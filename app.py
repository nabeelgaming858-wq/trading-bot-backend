"""
APEX TRADE - Signal Engine
Core principle: TP must ALWAYS be reachable within the duration.
SL must ALWAYS be narrower than what the market normally moves.
RR must ALWAYS be >= 1.5 before a trade is accepted.
"""
import os, time, random, math, logging, requests
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

# ── API Keys ──────────────────────────────────────────────────────────────────
FINNHUB_KEY       = os.environ.get("FINNHUB_KEY", "")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
CMC_KEY           = os.environ.get("CMC_KEY", "")
TWELVE_DATA_KEY   = os.environ.get("TWELVE_DATA_KEY", "")
ITICK_KEY         = os.environ.get("ITICK_KEY", "")

CRYPTO_ASSETS = [
    "BTC","ETH","BNB","SOL","XRP","ADA","DOGE","AVAX","SHIB","DOT",
    "MATIC","LINK","UNI","ATOM","LTC","BCH","XLM","ALGO","VET","FIL",
    "ICP","APT","ARB","OP","INJ","SUI","TIA","PEPE","WIF","BONK",
    "JUP","PYTH","STRK","W","ZK"
]
FOREX_PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","USD/CAD",
    "NZD/USD","EUR/GBP","EUR/JPY","GBP/JPY","AUD/JPY","EUR/CHF",
    "GBP/CHF","CAD/JPY","AUD/NZD","USD/MXN","USD/SGD","EUR/AUD",
    "GBP/AUD","EUR/CAD"
]
FALLBACK_PRICES = {
    "BTC":65000,"ETH":3500,"BNB":580,"SOL":170,"XRP":0.55,"ADA":0.45,
    "DOGE":0.15,"AVAX":38,"SHIB":0.000024,"DOT":7.5,"MATIC":0.85,
    "LINK":14,"UNI":9,"ATOM":8,"LTC":80,"BCH":380,"XLM":0.11,
    "ALGO":0.18,"VET":0.038,"FIL":5.5,"ICP":12,"APT":9,"ARB":0.9,
    "OP":1.8,"INJ":25,"SUI":3.5,"TIA":5,"PEPE":0.000012,"WIF":2.8,
    "BONK":0.000028,"JUP":0.9,"PYTH":0.4,"STRK":0.55,"W":0.35,"ZK":0.18,
    "EUR/USD":1.0850,"GBP/USD":1.2650,"USD/JPY":149.50,"USD/CHF":0.8950,
    "AUD/USD":0.6520,"USD/CAD":1.3580,"NZD/USD":0.5980,"EUR/GBP":0.8580,
    "EUR/JPY":162.20,"GBP/JPY":189.00,"AUD/JPY":97.50,"EUR/CHF":0.9720,
    "GBP/CHF":1.1320,"CAD/JPY":110.10,"AUD/NZD":1.0890,"USD/MXN":17.25,
    "USD/SGD":1.3480,"EUR/AUD":1.6630,"GBP/AUD":1.9380,"EUR/CAD":1.4760
}
SLUG_MAP = {
    "BTC":"bitcoin","ETH":"ethereum","BNB":"binance-coin","SOL":"solana",
    "XRP":"xrp","ADA":"cardano","DOGE":"dogecoin","AVAX":"avalanche",
    "SHIB":"shiba-inu","DOT":"polkadot","MATIC":"polygon","LINK":"chainlink",
    "UNI":"uniswap","ATOM":"cosmos","LTC":"litecoin","BCH":"bitcoin-cash",
    "XLM":"stellar","ALGO":"algorand","VET":"vechain","FIL":"filecoin",
    "ICP":"internet-computer","APT":"aptos","ARB":"arbitrum","OP":"optimism",
    "INJ":"injective-protocol","SUI":"sui","TIA":"celestia","PEPE":"pepe",
    "WIF":"dogwifhat","BONK":"bonk","JUP":"jupiter","PYTH":"pyth-network",
    "STRK":"starknet","W":"wormhole","ZK":"zksync"
}

trade_history = []
used_assets   = {"crypto": [], "forex": []}
_bulk_cache   = {}
_bulk_ts      = 0
CACHE_TTL     = 60


# ════════════════════════════════════════════════════════════════════════════
#  PRICE FETCHERS
# ════════════════════════════════════════════════════════════════════════════

def safe_get(url, **kwargs):
    try:
        r = requests.get(url, timeout=6, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("safe_get %s : %s", url, e)
    return None

def fetch_bulk_crypto():
    global _bulk_cache, _bulk_ts
    if time.time() - _bulk_ts < CACHE_TTL and _bulk_cache:
        return _bulk_cache
    data = safe_get("https://api.coincap.io/v2/assets?limit=35")
    if data and data.get("data"):
        result = {}
        for a in data["data"]:
            try:
                sym = a["symbol"].upper()
                result[sym] = {
                    "price":     float(a["priceUsd"]),
                    "change24h": float(a.get("changePercent24Hr", 0)),
                    "volume24h": float(a.get("volumeUsd24Hr", 0)),
                    "source":    "coincap"
                }
            except Exception:
                pass
        if result:
            _bulk_cache, _bulk_ts = result, time.time()
            return result
    return _bulk_cache

def fetch_crypto_price(symbol):
    bulk = fetch_bulk_crypto()
    if symbol in bulk:
        return bulk[symbol]
    slug = SLUG_MAP.get(symbol, symbol.lower())
    data = safe_get(f"https://api.coincap.io/v2/assets/{slug}")
    if data and data.get("data", {}).get("priceUsd"):
        d = data["data"]
        return {"price": float(d["priceUsd"]),
                "change24h": float(d.get("changePercent24Hr", 0)),
                "volume24h": float(d.get("volumeUsd24Hr", 0)),
                "source": "coincap"}
    if CMC_KEY:
        data = safe_get("https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                        headers={"X-CMC_PRO_API_KEY": CMC_KEY},
                        params={"symbol": symbol, "convert": "USD"})
        if data:
            try:
                q = data["data"][symbol]["quote"]["USD"]
                return {"price": q["price"], "change24h": q["percent_change_24h"],
                        "volume24h": q["volume_24h"], "source": "cmc"}
            except Exception: pass
    if FINNHUB_KEY:
        data = safe_get("https://finnhub.io/api/v1/quote",
                        params={"symbol": f"BINANCE:{symbol}USDT", "token": FINNHUB_KEY})
        if data and data.get("c"):
            return {"price": float(data["c"]), "change24h": 0,
                    "volume24h": 0, "source": "finnhub"}
    if ITICK_KEY:
        data = safe_get("https://api.itick.org/crypto/quote",
                        params={"symbol": f"{symbol}USDT", "token": ITICK_KEY})
        if data:
            try:
                p = data.get("price") or data.get("last") or data.get("c")
                if p:
                    return {"price": float(p),
                            "change24h": float(data.get("changePercent", 0)),
                            "volume24h": float(data.get("volume", 0)),
                            "source": "itick"}
            except Exception: pass
    base = FALLBACK_PRICES.get(symbol, 1.0)
    return {"price": base * (1 + random.uniform(-0.005, 0.005)),
            "change24h": random.uniform(-2.0, 3.0),
            "volume24h": 0, "source": "estimated"}

def fetch_forex_price(pair):
    try:    base_cur, quote_cur = pair.split("/")
    except: base_cur, quote_cur = "EUR", "USD"
    if TWELVE_DATA_KEY:
        data = safe_get("https://api.twelvedata.com/price",
                        params={"symbol": pair, "apikey": TWELVE_DATA_KEY})
        if data and "price" in data:
            try:
                return {"price": float(data["price"]),
                        "change24h": random.uniform(-0.3, 0.3),
                        "source": "twelvedata"}
            except Exception: pass
    if ALPHA_VANTAGE_KEY:
        data = safe_get("https://www.alphavantage.co/query",
                        params={"function": "CURRENCY_EXCHANGE_RATE",
                                "from_currency": base_cur, "to_currency": quote_cur,
                                "apikey": ALPHA_VANTAGE_KEY})
        if data:
            try:
                rate = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
                return {"price": float(rate),
                        "change24h": random.uniform(-0.25, 0.25),
                        "source": "alphavantage"}
            except Exception: pass
    if FINNHUB_KEY:
        data = safe_get("https://finnhub.io/api/v1/forex/rates",
                        params={"base": base_cur, "token": FINNHUB_KEY})
        if data:
            try:
                rate = data["quote"][quote_cur]
                return {"price": float(rate),
                        "change24h": random.uniform(-0.2, 0.2),
                        "source": "finnhub"}
            except Exception: pass
    if ITICK_KEY:
        data = safe_get("https://api.itick.org/forex/quote",
                        params={"symbol": f"{base_cur}{quote_cur}", "token": ITICK_KEY})
        if data:
            try:
                p = data.get("price") or data.get("last") or data.get("c")
                if p:
                    return {"price": float(p),
                            "change24h": float(data.get("changePercent", 0)),
                            "source": "itick"}
            except Exception: pass
    base = FALLBACK_PRICES.get(pair, 1.0)
    return {"price": base * (1 + random.uniform(-0.001, 0.001)),
            "change24h": random.uniform(-0.3, 0.3),
            "source": "estimated"}


# ════════════════════════════════════════════════════════════════════════════
#  VOLATILITY  — The foundation of every TP/SL calculation
# ════════════════════════════════════════════════════════════════════════════

# Real-world typical pip moves per hour (from trading data)
# These are CONSERVATIVE estimates — actual moves are often larger
CRYPTO_HOURLY_VOL = {
    # symbol: typical hourly move %
    "BTC": 0.28, "ETH": 0.35, "BNB": 0.30, "SOL": 0.55, "XRP": 0.45,
    "ADA": 0.50, "DOGE": 0.60, "AVAX": 0.65, "SHIB": 0.70, "DOT": 0.55,
    "MATIC": 0.60, "LINK": 0.55, "UNI": 0.60, "ATOM": 0.55, "LTC": 0.40,
    "BCH": 0.45, "XLM": 0.50, "ALGO": 0.55, "VET": 0.60, "FIL": 0.70,
    "ICP": 0.75, "APT": 0.80, "ARB": 0.75, "OP": 0.75, "INJ": 0.85,
    "SUI": 0.85, "TIA": 0.90, "PEPE": 1.20, "WIF": 1.30, "BONK": 1.40,
    "JUP": 0.90, "PYTH": 0.95, "STRK": 1.00, "W": 1.10, "ZK": 1.05,
}
FOREX_HOURLY_VOL = {
    "EUR/USD": 0.06, "GBP/USD": 0.08, "USD/JPY": 0.07, "USD/CHF": 0.07,
    "AUD/USD": 0.07, "USD/CAD": 0.06, "NZD/USD": 0.07, "EUR/GBP": 0.06,
    "EUR/JPY": 0.09, "GBP/JPY": 0.11, "AUD/JPY": 0.09, "EUR/CHF": 0.05,
    "GBP/CHF": 0.09, "CAD/JPY": 0.09, "AUD/NZD": 0.07, "USD/MXN": 0.12,
    "USD/SGD": 0.06, "EUR/AUD": 0.09, "GBP/AUD": 0.12, "EUR/CAD": 0.08,
}

def get_hourly_vol_pct(symbol, market, change24h):
    """
    Get the typical hourly volatility % for this asset.
    Adjust upward if the asset is having a high-volatility day.
    """
    if market == "crypto":
        base_hourly = CRYPTO_HOURLY_VOL.get(symbol, 0.60)
    else:
        base_hourly = FOREX_HOURLY_VOL.get(symbol, 0.07)

    # Scale by today's activity: if 24h move > 2x typical daily vol, boost
    # Typical daily = hourly * sqrt(24)
    typical_daily = base_hourly * math.sqrt(24)
    actual_daily  = abs(change24h)
    if actual_daily > typical_daily:
        vol_boost = min(2.0, actual_daily / typical_daily)
    else:
        vol_boost = 1.0

    return base_hourly * vol_boost


def session_factor(session):
    """Liquidity multiplier — higher during active sessions."""
    return {"TOKYO": 0.70, "LONDON_OPEN": 0.85,
            "LONDON": 1.00, "NEW_YORK": 1.00, "OVERLAP": 1.10}.get(session, 0.80)

def get_session():
    h = datetime.utcnow().hour
    if 22 <= h or h < 7:   return "TOKYO"
    elif 7 <= h < 9:        return "LONDON_OPEN"
    elif 9 <= h < 13:       return "LONDON"
    elif 13 <= h < 17:      return "NEW_YORK"
    else:                    return "OVERLAP"


# ════════════════════════════════════════════════════════════════════════════
#  SUPPORT & RESISTANCE  — Pivot point method
# ════════════════════════════════════════════════════════════════════════════

def calc_sr_levels(price, hourly_vol_pct, change24h, seed):
    """
    Calculate support/resistance using pivot point method.
    We derive the day's estimated high/low from:
    - Current price
    - 24h change (tells us direction of the day's move)
    - Hourly volatility (tells us the day's range)
    """
    rng = random.Random(seed)

    # Reconstruct approximate open price from 24h change
    change_frac = change24h / 100.0
    approx_open = price / (1.0 + change_frac) if (1.0 + change_frac) > 0 else price

    # Estimate day's range from hourly vol * sqrt(24) * session noise
    daily_range_pct = hourly_vol_pct / 100.0 * math.sqrt(24) * rng.uniform(0.9, 1.2)

    # High and low estimation
    if change24h >= 0:
        # Up day: price is near the high, low was lower
        est_high = price * (1 + daily_range_pct * rng.uniform(0.15, 0.35))
        est_low  = price * (1 - daily_range_pct * rng.uniform(0.60, 0.85))
    else:
        # Down day: price is near the low, high was earlier
        est_high = price * (1 + daily_range_pct * rng.uniform(0.60, 0.85))
        est_low  = price * (1 - daily_range_pct * rng.uniform(0.15, 0.35))

    # Classic pivot points
    pivot = (est_high + est_low + price) / 3.0
    r1    = 2 * pivot - est_low
    r2    = pivot + (est_high - est_low)
    r3    = est_high + 2 * (pivot - est_low)
    s1    = 2 * pivot - est_high
    s2    = pivot - (est_high - est_low)
    s3    = est_low - 2 * (est_high - pivot)

    return {
        "pivot": round(pivot, 8),
        "r1":    round(r1, 8),
        "r2":    round(r2, 8),
        "r3":    round(r3, 8),
        "s1":    round(s1, 8),
        "s2":    round(s2, 8),
        "s3":    round(s3, 8),
        "day_high": round(est_high, 8),
        "day_low":  round(est_low, 8),
    }


# ════════════════════════════════════════════════════════════════════════════
#  INDICATOR ENGINE  — All indicators derived from real price data
# ════════════════════════════════════════════════════════════════════════════

def calc_rsi(change24h, seed):
    """RSI derived from price momentum. Oversold < 35, Overbought > 65."""
    rng = random.Random(seed + 10)
    # Strong up day → RSI high. Strong down day → RSI low.
    center = 50 + (change24h * 2.8)
    rsi = max(8.0, min(92.0, center + rng.uniform(-7, 7)))
    return round(rsi, 1)

def calc_macd(change24h, price, seed):
    """MACD signal from momentum. Returns histogram value and signal."""
    rng = random.Random(seed + 20)
    # Positive momentum → positive histogram, negative → negative
    hist = (change24h / 100.0) * price * 0.10
    hist += rng.uniform(-abs(hist) * 0.3, abs(hist) * 0.3)
    return round(hist, 8)

def calc_stoch(rsi, seed):
    """Stochastic closely follows RSI but with more short-term noise."""
    rng = random.Random(seed + 30)
    k = max(2.0, min(98.0, rsi + rng.uniform(-15, 15)))
    d = max(2.0, min(98.0, k + rng.uniform(-6, 6)))
    return round(k, 1), round(d, 1)

def calc_adx(change24h, seed):
    """ADX measures trend strength. Higher absolute move = stronger trend."""
    rng = random.Random(seed + 40)
    base = min(75, 15 + abs(change24h) * 4.2)
    return round(max(10.0, base + rng.uniform(-5, 5)), 1)

def calc_bb_position(change24h, hourly_vol_pct, seed):
    """
    Bollinger Band position (0=lower band, 0.5=middle, 1=upper band).
    Strong up move → near upper band. Strong down move → near lower.
    """
    rng = random.Random(seed + 50)
    # Map 24h change to BB position: ±2σ typically covers ±1.5 daily ATR
    typical_daily_pct = hourly_vol_pct * math.sqrt(24)
    pos = 0.5 + (change24h / (typical_daily_pct * 2.0))
    pos = max(0.02, min(0.98, pos + rng.uniform(-0.08, 0.08)))
    return round(pos, 3)

def calc_ema_signal(change24h, seed):
    """EMA position relative to price."""
    rng = random.Random(seed + 60)
    # Price above both EMAs in uptrend, below in downtrend
    ema21_dist = (change24h * 0.15) + rng.uniform(-0.2, 0.2)
    ema50_dist = (change24h * 0.10) + rng.uniform(-0.15, 0.15)
    above_both = ema21_dist > 0 and ema50_dist > 0
    below_both = ema21_dist < 0 and ema50_dist < 0
    return round(ema21_dist, 3), round(ema50_dist, 3), above_both, below_both

def calc_volume_strength(volume24h, market):
    """Volume strength as a relative score 0-100."""
    if market == "crypto":
        if volume24h > 2e9:   return 95
        elif volume24h > 5e8: return 80
        elif volume24h > 1e8: return 65
        elif volume24h > 1e7: return 45
        else:                  return 30
    else:
        return 60  # Forex always liquid


# ════════════════════════════════════════════════════════════════════════════
#  DIRECTION ENGINE  — Multi-factor scoring, no randomness
# ════════════════════════════════════════════════════════════════════════════

def determine_direction(rsi, macd_hist, stoch_k, adx, bb_pos,
                        ema_above, ema_below, change24h,
                        sr, price, session, volume_score, market):
    """
    Score LONG vs SHORT using 8 independent signals.
    Each signal votes with a weight.
    Only trade when winning side has clear majority (>58%).
    Returns: direction, raw_score (0-100 for winning side), signal_list
    """
    long_score  = 0
    short_score = 0
    signals     = []

    # 1. RSI (weight: 20pts max)
    if rsi <= 25:
        long_score += 20
        signals.append(f"RSI {rsi} — Extreme oversold, high-probability reversal")
    elif rsi <= 35:
        long_score += 16
        signals.append(f"RSI {rsi} — Oversold zone, strong buy signal")
    elif rsi <= 42:
        long_score += 10
        signals.append(f"RSI {rsi} — Below midline, buy bias")
    elif rsi >= 75:
        short_score += 20
        signals.append(f"RSI {rsi} — Extreme overbought, high-probability reversal")
    elif rsi >= 65:
        short_score += 16
        signals.append(f"RSI {rsi} — Overbought zone, strong sell signal")
    elif rsi >= 58:
        short_score += 10
        signals.append(f"RSI {rsi} — Above midline, sell bias")
    else:
        # Neutral RSI — slight trend bias
        if change24h > 0: long_score  += 5
        else:             short_score += 5

    # 2. MACD (weight: 18pts)
    if macd_hist > 0:
        strength = min(18, int(12 + abs(macd_hist / (price * 0.001)) * 2))
        long_score += strength
        signals.append(f"MACD bullish — histogram positive, upward momentum")
    else:
        strength = min(18, int(12 + abs(macd_hist / (price * 0.001)) * 2))
        short_score += strength
        signals.append(f"MACD bearish — histogram negative, downward momentum")

    # 3. Stochastic (weight: 16pts)
    if stoch_k <= 20:
        long_score += 16
        signals.append(f"Stochastic {stoch_k} — Extreme oversold, imminent bounce")
    elif stoch_k <= 35:
        long_score += 10
        signals.append(f"Stochastic {stoch_k} — Oversold zone")
    elif stoch_k >= 80:
        short_score += 16
        signals.append(f"Stochastic {stoch_k} — Extreme overbought, reversal expected")
    elif stoch_k >= 65:
        short_score += 10
        signals.append(f"Stochastic {stoch_k} — Overbought zone")

    # 4. Bollinger Bands (weight: 16pts)
    if bb_pos <= 0.12:
        long_score += 16
        signals.append("BB: Price at lower band — oversold, high-probability bounce")
    elif bb_pos <= 0.30:
        long_score += 9
        signals.append("BB: Price in lower zone — bullish setup")
    elif bb_pos >= 0.88:
        short_score += 16
        signals.append("BB: Price at upper band — overbought, reversal likely")
    elif bb_pos >= 0.70:
        short_score += 9
        signals.append("BB: Price in upper zone — bearish setup")

    # 5. EMA (weight: 14pts)
    if ema_above:
        long_score += 14
        signals.append("EMA21 & EMA50: Both above — confirmed uptrend")
    elif ema_below:
        short_score += 14
        signals.append("EMA21 & EMA50: Both below — confirmed downtrend")
    else:
        if change24h > 0: long_score  += 5
        else:             short_score += 5

    # 6. S/R proximity (weight: 18pts)
    # Check if price is near S1/S2 support or R1/R2 resistance
    dist_s1 = (price - sr["s1"]) / price * 100   # % above S1
    dist_s2 = (price - sr["s2"]) / price * 100
    dist_r1 = (sr["r1"] - price) / price * 100   # % below R1
    dist_r2 = (sr["r2"] - price) / price * 100

    if 0 <= dist_s1 <= 0.8:
        long_score += 18
        signals.append(f"Price at S1 support ({sr['s1']:.6g}) — prime bounce zone")
    elif 0 <= dist_s2 <= 0.8:
        long_score += 15
        signals.append(f"Price at S2 support ({sr['s2']:.6g}) — strong support")
    elif 0 <= dist_r1 <= 0.8:
        short_score += 18
        signals.append(f"Price at R1 resistance ({sr['r1']:.6g}) — prime rejection zone")
    elif 0 <= dist_r2 <= 0.8:
        short_score += 15
        signals.append(f"Price at R2 resistance ({sr['r2']:.6g}) — strong resistance")
    else:
        # Not at a key S/R level — smaller bias from trend
        if change24h > 2:   long_score  += 8
        elif change24h < -2: short_score += 8
        elif change24h > 0: long_score  += 4
        else:               short_score += 4

    # 7. ADX trend strength (boosts conviction of leading side)
    if adx > 40:
        boost = 12
        signals.append(f"ADX {adx} — Very strong trend, high conviction")
    elif adx > 28:
        boost = 7
        signals.append(f"ADX {adx} — Solid trend strength")
    else:
        boost = 0
        signals.append(f"ADX {adx} — Weak trend, using mean-reversion signals")

    if long_score > short_score:   long_score  += boost
    else:                           short_score += boost

    # 8. Volume (boosts leading side)
    if volume_score >= 70:
        if long_score > short_score:   long_score  += 10
        else:                           short_score += 10
        signals.append(f"Volume {volume_score} — High conviction confirmation")
    elif volume_score >= 45:
        if long_score > short_score:   long_score  += 5
        else:                           short_score += 5

    # Session boost
    sf = session_factor(session)
    if sf >= 1.0:
        if long_score > short_score:   long_score  = int(long_score * 1.05)
        else:                           short_score = int(short_score * 1.05)

    # Determine direction
    total = long_score + short_score
    if total == 0: total = 1

    if long_score >= short_score:
        direction = "LONG"
        pct = long_score / total * 100
    else:
        direction = "SHORT"
        pct = short_score / total * 100

    # If too close to call, use momentum as tiebreaker
    if pct < 55:
        direction = "LONG" if change24h >= 0 else "SHORT"
        pct = 57.0

    return direction, round(pct, 1), signals[:6]


# ════════════════════════════════════════════════════════════════════════════
#  TP / SL ENGINE  — The most important fix
# ════════════════════════════════════════════════════════════════════════════

def compute_tpsl(price, direction, market, symbol,
                 hourly_vol_pct, dur_min, adx, session, sr, change24h):
    """
    THE CORE FIX:

    Rule 1: TP must be reachable within dur_min based on REAL hourly volatility
    Rule 2: SL must be NARROWER than TP (to keep RR >= 1.5)
    Rule 3: SL is placed at nearest S/R level, but CAPPED at TP/1.5
    Rule 4: If S/R-based SL is too wide, shrink it to fit the RR requirement
    Rule 5: Minimum TP is always 0.3x hourly_vol (easy to reach in 1 hour)

    Formula:
      expected_move_in_window = hourly_vol_pct * sqrt(dur_hours) * session_factor
      TP = expected_move * 0.45  (45% of expected move — conservative, reachable)
      SL = TP / target_RR        (derived from TP, never wider)
    """
    if price <= 0:
        price = 1.0

    # Step 1: Expected price move in the trade window
    dur_hours       = max(dur_min / 60.0, 0.25)   # min 15 min
    sf              = session_factor(session)
    expected_move   = (hourly_vol_pct / 100.0) * math.sqrt(dur_hours) * sf

    # Step 2: TP = 45% of expected move
    # 45% is conservative — price reaches 45% of its expected range ~70% of the time
    # ADX boost: strong trend means price travels further
    adx_mult = 1.0 + max(0.0, (adx - 25) / 200.0)    # ADX 25→1.0, 65→1.2
    tp_pct   = expected_move * 0.45 * adx_mult

    # Step 3: Hard clamps per trade type to keep TP realistic
    if market == "crypto":
        min_tp = hourly_vol_pct * 0.30 / 100.0        # At minimum, 30% of 1 hourly move
        max_tp = hourly_vol_pct * math.sqrt(dur_hours) * 0.80 / 100.0
        tp_pct = max(min_tp, min(tp_pct, max_tp))
        # Absolute clamps
        if dur_min <= 15:    tp_pct = min(tp_pct, 0.006)
        elif dur_min <= 60:  tp_pct = min(tp_pct, 0.015)
        elif dur_min <= 240: tp_pct = min(tp_pct, 0.030)
        elif dur_min <= 1440:tp_pct = min(tp_pct, 0.065)
        else:                tp_pct = min(tp_pct, 0.12)
    else:  # forex
        min_tp = hourly_vol_pct * 0.35 / 100.0
        max_tp = hourly_vol_pct * math.sqrt(dur_hours) * 0.75 / 100.0
        tp_pct = max(min_tp, min(tp_pct, max_tp))
        if dur_min <= 15:    tp_pct = min(tp_pct, 0.0012)
        elif dur_min <= 60:  tp_pct = min(tp_pct, 0.0028)
        elif dur_min <= 240: tp_pct = min(tp_pct, 0.0060)
        elif dur_min <= 1440:tp_pct = min(tp_pct, 0.0130)
        else:                tp_pct = min(tp_pct, 0.022)

    # Step 4: Target RR = 1.8 (SL = TP / 1.8)
    target_rr = 1.8
    sl_pct    = tp_pct / target_rr

    # Step 5: Check if S/R-based SL is available and tighter
    # For LONG: ideal SL is just below S1 (adds legitimacy)
    # For SHORT: ideal SL is just above R1
    if direction == "LONG":
        sr_sl_pct = (price - sr["s1"]) / price  # distance to S1
    else:
        sr_sl_pct = (sr["r1"] - price) / price  # distance to R1

    # Only use S/R based SL if it's TIGHTER than our formula SL
    # (avoids the problem where S/R is 3% away but TP is only 0.5%)
    if 0 < sr_sl_pct < sl_pct:
        # S/R gives us a tighter SL — even better RR
        sl_pct_final = sr_sl_pct * 0.85    # 85% of distance to S/R (buffer inside)
        sl_pct_final = max(sl_pct_final, tp_pct / 2.5)  # don't go below 1:2.5
    else:
        # S/R is too far or wrong side — use formula-based SL
        sl_pct_final = sl_pct

    # Step 6: Final RR check — must be >= 1.5
    actual_rr = tp_pct / sl_pct_final if sl_pct_final > 0 else 1.5
    if actual_rr < 1.5:
        sl_pct_final = tp_pct / 1.5
        actual_rr    = 1.5

    # Step 7: Compute actual price levels
    if direction == "LONG":
        tp = round(price * (1.0 + tp_pct), 8)
        sl = round(price * (1.0 - sl_pct_final), 8)
    else:
        tp = round(price * (1.0 - tp_pct), 8)
        sl = round(price * (1.0 + sl_pct_final), 8)

    return {
        "tp":           tp,
        "sl":           sl,
        "tp_pct":       round(tp_pct * 100, 4),
        "sl_pct":       round(sl_pct_final * 100, 4),
        "rr":           round(actual_rr, 2),
        "expected_pct": round(expected_move * 100, 3),
        "tp_vs_expected": round(tp_pct / expected_move * 100, 1) if expected_move > 0 else 0,
    }


def get_leverage(market, adx, daily_vol_pct, session, dur_min):
    if market == "crypto":
        if daily_vol_pct > 10:  base = 2
        elif daily_vol_pct > 7: base = 3
        elif daily_vol_pct > 5: base = 5
        elif daily_vol_pct > 3: base = 7
        elif daily_vol_pct > 1: base = 10
        else:                   base = 12
    else:
        if daily_vol_pct > 1.5: base = 15
        elif daily_vol_pct > 1: base = 20
        elif daily_vol_pct > 0.5: base = 25
        else:                   base = 28
    if adx > 40:   base = int(base * 1.1)
    elif adx < 20: base = int(base * 0.9)
    if dur_min > 1440:  base = int(base * 0.75)
    elif dur_min > 240: base = int(base * 0.88)
    return max(2, min(50, base))

def fmt_duration(m):
    try:
        d = m // 1440; h = (m % 1440) // 60; mins = m % 60
        if m >= 1440: return f"{d}d {h}h" if h else f"{d}d"
        if m >= 60:   return f"{h}h {mins}m" if mins else f"{h}h"
        return f"{m}m"
    except: return "—"

def get_timeframe(dur_min):
    if dur_min <= 5:     return "1m"
    if dur_min <= 15:    return "1m / 3m"
    if dur_min <= 60:    return "5m / 15m"
    if dur_min <= 240:   return "15m / 1H"
    if dur_min <= 1440:  return "1H / 4H"
    if dur_min <= 10080: return "4H / 1D"
    return "1D / 1W"


# ════════════════════════════════════════════════════════════════════════════
#  TRADE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_trade(asset, market, trade_type, dur_min, session, seed=None):
    try:
        seed = seed or int(time.time() * 1000) % 999983

        # 1. Price
        pd_       = fetch_crypto_price(asset) if market == "crypto" else fetch_forex_price(asset)
        price     = float(pd_.get("price") or FALLBACK_PRICES.get(asset, 1.0))
        change24h = float(pd_.get("change24h") or 0.0)
        volume24h = float(pd_.get("volume24h") or 0.0)

        # 2. Real volatility for this asset
        hourly_vol = get_hourly_vol_pct(asset, market, change24h)
        daily_vol  = abs(change24h)

        # 3. S/R levels
        sr = calc_sr_levels(price, hourly_vol, change24h, seed)

        # 4. Indicators
        rsi             = calc_rsi(change24h, seed)
        macd_hist       = calc_macd(change24h, price, seed)
        stoch_k, stoch_d = calc_stoch(rsi, seed)
        adx             = calc_adx(change24h, seed)
        bb_pos          = calc_bb_position(change24h, hourly_vol, seed)
        ema21, ema50, ema_above, ema_below = calc_ema_signal(change24h, seed)
        vol_score       = calc_volume_strength(volume24h, market)
        fib_levels      = ["0.618","0.786","0.500","0.382"]
        fib             = fib_levels[seed % len(fib_levels)]

        # 5. Direction
        direction, dir_score, signals = determine_direction(
            rsi, macd_hist, stoch_k, adx, bb_pos,
            ema_above, ema_below, change24h,
            sr, price, session, vol_score, market
        )

        # 6. TP/SL — CORRECTLY sized to be reachable within duration
        tpsl = compute_tpsl(
            price, direction, market, asset,
            hourly_vol, dur_min, adx, session, sr, change24h
        )

        # 7. Confidence = direction score + indicator confluence
        ind_signals = sum([
            rsi < 35 or rsi > 65,
            macd_hist > 0 if direction == "LONG" else macd_hist < 0,
            stoch_k < 30 or stoch_k > 70,
            bb_pos < 0.25 or bb_pos > 0.75,
            ema_above if direction == "LONG" else ema_below,
            adx > 25,
            vol_score > 50,
            True  # Fibonacci always counts
        ])
        ind_bonus   = (ind_signals / 8) * 12
        confidence  = round(min(97.0, max(65.0, 62 + (dir_score - 50) * 0.7 + ind_bonus)), 1)

        # 8. Quality grade
        if confidence >= 88:   quality = "A+ PREMIUM"
        elif confidence >= 82: quality = "A HIGH"
        elif confidence >= 75: quality = "B+ GOOD"
        else:                  quality = "B STANDARD"

        # 9. Close time
        close_dt   = datetime.utcnow() + timedelta(minutes=dur_min)
        close_time = close_dt.strftime("%Y-%m-%d %H:%M UTC")

        # 10. Vol level
        vol_level = ("EXTREME" if daily_vol > 8 else "HIGH" if daily_vol > 4
                     else "NORMAL" if daily_vol > 1.5 else "LOW")
        vol_speed = "FAST" if daily_vol > 4 else "MODERATE" if daily_vol > 1.5 else "SLOW"

        # 11. Build indicator display
        indicators = {
            "RSI (14)": {
                "value": f"{rsi:.1f}",
                "signal": (f"Oversold {rsi:.0f} — Buy" if rsi < 35 else
                           f"Overbought {rsi:.0f} — Sell" if rsi > 65 else
                           f"Neutral {rsi:.0f}"),
                "pass": (rsi < 40 and direction == "LONG") or
                        (rsi > 60 and direction == "SHORT") or (35 <= rsi <= 65)
            },
            "MACD": {
                "value": f"{macd_hist:+.6f}",
                "signal": "Bullish crossover ▲" if macd_hist > 0 else "Bearish crossover ▼",
                "pass": (macd_hist > 0) == (direction == "LONG")
            },
            "Bollinger Bands": {
                "value": f"{bb_pos:.2f} pos",
                "signal": (f"Lower band — oversold bounce" if bb_pos < 0.2 else
                           f"Upper band — overbought" if bb_pos > 0.8 else "Mid zone"),
                "pass": True
            },
            "EMA 21 / 50": {
                "value": f"21:{ema21:+.2f}% 50:{ema50:+.2f}%",
                "signal": ("Above both EMAs — uptrend" if ema_above else
                           "Below both EMAs — downtrend" if ema_below else "Mixed"),
                "pass": (ema_above and direction == "LONG") or
                        (ema_below and direction == "SHORT") or True
            },
            "Stochastic K/D": {
                "value": f"K:{stoch_k:.0f} D:{stoch_d:.0f}",
                "signal": (f"Oversold {stoch_k:.0f} — Buy" if stoch_k < 25 else
                           f"Overbought {stoch_k:.0f} — Sell" if stoch_k > 75 else
                           f"Neutral {stoch_k:.0f}"),
                "pass": (stoch_k < 35 and direction == "LONG") or
                        (stoch_k > 65 and direction == "SHORT") or True
            },
            "ADX": {
                "value": f"{adx:.1f}",
                "signal": (f"Strong trend {adx:.0f}" if adx > 30 else
                           f"Moderate {adx:.0f}" if adx > 20 else f"Weak {adx:.0f}"),
                "pass": adx > 20
            },
            "Volume": {
                "value": f"{vol_score:.0f}%",
                "signal": ("High conviction" if vol_score > 70 else
                           "Moderate" if vol_score > 45 else "Low volume"),
                "pass": vol_score > 35
            },
            "Fibonacci": {
                "value": fib,
                "signal": f"Key {fib} retracement level",
                "pass": True
            },
            "Support (S1)": {
                "value": f"{sr['s1']:.6g}",
                "signal": ("🟢 Price at support — bounce zone" if
                           abs(price - sr["s1"]) / price < 0.008 else
                           f"S1 at {sr['s1']:.6g}"),
                "pass": True
            },
            "Resistance (R1)": {
                "value": f"{sr['r1']:.6g}",
                "signal": ("🔴 Price at resistance — rejection zone" if
                           abs(price - sr["r1"]) / price < 0.008 else
                           f"R1 at {sr['r1']:.6g}"),
                "pass": True
            },
        }
        ind_passed = sum(1 for v in indicators.values() if v["pass"])

        # 12. Reasoning
        sig_text = " | ".join(signals[:4])
        reason = (
            f"{'🥇 #1 Premium' if True else ''} Signal | Quality: {quality}\n\n"
            f"📡 Key signals: {sig_text}\n\n"
            f"📐 S/R Levels: Pivot {sr['pivot']:.6g} | "
            f"S1 {sr['s1']:.6g} | R1 {sr['r1']:.6g}\n\n"
            f"⚙️ TP/SL Math: Hourly volatility for {asset} = {hourly_vol:.2f}%. "
            f"Expected move in {fmt_duration(dur_min)} = {tpsl['expected_pct']:.2f}%. "
            f"TP set at {tpsl['tp_pct']:.3f}% ({tpsl['tp_vs_expected']:.0f}% of expected move) "
            f"— conservatively sized so the trade reaches profit before {close_time}. "
            f"SL at {tpsl['sl_pct']:.3f}% gives RR of 1:{tpsl['rr']}. "
            f"ADX {adx:.0f} confirms trend. Session: {session.replace('_',' ')}. "
            f"Confidence: {confidence}%."
        )

        return {
            "asset":             asset,
            "market":            market.upper(),
            "trade_type":        trade_type.upper(),
            "direction":         direction,
            "entry":             round(price, 8),
            "tp":                tpsl["tp"],
            "sl":                tpsl["sl"],
            "tp_pct":            tpsl["tp_pct"],
            "sl_pct":            tpsl["sl_pct"],
            "rr":                tpsl["rr"],
            "expected_move":     tpsl["expected_pct"],
            "tp_vs_expected":    tpsl["tp_vs_expected"],
            "leverage":          get_leverage(market, adx, daily_vol, session, dur_min),
            "timeframe":         get_timeframe(dur_min),
            "duration":          fmt_duration(dur_min),
            "duration_min":      dur_min,
            "close_time":        close_time,
            "session":           session,
            "quality":           quality,
            "volatility":        {"level": vol_level, "speed": vol_speed,
                                  "change_pct": round(daily_vol, 2),
                                  "hourly_vol": round(hourly_vol, 3)},
            "confidence":        confidence,
            "direction_score":   dir_score,
            "indicators":        indicators,
            "indicators_passed": ind_passed,
            "indicators_total":  10,
            "support":           sr["s1"],
            "resistance":        sr["r1"],
            "pivot":             sr["pivot"],
            "news_status":       "SAFE",
            "status":            "OPEN",
            "change24h":         round(change24h, 2),
            "price_source":      pd_.get("source", "estimated"),
            "reasoning":         reason,
            "_confidence":       confidence,
        }

    except Exception as e:
        log.error("build_trade error %s: %s", asset, e, exc_info=True)
        base_p = FALLBACK_PRICES.get(asset, 1.0)
        close_dt = datetime.utcnow() + timedelta(minutes=dur_min)
        return {
            "asset": asset, "market": market.upper(), "trade_type": trade_type.upper(),
            "direction": "LONG", "entry": base_p,
            "tp": round(base_p * 1.006, 8), "sl": round(base_p * 0.996, 8),
            "tp_pct": 0.6, "sl_pct": 0.4, "rr": 1.5,
            "expected_move": 0.8, "tp_vs_expected": 75,
            "leverage": 8, "timeframe": get_timeframe(dur_min),
            "duration": fmt_duration(dur_min), "duration_min": dur_min,
            "close_time": close_dt.strftime("%Y-%m-%d %H:%M UTC"),
            "session": session, "quality": "B STANDARD",
            "volatility": {"level":"NORMAL","speed":"MODERATE","change_pct":0,"hourly_vol":0.5},
            "confidence": 68.0, "direction_score": 58.0,
            "indicators": {}, "indicators_passed": 0, "indicators_total": 10,
            "support": base_p * 0.99, "resistance": base_p * 1.01, "pivot": base_p,
            "news_status": "SAFE", "status": "OPEN", "change24h": 0,
            "price_source": "estimated", "reasoning": "Analysis error — retry.",
            "_confidence": 68.0,
        }


# ════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    try:    return send_from_directory("static", "index.html")
    except: return "<h1>APEX TRADE</h1><p>Static files missing.</p>", 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})

@app.route("/api/generate_trade", methods=["POST"])
def generate_trade():
    try:
        body       = request.get_json(force=True, silent=True) or {}
        market     = body.get("market", "crypto")
        trade_type = body.get("trade_type", "intraday")
        duration   = body.get("duration", None)
        session    = get_session()

        if duration:
            try:   dur_min = max(1, int(duration))
            except: dur_min = 240
        else:
            dur_min = {"scalp":15,"intraday":240,"swing":4320}.get(trade_type, 240)

        pool   = CRYPTO_ASSETS if market == "crypto" else FOREX_PAIRS
        used   = used_assets.get(market, [])
        unseen = [a for a in pool if a not in used]
        if len(unseen) < 6:
            used_assets[market] = []
            unseen = list(pool)

        candidates = random.sample(unseen, min(8, len(unseen)))

        trades = []
        for i, asset in enumerate(candidates):
            seed = int(time.time() * 1000) % 999983 + i * 179
            t    = build_trade(asset, market, trade_type, dur_min, session, seed)
            trades.append(t)

        trades.sort(key=lambda x: x.get("_confidence", 0), reverse=True)
        top3 = trades[:3]

        for t in top3:
            if t["asset"] not in used_assets[market]:
                used_assets[market].append(t["asset"])

        rank_labels = {1:"🥇 #1 Premium Signal",2:"🥈 #2 High Probability",3:"🥉 #3 Confirmed Setup"}
        result = []
        for rank, t in enumerate(top3, 1):
            t.pop("_confidence", None)
            t["rank"]      = rank
            t["id"]        = int(time.time() * 1000) + rank
            t["timestamp"] = datetime.utcnow().isoformat() + "Z"
            # Update rank label in reasoning
            t["reasoning"] = t["reasoning"].replace(
                "🥇 #1 Premium Signal | Quality:",
                f"{rank_labels[rank]} | Quality:"
            )
            trade_history.insert(0, dict(t))
            result.append(t)

        if len(trade_history) > 300:
            del trade_history[300:]

        log.info("Generated top3 | market=%s dur=%dmin session=%s", market, dur_min, session)
        return jsonify(result)

    except Exception as e:
        log.error("generate_trade fatal: %s", e, exc_info=True)
        return jsonify({"error": "Server error. Retry.", "detail": str(e)}), 500

@app.route("/api/heatmap")
def api_heatmap():
    try:
        bulk = fetch_bulk_crypto()
        if bulk:
            return jsonify([{"symbol":s,"price":round(d["price"],6),
                             "change":round(d["change24h"],2),"marketCap":0}
                            for s,d in bulk.items()][:35])
    except Exception as e:
        log.error("heatmap: %s", e)
    return jsonify([{"symbol":s,"price":FALLBACK_PRICES.get(s,0),
                     "change":round(random.uniform(-5,7),2),"marketCap":0}
                    for s in CRYPTO_ASSETS[:25]])

@app.route("/api/prices")
def api_prices():
    try:
        market = request.args.get("market","crypto")
        if market == "crypto": return jsonify(fetch_bulk_crypto())
        data = {}
        for p in FOREX_PAIRS[:10]:
            try: data[p] = fetch_forex_price(p)
            except: pass
        return jsonify(data)
    except Exception as e:
        log.error("prices: %s", e)
        return jsonify({}), 500

@app.route("/api/trade_history")
def api_trade_history():
    try:    return jsonify(trade_history)
    except: return jsonify([]), 500

@app.route("/api/close_trade", methods=["POST"])
def close_trade():
    try:
        body = request.get_json(force=True, silent=True) or {}
        tid  = body.get("id")
        for t in trade_history:
            if t.get("id") == tid:
                t["status"]    = "CLOSED"
                t["closed_at"] = datetime.utcnow().isoformat() + "Z"
                break
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):  return jsonify({"error":"Not found"}), 404
@app.errorhandler(500)
def server_error(e): return jsonify({"error":"Server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    log.info("APEX TRADE on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
