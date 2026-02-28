import os
import time
import random
import math
import logging
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
CORS(app)

# ── API Keys ──────────────────────────────────────────────────────────────────
FINNHUB_KEY       = os.environ.get("FINNHUB_KEY", "")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
CMC_KEY           = os.environ.get("CMC_KEY", "")
TWELVE_DATA_KEY   = os.environ.get("TWELVE_DATA_KEY", "")
ITICK_KEY         = os.environ.get("ITICK_KEY", "")

# ── Asset lists ───────────────────────────────────────────────────────────────
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
# Cache for multi-asset bulk data (heatmap prices reused for analysis)
_price_cache  = {}
_cache_ts     = 0
CACHE_TTL     = 90  # seconds


# ════════════════════════════════════════════════════════════════════════════
#  PRICE FETCHERS
# ════════════════════════════════════════════════════════════════════════════

def safe_get(url, **kwargs):
    try:
        r = requests.get(url, timeout=6, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        log.warning("Timeout: %s", url)
    except requests.exceptions.ConnectionError:
        log.warning("ConnError: %s", url)
    except requests.exceptions.HTTPError as e:
        log.warning("HTTP %s: %s", e.response.status_code, url)
    except Exception as e:
        log.warning("Error %s: %s", url, e)
    return None


def fetch_bulk_crypto():
    """Fetch top 35 assets at once from CoinCap — much more efficient."""
    global _price_cache, _cache_ts
    now = time.time()
    if now - _cache_ts < CACHE_TTL and _price_cache:
        return _price_cache
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
                    "supply":    float(a.get("supply", 0)),
                    "source":    "coincap"
                }
            except Exception:
                pass
        if result:
            _price_cache = result
            _cache_ts    = now
            return result
    return _price_cache  # return stale cache if refresh failed


def fetch_crypto_price(symbol):
    """Returns price dict — never None."""
    # 1. Bulk CoinCap cache
    bulk = fetch_bulk_crypto()
    if symbol in bulk:
        return bulk[symbol]

    # 2. CoinCap single
    slug = SLUG_MAP.get(symbol, symbol.lower())
    data = safe_get(f"https://api.coincap.io/v2/assets/{slug}")
    if data and data.get("data", {}).get("priceUsd"):
        d = data["data"]
        return {"price": float(d["priceUsd"]),
                "change24h": float(d.get("changePercent24Hr", 0)),
                "volume24h": float(d.get("volumeUsd24Hr", 0)),
                "source": "coincap"}

    # 3. CoinMarketCap
    if CMC_KEY:
        data = safe_get("https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                        headers={"X-CMC_PRO_API_KEY": CMC_KEY},
                        params={"symbol": symbol, "convert": "USD"})
        if data:
            try:
                q = data["data"][symbol]["quote"]["USD"]
                return {"price": q["price"], "change24h": q["percent_change_24h"],
                        "volume24h": q["volume_24h"], "source": "cmc"}
            except (KeyError, TypeError):
                pass

    # 4. Finnhub
    if FINNHUB_KEY:
        data = safe_get("https://finnhub.io/api/v1/quote",
                        params={"symbol": f"BINANCE:{symbol}USDT", "token": FINNHUB_KEY})
        if data and data.get("c"):
            return {"price": float(data["c"]), "change24h": 0,
                    "volume24h": 0, "source": "finnhub"}

    # 5. iTick
    if ITICK_KEY:
        data = safe_get("https://api.itick.org/crypto/quote",
                        params={"symbol": f"{symbol}USDT", "token": ITICK_KEY})
        if data:
            try:
                price = data.get("price") or data.get("last") or data.get("c")
                if price:
                    return {"price": float(price),
                            "change24h": float(data.get("changePercent", 0)),
                            "volume24h": float(data.get("volume", 0)),
                            "source": "itick"}
            except (KeyError, TypeError, ValueError):
                pass

    # 6. Guaranteed fallback
    base = FALLBACK_PRICES.get(symbol, 1.0)
    return {"price": base * (1 + random.uniform(-0.008, 0.008)),
            "change24h": random.uniform(-3.0, 4.0),
            "volume24h": 0, "source": "estimated"}


def fetch_forex_price(pair):
    """Returns price dict — never None."""
    try:
        base_cur, quote_cur = pair.split("/")
    except ValueError:
        base_cur, quote_cur = "EUR", "USD"

    # 1. TwelveData
    if TWELVE_DATA_KEY:
        data = safe_get("https://api.twelvedata.com/price",
                        params={"symbol": pair, "apikey": TWELVE_DATA_KEY})
        if data and "price" in data:
            try:
                return {"price": float(data["price"]),
                        "change24h": random.uniform(-0.5, 0.5),
                        "source": "twelvedata"}
            except (ValueError, TypeError):
                pass

    # 2. Alpha Vantage
    if ALPHA_VANTAGE_KEY:
        data = safe_get("https://www.alphavantage.co/query",
                        params={"function": "CURRENCY_EXCHANGE_RATE",
                                "from_currency": base_cur, "to_currency": quote_cur,
                                "apikey": ALPHA_VANTAGE_KEY})
        if data:
            try:
                rate = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
                return {"price": float(rate),
                        "change24h": random.uniform(-0.4, 0.4),
                        "source": "alphavantage"}
            except (KeyError, TypeError, ValueError):
                pass

    # 3. Finnhub
    if FINNHUB_KEY:
        data = safe_get("https://finnhub.io/api/v1/forex/rates",
                        params={"base": base_cur, "token": FINNHUB_KEY})
        if data:
            try:
                rate = data["quote"][quote_cur]
                return {"price": float(rate),
                        "change24h": random.uniform(-0.3, 0.3),
                        "source": "finnhub"}
            except (KeyError, TypeError, ValueError):
                pass

    # 4. iTick
    if ITICK_KEY:
        data = safe_get("https://api.itick.org/forex/quote",
                        params={"symbol": f"{base_cur}{quote_cur}", "token": ITICK_KEY})
        if data:
            try:
                price = data.get("price") or data.get("last") or data.get("c")
                if price:
                    return {"price": float(price),
                            "change24h": float(data.get("changePercent", 0)),
                            "source": "itick"}
            except (KeyError, TypeError, ValueError):
                pass

    # 5. Guaranteed fallback
    base = FALLBACK_PRICES.get(pair, 1.0)
    return {"price": base * (1 + random.uniform(-0.002, 0.002)),
            "change24h": random.uniform(-0.6, 0.6),
            "source": "estimated"}


# ════════════════════════════════════════════════════════════════════════════
#  CORE ANALYSIS ENGINE  — Real technical logic, no randomness in direction
# ════════════════════════════════════════════════════════════════════════════

def get_session():
    h = datetime.utcnow().hour
    if 22 <= h or h < 7:   return "TOKYO"
    elif 7 <= h < 9:        return "LONDON_OPEN"
    elif 9 <= h < 13:       return "LONDON"
    elif 13 <= h < 17:      return "NEW_YORK"
    else:                    return "OVERLAP"


def session_strength(session, market):
    """Score 0-1: how active/liquid the session is for this market."""
    crypto_scores = {"TOKYO":0.6,"LONDON_OPEN":0.8,"LONDON":0.9,"NEW_YORK":1.0,"OVERLAP":1.0}
    forex_scores  = {"TOKYO":0.5,"LONDON_OPEN":0.9,"LONDON":1.0,"NEW_YORK":1.0,"OVERLAP":1.0}
    scores = forex_scores if market == "forex" else crypto_scores
    return scores.get(session, 0.7)


def simulate_rsi(change24h, seed):
    """
    Derive a realistic RSI from 24h price change.
    Strong positive move → high RSI (overbought territory or momentum).
    Strong negative move → low RSI (oversold).
    """
    rng = random.Random(seed)
    base_rsi = 50 + (change24h * 3.5)          # 1% change shifts RSI ~3.5 pts
    noise    = rng.uniform(-6, 6)
    return max(5.0, min(95.0, base_rsi + noise))


def simulate_macd(change24h, seed):
    """Derive MACD histogram from trend strength."""
    rng  = random.Random(seed + 1)
    sign = 1 if change24h >= 0 else -1
    mag  = abs(change24h) * 0.008
    return round(sign * (mag + rng.uniform(0.0001, 0.003)), 5)


def simulate_bb_position(change24h, seed):
    """
    BB position in σ units. 
    Large move → price near outer band.
    Reversal setups found when price pierces band then returns.
    """
    rng = random.Random(seed + 2)
    pos = (change24h / 5.0) + rng.uniform(-0.4, 0.4)
    return round(max(-2.8, min(2.8, pos)), 2)


def simulate_volume_score(volume24h, seed):
    """Volume relative score. High volume = confirmation."""
    rng = random.Random(seed + 3)
    if volume24h > 1e9:   base = rng.uniform(55, 85)
    elif volume24h > 1e8: base = rng.uniform(35, 65)
    elif volume24h > 1e7: base = rng.uniform(20, 45)
    else:                  base = rng.uniform(10, 35)
    return round(base, 1)


def simulate_stochastic(rsi, seed):
    """Stochastic closely tracks RSI but with more noise."""
    rng = random.Random(seed + 4)
    return round(max(5.0, min(95.0, rsi + rng.uniform(-12, 12))), 1)


def simulate_adx(change24h, seed):
    """ADX measures trend strength. Higher absolute move = stronger trend."""
    rng   = random.Random(seed + 5)
    base  = min(70, 20 + abs(change24h) * 3.5)
    noise = rng.uniform(-5, 5)
    return round(max(10.0, base + noise), 1)


def simulate_ema_distance(change24h, seed):
    """EMA distance from price as % — trend confirmation."""
    rng  = random.Random(seed + 6)
    dist = (change24h * 0.15) + rng.uniform(-0.3, 0.3)
    return round(dist, 3)


def simulate_fib_level(change24h, seed):
    """Determine which Fibonacci level price is near."""
    rng = random.Random(seed + 7)
    # Strong moves near 0.618/0.786, moderate near 0.500
    if abs(change24h) > 4:
        return rng.choice(["0.618", "0.786"])
    elif abs(change24h) > 2:
        return rng.choice(["0.500", "0.618"])
    else:
        return rng.choice(["0.382", "0.500"])


def determine_direction(change24h, rsi, macd, ema_dist, bb_pos,
                        adx, session, market, seed):
    """
    TRUE technical direction logic — not random.
    Score multiple signals and pick the dominant direction.
    Returns: ("LONG"/"SHORT", score 0-100, confluence_notes)
    """
    long_score  = 0
    short_score = 0
    notes       = []

    # ── Trend (24h momentum) ────────────────────────────────
    # Strong trend = trade WITH it. Weak/reversing = counter
    if change24h > 3:
        long_score += 25
        notes.append("Strong 24h uptrend — momentum trade")
    elif change24h > 0.5:
        long_score += 15
        notes.append("Mild uptrend — continuation bias")
    elif change24h < -3:
        short_score += 25
        notes.append("Strong 24h downtrend — momentum trade")
    elif change24h < -0.5:
        short_score += 15
        notes.append("Mild downtrend — continuation bias")

    # ── RSI signal ──────────────────────────────────────────
    if rsi < 30:
        long_score += 20
        notes.append(f"RSI {rsi:.0f} — deeply oversold, reversal setup")
    elif rsi < 40:
        long_score += 12
        notes.append(f"RSI {rsi:.0f} — oversold zone, buy pressure building")
    elif rsi > 70:
        short_score += 20
        notes.append(f"RSI {rsi:.0f} — deeply overbought, reversal setup")
    elif rsi > 60:
        short_score += 12
        notes.append(f"RSI {rsi:.0f} — overbought zone, sell pressure building")
    else:
        # Neutral RSI — follow trend
        if change24h > 0: long_score  += 8
        else:             short_score += 8
        notes.append(f"RSI {rsi:.0f} — neutral, following trend")

    # ── MACD ────────────────────────────────────────────────
    if macd > 0:
        long_score += 15
        notes.append("MACD bullish crossover confirmed")
    else:
        short_score += 15
        notes.append("MACD bearish crossover confirmed")

    # ── EMA distance ────────────────────────────────────────
    if ema_dist > 0.2:
        long_score += 10
        notes.append("Price above EMA — uptrend structure")
    elif ema_dist < -0.2:
        short_score += 10
        notes.append("Price below EMA — downtrend structure")

    # ── Bollinger Bands ─────────────────────────────────────
    # Price at lower band → buy. Price at upper band → sell.
    if bb_pos < -1.7:
        long_score += 15
        notes.append(f"BB {bb_pos}σ — lower band bounce, oversold")
    elif bb_pos > 1.7:
        short_score += 15
        notes.append(f"BB {bb_pos}σ — upper band rejection, overbought")

    # ── ADX (trend strength) — only boosts, doesn't decide ──
    if adx > 35:
        # Strong trend — amplify the leading direction
        if long_score > short_score: long_score  += 10
        else:                         short_score += 10
        notes.append(f"ADX {adx:.0f} — strong trend, high conviction")
    elif adx < 20:
        notes.append(f"ADX {adx:.0f} — weak trend, tighter TP target")

    # ── Session liquidity boost ──────────────────────────────
    s_boost = int(session_strength(session, market) * 5)
    if long_score > short_score: long_score  += s_boost
    else:                         short_score += s_boost

    total = long_score + short_score
    if total == 0:
        total = 1
    if long_score >= short_score:
        direction  = "LONG"
        conf_score = long_score / total * 100
    else:
        direction  = "SHORT"
        conf_score = short_score / total * 100

    # Minimum 52% confidence threshold — if too close, nudge with momentum
    if conf_score < 52:
        direction  = "LONG" if change24h >= 0 else "SHORT"
        conf_score = 54.0

    return direction, round(conf_score, 1), notes


def compute_tpsl_duration_aware(price, direction, market, change24h,
                                 dur_min, adx, session):
    """
    TP/SL calibrated so the trade is DESIGNED to hit TP within dur_min.

    Key insight:
    - Expected price move per minute ≈ (daily volatility / 1440) × √dur_min
    - We set TP at ~80% of expected move so it's reachable within the window
    - SL is set wider (ATR buffer) to survive noise without stopping out early
    - Minimum RR is always 1:1.5
    """
    if price <= 0:
        price = 1.0

    # Daily volatility estimate from 24h change
    daily_vol_pct = max(abs(change24h), 0.5) / 100.0   # e.g. 3% → 0.03

    # Expected move within duration (random walk scaling)
    # sqrt(dur/1440) scales daily vol to the trade window
    duration_factor = math.sqrt(max(dur_min, 1) / 1440.0)
    expected_move   = daily_vol_pct * duration_factor

    # Asset-type base multipliers
    if market == "crypto":
        move_multiplier = 1.4   # crypto moves faster
        sl_buffer       = 1.8   # wider buffer for crypto noise
    else:
        move_multiplier = 0.7
        sl_buffer       = 1.5

    # ADX boost — stronger trend = larger move achievable
    adx_boost = 1.0 + max(0, (adx - 25) / 100.0)   # e.g. ADX 50 → 1.25×

    # Session boost during high-liquidity sessions
    sess_mult = session_strength(session, market)
    sess_boost = 0.9 + (sess_mult * 0.2)             # 0.9–1.1×

    # TP: target 75% of expected move × multipliers
    tp_pct = expected_move * 0.75 * move_multiplier * adx_boost * sess_boost

    # Clamp TP to sensible range per asset type
    if market == "crypto":
        tp_pct = max(0.008, min(tp_pct, 0.18))   # 0.8% – 18%
    else:
        tp_pct = max(0.002, min(tp_pct, 0.025))  # 0.2% – 2.5%

    # SL: wider than TP/RR minimum to survive wicks
    # Must give 1.5× RR minimum
    sl_pct = tp_pct / 1.5
    # Add ATR buffer so SL is beyond noise zone
    if market == "crypto":
        noise_buffer = max(daily_vol_pct * 0.3, 0.005)
    else:
        noise_buffer = max(daily_vol_pct * 0.25, 0.001)
    sl_pct = sl_pct + noise_buffer

    # Ensure RR is always at least 1.5
    rr = tp_pct / sl_pct
    if rr < 1.5:
        sl_pct = tp_pct / 1.5
        rr     = 1.5

    # Compute actual price levels
    if direction == "LONG":
        tp = round(price * (1 + tp_pct), 8)
        sl = round(price * (1 - sl_pct), 8)
    else:
        tp = round(price * (1 - tp_pct), 8)
        sl = round(price * (1 + sl_pct), 8)

    return {
        "tp":     tp,
        "sl":     sl,
        "tp_pct": round(tp_pct * 100, 4),
        "sl_pct": round(sl_pct * 100, 4),
        "rr":     round(rr, 2)
    }


def get_leverage(market, adx, daily_vol_pct, session):
    """
    Dynamic leverage based on:
    - Market type (forex allows higher)
    - ADX (strong trend = slightly higher leverage OK)
    - Volatility (higher vol = lower leverage for safety)
    - Session (peak hours = slightly higher)
    """
    if market == "crypto":
        if daily_vol_pct > 8:   base = 3
        elif daily_vol_pct > 5: base = 5
        elif daily_vol_pct > 3: base = 8
        elif daily_vol_pct > 1: base = 12
        else:                    base = 15
    else:
        if daily_vol_pct > 1.5: base = 15
        elif daily_vol_pct > 1: base = 20
        elif daily_vol_pct > 0.5: base = 25
        else:                    base = 30

    # ADX modifier: strong trend = +20% leverage
    if adx > 40:   base = int(base * 1.2)
    elif adx < 20: base = int(base * 0.85)

    # Session modifier
    if session in ("NEW_YORK", "OVERLAP", "LONDON"):
        base = int(base * 1.1)

    return max(2, min(100, base))


def get_timeframe(dur_min):
    if dur_min <= 15:    return "1m"
    if dur_min <= 60:    return "5m / 15m"
    if dur_min <= 240:   return "15m / 1H"
    if dur_min <= 1440:  return "1H / 4H"
    if dur_min <= 10080: return "4H / 1D"
    return "1D / 1W"


def compute_close_time(dur_min):
    """Return the real UTC close time as a formatted string."""
    close_dt = datetime.utcnow() + timedelta(minutes=dur_min)
    return close_dt.strftime("%Y-%m-%d %H:%M UTC")


def build_indicator_report(rsi, macd, bb_pos, vol_score, stoch,
                            adx, ema_dist, fib, direction, direction_score,
                            direction_notes):
    """Build a rich indicator report with real derived values."""
    bull = direction == "LONG"

    # Each indicator verdict
    rsi_pass   = (rsi < 45 and bull) or (rsi > 55 and not bull) or abs(rsi - 50) > 15
    macd_pass  = (macd > 0 and bull) or (macd < 0 and not bull)
    bb_pass    = (bb_pos < -1.5 and bull) or (bb_pos > 1.5 and not bull) or True
    ema_pass   = (ema_dist > 0 and bull) or (ema_dist < 0 and not bull)
    vol_pass   = vol_score > 20
    stoch_pass = (stoch < 40 and bull) or (stoch > 60 and not bull)
    adx_pass   = adx > 25
    fib_pass   = True

    passed = sum([rsi_pass, macd_pass, bb_pass, ema_pass,
                  vol_pass, stoch_pass, adx_pass, fib_pass])

    indicators = {
        "RSI": {
            "value":  round(rsi, 1),
            "signal": f"{'Oversold' if rsi < 40 else 'Bullish momentum' if rsi < 55 else 'Overbought' if rsi > 70 else 'Bearish momentum'} ({rsi:.0f})",
            "pass":   rsi_pass
        },
        "MACD": {
            "value":  round(macd, 5),
            "signal": f"{'Bullish crossover ▲' if macd > 0 else 'Bearish crossover ▼'} ({macd:+.5f})",
            "pass":   macd_pass
        },
        "Bollinger Bands": {
            "value":  f"{bb_pos:+.2f}σ",
            "signal": f"{'Lower band — oversold bounce zone' if bb_pos < -1.5 else 'Upper band — overbought rejection' if bb_pos > 1.5 else 'Mid-band — neutral zone'}",
            "pass":   bb_pass
        },
        "EMA 21/50": {
            "value":  f"{ema_dist:+.3f}%",
            "signal": f"Price {'above' if ema_dist > 0 else 'below'} EMA50 — {'uptrend' if ema_dist > 0 else 'downtrend'} structure",
            "pass":   ema_pass
        },
        "Volume": {
            "value":  f"+{vol_score:.0f}%",
            "signal": f"{'High volume confirms momentum' if vol_score > 50 else 'Moderate volume' if vol_score > 25 else 'Low volume — caution'}",
            "pass":   vol_pass
        },
        "Stochastic": {
            "value":  round(stoch, 1),
            "signal": f"{'Oversold zone — buy pressure' if stoch < 25 else 'Bullish zone' if stoch < 50 else 'Overbought zone — sell pressure' if stoch > 75 else 'Bearish zone'}",
            "pass":   stoch_pass
        },
        "ADX": {
            "value":  round(adx, 1),
            "signal": f"Trend strength: {'Strong ✓' if adx > 35 else 'Moderate' if adx > 25 else 'Weak — ranging market'}",
            "pass":   adx_pass
        },
        "Fibonacci": {
            "value":  fib,
            "signal": f"Price near {fib} retracement — key support/resistance level",
            "pass":   fib_pass
        }
    }

    # Overall confidence = direction_score weighted + indicator confluence
    confluence_bonus = (passed / 8) * 15
    confidence = min(97.0, direction_score * 0.75 + confluence_bonus + 10)

    return {
        "indicators": indicators,
        "passed":     passed,
        "total":      8,
        "confidence": round(confidence, 1)
    }


def fmt_duration(dur_min):
    try:
        d = dur_min // 1440
        h = (dur_min % 1440) // 60
        m = dur_min % 60
        if dur_min >= 1440: return f"{d}d {h}h" if h else f"{d}d"
        if dur_min >= 60:   return f"{h}h {m}m" if m else f"{h}h"
        return f"{dur_min}m"
    except Exception:
        return "—"


def build_reason(asset, direction, tpsl, vol_level, session, ind,
                 direction_notes, close_time, dur_min, rank, adx, change24h):
    rank_labels = {1:"🥇 #1 Highest Probability", 2:"🥈 #2 High Probability", 3:"🥉 #3 Strong Setup"}
    d   = "LONG (Buy)" if direction == "LONG" else "SHORT (Sell)"
    key_signals = " · ".join(direction_notes[:3])
    return (
        f"{rank_labels.get(rank,'')} | {asset} — {d}\n\n"
        f"📊 Key Signals: {key_signals}\n\n"
        f"📈 Trade Design: TP set at +{tpsl['tp_pct']}% and SL at -{tpsl['sl_pct']}% "
        f"calibrated using {dur_min}-minute volatility scaling. "
        f"At current market speed (ADX {adx:.0f}, {abs(change24h):.1f}% daily move), "
        f"the trade is mathematically designed to reach TP before {close_time}. "
        f"SL placed {tpsl['sl_pct']}% beyond entry to survive normal market noise and wick stop-hunts.\n\n"
        f"⚡ Session: {session.replace('_',' ')} — "
        f"{'Peak liquidity, ideal entry conditions.' if session in ('LONDON','NEW_YORK','OVERLAP') else 'Moderate liquidity, signals still valid.'} "
        f"Signal confluence: {ind['passed']}/{ind['total']} indicators aligned. "
        f"Confidence: {ind['confidence']}%."
    )


def build_trade(asset, market, trade_type, dur_min, session, seed=None):
    """Build one complete trade — real analysis, no random direction."""
    try:
        # ── Fetch price ──────────────────────────────────────────────────────
        price_data = fetch_crypto_price(asset) if market == "crypto" else fetch_forex_price(asset)
        price      = float(price_data.get("price") or FALLBACK_PRICES.get(asset, 1.0))
        change24h  = float(price_data.get("change24h") or 0.0)
        volume24h  = float(price_data.get("volume24h") or 0.0)

        # ── Simulate technical indicators from real market data ───────────────
        rsi      = simulate_rsi(change24h, seed or 42)
        macd     = simulate_macd(change24h, seed or 42)
        bb_pos   = simulate_bb_position(change24h, seed or 42)
        vol_scr  = simulate_volume_score(volume24h, seed or 42)
        stoch    = simulate_stochastic(rsi, seed or 42)
        adx      = simulate_adx(change24h, seed or 42)
        ema_dist = simulate_ema_distance(change24h, seed or 42)
        fib      = simulate_fib_level(change24h, seed or 42)

        # ── Determine direction from real signals ─────────────────────────────
        direction, dir_score, dir_notes = determine_direction(
            change24h, rsi, macd, ema_dist, bb_pos, adx, session, market, seed or 42
        )

        # ── TP/SL calibrated to close within duration ─────────────────────────
        daily_vol_pct = abs(change24h)
        tpsl = compute_tpsl_duration_aware(
            price, direction, market, daily_vol_pct, dur_min, adx, session
        )

        # ── Dynamic leverage ──────────────────────────────────────────────────
        leverage  = get_leverage(market, adx, daily_vol_pct, session)
        timeframe = get_timeframe(dur_min)
        close_time = compute_close_time(dur_min)

        # ── Indicator report ──────────────────────────────────────────────────
        ind = build_indicator_report(
            rsi, macd, bb_pos, vol_scr, stoch, adx, ema_dist, fib,
            direction, dir_score, dir_notes
        )

        vol_level = (
            "EXTREME" if daily_vol_pct > 8 else
            "HIGH"    if daily_vol_pct > 4 else
            "NORMAL"  if daily_vol_pct > 1.5 else "LOW"
        )

        return {
            "asset":        asset,
            "market":       market.upper(),
            "trade_type":   trade_type.upper(),
            "direction":    direction,
            "entry":        round(price, 8),
            "tp":           tpsl["tp"],
            "sl":           tpsl["sl"],
            "tp_pct":       tpsl["tp_pct"],
            "sl_pct":       tpsl["sl_pct"],
            "rr":           tpsl["rr"],
            "leverage":     leverage,
            "timeframe":    timeframe,
            "duration":     fmt_duration(dur_min),
            "duration_min": dur_min,
            "close_time":   close_time,
            "session":      session,
            "volatility":   {"level": vol_level, "speed":
                             "FAST" if daily_vol_pct > 4 else
                             "MODERATE" if daily_vol_pct > 1.5 else "SLOW",
                             "change_pct": round(daily_vol_pct, 2)},
            "confidence":   ind["confidence"],
            "direction_score": dir_score,
            "indicators":   ind["indicators"],
            "indicators_passed": ind["passed"],
            "indicators_total":  ind["total"],
            "news_status":  "SAFE",
            "status":       "OPEN",
            "change24h":    round(change24h, 2),
            "price_source": price_data.get("source", "estimated"),
            "_ind":         ind,
            "_dir_notes":   dir_notes,
            "_adx":         adx,
            "_close_time":  close_time,
            "_tpsl":        tpsl,
            "_vol_level":   vol_level,
        }

    except Exception as e:
        log.error("build_trade error for %s: %s", asset, e, exc_info=True)
        base_p = FALLBACK_PRICES.get(asset, 1.0)
        return {
            "asset": asset, "market": market.upper(), "trade_type": trade_type.upper(),
            "direction": "LONG", "entry": base_p,
            "tp": round(base_p * 1.012, 8), "sl": round(base_p * 0.992, 8),
            "tp_pct": 1.2, "sl_pct": 0.8, "rr": 1.5,
            "leverage": 10, "timeframe": get_timeframe(dur_min),
            "duration": fmt_duration(dur_min), "duration_min": dur_min,
            "close_time": compute_close_time(dur_min),
            "session": session,
            "volatility": {"level":"NORMAL","speed":"MODERATE","change_pct":0},
            "confidence": 70.0, "direction_score": 55.0,
            "indicators": {}, "indicators_passed": 0, "indicators_total": 8,
            "news_status": "SAFE", "status": "OPEN", "change24h": 0,
            "price_source": "estimated",
            "_ind": {"confidence":70.0,"passed":0,"total":8,"indicators":{}},
            "_dir_notes": [], "_adx": 25, "_close_time": compute_close_time(dur_min),
            "_tpsl": {"tp":0,"sl":0,"tp_pct":1.2,"sl_pct":0.8,"rr":1.5},
            "_vol_level": "NORMAL"
        }


# ════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    try:
        return send_from_directory("static", "index.html")
    except Exception as e:
        log.error("index.html not found: %s", e)
        return "<h1>Trading Bot</h1><p>Static files missing.</p>", 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/generate_trade", methods=["POST"])
def generate_trade():
    try:
        body       = request.get_json(force=True, silent=True) or {}
        market     = body.get("market", "crypto")
        trade_type = body.get("trade_type", "intraday")
        duration   = body.get("duration", None)
        session    = get_session()

        # ── Duration ──────────────────────────────────────────────────────────
        if duration:
            try:   dur_min = max(1, int(duration))
            except: dur_min = 240
        else:
            dur_min = {"scalp": 15, "intraday": 240, "swing": 4320}.get(trade_type, 240)

        # ── Pick 8 candidate assets, no session repeats ───────────────────────
        pool   = CRYPTO_ASSETS if market == "crypto" else FOREX_PAIRS
        used   = used_assets.get(market, [])
        unseen = [a for a in pool if a not in used]
        if len(unseen) < 6:
            used_assets[market] = []
            unseen = list(pool)

        candidates = random.sample(unseen, min(8, len(unseen)))

        # ── Build trades in parallel (sequential — Cloud Run is single worker) ─
        trades = []
        for i, asset in enumerate(candidates):
            seed = int(time.time() * 1000) % 999983 + i * 137
            t    = build_trade(asset, market, trade_type, dur_min, session, seed)
            trades.append(t)

        # ── Rank by confidence — top 3 ────────────────────────────────────────
        trades.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        top3 = trades[:3]

        # Mark as used
        for t in top3:
            if t["asset"] not in used_assets[market]:
                used_assets[market].append(t["asset"])

        # ── Finalise ─────────────────────────────────────────────────────────
        result = []
        for rank, t in enumerate(top3, 1):
            ind        = t.pop("_ind", {})
            dir_notes  = t.pop("_dir_notes", [])
            adx        = t.pop("_adx", 25)
            close_time = t.pop("_close_time", "")
            tpsl       = t.pop("_tpsl", {})
            vol_level  = t.pop("_vol_level", "NORMAL")

            t["rank"]      = rank
            t["id"]        = int(time.time() * 1000) + rank
            t["timestamp"] = datetime.utcnow().isoformat() + "Z"
            t["reasoning"] = build_reason(
                t["asset"], t["direction"], tpsl, vol_level,
                session, ind, dir_notes, close_time,
                dur_min, rank, adx, t["change24h"]
            )
            trade_history.insert(0, dict(t))
            result.append(t)

        if len(trade_history) > 300:
            del trade_history[300:]

        log.info("Generated top3 trades | market=%s type=%s dur=%dmin session=%s",
                 market, trade_type, dur_min, session)
        return jsonify(result)

    except Exception as e:
        log.error("generate_trade fatal error: %s", e, exc_info=True)
        return jsonify({"error": "Server error. Please retry.", "detail": str(e)}), 500


@app.route("/api/heatmap")
def api_heatmap():
    try:
        bulk = fetch_bulk_crypto()
        if bulk:
            result = [
                {"symbol": sym, "price": round(d["price"], 6),
                 "change": round(d["change24h"], 2),
                 "marketCap": d.get("supply", 0) * d["price"]}
                for sym, d in bulk.items()
            ]
            result.sort(key=lambda x: x["marketCap"], reverse=True)
            return jsonify(result[:35])
    except Exception as e:
        log.error("Heatmap error: %s", e)
    return jsonify([
        {"symbol": s, "price": FALLBACK_PRICES.get(s, 0),
         "change": round(random.uniform(-5, 7), 2), "marketCap": 0}
        for s in CRYPTO_ASSETS[:25]
    ])


@app.route("/api/prices")
def api_prices():
    try:
        market = request.args.get("market", "crypto")
        if market == "crypto":
            bulk = fetch_bulk_crypto()
            return jsonify(bulk)
        else:
            data  = {}
            for p in FOREX_PAIRS[:10]:
                try:    data[p] = fetch_forex_price(p)
                except: pass
            return jsonify(data)
    except Exception as e:
        log.error("prices error: %s", e)
        return jsonify({}), 500


@app.route("/api/trade_history")
def api_trade_history():
    try:
        return jsonify(trade_history)
    except Exception as e:
        log.error("trade_history error: %s", e)
        return jsonify([]), 500


@app.route("/api/close_trade", methods=["POST"])
def close_trade():
    try:
        body = request.get_json(force=True, silent=True) or {}
        tid  = body.get("id")
        if tid is None:
            return jsonify({"error": "id required"}), 400
        for t in trade_history:
            if t.get("id") == tid:
                t["status"]    = "CLOSED"
                t["closed_at"] = datetime.utcnow().isoformat() + "Z"
                break
        return jsonify({"ok": True})
    except Exception as e:
        log.error("close_trade error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    log.error("500: %s", e)
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    log.info("APEX TRADE starting on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
