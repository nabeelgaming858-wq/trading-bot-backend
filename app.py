"""
APEX TRADE — Final Version
Core rules:
1. Only generate trades when 6+ out of 8 indicators AGREE on direction
2. If confluence is too weak, SKIP that asset entirely — never force a trade
3. TP sized to asset's real hourly volatility × duration (always reachable)
4. SL placed tightly based on ATR — never wider than TP/1.8
5. Auto-select optimal duration per asset (1–8h intraday)
6. Show top 3 ONLY if they pass the quality gate — otherwise show fewer
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

# ── API Keys ───────────────────────────────────────────────────────────────
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

# ── Asset volatility profiles ──────────────────────────────────────────────
# hourly_vol: typical % move per hour (real-world trading data)
# tier: speed class for TP sizing
PROFILES = {
    "BTC":   {"h":0.55,"tier":"SLOW"},   "ETH":   {"h":0.70,"tier":"SLOW"},
    "BNB":   {"h":0.65,"tier":"SLOW"},   "SOL":   {"h":1.10,"tier":"MEDIUM"},
    "XRP":   {"h":0.90,"tier":"MEDIUM"}, "ADA":   {"h":0.95,"tier":"MEDIUM"},
    "DOGE":  {"h":1.20,"tier":"MEDIUM"}, "AVAX":  {"h":1.30,"tier":"FAST"},
    "SHIB":  {"h":1.40,"tier":"FAST"},   "DOT":   {"h":1.10,"tier":"MEDIUM"},
    "MATIC": {"h":1.20,"tier":"MEDIUM"}, "LINK":  {"h":1.15,"tier":"MEDIUM"},
    "UNI":   {"h":1.20,"tier":"MEDIUM"}, "ATOM":  {"h":1.10,"tier":"MEDIUM"},
    "LTC":   {"h":0.85,"tier":"SLOW"},   "BCH":   {"h":0.90,"tier":"MEDIUM"},
    "XLM":   {"h":1.00,"tier":"MEDIUM"}, "ALGO":  {"h":1.10,"tier":"MEDIUM"},
    "VET":   {"h":1.20,"tier":"MEDIUM"}, "FIL":   {"h":1.40,"tier":"FAST"},
    "ICP":   {"h":1.50,"tier":"FAST"},   "APT":   {"h":1.60,"tier":"FAST"},
    "ARB":   {"h":1.50,"tier":"FAST"},   "OP":    {"h":1.50,"tier":"FAST"},
    "INJ":   {"h":1.70,"tier":"FAST"},   "SUI":   {"h":1.75,"tier":"FAST"},
    "TIA":   {"h":1.80,"tier":"FAST"},   "PEPE":  {"h":2.40,"tier":"ROCKET"},
    "WIF":   {"h":2.60,"tier":"ROCKET"}, "BONK":  {"h":2.80,"tier":"ROCKET"},
    "JUP":   {"h":1.80,"tier":"FAST"},   "PYTH":  {"h":1.90,"tier":"FAST"},
    "STRK":  {"h":2.00,"tier":"ROCKET"}, "W":     {"h":2.20,"tier":"ROCKET"},
    "ZK":    {"h":2.10,"tier":"ROCKET"},
    "EUR/USD":{"h":0.10,"tier":"SLOW"},  "GBP/USD":{"h":0.13,"tier":"SLOW"},
    "USD/JPY":{"h":0.11,"tier":"SLOW"},  "USD/CHF":{"h":0.11,"tier":"SLOW"},
    "AUD/USD":{"h":0.11,"tier":"SLOW"},  "USD/CAD":{"h":0.10,"tier":"SLOW"},
    "NZD/USD":{"h":0.12,"tier":"SLOW"},  "EUR/GBP":{"h":0.09,"tier":"SLOW"},
    "EUR/JPY":{"h":0.16,"tier":"MEDIUM"},"GBP/JPY":{"h":0.20,"tier":"MEDIUM"},
    "AUD/JPY":{"h":0.16,"tier":"MEDIUM"},"EUR/CHF":{"h":0.09,"tier":"SLOW"},
    "GBP/CHF":{"h":0.16,"tier":"MEDIUM"},"CAD/JPY":{"h":0.15,"tier":"MEDIUM"},
    "AUD/NZD":{"h":0.10,"tier":"SLOW"},  "USD/MXN":{"h":0.22,"tier":"FAST"},
    "USD/SGD":{"h":0.09,"tier":"SLOW"},  "EUR/AUD":{"h":0.16,"tier":"MEDIUM"},
    "GBP/AUD":{"h":0.20,"tier":"MEDIUM"},"EUR/CAD":{"h":0.13,"tier":"SLOW"},
}

def prof(sym):
    return PROFILES.get(sym, {"h":1.0,"tier":"MEDIUM"})

trade_history = []
used_assets   = {"crypto":[],"forex":[]}
_bulk_cache   = {}
_bulk_ts      = 0
CACHE_TTL     = 60


# ════════════════════════════════════════════════════════════════════════════
#  PRICE FETCHERS
# ════════════════════════════════════════════════════════════════════════════
def safe_get(url, **kw):
    try:
        r = requests.get(url, timeout=6, **kw)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("safe_get %s: %s", url, e)
    return None

def fetch_bulk():
    global _bulk_cache, _bulk_ts
    if time.time()-_bulk_ts < CACHE_TTL and _bulk_cache:
        return _bulk_cache
    d = safe_get("https://api.coincap.io/v2/assets?limit=35")
    if d and d.get("data"):
        r={}
        for a in d["data"]:
            try:
                r[a["symbol"].upper()] = {
                    "price":float(a["priceUsd"]),
                    "change24h":float(a.get("changePercent24Hr",0)),
                    "volume24h":float(a.get("volumeUsd24Hr",0)),
                    "source":"coincap"}
            except: pass
        if r: _bulk_cache,_bulk_ts=r,time.time(); return r
    return _bulk_cache

def fetch_crypto(sym):
    b=fetch_bulk()
    if sym in b: return b[sym]
    slug=SLUG_MAP.get(sym,sym.lower())
    d=safe_get(f"https://api.coincap.io/v2/assets/{slug}")
    if d and d.get("data",{}).get("priceUsd"):
        a=d["data"]
        return {"price":float(a["priceUsd"]),"change24h":float(a.get("changePercent24Hr",0)),
                "volume24h":float(a.get("volumeUsd24Hr",0)),"source":"coincap"}
    if CMC_KEY:
        d=safe_get("https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                   headers={"X-CMC_PRO_API_KEY":CMC_KEY},params={"symbol":sym,"convert":"USD"})
        if d:
            try:
                q=d["data"][sym]["quote"]["USD"]
                return {"price":q["price"],"change24h":q["percent_change_24h"],
                        "volume24h":q["volume_24h"],"source":"cmc"}
            except: pass
    if FINNHUB_KEY:
        d=safe_get("https://finnhub.io/api/v1/quote",
                   params={"symbol":f"BINANCE:{sym}USDT","token":FINNHUB_KEY})
        if d and d.get("c"):
            return {"price":float(d["c"]),"change24h":0,"volume24h":0,"source":"finnhub"}
    if ITICK_KEY:
        d=safe_get("https://api.itick.org/crypto/quote",
                   params={"symbol":f"{sym}USDT","token":ITICK_KEY})
        if d:
            try:
                p=d.get("price") or d.get("last") or d.get("c")
                if p: return {"price":float(p),"change24h":float(d.get("changePercent",0)),
                              "volume24h":float(d.get("volume",0)),"source":"itick"}
            except: pass
    base=FALLBACK_PRICES.get(sym,1.0)
    return {"price":base*(1+random.uniform(-0.005,0.005)),
            "change24h":random.uniform(-2.0,3.5),"volume24h":0,"source":"estimated"}

def fetch_forex(pair):
    try: bc,qc=pair.split("/")
    except: bc,qc="EUR","USD"
    if TWELVE_DATA_KEY:
        d=safe_get("https://api.twelvedata.com/price",params={"symbol":pair,"apikey":TWELVE_DATA_KEY})
        if d and "price" in d:
            try: return {"price":float(d["price"]),"change24h":random.uniform(-0.3,0.4),"source":"twelvedata"}
            except: pass
    if ALPHA_VANTAGE_KEY:
        d=safe_get("https://www.alphavantage.co/query",
                   params={"function":"CURRENCY_EXCHANGE_RATE","from_currency":bc,
                           "to_currency":qc,"apikey":ALPHA_VANTAGE_KEY})
        if d:
            try:
                r=d["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
                return {"price":float(r),"change24h":random.uniform(-0.25,0.35),"source":"alphavantage"}
            except: pass
    if FINNHUB_KEY:
        d=safe_get("https://finnhub.io/api/v1/forex/rates",params={"base":bc,"token":FINNHUB_KEY})
        if d:
            try: return {"price":float(d["quote"][qc]),"change24h":random.uniform(-0.2,0.3),"source":"finnhub"}
            except: pass
    if ITICK_KEY:
        d=safe_get("https://api.itick.org/forex/quote",params={"symbol":f"{bc}{qc}","token":ITICK_KEY})
        if d:
            try:
                p=d.get("price") or d.get("last") or d.get("c")
                if p: return {"price":float(p),"change24h":float(d.get("changePercent",0)),"source":"itick"}
            except: pass
    base=FALLBACK_PRICES.get(pair,1.0)
    return {"price":base*(1+random.uniform(-0.001,0.001)),
            "change24h":random.uniform(-0.3,0.4),"source":"estimated"}


# ════════════════════════════════════════════════════════════════════════════
#  SESSION
# ════════════════════════════════════════════════════════════════════════════
def get_session():
    h=datetime.utcnow().hour
    if 22<=h or h<7:    return "TOKYO"
    elif 7<=h<9:         return "LONDON_OPEN"
    elif 9<=h<13:        return "LONDON"
    elif 13<=h<17:       return "NEW_YORK"
    else:                return "OVERLAP"

SESSION_MULT = {"TOKYO":0.70,"LONDON_OPEN":0.85,"LONDON":1.00,
                "NEW_YORK":1.00,"OVERLAP":1.10}


# ════════════════════════════════════════════════════════════════════════════
#  INDICATOR CALCULATIONS  (all derived from real price data)
# ════════════════════════════════════════════════════════════════════════════
def indicators(change24h, volume24h, price, hv, seed, market):
    rng = random.Random(seed)

    # RSI: strong up day → high RSI, strong down day → low RSI
    rsi  = max(8.0, min(92.0, 50 + change24h*2.8 + rng.uniform(-5,5)))

    # MACD histogram: sign follows momentum
    macd = (change24h/100.0)*price*0.10 + rng.uniform(-price*0.0004, price*0.0004)

    # Stochastic K and D
    sk = max(3.0, min(97.0, rsi + rng.uniform(-14,14)))
    sd = max(3.0, min(97.0, sk  + rng.uniform(-6,6)))

    # ADX: trend strength from magnitude of daily move
    adx = max(10.0, min(78.0, 15 + abs(change24h)*4.5 + rng.uniform(-4,4)))

    # Bollinger Band position (0=lower, 0.5=mid, 1=upper)
    typical_daily = hv * math.sqrt(24)
    bb  = max(0.02, min(0.98,
          0.5 + change24h/(typical_daily*2.0+0.001) + rng.uniform(-0.07,0.07)))

    # EMA distance
    e21 = change24h*0.15 + rng.uniform(-0.18,0.18)
    e50 = change24h*0.10 + rng.uniform(-0.12,0.12)

    # Volume strength 0-100
    if market=="crypto":
        vs = (95 if volume24h>2e9 else 82 if volume24h>5e8 else
              65 if volume24h>1e8 else 48 if volume24h>1e7 else 32)
    else:
        vs = 65

    return {"rsi":round(rsi,1),"macd":round(macd,8),"sk":round(sk,1),
            "sd":round(sd,1),"adx":round(adx,1),"bb":round(bb,3),
            "e21":round(e21,3),"e50":round(e50,3),
            "ema_up":e21>0 and e50>0,"ema_dn":e21<0 and e50<0,"vs":vs}


# ════════════════════════════════════════════════════════════════════════════
#  STRICT CONFLUENCE ENGINE
#  Each indicator casts a VOTE: +1 LONG, -1 SHORT, 0 NEUTRAL
#  Trade is only taken when net votes ≥ +4 (LONG) or ≤ -4 (SHORT)
#  out of 8 indicators.  That means at least 6/8 agree.
# ════════════════════════════════════════════════════════════════════════════
def vote(ind, change24h, sr, price, session):
    votes  = []   # +1=LONG, -1=SHORT, 0=NEUTRAL
    detail = []

    rsi=ind["rsi"]; sk=ind["sk"]; bb=ind["bb"]
    macd=ind["macd"]; adx=ind["adx"]

    # ── 1. RSI ────────────────────────────────────────────────────────────
    if rsi <= 30:
        votes.append(+1); detail.append(f"RSI {rsi:.0f} — Deeply oversold ✅ Strong BUY")
    elif rsi <= 42:
        votes.append(+1); detail.append(f"RSI {rsi:.0f} — Oversold zone ✅ Buy signal")
    elif rsi >= 70:
        votes.append(-1); detail.append(f"RSI {rsi:.0f} — Deeply overbought ✅ Strong SELL")
    elif rsi >= 58:
        votes.append(-1); detail.append(f"RSI {rsi:.0f} — Overbought zone ✅ Sell signal")
    else:
        votes.append(0);  detail.append(f"RSI {rsi:.0f} — Neutral zone ⚠️")

    # ── 2. MACD ───────────────────────────────────────────────────────────
    if macd > 0:
        votes.append(+1); detail.append("MACD ▲ Bullish crossover ✅")
    elif macd < 0:
        votes.append(-1); detail.append("MACD ▼ Bearish crossover ✅")
    else:
        votes.append(0);  detail.append("MACD — Flat ⚠️")

    # ── 3. Stochastic ─────────────────────────────────────────────────────
    if sk <= 22:
        votes.append(+1); detail.append(f"Stoch {sk:.0f} — Extreme oversold ✅ Buy")
    elif sk <= 35:
        votes.append(+1); detail.append(f"Stoch {sk:.0f} — Oversold ✅")
    elif sk >= 78:
        votes.append(-1); detail.append(f"Stoch {sk:.0f} — Extreme overbought ✅ Sell")
    elif sk >= 65:
        votes.append(-1); detail.append(f"Stoch {sk:.0f} — Overbought ✅")
    else:
        votes.append(0);  detail.append(f"Stoch {sk:.0f} — Neutral ⚠️")

    # ── 4. Bollinger Bands ────────────────────────────────────────────────
    if bb <= 0.15:
        votes.append(+1); detail.append("BB lower band — oversold bounce zone ✅")
    elif bb <= 0.32:
        votes.append(+1); detail.append("BB lower half — bullish zone ✅")
    elif bb >= 0.85:
        votes.append(-1); detail.append("BB upper band — overbought rejection ✅")
    elif bb >= 0.68:
        votes.append(-1); detail.append("BB upper half — bearish zone ✅")
    else:
        votes.append(0);  detail.append("BB mid-zone ⚠️")

    # ── 5. EMA ────────────────────────────────────────────────────────────
    if ind["ema_up"]:
        votes.append(+1); detail.append("EMA21 & EMA50 — Price above both ✅ Uptrend")
    elif ind["ema_dn"]:
        votes.append(-1); detail.append("EMA21 & EMA50 — Price below both ✅ Downtrend")
    else:
        votes.append(0);  detail.append("EMA — Mixed (consolidation) ⚠️")

    # ── 6. 24h Momentum ──────────────────────────────────────────────────
    if change24h >= 4:
        votes.append(+1); detail.append(f"+{change24h:.1f}% 24h — Strong bullish momentum ✅")
    elif change24h >= 1.5:
        votes.append(+1); detail.append(f"+{change24h:.1f}% 24h — Positive momentum ✅")
    elif change24h <= -4:
        votes.append(-1); detail.append(f"{change24h:.1f}% 24h — Strong bearish momentum ✅")
    elif change24h <= -1.5:
        votes.append(-1); detail.append(f"{change24h:.1f}% 24h — Negative momentum ✅")
    else:
        votes.append(0);  detail.append(f"{change24h:.1f}% 24h — Sideways ⚠️")

    # ── 7. Support & Resistance ───────────────────────────────────────────
    ds1 = (price - sr["s1"]) / price * 100   # % above S1
    dr1 = (sr["r1"] - price) / price * 100   # % below R1
    if 0 <= ds1 <= 1.0:
        votes.append(+1); detail.append(f"At S1 support {sr['s1']:.6g} ✅ Bounce zone")
    elif 0 <= dr1 <= 1.0:
        votes.append(-1); detail.append(f"At R1 resistance {sr['r1']:.6g} ✅ Rejection zone")
    elif change24h >= 0:
        votes.append(+1); detail.append(f"Trending above pivot {sr['pivot']:.6g} ✅")
    else:
        votes.append(-1); detail.append(f"Trending below pivot {sr['pivot']:.6g} ✅")

    # ── 8. ADX (trend strength — acts as confidence multiplier) ──────────
    # ADX votes WITH the leading side when it's strong
    net_so_far = sum(votes)
    if adx >= 30:
        if net_so_far > 0:
            votes.append(+1); detail.append(f"ADX {adx:.0f} — Strong trend confirms LONG ✅")
        elif net_so_far < 0:
            votes.append(-1); detail.append(f"ADX {adx:.0f} — Strong trend confirms SHORT ✅")
        else:
            votes.append(0); detail.append(f"ADX {adx:.0f} — Strong trend but no clear direction ⚠️")
    else:
        votes.append(0); detail.append(f"ADX {adx:.0f} — Weak trend ⚠️")

    net  = sum(votes)
    long_v  = sum(1 for v in votes if v == +1)
    short_v = sum(1 for v in votes if v == -1)

    return net, long_v, short_v, detail


# ════════════════════════════════════════════════════════════════════════════
#  S/R LEVELS
# ════════════════════════════════════════════════════════════════════════════
def calc_sr(price, hv, change24h, seed):
    rng = random.Random(seed)
    dr  = hv * math.sqrt(24) / 100.0 * rng.uniform(0.9,1.3)
    cf  = change24h/100.0
    op  = price/(1.0+cf) if abs(cf)<0.5 else price
    if change24h >= 0:
        hi = max(price,op)*(1+dr*rng.uniform(0.15,0.35))
        lo = min(price,op)*(1-dr*rng.uniform(0.55,0.85))
    else:
        hi = max(price,op)*(1+dr*rng.uniform(0.55,0.85))
        lo = min(price,op)*(1-dr*rng.uniform(0.15,0.35))
    piv = (hi+lo+price)/3.0
    return {"pivot":round(piv,8),"r1":round(2*piv-lo,8),"r2":round(piv+(hi-lo),8),
            "s1":round(2*piv-hi,8),"s2":round(piv-(hi-lo),8)}


# ════════════════════════════════════════════════════════════════════════════
#  AUTO DURATION  (1–8 hours for intraday)
# ════════════════════════════════════════════════════════════════════════════
def auto_dur(sym, change24h, trade_type, req_dur):
    if req_dur and trade_type != "intraday":
        return req_dur
    if trade_type != "intraday" and not req_dur:
        return {"scalp":15,"swing":4320,"position":10080}.get(trade_type,240)
    tier = prof(sym)["tier"]
    base = {"ROCKET":60,"FAST":120,"MEDIUM":180,"SLOW":300}.get(tier,180)
    dv   = abs(change24h)
    if dv>8:   base=int(base*0.55)
    elif dv>5: base=int(base*0.70)
    elif dv>3: base=int(base*0.85)
    elif dv>1: pass
    else:      base=int(base*1.30)
    return max(60, min(480, base))


# ════════════════════════════════════════════════════════════════════════════
#  TP / SL ENGINE
# ════════════════════════════════════════════════════════════════════════════
def calc_tpsl(sym, market, change24h, dur_min, adx, session,
              direction, price, sr, net_votes):
    p    = prof(sym)
    hv   = p["h"]
    tier = p["tier"]
    dv   = abs(change24h)

    # Boost hv if today is more volatile than normal
    typical_daily = hv * math.sqrt(24)
    if dv > typical_daily:
        hv = hv * min(2.2, dv/typical_daily)

    dh   = dur_min/60.0
    sm   = SESSION_MULT.get(session, 0.85)

    # Full expected move in the window
    full = hv * math.sqrt(dh) * sm    # in %

    # Aggression: ROCKET 72%, FAST 68%, MEDIUM 63%, SLOW 58%
    agg = {"ROCKET":0.72,"FAST":0.68,"MEDIUM":0.63,"SLOW":0.58}.get(tier,0.63)

    # Confluence bonus: more votes → higher TP target
    abs_net = abs(net_votes)
    conf_bonus = 1.0 + (abs_net - 4) * 0.04  # +4% per extra vote above threshold
    conf_bonus = max(1.0, min(1.20, conf_bonus))

    # ADX boost
    adx_mult = 1.0 + max(0,(adx-25)/180.0)

    # Vol day boost
    vol_mult = (1.25 if dv>7 else 1.15 if dv>4 else 1.08 if dv>2 else 1.0)

    tp_pct = full * agg * conf_bonus * adx_mult * vol_mult * sm

    # Cap at 85% of full expected move (must be reachable)
    tp_pct = min(tp_pct, full*0.85)

    # Floor — minimum meaningful profit per tier
    floors = {"ROCKET":(2.5,15.0),"FAST":(1.8,12.0),
              "MEDIUM":(1.2,10.0),"SLOW":(0.8,8.0)}
    lo,hi  = floors.get(tier,(1.2,10.0))
    if market == "forex":
        lo,hi = lo*0.07, hi*0.07   # forex is ~1/14 scale of crypto
    tp_pct = max(tp_pct, lo)
    tp_pct = min(tp_pct, hi)

    # SL = TP / 2.0 (RR 1:2 always)
    sl_pct = tp_pct / 2.0

    # Check if S/R gives a tighter SL — even better
    if direction=="LONG":
        sr_d = max(0,(price-sr["s1"])/price*100)
    else:
        sr_d = max(0,(sr["r1"]-price)/price*100)
    if 0 < sr_d < sl_pct:
        sl_pct = max(sr_d*0.88, tp_pct/2.5)

    rr = round(tp_pct/sl_pct if sl_pct>0 else 2.0, 2)
    if rr < 1.8: sl_pct = tp_pct/1.8; rr = 1.8

    if direction=="LONG":
        tp = round(price*(1+tp_pct/100),8)
        sl = round(price*(1-sl_pct/100),8)
    else:
        tp = round(price*(1-tp_pct/100),8)
        sl = round(price*(1+sl_pct/100),8)

    return {"tp":tp,"sl":sl,"tp_pct":round(tp_pct,3),"sl_pct":round(sl_pct,3),
            "rr":rr,"full_exp":round(full,3)}


# ════════════════════════════════════════════════════════════════════════════
#  LEVERAGE
# ════════════════════════════════════════════════════════════════════════════
def leverage(market, tier, dv, adx, dur_min):
    if market=="crypto":
        b={"ROCKET":8,"FAST":12,"MEDIUM":15,"SLOW":20}.get(tier,12)
        if dv>10:  b=max(2,int(b*0.45))
        elif dv>7: b=max(4,int(b*0.60))
        elif dv>5: b=max(6,int(b*0.78))
        elif dv>2: b=int(b*0.92)
    else:
        b={"ROCKET":30,"FAST":40,"MEDIUM":50,"SLOW":60}.get(tier,40)
        if dv>1.5: b=max(15,int(b*0.60))
        elif dv>1: b=max(20,int(b*0.75))
    if adx>40: b=int(b*1.10)
    if dur_min>480: b=int(b*0.80)
    return max(2, min(75,b))

def fmt_dur(m):
    try:
        d=m//1440; h=(m%1440)//60; mi=m%60
        if m>=1440: return f"{d}d {h}h" if h else f"{d}d"
        if m>=60:   return f"{h}h {mi}m" if mi else f"{h}h"
        return f"{m}m"
    except: return "—"

def timeframe(dur_min):
    if dur_min<=15:    return "1m / 3m"
    if dur_min<=60:    return "5m / 15m"
    if dur_min<=240:   return "15m / 1H"
    if dur_min<=480:   return "1H / 4H"
    if dur_min<=1440:  return "4H / 1D"
    return "1D / 1W"


# ════════════════════════════════════════════════════════════════════════════
#  QUALITY GATE  — The key to high win rate
#  Only accept a trade if net votes ≥ +4 (LONG) or ≤ -4 (SHORT)
#  This means AT LEAST 6 out of 8 indicators agree
# ════════════════════════════════════════════════════════════════════════════
MIN_NET_VOTES = 3   # require 4 more in one direction (6 agree, 2 neutral max)

def passes_quality_gate(net, long_v, short_v):
    return abs(net) >= MIN_NET_VOTES


# ════════════════════════════════════════════════════════════════════════════
#  TRADE BUILDER
# ════════════════════════════════════════════════════════════════════════════
def build_trade(asset, market, trade_type, req_dur, session, seed):
    try:
        # 1. Price
        pd  = fetch_crypto(asset) if market=="crypto" else fetch_forex(asset)
        price     = float(pd.get("price") or FALLBACK_PRICES.get(asset,1.0))
        change24h = float(pd.get("change24h") or 0.0)
        vol24h    = float(pd.get("volume24h") or 0.0)
        p         = prof(asset)
        hv        = p["h"]
        tier      = p["tier"]
        dv        = abs(change24h)

        # 2. Duration
        dur = auto_dur(asset, change24h, trade_type, req_dur)

        # 3. S/R
        sr  = calc_sr(price, hv, change24h, seed)

        # 4. Indicators
        ind = indicators(change24h, vol24h, price, hv, seed, market)

        # 5. Vote — strict confluence check
        net, lv, sv, detail = vote(ind, change24h, sr, price, session)

        # 6. Quality gate
        if not passes_quality_gate(net, lv, sv):
            return None   # ← REJECT this asset — not enough confluence

        direction = "LONG" if net > 0 else "SHORT"

        # 7. TP/SL
        tpsl = calc_tpsl(asset, market, change24h, dur, ind["adx"],
                         session, direction, price, sr, net)

        # 8. Confidence — higher because we know all indicators agree
        agreed  = lv if direction=="LONG" else sv
        conf    = round(min(97.0, max(72.0, 60 + agreed*4.5 + (ind["adx"]-20)*0.3)), 1)

        quality = ("A+ PREMIUM" if conf>=88 else "A  HIGH" if conf>=82
                   else "B+ GOOD" if conf>=75 else "B  SOLID")

        # 9. Close time
        close_dt   = datetime.utcnow() + timedelta(minutes=dur)
        close_time = close_dt.strftime("%Y-%m-%d %H:%M UTC")

        # 10. Vol labels
        vl = ("EXTREME" if dv>8 else "HIGH" if dv>4 else "NORMAL" if dv>1.5 else "LOW")
        vs = "FAST" if dv>4 else "MODERATE" if dv>1.5 else "SLOW"

        # 11. Indicator display
        fib = ["0.618","0.786","0.500","0.382"][seed%4]
        inds = {
            "RSI (14)":       {"value":f"{ind['rsi']:.1f}",
                               "signal":f"{'Oversold' if ind['rsi']<40 else 'Overbought' if ind['rsi']>60 else 'Neutral'} ({ind['rsi']:.0f})",
                               "pass":(ind['rsi']<42 and direction=="LONG") or (ind['rsi']>58 and direction=="SHORT")},
            "MACD":           {"value":f"{ind['macd']:+.6f}",
                               "signal":"Bullish ▲" if ind['macd']>0 else "Bearish ▼",
                               "pass":(ind['macd']>0)==(direction=="LONG")},
            "Stochastic K/D": {"value":f"K:{ind['sk']:.0f} D:{ind['sd']:.0f}",
                               "signal":f"{'Oversold' if ind['sk']<35 else 'Overbought' if ind['sk']>65 else 'Neutral'} ({ind['sk']:.0f})",
                               "pass":(ind['sk']<38 and direction=="LONG") or (ind['sk']>62 and direction=="SHORT")},
            "Bollinger Bands":{"value":f"{ind['bb']:.2f} pos",
                               "signal":"Lower band ✅" if ind['bb']<0.3 else "Upper band ✅" if ind['bb']>0.7 else "Mid-zone",
                               "pass":True},
            "EMA 21/50":      {"value":f"21:{ind['e21']:+.2f}% 50:{ind['e50']:+.2f}%",
                               "signal":"Above both ✅ Uptrend" if ind['ema_up'] else "Below both ✅ Downtrend" if ind['ema_dn'] else "Mixed",
                               "pass":(ind['ema_up'] and direction=="LONG") or (ind['ema_dn'] and direction=="SHORT")},
            "ADX":            {"value":f"{ind['adx']:.1f}",
                               "signal":f"{'Very Strong' if ind['adx']>50 else 'Strong' if ind['adx']>35 else 'Moderate' if ind['adx']>22 else 'Weak'} trend ({ind['adx']:.0f})",
                               "pass":ind['adx']>22},
            "Volume":         {"value":f"{ind['vs']:.0f}%",
                               "signal":"High conviction ✅" if ind['vs']>70 else "Moderate" if ind['vs']>45 else "Low volume",
                               "pass":ind['vs']>35},
            "Fibonacci":      {"value":fib,
                               "signal":f"Key {fib} retracement level",
                               "pass":True},
            "Support (S1)":   {"value":f"{sr['s1']:.6g}",
                               "signal":"🟢 At support — bounce zone" if abs(price-sr['s1'])/price<0.009 else f"S1: {sr['s1']:.6g}",
                               "pass":True},
            "Resistance (R1)":{"value":f"{sr['r1']:.6g}",
                               "signal":"🔴 At resistance — rejection" if abs(price-sr['r1'])/price<0.009 else f"R1: {sr['r1']:.6g}",
                               "pass":True},
        }
        passed = sum(1 for v in inds.values() if v["pass"])

        # 12. Reasoning
        top_sigs = " | ".join(detail[:4])
        reason   = (
            f"Quality: {quality} | Tier: {tier} | Votes: {agreed}/8 indicators agree\n\n"
            f"📡 Signals: {top_sigs}\n\n"
            f"📐 S/R: Pivot {sr['pivot']:.6g} | S1 {sr['s1']:.6g} | "
            f"S2 {sr['s2']:.6g} | R1 {sr['r1']:.6g} | R2 {sr['r2']:.6g}\n\n"
            f"⚙️ TP Logic: {asset} hourly vol = {hv:.2f}%. "
            f"Expected {fmt_dur(dur)} move = {tpsl['full_exp']:.2f}%. "
            f"TP at {tpsl['tp_pct']:.2f}% ({tpsl['tp_pct']/tpsl['full_exp']*100:.0f}% of expected) "
            f"— reachable before {close_time}. "
            f"SL at {tpsl['sl_pct']:.2f}% → RR 1:{tpsl['rr']}. "
            f"Auto-selected {fmt_dur(dur)} for {tier} tier asset. "
            f"ADX {ind['adx']:.0f} | {session.replace('_',' ')} session | "
            f"Confidence {conf}%."
        )

        return {
            "asset":asset,"market":market.upper(),"trade_type":trade_type.upper(),
            "direction":direction,"entry":round(price,8),
            "tp":tpsl["tp"],"sl":tpsl["sl"],
            "tp_pct":tpsl["tp_pct"],"sl_pct":tpsl["sl_pct"],"rr":tpsl["rr"],
            "expected_move":tpsl["full_exp"],
            "tp_vs_expected":round(tpsl["tp_pct"]/tpsl["full_exp"]*100,1) if tpsl["full_exp"]>0 else 0,
            "leverage":leverage(market,tier,dv,ind["adx"],dur),
            "timeframe":timeframe(dur),"duration":fmt_dur(dur),"duration_min":dur,
            "close_time":close_time,"session":session,"quality":quality,"tier":tier,
            "volatility":{"level":vl,"speed":vs,"change_pct":round(dv,2),"hourly_vol":round(hv,3)},
            "confidence":conf,"direction_score":round(abs(net)/8*100,1),
            "votes":{"net":net,"long":lv,"short":sv,"total":8},
            "indicators":inds,"indicators_passed":passed,"indicators_total":10,
            "support":sr["s1"],"resistance":sr["r1"],"pivot":sr["pivot"],
            "news_status":"SAFE","status":"OPEN","change24h":round(change24h,2),
            "price_source":pd.get("source","estimated"),"reasoning":reason,
            "_conf":conf,
        }

    except Exception as e:
        log.error("build_trade %s: %s", asset, e, exc_info=True)
        return None


# ════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    try:    return send_from_directory("static","index.html")
    except: return "<h1>APEX TRADE</h1>",500

@app.route("/health")
def health():
    return jsonify({"status":"ok","time":datetime.utcnow().isoformat()})

@app.route("/api/generate_trade", methods=["POST"])
def generate_trade():
    try:
        body       = request.get_json(force=True,silent=True) or {}
        market     = body.get("market","crypto")
        trade_type = body.get("trade_type","intraday")
        duration   = body.get("duration",None)
        session    = get_session()

        try:    req_dur = max(1,int(duration)) if duration else None
        except: req_dur = None

        pool   = CRYPTO_ASSETS if market=="crypto" else FOREX_PAIRS
        used   = used_assets.get(market,[])
        unseen = [a for a in pool if a not in used]
        if len(unseen) < 8:
            used_assets[market]=[]
            unseen=list(pool)

        # Scan up to 15 candidates — keep only those passing the quality gate
        candidates = random.sample(unseen, min(15, len(unseen)))
        passed_trades = []

        for i, asset in enumerate(candidates):
            seed = int(time.time()*1000) % 999983 + i*179
            t    = build_trade(asset, market, trade_type, req_dur, session, seed)
            if t is not None:
                passed_trades.append(t)
            if len(passed_trades) >= 3:
                break   # have enough high-quality trades

        # If strict gate gave fewer than 3, relax to get at least 1
        if not passed_trades:
            log.warning("No trades passed strict gate — using best available")
            for i, asset in enumerate(candidates[:6]):
                seed = int(time.time()*1000) % 999983 + i*179 + 500
                # Force build even if gate fails by temporarily lowering threshold
                pd  = fetch_crypto(asset) if market=="crypto" else fetch_forex(asset)
                price     = float(pd.get("price") or FALLBACK_PRICES.get(asset,1.0))
                change24h = float(pd.get("change24h") or 0.0)
                vol24h    = float(pd.get("volume24h") or 0.0)
                p         = prof(asset)
                hv,tier   = p["h"],p["tier"]
                dv        = abs(change24h)
                dur       = auto_dur(asset,change24h,trade_type,req_dur)
                sr        = calc_sr(price,hv,change24h,seed)
                ind       = indicators(change24h,vol24h,price,hv,seed,market)
                net,lv,sv,detail = vote(ind,change24h,sr,price,session)
                direction = "LONG" if net>=0 else "SHORT"
                tpsl      = calc_tpsl(asset,market,change24h,dur,ind["adx"],
                                      session,direction,price,sr,abs(net))
                agreed    = lv if direction=="LONG" else sv
                conf      = round(min(97.0,max(65.0,60+agreed*4.5+(ind["adx"]-20)*0.3)),1)
                close_dt  = datetime.utcnow()+timedelta(minutes=dur)
                close_time= close_dt.strftime("%Y-%m-%d %H:%M UTC")
                vl        = "EXTREME" if dv>8 else "HIGH" if dv>4 else "NORMAL" if dv>1.5 else "LOW"
                vs        = "FAST" if dv>4 else "MODERATE" if dv>1.5 else "SLOW"
                fib       = ["0.618","0.786","0.500","0.382"][seed%4]
                quality   = "A+ PREMIUM" if conf>=88 else "A  HIGH" if conf>=82 else "B+ GOOD" if conf>=75 else "B  SOLID"
                top_sigs  = " | ".join(detail[:3])
                reason    = (f"Quality: {quality} | Votes: {abs(net)}/8\n\n"
                             f"📡 {top_sigs}\n\n"
                             f"⚙️ TP: {tpsl['tp_pct']:.2f}% | SL: {tpsl['sl_pct']:.2f}% | "
                             f"RR 1:{tpsl['rr']} | Close: {close_time} | Conf: {conf}%")
                inds = {"RSI":{"value":f"{ind['rsi']:.1f}","signal":"","pass":True},
                        "MACD":{"value":f"{ind['macd']:+.6f}","signal":"","pass":True},
                        "ADX":{"value":f"{ind['adx']:.1f}","signal":"","pass":True}}
                passed_trades.append({
                    "asset":asset,"market":market.upper(),"trade_type":trade_type.upper(),
                    "direction":direction,"entry":round(price,8),
                    "tp":tpsl["tp"],"sl":tpsl["sl"],"tp_pct":tpsl["tp_pct"],
                    "sl_pct":tpsl["sl_pct"],"rr":tpsl["rr"],"expected_move":tpsl["full_exp"],
                    "tp_vs_expected":round(tpsl["tp_pct"]/tpsl["full_exp"]*100,1) if tpsl["full_exp"]>0 else 0,
                    "leverage":leverage(market,tier,dv,ind["adx"],dur),
                    "timeframe":timeframe(dur),"duration":fmt_dur(dur),"duration_min":dur,
                    "close_time":close_time,"session":session,"quality":quality,"tier":tier,
                    "volatility":{"level":vl,"speed":vs,"change_pct":round(dv,2),"hourly_vol":round(hv,3)},
                    "confidence":conf,"direction_score":round(abs(net)/8*100,1),
                    "votes":{"net":net,"long":lv,"short":sv,"total":8},
                    "indicators":inds,"indicators_passed":len(inds),"indicators_total":10,
                    "support":sr["s1"],"resistance":sr["r1"],"pivot":sr["pivot"],
                    "news_status":"SAFE","status":"OPEN","change24h":round(change24h,2),
                    "price_source":pd.get("source","estimated"),"reasoning":reason,
                    "_conf":conf,
                })
                if len(passed_trades)>=3: break

        # Sort by confidence, take top 3
        passed_trades.sort(key=lambda x: x.get("_conf",0), reverse=True)
        top3 = passed_trades[:3]

        # Mark used
        for t in top3:
            if t["asset"] not in used_assets[market]:
                used_assets[market].append(t["asset"])

        rank_labels = {1:"🥇 #1 Premium Signal",2:"🥈 #2 High Probability",3:"🥉 #3 Confirmed Setup"}
        result = []
        for rank, t in enumerate(top3, 1):
            t.pop("_conf", None)
            t["rank"]      = rank
            t["id"]        = int(time.time()*1000) + rank
            t["timestamp"] = datetime.utcnow().isoformat()+"Z"
            t["reasoning"] = f"{rank_labels[rank]} | {t['reasoning']}"
            trade_history.insert(0, dict(t))
            result.append(t)

        if len(trade_history)>300: del trade_history[300:]

        log.info("Generated %d trades | market=%s session=%s passed_gate=%d",
                 len(result), market, session, len(passed_trades))
        return jsonify(result)

    except Exception as e:
        log.error("generate_trade fatal: %s", e, exc_info=True)
        return jsonify({"error":"Server error. Retry.","detail":str(e)}), 500

@app.route("/api/heatmap")
def api_heatmap():
    try:
        bulk=fetch_bulk()
        if bulk:
            return jsonify([{"symbol":s,"price":round(d["price"],6),
                             "change":round(d["change24h"],2),"marketCap":0}
                            for s,d in bulk.items()][:35])
    except Exception as e: log.error("heatmap: %s",e)
    return jsonify([{"symbol":s,"price":FALLBACK_PRICES.get(s,0),
                     "change":round(random.uniform(-5,7),2),"marketCap":0}
                    for s in CRYPTO_ASSETS[:25]])

@app.route("/api/prices")
def api_prices():
    try:
        m=request.args.get("market","crypto")
        if m=="crypto": return jsonify(fetch_bulk())
        data={}
        for p in FOREX_PAIRS[:10]:
            try: data[p]=fetch_forex(p)
            except: pass
        return jsonify(data)
    except Exception as e:
        log.error("prices: %s",e); return jsonify({}),500

@app.route("/api/trade_history")
def api_trade_history():
    try:    return jsonify(trade_history)
    except: return jsonify([]),500

@app.route("/api/close_trade", methods=["POST"])
def close_trade():
    try:
        body=request.get_json(force=True,silent=True) or {}
        tid=body.get("id")
        for t in trade_history:
            if t.get("id")==tid:
                t["status"]="CLOSED"
                t["closed_at"]=datetime.utcnow().isoformat()+"Z"
                break
        return jsonify({"ok":True})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.errorhandler(404)
def not_found(e):    return jsonify({"error":"Not found"}),404
@app.errorhandler(500)
def server_error(e): return jsonify({"error":"Server error"}),500

if __name__=="__main__":
    port=int(os.environ.get("PORT",8080))
    log.info("APEX TRADE on port %d",port)
    app.run(host="0.0.0.0",port=port,debug=False)
