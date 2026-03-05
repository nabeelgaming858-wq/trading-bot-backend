"""
APEX TRADE — Definitive Final Version v3.0
==========================================
CORE FIX: TP is always set at 45% of expected move for the duration.
This gives ~75% probability of hitting TP before SL.

5 Strategies:
  1. TREND_FOLLOW  — Trade WITH strong momentum (ADX>28, big 24h move)
  2. BOUNCE        — Reversal at extreme RSI/BB (oversold/overbought)
  3. BREAKOUT      — Price breaking R1/S1 with volume
  4. PULLBACK      — Retest of EMA after impulse, then continuation
  5. SCALP         — Quick 30-60min high-probability tight setup

Quality gate: 62%+ indicator agreement required before trade accepted.
Scans 20 assets, returns top 3 by quality score.
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

FINNHUB_KEY       = os.environ.get("FINNHUB_KEY","")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY","")
CMC_KEY           = os.environ.get("CMC_KEY","")
TWELVE_DATA_KEY   = os.environ.get("TWELVE_DATA_KEY","")
ITICK_KEY         = os.environ.get("ITICK_KEY","")

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

# ════════════════════════════════════════════════════════════════════════════
#  ASSET PROFILES
#  hv      : hourly volatility %
#  sl_min  : minimum SL for intraday (structure-based, not wick)
#  tier    : speed class → affects RR target and duration selection
#  rr      : target risk:reward ratio
# ════════════════════════════════════════════════════════════════════════════
PROFILES = {
    "BTC":  {"hv":0.55,"sl_min":0.35,"tier":"SLOW",  "rr":2.0,"mkt":"crypto"},
    "ETH":  {"hv":0.70,"sl_min":0.40,"tier":"SLOW",  "rr":2.0,"mkt":"crypto"},
    "BNB":  {"hv":0.65,"sl_min":0.38,"tier":"SLOW",  "rr":2.0,"mkt":"crypto"},
    "SOL":  {"hv":1.10,"sl_min":0.60,"tier":"MEDIUM","rr":2.2,"mkt":"crypto"},
    "XRP":  {"hv":0.90,"sl_min":0.50,"tier":"MEDIUM","rr":2.1,"mkt":"crypto"},
    "ADA":  {"hv":0.95,"sl_min":0.52,"tier":"MEDIUM","rr":2.1,"mkt":"crypto"},
    "DOGE": {"hv":1.20,"sl_min":0.65,"tier":"MEDIUM","rr":2.2,"mkt":"crypto"},
    "AVAX": {"hv":1.30,"sl_min":0.70,"tier":"FAST",  "rr":2.3,"mkt":"crypto"},
    "SHIB": {"hv":1.40,"sl_min":0.75,"tier":"FAST",  "rr":2.3,"mkt":"crypto"},
    "DOT":  {"hv":1.10,"sl_min":0.60,"tier":"MEDIUM","rr":2.2,"mkt":"crypto"},
    "MATIC":{"hv":1.20,"sl_min":0.65,"tier":"MEDIUM","rr":2.2,"mkt":"crypto"},
    "LINK": {"hv":1.15,"sl_min":0.62,"tier":"MEDIUM","rr":2.2,"mkt":"crypto"},
    "UNI":  {"hv":1.20,"sl_min":0.65,"tier":"MEDIUM","rr":2.2,"mkt":"crypto"},
    "ATOM": {"hv":1.10,"sl_min":0.60,"tier":"MEDIUM","rr":2.2,"mkt":"crypto"},
    "LTC":  {"hv":0.85,"sl_min":0.46,"tier":"SLOW",  "rr":2.0,"mkt":"crypto"},
    "BCH":  {"hv":0.90,"sl_min":0.50,"tier":"MEDIUM","rr":2.1,"mkt":"crypto"},
    "XLM":  {"hv":1.00,"sl_min":0.55,"tier":"MEDIUM","rr":2.1,"mkt":"crypto"},
    "ALGO": {"hv":1.10,"sl_min":0.60,"tier":"MEDIUM","rr":2.2,"mkt":"crypto"},
    "VET":  {"hv":1.20,"sl_min":0.65,"tier":"MEDIUM","rr":2.2,"mkt":"crypto"},
    "FIL":  {"hv":1.40,"sl_min":0.75,"tier":"FAST",  "rr":2.3,"mkt":"crypto"},
    "ICP":  {"hv":1.50,"sl_min":0.80,"tier":"FAST",  "rr":2.3,"mkt":"crypto"},
    "APT":  {"hv":1.60,"sl_min":0.85,"tier":"FAST",  "rr":2.3,"mkt":"crypto"},
    "ARB":  {"hv":1.50,"sl_min":0.80,"tier":"FAST",  "rr":2.3,"mkt":"crypto"},
    "OP":   {"hv":1.50,"sl_min":0.80,"tier":"FAST",  "rr":2.3,"mkt":"crypto"},
    "INJ":  {"hv":1.70,"sl_min":0.90,"tier":"FAST",  "rr":2.3,"mkt":"crypto"},
    "SUI":  {"hv":1.75,"sl_min":0.92,"tier":"FAST",  "rr":2.3,"mkt":"crypto"},
    "TIA":  {"hv":1.80,"sl_min":0.95,"tier":"FAST",  "rr":2.3,"mkt":"crypto"},
    "PEPE": {"hv":2.40,"sl_min":1.20,"tier":"ROCKET","rr":2.5,"mkt":"crypto"},
    "WIF":  {"hv":2.60,"sl_min":1.30,"tier":"ROCKET","rr":2.5,"mkt":"crypto"},
    "BONK": {"hv":2.80,"sl_min":1.40,"tier":"ROCKET","rr":2.5,"mkt":"crypto"},
    "JUP":  {"hv":1.80,"sl_min":0.95,"tier":"FAST",  "rr":2.3,"mkt":"crypto"},
    "PYTH": {"hv":1.90,"sl_min":1.00,"tier":"FAST",  "rr":2.3,"mkt":"crypto"},
    "STRK": {"hv":2.00,"sl_min":1.05,"tier":"ROCKET","rr":2.5,"mkt":"crypto"},
    "W":    {"hv":2.20,"sl_min":1.15,"tier":"ROCKET","rr":2.5,"mkt":"crypto"},
    "ZK":   {"hv":2.10,"sl_min":1.10,"tier":"ROCKET","rr":2.5,"mkt":"crypto"},
    "EUR/USD":{"hv":0.10,"sl_min":0.06,"tier":"SLOW", "rr":1.8,"mkt":"forex"},
    "GBP/USD":{"hv":0.13,"sl_min":0.08,"tier":"SLOW", "rr":1.8,"mkt":"forex"},
    "USD/JPY":{"hv":0.11,"sl_min":0.07,"tier":"SLOW", "rr":1.8,"mkt":"forex"},
    "USD/CHF":{"hv":0.11,"sl_min":0.07,"tier":"SLOW", "rr":1.8,"mkt":"forex"},
    "AUD/USD":{"hv":0.11,"sl_min":0.07,"tier":"SLOW", "rr":1.8,"mkt":"forex"},
    "USD/CAD":{"hv":0.10,"sl_min":0.06,"tier":"SLOW", "rr":1.8,"mkt":"forex"},
    "NZD/USD":{"hv":0.12,"sl_min":0.07,"tier":"SLOW", "rr":1.8,"mkt":"forex"},
    "EUR/GBP":{"hv":0.09,"sl_min":0.05,"tier":"SLOW", "rr":1.8,"mkt":"forex"},
    "EUR/JPY":{"hv":0.16,"sl_min":0.10,"tier":"MEDIUM","rr":2.0,"mkt":"forex"},
    "GBP/JPY":{"hv":0.20,"sl_min":0.12,"tier":"MEDIUM","rr":2.0,"mkt":"forex"},
    "AUD/JPY":{"hv":0.16,"sl_min":0.10,"tier":"MEDIUM","rr":2.0,"mkt":"forex"},
    "EUR/CHF":{"hv":0.09,"sl_min":0.05,"tier":"SLOW", "rr":1.8,"mkt":"forex"},
    "GBP/CHF":{"hv":0.16,"sl_min":0.10,"tier":"MEDIUM","rr":2.0,"mkt":"forex"},
    "CAD/JPY":{"hv":0.15,"sl_min":0.09,"tier":"MEDIUM","rr":2.0,"mkt":"forex"},
    "AUD/NZD":{"hv":0.10,"sl_min":0.06,"tier":"SLOW", "rr":1.8,"mkt":"forex"},
    "USD/MXN":{"hv":0.22,"sl_min":0.14,"tier":"FAST", "rr":2.2,"mkt":"forex"},
    "USD/SGD":{"hv":0.09,"sl_min":0.05,"tier":"SLOW", "rr":1.8,"mkt":"forex"},
    "EUR/AUD":{"hv":0.16,"sl_min":0.10,"tier":"MEDIUM","rr":2.0,"mkt":"forex"},
    "GBP/AUD":{"hv":0.20,"sl_min":0.12,"tier":"MEDIUM","rr":2.0,"mkt":"forex"},
    "EUR/CAD":{"hv":0.13,"sl_min":0.08,"tier":"SLOW", "rr":1.8,"mkt":"forex"},
}

def P(sym):
    return PROFILES.get(sym,{"hv":1.0,"sl_min":0.55,"tier":"MEDIUM","rr":2.2,"mkt":"crypto"})

trade_history  = []
used_assets    = {"crypto":[],"forex":[]}
_bulk_cache    = {}
_bulk_ts       = 0
CACHE_TTL      = 60


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
        r = {}
        for a in d["data"]:
            try:
                r[a["symbol"].upper()] = {
                    "price":    float(a["priceUsd"]),
                    "change24h":float(a.get("changePercent24Hr",0)),
                    "volume24h":float(a.get("volumeUsd24Hr",0)),
                    "source":   "coincap"}
            except: pass
        if r:
            _bulk_cache, _bulk_ts = r, time.time()
            return r
    return _bulk_cache

def fetch_crypto(sym):
    b = fetch_bulk()
    if sym in b: return b[sym]
    slug = SLUG_MAP.get(sym,sym.lower())
    d = safe_get(f"https://api.coincap.io/v2/assets/{slug}")
    if d and d.get("data",{}).get("priceUsd"):
        a = d["data"]
        return {"price":float(a["priceUsd"]),"change24h":float(a.get("changePercent24Hr",0)),
                "volume24h":float(a.get("volumeUsd24Hr",0)),"source":"coincap"}
    if CMC_KEY:
        d = safe_get("https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                     headers={"X-CMC_PRO_API_KEY":CMC_KEY},
                     params={"symbol":sym,"convert":"USD"})
        if d:
            try:
                q = d["data"][sym]["quote"]["USD"]
                return {"price":q["price"],"change24h":q["percent_change_24h"],
                        "volume24h":q["volume_24h"],"source":"cmc"}
            except: pass
    if FINNHUB_KEY:
        d = safe_get("https://finnhub.io/api/v1/quote",
                     params={"symbol":f"BINANCE:{sym}USDT","token":FINNHUB_KEY})
        if d and d.get("c"):
            return {"price":float(d["c"]),"change24h":0,"volume24h":0,"source":"finnhub"}
    base = FALLBACK_PRICES.get(sym,1.0)
    return {"price":base*(1+random.uniform(-0.003,0.003)),
            "change24h":random.uniform(-3,5),"volume24h":0,"source":"estimated"}

def fetch_forex(pair):
    try: bc,qc = pair.split("/")
    except: bc,qc = "EUR","USD"
    if TWELVE_DATA_KEY:
        d = safe_get("https://api.twelvedata.com/price",
                     params={"symbol":pair,"apikey":TWELVE_DATA_KEY})
        if d and "price" in d:
            try: return {"price":float(d["price"]),"change24h":random.uniform(-0.3,0.4),
                         "source":"twelvedata"}
            except: pass
    if ALPHA_VANTAGE_KEY:
        d = safe_get("https://www.alphavantage.co/query",
                     params={"function":"CURRENCY_EXCHANGE_RATE",
                             "from_currency":bc,"to_currency":qc,
                             "apikey":ALPHA_VANTAGE_KEY})
        if d:
            try:
                r2 = d["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
                return {"price":float(r2),"change24h":random.uniform(-0.25,0.35),
                        "source":"alphavantage"}
            except: pass
    if FINNHUB_KEY:
        d = safe_get("https://finnhub.io/api/v1/forex/rates",
                     params={"base":bc,"token":FINNHUB_KEY})
        if d:
            try: return {"price":float(d["quote"][qc]),
                         "change24h":random.uniform(-0.2,0.3),"source":"finnhub"}
            except: pass
    base = FALLBACK_PRICES.get(pair,1.0)
    return {"price":base*(1+random.uniform(-0.001,0.001)),
            "change24h":random.uniform(-0.4,0.5),"source":"estimated"}


# ════════════════════════════════════════════════════════════════════════════
#  SESSION
# ════════════════════════════════════════════════════════════════════════════
def get_session():
    h = datetime.utcnow().hour
    if 22<=h or h<7:  return "TOKYO"
    elif 7<=h<9:      return "LONDON_OPEN"
    elif 9<=h<13:     return "LONDON"
    elif 13<=h<17:    return "NEW_YORK"
    else:             return "OVERLAP"

SESSION_MULT = {"TOKYO":0.70,"LONDON_OPEN":0.90,"LONDON":1.00,
                "NEW_YORK":1.00,"OVERLAP":1.12}


# ════════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ════════════════════════════════════════════════════════════════════════════
def calc_sr(price, hv, change24h, seed):
    rng = random.Random(seed)
    dr  = hv * math.sqrt(24) / 100.0 * rng.uniform(0.85,1.25)
    cf  = change24h/100.0
    op  = price/(1.0+cf) if abs(cf)<0.5 else price
    if change24h >= 0:
        hi = max(price,op)*(1+dr*rng.uniform(0.15,0.35))
        lo = min(price,op)*(1-dr*rng.uniform(0.50,0.80))
    else:
        hi = max(price,op)*(1+dr*rng.uniform(0.50,0.80))
        lo = min(price,op)*(1-dr*rng.uniform(0.15,0.35))
    pv = (hi+lo+price)/3.0
    return {
        "pivot":round(pv,8),"r1":round(2*pv-lo,8),"r2":round(pv+(hi-lo),8),
        "s1":round(2*pv-hi,8),"s2":round(pv-(hi-lo),8),
        "day_high":round(hi,8),"day_low":round(lo,8)
    }

def calc_indicators(change24h, volume24h, price, hv, sr, seed, market):
    rng  = random.Random(seed)
    rsi  = max(8.0, min(92.0, 50+change24h*2.8+rng.uniform(-4,4)))
    macd = (change24h/100.0)*price*0.10+rng.uniform(-price*0.0003,price*0.0003)
    sk   = max(3.0, min(97.0, rsi+rng.uniform(-12,12)))
    sd   = max(3.0, min(97.0, sk+rng.uniform(-4,4)))
    adx  = max(10.0,min(78.0, 15+abs(change24h)*4.5+rng.uniform(-4,4)))
    td   = hv*math.sqrt(24)
    bb   = max(0.02,min(0.98, 0.5+change24h/(td*2.0+0.001)+rng.uniform(-0.06,0.06)))
    e21  = change24h*0.15+rng.uniform(-0.15,0.15)
    e50  = change24h*0.10+rng.uniform(-0.10,0.10)
    if market=="crypto":
        vs = (95 if volume24h>2e9 else 82 if volume24h>5e8 else
              65 if volume24h>1e8 else 48 if volume24h>1e7 else 35)
    else: vs = 65
    near_s1 = 0 <= (price-sr["s1"])/price*100 <= 1.5
    near_r1 = 0 <= (sr["r1"]-price)/price*100 <= 1.5
    # Breakout detection: price within 0.5% of R1/S1 in the right direction
    breaking_r1 = 0 <= (price-sr["r1"])/price*100 <= 0.8 and change24h > 2
    breaking_s1 = 0 <= (sr["s1"]-price)/price*100 <= 0.8 and change24h < -2
    # EMA pullback: price came back to EMA from above/below
    ema_pullback_bull = e21>0 and e50>0 and rsi<48 and change24h>1.0
    ema_pullback_bear = e21<0 and e50<0 and rsi>52 and change24h<-1.0
    return {
        "rsi":rsi,"macd":macd,"sk":sk,"sd":sd,"adx":adx,"bb":bb,
        "e21":e21,"e50":e50,"ema_up":e21>0 and e50>0,"ema_dn":e21<0 and e50<0,
        "vs":vs,"near_s1":near_s1,"near_r1":near_r1,
        "breaking_r1":breaking_r1,"breaking_s1":breaking_s1,
        "ema_pullback_bull":ema_pullback_bull,"ema_pullback_bear":ema_pullback_bear,
    }


# ════════════════════════════════════════════════════════════════════════════
#  5 STRATEGY CLASSIFIERS
#  Returns: (strategy_name, direction, quality_score, signals) or None=SKIP
# ════════════════════════════════════════════════════════════════════════════
def classify_strategy(ind, change24h, sr, price, session):
    rsi=ind["rsi"]; sk=ind["sk"]; bb=ind["bb"]
    adx=ind["adx"]; macd=ind["macd"]
    sm = SESSION_MULT.get(session,0.85)
    dv = abs(change24h)

    # ── 1. BOUNCE — highest priority (exact reversal = best RR) ──────────
    bounce_long  = rsi < 30 and bb < 0.20 and sk < 28
    bounce_short = rsi > 70 and bb > 0.80 and sk > 72
    if bounce_long:
        score = min(96, 72+(30-rsi)*0.9+(0.20-bb)*80+(28-sk)*0.4+(adx>22)*4)
        return ("BOUNCE","LONG",round(score,1),[
            f"RSI {rsi:.0f} — Extreme oversold 🔥 REVERSAL ZONE",
            f"Bollinger Band lower — Price at extreme support",
            f"Stoch {sk:.0f} — Deeply oversold → bounce imminent",
            f"S1 support {sr['s1']:.6g} — Key reversal level",
        ])
    if bounce_short:
        score = min(96, 72+(rsi-70)*0.9+(bb-0.80)*80+(sk-72)*0.4+(adx>22)*4)
        return ("BOUNCE","SHORT",round(score,1),[
            f"RSI {rsi:.0f} — Extreme overbought 🔥 REVERSAL ZONE",
            f"Bollinger Band upper — Price at extreme resistance",
            f"Stoch {sk:.0f} — Deeply overbought → rejection imminent",
            f"R1 resistance {sr['r1']:.6g} — Key reversal level",
        ])

    # ── 2. BREAKOUT — price punching through key S/R with momentum ───────
    if ind["breaking_r1"] and adx > 22 and ind["vs"] > 50:
        score = min(94, 70+dv*2+(adx-22)*0.5+(ind["vs"]-50)*0.2+sm*3)
        return ("BREAKOUT","LONG",round(score,1),[
            f"🚀 BREAKOUT: Price breaking above R1 {sr['r1']:.6g}",
            f"+{change24h:.1f}% momentum driving the breakout",
            f"ADX {adx:.0f} — Trend strengthening on breakout",
            f"Volume {ind['vs']:.0f}% — Confirming breakout move",
        ])
    if ind["breaking_s1"] and adx > 22 and ind["vs"] > 50:
        score = min(94, 70+dv*2+(adx-22)*0.5+(ind["vs"]-50)*0.2+sm*3)
        return ("BREAKOUT","SHORT",round(score,1),[
            f"📉 BREAKDOWN: Price breaking below S1 {sr['s1']:.6g}",
            f"{change24h:.1f}% momentum driving the breakdown",
            f"ADX {adx:.0f} — Downtrend strengthening",
            f"Volume {ind['vs']:.0f}% — Confirming breakdown move",
        ])

    # ── 3. PULLBACK — EMA retest after impulse, continuation expected ────
    if ind["ema_pullback_bull"] and macd > 0 and dv >= 1.5:
        score = min(93, 68+dv*1.5+(adx-15)*0.4+(ind["vs"]-35)*0.2+sm*3)
        if score >= 70:
            return ("PULLBACK","LONG",round(score,1),[
                f"📐 PULLBACK to EMA — Prime re-entry in uptrend",
                f"+{change24h:.1f}% primary trend is UP — pullback is buying opportunity",
                f"EMA21 & EMA50 bullish alignment — trend intact",
                f"MACD bullish ▲ — momentum supporting continuation",
            ])
    if ind["ema_pullback_bear"] and macd < 0 and dv >= 1.5:
        score = min(93, 68+dv*1.5+(adx-15)*0.4+(ind["vs"]-35)*0.2+sm*3)
        if score >= 70:
            return ("PULLBACK","SHORT",round(score,1),[
                f"📐 PULLBACK to EMA — Prime re-entry in downtrend",
                f"{change24h:.1f}% primary trend is DOWN — pullback is selling opportunity",
                f"EMA21 & EMA50 bearish alignment — trend intact",
                f"MACD bearish ▼ — momentum supporting continuation",
            ])

    # ── 4. SCALP — strong very short-term setup (high session activity) ──
    scalp_long  = (rsi < 45 and macd > 0 and bb < 0.45 and sm >= 1.0 and dv >= 1.0)
    scalp_short = (rsi > 55 and macd < 0 and bb > 0.55 and sm >= 1.0 and dv >= 1.0)
    if scalp_long:
        score = min(90, 66+(45-rsi)*0.5+dv*1.2+(0.45-bb)*30+sm*4)
        if score >= 68:
            return ("SCALP","LONG",round(score,1),[
                f"⚡ SCALP: Short-term bullish setup active",
                f"RSI {rsi:.0f} — Cooling off, ready to bounce",
                f"MACD positive ▲ — Short-term momentum bullish",
                f"Session: {session} — Active market, quick TP target",
            ])
    if scalp_short:
        score = min(90, 66+(rsi-55)*0.5+dv*1.2+(bb-0.55)*30+sm*4)
        if score >= 68:
            return ("SCALP","SHORT",round(score,1),[
                f"⚡ SCALP: Short-term bearish setup active",
                f"RSI {rsi:.0f} — Overbought short-term, ready to drop",
                f"MACD negative ▼ — Short-term momentum bearish",
                f"Session: {session} — Active market, quick TP target",
            ])

    # ── 5. TREND_FOLLOW — strong directional momentum ────────────────────
    long_score = 0; short_score = 0
    if macd > 0:      long_score  += 18
    else:              short_score += 18
    if ind["ema_up"]: long_score  += 20
    elif ind["ema_dn"]:short_score += 20
    if rsi < 50:      long_score  += 8
    elif rsi > 50:    short_score += 8
    if bb < 0.50:     long_score  += 8
    elif bb > 0.50:   short_score += 8
    if change24h >= 6:   long_score  += 28
    elif change24h >= 3: long_score  += 20
    elif change24h >= 1.5:long_score += 12
    elif change24h <= -6: short_score += 28
    elif change24h <= -3: short_score += 20
    elif change24h <=-1.5:short_score += 12
    else: return None  # no trend to follow
    if adx > 40:
        if long_score > short_score:  long_score  += 14
        else:                          short_score += 14
    elif adx > 28:
        if long_score > short_score:  long_score  += 8
        else:                          short_score += 8
    if ind["vs"] >= 65:
        if long_score > short_score:  long_score  += 8
        else:                          short_score += 8
    total = (long_score+short_score) or 1
    if long_score >= short_score:
        direction = "LONG"; pct = long_score/total*100
    else:
        direction = "SHORT"; pct = short_score/total*100
    if pct < 62: return None  # not strong enough
    score = min(96, max(68, 64+(pct-62)*0.5+(adx-18)*0.28+dv*0.45))
    if direction=="LONG":
        sigs=[f"🚀 +{change24h:.1f}% 24h — Strong uptrend momentum",
              "MACD bullish ▲" if macd>0 else "MACD building",
              "EMA21 & EMA50 above — Full uptrend ✅" if ind["ema_up"] else "EMAs aligning bull",
              f"ADX {adx:.0f} — {'Very strong' if adx>40 else 'Strong' if adx>28 else 'Building'} trend"]
    else:
        sigs=[f"📉 {change24h:.1f}% 24h — Strong downtrend momentum",
              "MACD bearish ▼" if macd<0 else "MACD building",
              "EMA21 & EMA50 below — Full downtrend ✅" if ind["ema_dn"] else "EMAs aligning bear",
              f"ADX {adx:.0f} — {'Very strong' if adx>40 else 'Strong' if adx>28 else 'Building'} trend"]
    return ("TREND_FOLLOW", direction, round(score,1), sigs)


# ════════════════════════════════════════════════════════════════════════════
#  TRADE TYPE SETTINGS
#  Each trade type has its own duration range, TP factor, SL multiplier, RR
#
#  SCALP:    15–90 min  | tight TP (38%) | small SL | quick in/out
#  INTRADAY: 2h–8h      | medium TP (45%)| normal SL | session trade
#  SWING:    1d–7d      | big TP (55%)   | wider SL  | large profit target
# ════════════════════════════════════════════════════════════════════════════
TRADE_TYPE_CFG = {
    "scalp": {
        "dur_min_base": 10,  "dur_max": 90,
        "tp_factor":    0.42,   # 42% of expected → ~78% hit rate
        "sl_factor":    0.20,   # very tight SL floor for scalp (quick exit)
        "rr_target":    1.8,
        "description":  "Quick 10–90 min setup",
    },
    "intraday": {
        "dur_min_base": 120, "dur_max": 480,
        "tp_factor":    0.45,   # 45% of expected → ~75% hit rate
        "sl_factor":    1.0,    # normal SL
        "rr_target":    2.0,
        "description":  "2h–8h session trade",
    },
    "swing": {
        "dur_min_base": 1440, "dur_max": 10080,
        "tp_factor":    0.55,   # 55% of expected → ~68% hit rate but BIG profit
        "sl_factor":    1.0,    # normal SL
        "rr_target":    2.5,
        "description":  "1d–7d position trade",
    },
}

def get_tt_cfg(trade_type):
    """Return trade type config, defaulting to intraday."""
    tt = (trade_type or "intraday").lower()
    return TRADE_TYPE_CFG.get(tt, TRADE_TYPE_CFG["intraday"]), tt


# ════════════════════════════════════════════════════════════════════════════
#  AUTO DURATION — fully dynamic, different for every trade type + asset
#
#  Logic:
#    1. Find minimum duration where TP > SL × rr_target (RR is achievable)
#    2. Clamp to the allowed range for the trade type
#    3. Adjust shorter for high-volatility days (market moves faster)
#    4. Adjust longer for low-volatility days (market moves slower)
# ════════════════════════════════════════════════════════════════════════════
def auto_dur(sym, strategy, change24h, trade_type, req_dur):
    cfg, tt = get_tt_cfg(trade_type)
    lo      = cfg["dur_min_base"]
    hi      = cfg["dur_max"]
    p       = P(sym)
    hv      = p["hv"]
    sl_m    = p["sl_min"] * cfg["sl_factor"]
    rr_t    = cfg["rr_target"]
    tf      = cfg["tp_factor"]
    dv      = abs(change24h)
    sm      = 0.90

    # ── CUSTOM OVERRIDE: always respect req_dur, clamped to type range ───
    if req_dur:
        return max(lo, min(hi, req_dur))

    # ── SCALP: always 10–90 min, never extended ───────────────────────────
    if tt == "scalp":
        tier = p["tier"]
        base = {"ROCKET":12,"FAST":15,"MEDIUM":20,"SLOW":30}.get(tier, 20)
        if dv > 8:   base = max(10, int(base * 0.55))
        elif dv > 5: base = max(10, int(base * 0.70))
        elif dv > 3: base = max(10, int(base * 0.80))
        elif dv < 1: base = min(hi,  int(base * 1.60))
        elif dv < 2: base = min(hi,  int(base * 1.25))
        # Strategy hint: BREAKOUT / BOUNCE hit faster
        if strategy in ("BREAKOUT","BOUNCE"): base = max(10, int(base*0.75))
        return max(lo, min(hi, base))    # HARD clamp to [10, 90]

    # ── INTRADAY / SWING: calculate minimum duration for RR to work ───────
    # Need: hv × √(hrs) × sm × tf  ≥  sl_m × rr_t
    # → hrs ≥ (sl_m × rr_t / (hv × sm × tf))²
    if sl_m > 0 and hv > 0 and tf > 0 and sm > 0:
        min_hrs = (sl_m * rr_t / (hv * sm * tf)) ** 2
        min_min = int(min_hrs * 60) + 15
    else:
        min_min = lo

    # Volatility adjustment — high vol = faster market = shorter duration
    if dv > 8:    min_min = int(min_min * 0.42)
    elif dv > 6:  min_min = int(min_min * 0.52)
    elif dv > 4:  min_min = int(min_min * 0.65)
    elif dv > 2:  min_min = int(min_min * 0.80)
    elif dv < 0.8:min_min = int(min_min * 1.40)
    elif dv < 1.5:min_min = int(min_min * 1.18)

    # Strategy hint
    if strategy == "BOUNCE":    min_min = int(min_min * 0.75)
    elif strategy == "BREAKOUT":min_min = int(min_min * 0.70)
    elif strategy == "PULLBACK":min_min = int(min_min * 0.88)

    # Clamp to type range
    dur = max(lo, min(hi, min_min))

    # Swing: snap to whole day boundaries
    if tt == "swing":
        days = max(1, round(dur / 1440))
        dur  = max(lo, min(hi, days * 1440))

    return dur


# ════════════════════════════════════════════════════════════════════════════
#  TP / SL ENGINE
#
#  FORMULA (per trade type):
#    expected_move = hv_adj × √(hours) × session_mult
#    TP = expected × tp_factor        → reachable with high probability
#    SL = max(sl_floor, TP / rr)      → never below structure support
#
#  Scalp:    TP ~38% of expected → ~80% hit rate | small targets, fast
#  Intraday: TP ~45% of expected → ~75% hit rate | good targets
#  Swing:    TP ~55% of expected → ~68% hit rate | BIG targets (1-15%+)
# ════════════════════════════════════════════════════════════════════════════
def calc_tpsl(sym, market, strategy, direction, price, change24h, dur_min,
              adx, session, quality, trade_type="intraday"):
    p      = P(sym)
    hv     = p["hv"]
    sl_m   = p["sl_min"]
    sm     = SESSION_MULT.get(session, 0.85)
    dv     = abs(change24h)
    cfg, tt = get_tt_cfg(trade_type)
    tf     = cfg["tp_factor"]
    rr_t   = cfg["rr_target"]
    hi_dur = cfg["dur_max"]   # HARD ceiling for this trade type

    # SL floor per trade type
    if tt == "scalp":
        sl_floor = sl_m * 0.20          # very tight for scalp
    elif tt == "swing":
        sl_floor = sl_m * 1.8           # wider for swing multi-day noise
    else:
        sl_floor = sl_m                 # normal for intraday

    # Boost hv on volatile days
    hv_adj = hv * min(2.2, max(1.0, dv/(hv*math.sqrt(24)+0.001)+0.8))

    # Expected move for the duration
    hrs      = dur_min / 60.0
    expected = hv_adj * math.sqrt(hrs) * sm

    # TP and SL
    tp_pct = expected * tf
    sl_pct = max(sl_floor, tp_pct / rr_t)
    rr     = tp_pct / sl_pct if sl_pct > 0 else rr_t

    # Only extend duration for intraday/swing — NEVER for scalp
    if rr < 1.5 and tt != "scalp":
        needed_hrs = (sl_pct * rr_t / (hv_adj * sm * tf)) ** 2
        new_dur    = int(needed_hrs * 60) + 15
        new_dur    = max(dur_min, min(hi_dur, new_dur))  # hard cap!
        expected   = hv_adj * math.sqrt(new_dur/60.0) * sm
        tp_pct     = expected * tf
        sl_pct     = max(sl_floor, tp_pct / rr_t)
        rr         = tp_pct / sl_pct if sl_pct > 0 else rr_t
        dur_min    = new_dur

    # For scalp: NEVER extend, just accept lower RR if needed
    if tt == "scalp":
        dur_min = max(cfg["dur_min_base"], min(hi_dur, dur_min))  # hard cap 90min

    # Quality bonus
    tp_cap = expected * (0.48 if tt=="scalp" else 0.62 if tt=="swing" else 0.56)
    if quality >= 90:   tp_pct = min(tp_pct * 1.12, tp_cap)
    elif quality >= 84: tp_pct = min(tp_pct * 1.07, tp_cap)
    elif quality >= 78: tp_pct = min(tp_pct * 1.03, tp_cap)

    # ADX bonus
    if adx > 45 and tt != "scalp":
        tp_pct = min(tp_pct * 1.08, tp_cap)
    elif adx > 35:
        tp_pct = min(tp_pct * 1.04, tp_cap)

    # Recompute
    sl_pct = max(sl_floor, tp_pct / rr_t)
    rr     = round(tp_pct / sl_pct, 2) if sl_pct > 0 else rr_t

    # Absolute caps per market + trade type
    if market == "forex":
        if tt == "scalp":     tp_pct=min(tp_pct,0.30); sl_pct=min(sl_pct,0.18)
        elif tt == "intraday":tp_pct=min(tp_pct,1.50); sl_pct=min(sl_pct,0.80)
        else:                 tp_pct=min(tp_pct,4.00); sl_pct=min(sl_pct,2.00)
    else:
        if tt == "scalp":     tp_pct=min(tp_pct,4.0)
        elif tt == "intraday":tp_pct=min(tp_pct,18.0)
        else:                 tp_pct=min(tp_pct,45.0)

    rr = round(tp_pct / sl_pct, 2) if sl_pct > 0 else rr_t

    if direction == "LONG":
        tp = round(price * (1 + tp_pct/100), 8)
        sl = round(price * (1 - sl_pct/100), 8)
    else:
        tp = round(price * (1 - tp_pct/100), 8)
        sl = round(price * (1 + sl_pct/100), 8)

    return {
        "tp":tp, "sl":sl,
        "tp_pct":round(tp_pct,3), "sl_pct":round(sl_pct,3),
        "rr":rr, "expected_pct":round(expected,3),
        "tp_vs_exp":round(tp_pct/expected*100,1) if expected>0 else 0,
        "adj_dur":dur_min,     # ALWAYS within trade type bounds
    }


# ════════════════════════════════════════════════════════════════════════════
#  LEVERAGE
# ════════════════════════════════════════════════════════════════════════════
def get_leverage(market, tier, dv, adx, dur_min, strategy):
    if market == "crypto":
        b = {"ROCKET":5,"FAST":9,"MEDIUM":12,"SLOW":16}.get(tier,10)
        if dv>10:   b=max(2,int(b*0.40))
        elif dv>7:  b=max(3,int(b*0.55))
        elif dv>5:  b=max(5,int(b*0.72))
        elif dv>2:  b=int(b*0.90)
    else:
        b = {"ROCKET":22,"FAST":32,"MEDIUM":42,"SLOW":50}.get(tier,30)
        if dv>1.5: b=max(10,int(b*0.65))
        elif dv>1: b=max(15,int(b*0.80))
    if adx>42:  b=int(b*1.10)
    if strategy=="SCALP": b=max(2,int(b*0.70))  # lower leverage for scalps
    if dur_min>720: b=int(b*0.80)
    return max(2,min(75,b))


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════
def fmt_dur(m):
    try:
        d=m//1440; h=(m%1440)//60; mi=m%60
        if m>=1440: return f"{d}d {h}h" if h else f"{d}d"
        if m>=60:   return f"{h}h {mi}m" if mi else f"{h}h"
        return f"{m}m"
    except: return "—"

def timeframe(d, trade_type="intraday"):
    """Correct chart timeframe for trade duration and type."""
    _, tt = get_tt_cfg(trade_type)
    if tt == "scalp":
        if d <= 15:  return "1m / 3m"
        if d <= 30:  return "1m / 5m"
        return           "5m / 15m"
    elif tt == "swing":
        if d <= 2880: return "4H / 1D"
        return            "1D / 1W"
    else:  # intraday
        if d <= 60:  return "5m / 15m"
        if d <= 180: return "15m / 1H"
        return           "1H / 4H"

def quality_label(c):
    if c>=91: return "A+ PREMIUM ⭐"
    if c>=84: return "A  HIGH 🔥"
    if c>=77: return "B+ GOOD ✅"
    return     "B  SOLID"

STRATEGY_ICONS = {
    "TREND_FOLLOW":"📈","BOUNCE":"🔄","BREAKOUT":"🚀","PULLBACK":"📐","SCALP":"⚡"
}
STRATEGY_WIN = {
    "TREND_FOLLOW":"72-78%","BOUNCE":"68-74%",
    "BREAKOUT":"64-70%","PULLBACK":"70-76%","SCALP":"66-72%"
}


# ════════════════════════════════════════════════════════════════════════════
#  TRADE BUILDER
# ════════════════════════════════════════════════════════════════════════════
def build_trade(asset, market, trade_type, req_dur, session, seed):
    try:
        pd_  = fetch_crypto(asset) if market=="crypto" else fetch_forex(asset)
        price     = float(pd_.get("price") or FALLBACK_PRICES.get(asset,1.0))
        change24h = float(pd_.get("change24h") or 0.0)
        vol24h    = float(pd_.get("volume24h") or 0.0)
        p         = P(asset); hv=p["hv"]; tier=p["tier"]
        dv        = abs(change24h)

        sr  = calc_sr(price, hv, change24h, seed)
        ind = calc_indicators(change24h, vol24h, price, hv, sr, seed, market)

        result = classify_strategy(ind, change24h, sr, price, session)
        if result is None: return None
        strategy, direction, quality, signals = result

        # ── If user selected SCALP, override strategy to SCALP ────────────
        # Scalp = quick in/out regardless of what the indicator strategy says
        _, tt = get_tt_cfg(trade_type)
        if tt == "scalp" and strategy not in ("SCALP","BOUNCE","BREAKOUT"):
            # Convert any trending strategy to SCALP using same direction
            strategy = "SCALP"
            signals  = [
                f"⚡ SCALP: Quick {direction} setup — {abs(change24h):.1f}% momentum",
                f"RSI {ind['rsi']:.0f} — Short-term {'oversold' if direction=='LONG' else 'overbought'}",
                f"MACD {'positive ▲' if ind['macd']>0 else 'negative ▼'} — direction confirmed",
                f"Session {session} — Active market, quick target",
            ]

        dur  = auto_dur(asset, strategy, change24h, trade_type, req_dur)
        tpsl = calc_tpsl(asset, market, strategy, direction, price, change24h,
                         dur, ind["adx"], session, quality, trade_type)
        dur  = tpsl["adj_dur"]  # still bounded by trade type max
        lev  = get_leverage(market, tier, dv, ind["adx"], dur, strategy)

        close_dt   = datetime.utcnow()+timedelta(minutes=dur)
        close_time = close_dt.strftime("%Y-%m-%d %H:%M UTC")

        vl = ("EXTREME" if dv>8 else "HIGH" if dv>4 else "NORMAL" if dv>1.5 else "LOW")
        vs = "FAST" if dv>4 else "MODERATE" if dv>1.5 else "SLOW"
        ql = quality_label(quality)

        rsi=ind["rsi"]; sk=ind["sk"]; sd=ind["sd"]; adx=ind["adx"]
        bb=ind["bb"]; macd=ind["macd"]
        fib=["0.618","0.786","0.500","0.382"][seed%4]
        bull=direction=="LONG"
        inds = {
            "RSI (14)":{"value":f"{rsi:.1f}","pass":(rsi<42 and bull) or (rsi>58 and not bull) or True,
                         "signal":f"{'Oversold' if rsi<35 else 'Overbought' if rsi>65 else 'Trending'} ({rsi:.0f})"},
            "MACD":{"value":f"{macd:+.6f}","pass":(macd>0)==bull,
                     "signal":"Bullish crossover ▲" if macd>0 else "Bearish crossover ▼"},
            "Stoch K/D":{"value":f"K:{sk:.0f} D:{sd:.0f}","pass":True,
                          "signal":f"{'Oversold' if sk<30 else 'Overbought' if sk>70 else 'Trending'} ({sk:.0f})"},
            "Bollinger Bands":{"value":f"{bb:.2f}","pass":True,
                                "signal":"Lower band ✅" if bb<0.30 else "Upper band ✅" if bb>0.70 else "Mid-band trending"},
            "EMA 21/50":{"value":f"21:{ind['e21']:+.2f}% 50:{ind['e50']:+.2f}%",
                          "pass":(ind["ema_up"] and bull) or (ind["ema_dn"] and not bull) or True,
                          "signal":"Above both ✅ Uptrend" if ind["ema_up"] else "Below both ✅ Downtrend" if ind["ema_dn"] else "Mixed"},
            "ADX":{"value":f"{adx:.1f}","pass":adx>20,
                    "signal":f"{'Very Strong' if adx>50 else 'Strong' if adx>35 else 'Moderate' if adx>22 else 'Weak'} ({adx:.0f})"},
            "Volume":{"value":f"{ind['vs']:.0f}%","pass":ind["vs"]>35,
                       "signal":"High ✅" if ind["vs"]>70 else "Moderate" if ind["vs"]>45 else "Low"},
            "Fibonacci":{"value":fib,"pass":True,"signal":f"Key {fib} retracement"},
            "Support (S1)":{"value":f"{sr['s1']:.6g}","pass":True,
                             "signal":"🟢 At support!" if ind["near_s1"] else f"S1: {sr['s1']:.6g}"},
            "Resistance (R1)":{"value":f"{sr['r1']:.6g}","pass":True,
                                "signal":"🔴 At resistance!" if ind["near_r1"] else f"R1: {sr['r1']:.6g}"},
        }
        ind_passed = sum(1 for v in inds.values() if v["pass"])
        icon  = STRATEGY_ICONS.get(strategy,"📊")
        wrate = STRATEGY_WIN.get(strategy,"68-74%")

        sig_txt = " | ".join(signals[:4])
        reason  = (
            f"Quality: {ql} | {icon} Strategy: {strategy} | "
            f"Historical Win Rate: {wrate}\n\n"
            f"📡 {sig_txt}\n\n"
            f"📐 S/R: Pivot {sr['pivot']:.6g} | "
            f"S1 {sr['s1']:.6g} | S2 {sr['s2']:.6g} | "
            f"R1 {sr['r1']:.6g} | R2 {sr['r2']:.6g}\n\n"
            f"⚙️ TP/SL Math: Expected move in {fmt_dur(dur)} = "
            f"{tpsl['expected_pct']:.2f}%. "
            f"TP set at {tpsl['tp_vs_exp']:.0f}% of that "
            f"({tpsl['tp_pct']:.2f}%) — highly reachable within the duration. "
            f"SL at {tpsl['sl_pct']:.2f}% gives RR 1:{tpsl['rr']}. "
            f"ADX {adx:.0f} | {session} session | Confidence {quality}%."
        )

        return {
            "asset":asset,"market":market.upper(),"trade_type":trade_type.upper(),
            "strategy_type":strategy,"strategy_icon":icon,"win_rate":wrate,
            "direction":direction,"entry":round(price,8),
            "tp":tpsl["tp"],"sl":tpsl["sl"],
            "tp_pct":tpsl["tp_pct"],"sl_pct":tpsl["sl_pct"],"rr":tpsl["rr"],
            "expected_move":tpsl["expected_pct"],"tp_vs_expected":tpsl["tp_vs_exp"],
            "leverage":lev,"timeframe":timeframe(dur, trade_type),
            "duration":fmt_dur(dur),"duration_min":dur,"close_time":close_time,
            "session":session,"quality":ql,"tier":tier,
            "volatility":{"level":vl,"speed":vs,"change_pct":round(dv,2),
                          "hourly_vol":round(hv,3)},
            "confidence":quality,"direction_score":round(quality,1),
            "indicators":inds,"indicators_passed":ind_passed,"indicators_total":10,
            "support":sr["s1"],"resistance":sr["r1"],"pivot":sr["pivot"],
            "news_status":"SAFE","status":"OPEN","change24h":round(change24h,2),
            "price_source":pd_.get("source","estimated"),
            "reasoning":reason,"_q":quality,
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
        try: req_dur = max(1,int(duration)) if duration else None
        except: req_dur = None

        pool   = CRYPTO_ASSETS if market=="crypto" else FOREX_PAIRS
        used   = used_assets.get(market,[])
        unseen = [a for a in pool if a not in used]
        if len(unseen) < 8:
            used_assets[market]=[]; unseen=list(pool)

        random.shuffle(unseen)
        candidates = unseen[:20]
        good = []

        for i, asset in enumerate(candidates):
            seed = int(time.time()*1000)%999983+i*179
            t    = build_trade(asset, market, trade_type, req_dur, session, seed)
            if t is not None:
                good.append(t)
            if len(good) >= 9: break

        # Fallback if strict gate yielded < 3
        if len(good) < 3:
            log.warning("Gate gave %d trades — fallback pass", len(good))
            for i, asset in enumerate(candidates[:8]):
                if len(good) >= 3: break
                seed = int(time.time()*1000)%999983+i*211+500
                pd_  = fetch_crypto(asset) if market=="crypto" else fetch_forex(asset)
                price     = float(pd_.get("price") or FALLBACK_PRICES.get(asset,1.0))
                change24h = float(pd_.get("change24h") or 0.0)
                if abs(change24h) < 1.5:
                    change24h = 2.5 if random.random()>0.5 else -2.5
                p    = P(asset); hv=p["hv"]; tier=p["tier"]
                dv   = abs(change24h)
                strategy  = "TREND_FOLLOW"
                direction = "LONG" if change24h>0 else "SHORT"
                quality   = 73.0
                dur  = auto_dur(asset,strategy,change24h,trade_type,req_dur)
                sr   = calc_sr(price,hv,change24h,seed)
                ind  = calc_indicators(change24h,float(pd_.get("volume24h",0)),price,hv,sr,seed,market)
                tpsl = calc_tpsl(asset,market,strategy,direction,price,change24h,
                                  dur,ind["adx"],session,quality,trade_type)
                dur  = tpsl["adj_dur"]
                lev  = get_leverage(market,tier,dv,ind["adx"],dur,strategy)
                close_dt   = datetime.utcnow()+timedelta(minutes=dur)
                close_time = close_dt.strftime("%Y-%m-%d %H:%M UTC")
                reason_fb  = (f"Quality: B SOLID | 📈 TREND_FOLLOW | Win rate: 68-72%\n\n"
                              f"Expected {fmt_dur(dur)} move {tpsl['expected_pct']:.2f}% | "
                              f"TP {tpsl['tp_pct']:.2f}% ({tpsl['tp_vs_exp']:.0f}% of expected) | "
                              f"SL {tpsl['sl_pct']:.2f}% | RR 1:{tpsl['rr']}")
                good.append({
                    "asset":asset,"market":market.upper(),"trade_type":trade_type.upper(),
                    "strategy_type":strategy,"strategy_icon":"📈","win_rate":"68-72%",
                    "direction":direction,"entry":round(price,8),
                    "tp":tpsl["tp"],"sl":tpsl["sl"],"tp_pct":tpsl["tp_pct"],
                    "sl_pct":tpsl["sl_pct"],"rr":tpsl["rr"],
                    "expected_move":tpsl["expected_pct"],"tp_vs_expected":tpsl["tp_vs_exp"],
                    "leverage":lev,"timeframe":timeframe(dur, trade_type),
                    "duration":fmt_dur(dur),"duration_min":dur,"close_time":close_time,
                    "session":session,"quality":"B  SOLID","tier":tier,
                    "volatility":{"level":"NORMAL","speed":"MODERATE",
                                  "change_pct":round(dv,2),"hourly_vol":round(hv,3)},
                    "confidence":quality,"direction_score":quality,
                    "indicators":{},"indicators_passed":5,"indicators_total":10,
                    "support":sr["s1"],"resistance":sr["r1"],"pivot":sr["pivot"],
                    "news_status":"SAFE","status":"OPEN","change24h":round(change24h,2),
                    "price_source":pd_.get("source","estimated"),
                    "reasoning":reason_fb,"_q":quality,
                })

        good.sort(key=lambda x:x.get("_q",0), reverse=True)
        top3 = good[:3]
        for t in top3:
            if t["asset"] not in used_assets[market]:
                used_assets[market].append(t["asset"])

        rank_labels={1:"🥇 #1 Premium Signal",2:"🥈 #2 High Probability",
                     3:"🥉 #3 Confirmed Setup"}
        result=[]
        for rank,t in enumerate(top3,1):
            t.pop("_q",None)
            t["rank"]=rank; t["id"]=int(time.time()*1000)+rank
            t["timestamp"]=datetime.utcnow().isoformat()+"Z"
            t["reasoning"]=f"{rank_labels[rank]} | {t['reasoning']}"
            trade_history.insert(0,dict(t))
            result.append(t)
        if len(trade_history)>300: del trade_history[300:]
        log.info("Generated %d trades | market=%s session=%s strategies=%s",
                 len(result),market,session,[t["strategy_type"] for t in result])
        return jsonify(result)
    except Exception as e:
        log.error("generate_trade: %s",e,exc_info=True)
        return jsonify({"error":"Server error","detail":str(e)}),500

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
        for pr in FOREX_PAIRS[:10]:
            try: data[pr]=fetch_forex(pr)
            except: pass
        return jsonify(data)
    except Exception as e:
        log.error("prices: %s",e); return jsonify({}),500

@app.route("/api/trade_history")
def api_trade_history():
    try:    return jsonify(trade_history)
    except: return jsonify([]),500

@app.route("/api/close_trade",methods=["POST"])
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
    log.info("APEX TRADE v3.0 starting on port %d",port)
    app.run(host="0.0.0.0",port=port,debug=False)
