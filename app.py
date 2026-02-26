import os
import time
import threading
import random
from datetime import datetime, timedelta
from collections import deque

import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from ta import add_all_ta_features
from ta.volatility import AverageTrueRange

app = Flask(__name__)

# ==================== CONFIGURATION ====================
# API Keys from environment variables (set in Cloud Run)
COINMARKETCAP_KEY = 
FINNHUB_KEY = 
TWELVEDATA_KEY = 
ITICK_TOKEN = 
ALPHA_VANTAGE_KEY = 

CRYPTO_ASSETS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "DOTUSDT", "MATICUSDT", "SHIBUSDT", "TRXUSDT", "AVAXUSDT", "UNIUSDT", "ATOMUSDT",
    "LTCUSDT", "BCHUSDT", "NEARUSDT", "FILUSDT", "APTUSDT", "TONUSDT", "LINKUSDT",
    "XLMUSDT", "ALGOUSDT", "VETUSDT", "ICPUSDT"
]

FOREX_ASSETS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "GBPCHF", "USDNOK", "USDSEK"
]

TRADE_TIMEFRAMES = {
    "scalp": "15m",
    "intraday": "1h",
    "swing": "4h",
    "custom": "1h"
}

data_cache = {}
cache_lock = threading.Lock()
CACHE_DURATION = 60
last_assets = deque(maxlen=15)

# ==================== API FETCH FUNCTIONS ====================
# -------------------- CRYPTO --------------------
def fetch_coinmarketcap(symbol, interval, limit=100):
    """Fetch OHLC from CoinMarketCap (requires API key)."""
    if not COINMARKETCAP_KEY:
        return None
    # Map symbol to CMC ID (simplified – in production use /v1/cryptocurrency/map)
    mapping = {
        "BTCUSDT": "1", "ETHUSDT": "1027", "BNBUSDT": "1839", "SOLUSDT": "5426",
        "XRPUSDT": "52", "ADAUSDT": "2010", "DOGEUSDT": "74", "DOTUSDT": "6636",
        "MATICUSDT": "3890", "SHIBUSDT": "5994", "TRXUSDT": "1958", "AVAXUSDT": "5805",
        "UNIUSDT": "7083", "ATOMUSDT": "3794", "LTCUSDT": "2", "BCHUSDT": "1831",
        "NEARUSDT": "6535", "FILUSDT": "2280", "APTUSDT": "21794", "TONUSDT": "11419",
        "LINKUSDT": "1975", "XLMUSDT": "512", "ALGOUSDT": "4030", "VETUSDT": "3077",
        "ICPUSDT": "8916"
    }
    coin_id = mapping.get(symbol)
    if not coin_id:
        return None
    url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical"
    params = {
        "id": coin_id,
        "convert": "USD",
        "count": limit,
        "interval": "5m" if "15m" in interval else "hourly" if "1h" in interval else "daily"
    }
    headers = {"X-CMC_PRO_API_KEY": COINMARKETCAP_KEY}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        quotes = data["data"]["quotes"]
        records = []
        for q in quotes:
            records.append({
                "timestamp": pd.to_datetime(q["timestamp"]),
                "open": q["quote"]["USD"]["open"],
                "high": q["quote"]["USD"]["high"],
                "low": q["quote"]["USD"]["low"],
                "close": q["quote"]["USD"]["close"],
                "volume": q["quote"]["USD"]["volume"]
            })
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(f"CMC error: {e}")
        return None

def fetch_coingecko(symbol, interval, limit=100):
    """CoinGecko – no key required."""
    mapping = {
        "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "BNBUSDT": "binancecoin",
        "SOLUSDT": "solana", "XRPUSDT": "ripple", "ADAUSDT": "cardano",
        "DOGEUSDT": "dogecoin", "DOTUSDT": "polkadot", "MATICUSDT": "matic-network",
        "SHIBUSDT": "shiba-inu", "TRXUSDT": "tron", "AVAXUSDT": "avalanche-2",
        "UNIUSDT": "uniswap", "ATOMUSDT": "cosmos", "LTCUSDT": "litecoin",
        "BCHUSDT": "bitcoin-cash", "NEARUSDT": "near", "FILUSDT": "filecoin",
        "APTUSDT": "aptos", "TONUSDT": "the-open-network", "LINKUSDT": "chainlink",
        "XLMUSDT": "stellar", "ALGOUSDT": "algorand", "VETUSDT": "vechain",
        "ICPUSDT": "internet-computer"
    }
    coin_id = mapping.get(symbol)
    if not coin_id:
        return None
    days_map = {"15m": 2, "1h": 7, "4h": 30, "1d": 90, "1w": 365}
    days = days_map.get(interval, 7)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()  # list of [timestamp, open, high, low, close]
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['volume'] = 1e6  # dummy volume
        return df
    except Exception as e:
        print(f"CoinGecko error: {e}")
        return None

def fetch_finnhub_crypto(symbol, interval, limit=100):
    """Finnhub – requires API key."""
    if not FINNHUB_KEY:
        return None
    # Finnhub uses different symbol format (e.g., BINANCE:BTCUSDT)
    finnhub_symbol = f"BINANCE:{symbol}"
    resolution_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D", "1w": "W"}
    res = resolution_map.get(interval, "60")
    url = "https://finnhub.io/api/v1/crypto/candle"
    params = {
        "symbol": finnhub_symbol,
        "resolution": res,
        "count": limit,
        "token": FINNHUB_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("s") != "ok":
            return None
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["t"], unit="s"),
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"]
        })
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(f"Finnhub error: {e}")
        return None

# -------------------- FOREX --------------------
def fetch_twelvedata(symbol, interval, limit=100):
    """TwelveData – requires API key."""
    if not TWELVEDATA_KEY:
        return None
    interval_map = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1day", "1w": "1week"}
    td_interval = interval_map.get(interval, "1h")
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol.replace("USD", "/USD"),  # TwelveData expects EUR/USD format
        "interval": td_interval,
        "outputsize": limit,
        "apikey": TWELVEDATA_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "values" not in data:
            return None
        records = []
        for item in data["values"]:
            records.append({
                "timestamp": pd.to_datetime(item["datetime"]),
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
                "volume": 0
            })
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"TwelveData error: {e}")
        return None

def fetch_itick(symbol, interval, limit=100):
    """iTick – requires token."""
    if not ITICK_TOKEN:
        return None
    # Map interval to iTick kType: 1=1min, 2=5min, 3=15min, 4=30min, 5=1h, 6=4h, 7=1d, 8=1w
    ktype_map = {"15m": 3, "1h": 5, "4h": 6, "1d": 7, "1w": 8}
    ktype = ktype_map.get(interval, 5)
    url = "https://api.itick.org/forex/kline"
    params = {
        "region": "GB",
        "code": symbol,
        "kType": ktype,
        "limit": limit
    }
    headers = {"token": ITICK_TOKEN}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            return None
        klines = data["data"]
        records = []
        for k in klines:
            records.append({
                "timestamp": pd.to_datetime(k["t"], unit="ms"),
                "open": k["o"],
                "high": k["h"],
                "low": k["l"],
                "close": k["c"],
                "volume": k.get("v", 0)
            })
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(f"iTick error: {e}")
        return None

def fetch_alphavantage_forex(symbol, interval, limit=100):
    """Alpha Vantage – requires API key."""
    if not ALPHA_VANTAGE_KEY:
        return None
    # Map interval
    interval_map = {"15m": "15min", "1h": "60min", "4h": "60min", "1d": "DAILY", "1w": "WEEKLY"}
    av_interval = interval_map.get(interval, "60min")
    function = "FX_INTRADAY" if "min" in av_interval else "FX_DAILY"
    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "from_symbol": symbol[:3],
        "to_symbol": symbol[3:],
        "interval": av_interval if "min" in av_interval else None,
        "apikey": ALPHA_VANTAGE_KEY,
        "outputsize": "compact"
    }
    params = {k: v for k, v in params.items() if v is not None}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Find time series key
        ts_key = None
        for key in data:
            if "Time Series FX" in key:
                ts_key = key
                break
        if not ts_key:
            return None
        time_series = data[ts_key]
        records = []
        for dt_str, values in time_series.items():
            records.append({
                "timestamp": pd.to_datetime(dt_str),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": 0
            })
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()
        return df.iloc[-limit:]
    except Exception as e:
        print(f"Alpha Vantage error: {e}")
        return None

def fetch_exchangerate_forex(symbol, interval, limit=100):
    """Fallback: exchangerate.host (daily only)."""
    try:
        days_map = {"15m": 2, "1h": 7, "4h": 30, "1d": 90, "1w": 365}
        days = days_map.get(interval, 7)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        base = symbol[:3]
        quote = symbol[3:]
        url = "https://api.exchangerate.host/timeseries"
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "base": base,
            "symbols": quote
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("rates"):
            return None
        dates = sorted(data["rates"].keys())
        prices = [data["rates"][d][quote] for d in dates]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(dates),
            "close": prices
        })
        df.set_index("timestamp", inplace=True)
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.random.uniform(0, 0.002))
        df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.random.uniform(0, 0.002))
        df["volume"] = 1e6
        return df
    except Exception as e:
        print(f"Exchangerate error: {e}")
        return None

# -------------------- MASTER FETCHER WITH FALLBACKS --------------------
def get_ohlc(asset, asset_type, timeframe):
    cache_key = f"{asset}_{timeframe}"
    now = time.time()
    with cache_lock:
        if cache_key in data_cache and now - data_cache[cache_key]["timestamp"] < CACHE_DURATION:
            return data_cache[cache_key]["data"]

    df = None
    if asset_type == "crypto":
        # Chain: CoinMarketCap -> Finnhub -> CoinGecko
        if COINMARKETCAP_KEY:
            df = fetch_coinmarketcap(asset, timeframe)
        if df is None and FINNHUB_KEY:
            df = fetch_finnhub_crypto(asset, timeframe)
        if df is None:
            df = fetch_coingecko(asset, timeframe)  # no key needed
    else:
        # Chain: TwelveData -> iTick -> Alpha Vantage -> exchangerate.host
        if TWELVEDATA_KEY:
            df = fetch_twelvedata(asset, timeframe)
        if df is None and ITICK_TOKEN:
            df = fetch_itick(asset, timeframe)
        if df is None and ALPHA_VANTAGE_KEY:
            df = fetch_alphavantage_forex(asset, timeframe)
        if df is None:
            df = fetch_exchangerate_forex(asset, timeframe)

    if df is not None and not df.empty:
        with cache_lock:
            data_cache[cache_key] = {"data": df, "timestamp": now}
    return df

# ==================== INDICATORS & SIGNAL GENERATION ====================
def calculate_indicators(df):
    df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    return df

def find_swing_points(df, window=5):
    df["swing_high"] = df["high"].rolling(window=window, center=True).max()
    df["swing_low"] = df["low"].rolling(window=window, center=True).min()
    last_swing_high = df["swing_high"].iloc[-2] if len(df) > 2 else df["high"].iloc[-1]
    last_swing_low = df["swing_low"].iloc[-2] if len(df) > 2 else df["low"].iloc[-1]
    return last_swing_high, last_swing_low

def calculate_dynamic_leverage(volatility_percent, asset_type):
    if asset_type == "crypto":
        base_max = 10
    else:
        base_max = 50
    if volatility_percent < 0.5:
        return base_max
    elif volatility_percent < 1:
        return int(base_max * 0.7)
    elif volatility_percent < 2:
        return int(base_max * 0.5)
    elif volatility_percent < 3:
        return int(base_max * 0.3)
    else:
        return int(base_max * 0.2)

def generate_signal(asset, asset_type, trade_type, custom_timeframe=None):
    if custom_timeframe:
        timeframe = custom_timeframe
    else:
        timeframe = TRADE_TIMEFRAMES.get(trade_type, "1h")

    df = get_ohlc(asset, asset_type, timeframe)
    if df is None or len(df) < 30:
        return None

    df = calculate_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last["close"]
    atr = last.get("atr", (df["high"].iloc[-14:].max() - df["low"].iloc[-14:].min()) / 14)
    if pd.isna(atr) or atr == 0:
        atr = price * 0.01

    swing_high, swing_low = find_swing_points(df)

    ema21 = last.get("trend_ema_fast", last.get("EMA_21", price))
    ema50 = last.get("EMA_50", price)
    macd = last.get("MACD_macd", 0)
    macd_signal = last.get("MACD_signal", 0)
    rsi = last.get("momentum_rsi", 50)
    bb_high = last.get("volatility_bbh", price * 1.02)
    bb_low = last.get("volatility_bbl", price * 0.98)

    # Trend
    if price > ema21 and ema21 > ema50:
        trend = "bullish"
    elif price < ema21 and ema21 < ema50:
        trend = "bearish"
    else:
        trend = "neutral"

    # Confluence score (0-6 now with more factors)
    confluence = 0
    reasons = []

    if trend == "bullish":
        confluence += 1
        reasons.append("EMA bullish alignment")
    elif trend == "bearish":
        confluence += 1
        reasons.append("EMA bearish alignment")

    if price <= bb_low and last["close"] > bb_low:
        confluence += 1
        reasons.append("Bollinger bounce (bullish)")
    elif price >= bb_high and last["close"] < bb_high:
        confluence += 1
        reasons.append("Bollinger rejection (bearish)")

    if macd > macd_signal and prev.get("MACD_macd", 0) <= prev.get("MACD_signal", 0):
        confluence += 1
        reasons.append("MACD bullish cross")
    elif macd < macd_signal and prev.get("MACD_macd", 0) >= prev.get("MACD_signal", 0):
        confluence += 1
        reasons.append("MACD bearish cross")

    if rsi < 30:
        confluence += 1
        reasons.append("RSI oversold")
    elif rsi > 70:
        confluence += 1
        reasons.append("RSI overbought")

    # Additional: price vs previous close (momentum)
    if last["close"] > prev["close"]:
        confluence += 1
        reasons.append("Price up")
    elif last["close"] < prev["close"]:
        confluence += 1
        reasons.append("Price down")

    # Volume confirmation
    if last["volume"] > df["volume"].rolling(20).mean().iloc[-1]:
        confluence += 1
        reasons.append("Volume above average")

    # Determine direction based on confluence reasons
    bullish_count = sum(1 for r in reasons if "bull" in r.lower() or "up" in r.lower() or "oversold" in r.lower())
    bearish_count = sum(1 for r in reasons if "bear" in r.lower() or "down" in r.lower() or "overbought" in r.lower())
    if bullish_count > bearish_count:
        direction = "LONG"
    elif bearish_count > bullish_count:
        direction = "SHORT"
    else:
        direction = "LONG" if trend == "bullish" else "SHORT" if trend == "bearish" else None
    if not direction:
        return None

    # SL with buffer
    buffer = atr * 1.5
    if direction == "LONG":
        sl = swing_low - buffer
        entry = price
        tp = entry + (entry - sl) * 1.5
    else:
        sl = swing_high + buffer
        entry = price
        tp = entry - (sl - entry) * 1.5

    volatility_pct = (atr / price) * 100
    leverage = calculate_dynamic_leverage(volatility_pct, asset_type)
    duration = f"{timeframe}"  # simplified

    # Score: confluence + random tie-breaker
    score = confluence + random.random()

    signal = {
        "asset": asset,
        "signal": direction,
        "entry": round(entry, 4),
        "stop_loss": round(sl, 4),
        "take_profit": round(tp, 4),
        "leverage": leverage,
        "duration": duration,
        "timeframe": timeframe,
        "confidence": "High" if confluence >= 4 else "Medium" if confluence >= 2 else "Low",
        "rationale": " | ".join(reasons[:3]),
        "score": score
    }
    return signal

def get_top_trades(asset_type, trade_type, count=3, custom_timeframe=None):
    assets = CRYPTO_ASSETS if asset_type == "crypto" else FOREX_ASSETS
    signals = []
    for asset in assets:
        if asset in last_assets:
            continue
        signal = generate_signal(asset, asset_type, trade_type, custom_timeframe)
        if signal:
            signals.append(signal)
    signals.sort(key=lambda x: x["score"], reverse=True)
    top_signals = signals[:count]
    for s in top_signals:
        last_assets.append(s["asset"])
    return top_signals

# ==================== FLASK ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json()
    asset_type = data.get("asset_type", "crypto")
    trade_type = data.get("trade_type", "intraday")
    custom_timeframe = data.get("custom_timeframe")
    if custom_timeframe == "" or custom_timeframe is None:
        custom_timeframe = None
    top_trades = get_top_trades(asset_type, trade_type, count=3, custom_timeframe=custom_timeframe)
    if top_trades:
        return jsonify({"success": True, "signals": top_trades})
    else:
        return jsonify({"success": True, "signals": [], "message": "No trades met criteria. Try adjusting settings."})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
