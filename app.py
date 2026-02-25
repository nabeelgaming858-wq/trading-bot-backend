import os
import time
import threading
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

# Default timeframes for preset trade types
PRESET_TIMEFRAMES = {
    "scalp": "15m",
    "intraday": "1h",
    "swing": "4h"
}

# Optional: Twelve Data API key (set as environment variable for real-time forex)
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")

# Cache for OHLC data
data_cache = {}
cache_lock = threading.Lock()
CACHE_DURATION = 60  # seconds

# Keep track of last recommended assets to avoid repeats
last_assets = deque(maxlen=10)  # increased to accommodate top 3

# Binance API (no key needed)
BINANCE_BASE = "https://api.binance.com/api/v3"

# ==================== DATA FETCHING ====================
def fetch_binance_klines(symbol, interval, limit=100):
    """Fetch OHLC data from Binance for crypto assets."""
    url = f"{BINANCE_BASE}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        return df
    except Exception as e:
        print(f"Error fetching Binance {symbol} {interval}: {e}")
        return None

def fetch_twelvedata_forex(symbol, interval, limit=100):
    """Fetch real-time forex data from Twelve Data (requires API key)."""
    if not TWELVE_DATA_API_KEY:
        return None
    # Map interval to Twelve Data format
    interval_map = {
        "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "1h", "4h": "4h", "1d": "1day"
    }
    td_interval = interval_map.get(interval, "1h")
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol.replace("/", ""),
        "interval": td_interval,
        "outputsize": limit,
        "apikey": TWELVE_DATA_API_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "values" not in data:
            return None
        values = data["values"]
        df = pd.DataFrame(values)
        df = df.rename(columns={
            "datetime": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        return df.iloc[::-1]  # reverse to chronological
    except Exception as e:
        print(f"Twelve Data error: {e}")
        return None

def fetch_forex_ohlc_fallback(symbol, interval, limit=100):
    """Fallback forex using exchangerate.host (daily only). Simulates intraday."""
    try:
        days_map = {"1m": 1, "5m": 1, "15m": 2, "30m": 3, "1h": 7, "4h": 30, "1d": 90}
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
        df = pd.DataFrame({"timestamp": pd.to_datetime(dates), "close": prices})
        df.set_index("timestamp", inplace=True)
        df["open"] = df["close"].shift(1).fillna(df["close"])
        # Simulate high/low with random noise
        noise = np.random.uniform(0.001, 0.003, len(df))
        df["high"] = df[["open", "close"]].max(axis=1) * (1 + noise)
        df["low"] = df[["open", "close"]].min(axis=1) * (1 - noise)
        df["volume"] = 1e6
        return df
    except Exception as e:
        print(f"Forex fallback error: {e}")
        return None

def get_ohlc(asset, asset_type, timeframe):
    """Get OHLC data from cache or fetch."""
    cache_key = f"{asset}_{timeframe}"
    now = time.time()
    with cache_lock:
        if cache_key in data_cache and now - data_cache[cache_key]["timestamp"] < CACHE_DURATION:
            return data_cache[cache_key]["data"]

    if asset_type == "crypto":
        df = fetch_binance_klines(asset, timeframe)
    else:
        # Try Twelve Data first if key available
        if TWELVE_DATA_API_KEY:
            df = fetch_twelvedata_forex(asset, timeframe)
        else:
            df = None
        if df is None:
            df = fetch_forex_ohlc_fallback(asset, timeframe)

    if df is not None and not df.empty:
        with cache_lock:
            data_cache[cache_key] = {"data": df, "timestamp": now}
    return df

# ==================== INDICATORS & SIGNAL LOGIC ====================
def calculate_indicators(df):
    """Add technical indicators."""
    df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    return df

def find_swing_points(df, window=5):
    """Find recent swing high and low."""
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

def calculate_duration_from_timeframe(timeframe):
    """Convert timeframe like '15m' to human-readable duration for display."""
    if timeframe.endswith('m'):
        return timeframe
    elif timeframe.endswith('h'):
        return timeframe
    elif timeframe.endswith('d'):
        return timeframe
    else:
        return timeframe

def generate_signal(asset, asset_type, timeframe):
    """
    Generate a signal for a specific asset and timeframe.
    Returns dict or None.
    """
    df = get_ohlc(asset, asset_type, timeframe)
    if df is None or len(df) < 50:
        return None

    df = calculate_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last["close"]
    atr = last.get("atr", (df["high"].iloc[-14:].max() - df["low"].iloc[-14:].min()) / 14)
    if pd.isna(atr) or atr == 0:
        atr = price * 0.01

    swing_high, swing_low = find_swing_points(df)

    # Trend and confluence scoring (simplified for brevity, same as before)
    ema21 = last.get("trend_ema_fast", last.get("EMA_21", price))
    ema50 = last.get("EMA_50", price)
    macd = last.get("MACD_macd", 0)
    macd_signal = last.get("MACD_signal", 0)
    rsi = last.get("momentum_rsi", 50)
    bb_high = last.get("volatility_bbh", price * 1.02)
    bb_low = last.get("volatility_bbl", price * 0.98)

    if price > ema21 and ema21 > ema50:
        trend = "bullish"
    elif price < ema21 and ema21 < ema50:
        trend = "bearish"
    else:
        trend = "neutral"

    confluence = 0
    reasons = []

    if trend == "bullish":
        confluence += 1
        reasons.append("Price above EMA21/50")
    elif trend == "bearish":
        confluence += 1
        reasons.append("Price below EMA21/50")

    if price <= bb_low and last["close"] > bb_low:
        confluence += 1
        reasons.append("Bollinger bounce")
    elif price >= bb_high and last["close"] < bb_high:
        confluence += 1
        reasons.append("Bollinger rejection")

    if macd > macd_signal and prev.get("MACD_macd", 0) <= prev.get("MACD_signal", 0):
        confluence += 1
        reasons.append("MACD bullish cross")
    elif macd < macd_signal and prev.get("MACD_macd", 0) >= prev.get("MACD_signal", 0):
        confluence += 1
        reasons.append("MACD bearish cross")

    if rsi < 30 and last["close"] > prev["close"]:
        confluence += 1
        reasons.append("RSI oversold bounce")
    elif rsi > 70 and last["close"] < prev["close"]:
        confluence += 1
        reasons.append("RSI overbought drop")

    if confluence < 3:
        return None

    bullish_count = sum(1 for r in reasons if "bull" in r.lower())
    bearish_count = sum(1 for r in reasons if "bear" in r.lower())
    if bullish_count > bearish_count:
        direction = "LONG"
    elif bearish_count > bullish_count:
        direction = "SHORT"
    else:
        direction = "LONG" if trend == "bullish" else "SHORT" if trend == "bearish" else None
    if not direction:
        return None

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
    duration_display = calculate_duration_from_timeframe(timeframe)

    return {
        "asset": asset,
        "signal": direction,
        "entry": round(entry, 4),
        "stop_loss": round(sl, 4),
        "take_profit": round(tp, 4),
        "leverage": leverage,
        "duration": duration_display,
        "timeframe": timeframe,
        "confidence": "High" if confluence >= 4 else "Medium",
        "rationale": " | ".join(reasons[:2])  # shorter for display
    }

def get_top_trades(asset_type, timeframe, count=3):
    """Scan all assets and return the top N signals (by confidence)."""
    assets = CRYPTO_ASSETS if asset_type == "crypto" else FOREX_ASSETS
    signals = []
    for asset in assets:
        # Avoid recently recommended assets (if any)
        if asset in last_assets:
            continue
        sig = generate_signal(asset, asset_type, timeframe)
        if sig:
            # Score: High=2, Medium=1
            score = 2 if sig["confidence"] == "High" else 1
            signals.append((score, sig))
    # Sort by score descending, then maybe by risk-reward? Keep top N
    signals.sort(key=lambda x: x[0], reverse=True)
    top_signals = [sig for _, sig in signals[:count]]
    # Remember these assets to avoid repeats
    for sig in top_signals:
        last_assets.append(sig["asset"])
    return top_signals

# ==================== FLASK ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json()
    asset_type = data.get("asset_type", "crypto")
    # Support both preset trade_type and custom timeframe
    trade_type = data.get("trade_type")  # may be "scalp", "intraday", "swing"
    custom_timeframe = data.get("custom_timeframe")
    if custom_timeframe:
        timeframe = custom_timeframe
    elif trade_type and trade_type in PRESET_TIMEFRAMES:
        timeframe = PRESET_TIMEFRAMES[trade_type]
    else:
        timeframe = "1h"  # default
    top_signals = get_top_trades(asset_type, timeframe, count=3)
    if top_signals:
        return jsonify({"success": True, "signals": top_signals})
    else:
        return jsonify({"success": False, "message": "No high-probability trades found."})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
