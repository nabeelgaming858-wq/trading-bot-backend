import os
import json
import time
import threading
from datetime import datetime, timedelta
from collections import deque
from functools import lru_cache

import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

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

# Default timeframes for trade types
TRADE_TIMEFRAMES = {
    "scalp": "15m",
    "intraday": "1h",
    "swing": "4h"
}

# All available intervals for custom selection (Binance-compatible)
AVAILABLE_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# Cache
data_cache = {}
cache_lock = threading.Lock()
CACHE_DURATION = 60  # seconds

# APIs (free, no keys)
BINANCE_BASE = "https://api.binance.com/api/v3"
FOREX_BASE = "https://api.exchangerate.host"

# Keep track of last recommended assets to avoid repeats (optional, but we now return top 3)
last_assets = deque(maxlen=10)

# ==================== HELPER FUNCTIONS ====================
def fetch_binance_klines(symbol, interval, limit=100):
    """Fetch OHLC data from Binance."""
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

def fetch_forex_ohlc(symbol, interval, limit=100):
    """
    Fetch forex data (simulated intraday from daily).
    """
    try:
        days_map = {"1m": 1, "5m": 1, "15m": 2, "30m": 3, "1h": 7, "4h": 30, "1d": 90}
        days = days_map.get(interval, 7)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        base = symbol[:3]
        quote = symbol[3:]
        url = f"{FOREX_BASE}/timeseries"
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
        print(f"Error fetching forex {symbol}: {e}")
        return None

def get_ohlc(asset, asset_type, timeframe):
    cache_key = f"{asset}_{timeframe}"
    now = time.time()
    with cache_lock:
        if cache_key in data_cache and now - data_cache[cache_key]["timestamp"] < CACHE_DURATION:
            return data_cache[cache_key]["data"]
    if asset_type == "crypto":
        df = fetch_binance_klines(asset, timeframe)
    else:
        df = fetch_forex_ohlc(asset, timeframe)
    if df is not None and not df.empty:
        with cache_lock:
            data_cache[cache_key] = {"data": df, "timestamp": now}
    return df

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

def calculate_duration(asset_type, trade_type, atr, price):
    if trade_type == "scalp":
        return "2h"
    elif trade_type == "intraday":
        return "24h"
    else:  # swing
        return "7d"

def analyze_asset(asset, asset_type, timeframe, trade_type):
    """
    Generate a signal for a single asset. Returns None if data insufficient,
    otherwise a dict with signal details and a score.
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

    # Trend and indicators
    ema21 = last.get("trend_ema_fast", last.get("EMA_21", price))
    ema50 = last.get("EMA_50", price)
    macd = last.get("MACD_macd", 0)
    macd_signal = last.get("MACD_signal", 0)
    rsi = last.get("momentum_rsi", 50)
    bb_high = last.get("volatility_bbh", price * 1.02)
    bb_low = last.get("volatility_bbl", price * 0.98)

    # Trend direction
    if price > ema21 and ema21 > ema50:
        trend = "bullish"
    elif price < ema21 and ema21 < ema50:
        trend = "bearish"
    else:
        trend = "neutral"

    # Confluence scoring (0-4)
    confluence = 0
    reasons = []

    if trend == "bullish":
        confluence += 1
        reasons.append("Price above EMA21/50 (bullish trend)")
    elif trend == "bearish":
        confluence += 1
        reasons.append("Price below EMA21/50 (bearish trend)")

    if price <= bb_low and last["close"] > bb_low:
        confluence += 1
        reasons.append("Bollinger bounce (bullish)")
    elif price >= bb_high and last["close"] < bb_high:
        confluence += 1
        reasons.append("Bollinger rejection (bearish)")

    if macd > macd_signal and prev.get("MACD_macd", 0) <= prev.get("MACD_signal", 0):
        confluence += 1
        reasons.append("MACD bullish crossover")
    elif macd < macd_signal and prev.get("MACD_macd", 0) >= prev.get("MACD_signal", 0):
        confluence += 1
        reasons.append("MACD bearish crossover")

    if rsi < 30 and last["close"] > prev["close"]:
        confluence += 1
        reasons.append("RSI oversold bullish divergence")
    elif rsi > 70 and last["close"] < prev["close"]:
        confluence += 1
        reasons.append("RSI overbought bearish divergence")

    # Determine direction based on bullish/bearish reasons count
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

    # Compute SL with ATR buffer
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
    duration = calculate_duration(asset_type, trade_type, atr, price)

    # Compute a score for ranking (higher is better)
    # Base score from confluence (0-4) * 10 + extra for trend strength, etc.
    score = confluence * 10
    # Add small bonus for RSI momentum
    if 40 < rsi < 60:
        score += 2  # neutral zone, less risk
    if direction == "LONG" and trend == "bullish":
        score += 5
    elif direction == "SHORT" and trend == "bearish":
        score += 5
    # Penalize very high volatility
    if volatility_pct > 5:
        score -= 5

    signal = {
        "asset": asset,
        "signal": direction,
        "entry": round(entry, 4),
        "stop_loss": round(sl, 4),
        "take_profit": round(tp, 4),
        "leverage": leverage,
        "duration": duration,
        "timeframe": timeframe,
        "confidence": "High" if confluence >= 3 else "Medium",
        "rationale": " | ".join(reasons[:3]),
        "score": score
    }
    return signal

def get_top_trades(asset_type, trade_type, custom_timeframe=None, n=3):
    """
    Return top n trades for the given asset type, sorted by score descending.
    """
    assets = CRYPTO_ASSETS if asset_type == "crypto" else FOREX_ASSETS
    # Determine timeframe: custom overrides trade_type default
    if custom_timeframe and custom_timeframe in AVAILABLE_TIMEFRAMES:
        timeframe = custom_timeframe
    else:
        timeframe = TRADE_TIMEFRAMES.get(trade_type, "1h")

    signals = []
    for asset in assets:
        # Optionally skip recently used (to add variety)
        # if asset in last_assets:
        #     continue
        sig = analyze_asset(asset, asset_type, timeframe, trade_type)
        if sig:
            signals.append(sig)
    # Sort by score descending
    signals.sort(key=lambda x: x["score"], reverse=True)
    # Keep top n
    top = signals[:n]
    # Record assets used (optional)
    for sig in top:
        last_assets.append(sig["asset"])
    return top

# ==================== FLASK ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html", timeframes=AVAILABLE_TIMEFRAMES)

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json()
    asset_type = data.get("asset_type", "crypto")
    trade_type = data.get("trade_type", "intraday")
    custom_timeframe = data.get("custom_timeframe", "")
    top_trades = get_top_trades(asset_type, trade_type, custom_timeframe, n=3)
    if top_trades:
        return jsonify({"success": True, "signals": top_trades})
    else:
        # Fallback: should never happen because we have many assets, but just in case
        return jsonify({"success": False, "message": "No tradable assets at the moment."})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
