import os
import json
import time
import threading
from datetime import datetime, timedelta
from collections import deque
from functools import lru_cache
import math

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
# Asset lists (crypto 25, forex 15)
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

# Timeframes mapping for trade types
TRADE_TIMEFRAMES = {
    "scalp": "15m",
    "intraday": "1h",
    "swing": "4h",
    "custom": "1h"  # default for custom, will be overridden
}

# Cache for OHLC data
data_cache = {}
cache_lock = threading.Lock()
CACHE_DURATION = 60  # seconds

# Binance API base (no key needed for public endpoints)
BINANCE_BASE = "https://api.binance.com/api/v3"

# Forex API (exchangerate.host - free, no key)
FOREX_BASE = "https://api.exchangerate.host"

# Keep track of last recommended assets to avoid repeats
last_assets = deque(maxlen=15)  # Increased to track more assets

# ==================== HELPER FUNCTIONS ====================
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

def fetch_forex_ohlc(symbol, interval, limit=100):
    """Fetch OHLC for forex pairs from exchangerate.host."""
    try:
        days_map = {"15m": 2, "1h": 7, "4h": 30, "1d": 60, "1w": 180}
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
    """Get OHLC data from cache or fetch."""
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
    """Add technical indicators to dataframe."""
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
    """Determine leverage based on volatility."""
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

def calculate_duration(asset_type, trade_type, atr, price, custom_minutes=None):
    """Estimate duration based on ATR, trade type, or custom input."""
    if custom_minutes:
        # Convert custom minutes to readable format
        if custom_minutes < 60:
            return f"{custom_minutes}m"
        elif custom_minutes < 1440:
            hours = custom_minutes // 60
            return f"{hours}h"
        else:
            days = custom_minutes // 1440
            return f"{days}d"
    
    if trade_type == "scalp":
        return "2h"
    elif trade_type == "intraday":
        return "24h"
    elif trade_type == "swing":
        return "7d"
    else:
        return "1h"

def map_duration_to_timeframe(duration_str):
    """Convert duration string to Binance interval format."""
    if duration_str.endswith('m'):
        minutes = int(duration_str[:-1])
        if minutes <= 15:
            return "15m"
        elif minutes <= 30:
            return "30m"
        elif minutes <= 60:
            return "1h"
        elif minutes <= 240:
            return "4h"
        else:
            return "1h"
    elif duration_str.endswith('h'):
        hours = int(duration_str[:-1])
        if hours <= 1:
            return "1h"
        elif hours <= 4:
            return "4h"
        elif hours <= 24:
            return "1d"
        else:
            return "1d"
    elif duration_str.endswith('d'):
        return "1d"
    else:
        return "1h"

def calculate_signal_score(signal):
    """Calculate a numerical score for ranking signals."""
    score = 0
    # Confidence contributes
    if signal["confidence"] == "High":
        score += 50
    elif signal["confidence"] == "Medium":
        score += 30
    
    # Risk-reward ratio contributes
    entry = signal["entry"]
    sl = signal["stop_loss"]
    tp = signal["take_profit"]
    
    if signal["signal"] == "LONG":
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp
    
    if risk > 0:
        rr_ratio = reward / risk
        score += min(30, rr_ratio * 10)  # Cap at 30 points
    
    # Confluence count contributes
    rationale = signal.get("rationale", "")
    confluence_count = rationale.count("|") + 1
    score += confluence_count * 5
    
    return score

def generate_signal(asset, asset_type, trade_type, custom_duration=None):
    """
    Main signal generation logic.
    Returns a dict with signal details or None if no trade.
    """
    # Determine timeframe
    if custom_duration:
        timeframe = map_duration_to_timeframe(custom_duration)
    else:
        timeframe = TRADE_TIMEFRAMES.get(trade_type, "1h")
    
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

    # Trend identification
    ema21 = last.get("trend_ema_fast", last.get("EMA_21", price))
    ema50 = last.get("EMA_50", price)
    macd = last.get("MACD_macd", 0)
    macd_signal = last.get("MACD_signal", 0)
    rsi = last.get("momentum_rsi", 50)
    bb_high = last.get("volatility_bbh", price * 1.02)
    bb_low = last.get("volatility_bbl", price * 0.98)

    # Determine trend
    if price > ema21 and ema21 > ema50:
        trend = "bullish"
    elif price < ema21 and ema21 < ema50:
        trend = "bearish"
    else:
        trend = "neutral"

    # Confluence score
    confluence = 0
    reasons = []

    if trend == "bullish":
        confluence += 1
        reasons.append(f"Price above EMA21/50 ({trend} trend)")
    elif trend == "bearish":
        confluence += 1
        reasons.append(f"Price below EMA21/50 ({trend} trend)")

    if price <= bb_low and last["close"] > bb_low:
        confluence += 1
        reasons.append("Price bounced off lower Bollinger Band")
    elif price >= bb_high and last["close"] < bb_high:
        confluence += 1
        reasons.append("Price rejected off upper Bollinger Band")

    if macd > macd_signal and prev.get("MACD_macd", 0) <= prev.get("MACD_signal", 0):
        confluence += 1
        reasons.append("MACD bullish crossover")
    elif macd < macd_signal and prev.get("MACD_macd", 0) >= prev.get("MACD_signal", 0):
        confluence += 1
        reasons.append("MACD bearish crossover")

    if rsi < 30 and last["close"] > prev["close"]:
        confluence += 1
        reasons.append("RSI oversold with price increase")
    elif rsi > 70 and last["close"] < prev["close"]:
        confluence += 1
        reasons.append("RSI overbought with price decrease")

    if confluence < 3:
        return None

    # Determine direction
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

    # Calculate SL/TP with ATR buffer
    buffer = atr * 1.5
    if direction == "LONG":
        sl = swing_low - buffer
        entry = price
        tp = entry + (entry - sl) * 1.5
    else:
        sl = swing_high + buffer
        entry = price
        tp = entry - (sl - entry) * 1.5

    # Dynamic leverage
    volatility_pct = (atr / price) * 100
    leverage = calculate_dynamic_leverage(volatility_pct, asset_type)

    # Duration
    if custom_duration:
        # Extract minutes from custom duration string
        if custom_duration.endswith('m'):
            minutes = int(custom_duration[:-1])
        elif custom_duration.endswith('h'):
            minutes = int(custom_duration[:-1]) * 60
        elif custom_duration.endswith('d'):
            minutes = int(custom_duration[:-1]) * 1440
        else:
            minutes = 60
        duration = calculate_duration(asset_type, trade_type, atr, price, minutes)
    else:
        duration = calculate_duration(asset_type, trade_type, atr, price)

    signal = {
        "asset": asset,
        "signal": direction,
        "entry": round(entry, 4),
        "stop_loss": round(sl, 4),
        "take_profit": round(tp, 4),
        "leverage": leverage,
        "duration": duration,
        "timeframe": timeframe,
        "confidence": "High" if confluence >= 4 else "Medium",
        "rationale": " | ".join(reasons[:3]),
        "score": 0  # Will be calculated later
    }
    
    # Calculate and add score
    signal["score"] = calculate_signal_score(signal)
    return signal

def get_top_trades(asset_type, trade_type, count=3, custom_duration=None):
    """Scan all assets and return top N best signals (excluding repeats)."""
    assets = CRYPTO_ASSETS if asset_type == "crypto" else FOREX_ASSETS
    signals = []
    
    for asset in assets:
        # Skip recently recommended assets for variety
        if asset in last_assets:
            continue
        signal = generate_signal(asset, asset_type, trade_type, custom_duration)
        if signal:
            signals.append(signal)
    
    # Sort by score descending and take top 'count'
    signals.sort(key=lambda x: x["score"], reverse=True)
    top_signals = signals[:count]
    
    # Add to last assets to avoid repeats in future
    for signal in top_signals:
        last_assets.append(signal["asset"])
    
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
    custom_duration = data.get("custom_duration")
    
    signals = get_top_trades(asset_type, trade_type, count=3, custom_duration=custom_duration)
    
    if signals:
        return jsonify({"success": True, "signals": signals})
    else:
        return jsonify({"success": False, "message": "No high-probability trades found at this moment."})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
