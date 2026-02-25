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
# Asset lists (crypto 25, forex 15) - metals removed as requested
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
    "swing": "4h"
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
last_assets = deque(maxlen=5)

# ==================== HELPER FUNCTIONS ====================
def fetch_binance_klines(symbol, interval, limit=100):
    """Fetch OHLC data from Binance for crypto assets."""
    url = f"{BINANCE_BASE}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Binance returns: [open, high, low, close, volume, ...]
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
    Fetch OHLC for forex pairs from exchangerate.host.
    Note: exchangerate.host provides daily timeseries. For intraday, we simulate by using daily with random walk.
    In production, use a proper forex API (e.g., Twelve Data, Alpha Vantage).
    This is a simplified version for demonstration.
    """
    try:
        # Map interval to days back (approximation)
        days_map = {"15m": 2, "1h": 7, "4h": 30}
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
        # Build OHLC from daily rates (simulate intraday by interpolating)
        dates = sorted(data["rates"].keys())
        prices = [data["rates"][d][quote] for d in dates]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(dates),
            "close": prices
        })
        df.set_index("timestamp", inplace=True)
        # Generate fake OHLC from close (for demo)
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.random.uniform(0, 0.002))
        df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.random.uniform(0, 0.002))
        df["volume"] = 1e6  # dummy volume
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
    # Fetch fresh data
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
    # Required columns: open, high, low, close, volume
    df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    # Also add custom indicators if needed
    return df

def find_swing_points(df, window=5):
    """Find recent swing high and low."""
    # Simple method: use rolling max/min
    df["swing_high"] = df["high"].rolling(window=window, center=True).max()
    df["swing_low"] = df["low"].rolling(window=window, center=True).min()
    # Get last clear swing
    last_swing_high = df["swing_high"].iloc[-2] if len(df) > 2 else df["high"].iloc[-1]
    last_swing_low = df["swing_low"].iloc[-2] if len(df) > 2 else df["low"].iloc[-1]
    return last_swing_high, last_swing_low

def calculate_fib_levels(high, low):
    """Return Fibonacci retracement levels (0.236, 0.382, 0.5, 0.618, 0.786)."""
    diff = high - low
    levels = {
        0.236: high - 0.236 * diff,
        0.382: high - 0.382 * diff,
        0.5: high - 0.5 * diff,
        0.618: high - 0.618 * diff,
        0.786: high - 0.786 * diff
    }
    return levels

def calculate_dynamic_leverage(volatility_percent, asset_type):
    """
    Determine leverage based on volatility.
    Lower leverage for higher volatility.
    """
    if asset_type == "crypto":
        base_max = 10
    else:  # forex
        base_max = 50
    # volatility_percent is ATR% of price
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
    """
    Estimate duration based on ATR and trade type.
    Returns string like "2h", "1d", etc.
    """
    atr_pct = (atr / price) * 100
    if trade_type == "scalp":
        # Expect trade to complete within 1-2 hours
        base_hours = 2
    elif trade_type == "intraday":
        base_hours = 24
    else:  # swing
        base_hours = 7 * 24
    # Adjust for volatility: higher volatility -> shorter duration? Or longer? We'll keep simple.
    # For demonstration, we return a fixed string.
    if trade_type == "scalp":
        return "2h"
    elif trade_type == "intraday":
        return "24h"
    else:
        return "7d"

def generate_signal(asset, asset_type, trade_type):
    """
    Main signal generation logic.
    Returns a dict with signal details or None if no trade.
    """
    timeframe = TRADE_TIMEFRAMES[trade_type]
    df = get_ohlc(asset, asset_type, timeframe)
    if df is None or len(df) < 50:
        return None

    # Calculate indicators
    df = calculate_indicators(df)
    # Get latest values
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last["close"]
    atr = last["atr"] if "atr" in last else (df["high"].iloc[-14:].max() - df["low"].iloc[-14:].min()) / 14
    if pd.isna(atr) or atr == 0:
        atr = price * 0.01  # fallback 1%

    # Find swing high/low
    swing_high, swing_low = find_swing_points(df)

    # Trend identification: using EMA and MACD
    ema21 = last["trend_ema_fast"] if "trend_ema_fast" in last else last.get("EMA_21", price)
    ema50 = last.get("EMA_50", price)
    macd = last.get("MACD_macd", 0)
    macd_signal = last.get("MACD_signal", 0)
    rsi = last.get("momentum_rsi", 50)
    bb_high = last.get("volatility_bbh", price * 1.02)
    bb_low = last.get("volatility_bbl", price * 0.98)
    bb_mid = last.get("volatility_bbm", price)

    # Determine trend direction
    if price > ema21 and ema21 > ema50:
        trend = "bullish"
    elif price < ema21 and ema21 < ema50:
        trend = "bearish"
    else:
        trend = "neutral"

    # Confluence score (0-4)
    confluence = 0
    reasons = []

    # 1. EMA alignment
    if trend == "bullish":
        confluence += 1
        reasons.append("Price above EMA21/50 (bullish trend)")
    elif trend == "bearish":
        confluence += 1
        reasons.append("Price below EMA21/50 (bearish trend)")

    # 2. Bollinger Bands: price rejection off bands
    if price <= bb_low and last["close"] > bb_low:
        confluence += 1
        reasons.append("Price bounced off lower Bollinger Band (bullish)")
    elif price >= bb_high and last["close"] < bb_high:
        confluence += 1
        reasons.append("Price rejected off upper Bollinger Band (bearish)")

    # 3. MACD crossover
    if macd > macd_signal and prev.get("MACD_macd", 0) <= prev.get("MACD_signal", 0):
        confluence += 1
        reasons.append("MACD bullish crossover")
    elif macd < macd_signal and prev.get("MACD_macd", 0) >= prev.get("MACD_signal", 0):
        confluence += 1
        reasons.append("MACD bearish crossover")

    # 4. RSI divergence or overbought/oversold
    if rsi < 30 and last["close"] > prev["close"]:
        confluence += 1
        reasons.append("RSI oversold with price increase (bullish divergence)")
    elif rsi > 70 and last["close"] < prev["close"]:
        confluence += 1
        reasons.append("RSI overbought with price decrease (bearish divergence)")

    # Need at least 3 confluences for a signal
    if confluence < 3:
        return None

    # Determine direction based on confluence
    # Count bullish vs bearish reasons
    bullish_count = sum(1 for r in reasons if "bull" in r.lower())
    bearish_count = sum(1 for r in reasons if "bear" in r.lower())
    if bullish_count > bearish_count:
        direction = "LONG"
    elif bearish_count > bullish_count:
        direction = "SHORT"
    else:
        # Use trend as tie-breaker
        direction = "LONG" if trend == "bullish" else "SHORT" if trend == "bearish" else None
    if not direction:
        return None

    # Compute SL using ATR buffer
    buffer = atr * 1.5
    if direction == "LONG":
        sl = swing_low - buffer
        entry = price  # market entry
        tp = entry + (entry - sl) * 1.5  # 1:1.5 risk-reward
    else:
        sl = swing_high + buffer
        entry = price
        tp = entry - (sl - entry) * 1.5

    # Dynamic leverage
    volatility_pct = (atr / price) * 100
    leverage = calculate_dynamic_leverage(volatility_pct, asset_type)

    # Duration
    duration = calculate_duration(asset_type, trade_type, atr, price)

    # Build signal (news removed)
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
        "rationale": " | ".join(reasons[:3])
    }
    return signal

def get_best_trade(asset_type, trade_type):
    """Scan all assets of given type and return best signal (excluding repeats)."""
    assets = CRYPTO_ASSETS if asset_type == "crypto" else FOREX_ASSETS
    best_signal = None
    best_score = -1
    for asset in assets:
        # Skip recently recommended assets
        if asset in last_assets:
            continue
        signal = generate_signal(asset, asset_type, trade_type)
        if signal:
            # Score based on confidence and maybe risk-reward
            score = 2 if signal["confidence"] == "High" else 1
            if score > best_score:
                best_score = score
                best_signal = signal
    if best_signal:
        last_assets.append(best_signal["asset"])
    return best_signal

# ==================== FLASK ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json()
    asset_type = data.get("asset_type", "crypto")
    trade_type = data.get("trade_type", "intraday")
    signal = get_best_trade(asset_type, trade_type)
    if signal:
        return jsonify({"success": True, "signal": signal})
    else:
        return jsonify({"success": False, "message": "No high-probability trade found at this moment."})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
