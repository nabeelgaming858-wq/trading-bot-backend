import os
import time
import threading
import random
from datetime import datetime, timedelta
from collections import deque
from functools import lru_cache

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

# Default timeframes for trade types
TRADE_TIMEFRAMES = {
    "scalp": "15m",
    "intraday": "1h",
    "swing": "4h",
    "custom": "1h"  # placeholder, will be overridden
}

# Cache for OHLC data
data_cache = {}
cache_lock = threading.Lock()
CACHE_DURATION = 60  # seconds

# Keep track of last recommended assets to avoid repeats (now for top 3)
last_assets = deque(maxlen=10)

# ==================== API FUNCTIONS ====================
# Crypto APIs
def fetch_binance_klines(symbol, interval, limit=100):
    url = "https://api.binance.com/api/v3/klines"
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
        print(f"Binance error: {e}")
        return None

def fetch_coingecko_ohlc(symbol, interval, limit=100):
    # CoinGecko does not have USDT pairs directly; we map symbol to CoinGecko ID.
    # For simplicity, we'll map BTCUSDT -> bitcoin, ETHUSDT -> ethereum, etc.
    # This is a simplified mapping; in production, you'd need a proper mapping.
    # We'll use a rough approximation.
    mapping = {
        "BTCUSDT": "bitcoin",
        "ETHUSDT": "ethereum",
        "BNBUSDT": "binancecoin",
        "SOLUSDT": "solana",
        "XRPUSDT": "ripple",
        "ADAUSDT": "cardano",
        "DOGEUSDT": "dogecoin",
        "DOTUSDT": "polkadot",
        "MATICUSDT": "matic-network",
        "SHIBUSDT": "shiba-inu",
        "TRXUSDT": "tron",
        "AVAXUSDT": "avalanche-2",
        "UNIUSDT": "uniswap",
        "ATOMUSDT": "cosmos",
        "LTCUSDT": "litecoin",
        "BCHUSDT": "bitcoin-cash",
        "NEARUSDT": "near",
        "FILUSDT": "filecoin",
        "APTUSDT": "aptos",
        "TONUSDT": "the-open-network",
        "LINKUSDT": "chainlink",
        "XLMUSDT": "stellar",
        "ALGOUSDT": "algorand",
        "VETUSDT": "vechain",
        "ICPUSDT": "internet-computer"
    }
    coin_id = mapping.get(symbol)
    if not coin_id:
        return None
    # Map interval to days (CoinGecko ohlc endpoint uses days)
    days_map = {"1m": 1, "5m": 1, "15m": 2, "30m": 2, "1h": 7, "4h": 30, "1d": 90, "1w": 365}
    days = days_map.get(interval, 7)
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # data is list of [timestamp, open, high, low, close]
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['volume'] = 1e6  # dummy volume
        return df
    except Exception as e:
        print(f"CoinGecko error: {e}")
        return None

# Forex APIs
def fetch_alphavantage_forex(symbol, interval, limit=100):
    # Alpha Vantage requires API key; using public demo key (rate limited)
    api_key = "demo"  # free demo key; replace with your own if needed
    # Map interval to Alpha Vantage interval
    interval_map = {
        "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "60min", "4h": "60min", "1d": "daily", "1w": "weekly"
    }
    av_interval = interval_map.get(interval, "60min")
    function = "FX_INTRADAY" if "min" in av_interval else "FX_DAILY"
    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "from_symbol": symbol[:3],
        "to_symbol": symbol[3:],
        "interval": av_interval if "min" in av_interval else None,
        "apikey": api_key,
        "outputsize": "compact"
    }
    # Remove None params
    params = {k: v for k, v in params.items() if v is not None}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Parse response
        if "Time Series FX" in data:
            ts_key = "Time Series FX (1min)"  # adjust based on interval
            # find the correct key
            for key in data:
                if "Time Series FX" in key:
                    ts_key = key
                    break
            time_series = data[ts_key]
            records = []
            for dt_str, values in time_series.items():
                records.append({
                    'timestamp': pd.to_datetime(dt_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': 0
                })
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            return df.iloc[-limit:]  # limit
        else:
            return None
    except Exception as e:
        print(f"Alpha Vantage error: {e}")
        return None

def fetch_exchangerate_forex(symbol, interval, limit=100):
    # exchangerate.host provides daily only; we'll simulate intraday
    try:
        days_map = {"1m": 1, "5m": 1, "15m": 2, "30m": 2, "1h": 7, "4h": 30, "1d": 90, "1w": 365}
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
        # Simulate OHLC from close (for demo)
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.random.uniform(0, 0.002))
        df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.random.uniform(0, 0.002))
        df["volume"] = 1e6
        return df
    except Exception as e:
        print(f"Exchangerate error: {e}")
        return None

# Main data fetch with fallbacks
def get_ohlc(asset, asset_type, timeframe):
    cache_key = f"{asset}_{timeframe}"
    now = time.time()
    with cache_lock:
        if cache_key in data_cache and now - data_cache[cache_key]["timestamp"] < CACHE_DURATION:
            return data_cache[cache_key]["data"]

    df = None
    if asset_type == "crypto":
        # Try Binance first, then CoinGecko
        df = fetch_binance_klines(asset, timeframe)
        if df is None:
            df = fetch_coingecko_ohlc(asset, timeframe)
    else:
        # Try Alpha Vantage first, then exchangerate.host
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

def calculate_duration(asset_type, trade_type, atr, price):
    # Simple duration based on trade type
    if trade_type == "scalp":
        return "2h"
    elif trade_type == "intraday":
        return "24h"
    elif trade_type == "swing":
        return "7d"
    else:  # custom
        return "1h"  # placeholder; actual custom duration is determined by selected timeframe

def generate_signal(asset, asset_type, trade_type, custom_timeframe=None):
    """
    Generate a signal for a single asset.
    Returns a dict with signal details or None if no valid signal.
    """
    if custom_timeframe:
        timeframe = custom_timeframe
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

    # Confluence score (0-4)
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

    if rsi < 30 and last["close"] > prev["close"]:
        confluence += 1
        reasons.append("RSI oversold + price up")
    elif rsi > 70 and last["close"] < prev["close"]:
        confluence += 1
        reasons.append("RSI overbought + price down")

    if confluence < 2:  # lower threshold to ensure more signals; 2+ is acceptable
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
    duration = calculate_duration(asset_type, trade_type, atr, price)

    # Score for ranking (confluence * 10 + random tiny to break ties)
    score = confluence * 10 + random.random()

    signal = {
        "asset": asset,
        "signal": direction,
        "entry": round(entry, 4),
        "stop_loss": round(sl, 4),
        "take_profit": round(tp, 4),
        "leverage": leverage,
        "duration": duration,
        "timeframe": timeframe,
        "confidence": "High" if confluence >= 3 else "Medium" if confluence == 2 else "Low",
        "rationale": " | ".join(reasons[:2]),  # show top 2 reasons
        "score": score
    }
    return signal

def get_top_trades(asset_type, trade_type, count=3, custom_timeframe=None):
    """Return top N distinct signals for the given asset type."""
    assets = CRYPTO_ASSETS if asset_type == "crypto" else FOREX_ASSETS
    signals = []
    for asset in assets:
        if asset in last_assets:
            continue
        signal = generate_signal(asset, asset_type, trade_type, custom_timeframe)
        if signal:
            signals.append(signal)
    # Sort by score descending
    signals.sort(key=lambda x: x["score"], reverse=True)
    top_signals = signals[:count]
    # Add selected assets to last_assets to avoid repeats next time
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
    custom_timeframe = data.get("custom_timeframe")  # e.g., "1h", "4h", etc.
    if custom_timeframe == "" or custom_timeframe is None:
        custom_timeframe = None
    top_trades = get_top_trades(asset_type, trade_type, count=3, custom_timeframe=custom_timeframe)
    if top_trades:
        return jsonify({"success": True, "signals": top_trades})
    else:
        # In case no trades found (should rarely happen), return an empty list with a message
        return jsonify({"success": True, "signals": [], "message": "No trades met the criteria. Try adjusting settings."})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
