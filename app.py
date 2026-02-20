from flask import Flask, jsonify, render_template, request
import requests
import pandas as pd
import numpy as np
import time
import math

app = Flask(__name__, template_folder="templates")

# =========================
# CONFIG
# =========================

CRYPTO_ASSETS = [
"BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT",
"SOLUSDT","DOGEUSDT","DOTUSDT","MATICUSDT","AVAXUSDT",
"LTCUSDT","TRXUSDT","LINKUSDT","ATOMUSDT","ETCUSDT",
"XLMUSDT","ICPUSDT","FILUSDT","APTUSDT","ARBUSDT",
"OPUSDT","NEARUSDT","SANDUSDT","AAVEUSDT","EOSUSDT"
]

FOREX_PAIRS = [
"EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD",
"USDCHF","NZDUSD","EURGBP","EURJPY","GBPJPY",
"AUDJPY","EURAUD","GBPAUD","EURCAD","GBPCAD"
]

# =========================
# INDICATORS
# =========================

def calculate_indicators(df):

    df["EMA21"] = df["close"].ewm(span=21).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()

    return df

# =========================
# TIMEFRAME MAPPING
# =========================

def map_duration(duration_type, custom_minutes):

    if duration_type == "scalp":
        return "5m"
    if duration_type == "intraday":
        return "15m"
    if duration_type == "swing":
        return "1h"
    if duration_type == "custom":
        if custom_minutes <= 15:
            return "5m"
        if custom_minutes <= 60:
            return "15m"
        if custom_minutes <= 240:
            return "1h"
        return "4h"

# =========================
# DATA FETCH
# =========================

def get_crypto_ohlc(symbol, interval):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=100"
    r = requests.get(url, timeout=5)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tbav","tqav","ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    return df

# =========================
# SCORING
# =========================

def score_asset(df):

    latest = df.iloc[-1]
    score = 0
    direction = None

    # Trend
    if latest["EMA21"] > latest["EMA50"]:
        score += 2
        direction = "BUY"
    elif latest["EMA21"] < latest["EMA50"]:
        score += 2
        direction = "SELL"

    # Momentum
    if direction == "BUY" and latest["RSI"] > 55:
        score += 2
    if direction == "SELL" and latest["RSI"] < 45:
        score += 2

    # Volatility
    if latest["ATR"] > df["ATR"].mean():
        score += 1

    # Stability
    if 35 < latest["RSI"] < 65:
        score += 1

    return score, direction

# =========================
# ROUTE
# =========================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/scan")
def scan():

    market = request.args.get("market","crypto")
    duration_type = request.args.get("type","scalp")
    custom_minutes = int(request.args.get("minutes","15"))

    interval = map_duration(duration_type, custom_minutes)

    results = []
    assets = CRYPTO_ASSETS if market=="crypto" else CRYPTO_ASSETS

    for asset in assets[:12]:

        try:
            df = get_crypto_ohlc(asset, interval)
            df = calculate_indicators(df)

            score, direction = score_asset(df)

            if score >= 6:

                latest = df.iloc[-1]
                price = latest["close"]
                atr = latest["ATR"]

                # Duration scaling factor
                duration_factor = max(custom_minutes/30, 1)

                tp_distance = atr * 1.8 * duration_factor
                sl_distance = atr * 1.2 * duration_factor

                if direction == "BUY":
                    tp = price + tp_distance
                    sl = price - sl_distance
                else:
                    tp = price - tp_distance
                    sl = price + sl_distance

                leverage = min(max(int(10 / (atr/price)), 3), 20)

                results.append({
                    "asset": asset,
                    "direction": direction,
                    "entry": round(price,6),
                    "take_profit": round(tp,6),
                    "stop_loss": round(sl,6),
                    "score": score,
                    "leverage": leverage,
                    "interval": interval,
                    "logic": "EMA21/50 + RSI Momentum + ATR Duration Scaling"
                })

        except:
            continue

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:3]

    return jsonify({
        "results": results,
        "interval": interval,
        "duration_minutes": custom_minutes,
        "timestamp": int(time.time())
    })

# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
