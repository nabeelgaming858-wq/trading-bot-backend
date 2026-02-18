from flask import Flask, jsonify, render_template, request
import requests
import random
import pandas as pd
import numpy as np
import time

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
# SCORING ENGINE
# =========================

def score_asset(df):

    latest = df.iloc[-1]
    score = 0
    direction = None

    if latest["EMA21"] > latest["EMA50"]:
        score += 2
        direction = "BUY"
    elif latest["EMA21"] < latest["EMA50"]:
        score += 2
        direction = "SELL"

    if direction == "BUY" and latest["RSI"] > 50:
        score += 2
    if direction == "SELL" and latest["RSI"] < 50:
        score += 2

    if latest["ATR"] > df["ATR"].mean():
        score += 1

    if 30 < latest["RSI"] < 70:
        score += 1

    return score, direction

# =========================
# DATA FETCH
# =========================

def get_crypto_ohlc(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit=100"
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

def get_forex_ohlc(symbol):
    base = symbol[:3]
    quote = symbol[3:]
    url = f"https://api.twelvedata.com/time_series?symbol={base}/{quote}&interval=15min&outputsize=100&apikey=demo"
    r = requests.get(url, timeout=5)
    data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    return df.iloc[::-1]

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/scan")
def scan():

    market = request.args.get("market","crypto").lower()
    results = []

    assets = CRYPTO_ASSETS if market=="crypto" else FOREX_PAIRS

    for asset in random.sample(assets,8):

        try:
            df = get_crypto_ohlc(asset) if market=="crypto" else get_forex_ohlc(asset)
            if df is None:
                continue

            df = calculate_indicators(df)
            score, direction = score_asset(df)

            if score >= 5 and direction:

                price = df.iloc[-1]["close"]
                atr = df.iloc[-1]["ATR"]

                if direction == "BUY":
                    sl = price - (1.5 * atr)
                    tp = price + (2 * atr)
                else:
                    sl = price + (1.5 * atr)
                    tp = price - (2 * atr)

                results.append({
                    "asset": asset,
                    "market": market.capitalize(),
                    "direction": direction,
                    "entry": round(price,6),
                    "take_profit": round(tp,6),
                    "stop_loss": round(sl,6),
                    "score": score,
                    "volatility": round(atr,6)
                })

        except:
            continue

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:3]

    return jsonify({
        "timestamp": int(time.time()),
        "results": results
    })

# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
        
