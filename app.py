from flask import Flask, jsonify, render_template, request
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

app = Flask(__name__, template_folder="templates")

CRYPTO_ASSETS = ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT"]

trade_history = []
adaptive_multiplier = 1.0
analytics_cache = {"data": None, "timestamp": 0}

# =========================
# INDICATORS
# =========================

def calculate(df):
    df["EMA21"] = df["close"].ewm(span=21).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100/(1+rs))

    df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
    return df

def get_crypto(symbol, limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit={limit}"
    r = requests.get(url, timeout=5)
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "t","open","high","low","close","volume",
        "ct","q","n","tb","tq","ig"
    ])

    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    return df

# =========================
# VOLATILITY REGIME
# =========================

def volatility_regime(df):
    atr = df["ATR"].iloc[-1]
    high = np.percentile(df["ATR"].dropna(), 70)
    low = np.percentile(df["ATR"].dropna(), 30)

    if atr > high:
        return "HIGH"
    elif atr < low:
        return "LOW"
    return "NORMAL"

# =========================
# PROBABILITY MODEL
# =========================

def probability_model(df):
    ema_gap = abs(df["EMA21"].iloc[-1] - df["EMA50"].iloc[-1])
    trend_strength = min(ema_gap * 100, 1)

    rsi_strength = abs(df["RSI"].iloc[-1] - 50) / 50
    volatility_factor = df["ATR"].iloc[-1] / df["ATR"].mean()

    base_winrate = 0.55

    probability = (
        (base_winrate * 0.6) +
        (trend_strength * 0.2) +
        (volatility_factor * 0.2)
    )

    probability = min(probability, 0.95)
    return round(probability * 100, 1)

# =========================
# EDGE MODEL
# =========================

def edge_model(winrate, avg_rr):

    expectancy = (winrate/100 * avg_rr) - ((1 - winrate/100) * 1)

    stability = min(winrate/100, 1)

    edge_score = (
        (winrate * 0.4) +
        (avg_rr * 15 * 0.3) +
        (stability * 100 * 0.3)
    )

    edge_score = min(edge_score, 100)

    if edge_score > 85:
        grade = "Platinum"
    elif edge_score > 70:
        grade = "Gold"
    elif edge_score > 55:
        grade = "Silver"
    else:
        grade = "Bronze"

    return round(edge_score,1), grade, round(expectancy,3)

# =========================
# SCAN
# =========================

@app.route("/api/scan")
def scan():

    global adaptive_multiplier

    duration = int(request.args.get("duration","30"))
    results = []

    for asset in CRYPTO_ASSETS:

        df = calculate(get_crypto(asset))
        latest = df.iloc[-1]

        direction = None
        score = 0

        if latest["EMA21"] > latest["EMA50"]:
            direction = "BUY"
            score += 2
        elif latest["EMA21"] < latest["EMA50"]:
            direction = "SELL"
            score += 2

        if direction == "BUY" and latest["RSI"] > 55:
            score += 2
        if direction == "SELL" and latest["RSI"] < 45:
            score += 2

        if score < 3:
            continue

        regime = volatility_regime(df)
        price = latest["close"]
        atr = latest["ATR"]

        duration_factor = max(duration/30,1)

        sl_mult = 1.3 * adaptive_multiplier
        tp_mult = 2.2 * adaptive_multiplier

        if direction == "BUY":
            tp = price + (tp_mult * atr * duration_factor)
            sl = price - (sl_mult * atr * duration_factor)
        else:
            tp = price - (tp_mult * atr * duration_factor)
            sl = price + (sl_mult * atr * duration_factor)

        probability = probability_model(df)

        trade = {
            "asset": asset,
            "direction": direction,
            "entry": round(price,6),
            "tp": round(tp,6),
            "sl": round(sl,6),
            "probability": probability,
            "regime": regime
        }

        trade_history.append(trade)
        if len(trade_history) > 20:
            trade_history.pop(0)

        results.append(trade)

    return jsonify({"results": results[:3]})

# =========================
# ANALYTICS DASHBOARD
# =========================

@app.route("/api/analytics")
def analytics():

    global analytics_cache

    if time.time() - analytics_cache["timestamp"] < 300:
        return jsonify(analytics_cache["data"])

    total = 0
    wins = 0
    rr_list = []

    for asset in CRYPTO_ASSETS:

        df = calculate(get_crypto(asset, limit=150))

        for i in range(50, len(df)-5):

            row = df.iloc[i]
            future = df.iloc[i+1:i+6]

            if row["EMA21"] > row["EMA50"] and row["RSI"] > 55:

                entry = row["close"]
                tp = entry + (2 * row["ATR"])
                sl = entry - (1.5 * row["ATR"])

                total += 1

                if any(future["high"] >= tp):
                    wins += 1
                    rr_list.append(2/1.5)
                elif any(future["low"] <= sl):
                    rr_list.append(-1)

    winrate = (wins/total*100) if total>0 else 0
    avg_rr = np.mean(rr_list) if rr_list else 0

    edge_score, grade, expectancy = edge_model(winrate, avg_rr)

    data = {
        "winrate": round(winrate,1),
        "avg_rr": round(avg_rr,2),
        "edge_score": edge_score,
        "edge_grade": grade,
        "expectancy": expectancy
    }

    analytics_cache = {
        "data": data,
        "timestamp": time.time()
    }

    return jsonify(data)

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
