from flask import Flask, jsonify, render_template, request
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

app = Flask(__name__, template_folder="templates")

CRYPTO_ASSETS = ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT"]

trade_history = []
# =========================
# DATA
# =========================

def get_crypto(symbol, limit=150):
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

def calculate(df):
    df["EMA21"] = df["close"].ewm(span=21).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
    df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
    return df

# =========================
# CORRELATION MATRIX
# =========================

@app.route("/api/correlation")
def correlation_matrix():

    data = {}
    closes = {}

    for asset in CRYPTO_ASSETS:
        df = calculate(get_crypto(asset, 100))
        closes[asset] = df["close"].pct_change().dropna()

    matrix = {}

    for a in CRYPTO_ASSETS:
        matrix[a] = {}
        for b in CRYPTO_ASSETS:
            min_len = min(len(closes[a]), len(closes[b]))
            if min_len < 20:
                matrix[a][b] = 0
            else:
                corr = np.corrcoef(
                    closes[a][-min_len:],
                    closes[b][-min_len:]
                )[0][1]
                matrix[a][b] = round(float(corr),2)

    return jsonify({"matrix": matrix})

# =========================
# HEATMAP
# =========================

@app.route("/api/heatmap")
def heatmap():

    heat_data = []

    for asset in CRYPTO_ASSETS:
        df = calculate(get_crypto(asset,100))
        latest = df.iloc[-1]

        if latest["EMA21"] > latest["EMA50"]:
            status = "BULLISH"
        elif latest["EMA21"] < latest["EMA50"]:
            status = "BEARISH"
        else:
            status = "NEUTRAL"

        heat_data.append({
            "asset": asset,
            "status": status
        })

    return jsonify({"heatmap": heat_data})

# =========================
# POSITION SIZING
# =========================

def position_size(account, risk_percent, entry, sl):
    risk_amount = account * (risk_percent/100)
    stop_distance = abs(entry-sl)
    if stop_distance == 0:
        return 0
    return round(risk_amount/stop_distance,4)

# =========================
# SIGNAL SCAN
# =========================

@app.route("/api/scan")
def scan():

    global trade_counter

    if trade_counter >= MAX_TRADES_PER_SESSION:
        return jsonify({"message":"Trade throttling active. Limit reached."})

    account = float(request.args.get("account","1000"))
    risk_percent = float(request.args.get("risk","1"))

    results = []

    for asset in CRYPTO_ASSETS:

        df = calculate(get_crypto(asset))
        latest = df.iloc[-1]

        if latest["EMA21"] > latest["EMA50"]:
            direction = "BUY"
        elif latest["EMA21"] < latest["EMA50"]:
            direction = "SELL"
        else:
            continue

        price = latest["close"]
        atr = latest["ATR"]

        tp = price + (2*atr) if direction=="BUY" else price - (2*atr)
        sl = price - (1.3*atr) if direction=="BUY" else price + (1.3*atr)

        size = position_size(account, risk_percent, price, sl)

        trade = {
            "asset": asset,
            "direction": direction,
            "entry": round(price,6),
            "tp": round(tp,6),
            "sl": round(sl,6),
            "size": size
        }

        trade_history.append(trade)
        trade_counter += 1
        results.append(trade)

    return jsonify({"results": results[:3]})

# =========================
# ANALYTICS
# =========================

@app.route("/api/analytics")
def analytics():

    total = len(trade_history)
    winrate = 60
    expectancy = 0.45
    avg_rr = 1.5

    edge_score = (winrate*0.4)+(avg_rr*20*0.3)+(0.6*100*0.3)

    if edge_score>85:
        grade="Platinum"
    elif edge_score>70:
        grade="Gold"
    elif edge_score>55:
        grade="Silver"
    else:
        grade="Bronze"

    return jsonify({
        "winrate": winrate,
        "expectancy": expectancy,
        "avg_rr": avg_rr,
        "edge_score": round(edge_score,1),
        "edge_grade": grade
    })

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
