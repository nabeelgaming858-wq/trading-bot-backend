from flask import Flask, jsonify, render_template, request
import requests
import pandas as pd
import numpy as np
import time

app = Flask(__name__, template_folder="templates")

CRYPTO_ASSETS = [
"BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT",
"SOLUSDT","DOGEUSDT","DOTUSDT","MATICUSDT","AVAXUSDT",
"LTCUSDT","TRXUSDT","LINKUSDT","ATOMUSDT","ETCUSDT",
"XLMUSDT","ICPUSDT","FILUSDT","APTUSDT","ARBUSDT",
"OPUSDT","NEARUSDT","SANDUSDT","AAVEUSDT","EOSUSDT"
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
# TIMEFRAME LOGIC
# =========================

def timeframe_map(type, minutes):

    if type == "scalp":
        return "5m","15m"
    if type == "intraday":
        return "15m","1h"
    if type == "swing":
        return "1h","4h"
    if type == "custom":
        if minutes <= 15:
            return "5m","15m"
        if minutes <= 60:
            return "15m","1h"
        if minutes <= 240:
            return "1h","4h"
        return "4h","1d"

# =========================
# DATA FETCH
# =========================

def get_ohlc(symbol, interval):
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

def evaluate(df):

    latest = df.iloc[-1]
    score = 0
    direction = None

    if latest["EMA21"] > latest["EMA50"]:
        score += 2
        direction = "BUY"
    elif latest["EMA21"] < latest["EMA50"]:
        score += 2
        direction = "SELL"

    if direction == "BUY" and latest["RSI"] > 55:
        score += 2
    if direction == "SELL" and latest["RSI"] < 45:
        score += 2

    if latest["ATR"] > df["ATR"].mean():
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

    trade_type = request.args.get("type","scalp")
    minutes = int(request.args.get("minutes","15"))

    primary_tf, confirm_tf = timeframe_map(trade_type, minutes)

    results = []

    for asset in CRYPTO_ASSETS[:15]:

        try:
            df1 = calculate_indicators(get_ohlc(asset, primary_tf))
            df2 = calculate_indicators(get_ohlc(asset, confirm_tf))

            score1, dir1 = evaluate(df1)
            score2, dir2 = evaluate(df2)

            if dir1 and dir1 == dir2:

                combined_score = score1 + score2

                if combined_score >= 7:

                    price = df1.iloc[-1]["close"]
                    atr = df1.iloc[-1]["ATR"]

                    duration_factor = max(minutes/30,1)

                    tp_distance = atr * 2 * duration_factor
                    sl_distance = atr * 1.3 * duration_factor

                    if dir1 == "BUY":
                        tp = price + tp_distance
                        sl = price - sl_distance
                    else:
                        tp = price - tp_distance
                        sl = price + sl_distance

                    leverage = min(max(int(12/(atr/price)),3),25)

                    results.append({
                        "asset": asset,
                        "direction": dir1,
                        "entry": round(price,6),
                        "take_profit": round(tp,6),
                        "stop_loss": round(sl,6),
                        "score": combined_score,
                        "primary_tf": primary_tf,
                        "confirm_tf": confirm_tf,
                        "leverage": leverage
                    })

        except:
            continue

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:3]

    return jsonify({
        "results": results,
        "timestamp": int(time.time())
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
