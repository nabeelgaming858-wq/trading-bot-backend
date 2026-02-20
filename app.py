from flask import Flask, jsonify, render_template, request
import requests
import pandas as pd
import numpy as np
import time
import uuid

app = Flask(__name__, template_folder="templates")

CRYPTO_ASSETS = [
"BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT",
"SOLUSDT","DOGEUSDT","DOTUSDT","MATICUSDT","AVAXUSDT",
"LTCUSDT","TRXUSDT","LINKUSDT","ATOMUSDT","ETCUSDT"
]

TRADE_LOG = []

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

def get_ohlc(symbol):
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

# =========================
# GENERATE TRADES
# =========================

@app.route("/api/scan")
def scan():

    duration_minutes = int(request.args.get("minutes","60"))

    results = []

    for asset in CRYPTO_ASSETS[:8]:

        try:
            df = calculate_indicators(get_ohlc(asset))
            latest = df.iloc[-1]

            if latest["EMA21"] > latest["EMA50"] and latest["RSI"] > 55:

                price = latest["close"]
                atr = latest["ATR"]

                tp = price + (2 * atr)
                sl = price - (1.5 * atr)

                trade = {
                    "id": str(uuid.uuid4()),
                    "asset": asset,
                    "direction": "BUY",
                    "entry": price,
                    "tp": tp,
                    "sl": sl,
                    "status": "OPEN",
                    "created_at": time.time(),
                    "duration_minutes": duration_minutes,
                    "expires_at": time.time() + (duration_minutes * 60)
                }

                TRADE_LOG.append(trade)
                results.append(trade)

        except:
            continue

    return jsonify({"results": results})

# =========================
# MONITOR TRADES
# =========================

@app.route("/api/monitor")
def monitor():

    now = time.time()

    for trade in TRADE_LOG:

        if trade["status"] != "OPEN":
            continue

        # Expiration check
        if now > trade["expires_at"]:
            trade["status"] = "EXPIRED"
            continue

        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={trade['asset']}"
            r = requests.get(url, timeout=5)
            current_price = float(r.json()["price"])

            if current_price >= trade["tp"]:
                trade["status"] = "TP HIT"

            elif current_price <= trade["sl"]:
                trade["status"] = "SL HIT"

        except:
            continue

    total = len(TRADE_LOG)
    tp_hits = len([t for t in TRADE_LOG if t["status"]=="TP HIT"])
    sl_hits = len([t for t in TRADE_LOG if t["status"]=="SL HIT"])
    expired = len([t for t in TRADE_LOG if t["status"]=="EXPIRED"])
    open_trades = len([t for t in TRADE_LOG if t["status"]=="OPEN"])

    completed = tp_hits + sl_hits
    winrate = (tp_hits / completed * 100) if completed > 0 else 0

    return jsonify({
        "total_trades": total,
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "expired": expired,
        "open_trades": open_trades,
        "win_rate": round(winrate,1),
        "trades": TRADE_LOG
    })

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
