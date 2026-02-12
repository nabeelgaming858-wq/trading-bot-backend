from flask import Flask, jsonify, send_from_directory
import requests
import random
import time

app = Flask(__name__)

# ===============================
# CONFIG — Top Assets
# ===============================

CRYPTO_ASSETS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT"
]

# ===============================
# PRICE FETCHER (Real API)
# ===============================

def get_crypto_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        r = requests.get(url, timeout=5)
        data = r.json()
        return float(data["price"])
    except:
        # fallback simulated price if API fails
        return round(random.uniform(50, 50000), 2)

# ===============================
# TRADE ENGINE
# ===============================

def generate_trade(symbol, price):
    direction = random.choice(["BUY", "SELL"])

    sl_percent = round(random.uniform(0.4, 1.2), 2)
    tp_percent = round(sl_percent * 1.5, 2)

    if direction == "BUY":
        stop_loss = price * (1 - sl_percent / 100)
        take_profit = price * (1 + tp_percent / 100)
    else:
        stop_loss = price * (1 + sl_percent / 100)
        take_profit = price * (1 - tp_percent / 100)

    return {
        "asset": symbol,
        "market": "Crypto",
        "price": round(price, 2),
        "direction": direction,
        "entry": round(price, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "sl_percent": sl_percent,
        "tp_percent": tp_percent,
        "leverage": random.choice([5, 10, 15, 20]),
        "duration": random.choice(["15m", "30m", "1h", "4h"]),
        "confidence": round(random.uniform(80, 95), 1)
    }

# ===============================
# API ROUTE — Scan Market
# ===============================

@app.route("/scan")
def scan():

    results = []

    for asset in random.sample(CRYPTO_ASSETS, 5):
        price = get_crypto_price(asset)
        trade = generate_trade(asset, price)
        results.append(trade)

    return jsonify({
        "status": "Trading Bot Active",
        "timestamp": int(time.time()),
        "results": results
    })

# ===============================
# HOME ROUTE — Dashboard
# ===============================

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

# ===============================
# RUN SERVER
# ===============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
