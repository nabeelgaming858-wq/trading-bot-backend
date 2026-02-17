from flask import Flask, jsonify, send_from_directory
import requests
import random
import time

app = Flask(__name__, static_folder="static", template_folder="templates")


# =========================
# HEALTH CHECK (VERY IMPORTANT FOR CLOUD RUN)
# =========================
@app.route("/health")
def health():
    return "OK", 200


# =========================
# HOMEPAGE (FAST LOAD — NO CALCULATIONS)
# =========================
@app.route("/")
def home():
    return send_from_directory("templates", "index.html")


# =========================
# SIMPLE PRICE FETCH (FAST API ONLY)
# =========================
CRYPTO_LIST = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT",
    "MATICUSDT","DOTUSDT","TRXUSDT","LTCUSDT","LINKUSDT","ATOMUSDT","ETCUSDT",
    "XLMUSDT","FILUSDT","APTUSDT","NEARUSDT","ARBUSDT","OPUSDT","SUIUSDT",
    "PEPEUSDT","SHIBUSDT","AAVEUSDT"
]

def get_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        r = requests.get(url, timeout=3)
        return float(r.json()["price"])
    except:
        return None


# =========================
# TRADE GENERATOR (ONLY RUNS WHEN BUTTON CLICKED)
# =========================
@app.route("/api/scan")
def scan():
    results = []

    selected = random.sample(CRYPTO_LIST, 5)

    for sym in selected:
        price = get_price(sym)
        if not price:
            continue

        direction = random.choice(["BUY", "SELL"])

        volatility = random.uniform(0.3, 1.2)

        tp_percent = round(volatility * 1.4, 2)
        sl_percent = round(volatility * 0.8, 2)

        if direction == "BUY":
            tp = price * (1 + tp_percent/100)
            sl = price * (1 - sl_percent/100)
        else:
            tp = price * (1 - tp_percent/100)
            sl = price * (1 + sl_percent/100)

        trade = {
            "asset": sym,
            "market": "Crypto",
            "price": round(price, 4),
            "direction": direction,
            "entry": round(price, 4),
            "take_profit": round(tp, 4),
            "tp_percent": tp_percent,
            "stop_loss": round(sl, 4),
            "sl_percent": sl_percent,
            "leverage": random.choice([5,10,15,20]),
            "confidence": random.randint(82, 96),
            "duration": random.choice(["5m","15m","30m","1h","4h"])
        }

        results.append(trade)

    return jsonify({
        "status": "success",
        "results": results,
        "server_time": int(time.time())
    })


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
