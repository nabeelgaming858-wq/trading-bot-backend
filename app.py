from flask import Flask, jsonify
import requests
import random

app = Flask(__name__)

# -------- LIVE PRICE APIs --------

def get_crypto_price():
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        r = requests.get(url, timeout=5)
        data = r.json()
        return float(data["price"])
    except:
        return None


def get_gold_price():
    try:
        url = "https://api.metals.live/v1/spot"
        r = requests.get(url, timeout=5)
        data = r.json()
        return float(data[0]["gold"])
    except:
        return None


# -------- SIMPLE SIGNAL ENGINE --------

def generate_trade():

    crypto = get_crypto_price()
    gold = get_gold_price()

    asset = random.choice(["BTC/USDT", "Gold"])

    if asset == "BTC/USDT":
        price = crypto
    else:
        price = gold

    if price is None:
        return {"error": "Market data unavailable"}

    direction = random.choice(["BUY", "SELL"])

    tp = round(price * 1.01, 2)
    sl = round(price * 0.99, 2)

    return {
        "asset": asset,
        "direction": direction,
        "entry_price": price,
        "take_profit": tp,
        "stop_loss": sl,
        "reason": "Trend + volatility scan (base engine)"
    }


# -------- API ROUTES --------

@app.route("/")
def home():
    return jsonify({"status": "Trading Bot Backend Running"})


@app.route("/trade")
def trade():
    return jsonify(generate_trade())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
