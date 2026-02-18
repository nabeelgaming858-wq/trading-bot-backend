from flask import Flask, jsonify, render_template, request
import requests
import random
import time

app = Flask(__name__, template_folder="templates")

# ==============================
# CONFIG
# ==============================

CRYPTO_ASSETS = [
"BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT",
"SOLUSDT","DOGEUSDT","DOTUSDT","MATICUSDT","AVAXUSDT",
"LTCUSDT","TRXUSDT","LINKUSDT","ATOMUSDT","ETCUSDT",
"XLMUSDT","ICPUSDT","FILUSDT","APTUSDT","ARBUSDT",
"OPUSDT","NEARUSDT","SANDUSDT","AAVEUSDT","EOSUSDT"
]

FOREX_PAIRS = [
"EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD",
"USD/CHF","NZD/USD","EUR/GBP","EUR/JPY","GBP/JPY",
"AUD/JPY","EUR/AUD","GBP/AUD","EUR/CAD","GBP/CAD"
]

# ==============================
# PRICE FETCHERS
# ==============================

def get_crypto_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        r = requests.get(url, timeout=5)
        return float(r.json()["price"])
    except:
        return None

def get_forex_price(pair):
    base, quote = pair.split("/")
    try:
        url = f"https://api.exchangerate.host/latest?base={base}&symbols={quote}"
        r = requests.get(url, timeout=5)
        return float(r.json()["rates"][quote])
    except:
        return None

# ==============================
# SIGNAL ENGINE (LIGHTWEIGHT)
# ==============================

def generate_trade(asset, market, price):

    direction = random.choice(["BUY","SELL"])

    volatility = random.uniform(0.4,1.2)

    tp_percent = round(volatility * 1.5,2)
    sl_percent = round(volatility * 0.8,2)

    if direction == "BUY":
        take_profit = price * (1 + tp_percent/100)
        stop_loss = price * (1 - sl_percent/100)
    else:
        take_profit = price * (1 - tp_percent/100)
        stop_loss = price * (1 + sl_percent/100)

    return {
        "asset": asset,
        "market": market,
        "direction": direction,
        "entry": round(price,4),
        "take_profit": round(take_profit,4),
        "tp_percent": tp_percent,
        "stop_loss": round(stop_loss,4),
        "sl_percent": sl_percent,
        "leverage": random.choice([5,10,15,20]),
        "confidence": random.randint(82,95),
        "duration": random.choice(["5m","15m","30m","1h","4h"])
    }

# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/scan")
def scan():

    market = request.args.get("market","crypto").lower()

    results = []

    if market == "crypto":
        selected = random.sample(CRYPTO_ASSETS,5)
        for asset in selected:
            price = get_crypto_price(asset)
            if price:
                results.append(generate_trade(asset,"Crypto",price))

    elif market == "forex":
        selected = random.sample(FOREX_PAIRS,5)
        for pair in selected:
            price = get_forex_price(pair)
            if price:
                results.append(generate_trade(pair,"Forex",price))

    return jsonify({
        "timestamp": int(time.time()),
        "results": results
    })

# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
