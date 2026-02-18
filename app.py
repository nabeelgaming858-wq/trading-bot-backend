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
"EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD",
"USDCHF","NZDUSD","EURGBP","EURJPY","GBPJPY",
"AUDJPY","EURAUD","GBPAUD","EURCAD","GBPCAD"
]

# ==============================
# PRICE FETCHERS
# ==============================

def get_crypto_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        r = requests.get(url, timeout=5)
        data = r.json()
        return float(data["price"])
    except:
        return None

def get_forex_price(symbol):
    try:
        base = symbol[:3]
        quote = symbol[3:]
        url = f"https://api.exchangerate.host/latest?base={base}&symbols={quote}"
        r = requests.get(url, timeout=5)
        data = r.json()
        return float(data["rates"][quote])
    except:
        return None

# ==============================
# SIGNAL ENGINE
# ==============================

def generate_trade(asset, market, price):

    direction = random.choice(["BUY","SELL"])

    volatility = random.uniform(0.4, 1.2)

    tp_percent = round(volatility * 1.5, 2)
    sl_percent = round(volatility * 0.8, 2)

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
        "entry": round(price,6),
        "take_profit": round(take_profit,6),
        "tp_percent": tp_percent,
        "stop_loss": round(stop_loss,6),
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
