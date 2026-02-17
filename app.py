from flask import Flask, jsonify, request, render_template
import requests
import random
import time

app = Flask(__name__)

# ===============================
# ASSET LISTS
# ===============================

CRYPTO_ASSETS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT",
    "SOLUSDT","DOGEUSDT","DOTUSDT","MATICUSDT","AVAXUSDT",
    "LTCUSDT","TRXUSDT","LINKUSDT","ATOMUSDT","ETCUSDT",
    "XLMUSDT","ICPUSDT","FILUSDT","APTUSDT","ARBUSDT",
    "OPUSDT","NEARUSDT","VETUSDT","SANDUSDT","AAVEUSDT"
]

FOREX_PAIRS = [
    "EUR/USD","GBP/USD","USD/JPY","AUD/USD","USD/CAD",
    "USD/CHF","NZD/USD","EUR/GBP","EUR/JPY","GBP/JPY",
    "AUD/JPY","EUR/AUD","GBP/AUD","EUR/CAD","GBP/CAD"
]

# ===============================
# CRYPTO PRICE ENGINE
# ===============================

def get_crypto_price(symbol):

    # Binance primary
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        r = requests.get(url, timeout=3)
        data = r.json()
        return float(data["price"])
    except:
        pass

    # CoinGecko fallback
    try:
        coin = symbol.replace("USDT","").lower()
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
        r = requests.get(url, timeout=3)
        data = r.json()
        return float(data[coin]["usd"])
    except:
        return None

# ===============================
# FOREX PRICE ENGINE
# ===============================

def get_forex_price(pair):

    base, quote = pair.split("/")

    # TwelveData primary
    try:
        url = f"https://api.twelvedata.com/price?symbol={base}/{quote}&apikey=demo"
        r = requests.get(url, timeout=3)
        data = r.json()
        return float(data["price"])
    except:
        pass

    # Alpha Vantage fallback
    try:
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={base}&to_currency={quote}&apikey=demo"
        r = requests.get(url, timeout=3)
        data = r.json()
        rate = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
        return float(rate)
    except:
        return None

# ===============================
# SIMPLE ANALYSIS ENGINE
# ===============================

def generate_trade(asset, market, price):

    direction = random.choice(["BUY","SELL"])

    sl_percent = random.uniform(0.5,1.2)
    tp_percent = random.uniform(1.0,2.0)

    if direction == "BUY":
        stop_loss = price * (1 - sl_percent/100)
        take_profit = price * (1 + tp_percent/100)
    else:
        stop_loss = price * (1 + sl_percent/100)
        take_profit = price * (1 - tp_percent/100)

    confidence = round(random.uniform(82,95),1)
    leverage = random.choice([5,10,15,20])
    duration = random.choice(["15m","30m","1h","4h"])

    return {
        "asset": asset,
        "market": market,
        "direction": direction,
        "entry": round(price,4),
        "take_profit": round(take_profit,4),
        "tp_percent": round(tp_percent,2),
        "stop_loss": round(stop_loss,4),
        "sl_percent": round(sl_percent,2),
        "leverage": leverage,
        "confidence": confidence,
        "duration": duration
    }

# ===============================
# API ROUTE
# ===============================

@app.route("/scan")
def scan():

    market = request.args.get("market","crypto").lower()

    results = []

    if market == "crypto":
        assets = random.sample(CRYPTO_ASSETS, 5)

        for a in assets:
            price = get_crypto_price(a)
            if price:
                results.append(generate_trade(a,"Crypto",price))

    if market == "forex":
        pairs = random.sample(FOREX_PAIRS, 5)

        for p in pairs:
            price = get_forex_price(p)
            if price:
                results.append(generate_trade(p,"Forex",price))

    return jsonify({"results": results})

# ===============================
# MAIN PAGE
# ===============================

@app.route("/")
def home():
    return render_template("index.html")

# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
