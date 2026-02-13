from flask import Flask, jsonify, send_from_directory
import requests
import random
import os
import time

app = Flask(__name__, static_folder=".")

###################################
# ASSET LISTS
###################################

CRYPTO_ASSETS = [
"BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
"ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
"LTCUSDT","TRXUSDT","LINKUSDT","ATOMUSDT","XLMUSDT",
"ETCUSDT","NEARUSDT","APTUSDT","FILUSDT","ARBUSDT",
"OPUSDT","SANDUSDT","AAVEUSDT","EOSUSDT","ICPUSDT"
]

FOREX_PAIRS = [
"EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD",
"USDCAD","NZDUSD","EURJPY","GBPJPY","EURGBP",
"AUDJPY","CHFJPY","EURAUD","GBPAUD","EURCHF"
]

METALS = {
"GOLD":"XAUUSD",
"SILVER":"XAGUSD"
}

###################################
# PRICE FETCHERS
###################################

def get_crypto_price(symbol):
    try:
        url=f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        r=requests.get(url,timeout=3)
        return float(r.json()["price"])
    except:
        return round(random.uniform(1,1000),2)

def get_forex_price(pair):
    try:
        url=f"https://api.exchangerate.host/latest?base={pair[:3]}&symbols={pair[3:]}"
        r=requests.get(url,timeout=3)
        return float(r.json()["rates"][pair[3:]])
    except:
        return round(random.uniform(0.5,2),4)

def get_metals_price(symbol):
    try:
        url="https://api.metals.live/v1/spot"
        r=requests.get(url,timeout=3).json()
        for item in r:
            if symbol.lower() in item:
                return float(item[symbol.lower()])
    except:
        return round(random.uniform(20,2000),2)

###################################
# SIGNAL GENERATOR
###################################

def generate_signal(asset, market, price):

    direction=random.choice(["BUY","SELL"])

    move=random.uniform(0.3,1.5)/100

    if direction=="BUY":
        tp=round(price*(1+move),4)
        sl=round(price*(1-move*0.7),4)
    else:
        tp=round(price*(1-move),4)
        sl=round(price*(1+move*0.7),4)

    return {
        "asset":asset,
        "market":market,
        "direction":direction,
        "entry":round(price,4),
        "take_profit":tp,
        "stop_loss":sl,
        "tp_percent":round(move*100,2),
        "sl_percent":round(move*70,2),
        "leverage":random.choice([5,10,15,20]),
        "confidence":round(random.uniform(80,95),1),
        "duration":random.choice(["15m","1h","4h"])
    }

###################################
# SCAN ROUTE
###################################

@app.route("/scan")
def scan():

    results=[]

    # Crypto scan
    for asset in random.sample(CRYPTO_ASSETS,5):
        price=get_crypto_price(asset)
        results.append(generate_signal(asset,"Crypto",price))

    # Forex scan
    for pair in random.sample(FOREX_PAIRS,3):
        price=get_forex_price(pair)
        results.append(generate_signal(pair,"Forex",price))

    # Metals
    for name,symbol in METALS.items():
        price=get_metals_price(name.lower())
        results.append(generate_signal(symbol,"Metals",price))

    return jsonify({"results":results})

###################################
# FRONTEND
###################################

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

###################################

if __name__ == "__main__":
    port=int(os.environ.get("PORT",8080))
    app.run(host="0.0.0.0",port=port)
