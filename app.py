import asyncio
import json
import threading
import requests
from flask import Flask, render_template, jsonify
from flask_sock import Sock
import websockets

app = Flask(__name__)
sock = Sock(app)

# ===============================
# CONFIG
# ===============================

CRYPTO_SYMBOLS = [
    "btcusdt","ethusdt","bnbusdt","xrpusdt","adausdt","dogeusdt",
    "solusdt","dotusdt","maticusdt","ltcusdt","linkusdt","avaxusdt",
    "trxusdt","atomusdt","nearusdt","algo_usdt","filusdt","aptusdt",
    "sandusdt","manausdt","opususdt","arbousdt","injusdt","tiausdt",
    "pepeusdt"
]

FOREX_SYMBOLS = [
    "EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD",
    "USD/CAD","NZD/USD","EUR/JPY","GBP/JPY","EUR/GBP",
    "EUR/AUD","GBP/AUD","AUD/JPY","CHF/JPY","EUR/CHF"
]

latest_prices = {}
lock = threading.Lock()

# ===============================
# BINANCE WEBSOCKET
# ===============================

async def binance_ws():
    streams = "/".join([f"{s}@miniTicker" for s in CRYPTO_SYMBOLS])
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"

    while True:
        try:
            async with websockets.connect(url) as ws:
                async for msg in ws:
                    data = json.loads(msg)
                    if "data" in data:
                        symbol = data["data"]["s"].lower()
                        price = float(data["data"]["c"])
                        with lock:
                            latest_prices[symbol] = price
        except:
            await asyncio.sleep(5)

# ===============================
# FOREX POLLING (Free API Limit)
# ===============================

def update_forex():
    while True:
        try:
            for pair in FOREX_SYMBOLS:
                url = f"https://api.twelvedata.com/price?symbol={pair}&apikey=demo"
                r = requests.get(url, timeout=5)
                data = r.json()
                if "price" in data:
                    with lock:
                        latest_prices[pair] = float(data["price"])
        except:
            pass
        asyncio.sleep(10)

# ===============================
# BACKGROUND START
# ===============================

def start_background():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(binance_ws())

threading.Thread(target=start_background, daemon=True).start()

# ===============================
# ROUTES
# ===============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prices")
def prices():
    with lock:
        return jsonify(latest_prices)

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
