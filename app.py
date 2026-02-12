from flask import Flask, jsonify
import requests
import random
import time

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------

CRYPTO_ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "MATICUSDT", "LTCUSDT", "TRXUSDT", "UNIUSDT", "ATOMUSDT",
    "ETCUSDT", "FILUSDT", "APTUSDT", "NEARUSDT", "ARBUSDT",
    "OPUSDT", "SUIUSDT", "INJUSDT", "XLMUSDT", "ICPUSDT"
]

FOREX_METALS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "XAUUSD", "XAGUSD"
]

# -----------------------------
# API FUNCTIONS
# -----------------------------

def get_crypto_price(symbol):
    """Primary: Binance"""
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        r = requests.get(url, timeout=3)
        data = r.json()
        return float(data["price"])
    except:
        return None


def get_crypto_price_backup(symbol):
    """Backup: CoinGecko"""
    try:
        mapping = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "SOLUSDT": "solana",
            "BNBUSDT": "binancecoin"
        }

        coin = mapping.get(symbol)
        if not coin:
            return None

        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
        r = requests.get(url, timeout=3)
        data = r.json()
        return float(data[coin]["usd"])
    except:
        return None


def get_forex_price(pair):
    """Forex/Metals via exchangerate.host"""
    try:
        base = pair[:3]
        quote = pair[3:]

        url = f"https://api.exchangerate.host/latest?base={base}&symbols={quote}"
        r = requests.get(url, timeout=3)
        data = r.json()

        return float(data["rates"][quote])
    except:
        return None


# -----------------------------
# SIMPLE ANALYSIS ENGINE
# -----------------------------

def generate_signal(price):
    """Basic placeholder analysis (will upgrade later)"""

    direction = random.choice(["BUY", "SELL"])

    tp_percent = random.uniform(0.5, 1.5)
    sl_percent = random.uniform(0.3, 0.8)

    if direction == "BUY":
        tp = price * (1 + tp_percent / 100)
        sl = price * (1 - sl_percent / 100)
    else:
        tp = price * (1 - tp_percent / 100)
        sl = price * (1 + sl_percent / 100)

    return {
        "direction": direction,
        "entry": round(price, 4),
        "take_profit": round(tp, 4),
        "stop_loss": round(sl, 4),
        "tp_percent": round(tp_percent, 2),
        "sl_percent": round(sl_percent, 2),
        "leverage": random.choice([5, 10, 15, 20]),
        "duration": random.choice(["5m", "15m", "1h", "4h"]),
        "confidence": random.randint(85, 98)
    }


# -----------------------------
# SCANNER
# -----------------------------

def scan_market():

    results = []

    # Scan crypto
    for symbol in random.sample(CRYPTO_ASSETS, 5):

        price = get_crypto_price(symbol)

        if price is None:
            price = get_crypto_price_backup(symbol)

        if price is None:
            continue

        signal = generate_signal(price)

        results.append({
            "asset": symbol,
            "market": "Crypto",
            "price": price,
            "signal": signal
        })

    # Scan forex/metals
    for pair in random.sample(FOREX_METALS, 3):

        price = get_forex_price(pair)

        if price is None:
            continue

        signal = generate_signal(price)

        results.append({
            "asset": pair,
            "market": "Forex/Metals",
            "price": price,
            "signal": signal
        })

    return results


# -----------------------------
# ROUTES
# -----------------------------

@app.route("/")
def home():
    """Main URL auto runs scan"""
    data = scan_market()

    return jsonify({
        "status": "Trading Bot Active",
        "timestamp": int(time.time()),
        "results": data
    })


@app.route("/scan")
def scan():
    """Optional manual scan endpoint"""
    return home()


# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
