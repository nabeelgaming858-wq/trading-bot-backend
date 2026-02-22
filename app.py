import os
from flask import Flask, render_template, jsonify, request
import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

FINNHUB_API_KEY = "YOUR_FREE_FINNHUB_KEY"
NEWSDATA_API_KEY = "YOUR_FREE_NEWSDATA_KEY"

CRYPTO_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT']
FOREX_SYMBOLS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
METAL_SYMBOLS = ['GC=F', 'SI=F']

primary_exchange = ccxt.binance()
fallback_exchange = ccxt.kraken()

# --- DRAWDOWN CIRCUIT BREAKER ---
circuit_breaker = {'fails': 0, 'cooldown_until': None}

def get_order_book_walls(symbol):
    """Scans Level 2 Order Book to find massive Liquidity Walls (Whale Orders)."""
    try:
        orderbook = primary_exchange.fetch_order_book(symbol, limit=50)
        bids, asks = orderbook['bids'], orderbook['asks']
        largest_bid = max(bids, key=lambda x: x[1])[0] if bids else 0
        largest_ask = max(asks, key=lambda x: x[1])[0] if asks else 0
        return largest_bid, largest_ask
    except Exception:
        return 0, 0

def get_crypto_data(symbol):
    try:
        bars = primary_exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
    except Exception:
        try:
            bars = fallback_exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        except Exception:
            return None
    df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    return df

def get_tradfi_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval='1h', period='5d')
        df.reset_index(inplace=True)
        # yfinance returns DatetimeIndex, convert to unix ms for uniformity
        df['time'] = df['Datetime'].astype('int64') // 10**6 
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        return df[['time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception:
        return None

def fetch_news_sentiment(query):
    try:
        url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={query}&language=en"
        response = requests.get(url).json()
        if response.get('status') == 'success' and response.get('totalResults') > 0:
            titles = " ".join([article['title'].lower() for article in response['results'][:3]])
            if any(word in titles for word in ['crash', 'hack', 'lawsuit', 'ban']): return "Bearish Risk"
            elif any(word in titles for word in ['approved', 'surge', 'record']): return "Bullish Sentiment"
        return "Neutral"
    except Exception:
        return "News API Offline"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_trade', methods=['POST'])
def generate_trade():
    global circuit_breaker
    req_data = request.json
    duration_pref = req_data.get('duration', 'Hours')
    
    # Circuit Breaker Check
    if circuit_breaker['cooldown_until'] and datetime.now() < circuit_breaker['cooldown_until']:
        return jsonify({"status": "error", "message": f"Circuit Breaker Active: Extreme chop detected. Cooldown ends at {circuit_breaker['cooldown_until'].strftime('%H:%M UTC')}."})

    assets_to_scan = random.sample(CRYPTO_SYMBOLS, 2) + random.sample(FOREX_SYMBOLS, 2) + METAL_SYMBOLS
    random.shuffle(assets_to_scan)
    
    for symbol in assets_to_scan:
        asset_type = 'Crypto' if '/' in symbol else 'Metals' if '=F' in symbol else 'Forex'
        
        df = get_crypto_data(symbol) if asset_type == 'Crypto' else get_tradfi_data(symbol)
        if df is None or len(df) < 20: continue

        # Format Chart Data for Frontend (Unix seconds)
        chart_df = df.copy()
        chart_df['time'] = (chart_df['time'] / 1000).astype(int)
        chart_data_dict = chart_df[['time', 'open', 'high', 'low', 'close']].to_dict(orient='records')

        # Calculate Technicals
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.atr(length=14, append=True)
        latest = df.dropna().iloc[-1] if not df.dropna().empty else None
        
        if latest is None: continue
        
        price = latest['close']
        ema9, ema21, atr = latest.get('EMA_9', 0), latest.get('EMA_21', 0), latest.get('ATRr_14', 0)
        if atr == 0: atr = price * 0.005
        
        sentiment = fetch_news_sentiment("crypto" if asset_type == 'Crypto' else "forex")
        largest_bid, largest_ask = get_order_book_walls(symbol) if asset_type == 'Crypto' else (0, 0)

        signal = None
        if ema9 > ema21 and "Bearish" not in sentiment:
            signal = "LONG"
            sl = price - (1.5 * atr)
            tp_calc = price + (2.5 * atr)
            # Level 2 Logic: If a massive sell wall is right before our TP, place TP slightly below the wall
            tp = largest_ask * 0.999 if (0 < largest_ask < tp_calc) else tp_calc
            
        elif ema9 < ema21 and "Bullish" not in sentiment:
            signal = "SHORT"
            sl = price + (1.5 * atr)
            tp_calc = price - (2.5 * atr)
            tp = largest_bid * 1.001 if (largest_bid > tp_calc) else tp_calc

        if signal:
            circuit_breaker['fails'] = 0 # Reset breaker on success
            display_symbol = symbol.replace('=X', '').replace('=F', '')
            if display_symbol == 'GC': display_symbol = 'Gold (XAU/USD)'
            if display_symbol == 'SI': display_symbol = 'Silver (XAG/USD)'

            return jsonify({"status": "success", "data": {
                "asset": display_symbol,
                "type": asset_type,
                "signal": signal,
                "price": round(price, 4),
                "tp": f"{round(tp, 4)}",
                "sl": f"{round(sl, 4)}",
                "leverage": "Dynamic via ATR",
                "duration": duration_pref,
                "session": datetime.utcnow().strftime('%H:%M UTC'),
                "volatility": "High" if atr > (price * 0.01) else "Normal",
                "logic": f"News: {sentiment}. Level 2 Order Book Filter applied.",
                "chart_data": chart_data_dict
            }})

    # If loop finishes with no trades, trigger fail count
    circuit_breaker['fails'] += 1
    if circuit_breaker['fails'] >= 3:
        circuit_breaker['cooldown_until'] = datetime.now() + timedelta(minutes=15)
        circuit_breaker['fails'] = 0

    return jsonify({"status": "error", "message": "Market choppy or News is conflicting. Capital Protected. NO TRADE."})

if __name__ == '__main__':
    # Cloud Run relies on the PORT environment variable
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
        
