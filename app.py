import os
import time
import json
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime
import threading
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ==================== CONFIG ====================
CRYPTO_SYMBOLS = [
    'BTC/USD', 'ETH/USD', 'BNB/USD', 'SOL/USD', 'ADA/USD', 'XRP/USD', 'DOT/USD',
    'DOGE/USD', 'SHIB/USD', 'AVAX/USD', 'LINK/USD', 'MATIC/USD', 'LTC/USD',
    'BCH/USD', 'ALGO/USD', 'XLM/USD', 'VET/USD', 'FIL/USD', 'TRX/USD',
    'EOS/USD', 'AAVE/USD', 'MKR/USD', 'YFI/USD', 'SUSHI/USD', 'UNI/USD'
]

FOREX_SYMBOLS = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD',
    'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'EUR/AUD', 'GBP/AUD', 'USD/ZAR',
    'USD/TRY'
]

ALL_SYMBOLS = CRYPTO_SYMBOLS + FOREX_SYMBOLS

SESSIONS = {
    'asia': (0, 9),
    'london': (8, 17),
    'ny': (13, 22),
    'pacific': (22, 24)
}

# Cache
price_cache = {}
price_cache_ts = {}
CACHE_DURATION = 10

# ==================== PRICE FETCHING (FREE, NO KEY) ====================
async def fetch_coingecko(symbol):
    coin_map = {
        'BTC/USD': 'bitcoin', 'ETH/USD': 'ethereum', 'BNB/USD': 'binancecoin',
        'SOL/USD': 'solana', 'ADA/USD': 'cardano', 'XRP/USD': 'ripple',
        'DOT/USD': 'polkadot', 'DOGE/USD': 'dogecoin', 'SHIB/USD': 'shiba-inu',
        'AVAX/USD': 'avalanche-2', 'LINK/USD': 'chainlink', 'MATIC/USD': 'matic-network',
        'LTC/USD': 'litecoin', 'BCH/USD': 'bitcoin-cash', 'ALGO/USD': 'algorand',
        'XLM/USD': 'stellar', 'VET/USD': 'vechain', 'FIL/USD': 'filecoin',
        'TRX/USD': 'tron', 'EOS/USD': 'eos', 'AAVE/USD': 'aave',
        'MKR/USD': 'maker', 'YFI/USD': 'yearn-finance', 'SUSHI/USD': 'sushi',
        'UNI/USD': 'uniswap'
    }
    coin_id = coin_map.get(symbol)
    if not coin_id:
        return None, None
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None, None
                data = await resp.json()
                price = data.get(coin_id, {}).get('usd')
                change = data.get(coin_id, {}).get('usd_24h_change')
                return price, change
    except Exception as e:
        logger.error(f"CoinGecko error {symbol}: {e}")
    return None, None

async def fetch_frankfurter(symbol):
    try:
        base, quote = symbol.split('/')
        if base == 'EUR':
            url = f"https://api.frankfurter.app/latest?from={base}&to={quote}"
        elif quote == 'EUR':
            url = f"https://api.frankfurter.app/latest?from={base}&to={quote}"
        else:
            # Cross rate via EUR
            async with aiohttp.ClientSession() as session:
                url1 = f"https://api.frankfurter.app/latest?from=EUR&to={base}"
                url2 = f"https://api.frankfurter.app/latest?from=EUR&to={quote}"
                async with session.get(url1) as resp1, session.get(url2) as resp2:
                    if resp1.status != 200 or resp2.status != 200:
                        return None, None
                    data1 = await resp1.json()
                    data2 = await resp2.json()
                    rate_base = data1['rates'].get(base)
                    rate_quote = data2['rates'].get(quote)
                    if rate_base and rate_quote:
                        base_per_eur = 1 / rate_base
                        price = base_per_eur * rate_quote
                        return price, None
            return None, None
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None, None
                data = await resp.json()
                price = data['rates'].get(quote)
                if base == 'EUR':
                    return price, None
                else:  # quote == 'EUR'
                    return 1/price, None
    except Exception as e:
        logger.error(f"Frankfurter error {symbol}: {e}")
    return None, None

async def get_price(symbol):
    now = time.time()
    if symbol in price_cache and now - price_cache_ts.get(symbol, 0) < CACHE_DURATION:
        return price_cache[symbol]

    price, change = None, None
    if symbol in CRYPTO_SYMBOLS:
        price, change = await fetch_coingecko(symbol)
    else:
        price, change = await fetch_frankfurter(symbol)

    # Fallback to mock if all APIs fail (for demo only)
    if price is None:
        logger.warning(f"Using mock for {symbol}")
        mock_prices = {
            'BTC/USD': 60000, 'ETH/USD': 3000, 'EUR/USD': 1.08, 'GBP/USD': 1.27,
            'USD/JPY': 150.5, 'AUD/USD': 0.66
        }
        price = mock_prices.get(symbol, 1.0)
        change = 0

    price_cache[symbol] = (price, change)
    price_cache_ts[symbol] = now
    return price, change

# ==================== HISTORICAL DATA (SIMULATED FOR FOREX) ====================
async def fetch_historical(symbol, interval='1h', bars=100):
    """For crypto, use CoinGecko; for forex, simulate with random walk (for demo). 
       In production, integrate a proper API like Alpha Vantage (needs key)."""
    # Simplified: generate synthetic data based on current price and volatility
    price, _ = await get_price(symbol)
    if not price:
        return None
    # Create a simple DataFrame with some volatility
    dates = pd.date_range(end=datetime.now(), periods=bars, freq='H')
    closes = [price * (1 + np.random.randn()*0.01) for _ in range(bars)]
    # Ensure last price is current
    closes[-1] = price
    df = pd.DataFrame({
        'date': dates,
        'close': closes,
        'high': [c*1.005 for c in closes],
        'low': [c*0.995 for c in closes],
        'open': [c*0.998 for c in closes],
        'volume': [1000] * bars
    })
    return df

# ==================== INDICATORS ====================
def compute_indicators(df):
    if df is None or len(df) < 30:
        return None
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['open'] = pd.to_numeric(df['open'])
    df['volume'] = pd.to_numeric(df['volume'])

    # EMAs
    df['ema7'] = df['close'].ewm(span=7).mean()
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()

    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std

    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # Swing highs/lows
    window = 10
    df['swing_high'] = df['high'].rolling(window, center=True).max()
    df['swing_low'] = df['low'].rolling(window, center=True).min()
    return df

def get_current_session():
    hour = datetime.utcnow().hour
    for name, (start, end) in SESSIONS.items():
        if start <= hour < end:
            return name
    return 'asia'

def score_signal(df, direction, price, atr):
    latest = df.iloc[-1]
    score = 50
    if direction == 'long':
        if latest['ema7'] > latest['ema21']:
            score += 10
        if latest['rsi'] > 40 and latest['rsi'] < 70:
            score += 5
        if latest['macd'] > latest['macd_signal']:
            score += 5
        if price <= latest['bb_lower'] * 1.01:
            score += 5
    else:
        if latest['ema7'] < latest['ema21']:
            score += 10
        if latest['rsi'] < 60 and latest['rsi'] > 30:
            score += 5
        if latest['macd'] < latest['macd_signal']:
            score += 5
        if price >= latest['bb_upper'] * 0.99:
            score += 5
    # Volume
    if latest['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.2:
        score += 5
    atr_pct = atr / price
    if 0.002 < atr_pct < 0.05:
        score += 5
    # Support/Resistance
    if direction == 'long' and price <= latest['swing_low'] * 1.02:
        score += 10
    if direction == 'short' and price >= latest['swing_high'] * 0.98:
        score += 10
    return min(95, max(0, score))

async def analyze_symbol(symbol, duration_minutes):
    price, change = await get_price(symbol)
    if not price:
        return None

    df = await fetch_historical(symbol, bars=100)
    if df is None or len(df) < 30:
        return None

    df = compute_indicators(df)
    if df is None:
        return None

    latest = df.iloc[-1]
    session = get_current_session()
    atr = latest['atr']
    atr_percent = atr / price

    # Determine direction
    long_score = 0
    short_score = 0

    if latest['ema7'] > latest['ema21']:
        long_score += 1
    elif latest['ema7'] < latest['ema21']:
        short_score += 1

    if latest['macd'] > latest['macd_signal']:
        long_score += 1
    elif latest['macd'] < latest['macd_signal']:
        short_score += 1

    if latest['rsi'] < 40:
        long_score += 1
    elif latest['rsi'] > 60:
        short_score += 1

    if price <= latest['bb_lower']:
        long_score += 1
    elif price >= latest['bb_upper']:
        short_score += 1

    swing_low = latest['swing_low']
    swing_high = latest['swing_high']
    if price <= swing_low * 1.01:
        long_score += 1
    if price >= swing_high * 0.99:
        short_score += 1

    direction = None
    if long_score >= 3 and long_score > short_score:
        direction = 'long'
    elif short_score >= 3 and short_score > long_score:
        direction = 'short'
    else:
        return None

    confidence = score_signal(df, direction, price, atr)

    entry = price
    if direction == 'long':
        stop = swing_low - atr * 1.5
        tp = entry + (entry - stop) * 2
    else:
        stop = swing_high + atr * 1.5
        tp = entry - (stop - entry) * 2

    # Adjust TP for duration (simplified)
    atr_per_hour = atr
    required_minutes = (abs(tp - entry) / atr_per_hour) * 60 if atr_per_hour > 0 else 999
    if required_minutes > duration_minutes:
        max_distance = atr_per_hour * (duration_minutes / 60) * 0.8
        if direction == 'long':
            tp = entry + max_distance
        else:
            tp = entry - max_distance

    asset_class = 'crypto' if symbol in CRYPTO_SYMBOLS else 'forex'
    session_factor = 1.0
    if session == 'asia':
        session_factor = 0.8
    elif session == 'ny':
        session_factor = 1.2
    base_leverage = 5 if asset_class == 'crypto' else 10
    vol_factor = max(0.2, min(1.0, 1.0 - (atr_percent * 10)))
    leverage = int(base_leverage * vol_factor * session_factor)
    leverage = max(1, min(leverage, 50))

    return {
        'asset': symbol,
        'direction': direction.upper(),
        'entry': round(entry, 5),
        'stop_loss': round(stop, 5),
        'take_profit': round(tp, 5),
        'leverage': leverage,
        'duration': duration_minutes,
        'session': session,
        'confidence': confidence,
        'change_24h': change,
        'tp_percent': round(abs((tp - entry)/entry)*100, 2),
        'sl_percent': round(abs((stop - entry)/entry)*100, 2),
        'reason': f"{direction.upper()} with {confidence}% confidence. ATR: {round(atr_percent*100,2)}%."
    }

async def scan_top_trades(asset_class, duration_minutes, top_n=3):
    symbols = CRYPTO_SYMBOLS[:10] if asset_class == 'crypto' else FOREX_SYMBOLS[:8]
    tasks = [analyze_symbol(sym, duration_minutes) for sym in symbols]
    results = await asyncio.gather(*tasks)
    signals = [r for r in results if r is not None]
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    return signals[:top_n]

async def get_heatmap_data(asset_class):
    symbols = CRYPTO_SYMBOLS[:20] if asset_class == 'crypto' else FOREX_SYMBOLS[:15]
    result = []
    for sym in symbols:
        price, change = await get_price(sym)
        result.append({
            'symbol': sym,
            'price': round(price, 5) if price else 0,
            'change': round(change, 2) if change else 0
        })
    return result

async def fetch_news(asset_class):
    # Use free RSS feeds
    if asset_class == 'crypto':
        url = "https://cryptopanic.com/news/rss/"
    else:
        url = "https://www.forexfactory.com/news.xml"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return {'articles': []}
                text = await resp.text()
                import xml.etree.ElementTree as ET
                root = ET.fromstring(text)
                articles = []
                for item in root.findall('.//item'):
                    title = item.find('title').text
                    desc = item.find('description').text if item.find('description') is not None else ''
                    articles.append({'title': title, 'description': desc[:200]})
                return {'articles': articles[:5]}
    except Exception as e:
        logger.error(f"News error: {e}")
        return {'articles': []}

# ==================== BACKGROUND UPDATER ====================
def update_prices():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        for sym in ALL_SYMBOLS[:10]:
            loop.run_until_complete(get_price(sym))
        time.sleep(30)

threading.Thread(target=update_prices, daemon=True).start()

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Template error: {e}")
        return "Server error: Template not found. Please check deployment.", 500

@app.route('/api/scan', methods=['POST'])
def api_scan():
    data = request.json
    asset_class = data.get('asset_class', 'crypto')
    duration = int(data.get('duration', 60))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        signals = loop.run_until_complete(scan_top_trades(asset_class, duration))
        return jsonify({'success': True, 'signals': signals})
    except Exception as e:
        logger.exception("Scan error")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/prices', methods=['POST'])
def api_prices():
    data = request.json
    asset_class = data.get('asset_class', 'crypto')
    symbols = CRYPTO_SYMBOLS if asset_class == 'crypto' else FOREX_SYMBOLS
    symbols = symbols[:10]
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    prices = {}
    for sym in symbols:
        price, _ = loop.run_until_complete(get_price(sym))
        prices[sym] = price
    return jsonify(prices)

@app.route('/api/news', methods=['POST'])
def api_news():
    data = request.json
    asset_class = data.get('asset_class', 'crypto')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    news = loop.run_until_complete(fetch_news(asset_class))
    return jsonify(news)

@app.route('/api/heatmap', methods=['POST'])
def api_heatmap():
    data = request.json
    asset_class = data.get('asset_class', 'crypto')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    heat = loop.run_until_complete(get_heatmap_data(asset_class))
    return jsonify(heat)

@app.route('/api/session')
def api_session():
    return jsonify({'session': get_current_session()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
