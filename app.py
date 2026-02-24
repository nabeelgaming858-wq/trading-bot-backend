import os
import time
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import threading
import ta
import yfinance as yf
import logging
from functools import lru_cache
import xml.etree.ElementTree as ET

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

# ==================== PRICE FETCHING (MULTI-API) ====================
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
        return None
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                price = data.get(coin_id, {}).get('usd')
                change = data.get(coin_id, {}).get('usd_24h_change')
                return price, change
    except Exception as e:
        logger.error(f"CoinGecko error {symbol}: {e}")
    return None, None

async def fetch_coincap(symbol):
    # CoinCap uses ids like 'bitcoin'
    coin_map = {
        'BTC/USD': 'bitcoin', 'ETH/USD': 'ethereum', 'BNB/USD': 'binance-coin',
        'SOL/USD': 'solana', 'ADA/USD': 'cardano', 'XRP/USD': 'xrp',
        'DOT/USD': 'polkadot', 'DOGE/USD': 'dogecoin', 'SHIB/USD': 'shiba-inu',
        'AVAX/USD': 'avalanche', 'LINK/USD': 'chainlink', 'MATIC/USD': 'polygon',
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
        url = f"https://api.coincap.io/v2/assets/{coin_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None, None
                data = await resp.json()
                price = float(data['data']['priceUsd'])
                change = float(data['data']['changePercent24Hr'])
                return price, change
    except Exception as e:
        logger.error(f"CoinCap error {symbol}: {e}")
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
                        # Frankfurter doesn't provide 24h change, so return None
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

async def fetch_yfinance(symbol):
    try:
        if symbol in CRYPTO_SYMBOLS:
            ticker = symbol.replace('/', '-')
        else:
            ticker = symbol.replace('/', '') + '=X'
        loop = asyncio.get_event_loop()
        ticker_obj = await loop.run_in_executor(None, lambda: yf.Ticker(ticker))
        data = await loop.run_in_executor(None, lambda: ticker_obj.history(period='2d', interval='1m').iloc[-1])
        if data.empty:
            return None, None
        price = data['Close']
        # Calculate 24h change using previous day's close
        hist = await loop.run_in_executor(None, lambda: ticker_obj.history(period='2d', interval='1d'))
        if len(hist) >= 2:
            prev_close = hist['Close'].iloc[-2]
            change = ((price - prev_close) / prev_close) * 100
        else:
            change = None
        return price, change
    except Exception as e:
        logger.error(f"yFinance error {symbol}: {e}")
    return None, None

async def get_price(symbol):
    now = time.time()
    if symbol in price_cache and now - price_cache_ts.get(symbol, 0) < CACHE_DURATION:
        return price_cache[symbol]

    price, change = None, None
    if symbol in CRYPTO_SYMBOLS:
        price, change = await fetch_coingecko(symbol)
        if not price:
            price, change = await fetch_coincap(symbol)
        if not price:
            price, change = await fetch_yfinance(symbol)
    else:
        price, change = await fetch_frankfurter(symbol)
        if not price:
            price, change = await fetch_yfinance(symbol)

    if price is None:
        logger.warning(f"All APIs failed for {symbol}, using mock")
        mock_prices = {
            'BTC/USD': 60000, 'ETH/USD': 3000, 'EUR/USD': 1.08, 'GBP/USD': 1.27,
            'USD/JPY': 150.5, 'AUD/USD': 0.66
        }
        price = mock_prices.get(symbol, 1.0)
        change = 0

    price_cache[symbol] = (price, change)
    price_cache_ts[symbol] = now
    return price, change

# ==================== HISTORICAL DATA ====================
@lru_cache(maxsize=100)
def get_historical_yf(symbol, interval='1h', period='5d'):
    try:
        if symbol in CRYPTO_SYMBOLS:
            ticker = symbol.replace('/', '-')
        else:
            ticker = symbol.replace('/', '') + '=X'
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if 'datetime' in df.columns:
            df.rename(columns={'datetime': 'date'}, inplace=True)
        return df
    except Exception as e:
        logger.error(f"yFinance historical error {symbol}: {e}")
        return None

async def fetch_historical(symbol, interval='1h', bars=100):
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(None, get_historical_yf, symbol, interval, '5d')
    if df is not None and len(df) >= bars:
        return df.tail(bars)
    return None

# ==================== INDICATORS ====================
def compute_indicators(df):
    if df is None or len(df) < 50:
        return None
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['open'] = pd.to_numeric(df['open'])
    df['volume'] = pd.to_numeric(df['volume']) if 'volume' in df else 0

    # Trend
    df['ema7'] = ta.trend.ema_indicator(df['close'], window=7)
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

    # Momentum
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # Volatility
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    # Volume
    df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)

    # Support/Resistance (simple rolling max/min)
    window = 10
    df['swing_high'] = df['high'].rolling(window, center=True).max()
    df['swing_low'] = df['low'].rolling(window, center=True).min()
    return df

def detect_market_structure(df):
    last_rows = df.tail(20)
    highs = last_rows['high'].values
    lows = last_rows['low'].values
    if len(highs) < 5:
        return 'ranging'
    recent_high = max(highs[-5:])
    prev_high = max(highs[-10:-5]) if len(highs) >= 10 else recent_high
    recent_low = min(lows[-5:])
    prev_low = min(lows[-10:-5]) if len(lows) >= 10 else recent_low
    if recent_high > prev_high and recent_low > prev_low:
        return 'uptrend'
    elif recent_high < prev_high and recent_low < prev_low:
        return 'downtrend'
    else:
        return 'ranging'

def get_current_session():
    hour = datetime.utcnow().hour
    for name, (start, end) in SESSIONS.items():
        if start <= hour < end:
            return name
    return 'asia'

def calculate_fib_levels(high, low):
    diff = high - low
    return {
        '0.236': low + 0.236 * diff,
        '0.382': low + 0.382 * diff,
        '0.5': low + 0.5 * diff,
        '0.618': low + 0.618 * diff,
        '0.786': low + 0.786 * diff,
    }

# ==================== NEWS ====================
async def fetch_cryptopanic():
    # CryptoPanic provides a free RSS feed without key
    try:
        url = "https://cryptopanic.com/news/rss/"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                text = await resp.text()
                root = ET.fromstring(text)
                articles = []
                for item in root.findall('.//item'):
                    title = item.find('title').text
                    description = item.find('description').text
                    articles.append({'title': title, 'description': description})
                return {'articles': articles[:5], 'sentiment': 'neutral'}
    except Exception as e:
        logger.error(f"CryptoPanic error: {e}")
    return None

async def fetch_forex_rss():
    # Example: use ForexFactory RSS (may not have descriptions)
    try:
        url = "https://www.forexfactory.com/news.xml"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                text = await resp.text()
                root = ET.fromstring(text)
                articles = []
                for item in root.findall('.//item'):
                    title = item.find('title').text
                    description = item.find('description').text if item.find('description') is not None else ''
                    articles.append({'title': title, 'description': description})
                return {'articles': articles[:5], 'sentiment': 'neutral'}
    except Exception as e:
        logger.error(f"Forex RSS error: {e}")
    return None

async def fetch_news(asset_class):
    if asset_class == 'crypto':
        news = await fetch_cryptopanic()
        if news:
            return news
    else:
        news = await fetch_forex_rss()
        if news:
            return news
    # Fallback mock
    return {
        'articles': [
            {'title': 'Market update: No major news', 'description': 'Stay tuned for updates.'}
        ],
        'sentiment': 'neutral'
    }

# ==================== SIGNAL SCORING ====================
def score_signal(df, direction, price, atr):
    """Return a confidence score 0-100 based on multiple factors"""
    latest = df.iloc[-1]
    score = 50  # base

    # Trend alignment
    if direction == 'long':
        if latest['ema7'] > latest['ema21']:
            score += 10
        if latest['adx'] > 25:
            score += 5
        if latest['rsi'] > 50 and latest['rsi'] < 70:
            score += 5
        if latest['macd'] > latest['macd_signal']:
            score += 5
        if price <= latest['bb_lower'] * 1.01:
            score += 5  # oversold bounce
    else:  # short
        if latest['ema7'] < latest['ema21']:
            score += 10
        if latest['adx'] > 25:
            score += 5
        if latest['rsi'] < 50 and latest['rsi'] > 30:
            score += 5
        if latest['macd'] < latest['macd_signal']:
            score += 5
        if price >= latest['bb_upper'] * 0.99:
            score += 5  # overbought rejection

    # Volume confirmation
    if latest['volume'] > latest['volume_sma'] * 1.2:
        score += 5

    # Volatility (ATR) – avoid extremely low or high
    atr_pct = atr / price
    if 0.005 < atr_pct < 0.03:
        score += 5

    # Support/Resistance proximity
    if direction == 'long' and price <= latest['swing_low'] * 1.02:
        score += 10
    if direction == 'short' and price >= latest['swing_high'] * 0.98:
        score += 10

    # Stochastic
    if direction == 'long' and latest['stoch_k'] < 20 and latest['stoch_k'] > latest['stoch_d']:
        score += 5
    if direction == 'short' and latest['stoch_k'] > 80 and latest['stoch_k'] < latest['stoch_d']:
        score += 5

    return min(95, max(0, score))

# ==================== TRADE GENERATION ====================
async def analyze_symbol(symbol, duration_minutes):
    price, change = await get_price(symbol)
    if not price:
        return None

    df = await fetch_historical(symbol, interval='1h', bars=100)
    if df is None or len(df) < 50:
        return None

    df = compute_indicators(df)
    if df is None:
        return None

    latest = df.iloc[-1]
    trend = detect_market_structure(df)
    session = get_current_session()
    atr = latest['atr']
    atr_percent = atr / price

    # Determine direction candidates
    long_score = 0
    short_score = 0

    # EMA alignment
    if latest['ema7'] > latest['ema9'] > latest['ema21']:
        long_score += 1
    elif latest['ema7'] < latest['ema9'] < latest['ema21']:
        short_score += 1

    # MACD
    if latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > 0:
        long_score += 1
    elif latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < 0:
        short_score += 1

    # RSI
    if latest['rsi'] < 30 and trend == 'uptrend':
        long_score += 1
    elif latest['rsi'] > 70 and trend == 'downtrend':
        short_score += 1

    # Bollinger
    if price <= latest['bb_lower'] and latest['close'] > latest['bb_lower']:
        long_score += 1
    elif price >= latest['bb_upper'] and latest['close'] < latest['bb_upper']:
        short_score += 1

    # Support/Resistance
    swing_low = latest['swing_low']
    swing_high = latest['swing_high']
    if price <= swing_low * 1.01 and latest['close'] > price:
        long_score += 1
    if price >= swing_high * 0.99 and latest['close'] < price:
        short_score += 1

    # ADX trend strength
    if latest['adx'] > 25:
        if trend == 'uptrend':
            long_score += 1
        elif trend == 'downtrend':
            short_score += 1

    # Decide direction
    direction = None
    if long_score >= 3 and long_score > short_score:
        direction = 'long'
    elif short_score >= 3 and short_score > long_score:
        direction = 'short'
    else:
        return None

    # Compute confidence score
    confidence = score_signal(df, direction, price, atr)

    # Calculate entry, stop, TP
    entry = price
    if direction == 'long':
        stop_level = swing_low
        stop = stop_level - (atr * 1.5)
        tp = entry + (abs(entry - stop) * 2)
    else:
        stop_level = swing_high
        stop = stop_level + (atr * 1.5)
        tp = entry - (abs(entry - stop) * 2)

    # Adjust TP for duration
    atr_per_hour = atr
    distance = abs(tp - entry)
    required_hours = distance / atr_per_hour if atr_per_hour > 0 else 999
    required_minutes = required_hours * 60
    if required_minutes > duration_minutes:
        max_distance = atr_per_hour * (duration_minutes / 60) * 0.8
        if direction == 'long':
            tp = entry + max_distance
        else:
            tp = entry - max_distance

    # Leverage
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
        'trend': trend,
        'confidence': confidence,
        'change_24h': change,
        'tp_percent': round(abs((tp - entry)/entry)*100, 2),
        'sl_percent': round(abs((stop - entry)/entry)*100, 2),
        'reason': f"{direction.upper()} signal with {confidence}% confidence. Trend: {trend}. ATR: {round(atr_percent*100,2)}%."
    }

async def scan_top_trades(asset_class, duration_minutes, top_n=3):
    symbols = CRYPTO_SYMBOLS[:15] if asset_class == 'crypto' else FOREX_SYMBOLS[:10]
    tasks = [analyze_symbol(sym, duration_minutes) for sym in symbols]
    results = await asyncio.gather(*tasks)
    signals = [r for r in results if r is not None]
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    return signals[:top_n]

# ==================== HEATMAP ====================
async def get_heatmap_data(asset_class):
    symbols = CRYPTO_SYMBOLS if asset_class == 'crypto' else FOREX_SYMBOLS
    symbols = symbols[:25]
    result = []
    for sym in symbols:
        price, change = await get_price(sym)
        result.append({
            'symbol': sym,
            'price': round(price, 5) if price else 0,
            'change': round(change, 2) if change else 0
        })
    return result

# ==================== BACKGROUND PRICE UPDATER ====================
def update_prices():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        for sym in ALL_SYMBOLS[:20]:
            loop.run_until_complete(get_price(sym))
        time.sleep(30)

threading.Thread(target=update_prices, daemon=True).start()

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scan', methods=['POST'])
def api_scan():
    data = request.json
    asset_class = data.get('asset_class', 'crypto')
    duration = data.get('duration', 60)
    try:
        duration = int(duration)
    except:
        duration = 60

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        signals = loop.run_until_complete(scan_top_trades(asset_class, duration, top_n=3))
    except Exception as e:
        logger.exception("Scan error")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500
    return jsonify({'success': True, 'signals': signals})

@app.route('/api/prices', methods=['POST'])
def api_prices():
    data = request.json
    asset_class = data.get('asset_class', 'crypto')
    symbols = CRYPTO_SYMBOLS if asset_class == 'crypto' else FOREX_SYMBOLS
    symbols = symbols[:15]
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
