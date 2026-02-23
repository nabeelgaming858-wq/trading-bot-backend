import os
import json
import time
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime
import threading
import ta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ==================== CONFIGURATION ====================
# Free API keys – set as environment variables in production
GNEWS_API_KEY = os.environ.get('GNEWS_API_KEY', '')  # Get from https://gnews.io/
EXCHANGERATE_API_KEY = os.environ.get('EXCHANGERATE_API_KEY', '')  # Get from https://exchangerate-api.com/

# Asset lists (no metals)
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

# Session times (UTC)
SESSIONS = {
    'asia': (0, 9),
    'london': (8, 17),
    'ny': (13, 22),
    'pacific': (22, 24)
}

# Cache for prices
price_cache = {}
price_cache_ts = {}
CACHE_DURATION = 10  # seconds

last_asset = None  # To avoid repeats

# ==================== DATA FETCHING ====================
async def fetch_coingecko(symbol):
    """Fetch crypto price from CoinGecko (no key required)"""
    try:
        # Map symbol to CoinGecko ID
        coin_map = {
            'BTC/USD': 'bitcoin',
            'ETH/USD': 'ethereum',
            'BNB/USD': 'binancecoin',
            'SOL/USD': 'solana',
            'ADA/USD': 'cardano',
            'XRP/USD': 'ripple',
            'DOT/USD': 'polkadot',
            'DOGE/USD': 'dogecoin',
            'SHIB/USD': 'shiba-inu',
            'AVAX/USD': 'avalanche-2',
            'LINK/USD': 'chainlink',
            'MATIC/USD': 'matic-network',
            'LTC/USD': 'litecoin',
            'BCH/USD': 'bitcoin-cash',
            'ALGO/USD': 'algorand',
            'XLM/USD': 'stellar',
            'VET/USD': 'vechain',
            'FIL/USD': 'filecoin',
            'TRX/USD': 'tron',
            'EOS/USD': 'eos',
            'AAVE/USD': 'aave',
            'MKR/USD': 'maker',
            'YFI/USD': 'yearn-finance',
            'SUSHI/USD': 'sushi',
            'UNI/USD': 'uniswap'
        }
        coin_id = coin_map.get(symbol)
        if not coin_id:
            return None
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data.get(coin_id, {}).get('usd')
    except Exception as e:
        logger.error(f"CoinGecko error for {symbol}: {e}")
    return None

async def fetch_frankfurter(symbol):
    """Fetch forex price from Frankfurter (free, no key, EUR-based)"""
    try:
        base, quote = symbol.split('/')
        # Frankfurter uses EUR as base, so we need to get rates
        # For pairs involving EUR, direct; for others, derive
        if base == 'EUR':
            url = f"https://api.frankfurter.app/latest?from={base}&to={quote}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    return data['rates'].get(quote)
        elif quote == 'EUR':
            url = f"https://api.frankfurter.app/latest?from={base}&to={quote}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                rate = data['rates'].get(quote)
                return 1/rate if rate else None
        else:
            # Cross rate via EUR
            url1 = f"https://api.frankfurter.app/latest?from=EUR&to={base}"
            url2 = f"https://api.frankfurter.app/latest?from=EUR&to={quote}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url1) as resp1, session.get(url2) as resp2:
                    if resp1.status != 200 or resp2.status != 200:
                        return None
                    data1 = await resp1.json()
                    data2 = await resp2.json()
                    rate_base = data1['rates'].get(base)  # EUR to base
                    rate_quote = data2['rates'].get(quote)  # EUR to quote
                    if rate_base and rate_quote:
                        # base/usd = (eur/usd) / (eur/base) ??? Wait, careful.
                        # We want base/quote = (base/eur) * (eur/quote)
                        # base/eur = 1/(eur/base)
                        base_per_eur = 1 / rate_base if rate_base else None
                        eur_per_quote = rate_quote
                        if base_per_eur and eur_per_quote:
                            return base_per_eur * eur_per_quote
    except Exception as e:
        logger.error(f"Frankfurter error for {symbol}: {e}")
    return None

async def fetch_exchangerate_api(symbol):
    """Fetch forex price from ExchangeRate-API (free tier requires key)"""
    if not EXCHANGERATE_API_KEY:
        return None
    try:
        base, quote = symbol.split('/')
        url = f"https://v6.exchangerate-api.com/v6/{EXCHANGERATE_API_KEY}/pair/{base}/{quote}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data['result'] == 'success':
                    return data['conversion_rate']
    except Exception as e:
        logger.error(f"ExchangeRate-API error for {symbol}: {e}")
    return None

async def get_price(symbol):
    """Get latest price using multiple free APIs with fallback"""
    now = time.time()
    if symbol in price_cache and now - price_cache_ts.get(symbol, 0) < CACHE_DURATION:
        return price_cache[symbol]

    price = None
    if symbol in CRYPTO_SYMBOLS:
        price = await fetch_coingecko(symbol)
        # No fallback for crypto except maybe mock (but CoinGecko is reliable)
    else:  # forex
        price = await fetch_frankfurter(symbol)
        if not price and EXCHANGERATE_API_KEY:
            price = await fetch_exchangerate_api(symbol)

    # If still None, use a reasonable mock (for development only)
    if not price:
        logger.warning(f"All APIs failed for {symbol}, using mock")
        mock_prices = {
            'BTC/USD': 60000, 'ETH/USD': 3000, 'EUR/USD': 1.08, 'GBP/USD': 1.27,
            'USD/JPY': 150.5, 'AUD/USD': 0.66
        }
        price = mock_prices.get(symbol, 1.0)

    price_cache[symbol] = price
    price_cache_ts[symbol] = now
    return price

# ==================== TECHNICAL INDICATORS ====================
async def fetch_historical(symbol, interval='1h', bars=100):
    """Fetch historical data from free sources"""
    # For crypto, we can use CoinGecko historical; for forex, Frankfurter doesn't provide.
    # We'll use a fallback: for crypto, use CoinGecko; for forex, we'll simulate or use another free API.
    # To keep it simple, we'll use TwelveData's free demo? But that requires key.
    # Instead, we'll use a local generator (not accurate) or skip indicators for forex?
    # Given the requirement for real analysis, we need historical data. We'll use Alpha Vantage's free tier (demo key) – it's free but limited.
    # But the user said "I don't want to put my one's use the free and put it in" – meaning they don't want to provide keys. So we'll use free APIs that don't require keys: for crypto, CoinGecko provides historical; for forex, we can use Frankfurter's historical? It doesn't. So we'll have to make a compromise: for forex, we'll use a simple volatility model based on recent price changes (simulated) or skip forex scanning? That's not ideal.
    # Alternative: use Yahoo Finance (yfinance) – free, no key, but requires yfinance library. That might be acceptable.
    # We'll add yfinance as a dependency and use it for both crypto and forex historical.
    # However, yfinance may not have all forex pairs. We'll use it as a fallback.
    # Let's implement using yfinance (async? we can use asyncio.to_thread).

    # For now, we'll implement a placeholder that returns None for forex historical, and for crypto we'll use CoinGecko.
    # But the bot needs indicators, so we must have historical data. We'll use yfinance.
    import yfinance as yf
    try:
        # Convert symbol to yfinance format
        if symbol in CRYPTO_SYMBOLS:
            ticker = symbol.replace('/', '-')  # BTC-USD
        else:
            ticker = symbol.replace('/', '') + '=X'  # EURUSD=X
        # Download data (async in thread)
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, lambda: yf.download(ticker, period='5d', interval='1h'))
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={'datetime': 'date'}, inplace=True)
        return df
    except Exception as e:
        logger.error(f"yFinance error for {symbol}: {e}")
        return None

def compute_indicators(df):
    """Compute all technical indicators on DataFrame"""
    if df is None or len(df) < 50:
        return None
    # Ensure numeric
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['open'] = pd.to_numeric(df['open'])
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'])
    else:
        df['volume'] = 0
    # EMAs
    df['ema7'] = ta.trend.ema_indicator(df['close'], window=7)
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    # Support/Resistance (simplified: recent swing highs/lows)
    window = 10
    df['swing_high'] = df['high'].rolling(window, center=True).max()
    df['swing_low'] = df['low'].rolling(window, center=True).min()
    return df

def detect_market_structure(df):
    """Identify trend: HH/HL, LH/LL"""
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
    for session_name, (start, end) in SESSIONS.items():
        if start <= hour < end:
            return session_name
    return 'asia'

def calculate_atr_based_stop(entry, swing_level, atr, direction='long', multiplier=1.5):
    if direction == 'long':
        stop = swing_level - (atr * multiplier)
    else:
        stop = swing_level + (atr * multiplier)
    return stop

def calculate_tp(entry, stop, direction, risk_reward=2):
    risk = abs(entry - stop)
    if direction == 'long':
        tp = entry + (risk * risk_reward)
    else:
        tp = entry - (risk * risk_reward)
    return tp

def adjust_tp_for_duration(entry, tp, direction, duration_minutes, atr_per_hour):
    distance = abs(tp - entry)
    required_hours = distance / atr_per_hour if atr_per_hour > 0 else 999
    required_minutes = required_hours * 60
    if required_minutes > duration_minutes:
        max_distance = atr_per_hour * (duration_minutes / 60) * 0.8
        if direction == 'long':
            tp = entry + max_distance
        else:
            tp = entry - max_distance
    return tp

def calculate_dynamic_leverage(asset_class, atr_percent, session_volatility_factor=1.0):
    if asset_class == 'crypto':
        base_leverage = 5
    else:  # forex
        base_leverage = 10
    vol_factor = max(0.2, min(1.0, 1.0 - (atr_percent * 10)))
    leverage = int(base_leverage * vol_factor * session_volatility_factor)
    return max(1, min(leverage, 50))

def get_asset_class(symbol):
    return 'crypto' if symbol in CRYPTO_SYMBOLS else 'forex'

# ==================== NEWS SENTIMENT ====================
async def fetch_gnews(query):
    if not GNEWS_API_KEY:
        return None
    try:
        url = f"https://gnews.io/api/v4/search?q={query}&token={GNEWS_API_KEY}&lang=en&max=3"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                articles = data.get('articles', [])
                headlines = [a['title'] for a in articles]
                sentiment = 'neutral'
                for title in headlines:
                    lower = title.lower()
                    if any(w in lower for w in ['bull', 'surge', 'rally', 'gain']):
                        sentiment = 'bullish'
                    elif any(w in lower for w in ['bear', 'crash', 'plunge', 'drop']):
                        sentiment = 'bearish'
                return {'sentiment': sentiment, 'headlines': headlines}
    except Exception as e:
        logger.error(f"GNews error: {e}")
    return None

async def fetch_news(asset_class, symbol):
    query = symbol.split('/')[0]
    if query in ['XAU', 'XAG']:  # should not happen now
        return {'sentiment': 'neutral', 'headlines': []}
    news = await fetch_gnews(query)
    if news:
        return news
    return {'sentiment': 'neutral', 'headlines': []}

# ==================== SIGNAL GENERATION ====================
async def scan_symbol(symbol, duration_minutes):
    global last_asset
    if symbol == last_asset:
        return None

    price = await get_price(symbol)
    if not price:
        return None

    df = await fetch_historical(symbol, interval='1h', bars=100)
    if df is None or len(df) < 50:
        return None

    df = compute_indicators(df)
    if df is None:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    trend = detect_market_structure(df)
    session = get_current_session()

    atr = latest['atr']
    atr_percent = atr / price

    long_score = 0
    short_score = 0

    # Condition 1: EMAs
    if latest['ema7'] > latest['ema9'] > latest['ema21']:
        long_score += 1
    elif latest['ema7'] < latest['ema9'] < latest['ema21']:
        short_score += 1

    # Condition 2: MACD
    if latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > 0:
        long_score += 1
    elif latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < 0:
        short_score += 1

    # Condition 3: RSI
    if latest['rsi'] < 30 and trend == 'uptrend':
        long_score += 1
    elif latest['rsi'] > 70 and trend == 'downtrend':
        short_score += 1

    # Condition 4: Bollinger Bands
    if price <= latest['bb_lower'] and latest['close'] > latest['bb_lower']:
        long_score += 1
    elif price >= latest['bb_upper'] and latest['close'] < latest['bb_upper']:
        short_score += 1

    # Condition 5: Support/Resistance
    swing_low = latest['swing_low']
    swing_high = latest['swing_high']
    if price <= swing_low * 1.01 and latest['close'] > price:
        long_score += 1
    if price >= swing_high * 0.99 and latest['close'] < price:
        short_score += 1

    direction = None
    if long_score >= 3 and long_score > short_score and trend in ['uptrend', 'ranging']:
        direction = 'long'
    elif short_score >= 3 and short_score > long_score and trend in ['downtrend', 'ranging']:
        direction = 'short'
    else:
        return None

    news = await fetch_news(get_asset_class(symbol), symbol)
    if news['sentiment'] == 'bearish' and direction == 'long':
        return None
    if news['sentiment'] == 'bullish' and direction == 'short':
        return None

    entry = price
    if direction == 'long':
        stop_level = swing_low
        stop = calculate_atr_based_stop(entry, stop_level, atr, 'long')
        tp = calculate_tp(entry, stop, 'long', risk_reward=2)
    else:
        stop_level = swing_high
        stop = calculate_atr_based_stop(entry, stop_level, atr, 'short')
        tp = calculate_tp(entry, stop, 'short', risk_reward=2)

    atr_per_hour = atr
    tp = adjust_tp_for_duration(entry, tp, direction, duration_minutes, atr_per_hour)

    if direction == 'long' and tp <= entry:
        tp = entry + (atr * 1.5)
    if direction == 'short' and tp >= entry:
        tp = entry - (atr * 1.5)

    asset_class = get_asset_class(symbol)
    session_volatility_factor = 1.0
    if session in ['asia']:
        session_volatility_factor = 0.8
    elif session in ['ny']:
        session_volatility_factor = 1.2
    leverage = calculate_dynamic_leverage(asset_class, atr_percent, session_volatility_factor)

    signal = {
        'asset': symbol,
        'direction': direction.upper(),
        'entry': round(entry, 5),
        'stop_loss': round(stop, 5),
        'take_profit': round(tp, 5),
        'leverage': leverage,
        'duration': duration_minutes,
        'session': session,
        'trend': trend,
        'confidence': min(95, int((long_score if direction=='long' else short_score) * 20)),
        'news': news['headlines'][0] if news['headlines'] else 'No major news',
        'reason': f"{direction.upper()} signal due to {long_score if direction=='long' else short_score} indicators confluence, ATR-adjusted SL, TP based on {duration_minutes}min duration.",
        'tp_percent': round(abs((tp - entry)/entry)*100, 2),
        'sl_percent': round(abs((stop - entry)/entry)*100, 2),
        'entry_price': round(entry, 5),
        'current_price': round(price, 5)
    }
    return signal

async def scan_all(duration_minutes):
    # Limit to top 10 crypto + 10 forex for speed
    symbols_to_scan = CRYPTO_SYMBOLS[:10] + FOREX_SYMBOLS[:10]
    tasks = [scan_symbol(sym, duration_minutes) for sym in symbols_to_scan]
    results = await asyncio.gather(*tasks)
    signals = [r for r in results if r is not None]
    if not signals:
        return None
    best = max(signals, key=lambda x: x['confidence'])
    return best

# ==================== BACKGROUND TASKS ====================
def update_prices_periodically():
    while True:
        for sym in ALL_SYMBOLS[:20]:  # limit to avoid rate limits
            asyncio.run(get_price(sym))
        time.sleep(30)

threading.Thread(target=update_prices_periodically, daemon=True).start()

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scan', methods=['POST'])
def api_scan():
    data
