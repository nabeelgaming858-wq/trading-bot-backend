import os
import json
import random
import time
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, render_template, session, request, jsonify
import requests
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# ==================== CONFIGURATION ====================
# Asset lists (will be dynamically updated from APIs)
CRYPTO_SYMBOLS = [
    'BTC/USD', 'ETH/USD', 'BNB/USD', 'SOL/USD', 'XRP/USD',
    'ADA/USD', 'AVAX/USD', 'DOGE/USD', 'DOT/USD', 'MATIC/USD',
    'SHIB/USD', 'TRX/USD', 'LTC/USD', 'BCH/USD', 'ATOM/USD',
    'LINK/USD', 'ETC/USD', 'XLM/USD', 'APT/USD', 'FIL/USD',
    'NEAR/USD', 'ALGO/USD', 'VET/USD', 'ICP/USD', 'HBAR/USD'
]
FOREX_SYMBOLS = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD',
    'USD/CHF', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY',
    'AUD/JPY', 'EUR/AUD', 'GBP/AUD', 'USD/TRY', 'USD/ZAR'
]

# API Keys from environment
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')
CRYPTO_API_KEY = os.environ.get('CRYPTO_API_KEY', '')   # e.g., CoinMarketCap
FOREX_API_KEY = os.environ.get('FOREX_API_KEY', '')     # e.g., TwelveData
ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY', '')

# ==================== HELPER FUNCTIONS ====================
def fetch_with_fallback(urls, params=None, headers=None):
    """Try multiple URLs until one succeeds."""
    for url in urls:
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except:
            continue
    return None

def get_crypto_prices():
    """Fetch top 25 crypto prices with fallback."""
    # Primary: CoinGecko (free, no key)
    urls = [
        'https://api.coingecko.com/api/v3/coins/markets',
        'https://api.coinpaprika.com/v1/tickers'  # fallback
    ]
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 25,
        'page': 1,
        'sparkline': 'false'
    }
    data = fetch_with_fallback([urls[0]], params=params)
    if data:
        return {f"{item['symbol'].upper()}/USD": item['current_price'] for item in data}
    # Fallback to mock
    return {sym: round(1000 + 5000 * random.random(), 2) for sym in CRYPTO_SYMBOLS}

def get_forex_prices():
    """Fetch forex prices (fallback to mock)."""
    # Use Alpha Vantage or TwelveData if keys available
    if ALPHA_VANTAGE_KEY:
        prices = {}
        for sym in FOREX_SYMBOLS:
            from_curr, to_curr = sym.replace('/', ''), 'USD'  # simplification
            url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_curr}&to_currency={to_curr}&apikey={ALPHA_VANTAGE_KEY}'
            try:
                resp = requests.get(url, timeout=5)
                data = resp.json()
                rate = data['Realtime Currency Exchange Rate']['5. Exchange Rate']
                prices[sym] = float(rate)
            except:
                prices[sym] = round(1.0 + 0.5 * random.random(), 4)
        return prices
    # Mock
    return {sym: round(1.0 + 0.5 * random.random(), 4) for sym in FOREX_SYMBOLS}

def get_ohlc(symbol, interval='1h', limit=100):
    """
    Fetch OHLC data.
    For crypto: use Binance public API.
    For forex: use TwelveData or fallback to simulated data.
    """
    if 'USD' in symbol and symbol.split('/')[0] in ['BTC','ETH','BNB','SOL','XRP']:
        # Crypto
        base = symbol.split('/')[0].lower()
        url = f'https://api.binance.com/api/v3/klines?symbol={base.upper()}USDT&interval={interval}&limit={limit}'
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            ohlc = []
            for k in data:
                ohlc.append({
                    'time': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4])
                })
            return ohlc
        except:
            pass
    # Simulate OHLC for demo
    return generate_simulated_ohlc(limit)

def generate_simulated_ohlc(limit=100):
    """Generate fake OHLC for testing."""
    base = 1000
    ohlc = []
    for i in range(limit):
        close = base + random.uniform(-10, 10)
        high = close + random.uniform(0, 5)
        low = close - random.uniform(0, 5)
        ohlc.append({
            'time': i,
            'open': base,
            'high': high,
            'low': low,
            'close': close
        })
        base = close
    return ohlc

def calculate_atr(ohlc, period=14):
    """Average True Range."""
    if len(ohlc) < period:
        return 0
    high = np.array([x['high'] for x in ohlc])
    low = np.array([x['low'] for x in ohlc])
    close = np.array([x['close'] for x in ohlc])
    tr = np.maximum(high[1:] - low[1:], 
                    np.abs(high[1:] - close[:-1]), 
                    np.abs(low[1:] - close[:-1]))
    atr = np.mean(tr[-period:])
    return atr

def calculate_indicators(ohlc):
    """Return trend, support, resistance, EMA, RSI, MACD, Bollinger, Fibonacci."""
    closes = [x['close'] for x in ohlc]
    highs = [x['high'] for x in ohlc]
    lows = [x['low'] for x in ohlc]
    
    # Simple trend: compare recent highs/lows
    recent_highs = highs[-10:]
    recent_lows = lows[-10:]
    if recent_highs[-1] > max(recent_highs[:-1]) and recent_lows[-1] > max(recent_lows[:-1]):
        trend = 'UPTREND'
    elif recent_highs[-1] < min(recent_highs[:-1]) and recent_lows[-1] < min(recent_lows[:-1]):
        trend = 'DOWNTREND'
    else:
        trend = 'RANGING'
    
    # Swing low/high for stop loss
    swing_low = min(lows[-20:])
    swing_high = max(highs[-20:])
    
    # EMA 21
    ema21 = np.mean(closes[-21:]) if len(closes)>=21 else closes[-1]
    
    # RSI
    gains = []
    losses = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    avg_gain = np.mean(gains[-14:]) if gains else 0
    avg_loss = np.mean(losses[-14:]) if losses else 1
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    rsi = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = np.mean(closes[-12:])
    ema26 = np.mean(closes[-26:])
    macd = ema12 - ema26
    signal = np.mean([macd])  # simplified
    
    # Bollinger Bands
    sma20 = np.mean(closes[-20:])
    std = np.std(closes[-20:])
    upper = sma20 + 2*std
    lower = sma20 - 2*std
    
    # Fibonacci retracement levels from recent swing
    fib_levels = {}
    if swing_high > swing_low:
        diff = swing_high - swing_low
        fib_levels = {0.236: swing_low + 0.236*diff, 0.382: swing_low + 0.382*diff,
                      0.5: swing_low + 0.5*diff, 0.618: swing_low + 0.618*diff, 0.786: swing_low + 0.786*diff}
    
    return {
        'trend': trend,
        'swing_low': swing_low,
        'swing_high': swing_high,
        'ema21': ema21,
        'rsi': rsi,
        'macd': macd,
        'signal': signal,
        'bollinger_upper': upper,
        'bollinger_lower': lower,
        'fib_levels': fib_levels,
        'current_price': closes[-1]
    }

def assess_news(asset):
    """Check news sentiment. Return (impact, sentiment, headline)."""
    if not NEWS_API_KEY:
        return ('LOW', 'NEUTRAL', 'No news API key')
    
    # Extract keyword from asset (e.g., 'BTC' from 'BTC/USD')
    keyword = asset.split('/')[0]
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': keyword,
        'apiKey': NEWS_API_KEY,
        'pageSize': 5,
        'sortBy': 'publishedAt'
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        if data['status'] == 'ok' and data['articles']:
            headlines = [a['title'] for a in data['articles']]
            # Simple sentiment: look for bullish/bearish words
            bullish = ['surge', 'gain', 'bull', 'buy', 'positive']
            bearish = ['drop', 'fall', 'bear', 'sell', 'negative', 'hack', 'ban']
            score = 0
            for h in headlines:
                if any(w in h.lower() for w in bullish):
                    score += 1
                if any(w in h.lower() for w in bearish):
                    score -= 1
            if score > 0:
                sentiment = 'BULLISH'
            elif score < 0:
                sentiment = 'BEARISH'
            else:
                sentiment = 'NEUTRAL'
            impact = 'HIGH' if abs(score) >= 2 else 'MEDIUM'
            return (impact, sentiment, headlines[0][:100])
    except:
        pass
    return ('LOW', 'NEUTRAL', 'No recent news')

def calculate_dynamic_tp_sl(entry, direction, atr, swing_low, swing_high, duration_minutes, current_price):
    """
    Compute TP and SL with breathing room (1.5*ATR buffer) and duration adjustment.
    Returns (stop_loss, take_profit).
    """
    # Base SL using structure + ATR buffer
    if direction == 'LONG':
        base_sl = swing_low - 1.5 * atr
        # TP based on risk:reward 1:1.5
        base_tp = entry + (entry - base_sl) * 1.5
    else:
        base_sl = swing_high + 1.5 * atr
        base_tp = entry - (base_sl - entry) * 1.5
    
    # Adjust for duration: expected move = atr * sqrt(duration_minutes / 60) (if hourly ATR)
    # Assuming atr is based on 1h candles
    expected_move = atr * np.sqrt(duration_minutes / 60.0)
    
    # Ensure TP is not too far for the duration
    if direction == 'LONG':
        tp = min(base_tp, entry + expected_move)
    else:
        tp = max(base_tp, entry - expected_move)
    
    # Ensure SL is not too tight: at least 0.5*expected_move for breathing
    min_sl_distance = 0.5 * expected_move
    if direction == 'LONG':
        sl = min(base_sl, entry - min_sl_distance)
    else:
        sl = max(base_sl, entry + min_sl_distance)
    
    return round(sl, 4), round(tp, 4)

def calculate_leverage(atr_percent):
    """Dynamic leverage based on volatility."""
    if atr_percent < 0.5:
        return 20
    elif atr_percent < 1.0:
        return 10
    elif atr_percent < 2.0:
        return 5
    else:
        return 2

def probability_score(indicators, news_sentiment, news_impact):
    """Compute a score 0-100 based on confluence."""
    score = 50  # base
    # Trend alignment
    if indicators['trend'] == 'UPTREND':
        score += 10
    elif indicators['trend'] == 'DOWNTREND':
        score += 10
    # RSI (30-70)
    if 30 < indicators['rsi'] < 70:
        score += 5
    # MACD
    if indicators['macd'] > indicators['signal']:
        score += 5
    # Bollinger position (if price near lower band for long, near upper for short)
    # Simplified: assume we will decide direction later; for now just add if bands are wide
    if indicators['bollinger_upper'] - indicators['bollinger_lower'] > indicators['current_price'] * 0.05:
        score += 5
    # News
    if news_sentiment == 'BULLISH' or news_sentiment == 'BEARISH':
        score += 10
    if news_impact == 'HIGH':
        score += 5
    elif news_impact == 'MEDIUM':
        score += 2
    # Cap
    return min(100, score)

# ==================== TRADE GENERATION ====================
def generate_trade(asset, asset_type, duration_minutes, used_assets):
    """
    Generate a trade signal for a single asset.
    Returns dict or None if no trade.
    """
    # Skip if already used
    if asset in used_assets:
        return None
    
    # Fetch OHLC
    ohlc = get_ohlc(asset)
    if not ohlc:
        return None
    
    # Calculate indicators and ATR
    indicators = calculate_indicators(ohlc)
    atr = calculate_atr(ohlc)
    current_price = indicators['current_price']
    
    # News check
    news_impact, news_sentiment, headline = assess_news(asset)
    if news_impact == 'HIGH' and 'red folder' in headline.lower():  # simulate high impact news filter
        return None
    
    # Determine trade direction based on trend and indicators
    direction = None
    entry = current_price  # we would ideally wait for candle close; for simplicity use current
    if indicators['trend'] == 'UPTREND' and indicators['rsi'] < 70 and current_price > indicators['ema21']:
        direction = 'LONG'
    elif indicators['trend'] == 'DOWNTREND' and indicators['rsi'] > 30 and current_price < indicators['ema21']:
        direction = 'SHORT'
    else:
        # Additional confluence
        if current_price <= indicators['bollinger_lower'] * 1.02 and indicators['rsi'] < 40:
            direction = 'LONG'
        elif current_price >= indicators['bollinger_upper'] * 0.98 and indicators['rsi'] > 60:
            direction = 'SHORT'
    
    if not direction:
        return None
    
    # Calculate SL/TP
    sl, tp = calculate_dynamic_tp_sl(entry, direction, atr, 
                                      indicators['swing_low'], indicators['swing_high'],
                                      duration_minutes, current_price)
    
    # ATR percent for leverage
    atr_percent = (atr / current_price) * 100 if current_price else 1.0
    leverage = calculate_leverage(atr_percent)
    
    # Probability score
    prob = probability_score(indicators, news_sentiment, news_impact)
    
    # Build trade object
    trade = {
        'asset': asset,
        'type': asset_type,
        'direction': direction,
        'entry': round(entry, 4),
        'stop_loss': sl,
        'take_profit': tp,
        'leverage': leverage,
        'duration_minutes': duration_minutes,
        'probability': prob,
        'timestamp': datetime.now().isoformat(),
        'news': headline,
        'logic': f"{indicators['trend']} trend, RSI {indicators['rsi']:.1f}, ATR {atr:.2f}, SL placed with 1.5xATR buffer."
    }
    return trade

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    """Render main page."""
    # Initialize session for used assets and history
    if 'used_crypto' not in session:
        session['used_crypto'] = []
    if 'used_forex' not in session:
        session['used_forex'] = []
    if 'trade_history' not in session:
        session['trade_history'] = []
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate top 3 trades based on user selection."""
    data = request.get_json()
    asset_type = data.get('type', 'crypto')  # 'crypto' or 'forex'
    duration = data.get('duration', 60)      # minutes
    custom_duration = data.get('custom_duration')
    if custom_duration:
        try:
            duration = int(custom_duration)
        except:
            duration = 60
    
    # Select asset list and used set
    if asset_type == 'crypto':
        symbols = CRYPTO_SYMBOLS
        used_key = 'used_crypto'
    else:
        symbols = FOREX_SYMBOLS
        used_key = 'used_forex'
    
    used = session.get(used_key, [])
    
    # If all assets used, reset
    if len(used) >= len(symbols):
        used = []
        session[used_key] = used
    
    # Generate trades for each asset
    trades = []
    for sym in symbols:
        if sym in used:
            continue
        trade = generate_trade(sym, asset_type, duration, used)
        if trade:
            trades.append(trade)
    
    # Sort by probability descending, take top 3
    trades = sorted(trades, key=lambda x: x['probability'], reverse=True)[:3]
    
    # Update used assets
    for trade in trades:
        used.append(trade['asset'])
    session[used_key] = used
    
    # Add to history
    history = session.get('trade_history', [])
    history.extend(trades)
    # Keep last 50
    session['trade_history'] = history[-50:]
    
    return jsonify({'trades': trades})

@app.route('/history')
def history():
    """Return trade history."""
    return jsonify(session.get('trade_history', []))

@app.route('/heatmap')
def heatmap():
    """Return data for heatmap (top movers)."""
    # For simplicity, return top 10 gainers/losers from current prices
    crypto_prices = get_crypto_prices()
    # Calculate 24h change would require historical; mock
    heat = []
    for sym, price in crypto_prices.items():
        change = random.uniform(-10, 10)
        heat.append({
            'asset': sym,
            'price': price,
            'change': round(change, 2)
        })
    # Sort by absolute change
    heat = sorted(heat, key=lambda x: abs(x['change']), reverse=True)[:20]
    return jsonify(heat)

@app.route('/prices')
def prices():
    """Get current prices for all assets."""
    crypto = get_crypto_prices()
    forex = get_forex_prices()
    return jsonify({'crypto': crypto, 'forex': forex})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
