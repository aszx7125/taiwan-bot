import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import datetime
import urllib.parse
import xml.etree.ElementTree as ET
import asyncio
import aiohttp
from indicators import add_advanced_indicators

yf_session = requests.Session()
yf_session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

@st.cache_data(ttl=3600*12)
def load_all_market_tickers():
    try:
        df = pd.read_csv("all_tw_stocks.csv")
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def get_market_index_data():
    try:
        df = yf.Ticker("^TWII", session=yf_session).history(period="6mo")
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def get_market_summary():
    indices = {"加權指數": "^TWII", "櫃買指數": "^TWOTC", "台灣50": "0050.TW"}
    summary_data = {}
    for name, ticker in indices.items():
        try:
            df = yf.Ticker(ticker, session=yf_session).history(period="2d")
            if len(df) >= 2:
                p_now, p_prev = df['Close'].iloc[-1], df['Close'].iloc[-2]
                summary_data[name] = {"price": p_now, "change": p_now - p_prev, "pct": ((p_now - p_prev) / p_prev) * 100}
        except: pass
    return summary_data

@st.cache_data(ttl=30) 
def get_kline_with_fugle(ticker_code, fugle_api_key=""):
    market_df = get_market_index_data()
    symbols_to_try = [f"{ticker_code}.TW", f"{ticker_code}.TWO"]
    df, actual_symbol = pd.DataFrame(), ""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for symbol in symbols_to_try:
            try:
                temp_df = yf.Ticker(symbol, session=yf_session).history(period="6mo")
                if not temp_df.empty: 
                    df, actual_symbol = temp_df, symbol
                    break 
            except: pass

    if df.empty or len(df) < 20: return df, actual_symbol 

    if fugle_api_key:
        try:
            res = requests.get(f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{ticker_code}", headers={"X-API-KEY": fugle_api_key}, timeout=3)
            if res.status_code == 200:
                data = res.json()
                rt_price = data.get('closePrice') or data.get('lastTrade', {}).get('price')
                rt_vol = data.get('total', {}).get('tradeVolume', 0)
                tz_tw = datetime.timezone(datetime.timedelta(hours=8))
                today_date, last_candle_date = datetime.datetime.now(tz_tw).date(), df.index[-1].date()
                if last_candle_date < today_date and rt_price:
                    new_row = df.iloc[-1].copy()
                    new_row.name = pd.Timestamp(today_date, tz=df.index.tz)
                    df = pd.concat([df, pd.DataFrame([new_row])])
                if rt_price:
                    df.iloc[-1, df.columns.get_loc('Close')] = rt_price
                    if data.get('highPrice'): df.iloc[-1, df.columns.get_loc('High')] = data['highPrice']
                    if data.get('lowPrice'): df.iloc[-1, df.columns.get_loc('Low')] = data['lowPrice']
                if rt_vol > 0: df.iloc[-1, df.columns.get_loc('Volume')] = rt_vol
        except: pass
        
    df = add_advanced_indicators(df, market_df)
    return df, actual_symbol

@st.cache_data(ttl=300)
def get_stock_news(keyword):
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(keyword)}+when:7d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        root = ET.fromstring(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5).text)
        return [{"title": i.find('title').text, "link": i.find('link').text, "date": i.find('pubDate').text} for i in root.findall('./channel/item')[:5]]
    except: return []

@st.cache_data(ttl=300)
def get_macro_news():
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote('台股 OR 聯準會 OR 財報')}+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        root = ET.fromstring(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5).text)
        return [{"title": i.find('title').text, "link": i.find('link').text, "date": i.find('pubDate').text} for i in root.findall('./channel/item')[:6]]
    except: return []

async def fetch_yahoo_history(session, symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=90d&interval=1d"
    try:
        async with session.get(url, timeout=6) as response:
            if response.status == 200:
                data = await response.json()
                res = data.get('chart', {}).get('result', [])
                if res:
                    adj_close = res[0].get('indicators', {}).get('adjclose', [{}])[0].get('adjclose', [])
                    closes = res[0].get('indicators', {}).get('quote', [{}])[0].get('close', [])
                    volumes = res[0].get('indicators', {}).get('quote', [{}])[0].get('volume', [])
                    highs = res[0].get('indicators', {}).get('quote', [{}])[0].get('high', [])
                    lows = res[0].get('indicators', {}).get('quote', [{}])[0].get('low', [])
                    
                    final_closes = adj_close if adj_close else closes
                    valid_idx = [i for i, c in enumerate(final_closes) if c is not None and i < len(volumes) and volumes[i] is not None]
                    
                    if len(valid_idx) >= 35:
                        df = pd.DataFrame({
                            'Close': [final_closes[i] for i in valid_idx],
                            'Volume': [volumes[i] for i in valid_idx],
                            'High': [highs[i] for i in valid_idx],
                            'Low': [lows[i] for i in valid_idx]
                        })
                        return symbol, df
    except: pass
    return symbol, None

async def async_scan_market(tickers_to_scan, c_vol, c_ma, c_rsi, c_macd, progress_bar, status_text, names_dict, market_df):
    results = []
    connector = aiohttp.TCPConnector(limit=60) 
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_yahoo_history(session, t) for t in tickers_to_scan]
        completed, total = 0, len(tasks)
        for future in asyncio.as_completed(tasks):
            symbol, df = await future
            completed += 1
            if completed % 15 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"🚀 量化核心異步高頻掃描中... ({completed}/{total})")

            if df is not None:
                df = add_advanced_indicators(df, market_df)
                if len(df) < 5: continue
                today = df.iloc[-1]
                
                # 策略控制閥開關判定
                f_vol = (today['Volume'] > today['Vol_SMA5'] * 1.5) if c_vol else True
                f_ma = (today['Close'] > today['SMA_20']) if c_ma else True
                f_rsi = (today['RSI'] < 35) if c_rsi else True
                f_macd = (today['MACD'] > today['Signal'] and df['MACD'].iloc[-2] <= df['Signal'].iloc[-2]) if c_macd else True
                
                if f_vol and f_ma and f_rsi and f_macd:
                    pct = ((today['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                    code = symbol.split('.')[0]
                    results.append({
                        "代號": code, "名稱": names_dict.get(code, "市場焦點"), 
                        "現價": f"{today['Close']:.2f}", "今日漲跌": f"{pct:+.2f}%", 
                        "量比": f"{today['Volume']/today['Vol_SMA5']:.1f}x", "RSI": f"{today['RSI']:.1f}",
                        "AI綜合評分": int(today['Score']),
                        "相對強弱度": f"{today['RS_Index']*100:+.1f}%",
                        "型態特徵": "💥 波動臨界突破" if (df['Squeeze_On'].iloc[-2] and today['Close'] > today['BB_Upper']) else ("🛡️ 區間收斂" if today['Squeeze_On'] else "📈 趨勢多頭")
                    })
    return results