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

# 全域連線設定
yf_session = requests.Session()
yf_session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

@st.cache_data(ttl=3600*24)
def load_all_market_tickers():
    """讀取全市場 CSV 資料"""
    try:
        df = pd.read_csv("all_tw_stocks.csv")
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def get_market_summary():
    """獲取大盤摘要"""
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

@st.cache_data(ttl=60) 
def get_kline_with_fugle(ticker_code, fugle_api_key=""):
    """獲取單一個股歷史與即時資料"""
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
        
    df = add_advanced_indicators(df)
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
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=60d&interval=1d"
    try:
        async with session.get(url, timeout=5) as response:
            if response.status == 200:
                data = await response.json()
                result = data.get('chart', {}).get('result', [])
                if result:
                    indicators = result[0].get('indicators', {}).get('quote', [{}])[0]
                    closes, volumes = indicators.get('close', []), indicators.get('volume', [])
                    closes, volumes = [c for c in closes if c is not None], [v for v in volumes if v is not None]
                    if len(closes) >= 30:
                        df = pd.DataFrame({'Close': closes, 'Volume': volumes})
                        df = add_advanced_indicators(df)
                        return symbol, df
    except: pass
    return symbol, None

async def async_scan_market(tickers_to_scan, cond_vol, cond_ma, cond_rsi, cond_macd, progress_bar, status_text, stock_names_dict):
    results = []
    connector = aiohttp.TCPConnector(limit=50) 
    async with aiohttp.ClientSession(connector=connector, headers={"User-Agent": "Mozilla/5.0"}) as session:
        tasks = [fetch_yahoo_history(session, t) for t in tickers_to_scan]
        completed, total = 0, len(tasks)
        for future in asyncio.as_completed(tasks):
            symbol, df = await future
            completed += 1
            if completed % 10 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"🚀 光速異步掃描中... ({completed}/{total})")

            if df is not None:
                c_close, c_vol = df['Close'].iloc[-1], df['Volume'].iloc[-1]
                sma20, vol_sma5 = df['SMA_20'].iloc[-1], df['Vol_SMA5'].iloc[-1]
                rsi, macd, signal = df['RSI'].iloc[-1], df['MACD'].iloc[-1], df['Signal'].iloc[-1]
                macd_prev, signal_prev = df['MACD'].iloc[-2], df['Signal'].iloc[-2]
                
                pass_vol = (c_vol > vol_sma5 * 1.5) if cond_vol else True
                pass_ma = (c_close > sma20) if cond_ma else True
                pass_rsi = (rsi < 35) if cond_rsi else True
                pass_macd = (macd > signal and macd_prev <= signal_prev) if cond_macd else True
                
                if pass_vol and pass_ma and pass_rsi and pass_macd:
                    pct = ((c_close - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                    code = symbol.split('.')[0]
                    results.append({
                        "代號": code, "名稱": stock_names_dict.get(code, "大盤個股"), 
                        "現價": f"{c_close:.2f}", "今日漲跌": f"{pct:+.2f}%", 
                        "量比": f"{c_vol/vol_sma5:.1f}x" if vol_sma5>0 else "-", "RSI": f"{rsi:.1f}"
                    })
    return results