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
yf_session.headers.update({"User-Agent": "Mozilla/5.0"})

@st.cache_data(ttl=3600*12)
def load_all_market_tickers():
    try: return pd.read_csv("all_tw_stocks.csv")
    except: return pd.DataFrame()

@st.cache_data(ttl=120)
def get_market_index_data():
    try: return yf.Ticker("^TWII", session=yf_session).history(period="6mo")
    except: return pd.DataFrame()

@st.cache_data(ttl=120)
def get_market_summary():
    indices = {"加權指數": "^TWII", "櫃買指數": "^TWOTC", "台灣50": "0050.TW"}
    res = {}
    for name, t in indices.items():
        try:
            df = yf.Ticker(t, session=yf_session).history(period="2d")
            if len(df) >= 2: res[name] = {"price": df['Close'].iloc[-1], "change": df['Close'].iloc[-1] - df['Close'].iloc[-2], "pct": ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100}
        except: pass
    return res

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
                temp = yf.Ticker(symbol, session=yf_session).history(period="6mo")
                if not temp.empty: df, actual_symbol = temp, symbol; break 
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
                if rt_vol > 0: 
                    df.iloc[-1, df.columns.get_loc('Volume')] = rt_vol
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

# ⚡ 安全的異步底層引擎 (加入 DataFrame 安全檢核)
async def _async_fetch_and_score(session, symbol, market_df, conds, names_dict, p_bar, s_text, total, counter, mode):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=90d&interval=1d"
    try:
        async with session.get(url, timeout=6) as response:
            if response.status == 200:
                data = await response.json()
                res = data.get('chart', {}).get('result', [])
                if res:
                    adj = res[0].get('indicators', {}).get('adjclose', [{}])[0].get('adjclose', [])
                    c = res[0].get('indicators', {}).get('quote', [{}])[0].get('close', [])
                    v = res[0].get('indicators', {}).get('quote', [{}])[0].get('volume', [])
                    h = res[0].get('indicators', {}).get('quote', [{}])[0].get('high', [])
                    l = res[0].get('indicators', {}).get('quote', [{}])[0].get('low', [])
                    
                    fc = adj if adj else c
                    v_idx = [i for i, val in enumerate(fc) if val is not None and i < len(v) and v[i] is not None]
                    
                    if len(v_idx) >= 35:
                        df = pd.DataFrame({'Close': [fc[i] for i in v_idx], 'Volume': [v[i] for i in v_idx], 'High': [h[i] for i in v_idx], 'Low': [l[i] for i in v_idx]})
                        df = add_advanced_indicators(df, market_df)
                        
                        # 🛡️ 核心防護：如果資料太少導致沒算出 Score，直接跳過，避免 KeyError 崩潰
                        if 'Score' not in df.columns: return None
                        
                        counter[0] += 1
                        if counter[0] % 15 == 0 or counter[0] == total:
                            p_bar.progress(counter[0] / total)
                            s_text.text(f"🚀 量化運算中... ({counter[0]}/{total})")

                        last = df.iloc[-1]
                        code = symbol.split('.')[0]
                        name = names_dict.get(code, "市場焦點")
                        pct = ((last['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                        vol_ratio = last['Volume'] / last['Vol_SMA5'] if last['Vol_SMA5'] > 0 else 1.0
                        squeeze_status = "💥 臨界突破" if (df['Squeeze_On'].iloc[-2] and last['Close'] > last['BB_Upper']) else ("🛡️ 區間收斂" if last['Squeeze_On'] else "📈 趨勢多頭")

                        if mode == "radar":
                            f_vol = (last['Volume'] > last['Vol_SMA5'] * 1.5) if conds.get('vol') else True
                            f_ma = (last['Close'] > last['SMA_20']) if conds.get('ma') else True
                            f_rsi = (last['RSI'] < 35) if conds.get('rsi') else True
                            f_macd = (last['MACD'] > last['Signal'] and df['MACD'].iloc[-2] <= df['Signal'].iloc[-2]) if conds.get('macd') else True
                            
                            if f_vol and f_ma and f_rsi and f_macd:
                                return {"代號": code, "名稱": name, "現價": f"{last['Close']:.2f}", "今日漲跌": f"{pct:+.2f}%", "量比": f"{vol_ratio:.1f}x", "RSI": f"{last['RSI']:.1f}", "型態特徵": squeeze_status}
                        
                        elif mode == "score":
                            rs_val = last['RS_Index'] if pd.notna(last['RS_Index']) else 0.0
                            rsi_val = last['RSI'] if pd.notna(last['RSI']) else 50.0
                            return {"代號": code, "名稱": name, "量化總分": int(last['Score']), "相對大盤強度": f"{rs_val*100:+.2f}%", "現價": f"{last['Close']:.2f}", "RSI": round(rsi_val, 1), "週線趨勢": "🟢 多頭共振" if last['Weekly_Trend_Up'] else "🔴 趨勢壓制"}
    except: pass
    return None

async def _main_async_runner(tickers, conds, p_bar, s_text, names_dict, market_df, mode):
    connector = aiohttp.TCPConnector(limit=50)
    counter = [0]
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_async_fetch_and_score(session, t, market_df, conds, names_dict, p_bar, s_text, len(tickers), counter, mode) for t in tickers]
        results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

def run_async_market_scan(tickers, conds, p_bar, s_text, names_dict, market_df, mode="radar"):
    try: loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    return new_loop.run_until_complete(_main_async_runner(tickers, conds, p_bar, s_text, names_dict, market_df, mode))