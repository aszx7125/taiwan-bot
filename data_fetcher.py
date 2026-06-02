import streamlit as st
import pandas as pd
import datetime
import urllib.parse
import xml.etree.ElementTree as ET
import concurrent.futures
import cloudscraper
from indicators import add_advanced_indicators, add_intraday_indicators

stealth_scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
)

def fetch_yahoo_robust(symbol, period="6mo", interval="1d"):
    # 🚀 動態支援多時區請求
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?range={period}&interval={interval}"
    try:
        res = stealth_scraper.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            res_data = data.get('chart', {}).get('result', [])
            if res_data:
                timestamps = res_data[0].get('timestamp', [])
                quote = res_data[0].get('indicators', {}).get('quote', [{}])[0]
                if not timestamps or not quote: return pd.DataFrame()
                df = pd.DataFrame({
                    'Open': quote.get('open', []),
                    'High': quote.get('high', []),
                    'Low': quote.get('low', []),
                    'Close': quote.get('close', []),
                    'Volume': quote.get('volume', [])
                }, index=pd.to_datetime(timestamps, unit='s', utc=True))
                df.index = df.index.tz_convert('Asia/Taipei').tz_localize(None)
                return df.dropna(subset=['Close', 'Volume'])
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=3600*12)
def load_all_market_tickers():
    try: return pd.read_csv("all_tw_stocks.csv")
    except: return pd.DataFrame()

@st.cache_data(ttl=120)
def get_market_index_data():
    return fetch_yahoo_robust("^TWII", "6mo", "1d")

@st.cache_data(ttl=120)
def get_market_summary():
    indices = {"加權指數": "^TWII", "櫃買指數": "^TWOTC", "台灣50": "0050.TW"}
    res = {}
    for name, t in indices.items():
        df = fetch_yahoo_robust(t, "5d", "1d")
        if len(df) >= 2: 
            p_now, p_prev = df['Close'].iloc[-1], df['Close'].iloc[-2]
            res[name] = {"price": p_now, "change": p_now - p_prev, "pct": ((p_now - p_prev) / p_prev) * 100}
    return res

def get_precalculated_market_ret():
    market_df = get_market_index_data()
    if not market_df.empty:
        market_df.index = pd.to_datetime(market_df.index).tz_localize(None).normalize()
        market_df = market_df[~market_df.index.duplicated(keep='last')]
        return market_df['Close'].pct_change(20)
    return None

@st.cache_data(ttl=30) 
def get_kline_with_fugle(ticker_code, fugle_api_key=""):
    market_ret_20 = get_precalculated_market_ret()
    clean_ticker = ticker_code.split('.')[0]
    
    # 1. 抓取巨觀日線 (Daily)
    df_daily = fetch_yahoo_robust(f"{clean_ticker}.TW", "6mo", "1d")
    actual_symbol = f"{clean_ticker}.TW"
    if df_daily.empty or len(df_daily) < 20:
        df_daily = fetch_yahoo_robust(f"{clean_ticker}.TWO", "6mo", "1d")
        actual_symbol = f"{clean_ticker}.TWO"

    # 2. 抓取微觀小時線 (Hourly) - 限制 1個月 避免 Yahoo 阻擋
    df_hourly = pd.DataFrame()
    if not df_daily.empty:
        df_hourly = fetch_yahoo_robust(actual_symbol, "1mo", "1h")

    if df_daily.empty or len(df_daily) < 20: 
        return pd.DataFrame(), pd.DataFrame(), actual_symbol 

    # Fugle 即時資料輔助更新日線最後一根 K 棒
    if fugle_api_key:
        try:
            res = stealth_scraper.get(f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{clean_ticker}", headers={"X-API-KEY": fugle_api_key}, timeout=3)
            if res.status_code == 200:
                data = res.json()
                rt_price = data.get('closePrice') or data.get('lastTrade', {}).get('price')
                rt_vol = data.get('total', {}).get('tradeVolume', 0)
                tz_tw = datetime.timezone(datetime.timedelta(hours=8))
                today_date, last_candle_date = datetime.datetime.now(tz_tw).date(), df_daily.index[-1].date()
                if last_candle_date < today_date and rt_price:
                    new_row = df_daily.iloc[-1].copy()
                    new_row.name = pd.Timestamp(today_date)
                    df_daily = pd.concat([df_daily, pd.DataFrame([new_row])])
                if rt_price: df_daily.iloc[-1, df_daily.columns.get_loc('Close')] = float(rt_price)
                if rt_vol > 0: df_daily.iloc[-1, df_daily.columns.get_loc('Volume')] = float(rt_vol)
        except Exception:
            pass
        
    df_daily = add_advanced_indicators(df_daily, market_ret_20)
    df_hourly = add_intraday_indicators(df_hourly)
    
    return df_daily, df_hourly, actual_symbol

@st.cache_data(ttl=300)
def get_stock_news(keyword):
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(keyword)}+when:7d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        root = ET.fromstring(stealth_scraper.get(url, timeout=4).text)
        return [{"title": i.find('title').text, "link": i.find('link').text, "date": i.find('pubDate').text} for i in root.findall('./channel/item')[:5]]
    except: return []

@st.cache_data(ttl=300)
def get_macro_news():
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote('台股 OR 聯準會 OR 財報')}+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        root = ET.fromstring(stealth_scraper.get(url, timeout=4).text)
        return [{"title": i.find('title').text, "link": i.find('link').text, "date": i.find('pubDate').text} for i in root.findall('./channel/item')[:6]]
    except: return []

def _fetch_and_score_sync(symbol, market_ret_20, conds, names_dict, mode):
    try:
        df_daily = fetch_yahoo_robust(symbol, "6mo", "1d")
        if df_daily.empty or len(df_daily) < 35: return None
        
        df_daily = add_advanced_indicators(df_daily, market_ret_20)
        if 'Score' not in df_daily.columns: return None
        
        last = df_daily.iloc[-1]
        code = symbol.split('.')[0]
        name = names_dict.get(code, "市場焦點")
        
        score = int(last.get('Score', 0)) if pd.notna(last.get('Score')) else 0
        rs_val = float(last.get('RS_Index', 0)) if pd.notna(last.get('RS_Index')) else 0.0
        rsi_val = float(last.get('RSI', 50)) if pd.notna(last.get('RSI')) else 50.0
        
        liq_sweep = bool(last.get('Liquidity_Sweep_Bull', False))
        low_vol_pb = bool(last.get('Low_Vol_Pullback', False))
        broker_conc = float(last.get('Broker_Concentration', 0.0))
        
        pattern_list = []
        if low_vol_pb: pattern_list.append("📉 量縮回踩")
        if broker_conc > 0.3: pattern_list.append("🏦 分點囤貨")
        if liq_sweep: pattern_list.append("🧹 破底翻")
        if df_daily['Squeeze_On'].iloc[-2] and last['Close'] > last['BB_Upper']: pattern_list.append("💥 壓縮突破")
        
        if not pattern_list: pattern_list.append("多頭" if last['Close'] > last['SMA_20'] else "空頭")
        pattern_str = " + ".join(pattern_list[:3]) 
        
        if mode == "score":
            return {"代號": code, "名稱": name, "量化總分": score, "機構籌碼/型態": pattern_str, "大盤相對強度": f"{rs_val*100:+.2f}%", "現價": f"{last['Close']:.2f}"}
    except: pass
    return None

def run_robust_market_scan(tickers, conds, p_bar, s_text, names_dict, market_data_raw, mode="score"):
    results = []
    total = len(tickers)
    completed = 0
    market_ret_20 = get_precalculated_market_ret()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        future_to_ticker = {executor.submit(_fetch_and_score_sync, t, market_ret_20, conds, names_dict, mode): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                p_bar.progress(min(completed / total, 1.0))
                s_text.text(f"🚀 AI 演算法極速運算中... ({completed}/{total})")
            try:
                res = future.result()
                if res: results.append(res)
            except: pass
    return results