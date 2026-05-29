import streamlit as st
import requests
import pandas as pd
import datetime
import urllib.parse
import xml.etree.ElementTree as ET
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from indicators import add_advanced_indicators

# 🛡️ 建立穩健的 HTTP Session，帶有自動重試機制與強勢偽裝
retry_strategy = Retry(
    total=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
    backoff_factor=0.5
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=50, pool_maxsize=50)

general_session = requests.Session()
general_session.mount("https://", adapter)
general_session.mount("http://", adapter)
general_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
})

def fetch_yahoo_robust(symbol, period="6mo"):
    """
    🚀 終極直連爬蟲：徹底放棄 yfinance 套件。
    使用強制 timeout=3，保證 3 秒內一定有回應，絕對不會死結卡死。
    """
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?range={period}&interval=1d"
    try:
        res = general_session.get(url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            res_data = data.get('chart', {}).get('result', [])
            if res_data:
                timestamps = res_data[0].get('timestamp', [])
                quote = res_data[0].get('indicators', {}).get('quote', [{}])[0]
                
                if not timestamps or not quote: 
                    return pd.DataFrame()
                    
                df = pd.DataFrame({
                    'Open': quote.get('open', []),
                    'High': quote.get('high', []),
                    'Low': quote.get('low', []),
                    'Close': quote.get('close', []),
                    'Volume': quote.get('volume', [])
                }, index=pd.to_datetime(timestamps, unit='s', utc=True))
                
                df.index = df.index.tz_convert('Asia/Taipei').tz_localize(None).normalize()
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
    return fetch_yahoo_robust("^TWII", "6mo")

@st.cache_data(ttl=120)
def get_market_summary():
    indices = {"加權指數": "^TWII", "櫃買指數": "^TWOTC", "台灣50": "0050.TW"}
    res = {}
    for name, t in indices.items():
        df = fetch_yahoo_robust(t, "5d")
        if len(df) >= 2: 
            p_now, p_prev = df['Close'].iloc[-1], df['Close'].iloc[-2]
            res[name] = {"price": p_now, "change": p_now - p_prev, "pct": ((p_now - p_prev) / p_prev) * 100}
    return res

@st.cache_data(ttl=30) 
def get_kline_with_fugle(ticker_code, fugle_api_key=""):
    market_df = get_market_index_data()
    clean_ticker = ticker_code.split('.')[0]
    
    # 依序尋訪上市與上櫃
    df = fetch_yahoo_robust(f"{clean_ticker}.TW")
    actual_symbol = f"{clean_ticker}.TW"
    if df.empty or len(df) < 20:
        df = fetch_yahoo_robust(f"{clean_ticker}.TWO")
        actual_symbol = f"{clean_ticker}.TWO"

    if df.empty or len(df) < 20: 
        return pd.DataFrame(), actual_symbol 

    # Fugle 零延遲報價整合
    if fugle_api_key:
        try:
            res = general_session.get(f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{clean_ticker}", headers={"X-API-KEY": fugle_api_key}, timeout=2)
            if res.status_code == 200:
                data = res.json()
                rt_price = data.get('closePrice') or data.get('lastTrade', {}).get('price')
                rt_vol = data.get('total', {}).get('tradeVolume', 0)
                
                tz_tw = datetime.timezone(datetime.timedelta(hours=8))
                today_date, last_candle_date = datetime.datetime.now(tz_tw).date(), df.index[-1].date()
                
                if last_candle_date < today_date and rt_price:
                    new_row = df.iloc[-1].copy()
                    new_row.name = pd.Timestamp(today_date)
                    df = pd.concat([df, pd.DataFrame([new_row])])
                    
                if rt_price: df.iloc[-1, df.columns.get_loc('Close')] = float(rt_price)
                if rt_vol > 0: df.iloc[-1, df.columns.get_loc('Volume')] = float(rt_vol)
        except Exception:
            pass
        
    df = add_advanced_indicators(df, market_df)
    return df, actual_symbol

@st.cache_data(ttl=300)
def get_stock_news(keyword):
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(keyword)}+when:7d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        root = ET.fromstring(general_session.get(url, timeout=5).text)
        return [{"title": i.find('title').text, "link": i.find('link').text, "date": i.find('pubDate').text} for i in root.findall('./channel/item')[:5]]
    except: return []

@st.cache_data(ttl=300)
def get_macro_news():
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote('台股 OR 聯準會 OR 財報')}+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        root = ET.fromstring(general_session.get(url, timeout=5).text)
        return [{"title": i.find('title').text, "link": i.find('link').text, "date": i.find('pubDate').text} for i in root.findall('./channel/item')[:6]]
    except: return []

def _fetch_and_score_sync(symbol, market_df, conds, names_dict, mode):
    try:
        df = fetch_yahoo_robust(symbol)
        if df.empty or len(df) < 35: return None
        
        df = add_advanced_indicators(df, market_df)
        if 'Score' not in df.columns: return None
        
        last = df.iloc[-1]
        code = symbol.split('.')[0]
        name = names_dict.get(code, "市場焦點")
        
        score = int(last.get('Score', 0)) if pd.notna(last.get('Score')) else 0
        rs_val = float(last.get('RS_Index', 0)) if pd.notna(last.get('RS_Index')) else 0.0
        rsi_val = float(last.get('RSI', 50)) if pd.notna(last.get('RSI')) else 50.0
        chip_trend = int(last.get('Smart_Money_Trend', 0)) if pd.notna(last.get('Smart_Money_Trend')) else 0
        wt_up = bool(last.get('Weekly_Trend_Up', False)) if pd.notna(last.get('Weekly_Trend_Up')) else False
        
        bull_div = bool(last.get('Bullish_Div', False))
        bear_div = bool(last.get('Bearish_Div', False))
        
        pattern_list = []
        if df['Squeeze_On'].iloc[-2] and last['Close'] > last['BB_Upper']: pattern_list.append("💥 壓縮突破")
        elif last['Squeeze_On']: pattern_list.append("🛡️ 區間收斂")
        
        if bull_div: pattern_list.append("🟢 RSI底背離")
        elif bear_div: pattern_list.append("🚨 RSI頂背離")
        
        if not pattern_list: pattern_list.append("多頭" if last['Close'] > last['SMA_20'] else "空頭")
        pattern_str = " + ".join(pattern_list)
        
        chip_status = "👽 大戶連買" if chip_trend >= 1 else ("🚶 散戶接盤" if chip_trend <= -1 else "⚖️ 無方向")

        if mode == "radar":
            f_vol = (last['Volume'] > last['Vol_SMA5'] * 1.5) if conds.get('vol') else True
            f_ma = (last['Close'] > last['SMA_20']) if conds.get('ma') else True
            f_rsi = (last['RSI'] < 35) if conds.get('rsi') else True
            f_macd = (last['MACD'] > last['Signal'] and df['MACD'].iloc[-2] <= df['Signal'].iloc[-2]) if conds.get('macd') else True
            
            if f_vol and f_ma and f_rsi and f_macd:
                pct = ((last['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                vol_ratio = last['Volume'] / last['Vol_SMA5'] if last['Vol_SMA5'] > 0 else 1.0
                return {"代號": code, "名稱": name, "現價": f"{last['Close']:.2f}", "今日漲跌": f"{pct:+.2f}%", "量比": f"{vol_ratio:.1f}x", "RSI": f"{rsi_val:.1f}", "型態特徵": pattern_str}
        
        elif mode == "score":
            return {"代號": code, "名稱": name, "量化總分": score, "籌碼/型態": f"{chip_status} | {pattern_str}", "大盤相對強度": f"{rs_val*100:+.2f}%", "現價": f"{last['Close']:.2f}", "週線趨勢": "🟢 共振" if wt_up else "🔴 壓制"}
    except: pass
    return None

def run_robust_market_scan(tickers, conds, p_bar, s_text, names_dict, market_df, mode="radar"):
    results = []
    total = len(tickers)
    completed = 0
    # 策略雷達因為不會卡到 Streamlit 的 cache，使用執行緒是安全的
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_ticker = {executor.submit(_fetch_and_score_sync, t, market_df, conds, names_dict, mode): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                p_bar.progress(min(completed / total, 1.0))
                s_text.text(f"🚀 量化核心運算中... ({completed}/{total})")
            try:
                res = future.result()
                if res: results.append(res)
            except: pass
    return results