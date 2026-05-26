import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import datetime
import time
import urllib.parse
import xml.etree.ElementTree as ET

# --- 頁面配置 ---
st.set_page_config(page_title="台股量化決策平台", page_icon="📊", layout="wide")

# --- 設定與初始化 ---
try: FUGLE_API_KEY = st.secrets["FUGLE_API_KEY"]
except: FUGLE_API_KEY = ""

yf_session = requests.Session()
yf_session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})

stock_clusters = {
    "半導體": ["2330.TW", "2454.TW", "2303.TW", "5347.TWO", "3034.TW"],
    "伺服器": ["2382.TW", "3231.TW", "6669.TW", "2376.TW", "3017.TW"],
    "金融股": ["2881.TW", "2882.TW", "2886.TW", "2891.TW"],
    "傳統產業": ["1101.TW", "2002.TW", "2603.TW", "2609.TW"],
    "低軌衛星": ["3491.TWO", "3138.TW", "6285.TW"]
}

# --- 核心數據獲取邏輯 ---
@st.cache_data(ttl=600)
def get_kline_with_fugle(ticker_code):
    symbols = [f"{ticker_code}.TW", f"{ticker_code}.TWO"]
    df = pd.DataFrame()
    actual_symbol = ""
    for s in symbols:
        df = yf.Ticker(s, session=yf_session).history(period="6mo")
        if not df.empty:
            actual_symbol = s
            break
    
    if df.empty or len(df) < 20: return df, actual_symbol
    
    # 基礎技術指標運算
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    df['Res_20'] = df['High'].shift(1).rolling(window=20).max()
    df['Sup_20'] = df['Low'].shift(1).rolling(window=20).min()
    df['ATR'] = df[['High', 'Low', 'Close']].apply(lambda x: max(x['High'] - x['Low'], abs(x['High'] - df['Close'].shift(1).loc[x.name])), axis=1).rolling(window=14).mean()
    
    # 缺口狀態
    prev, curr = df.iloc[-2], df.iloc[-1]
    if curr['Low'] > prev['High']: df.loc[df.index[-1], 'Gap'] = "向上缺口"
    elif curr['High'] < prev['Low']: df.loc[df.index[-1], 'Gap'] = "向下缺口"
    elif curr['Low'] < prev['Close'] < curr['High']: df.loc[df.index[-1], 'Gap'] = "缺口已補"
    else: df.loc[df.index[-1], 'Gap'] = "無缺口"
    
    return df, actual_symbol

# --- 全球新聞引擎 ---
@st.cache_data(ttl=1800)
def get_macro_news():
    url = "https://news.google.com/rss/search?q=台股+OR+財報&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.text)
        return [{"title": i.find('title').text, "link": i.find('link').text} for i in root.findall('./channel/item')[:5]]
    except: return []

# --- 主程式控制 ---
if 'target_ticker' not in st.session_state: st.session_state.target_ticker = None

# 儀表板視圖
if st.session_state.target_ticker is None:
    st.sidebar.header("📂 產業監控")
    selected_cluster = st.sidebar.selectbox("選擇群組", list(stock_clusters.keys()))
    st.title(f"📊 {selected_cluster} 監控儀表板")
    
    # 構建監控表格
    table_data = []
    for t in stock_clusters[selected_cluster]:
        df, _ = get_kline_with_fugle(t.split('.')[0])
        if not df.empty:
            last = df.iloc[-1]
            table_data.append({"代號": t.split('.')[0], "現價": last['Close'], "缺口狀態": last['Gap']})
    
    df_view = pd.DataFrame(table_data)
    # 條件式顏色標示
    st.dataframe(df_view.style.map(lambda x: 'background-color: #d4edda' if x == '缺口已補' else '', subset=['缺口狀態']), use_container_width=True)
    
    # 搜尋區
    manual_input = st.text_input("輸入代號進行詳細診斷")
    if st.button("執行診斷"):
        st.session_state.target_ticker = manual_input.upper()
        st.rerun()

# 詳細診斷視圖
else:
    df, actual_symbol = get_kline_with_fugle(st.session_state.target_ticker)
    if not df.empty:
        st.subheader(f"🧬 {st.session_state.target_ticker} 診斷報告")
        tab1, tab2 = st.tabs(["📊 技術測幅分析", "📰 最新財報新聞"])
        with tab1:
            st.metric("現價", f"{df['Close'].iloc[-1]:.1f}")
            st.write(f"壓力位: {df['Res_20'].iloc[-1]:.1f} | 預測目標: {round(df['Res_20'].iloc[-1] + df['ATR'].iloc[-1], 1)}")
        with tab2:
            news = yf.Ticker(actual_symbol).news
            for n in news[:5]: st.markdown(f"- [{n['title']}]({n['link']})")
    
    if st.button("⬅️ 返回監控儀表板"):
        st.session_state.target_ticker = None
        st.rerun()
