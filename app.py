import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import datetime
import urllib.parse
import xml.etree.ElementTree as ET

st.set_page_config(page_title="台股專業決策終端", layout="wide")

# --- 設定：股票字典 ---
stock_names = {"2330": "台積電", "2454": "聯發科", "2303": "聯電", "3034": "聯詠", "2382": "廣達", "3231": "緯創", "2881": "富邦金", "2886": "兆豐金", "2603": "長榮", "3491": "昇達科", "3105": "穩懋"}
stock_clusters = {
    "半導體": ["2330.TW", "2454.TW", "2303.TW", "3034.TW", "5347.TWO", "3105.TWO"],
    "伺服器": ["2382.TW", "3231.TW", "6669.TW"],
    "金融股": ["2881.TW", "2886.TW", "2891.TW"],
    "傳產": ["1101.TW", "2603.TW", "2609.TW"],
    "低軌衛星": ["3491.TWO", "3138.TW"]
}

# --- 核心邏輯引擎 (恢復詳細分析功能) ---
@st.cache_data(ttl=600)
def get_full_analysis(ticker_code):
    symbols = [f"{ticker_code}.TW", f"{ticker_code}.TWO"]
    df = pd.DataFrame()
    actual_symbol = ""
    for s in symbols:
        df = yf.Ticker(s).history(period="6mo")
        if not df.empty:
            actual_symbol = s
            break
    if df.empty: return None, None, None
    
    # 運算邏輯
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['ATR'] = df[['High', 'Low', 'Close']].apply(lambda x: max(x['High'] - x['Low'], abs(x['High'] - df['Close'].shift(1).loc[x.name])), axis=1).rolling(14).mean()
    df['Res_20'] = df['High'].shift(1).rolling(20).max()
    df['Sup_20'] = df['Low'].shift(1).rolling(20).min()
    
    # 缺口
    prev, curr = df.iloc[-2], df.iloc[-1]
    gap = "缺口已補" if (curr['Low'] < prev['Close'] < curr['High']) else ("向上缺口" if curr['Low'] > prev['High'] else ("向下缺口" if curr['High'] < prev['Low'] else "無缺口"))
    
    return df, gap, actual_symbol

# --- 新聞獲取 ---
@st.cache_data(ttl=1800)
def get_news(symbol):
    try:
        news = yf.Ticker(symbol).news
        return news[:5]
    except: return []

# --- 邏輯控制 ---
if 'target_ticker' not in st.session_state: st.session_state.target_ticker = None

# --- 主畫面模式 ---
if st.session_state.target_ticker is None:
    st.title("📊 台股專業決策監控平台")
    selected_cluster = st.selectbox("選擇產業群組", list(stock_clusters.keys()))
    
    data_list = []
    for t in stock_clusters[selected_cluster]:
        code = t.split('.')[0]
        res = get_full_analysis(code)
        if res and res[0] is not None:
            df, gap, _ = res
            data_list.append({"代號": code, "名稱": stock_names.get(code, "未知"), "現價": round(df['Close'].iloc[-1], 2), "缺口狀態": gap, "20MA": round(df['SMA_20'].iloc[-1], 2)})
    
    df_view = pd.DataFrame(data_list)
    st.dataframe(df_view.style.map(lambda x: 'background-color: #d4edda' if x == '缺口已補' else '', subset=['缺口狀態']), use_container_width=True)
    
    manual_input = st.text_input("搜尋代號進行詳細診斷")
    if st.button("執行診斷"):
        st.session_state.target_ticker = manual_input.upper()
        st.rerun()

# --- 詳細診斷模式 ---
else:
    df, gap, symbol = get_full_analysis(st.session_state.target_ticker)
    if df is not None:
        st.title(f"🧬 {st.session_state.target_ticker} 深度分析")
        tab1, tab2 = st.tabs(["📊 技術測幅與策略", "📰 個股新聞"])
        
        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("當前現價", f"{df['Close'].iloc[-1]:.1f}")
            c2.metric("型態缺口", gap)
            c3.metric("ATR 風險值", f"{df['ATR'].iloc[-1]:.1f}")
            
            # 完整策略推演
            st.write(f"### 🎯 科學測幅與壓力")
            target = round(df['Res_20'].iloc[-1] + df['ATR'].iloc[-1], 1)
            st.info(f"技術壓力: {df['Res_20'].iloc[-1]:.1f} | 預期目標: **{target}**")
            st.write(f"支撐防守: {df['Sup_20'].iloc[-1]:.1f}")
            st.warning("策略建議：依照 ATR 計算停利停損點，突破壓力位可順勢佈局。")

        with tab2:
            st.write("### 🌍 最新新聞")
            news = get_news(symbol)
            for n in news: st.markdown(f"- [{n['title']}]({n['link']})")
        
        if st.button("⬅️ 返回儀表板"):
            st.session_state.target_ticker = None
            st.rerun()
    else:
        st.error("找不到代號")
        if st.button("返回"):
            st.session_state.target_ticker = None
            st.rerun()
