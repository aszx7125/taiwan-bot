import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import datetime
import time
import urllib.parse
import xml.etree.ElementTree as ET

# --- 頁面配置 ---
st.set_page_config(page_title="台股量化決策平台", layout="wide")

# --- 設定：股票字典與清單 ---
stock_names = {"2330": "台積電", "2454": "聯發科", "2303": "聯電", "3034": "聯詠", "2382": "廣達", "3231": "緯創", "2881": "富邦金", "2886": "兆豐金", "2603": "長榮", "3491": "昇達科", "3105": "穩懋"}
stock_clusters = {
    "半導體": ["2330.TW", "2454.TW", "2303.TW", "3034.TW", "5347.TWO", "3105.TWO"],
    "伺服器": ["2382.TW", "3231.TW", "6669.TW"],
    "金融股": ["2881.TW", "2886.TW", "2891.TW"],
    "傳產": ["1101.TW", "2603.TW", "2609.TW"],
    "低軌衛星": ["3491.TWO", "3138.TW"]
}

# --- 核心數據引擎 (整合邏輯) ---
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
    if df.empty: return None, None
    
    # 指標運算
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['ATR'] = df[['High', 'Low', 'Close']].apply(lambda x: max(x['High'] - x['Low'], abs(x['High'] - df['Close'].shift(1).loc[x.name])), axis=1).rolling(14).mean()
    df['Res_20'] = df['High'].shift(1).rolling(20).max()
    df['Sup_20'] = df['Low'].shift(1).rolling(20).min()
    
    # 缺口與突破狀態
    prev, curr = df.iloc[-2], df.iloc[-1]
    if curr['Low'] > prev['High']: gap = "向上缺口"
    elif curr['High'] < prev['Low']: gap = "向下缺口"
    elif curr['Low'] < prev['Close'] < curr['High']: gap = "缺口已補"
    else: gap = "無缺口"
    
    return df, gap

# --- UI 邏輯控制 ---
if 'target_ticker' not in st.session_state: st.session_state.target_ticker = None

# --- 主畫面：儀表板模式 ---
if st.session_state.target_ticker is None:
    st.title("📊 台股專業決策監控平台")
    selected_cluster = st.selectbox("選擇產業群組", list(stock_clusters.keys()))
    
    data_list = []
    for t in stock_clusters[selected_cluster]:
        code = t.split('.')[0]
        res = get_full_analysis(code)
        if res and res[0] is not None:
            df, gap = res
            data_list.append({
                "代號": code, "名稱": stock_names.get(code, "未知"), 
                "現價": round(df['Close'].iloc[-1], 2),
                "缺口狀態": gap, "20MA": round(df['SMA_20'].iloc[-1], 2)
            })
    
    df_view = pd.DataFrame(data_list)
    # 表格視覺化
    st.dataframe(df_view.style.map(lambda x: 'background-color: #d4edda' if x == '缺口已補' else '', subset=['缺口狀態']), use_container_width=True)
    
    st.markdown("---")
    manual_input = st.text_input("搜尋代號進行詳細診斷")
    if st.button("執行詳細診斷"):
        st.session_state.target_ticker = manual_input.upper()
        st.rerun()

# --- 診斷模式 ---
else:
    df, gap = get_full_analysis(st.session_state.target_ticker)
    if df is not None:
        st.title(f"🧬 {st.session_state.target_ticker} {stock_names.get(st.session_state.target_ticker, '')} 深度診斷")
        
        tab1, tab2 = st.tabs(["📊 技術測幅與策略", "📰 財報新聞"])
        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("當前現價", f"{df['Close'].iloc[-1]:.1f}")
            c2.metric("型態缺口", gap)
            c3.metric("ATR 風險值", f"{df['ATR'].iloc[-1]:.1f}")
            
            st.write(f"### 🎯 測幅目標與壓力")
            target = round(df['Res_20'].iloc[-1] + df['ATR'].iloc[-1], 1)
            st.info(f"技術面壓力位: {df['Res_20'].iloc[-1]:.1f} | 突破預期目標: **{target}**")
            
        with tab2:
            st.write("### 🌍 最新新聞動態")
            # 整合新聞邏輯...
            
        if st.button("⬅️ 返回監控儀表板"):
            st.session_state.target_ticker = None
            st.rerun()
    else:
        st.error("找不到代號，請重新輸入")
        if st.button("返回"):
            st.session_state.target_ticker = None
            st.rerun()
