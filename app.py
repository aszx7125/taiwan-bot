# app.py
import streamlit as st
from config import STOCK_CLUSTERS, STOCK_NAMES # 匯入設定
from data_fetcher import get_kline_data        # 匯入爬蟲工具

st.set_page_config(page_title="台股量化旗艦終端", layout="wide")

st.title("⚡ 台股戰情分析終端")

# --- 側邊欄 UI ---
with st.sidebar:
    selected_cluster = st.selectbox("選擇群組", list(STOCK_CLUSTERS.keys()))
    if st.button("啟動雷達"):
        st.session_state.scan_trigger = True

# --- 主畫面 UI 渲染 ---
if 'scan_trigger' in st.session_state and st.session_state.scan_trigger:
    st.write(f"正在掃描群組: {selected_cluster}")
    
    for ticker in STOCK_CLUSTERS[selected_cluster]:
        code = ticker.split('.')[0]
        # 直接呼叫我們寫好的模組工具
        df, _ = get_kline_data(code) 
        
        if not df.empty:
            st.metric(label=STOCK_NAMES.get(code, code), value=round(df['Close'].iloc[-1], 2))