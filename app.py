import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

st.set_page_config(page_title="台股專業決策終端", layout="wide")

# --- 設定：股票字典 ---
# 確保代號與名稱對應正確
stock_names = {
    "2330": "台積電", "2454": "聯發科", "2303": "聯電", "3034": "聯詠", "5347": "世界先進",
    "2382": "廣達", "3231": "緯創", "6669": "緯穎", "2376": "技嘉", "3017": "奇鋐",
    "2881": "富邦金", "2882": "國泰金", "2886": "兆豐金", "2891": "中信金",
    "1101": "台泥", "2002": "中鋼", "2603": "長榮", "2609": "陽明",
    "3491": "昇達科", "3138": "耀登", "6285": "啟碁"
}

# --- 數據核心 (強化版) ---
@st.cache_data(ttl=600)
def get_full_analysis(ticker_code):
    symbol = f"{ticker_code}.TW" if not ticker_code.endswith(".TWO") else ticker_code
    df = yf.Ticker(symbol).history(period="6mo")
    if df.empty: return None
    
    # 邏輯計算
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['ATR'] = df[['High', 'Low', 'Close']].apply(lambda x: max(x['High'] - x['Low'], abs(x['High'] - df['Close'].shift(1).loc[x.name])), axis=1).rolling(14).mean()
    
    # 缺口判斷
    prev, curr = df.iloc[-2], df.iloc[-1]
    gap = "缺口已補" if (curr['Low'] < prev['Close'] < curr['High']) else ("向上缺口" if curr['Low'] > prev['High'] else ("向下缺口" if curr['High'] < prev['Low'] else "無缺口"))
    
    return df, gap

# --- 儀表板邏輯 ---
if 'target_ticker' not in st.session_state: st.session_state.target_ticker = None

if st.session_state.target_ticker is None:
    st.title("📊 產業監控儀表板")
    selected_cluster = st.selectbox("選擇監控群組", list(stock_clusters.keys()))
    
    data_list = []
    for t in stock_clusters[selected_cluster]:
        code = t.split('.')[0]
        df, gap = get_full_analysis(code)
        if df is not None:
            data_list.append({
                "代號": code,
                "名稱": stock_names.get(code, "未知"),
                "現價": round(df['Close'].iloc[-1], 2),
                "漲跌幅%": round(((df['Close'].iloc[-1] - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100, 2),
                "缺口狀態": gap
            })
    
    df_view = pd.DataFrame(data_list)
    # 條件式顏色標示：缺口已補變綠色
    st.dataframe(df_view.style.map(lambda x: 'background-color: #d4edda' if x == '缺口已補' else '', subset=['缺口狀態']), use_container_width=True)
    
    manual = st.text_input("輸入代號診斷")
    if st.button("執行診斷"):
        st.session_state.target_ticker = manual
        st.rerun()

else:
    # 診斷頁面邏輯
    st.title(f"🔍 {st.session_state.target_ticker} 深度診斷")
    df, _ = get_full_analysis(st.session_state.target_ticker)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("當前價格", f"{df['Close'].iloc[-1]:.1f}")
        st.write(f"均線支撐: {df['SMA_20'].iloc[-1]:.1f}")
    with col2:
        st.write(f"波動風險 (ATR): {df['ATR'].iloc[-1]:.1f}")
    
    if st.button("⬅️ 返回列表"):
        st.session_state.target_ticker = None
        st.rerun()
