import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

st.set_page_config(page_title="台股專業決策終端", layout="wide")

# --- 資料對應表 ---
stock_names = {"2330": "台積電", "2454": "聯發科", "2303": "聯電", "3034": "聯詠", "2382": "廣達", "3231": "緯創", "2881": "富邦金", "2886": "兆豐金", "2603": "長榮", "3491": "昇達科"}
stock_clusters = {"半導體": ["2330.TW", "2454.TW", "2303.TW", "3034.TW"], "伺服器": ["2382.TW", "3231.TW"], "金融股": ["2881.TW", "2886.TW"], "傳產": ["2603.TW"], "低軌衛星": ["3491.TWO"]}

# --- 詳細診斷邏輯核心 (恢復旗艦版功能) ---
@st.cache_data(ttl=600)
def get_full_analysis(ticker_code):
    symbol = f"{ticker_code}.TW" if not ticker_code.endswith(".TWO") else ticker_code
    df = yf.Ticker(symbol).history(period="6mo")
    if df.empty: return None
    
    # 計算指標
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['ATR'] = df[['High', 'Low', 'Close']].apply(lambda x: max(x['High'] - x['Low'], abs(x['High'] - df['Close'].shift(1).loc[x.name])), axis=1).rolling(14).mean()
    df['Res_20'] = df['High'].shift(1).rolling(20).max()
    df['Sup_20'] = df['Low'].shift(1).rolling(20).min()
    
    # 缺口狀態
    prev, curr = df.iloc[-2], df.iloc[-1]
    gap = "缺口已補" if (curr['Low'] < prev['Close'] < curr['High']) else ("向上缺口" if curr['Low'] > prev['High'] else ("向下缺口" if curr['High'] < prev['Low'] else "無缺口"))
    
    return df, gap

# --- UI 邏輯 ---
if 'target_ticker' not in st.session_state: st.session_state.target_ticker = None

if st.session_state.target_ticker is None:
    st.title("📊 產業監控儀表板")
    selected_cluster = st.selectbox("選擇群組", list(stock_clusters.keys()))
    
    data_list = []
    for t in stock_clusters[selected_cluster]:
        code = t.split('.')[0]
        res = get_full_analysis(code)
        if res:
            df, gap = res
            data_list.append({"代號": code, "名稱": stock_names.get(code, "未知"), "現價": round(df['Close'].iloc[-1], 2), "缺口狀態": gap})
    
    df_view = pd.DataFrame(data_list)
    st.dataframe(df_view.style.map(lambda x: 'background-color: #d4edda' if x == '缺口已補' else '', subset=['缺口狀態']), use_container_width=True)
    
    if st.button("診斷單股"):
        st.session_state.target_ticker = st.text_input("輸入代號")
        st.rerun()

else:
    df, gap = get_full_analysis(st.session_state.target_ticker)
    st.title(f"🧬 {st.session_state.target_ticker} 深度診斷報告")
    
    # 恢復之前的詳細分析介面
    col1, col2 = st.columns(2)
    with col1:
        st.metric("當前現價", f"{df['Close'].iloc[-1]:.1f}")
        st.write(f"均線支撐 (20MA): {df['SMA_20'].iloc[-1]:.1f}")
        st.write(f"型態缺口: **{gap}**")
    with col2:
        st.write(f"科學防守 (ATR): {df['ATR'].iloc[-1]:.1f}")
        st.write(f"預估目標價 (測幅): {round(df['Res_20'].iloc[-1] + df['ATR'].iloc[-1], 1)}")
        
    st.info("策略推演：均線呈現多頭，具備技術面優勢。若跌破月線請嚴守停損。")
    
    if st.button("⬅️ 返回儀表板"):
        st.session_state.target_ticker = None
        st.rerun()
