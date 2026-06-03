import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random 
import concurrent.futures
import requests
import json
import os

from config import get_fugle_key, DEFAULT_CLUSTERS, DEFAULT_NAMES, INDUSTRY_CHAINS
from data_fetcher import (
    load_all_market_tickers, get_market_index_data, get_market_summary, 
    get_kline_with_fugle, get_stock_news, get_macro_news, run_robust_market_scan, 
    get_precalculated_market_ret, fetch_yahoo_robust
)

st.set_page_config(page_title="台股量化旗艦終端", page_icon="📈", layout="wide")

if 'stock_clusters' not in st.session_state: st.session_state.stock_clusters = DEFAULT_CLUSTERS.copy()
if 'stock_names' not in st.session_state: st.session_state.stock_names = DEFAULT_NAMES.copy()

FUGLE_API_KEY = get_fugle_key()
csv_df = load_all_market_tickers()
if not csv_df.empty:
    for index, row in csv_df.iterrows():
        code = str(row['Ticker']).split('.')[0]
        if code not in st.session_state.stock_names: st.session_state.stock_names[code] = str(row['Name'])

def load_market_snapshot():
    if os.path.exists("market_snapshot.json"):
        try:
            with open("market_snapshot.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# 🧠 將全新的 LightGBM 大腦鎖進常駐記憶體，避免重複讀取硬碟榨乾效能
@st.cache_resource
def get_ai_model():
    if os.path.exists("quant_model.joblib") and os.path.exists("model_features.joblib"):
        try:
            import joblib
            return joblib.load("quant_model.joblib"), joblib.load("model_features.joblib")
        except: pass
    return None, None

@st.cache_data(ttl=3600*6) 
def fetch_and_calculate_backtest(holding_period=5, threshold=60):
    try:
        from supabase import create_client
        
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            try:
                url = st.secrets["SUPABASE_URL"]
                key = st.secrets["SUPABASE_KEY"]
            except: pass
            
        if not url or not key: return {"status": "no_key"}
        
        supabase = create_client(url, key)
        all_data = []
        offset, limit = 0, 1000
        while True:
            res = supabase.table("quant_history").select("date, ticker, close_price, score").range(offset, offset+limit-1).execute()
            if not res.data: break
            all_data.extend(res.data)
            offset += limit

        if not all_data: return {"status": "empty"}
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)

        df[f'future_close_{holding_period}d'] = df.groupby('ticker')['close_price'].shift(-holding_period)
        df[f'return_{holding_period}d'] = (df[f'future_close_{holding_period}d'] - df['close_price']) / df['close_price']

        signals = df[(df['score'] >= threshold) & (df[f'future_close_{holding_period}d'].notna())].copy()
        total_signals = len(signals)
        
        if total_signals == 0:
            pending = len(df[(df['score'] >= threshold) & (df[f'future_close_{holding_period}d'].isna())])
            return {"status": "pending", "pending_count": pending}

        signals['is_win'] = signals[f'return_{holding_period}d'] > 0
        win_rate = len(signals[signals['is_win']]) / total_signals
        avg_win = signals[signals['is_win']][f'return_{holding_period}d'].mean() if len(signals[signals['is_win']]) > 0 else 0
        avg_loss = signals[~signals['is_win']][f'return_{holding_period}d'].mean() if len(signals[~signals['is_win']]) > 0 else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return {
            "status": "ready",
            "total": total_signals,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}

with st.sidebar:
    st.header("📂 我的自選清單")
    selected_cluster = st.selectbox("1. 選擇產業群組", list(st.session_state.stock_clusters.keys()))
    cluster_stocks = st.session_state.stock_clusters[selected_cluster]
    display_options = [f"{t.split('.')[0]} {st.session_state.stock_names.get(t.split('.')[0], '')}".strip() for t in cluster_stocks]
    sidebar_ticker = st.selectbox("2. 選擇分析標的", display_options).split(' ')[0]
    st.write("") 
    if st.button("📊 診斷此自選股", use_container_width=True, type="primary"):
        st.session_state.analyze_trigger = sidebar_ticker 
        st.rerun()
    st.markdown("---")
    
    if FUGLE_API_KEY: st.success("🟢 零延遲即時引擎已啟動")
    else: st.warning("🟡 目前使用 Yahoo 延遲報價")
    st.markdown("---")
    
    with st.expander("🛠️ 管理自選群組", expanded=False):
        new_cluster_name = st.text_input("群組名稱")
        new_cluster_tickers = st.text_area("股票代號 (逗號分隔)")
        if st.button("➕ 儲存配置"):
            if new_cluster_name and new_cluster_tickers:
                st.session_state.stock_clusters[new_cluster_name] = [t if '.TW' in t or '.TWO' in t else f"{t}.TW" for t in new_cluster_tickers.split(',')]
                st.success("配置成功！"); st.rerun()

st.title("⚡ 台股戰情分析終端")
col1, col2 = st.columns([3, 1])
with col1: manual_ticker = st.text_input("輸入股票代號", "", label_visibility="collapsed")
with col2: analyze_manual_btn = st.button("單股掃描", use_container_width=True)
st.markdown("---")

target_ticker = st.session_state.pop('analyze_trigger', None) or (manual_ticker.strip().upper() if analyze_manual_btn else None)

if target_ticker:
    base_ticker = target_ticker.split('.')[0]
    c_name = st.session_state.stock_names.get(base_ticker, target_ticker)
    
    with st.spinner(f"正在分析 {target_ticker}... 多維度時區運算與 AI 預測中"):
        df_daily, df_hourly, actual_symbol = get_kline_with_fugle(target_ticker, FUGLE_API_KEY)
        
        if df_daily.empty or len(df_daily) < 40: 
            st.error("❌ 該標的數據深度不足，無法執行複雜演算法。")
        else:
            news_s = get_stock_news(c_name)
            news_m = get_macro_news()

            today, yesterday = df_daily.iloc[-1], df_daily.iloc[-2]
            
            try:
                entry_price = float(today.get('Close', 0.0))
                t_high = float(today.get('High', entry_price))
                t_low = float(today.get('Low', entry_price))
                y_close = float(yesterday.get('Close', entry_price))
                
                if pd.isna(entry_price): entry_price = 0.0
                if pd.isna(y_close) or y_close == 0: y_close = entry_price if entry_price > 0 else 1.0
                
                vol_sma5 = float(today.get('Vol_SMA5', 1.0))
                if pd.isna(vol_sma5) or vol_sma5 <= 0: vol_sma5 = 1.0
                vol_ratio = float(today.get('Volume', 0.0)) / vol_sma5
                p_change = ((entry_price - y_close) / y_close) * 100
                
                res_level = float(today.get('Res_20', entry_price * 1.05))
                sup_level = float(today.get('Sup_20', entry_price * 0.95))
                if pd.isna(res_level): res_level = entry_price * 1.05
                if pd.isna(sup_level): sup_level = entry_price * 0.95
                box_height = max(res_level - sup_level, 0.01)
                
                atr_14 = float(yesterday.get('ATR_14', entry_price * 0.05))
                if pd.isna(atr_14) or atr_14 == 0: atr_14 = entry_price * 0.05
                
                recent_20_df = df_daily.iloc[-21:-1]
                res_tests = int(len(recent_20_df[recent_20_df['High'] >= res_level * 0.985])) if not recent_20_df.empty else 0
                sup_tests = int(len(recent_20_df[recent_20_df['Low'] <= sup_level * 1.015])) if not recent_20_df.empty else 0
                    
                ai_score = int(today.get('Score', 0))
                rs_index = float(today.get('RS_Index', 0.0))
                broker_conc = float(today.get('Broker_Concentration', 0.0))
                if pd.isna(rs_index): rs_index = 0.0
                if pd.isna(broker_conc): broker_conc = 0.0
            except Exception as e:
                entry_price, y_close, t_high, t_low, vol_ratio, p_change = 0.0, 1.0, 0.0, 0.0, 1.0, 0.0
                res_level, sup_level, box_height, atr_14 = 0.0, 0.0, 0.0, 0.0
                res_tests, sup_tests, ai_score, rs_index, broker_conc = 0, 0, 0, 0.0, 0.0
            
            bull_div = bool(today.get('Bullish_Div', False))
            bear_div = bool(today.get('Bearish_Div', False))
            div_status = "🟢 底背離 (空頭力竭，準備反轉)" if bull_div else ("🚨 頂背離 (多頭力竭，注意回檔)" if bear_div else "無顯著背離")

            liq_sweep = bool(today.get('Liquidity_Sweep_Bull', False))
            low_vol_pb = bool(today.get('Low_Vol_Pullback', False))
            squeeze_on = bool(today.get('Squeeze_On', False))
            
            smc_status = []
            if low_vol_pb: smc_status.append("📉 量縮回踩")
            if squeeze_on: smc_status.append("🛡️ 區間極度壓縮")
            if liq_sweep: smc_status.append("🌊 流動性掠奪")
            smc_text = " + ".join(smc_status) if smc_status else "一般常態震盪"
            
            block_trade = bool(today.get('Block_Trade_Inflow', False))
            
            breakout_status = "區間震盪 (未突破)"
            if entry_price > res_level: breakout_status = "🚀 向上突破前高"
            elif entry_price < sup_level: breakout_status = "⚠️ 向下摜破前低"
            elif entry_price >= res_level * 0.98: breakout_status = "⚔️ 兵臨城下 (挑戰前高)"
            elif entry_price <= sup_level * 1.02: breakout_status = "🛡️ 支撐保衛戰 (回測前低)"

            atr_stop = entry_price - (1.5 * atr_14)
            structural_stop = sup_level * 0.985 
            stop_loss = round(min(atr_stop, structural_stop), 2)
            if entry_price > res_level: stop_loss = round(res_level * 0.985, 2)
            risk_per_share = max(entry_price - stop_loss, 0.01)

            if entry_price > res_level:
                take_profit = round(res_level + box_height, 2)
                profit_reason = "🚀 噴發目標：等距測幅擴展位"
            elif low_vol_pb or liq_sweep or bull_div:
                take_profit = round(res_level, 2)
                profit_reason = "🎯 潛伏目標：前高/箱頂壓力區"
            else:
                take_profit = round(res_level + (atr_14 * 1.0), 2)
                profit_reason = "⚔️ 波段目標：前高波動擴張位"
                
            real_rr_ratio = round((take_profit - entry_price) / risk_per_share, 2)
            
            micro_trigger = False
            micro_status_text = "數據不足"
            if not df_hourly.empty and len(df_hourly) >= 2:
                last_hour = df_hourly.iloc[-1]
                micro_trigger = bool(last_hour.get('Micro_Sniper_Trigger', False))
                h_macd_cross = bool(last_hour.get('MACD_Cross_Up', False))
                h_vol_surge = bool(last_hour.get('Vol_Surge_1h', False))
                
                if micro_trigger: micro_status_text = "🔥 帶量突破 1h 均線 (強烈買點)"
                elif h_macd_cross: micro_status_text = "📈 1h MACD 金叉發動"
                elif h_vol_surge: micro_status_text = "🌊 1h 微觀異常爆量"
                elif last_hour['Close'] > last_hour.get('SMA_20_1h', entry_price): micro_status_text = "🟢 站穩 1h 均線 (短線強勢)"
                else: micro_status_text = "⚪ 1h 均線下弱勢震盪"

            ai_win_rate_str = "等待 AI 訓練"
            ai_recommendation = "⏸️ 勝率偏低或追高風險，強制觀望"
            box_color = "#555555"

            # 🚀 從常駐快取讀取全新的 LightGBM 大腦 (完美傳遞 7 大特徵)
            model, features = get_ai_model()
            if model:
                try:
                    input_data = {}
                    input_data['is_pullback'] = 1 if low_vol_pb else 0
                    input_data['is_sweep'] = 1 if liq_sweep else 0
                    input_data['is_squeeze'] = 1 if squeeze_on else 0
                    input_data['is_divergence'] = 1 if bull_div else 0
                    input_data['rs_index'] = float(rs_index) if not pd.isna(rs_index) else 0.0
                    
                    # 🚀 即時動態計算股性波動率與規模特徵，餵給 LightGBM 交叉審查
                    input_data['volatility'] = float(atr_14 / entry_price) if entry_price > 0 else 0.0
                    input_data['turnover'] = float(entry_price * today.get('Volume', 0)) if 'Volume' in today else 0.0
                    
                    # 確保按照模型訓練時的 7 大特徵順序排列
                    input_df = pd.DataFrame([input_data], columns=features).fillna(0)
                    win_prob = float(model.predict_proba(input_df)[0][1])
                    ai_win_rate_str = f"{win_prob * 100:.1f}%"
                    
                    if win_prob > 0.60 and real_rr_ratio >= 1.5:
                        ai_recommendation = "⭐⭐⭐ 極致期望值！(高勝率 + 高風報比)"
                        box_color = "#00cc96"
                    elif win_prob > 0.50 and real_rr_ratio >= 1.0:
                        ai_recommendation = "⭐⭐ 溫和佈局 (具備正向期望值)"
                        box_color = "#ffc107"
                    elif win_prob <= 0.50:
                        ai_recommendation = "⚠️ 預測敗率較高，建議嚴格觀望"
                        box_color = "#555555"
                    else:
                        ai_recommendation = "⏸️ 風報比不足，防守空間過窄"
                        box_color = "#555555"
                except Exception as e:
                    pass

            st.subheader(f"🧬 {target_ticker} {c_name} 多時區量化診斷報告")
            
            st.markdown(f"""
            <div style="border: 2px solid {box_color}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;">
                <h4 style="color: {box_color}; margin-top: 0;">🎯 AI 深度學習 x 結構價格 戰術計畫</h4>
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 180px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">1. AI 真實勝率預測</span><br>
                        <b style="font-size: 24px; color: {box_color};">{ai_win_rate_str}</b><br>
                        <span style="font-size: 14px; font-weight: bold;">{ai_recommendation}</span>
                    </div>
                    <div style="flex: 1; min-width: 130px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">2. 建議進場價</span><br>
                        <b style="font-size: 22px;">{entry_price:.2f}</b><br>
                        <span style="font-size: 12px; color: gray;">(現價/限價單)</span>
                    </div>
                    <div style="flex: 1; min-width: 200px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">3. 結構停利點</span><br>
                        <b style="font-size: 22px; color: #00cc96;">{take_profit:.2f}</b><br>
                        <span style="font-size: 12px; color: #00cc96; font-weight: bold;">{profit_reason}</span><br>
                        <span style="font-size: 12px; color: gray;">(實況風報比 1 : {real_rr_ratio})</span>
                    </div>
                    <div style="flex: 1; min-width: 130px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">4. 嚴格防守價</span><br>
                        <b style="font-size: 22px; color: #ff4b4b;">{stop_loss:.2f}</b><br>
                        <span style="font-size: 12px; color: gray;">(跌破無條件停損)</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("當前現價", f"{entry_price:.2f}", f"{p_change:+.2f}%")
            m2.metric("日K 巨觀潛伏分數", f"{ai_score} 分")
            m3.metric("1h 小時區微觀狀態", micro_status_text)
            m4.metric("機構囤貨集中度", f"{broker_conc*100:.1f}%")
            
            st.markdown("---")
            t1, t2, t3, t4 = st.tabs(["🧱 多時區結構解析", "🔍 策略回測對撞", "🏦 大單流與分點籌碼", "📰 專屬新聞動態"])
            
            with t1:
                c_l, c_r = st.columns(2)
                with c_l:
                    st.markdown("#### 📐 日K與1h共振狀態")
                    st.write(f"- **前高壓力 (近20日):** {res_level:.2f}")
                    st.write(f"- **前低支撐 (近20日):** {sup_level:.2f}")
                    st.markdown(f"- **大週期(日K)潛伏型態:** <span style='color:{'#ffc107' if low_vol_pb or squeeze_on or liq_sweep else 'gray'}; font-weight:bold;'>{'量縮回踩/極度壓縮' if low_vol_pb or squeeze_on else '破底翻' if liq_sweep else '常態震盪'}</span>", unsafe_allow_html=True)
                    st.markdown(f"- **小週期(1h)板機狀態:** <span style='color:{'#ff4b4b' if micro_trigger else '#00cc96' if '站穩' in micro_status_text else 'gray'}; font-weight:bold;'>{micro_status_text}</span>", unsafe_allow_html=True)
                with c_r:
                    st.markdown("#### 💡 三重濾網實戰解析")
                    if ai_score >= 65 and micro_trigger:
                        st.error("🎯 **【狙擊點確認】** 日K線大格局處於安全潛伏區，且小時區突然爆發突破訊號！這是極高勝率的多時區共振買點，建議立刻佈局。")
                    elif ai_score >= 65 and not micro_trigger:
                        st.success("👀 **【耐心埋伏】** 大格局非常漂亮，勝率極高，但小時區動能尚未發動。您可以先建立基本底倉，等待小時區金叉再加碼。")
                    elif ai_score < 50 and micro_trigger:
                        st.warning("🚨 **【假突破陷阱】** 小時區雖然大漲，但大週期(日K)不支援，追高套牢風險極大！建議嚴格觀望。")
                    else: 
                        st.info("⏸️ **【動能休眠】** 大小週期皆無明顯發動跡象，屬於機構垃圾時間。")
            
            with t2:
                st.markdown("### 🔍 多週期策略回測與實況對撞")
                sub_t1, sub_t2, sub_t3 = st.tabs(["1️⃣ 昨日對撞 (1日)", "5️⃣ 一週波段 (5日)", "🈷️ 單月波段 (20日)"])
                
                with sub_t1:
                    y_target = res_level + atr_14
                    col_r1, col_r2 = st.columns(2)
                    with col_r1: st.info(f"**昨日盤後預測基準**\n- 壓力位: **{res_level:.2f}**\n- 支撐位: **{sup_level:.2f}**\n- 測幅目標: **{y_target:.2f}**")
                    with col_r2: st.warning(f"**今日實況極值**\n- 最高價: **{t_high:.2f}**\n- 最低價: **{t_low:.2f}**\n- 收盤現價: **{entry_price:.2f}**")
                
                with sub_t2:
                    if len(df_daily) >= 26:
                        d_base = df_daily.iloc[-6]
                        d_5_days = df_daily.iloc[-5:]
                        try:
                            w_res = float(d_base.get('Res_20', entry_price))
                            w_sup = float(d_base.get('Sup_20', entry_price))
                            w_target = w_res + float(d_base.get('ATR_14', 1.0))
                            max_h_5d = float(d_5_days['High'].max())
                            min_l_5d = float(d_5_days['Low'].min())
                            
                            col_w1, col_w2 = st.columns(2)
                            with col_w1: st.info(f"**5 天前預測基準**\n- 當時壓力: **{w_res:.2f}**\n- 當時支撐: **{w_sup:.2f}**\n- 測幅目標: **{w_target:.2f}**")
                            with col_w2: st.warning(f"**本週實況極值 (近5日)**\n- 波段最高: **{max_h_5d:.2f}**\n- 波段最低: **{min_l_5d:.2f}**\n- 目前收盤: **{entry_price:.2f}**")
                        except: st.warning("⚠️ 此區間資料存在空值，無法計算。")
                    else:
                        st.warning("⚠️ 數據不足，無法進行一週歷史回測。")

                with sub_t3:
                    if len(df_daily) >= 45:
                        d_base_m = df_daily.iloc[-21] 
                        d_20_days = df_daily.iloc[-20:]
                        try:
                            m_res = float(d_base_m.get('Res_20', entry_price))
                            m_sup = float(d_base_m.get('Sup_20', entry_price))
                            m_target = m_res + float(d_base_m.get('ATR_14', 1.0))
                            max_h_20d = float(d_20_days['High'].max())
                            min_l_20d = float(d_20_days['Low'].min())
                            
                            col_m1, col_m2 = st.columns(2)
                            with col_m1: st.info(f"**20 天前預測基準**\n- 當時壓力: **{m_res:.2f}**\n- 當時支撐: **{m_sup:.2f}**\n- 測幅目標: **{m_target:.2f}**")
                            with col_m2: st.warning(f"**本月實況極值 (近20日)**\n- 波段最高: **{max_h_20d:.2f}**\n- 波段最低: **{min_l_20d:.2f}**\n- 目前收盤: **{entry_price:.2f}**")
                        except: st.warning("⚠️ 此區間資料存在空值，無法計算。")
                    else:
                        st.warning("⚠️ 歷史數據深度不足，無法進行單月回測。")
            
            with t3:
                st.markdown("#### 🏦 微觀結構：特大單與贏家分點籌碼追蹤")
                c_flow1, c_flow2 = st.columns(2)
                with c_flow1:
                    st.markdown("##### 🌊 盤中特大單淨流入")
                    if today.get('Block_Trade_Inflow', False): st.success("🚨 **異常大單狂敲**\n今日成交量大，價格強勢推升，機構不計代價掃貨。")
                    else: st.info("📉 **無明顯大單流入**\n量能平穩，屬自然換手。")
                with c_flow2:
                    st.markdown("##### 🏦 贏家分點集中囤貨")
                    if broker_conc > 0.3: st.success(f"🔥 **高度集中 ({broker_conc*100:.1f}%)**\n近 5 日資金異常淨流入，極可能有主力分點暗中大舉囤貨！")
                    elif broker_conc > 0: st.warning(f"🔍 **溫和吃貨 ({broker_conc*100:.1f}%)**\n籌碼緩步集中。")
                    else: st.info(f"⚖️ **籌碼發散 ({broker_conc*100:.1f}%)**\n近期無特定分點囤貨跡象。")
            
            with t4:
                nl, nr = st.columns(2)
                with nl:
                    st.markdown("#### 🎯 個股專屬新聞")
                    if news_s:
                        for n in news_s[:5]: st.markdown(f"**[{n['title']}]({n['link']})**\n<span style='color:gray;font-size:14px;'>🕒 {n['date'].replace(' GMT','')}</span>", unsafe_allow_html=True)
                    else: st.info("無相關新聞")
                with nr:
                    st.markdown("#### 🌍 總經大盤焦點")
                    if news_m:
                        for n in news_m[:5]: st.markdown(f"**[{n['title']}]({n['link']})**\n<span style='color:gray;font-size:14px;'>🕒 {n['date'].replace(' GMT','')}</span>", unsafe_allow_html=True)
                    else: st.info("無大盤新聞")
                    
        if st.button("⬅️ 返回戰情室主頁", use_container_width=True):
            st.session_state.analyze_trigger = None
            st.rerun()

else:
    st.markdown("### 🌍 台股大盤與情緒摘要")
    summary = get_market_summary()
    if summary:
        twii_data = summary.get("加權指數", {"pct": 0})
        base_greed = 50 + (twii_data['pct'] * 15)
        greed_index = int(max(0, min(100, base_greed + random.randint(-5, 5))))
        greed_status = "極度恐懼 🥶" if greed_index < 25 else ("恐懼 😨" if greed_index < 45 else ("中立 😐" if greed_index < 55 else ("貪婪 😏" if greed_index < 75 else "極度貪婪 🤑")))
        
        c_idx, c_greed = st.columns([3, 1])
        with c_idx:
            cols = st.columns(len(summary))
            for i, (name, data) in enumerate(summary.items()): 
                cols[i].metric(name, f"{data['price']:.2f}", f"{data['change']:+.2f} ({data['pct']:+.2f}%)")
            st.markdown("""<style>[data-testid="stMetricDelta"] svg { display: none; } [data-testid="stMetricDelta"] > div { flex-direction: row; } [data-testid="stMetricDelta"] > div:has(div:contains("+")) { color: #ff4b4b !important; } [data-testid="stMetricDelta"] > div:has(div:contains("-")) { color: #00cc96 !important; }</style>""", unsafe_allow_html=True)
        
        with c_greed:
            st.metric("台股恐懼貪婪指數", f"{greed_index} / 100", greed_status, delta_color="off")
            bar_color = "#00cc96" if greed_index < 45 else ("#ffc107" if greed_index < 55 else "#ff4b4b")
            st.markdown(f"""
                <div style="width: 100%; background-color: #333; border-radius: 10px; height: 10px; margin-top: 5px;">
                  <div style="width: {greed_index}%; background-color: {bar_color}; height: 100%; border-radius: 10px;"></div>
                </div>
            """, unsafe_allow_html=True)

    with st.expander("🧠 系統核心：多時區三重濾網策略白皮書 (點此展開)", expanded=False):
        st.markdown("""
        #### 1. 宏觀濾網 (日線巨觀潛伏)
        摒棄追高風險。系統在日K級別專注尋找：
        * **📉 量縮回踩**：趨勢偏多，但今日量縮下跌。主力洗盤最安全的買點。
        * **🛡️ 區間壓縮 (Squeeze)**：布林通道收斂，等待爆發。

        #### 2. 微觀濾網 (小時線精確狙擊)
        大週期確定潛伏後，系統會啟用 1 小時 (1h) 線的微觀狙擊雷達。只有當 1h K線突然出現「帶量突破均線且MACD金叉」時，系統才會觸發極限買點，幫您過濾掉盤中假突破的騙線。
        """)
            
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 自選即時流", "🎯 全市場 AI 進出場戰術面板", "🕸️ 產業鏈資金共振 (精選)", "🔬 策略回測實驗室 (實盤)"])
    
    # ==========================================================
    # 📊 TAB 1: 自選即時流 x 完美對齊的 LightGBM 動態勝率膠囊標籤
    # ==========================================================
    with tab1:
        c_title, c_slider = st.columns([2, 1])
        with c_title: st.markdown(f"#### 【{selected_cluster}】即時行情流 x AI 戰術標籤")
        with c_slider:
            with st.expander("⚙️ 畫幅設定"): user_font_size = st.slider("表格文字大小", 12, 40, 22, 2)
            
        @st.fragment(run_every=datetime.timedelta(seconds=15))
        def render_rt():
            rows = []
            current_names = st.session_state.stock_names.copy()
            
            # 🧠 同步加載全局新大腦與大盤快取
            model, features = get_ai_model()
            snapshot = load_market_snapshot()
            snapshot_dict = {}
            
            # 快取矩陣轉雜湊字典，實現 O(1) 極速搜尋，防止多執行緒阻塞
            if snapshot and 'data' in snapshot:
                for item in snapshot['data']:
                    tk = str(item.get('代號', item.get('ticker', ''))).split('.')[0].strip()
                    snapshot_dict[tk] = item
            
            def fetch_single_rt(t, names_dict):
                try:
                    clean_ticker = t.split('.')[0]
                    base_name = names_dict.get(clean_ticker, clean_ticker)
                    
                    # 🚀 核心：非同步即時為自選股打上 100% 同步的 LightGBM 勝率標籤
                    ai_badge_html = ""
                    if model and clean_ticker in snapshot_dict:
                        try:
                            match_item = snapshot_dict[clean_ticker]
                            pattern_str = str(match_item.get('pattern', match_item.get('Pattern', match_item.get('型態', ''))))
                            rs_val_str = str(match_item.get('RS_Index', match_item.get('rs_index', match_item.get('大盤相對強度', '0')))).replace('%', '')
                            
                            try: rs_index = float(rs_val_str)
                            except: rs_index = 0.0
                            
                            volatility = float(match_item.get('volatility', match_item.get('Volatility', 0.0)))
                            turnover = float(match_item.get('turnover', match_item.get('Turnover', 0.0)))
                            
                            # 完美對齊 7 大維度特徵
                            input_data = {
                                'is_pullback': 1 if "量縮回踩" in pattern_str else 0,
                                'is_sweep': 1 if "流動性掠奪" in pattern_str else 0,
                                'is_squeeze': 1 if "區間壓縮" in pattern_str else 0,
                                'is_divergence': 1 if "底背離" in pattern_str else 0,
                                'rs_index': rs_index,
                                'volatility': volatility,
                                'turnover': turnover
                            }
                            
                            input_df = pd.DataFrame([input_data], columns=features).fillna(0)
                            win_prob = float(model.predict_proba(input_df)[0][1])
                            win_rate_pct = win_prob * 100
                            
                            # 高質感膠囊視覺樣式
                            if win_rate_pct >= 60:
                                badge_style = "background-color: #00cc96; color: black; padding: 10px 16px; border-radius: 4px; font-size: 12px; font-weight: bold;"
                                prefix = "⭐ 核心強勢"
                            elif win_rate_pct >= 50:
                                badge_style = "background-color: #ffc107; color: black; padding: 10px 16px; border-radius: 4px; font-size: 12px; font-weight: bold;"
                                prefix = "⚖️ 溫和觀察"
                            else:
                                badge_style = "background-color: #555555; color: #bbb; padding: 10px 16px; border-radius: 4px; font-size: 12px;"
                                prefix = "⏸️ 暫無動能"
                                
                            ai_badge_html = f"<br><span style='{badge_style}'>{prefix} {win_rate_pct:.1f}%</span>"
                        except: pass
                    
                    # 標籤嵌入標的欄位
                    name_str = f"<b>{base_name}</b><br><span style='font-size:0.8em;color:gray;'>{clean_ticker}</span>{ai_badge_html}"
                    
                    # 即時報價串接 (Fugle / Yahoo Fallback)
                    if FUGLE_API_KEY:
                        try:
                            res = requests.get(f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{clean_ticker}", headers={"X-API-KEY": FUGLE_API_KEY}, timeout=2)
                            if res.status_code == 200:
                                data = res.json()
                                rt_price = data.get('closePrice')
                                if not rt_price and data.get('lastTrade'): rt_price = data.get('lastTrade').get('price')
                                prev_close = data.get('previousClose') or data.get('referencePrice')
                                rt_vol = data.get('total', {}).get('tradeVolume', 0)
                                
                                if rt_price and prev_close:
                                    rt_price, prev_close = float(rt_price), float(prev_close)
                                    change_amt = rt_price - prev_close
                                    change_pct = (change_amt / prev_close) * 100 if prev_close > 0 else 0
                                    
                                    price_vol = f"<b>{rt_price:.2f}</b><br><span style='font-size:0.7em;color:gray;'>({int(rt_vol):,} 張)</span>"
                                    change_str = f"<span style='color:#ff4b4b;font-weight:bold;'>+{change_amt:.2f}<br>(+{change_pct:.2f}%)</span>" if change_amt > 0 else (f"<span style='color:#00cc96;font-weight:bold;'>{change_amt:.2f}<br>({change_pct:.2f}%)</span>" if change_amt < 0 else "0.00")
                                    return {"標的": name_str, "及時價 (成交量)": price_vol, "今日漲跌幅": change_str, "raw_pct": change_pct}
                        except: pass

                    try:
                        df = fetch_yahoo_robust(f"{clean_ticker}.TW", period="5d", interval="1d")
                        if df.empty: df = fetch_yahoo_robust(f"{clean_ticker}.TWO", period="5d", interval="1d")
                        if not df.empty and len(df) >= 2:
                            c, p = df.iloc[-1], df.iloc[-2]
                            rt_price, float_p = float(c['Close']), float(p['Close'])
                            rt_vol = float(c['Volume'])
                            change_amt = rt_price - float_p
                            change_pct = (change_amt / float_p) * 100 if float_p > 0 else 0
                            
                            price_vol = f"<b>{rt_price:.2f}</b><br><span style='font-size:0.7em;color:gray;'>({int(rt_vol):,} 張)</span>"
                            change_str = f"<span style='color:#ff4b4b;font-weight:bold;'>+{change_amt:.2f}<br>(+{change_pct:.2f}%)</span>" if change_amt > 0 else (f"<span style='color:#00cc96;font-weight:bold;'>{change_amt:.2f}<br>({change_pct:.2f}%)</span>" if change_amt < 0 else "0.00")
                            return {"標的": name_str, "及時價 (成交量)": price_vol, "今日漲跌幅": change_str, "raw_pct": change_pct}
                    except: pass
                except: pass
                return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                futures = [executor.submit(fetch_single_rt, t, current_names) for t in cluster_stocks]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res: rows.append(res)
                
            if rows:
                sorted_by_pct = sorted(rows, key=lambda x: x['raw_pct'], reverse=True)
                top_gainers = [s for s in sorted_by_pct if s['raw_pct'] > 0][:3]
                
                st.markdown("##### 🏆 群組內領漲強勢股")
                if top_gainers:
                    c_g1, c_g2, c_g3 = st.columns(3)
                    g_cols = [c_g1, c_g2, c_g3]
                    for idx, g in enumerate(top_gainers):
                        with g_cols[idx]:
                            clean_name = g['標的'].split('<br>')[0].replace('<b>','').replace('</b>','')
                            st.markdown(f"<div style='background:#2b1111;padding:10px;border-left:4px solid #ff4b4b;border-radius:5px;text-align:center;'><b>{clean_name}</b><br><span style='color:#ff4b4b;font-size:1.2em;font-weight:bold;'>+{g['raw_pct']:.2f}%</span></div>", unsafe_allow_html=True)
                else: st.info("群組內暫無上漲標的。")
                st.write("")

                html_table = pd.DataFrame(rows)[["標的", "及時價 (成交量)", "今日漲跌幅"]].to_html(escape=False, index=False, border=0).replace('\n', '')
                css = f"<style>.watch-board {{ width: 100%; }} .watch-board table {{ width: 100% !important; border-collapse: collapse; }} .watch-board th {{ text-align: center !important; font-size: {max(14, user_font_size-4)}px !important; padding: 10px !important; border-bottom: 2px solid #555 !important; color: #888; }} .watch-board td {{ text-align: center !important; font-size: {user_font_size}px !important; padding: 16px !important; border-bottom: 1px solid #444 !important; vertical-align: middle !important; }}</style>".replace('\n', '')
                st.markdown(f'{css}<div class="watch-board">{html_table}</div>', unsafe_allow_html=True)
            else: st.info("同步流介接中...")
        render_rt()

    # ==========================================================
    # 🎯 TAB 2: 全市場 AI 勝率最高進出場戰術面板 (100% 矩陣批量推論)
    # ==========================================================
    with tab2:
        st.markdown("#### 🎯 全市場 AI 勝率最高進出場戰術面板 (TOP 20)")
        st.caption("系統透過矩陣平行運算，將全市場數據送入具備『股性與規模特徵』的神經網路推演，瞬間篩選出明日預測勝率最高的 20 檔強勢標的。")
        
        snapshot = load_market_snapshot()
        if snapshot:
            st.success(f"⏱️ 數據最後更新時間: {snapshot['update_time']} (極速推演完成)")
            
            raw_list = snapshot['data']
            valid_items = []
            bulk_features = []
            
            model, features = get_ai_model()
            
            # 第一階段：提取有效數據並打包成矩陣 (支援大、小寫與新股性欄位防呆)
            for item in raw_list:
                entry_price = float(item.get('現價', item.get('close_price', item.get('Close', 0.0))))
                if entry_price == 0: continue
                
                valid_items.append(item)
                
                if model:
                    pattern_str = str(item.get('pattern', item.get('Pattern', item.get('型態', ''))))
                    rs_val_str = str(item.get('RS_Index', item.get('rs_index', item.get('大盤相對強度', '0')))).replace('%', '')
                    try: rs_index = float(rs_val_str)
                    except: rs_index = 0.0
                    
                    volatility = float(item.get('volatility', item.get('Volatility', 0.0)))
                    turnover = float(item.get('turnover', item.get('Turnover', 0.0)))
                    
                    bulk_features.append({
                        'is_pullback': 1 if "量縮回踩" in pattern_str else 0,
                        'is_sweep': 1 if "流動性掠奪" in pattern_str else 0,
                        'is_squeeze': 1 if "區間壓縮" in pattern_str else 0,
                        'is_divergence': 1 if "底背離" in pattern_str else 0,
                        'rs_index': rs_index,
                        'volatility': volatility,
                        'turnover': turnover
                    })

            # 第二階段：一鍵執行全市場批量向量化推論 (Vectorized Inference)
            all_probs = []
            if model and bulk_features:
                try:
                    input_df = pd.DataFrame(bulk_features, columns=features).fillna(0)
                    all_probs = model.predict_proba(input_df)[:, 1] 
                except: pass

            # 第三階段：合併排序預測結果
            processed_stocks = []
            for idx, item in enumerate(valid_items):
                ticker = str(item.get('代號', item.get('ticker', ''))).split('.')[0].strip()
                name = str(item.get('名稱', item.get('name', '')))
                entry_price = float(item.get('現價', item.get('close_price', item.get('Close', 0.0))))
                ai_score = int(item.get('量化總分', item.get('score', 0)))
                
                res_level = float(item.get('Res_20', entry_price * 1.05))
                sup_level = float(item.get('Sup_20', entry_price * 0.95))
                atr_14 = float(item.get('ATR_14', entry_price * 0.05))
                
                if model and len(all_probs) > idx:
                    win_prob = float(all_probs[idx])
                else:
                    win_prob = ai_score / 100.0
                
                box_height = max(res_level - sup_level, 0.01)
                pattern_str = str(item.get('pattern', item.get('Pattern', item.get('型態', ''))))
                
                if entry_price > res_level:
                    stop_loss = round(res_level * 0.985, 2)
                    take_profit = round(res_level + box_height, 2)
                    profit_reason = "🚀 噴發目標：等距測幅擴展位"
                elif "量縮回踩" in pattern_str or "流動性掠奪" in pattern_str or "底背離" in pattern_str:
                    stop_loss = round(min(entry_price - (1.5 * atr_14), sup_level * 0.985), 2)
                    take_profit = round(res_level, 2)
                    profit_reason = "🎯 潛伏目標：前高/箱頂壓力區"
                else:
                    stop_loss = round(min(entry_price - (1.5 * atr_14), sup_level * 0.985), 2)
                    take_profit = round(res_level + (atr_14 * 1.0), 2)
                    profit_reason = "⚔️ 波段目標：前高波動擴張位"
                    
                risk_per_share = max(entry_price - stop_loss, 0.01)
                real_rr_ratio = round((take_profit - entry_price) / risk_per_share, 2)
                
                if model:
                    ai_win_rate_str = f"{win_prob * 100:.1f}%"
                    if win_prob > 0.60 and real_rr_ratio >= 1.5:
                        ai_recommendation = "⭐⭐⭐ 極致期望值！(高勝率 + 高風報比)"
                        box_color = "#00cc96" 
                    elif win_prob > 0.50 and real_rr_ratio >= 1.0:
                        ai_recommendation = "⭐⭐ 溫和佈局 (具備正向期望值)"
                        box_color = "#ffc107" 
                    elif win_prob <= 0.50:
                        ai_recommendation = "⚠️ 預測敗率較高，建議嚴格觀望"
                        box_color = "#555555" 
                    else:
                        ai_recommendation = "⏸️ 風報比不足，防守空間過窄"
                        box_color = "#555555"
                else:
                    ai_win_rate_str = f"基底分數: {ai_score} 分"
                    ai_recommendation = "⏳ 尚未載入 AI 訓練大腦，顯示基底推薦"
                    box_color = "#ffc107" if ai_score >= 60 else "#555555"
                
                processed_stocks.append({
                    'ticker': ticker, 'name': name, 'win_prob': win_prob, 
                    'ai_win_rate_str': ai_win_rate_str, 'ai_recommendation': ai_recommendation, 
                    'box_color': box_color, 'entry_price': entry_price, 
                    'take_profit': take_profit, 'stop_loss': stop_loss, 
                    'profit_reason': profit_reason, 'real_rr_ratio': real_rr_ratio
                })
            
            # 第四階段：批量渲染渲染前 20 名旗艦圖卡
            top_20 = sorted(processed_stocks, key=lambda x: x['win_prob'], reverse=True)[:20]
            
            for s in top_20:
                st.markdown(f"""
                <div style="border: 2px solid {s['box_color']}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;">
                    <h4 style="color: {s['box_color']}; margin-top: 0;">🎯 AI 深度學習 x 結構價格 戰術計畫 ({s['ticker']} {s['name']})</h4>
                    <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 180px; margin-bottom: 10px;">
                            <span style="color: gray; font-size: 14px;">1. AI 真實勝率預測</span><br>
                            <b style="font-size: 24px; color: {s['box_color']};">{s['ai_win_rate_str']}</b><br>
                            <span style="font-size: 14px; font-weight: bold;">{s['ai_recommendation']}</span>
                        </div>
                        <div style="flex: 1; min-width: 130px; margin-bottom: 10px;">
                            <span style="color: gray; font-size: 14px;">2. 建議進場價</span><br>
                            <b style="font-size: 22px;">{s['entry_price']:.2f}</b><br>
                            <span style="font-size: 12px; color: gray;">(現價/限價單)</span>
                        </div>
                        <div style="flex: 1; min-width: 200px; margin-bottom: 10px;">
                            <span style="color: gray; font-size: 14px;">3. 結構停利點</span><br>
                            <b style="font-size: 22px; color: #00cc96;">{s['take_profit']:.2f}</b><br>
                            <span style="font-size: 12px; color: #00cc96; font-weight: bold;">{s['profit_reason']}</span><br>
                            <span style="font-size: 12px; color: gray;">(實況風報比 1 : {s['real_rr_ratio']})</span>
                        </div>
                        <div style="flex: 1; min-width: 130px; margin-bottom: 10px;">
                            <span style="color: gray; font-size: 14px;">4. 嚴格防守價</span><br>
                            <b style="font-size: 22px; color: #ff4b4b;">{s['stop_loss']:.2f}</b><br>
                            <span style="font-size: 12px; color: gray;">(跌破無條件停損)</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ 全市場快取準備中，請先前往 GitHub Actions 觸發每日全市場掃描...")

    # ==========================================================
    # 🕸️ TAB 3: 上中下游產業鏈資金共振分析
    # ==========================================================
    with tab3:
        st.markdown("#### 🕸️ 上中下游產業鏈資金共振分析 (Top-Down)")
        selected_chain = st.selectbox("選擇要檢視的產業鏈", list(INDUSTRY_CHAINS.keys()))
        chain_data = INDUSTRY_CHAINS[selected_chain]
        
        snapshot = load_market_snapshot()
        if snapshot:
            res_df = pd.DataFrame(snapshot['data'])
            if '代號' in res_df.columns: 
                res_df['代號'] = res_df['代號'].astype(str).str.replace('.0', '', regex=False).str.strip()
            
            st.markdown("---")
            cols = st.columns(len(chain_data))
            
            for idx, (sub_name, tickers) in enumerate(chain_data.items()):
                with cols[idx]:
                    sub_codes = [str(t).split('.')[0].strip() for t in tickers]
                    sub_res = res_df[res_df['代號'].isin(sub_codes)].copy()
                    
                    if not sub_res.empty:
                        avg_score = int(sub_res['量化總分'].mean())
                        heat_color = "#ff4b4b" if avg_score >= 65 else ("#ffc107" if avg_score >= 45 else "#00cc96")
                        
                        st.markdown(f"<div style='background:#1e1e1e;padding:15px;border-top:4px solid {heat_color};border-radius:5px;margin-bottom:15px;'><b>{sub_name}</b><br><span style='font-size:24px;color:{heat_color};'>板塊熱度: {avg_score} 分</span></div>", unsafe_allow_html=True)
                        st.dataframe(sub_res[['名稱', '現價', '量化總分']].sort_values('量化總分', ascending=False), hide_index=True, use_container_width=True)
                    else:
                        st.markdown(f"**{sub_name}**\n暫無高分數據")
        else:
            st.warning("⚠️ 系統快取準備中...")

    # ==========================================================
    # 🔬 TAB 4: 策略回測實驗室 (實盤期望值盲測)
    # ==========================================================
    with tab4:
        st.markdown("#### 🔬 AI 演算法真實勝率與期望值 (Out-of-Sample)")
        st.caption("系統自動從 Supabase 大腦記憶庫撈取歷史訊號，與未來真實收盤價對撞，計算出策略目前的真實期望值。")
        
        test_threshold = st.slider("🎚️ 設定 AI 分進場門檻", min_value=20, max_value=100, value=50, step=5)
        
        b_col1, b_col2 = st.columns([1, 1])
        with b_col1:
            st.markdown("##### ⏳ 短波段策略 (持倉 5 天)")
            res_5d = fetch_and_calculate_backtest(holding_period=5, threshold=test_threshold)
            
            if res_5d["status"] == "no_key":
                st.error("⚠️ 找不到資料庫金鑰。")
            elif res_5d["status"] == "empty":
                st.warning("⚠️ Supabase 資料庫為空，請先補齊歷史。")
            elif res_5d["status"] == "pending":
                st.info(f"⏸️ 門檻設定為 {test_threshold} 分。目前有 **{res_5d['pending_count']}** 筆訊號等待開獎。")
            elif res_5d["status"] == "error":
                st.error(f"❌ 運算發生錯誤: {res_5d['msg']}")
            elif res_5d["status"] == "ready":
                rr_ratio = abs(res_5d['avg_win'] / res_5d['avg_loss']) if res_5d['avg_loss'] != 0 else 0
                st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-top: 4px solid #00cc96;">
                    <h3 style="margin-top: 0; color: #00cc96;">期望值: {res_5d['expectancy']*100:+.2f}%</h3>
                    <p style="color: gray; margin-bottom: 5px;">總樣本數: {res_5d['total']} 次</p>
                    <b>勝率:</b> {res_5d['win_rate']*100:.1f}%<br>
                    <b>平均獲利:</b> <span style="color:#ff4b4b;">+{res_5d['avg_win']*100:.2f}%</span><br>
                    <b>平均虧損:</b> <span style="color:#00cc96;">{res_5d['avg_loss']*100:.2f}%</span><br>
                    <b>盈虧比:</b> {rr_ratio:.2f}
                </div>
                """, unsafe_allow_html=True)

        with b_col2:
            st.markdown("##### 🈷️ 長波段策略 (持倉 20 天)")
            res_20d = fetch_and_calculate_backtest(holding_period=20, threshold=test_threshold)
            
            if res_20d["status"] == "pending":
                st.info(f"⏸️ 門檻設定為 {test_threshold} 分。目前有 **{res_20d['pending_count']}** 筆訊號等待開獎。")
            elif res_20d["status"] == "error":
                st.error(f"❌ 運算發生錯誤: {res_20d['msg']}")
            elif res_20d["status"] == "ready":
                rr_ratio = abs(res_20d['avg_win'] / res_20d['avg_loss']) if res_20d['avg_loss'] != 0 else 0
                st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-top: 4px solid #ffc107;">
                    <h3 style="margin-top: 0; color: #ffc107;">期望值: {res_20d['expectancy']*100:+.2f}%</h3>
                    <p style="color: gray; margin-bottom: 5px;">總樣本數: {res_20d['total']} 次</p>
                    <b>勝率:</b> {res_20d['win_rate']*100:.1f}%<br>
                    <b>平均獲利:</b> <span style="color:#ff4b4b;">+{res_20d['avg_win']*100:.2f}%</span><br>
                    <b>平均虧損:</b> <span style="color:#00cc96;">{res_20d['avg_loss']*100:.2f}%</span><br>
                    <b>盈虧比:</b> {rr_ratio:.2f}
                </div>
                """, unsafe_allow_html=True)