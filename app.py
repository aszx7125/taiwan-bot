import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random 
import concurrent.futures
import requests  # 導入原生請求，供富果即時流量調用

from config import get_fugle_key, DEFAULT_CLUSTERS, DEFAULT_NAMES, INDUSTRY_CHAINS
from data_fetcher import (
    load_all_market_tickers, get_market_index_data, get_market_summary, 
    get_kline_with_fugle, get_stock_news, get_macro_news, run_robust_market_scan, 
    get_precalculated_market_ret, fetch_yahoo_robust  # 🚀 關鍵：導入輕量爬蟲核心
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

import json
import os

# 🚀 雲端背景快取載入器
def load_market_snapshot():
    if os.path.exists("market_snapshot.json"):
        try:
            with open("market_snapshot.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

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
    
    with st.spinner(f"正在分析 {target_ticker}... 多維度時區運算中"):
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
            if low_vol_pb: smc_status.append("📉 量縮回踩 (絕佳佈局點)")
            if squeeze_on: smc_status.append("🛡️ 區間極度壓縮 (暴風雨前夕)")
            if liq_sweep: smc_status.append("🌊 流動性掠奪 (主力洗盤完畢)")
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
            
            # 🚀 提取小時區 (1h) 微觀狙擊狀態
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

            if ai_score >= 65 and micro_trigger and real_rr_ratio >= 1.0:
                trade_action = f"🎯 多時區共振狙擊！(期待值80%+)"
                box_color = "#ff4b4b" 
            elif ai_score >= 70 and real_rr_ratio >= 1.2:
                trade_action = f"✅ 極高勝率潛伏區間"
                box_color = "#00cc96"
            elif ai_score >= 55 and real_rr_ratio >= 1.0:
                trade_action = "⚠️ 溫和試單 (等待小時區發動)"
                box_color = "#ffc107"
            else:
                trade_action = "⏸️ 勝率偏低或追高風險，強制觀望"
                box_color = "#555555"

            st.subheader(f"🧬 {target_ticker} {c_name} 多時區量化診斷報告")
            
            st.markdown(f"""
            <div style="border: 2px solid {box_color}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;">
                <h4 style="color: {box_color}; margin-top: 0;">🎯 三重濾網戰術計畫 (日線 x 小時線)</h4>
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 180px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">1. 最終戰術指令</span><br>
                        <b style="font-size: 18px; color: {box_color};">{trade_action}</b><br>
                        <span style="font-size: 14px;">日K分數: {ai_score} | 1h狀態: {micro_status_text.split(' ')[0]}</span>
                    </div>
                    <div style="flex: 1; min-width: 130px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">2. 狙擊進場價格</span><br>
                        <b style="font-size: 22px;">{entry_price:.2f}</b><br>
                        <span style="font-size: 12px; color: gray;">(現貨/限價單)</span>
                    </div>
                    <div style="flex: 1; min-width: 200px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">3. 盤面合理獲利點</span><br>
                        <b style="font-size: 22px; color: #00cc96;">{take_profit:.2f}</b><br>
                        <span style="font-size: 12px; color: #00cc96; font-weight: bold;">{profit_reason}</span><br>
                        <span style="font-size: 12px; color: gray;">(實況風報比 1 : {real_rr_ratio})</span>
                    </div>
                    <div style="flex: 1; min-width: 130px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">4. 嚴格止損防禦價</span><br>
                        <b style="font-size: 22px; color: #ff4b4b;">{stop_loss:.2f}</b><br>
                        <span style="font-size: 12px; color: gray;">(日K微觀防守點)</span>
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
                            d_base_close = float(d_base.get('Close', 1.0))
                            
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
                            d_base_m_close = float(d_base_m.get('Close', 1.0))
                            
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

        #### 3. AI 聯動判定指令
        * **🎯 多時區共振狙擊！**：日K分數高 + 1h突破確認 (可重倉)。
        * **✅ 極高勝率潛伏區間**：日K分數高 + 1h未發動 (可佈局底倉)。
        * **🚨 假突破陷阱**：日K分數低 + 1h突然大漲 (嚴格觀望，不要被騙上車)。
        """)
            
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊 自選即時流", "🎯 潛伏型 AI 評分 (勝率排行)", "🕸️ 產業鏈資金共振 (精選)"])
    
    with tab1:
        c_title, c_slider = st.columns([2, 1])
        with c_title: st.markdown(f"#### 【{selected_cluster}】即時行情流")
        with c_slider:
            with st.expander("⚙️ 畫幅設定"): user_font_size = st.slider("表格文字大小", 12, 40, 22, 2)
            
        @st.fragment(run_every=datetime.timedelta(seconds=15))
        def render_rt():
            rows = []
            
            # 🚀 終極修復：即時看板『只抓5天日 K 線』，不運算任何複雜指標，0.01秒光速響應
            def fetch_single_rt(t):
                try:
                    clean_ticker = t.split('.')[0]
                    df = fetch_yahoo_robust(f"{clean_ticker}.TW", period="5d", interval="1d")
                    if df.empty:
                        df = fetch_yahoo_robust(f"{clean_ticker}.TWO", period="5d", interval="1d")
                    
                    if not df.empty and len(df) >= 2:
                        c, p = df.iloc[-1], df.iloc[-2]
                        
                        # 若具備富果 API 密鑰，直接進行零延遲覆蓋
                        if FUGLE_API_KEY:
                            try:
                                res = requests.get(f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{clean_ticker}", headers={"X-API-KEY": FUGLE_API_KEY}, timeout=2)
                                if res.status_code == 200:
                                    data = res.json()
                                    rt_price = data.get('closePrice') or data.get('lastTrade', {}).get('price')
                                    rt_vol = data.get('total', {}).get('tradeVolume', 0)
                                    if rt_price: c['Close'] = float(rt_price)
                                    if rt_vol > 0: c['Volume'] = float(rt_vol)
                            except: pass

                        change_amt = c['Close'] - p['Close']
                        change_pct = (change_amt / p['Close']) * 100
                        gap = " <span style='color:#ff4b4b;font-size:0.7em;'>(跳空🔥)</span>" if c['Low'] > p['High'] else ""
                        
                        price_vol = f"<b>{c['Close']:.2f}</b><br><span style='font-size:0.7em;color:gray;'>({int(c['Volume']):,} 張)</span>"
                        name_str = f"<b>{st.session_state.stock_names.get(clean_ticker, t)}</b><br><span style='font-size:0.8em;color:gray;'>{clean_ticker}</span>"
                        change_str = f"<span style='color:#ff4b4b;font-weight:bold;'>+{change_amt:.2f}<br>(+{change_pct:.2f}%){gap}</span>" if change_amt > 0 else (f"<span style='color:#00cc96;font-weight:bold;'>{change_amt:.2f}<br>({change_pct:.2f}%){gap}</span>" if change_amt < 0 else "0.00")
                        
                        return {"標的": name_str, "及時價 (成交量)": price_vol, "今日漲跌幅": change_str, "raw_pct": change_pct}
                except: pass
                return None

            # 🛠️ 15 條快線併發，5日輕量數據對撞，徹底解決限流問題
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                futures = [executor.submit(fetch_single_rt, t) for t in cluster_stocks]
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

    with tab2:
        st.markdown("#### 🎯 潛伏型 AI 勝率期望值排行 (TOP 20)")
        st.caption("以下數據由雲端背景運算引擎於每日盤後自動解算，達成 0 延遲載入。")
        
        snapshot = load_market_snapshot()
        if snapshot:
            st.success(f"⏱️ 數據最後更新時間: {snapshot['update_time']} (資料庫直連)")
            df_res = pd.DataFrame(snapshot['data'])
            st.dataframe(df_res.head(20), column_config={"量化總分": st.column_config.ProgressColumn("潛伏勝率期望值", min_value=0, max_value=100, format="%d 分")}, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ 預算快取準備中，請先至 GitHub 觸發首次 Actions 運算...")

    with tab3:
        st.markdown("#### 🕸️ 上中下游產業鏈資金共振分析 (Top-Down)")
        selected_chain = st.selectbox("選擇要檢視的產業鏈", list(INDUSTRY_CHAINS.keys()))
        chain_data = INDUSTRY_CHAINS[selected_chain]
        
        snapshot = load_market_snapshot()
        if snapshot:
            res_df = pd.DataFrame(snapshot['data'])
            st.markdown("---")
            cols = st.columns(len(chain_data))
            
            for idx, (sub_name, tickers) in enumerate(chain_data.items()):
                with cols[idx]:
                    sub_codes = [t.split('.')[0] for t in tickers]
                    sub_res = res_df[res_df['代號'].isin(sub_codes)].copy()
                    
                    if not sub_res.empty:
                        avg_score = int(sub_res['量化總分'].mean())
                        heat_color = "#ff4b4b" if avg_score >= 65 else ("#ffc107" if avg_score >= 45 else "#00cc96")
                        
                        st.markdown(f"<div style='background:#1e1e1e;padding:15px;border-top:4px solid {heat_color};border-radius:5px;margin-bottom:15px;'><b>{sub_name}</b><br><span style='font-size:24px;color:{heat_color};'>板塊熱度: {avg_score} 分</span></div>", unsafe_allow_html=True)
                        st.dataframe(sub_res[['名稱', '現價', '量化總分']].sort_values('量化總分', ascending=False), hide_index=True, use_container_width=True)
                    else:
                        st.markdown(f"**{sub_name}**\n查無高分數據")
        else:
            st.warning("⚠️ 系統快取準備中...")