import streamlit as st
import pandas as pd
import datetime
import random 
import concurrent.futures

from config import get_fugle_key, DEFAULT_CLUSTERS, DEFAULT_NAMES, INDUSTRY_CHAINS
from data_fetcher import (
    load_all_market_tickers, get_market_index_data, get_market_summary, 
    get_kline_with_fugle, get_stock_news, get_macro_news, run_robust_market_scan, get_precalculated_market_ret
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
    
    with st.spinner(f"正在分析 {target_ticker}... 提取深度資料中"):
        df, actual_symbol = get_kline_with_fugle(target_ticker, FUGLE_API_KEY)
        
        if df.empty or len(df) < 40: 
            st.error("❌ 該標的數據深度不足，無法執行複雜演算法。")
        else:
            news_s = get_stock_news(c_name)
            news_m = get_macro_news()

            today, yesterday = df.iloc[-1], df.iloc[-2]
            vol_ratio = (today['Volume'] / today['Vol_SMA5']) if today['Vol_SMA5'] > 0 else 1.0
            p_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100
            
            res_level, sup_level = today['Res_20'], today['Sup_20']
            box_height = res_level - sup_level
            recent_20_df = df.iloc[-21:-1]
            res_tests = len(recent_20_df[recent_20_df['High'] >= res_level * 0.985])
            sup_tests = len(recent_20_df[recent_20_df['Low'] <= sup_level * 1.015])
            
            bull_div = bool(today.get('Bullish_Div', False))
            bear_div = bool(today.get('Bearish_Div', False))
            div_status = "🟢 底背離 (空頭力竭，醞釀反彈)" if bull_div else ("🚨 頂背離 (多頭力竭，注意回檔)" if bear_div else "無顯著背離")
            
            liq_sweep = bool(today.get('Liquidity_Sweep_Bull', False))
            fvg_bull = bool(today.get('FVG_Bull', False))
            smc_status = []
            if liq_sweep: smc_status.append("🌊 流動性掠奪 (破底翻)")
            if fvg_bull: smc_status.append("🧱 FVG大戶缺口")
            smc_text = " + ".join(smc_status) if smc_status else "無觸發"
            
            block_trade = bool(today.get('Block_Trade_Inflow', False))
            broker_conc = float(today.get('Broker_Concentration', 0.0))
            
            breakout_status = "區間震盪 (未突破)"
            if today['Close'] > res_level: breakout_status = "🚀 向上突破前高"
            elif today['Close'] < sup_level: breakout_status = "⚠️ 向下摜破前低"
            elif today['Close'] >= res_level * 0.98: breakout_status = "⚔️ 兵臨城下 (挑戰前高)"
            elif today['Close'] <= sup_level * 1.02: breakout_status = "🛡️ 支撐保衛戰 (回測前低)"

            # ==========================================
            # 🎯 核心升級：AI 戰術執行面板 (四要件計算)
            # ==========================================
            entry_price = today['Close']
            
            # 4. 止損價格 (防禦)：取 ATR 吊燈止損 與 前低支撐 的較低者，給予足夠洗盤空間
            atr_stop = entry_price - (1.5 * yesterday['ATR_14'])
            structural_stop = sup_level * 0.985 # 跌破前低 1.5% 視為真跌破
            stop_loss = round(min(atr_stop, structural_stop), 2)
            
            # 若為強勢突破，將停損點拉高保護利潤 (Trailing Stop)
            if today['Close'] > res_level:
                stop_loss = round(res_level * 0.985, 2)
                
            risk_per_share = entry_price - stop_loss
            
            # 3. 停利價格 (攻擊)：嚴格執行 1:2.5 的風報比 (賺要賺賠的 2.5 倍)
            # 若等距測幅的目標價更高，則取更高者
            rr_target = entry_price + (risk_per_share * 2.5)
            structural_target = res_level + box_height
            take_profit = round(max(rr_target, structural_target), 2)
            
            # 交易建議判定
            ai_score = int(today.get('Score', 0))
            if ai_score >= 75 and (liq_sweep or fvg_bull or today['Close'] > res_level or block_trade):
                trade_action = f"✅ 強烈建議買進 (AI評分: {ai_score} 分)"
                box_color = "#00cc96"
            elif ai_score >= 60:
                trade_action = "⚠️ 建議小部位試單 (條件尚未完全共振)"
                box_color = "#ffc107"
            else:
                trade_action = "⏸️ 建議空手觀望 (動能不足或風險過高)"
                box_color = "#555555"

            st.subheader(f"🧬 {target_ticker} {c_name} 深度量化診斷報告")
            
            # 🎯 獨立顯示的 AI 戰術面板
            st.markdown(f"""
            <div style="border: 2px solid {box_color}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;">
                <h4 style="color: {box_color}; margin-top: 0;">🎯 AI 戰術執行計畫</h4>
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 150px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">1. 建議動作與標的</span><br>
                        <b style="font-size: 18px; color: {box_color};">{trade_action}</b><br>
                        <span style="font-size: 16px;">{c_name} ({target_ticker})</span>
                    </div>
                    <div style="flex: 1; min-width: 150px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">2. 建議買入價格</span><br>
                        <b style="font-size: 22px;">{entry_price:.2f}</b><br>
                        <span style="font-size: 12px; color: gray;">(現價進場或限價掛單)</span>
                    </div>
                    <div style="flex: 1; min-width: 150px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">3. 獲利了結目標價</span><br>
                        <b style="font-size: 22px; color: #00cc96;">{take_profit:.2f}</b><br>
                        <span style="font-size: 12px; color: gray;">(風報比 1:2.5 測幅)</span>
                    </div>
                    <div style="flex: 1; min-width: 150px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">4. 嚴格止損防禦價</span><br>
                        <b style="font-size: 22px; color: #ff4b4b;">{stop_loss:.2f}</b><br>
                        <span style="font-size: 12px; color: gray;">(跌破此價無條件砍倉)</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("當前現價", f"{today['Close']:.2f}", f"{p_change:+.2f}%")
            m2.metric("即時量比", f"{vol_ratio:.1f}x", f"今日成交 {int(today['Volume']):,} 張", delta_color="off")
            m3.metric("AI 綜合評分", f"{ai_score} 分")
            m4.metric("大盤相對強度", f"{today.get('RS_Index', 0)*100:+.2f}%")
            
            st.markdown("---")
            t1, t2, t3, t4 = st.tabs(["🧱 結構與前瞻推演", "🔍 多週期策略回測", "🏦 大單流與分點籌碼", "📰 專屬新聞動態"])
            
            with t1:
                c_l, c_r = st.columns(2)
                with c_l:
                    st.markdown("#### 📐 當前關鍵結構與預判")
                    st.write(f"- **前高壓力 (近20日):** {res_level:.2f} | **已測試:** {res_tests} 次")
                    st.write(f"- **前低支撐 (近20日):** {sup_level:.2f} | **已測試:** {sup_tests} 次")
                    st.write(f"- **盤勢型態判定:** {breakout_status}")
                    st.markdown(f"- **RSI 動能背離偵測:** <span style='color:{'#00cc96' if bull_div else ('#ff4b4b' if bear_div else 'gray')}; font-weight:bold;'>{div_status}</span>", unsafe_allow_html=True)
                    st.markdown(f"- **SMC 微觀結構:** <span style='color:{'#ffc107' if smc_status else 'gray'}; font-weight:bold;'>{smc_text}</span>", unsafe_allow_html=True)
                with c_r:
                    st.markdown("#### 💡 當日訊號解析")
                    if "破底翻" in smc_text or "底背離" in div_status:
                        st.success("🎯 **【起漲潛力】** 出現左側底背離或主力洗盤結束訊號！此處為風報比極佳的低價試單區間。")
                    elif "突破前高" in breakout_status: 
                        st.success("🚀 **【右側追隨】** 型態正式帶量向上突破！屬強勢追隨訊號，防守點位移至前高。")
                    elif "挑戰前高" in breakout_status: 
                        st.warning("⚔️ **【突破預備】** 兵臨城下，即將挑戰壓力！若帶量突破可試單。")
                    else: 
                        st.info("⏸️ **【箱體觀望】** 股價處於內部結構震盪，動能未發動前不建議重倉。")
            
            with t2:
                st.markdown("### 🔍 多週期策略回測與實況對撞")
                sub_t1, sub_t2 = st.tabs(["5️⃣ 一週波段 (5日)", "🈷️ 單月波段 (20日)"])
                with sub_t1:
                    if len(df) >= 26:
                        d_base = df.iloc[-6]
                        d_5_days = df.iloc[-5:]
                        w_res, w_sup, w_atr = d_base['Res_20'], d_base['Sup_20'], d_base['ATR_14']
                        w_target = w_res + w_atr
                        max_h_5d = d_5_days['High'].max()
                        min_l_5d = d_5_days['Low'].min()
                        c_now = today['Close']
                        
                        col_w1, col_w2 = st.columns(2)
                        with col_w1: st.info(f"**5 天前預測基準**\n- 當時壓力: **{w_res:.2f}**\n- 當時支撐: **{w_sup:.2f}**\n- 測幅目標: **{w_target:.2f}**")
                        with col_w2: st.warning(f"**本週實況極值 (近5日)**\n- 波段最高: **{max_h_5d:.2f}**\n- 波段最低: **{min_l_5d:.2f}**\n- 目前收盤: **{c_now:.2f}**")
                        
                        max_gain = ((max_h_5d - d_base['Close']) / d_base['Close']) * 100
                        max_loss = ((min_l_5d - d_base['Close']) / d_base['Close']) * 100
                        st.markdown(f"**📈 本週最大潛在獲利:** <span style='color:#ff4b4b;'>+{max_gain:.2f}%</span> | **📉 本週最大潛在回撤:** <span style='color:#00cc96;'>{max_loss:.2f}%</span>", unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ 數據不足，無法進行一週歷史回測。")

                with sub_t2:
                    if len(df) >= 45:
                        d_base_m = df.iloc[-21] 
                        d_20_days = df.iloc[-20:]
                        m_res, m_sup, m_atr = d_base_m['Res_20'], d_base_m['Sup_20'], d_base_m['ATR_14']
                        m_target = m_res + m_atr
                        max_h_20d = d_20_days['High'].max()
                        min_l_20d = d_20_days['Low'].min()
                        c_now = today['Close']
                        
                        col_m1, col_m2 = st.columns(2)
                        with col_m1: st.info(f"**20 天前預測基準**\n- 當時壓力: **{m_res:.2f}**\n- 當時支撐: **{m_sup:.2f}**\n- 測幅目標: **{m_target:.2f}**")
                        with col_m2: st.warning(f"**本月實況極值 (近20日)**\n- 波段最高: **{max_h_20d:.2f}**\n- 波段最低: **{min_l_20d:.2f}**\n- 目前收盤: **{c_now:.2f}**")
                        
                        max_gain_m = ((max_h_20d - d_base_m['Close']) / d_base_m['Close']) * 100
                        max_loss_m = ((min_l_20d - d_base_m['Close']) / d_base_m['Close']) * 100
                        st.markdown(f"**📈 本月最大潛在獲利:** <span style='color:#ff4b4b;'>+{max_gain_m:.2f}%</span> | **📉 本月最大潛在回撤:** <span style='color:#00cc96;'>{max_loss_m:.2f}%</span>", unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ 歷史數據深度不足，無法進行單月回測。")
            
            with t3:
                st.markdown("#### 🏦 微觀結構：特大單與贏家分點籌碼追蹤")
                
                c_flow1, c_flow2 = st.columns(2)
                with c_flow1:
                    st.markdown("##### 🌊 盤中特大單淨流入 (Order Flow)")
                    if block_trade: st.success("🚨 **偵測到異常大單狂敲！**\n今日成交量大幅超越均量，且價格強勢推升收在高點附近，顯示有機構級資金不計代價掃貨。")
                    else: st.info("📉 **無明顯大單流入**\n今日成交量能平穩，屬於一般散戶或程式單的自然換手。")
                        
                with c_flow2:
                    st.markdown("##### 🏦 贏家分點集中囤貨 (Broker Concentration)")
                    if broker_conc > 0.3: st.success(f"🔥 **高度集中 (集中度: {broker_conc*100:.1f}%)**\n近 5 日資金呈現異常淨流入，極可能有特定主力分點正在暗中大舉囤貨！")
                    elif broker_conc > 0: st.warning(f"🔍 **溫和吃貨 (集中度: {broker_conc*100:.1f}%)**\n籌碼緩步集中，有特定買盤正在吸收浮額。")
                    else: st.info(f"⚖️ **籌碼發散 (集中度: {broker_conc*100:.1f}%)**\n近 5 日無特定分點囤貨跡象，籌碼呈現自然發散。")
            
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
                <div style="display: flex; justify-content: space-between; font-size: 10px; color: gray; margin-top: 5px;">
                    <span>0 (恐懼)</span><span>100 (貪婪)</span>
                </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 板塊實時監控", "⚡ 條件設定全市場雷達", "🎯 多因子 AI 評分 (含籌碼)", "🕸️ 產業鏈資金共振 (精選)"])
    
    with tab1:
        c_title, c_slider = st.columns([2, 1])
        with c_title: st.markdown(f"#### 【{selected_cluster}】即時行情流")
        with c_slider:
            with st.expander("⚙️ 畫幅設定"): user_font_size = st.slider("表格文字大小", 12, 40, 22, 2)
            
        @st.fragment(run_every=datetime.timedelta(seconds=15))
        def render_rt():
            rows = []
            for t in cluster_stocks:
                try:
                    df, _ = get_kline_with_fugle(t.split('.')[0], FUGLE_API_KEY)
                    if len(df) >= 3:
                        c, p = df.iloc[-1], df.iloc[-2]
                        change_amt = c['Close'] - p['Close']
                        change_pct = (change_amt / p['Close']) * 100
                        gap = " <span style='color:#ff4b4b;font-size:0.7em;'>(跳空🔥)</span>" if c['Low'] > p['High'] else ""
                        
                        price_vol = f"<b>{c['Close']:.2f}</b><br><span style='font-size:0.7em;color:gray;'>({int(c['Volume']):,} 張)</span>"
                        name_str = f"<b>{st.session_state.stock_names.get(t.split('.')[0], t)}</b><br><span style='font-size:0.8em;color:gray;'>{t.split('.')[0]}</span>"
                        change_str = f"<span style='color:#ff4b4b;font-weight:bold;'>+{change_amt:.2f}<br>(+{change_pct:.2f}%){gap}</span>" if change_amt > 0 else (f"<span style='color:#00cc96;font-weight:bold;'>{change_amt:.2f}<br>({change_pct:.2f}%){gap}</span>" if change_amt < 0 else "0.00")
                        
                        rows.append({"標的": name_str, "及時價 (成交量)": price_vol, "今日漲跌幅": change_str, "raw_pct": change_pct})
                except: pass
                
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
        st.markdown("#### ⚡ 條件設定全市場雷達")
        c1, c2 = st.columns(2)
        conds = {
            'vol': c1.checkbox("🔥 量能異常 (> 1.5倍)", value=True), 'rsi': c1.checkbox("📉 RSI谷底 (< 35)", value=False),
            'ma': c2.checkbox("📈 均線多頭 (> 月線)", value=True), 'macd': c2.checkbox("📊 MACD剛金叉", value=True)
        }
        scan_mode = st.radio("範圍", ["自選群組", "全市場 (需 CSV)"], horizontal=True)
        
        if st.button("🚀 啟動條件掃描", type="primary"):
            custom = [t for g in st.session_state.stock_clusters.values() for t in g]
            tickers = list(set(custom + (csv_df['Ticker'].tolist() if "全市場" in scan_mode and not csv_df.empty else [])))
            p_bar, s_text = st.progress(0), st.empty()
            
            res = run_robust_market_scan(tickers, conds, p_bar, s_text, st.session_state.stock_names, get_precalculated_market_ret(), "radar")
            s_text.empty(); p_bar.empty()
            if res: st.dataframe(pd.DataFrame(res), use_container_width=True, hide_index=True)
            else: st.warning("⚠️ 查無符合標的，請嘗試放寬條件。")

    with tab3:
        st.markdown("#### 🎯 多因子 AI 評分與戰術預選 (TOP 20)")
        st.caption("請從下方高分榜中，挑選呈現「買進建議」的標的，點擊『單股掃描』取得精確的進出場價格。")
        if st.button("🔮 執行全權重深度矩陣運算", type="primary"):
            custom = [t for g in st.session_state.stock_clusters.values() for t in g]
            tickers = list(set(custom + (csv_df['Ticker'].tolist() if not csv_df.empty else [])))
            p_bar, s_text = st.progress(0), st.empty()
            res = run_robust_market_scan(tickers, {}, p_bar, s_text, st.session_state.stock_names, get_precalculated_market_ret(), "score")
            s_text.empty(); p_bar.empty()
            if res: 
                df_res = pd.DataFrame(res).sort_values("量化總分", ascending=False).head(20)
                st.dataframe(df_res, column_config={"量化總分": st.column_config.ProgressColumn("多空綜合能量", min_value=0, max_value=100, format="%d 分")}, use_container_width=True, hide_index=True)
            else: st.warning("⚠️ 運算失敗。")

    with tab4:
        st.markdown("#### 🕸️ 上中下游產業鏈資金共振分析 (Top-Down)")
        selected_chain = st.selectbox("選擇要掃描的產業鏈", list(INDUSTRY_CHAINS.keys()))
        chain_data = INDUSTRY_CHAINS[selected_chain]
        
        if st.button("🚀 啟動產業鏈深度掃描", type="primary"):
            all_chain_tickers = [ticker for sublist in chain_data.values() for ticker in sublist]
            p_bar, s_text = st.progress(0), st.empty()
            res = run_robust_market_scan(list(set(all_chain_tickers)), {}, p_bar, s_text, st.session_state.stock_names, get_precalculated_market_ret(), "score")
            s_text.empty(); p_bar.empty()
            
            if res:
                res_df = pd.DataFrame(res)
                st.markdown("---")
                cols = st.columns(len(chain_data))
                recommendations = []
                
                for idx, (sub_name, tickers) in enumerate(chain_data.items()):
                    with cols[idx]:
                        sub_codes = [t.split('.')[0] for t in tickers]
                        sub_res = res_df[res_df['代號'].isin(sub_codes)].copy()
                        
                        if not sub_res.empty:
                            avg_score = int(sub_res['量化總分'].mean())
                            heat_color = "#ff4b4b" if avg_score >= 65 else ("#ffc107" if avg_score >= 45 else "#00cc96")
                            
                            st.markdown(f"<div style='background:#1e1e1e;padding:15px;border-top:4px solid {heat_color};border-radius:5px;margin-bottom:15px;'><b>{sub_name}</b><br><span style='font-size:24px;color:{heat_color};'>板塊熱度: {avg_score} 分</span></div>", unsafe_allow_html=True)
                            st.dataframe(sub_res[['名稱', '現價', '量化總分']].sort_values('量化總分', ascending=False), hide_index=True, use_container_width=True)
                            
                            high_scorers = sub_res[sub_res['量化總分'] >= 75]
                            for _, r in high_scorers.iterrows():
                                recommendations.append(f"**{r['名稱']} ({r['代號']})** - AI總分: {r['量化總分']} 分 _(位於 {sub_name})_")
                        else:
                            st.markdown(f"**{sub_name}**\n查無數據")
                
                st.markdown("---")
                st.markdown("### 🌟 AI 產業鏈精選建議標的")
                if recommendations:
                    st.success("🎯 以下標的為該產業鏈中，具備「微觀籌碼流入、SMC前瞻訊號與技術面突破」的高分強勢股。請將代號輸入上方『單股掃描』以取得精確買賣點：")
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                else:
                    st.info("⏸️ 目前該產業鏈中，尚無綜合評分大於 75 分的強勢發動標的，建議持續觀望。")
            else:
                st.warning("掃描失敗，請稍後再試。")