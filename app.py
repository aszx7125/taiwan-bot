import streamlit as st
import pandas as pd
import datetime
import concurrent.futures

from config import get_fugle_key, DEFAULT_CLUSTERS, DEFAULT_NAMES
from data_fetcher import (
    load_all_market_tickers, get_market_index_data, get_market_summary, 
    get_kline_with_fugle, get_stock_news, get_macro_news, run_robust_market_scan
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

# ==========================================
# 📱 側邊欄控制
# ==========================================
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

# ==========================================
# 🖥️ 主路由
# ==========================================
st.title("⚡ 台股戰情分析終端")
col1, col2 = st.columns([3, 1])
with col1: manual_ticker = st.text_input("輸入股票代號", "", label_visibility="collapsed")
with col2: analyze_manual_btn = st.button("單股掃描", use_container_width=True)
st.markdown("---")

target_ticker = st.session_state.pop('analyze_trigger', None) or (manual_ticker.strip().upper() if analyze_manual_btn else None)

if target_ticker:
    base_ticker = target_ticker.split('.')[0]
    c_name = st.session_state.stock_names.get(base_ticker, target_ticker)
    
    with st.spinner(f"正在以多執行緒全速分析 {target_ticker}..."):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_kline = executor.submit(get_kline_with_fugle, target_ticker, FUGLE_API_KEY)
                future_news_s = executor.submit(get_stock_news, c_name)
                future_news_m = executor.submit(get_macro_news)
                
                df, actual_symbol = future_kline.result()
                news_s = future_news_s.result()
                news_m = future_news_m.result()

            if df.empty or len(df) < 40: 
                st.error("❌ 該標的數據深度不足，無法執行複雜演算法。")
            else:
                today, yesterday = df.iloc[-1], df.iloc[-2]
                vol_ratio = (today['Volume'] / today['Vol_SMA5']) if today['Vol_SMA5'] > 0 else 1.0
                p_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100
                
                res_level, sup_level = today['Res_20'], today['Sup_20']
                box_height = res_level - sup_level
                
                recent_20_df = df.iloc[-21:-1]
                res_tests = len(recent_20_df[recent_20_df['High'] >= res_level * 0.985])
                sup_tests = len(recent_20_df[recent_20_df['Low'] <= sup_level * 1.015])
                
                breakout_status, target_proj, breakout_prob = "區間震盪 (未突破)", "無明確突破方向", "中立"
                if today['Close'] > res_level:
                    breakout_status, target_proj, breakout_prob = "🚀 向上突破前高", f"目標上看 **{round(res_level + box_height, 1)}**", "強勢發動"
                elif today['Close'] < sup_level:
                    breakout_status, target_proj, breakout_prob = "⚠️ 向下摜破前低", f"下看 **{round(sup_level - box_height, 1)}**", "弱勢探底"
                elif today['Close'] >= res_level * 0.98:
                    breakout_status = "⚔️ 兵臨城下 (挑戰前高)"
                    if vol_ratio > 1.3 and today['Close'] > today['Open']:
                        breakout_prob, target_proj = "高機率突破", f"目標上看 **{round(res_level + box_height, 1)}**"
                    else:
                        breakout_prob, target_proj = "機率中等 (量縮)", f"壓力位 {round(res_level, 1)} 附近震盪"
                elif today['Close'] <= sup_level * 1.02:
                    breakout_status = "🛡️ 支撐保衛戰 (回測前低)"
                    if vol_ratio > 1.3 and today['Close'] < today['Open']:
                        breakout_prob, target_proj = "高機率破底", f"下測 **{round(sup_level - box_height, 1)}**"
                    else:
                        breakout_prob, target_proj = "機率中等 (量縮)", f"支撐位 {round(sup_level, 1)} 防守戰"

                st.subheader(f"🧬 {target_ticker} {c_name} 深度量化診斷報告")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("當前現價", f"{today['Close']:.2f}", f"{p_change:+.2f}%")
                m2.metric("即時量比", f"{vol_ratio:.1f}x", f"今日成交 {int(today['Volume']):,} 張", delta_color="off")
                m3.metric("AI 綜合評分", f"{int(today.get('Score', 0))} 分")
                m4.metric("大盤相對強度", f"{today.get('RS_Index', 0)*100:+.2f}%")
                
                st.markdown("---")
                t1, t2, t3, t4 = st.tabs(["🧱 測幅與策略推演", "🔍 前向驗證 (昨日對今日)", "🕵️‍♂️ 籌碼動向矩陣", "📰 專屬新聞動態"])
                
                with t1:
                    c_l, c_r = st.columns(2)
                    with c_l:
                        st.markdown("#### 📐 關鍵結構水位與型態")
                        st.write(f"- **前高壓力 (近20日):** {res_level:.2f} | **已測試:** {res_tests} 次")
                        st.write(f"- **前低支撐 (近20日):** {sup_level:.2f} | **已測試:** {sup_tests} 次")
                        st.write(f"- **盤勢型態判定:** {breakout_status}")
                        st.write(f"- **突破機率評估:** {breakout_prob}")
                        st.write(f"- **波動壓縮狀態 (Squeeze):** {'⚠️ 處於極度擠壓收斂期 (即將噴發)' if today.get('Squeeze_On', False) else '🟢 波動度處於常態分佈'}")
                        st.write(f"- **大週期週線共振:** {'📈 週線處於波段多頭保護期' if today.get('Weekly_Trend_Up', False) else '📉 週線空頭趨勢壓制'}")
                    with c_r:
                        st.markdown("#### 💡 操作劇本規劃")
                        st.write(f"**🔴 漲停極限:** {round(yesterday['Close'] * 1.10, 1)} | **🟢 跌停極限:** {round(yesterday['Close'] * 0.90, 1)}")
                        st.write(f"**等距測幅 (目標):** {target_proj}")
                        
                        if "突破前高" in breakout_status: st.success("🚀 型態正式帶量向上突破！屬強勢追隨訊號，防守點位移至前高。")
                        elif "挑戰前高" in breakout_status: st.warning("⚔️ 兵臨城下，即將挑戰壓力！若帶量突破布林上軌可試單。")
                        elif "回測前低" in breakout_status: st.warning("🛡️ 測試底部支撐，若跌破前低應果斷停損。")
                        elif "摜破前低" in breakout_status: st.error("⚠️ 破底危機！嚴格執行停損，切勿攤平。")
                        else: st.info("⏸️ 股價處於箱體內部結構震盪，採取下軌附近低吸、上軌調節之區間策略。")
                
                with t2:
                    st.markdown("### 🔍 昨日策略劇本與今日實況對撞")
                    y_res, y_sup, y_atr = today['Res_20'], today['Sup_20'], yesterday['ATR_14']
                    y_target = y_res + y_atr

                    col_r1, col_r2 = st.columns(2)
                    with col_r1: st.info(f"**昨日盤後預測基準**\n- 壓力位: **{y_res:.1f}**\n- 支撐位: **{y_sup:.1f}**\n- 測幅目標: **{y_target:.1f}**")
                    with col_r2: st.warning(f"**今日實況極值**\n- 最高價: **{today['High']:.1f}**\n- 最低價: **{today['Low']:.1f}**\n- 收盤現價: **{today['Close']:.1f}**")

                    if today['Close'] > y_res:
                        if today['High'] >= y_target: st.success(f"⭐⭐⭐ **超前達標**：今日強勢突破壓力位 {y_res:.1f}，成功觸及測幅目標 {y_target:.1f}！")
                        else: st.success(f"⭐⭐ **突破確認**：今日收盤 {today['Close']:.1f} 站上壓力位 {y_res:.1f}，多頭正式發動。")
                    elif today['Close'] < y_sup:
                        st.error(f"⚠️ **跌破防線**：今日收盤 {today['Close']:.1f} 跌破支撐位 {y_sup:.1f}，觸發停損機制。")
                    elif today['High'] >= y_res and today['Close'] <= y_res:
                        st.warning(f"👀 **假突破 / 壓力沉重**：今日盤中突破壓力，但收盤未能站穩。")
                    elif today['Low'] <= y_sup and today['Close'] >= y_sup:
                        st.info(f"🛡️ **支撐有守 (破底翻)**：今日下探支撐，但獲得買盤承接拉回。")
                    else:
                        st.write(f"⏸️ **區間震盪**：走勢在預設箱體 ({y_sup:.1f} ~ {y_res:.1f}) 內震盪，符合觀望預期。")
                
                with t3:
                    st.markdown("#### 🕵️‍♂️ 法人大戶籌碼控盤度矩陣")
                    sm = int(today.get('Smart_Money_Trend', 0))
                    chip_txt = "👽 外資/大戶積極建倉中 (價漲量增)" if sm >= 1 else ("🚶 散戶接盤/大戶出貨 (價跌量增)" if sm <= -1 else "⚖️ 籌碼無明確方向，量能萎縮")
                    st.info(f"**智能籌碼動向判定:** {chip_txt}")
                    st.progress(int(today.get('Score', 0)), text=f"量化綜合控盤度：{int(today.get('Score', 0))}%")
                    st.caption("基於價量背離、RS 指標與週線多時區共振加權推算。分數越高，代表法蘭與大戶資金沉澱度越高。")
                
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
                        
        except Exception as e:
            st.error(f"量化引擎運算碰撞錯誤: {e}")
            
        if st.button("⬅️ 返回戰情室主頁", use_container_width=True):
            st.session_state.analyze_trigger = None
            st.rerun()

else:
    # ─── 【模組 B】主頁儀表板 ───
    st.markdown("### 🌍 台股大盤摘要")
    summary = get_market_summary()
    if summary:
        cols = st.columns(len(summary))
        for i, (name, data) in enumerate(summary.items()): cols[i].metric(name, f"{data['price']:.2f}", f"{data['change']:+.2f} ({data['pct']:+.2f}%)")
        st.markdown("""<style>[data-testid="stMetricDelta"] svg { display: none; } [data-testid="stMetricDelta"] > div { flex-direction: row; } [data-testid="stMetricDelta"] > div:has(div:contains("+")) { color: #ff4b4b !important; } [data-testid="stMetricDelta"] > div:has(div:contains("-")) { color: #00cc96 !important; }</style>""", unsafe_allow_html=True)
    
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊 板塊實時監控", "⚡ 條件設定全市場雷達", "🎯 多因子 AI 評分 (含籌碼)"])
    
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
            
            res = run_robust_market_scan(tickers, conds, p_bar, s_text, st.session_state.stock_names, get_market_index_data(), "radar")
            s_text.empty(); p_bar.empty()
            
            if res: st.dataframe(pd.DataFrame(res), use_container_width=True)
            else: st.warning("⚠️ 查無符合標的，請嘗試放寬條件。")

    with tab3:
        st.markdown("#### 🎯 多因子演算法綜合評分排行榜 (TOP 20)")
        st.caption("結合 日/週雙時區共振、大盤相對強度矩陣 (RS) 及 大戶籌碼估算模型 進行深度排行。")
        
        with st.expander("ℹ️ 了解 AI 多因子評分標準 (滿分 100 分)"):
            st.markdown("""
            本系統採用 **7 大核心量化因子** 進行權重計分，尋找市場上具備「多頭共振」的最強勢標的：
            
            * **📈 趨勢防護 (10分)**：收盤價必須站上 20 日均線 (月線)，確保處於多頭基底。
            * **🔥 量能發動 (15分)**：今日成交量大於近 5 日均量的 1.5 倍，顯示資金實質進駐。
            * **📊 動能金叉 (15分)**：日線 MACD 大於 Signal 線，確認短波段動能向上。
            * **🚀 領漲抗跌 (15分)**：近 20 日相對大盤強度 (RS Index) 大於 0.02 (贏過大盤 2%)。
            * **🌍 週線共振 (15分)**：模擬週線級別 MACD 處於多頭，提供大週期趨勢保護。
            * **💥 壓縮突破 (15分)**：布林與肯特納通道之 Squeeze 狀態解除並向上突破上軌。
            * **👽 籌碼動能 (15分)**：近 5 日「聰明錢指標 (價漲量增/價跌量縮)」累積大於 1，顯示大戶偏多操作。
            """)
        
        if st.button("🔮 執行全權重深度矩陣運算", type="primary"):
            custom = [t for g in st.session_state.stock_clusters.values() for t in g]
            tickers = list(set(custom + (csv_df['Ticker'].tolist() if not csv_df.empty else [])))
            p_bar, s_text = st.progress(0), st.empty()
            
            res = run_robust_market_scan(tickers, {}, p_bar, s_text, st.session_state.stock_names, get_market_index_data(), "score")
            s_text.empty(); p_bar.empty()
            
            if res: 
                df_res = pd.DataFrame(res).sort_values("量化總分", ascending=False).head(20)
                # 🛡️ 安全渲染表格，絕不使用 index=False 觸發 TypeError
                st.dataframe(
                    df_res, 
                    column_config={"量化總分": st.column_config.ProgressColumn("多空綜合能量", min_value=0, max_value=100, format="%d 分")}, 
                    use_container_width=True
                )
            else: st.warning("⚠️ 運算失敗，無法取得足夠數據。")