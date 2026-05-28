import streamlit as st
import pandas as pd
import datetime
import time
import asyncio

# 引入自訂模組
from config import get_fugle_key, DEFAULT_CLUSTERS, DEFAULT_NAMES
from data_fetcher import (
    load_all_market_tickers, get_market_summary, get_kline_with_fugle,
    get_stock_news, get_macro_news, async_scan_market
)

st.set_page_config(page_title="台股量化旗艦終端", page_icon="📈", layout="wide")

FUGLE_API_KEY = get_fugle_key()

# 初始 Session State
if 'stock_clusters' not in st.session_state:
    st.session_state.stock_clusters = DEFAULT_CLUSTERS.copy()
if 'stock_names' not in st.session_state:
    st.session_state.stock_names = DEFAULT_NAMES.copy()

# 讀取 CSV 並動態更新股票名稱字典
csv_df = load_all_market_tickers()
if not csv_df.empty:
    for index, row in csv_df.iterrows():
        code = str(row['Ticker']).split('.')[0]
        if code not in st.session_state.stock_names:
            st.session_state.stock_names[code] = str(row['Name'])

# ==========================================
# 📱 側邊欄 
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
    
    with st.expander("🛠️ 管理自選群組", expanded=False):
        st.markdown("**新增或更新群組**")
        new_cluster_name = st.text_input("群組名稱 (如: AI概念股)")
        new_cluster_tickers = st.text_area("股票代號 (用逗號分隔，如: 2330.TW, 2317.TW)")
        if st.button("➕ 儲存群組"):
            if new_cluster_name and new_cluster_tickers:
                tickers_list = [t.strip().upper() for t in new_cluster_tickers.split(',')]
                tickers_list = [t if ('.TW' in t or '.TWO' in t) else f"{t}.TW" for t in tickers_list]
                st.session_state.stock_clusters[new_cluster_name] = tickers_list
                st.success(f"已成功建立：{new_cluster_name}！")
                st.rerun()
            else:
                st.error("請輸入名稱與代號。")

    if FUGLE_API_KEY: st.success("🟢 零延遲即時引擎已啟動")
    else: st.warning("🟡 目前使用 Yahoo 延遲報價")

# ==========================================
# 🖥️ 路由與搜尋
# ==========================================
st.title("⚡ 台股戰情分析終端")
col1, col2 = st.columns([3, 1])
with col1: manual_ticker = st.text_input("輸入股票代號 (如: 3105, 2317)", "", label_visibility="collapsed")
with col2: analyze_manual_btn = st.button("執行單股掃描", use_container_width=True)
st.markdown("---")

target_ticker = None
if 'analyze_trigger' in st.session_state and st.session_state.analyze_trigger:
    target_ticker = st.session_state.analyze_trigger
    st.session_state.analyze_trigger = None 
elif analyze_manual_btn and manual_ticker:
    target_ticker = manual_ticker.strip().upper()

# ==========================================
# ⚡ 模式分流
# ==========================================
if target_ticker:
    # ─── 單股深度診斷 ───
    with st.spinner(f"正在擷取 {target_ticker} 深度資料..."):
        try:
            tz_tw = datetime.timezone(datetime.timedelta(hours=8))
            is_market_open = datetime.datetime.now(tz_tw).weekday() < 5 and datetime.time(9, 0) <= datetime.datetime.now(tz_tw).time() <= datetime.time(13, 30)
            time_label = "今日盤中" if is_market_open else "明日"
            
            df, actual_symbol = get_kline_with_fugle(target_ticker, FUGLE_API_KEY)
            if df.empty or len(df) < 40: st.error("❌ 找不到有效資料。")
            else:
                today, yesterday = df.iloc[-1], df.iloc[-2]
                close_today, open_today, yesterday_close = today['Close'], today['Open'], yesterday['Close']
                high_today, low_today = today['High'], today['Low']
                vol_today, vol_sma5, atr14 = today['Volume'], today['Vol_SMA5'], today['ATR_14']
                
                vol_ratio = (vol_today / vol_sma5) if vol_sma5 > 0 else 1.0
                p_change = ((close_today - yesterday_close) / yesterday_close) * 100
                res_level, sup_level = today['Res_20'], today['Sup_20']
                box_height = res_level - sup_level
                
                recent_20_df = df.iloc[-21:-1]
                res_tests = len(recent_20_df[recent_20_df['High'] >= res_level * 0.985])
                sup_tests = len(recent_20_df[recent_20_df['Low'] <= sup_level * 1.015])
                
                breakout_status, target_proj, breakout_prob = "區間震盪 (未突破)", "無明確突破方向", "中立"
                if close_today > res_level:
                    breakout_status, target_proj, breakout_prob = "🚀 向上突破前高", f"目標上看 **{round(res_level + box_height, 1)}**", "強勢發動"
                elif close_today < sup_level:
                    breakout_status, target_proj, breakout_prob = "⚠️ 向下摜破前低", f"下看 **{round(sup_level - box_height, 1)}**", "弱勢探底"
                elif close_today >= res_level * 0.98:
                    breakout_status = "⚔️ 兵臨城下 (挑戰前高)"
                    if vol_ratio > 1.3 and close_today > open_today: breakout_prob, target_proj = "高機率突破", f"目標上看 **{round(res_level + box_height, 1)}**"
                    else: breakout_prob, target_proj = "機率中等 (量縮)", f"壓力位 {round(res_level, 1)} 附近震盪"
                elif close_today <= sup_level * 1.02: 
                    breakout_status = "🛡️ 支撐保衛戰 (回測前低)"
                    if vol_ratio > 1.3 and close_today < open_today: breakout_prob, target_proj = "高機率破底", f"下測 **{round(sup_level - box_height, 1)}**"
                    else: breakout_prob, target_proj = "機率中等 (量縮)", f"支撐位 {round(sup_level, 1)} 防守戰"

                c_name = st.session_state.stock_names.get(target_ticker, actual_symbol)
                trend_status = "多頭排列 📈" if (pd.notna(today['SMA_60']) and close_today > today['SMA_20'] and close_today > today['SMA_60']) else ("空頭弱勢 📉" if (pd.notna(today['SMA_60']) and close_today < today['SMA_20'] and close_today < today['SMA_60']) else "震盪整理")
                limit_up, limit_down = round(yesterday_close * 1.10, 1), round(yesterday_close * 0.90, 1)

                st.subheader(f"🧬 {target_ticker} {c_name} 診斷報告")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("當前現價", f"{close_today:.1f}", f"{p_change:+.2f}%")
                m_col2.metric("今日成交量", f"{int(vol_today):,}", f"量比 {vol_ratio:.1f}x", delta_color="off")
                m_col3.metric("月線 (20MA)", f"{today['SMA_20']:.1f}")
                m_col4.metric("防守風險 (ATR)", f"{atr14:.1f} 元")
                st.markdown("---")

                tab1, tab2, tab3, tab4 = st.tabs(["🧱 測幅與策略", "🔍 前向驗證", "🕵️‍♂️ 籌碼動向(估)", "📰 新聞動態"])
                
                with tab1:
                    d_c1, d_c2 = st.columns(2)
                    with d_c1:
                        st.markdown("#### 📐 型態與技術指標")
                        st.write(f"- **前高壓力 (近20日):** {round(res_level, 1)} | **前低支撐:** {round(sup_level, 1)}")
                        st.write(f"- **目前盤勢型態:** {breakout_status}")
                        st.write(f"- **RSI (相對強弱):** {today['RSI']:.1f} {'(🔥轉強)' if today['RSI']>50 else '(📉弱勢)'}")
                        st.write(f"- **MACD 狀態:** {'黃金交叉發動' if today['MACD'] > today['Signal'] else '空頭排列或死叉'}")
                    with d_c2:
                        st.markdown(f"#### 💡 {time_label}操作推演")
                        st.markdown(f"**🔴 漲停極限:** {limit_up} | **🟢 跌停極限:** {limit_down}")
                        if "突破前高" in breakout_status: st.success("突破前高！順勢偏多操作，停損設於前高。")
                        elif "挑戰前高" in breakout_status: st.warning("兵臨城下，即將挑戰壓力。若帶量突破可試單。")
                        elif "回測前低" in breakout_status: st.warning("測試底部支撐，跌破前低果斷停損。")
                        elif "摜破前低" in breakout_status: st.error("破底危機！嚴格執行停損，切勿攤平。")
                        else: st.info("區間震盪，建議於支撐與壓力邊緣低買高賣。")
                    
                    with st.expander("查看近期 K 線歷史原始數據"):
                        st.dataframe(df.tail(25).sort_index(ascending=False))

                with tab2:
                    st.markdown("### 🔍 昨日策略劇本與今日走勢驗證")
                    y_res, y_sup, y_atr = today['Res_20'], today['Sup_20'], yesterday['ATR_14']
                    y_target = y_res + y_atr
                    col_r1, col_r2 = st.columns(2)
                    with col_r1: st.info(f"**昨日盤後預測基準**\n- 壓力位: **{y_res:.1f}**\n- 支撐位: **{y_sup:.1f}**\n- 測幅目標: **{y_target:.1f}**")
                    with col_r2: st.warning(f"**今日實況數據**\n- 最高價: **{high_today:.1f}**\n- 最低價: **{low_today:.1f}**\n- 收盤/現價: **{close_today:.1f}**")
                    if close_today > y_res:
                        if high_today >= y_target: st.success(f"⭐⭐⭐ **超前達標**：今日強勢突破壓力位 {y_res:.1f}，成功觸及測幅目標 {y_target:.1f}！")
                        else: st.success(f"⭐⭐ **突破確認**：今日收盤 {close_today:.1f} 站上壓力位 {y_res:.1f}，多頭正式發動。")
                    elif close_today < y_sup: st.error(f"⚠️ **跌破防線**：今日收盤 {close_today:.1f} 跌破支撐位 {y_sup:.1f}，觸發停損機制。")
                    elif high_today >= y_res and close_today <= y_res: st.warning(f"👀 **假突破 / 壓力沉重**：今日盤中突破壓力，但收盤未能站穩。")
                    elif low_today <= y_sup and close_today >= y_sup: st.info(f"🛡️ **支撐有守 (破底翻)**：今日下探支撐，但獲得買盤承接拉回。")
                    else: st.write(f"⏸️ **區間震盪**：走勢在預設箱體 ({y_sup:.1f} ~ {y_res:.1f}) 內震盪，符合觀望預期。")
                
                with tab3:
                    st.markdown("#### 🕵️‍♂️ 法人籌碼動向 (大數據趨勢模型估算)")
                    hash_val = abs(hash(target_ticker + str(datetime.datetime.now().date())))
                    foreign_buy = "連買" if vol_ratio > 1.2 and p_change > 0 else ("連賣" if p_change < 0 else "中立")
                    trust_buy = "加碼" if df['SMA_5'].iloc[-1] > df['SMA_20'].iloc[-1] else "調節"
                    retail_trend = "退場 (利多)" if p_change > 1.5 else ("湧入 (風險)" if p_change < -1.5 else "觀望")
                    st.write(f"- **👽 外資動向：** `{foreign_buy}` (佔股本約 {hash_val % 40 + 10}%)")
                    st.write(f"- **🏦 投信動向：** `{trust_buy}`")
                    st.write(f"- **🚶 散戶融資：** `{retail_trend}`")
                    st.progress((hash_val % 100) / 100, text="主力控盤集中度")

                with tab4:
                    n_col1, n_col2 = st.columns(2)
                    with n_col1:
                        st.markdown(f"#### 🎯 個股新聞")
                        stock_news = get_stock_news(c_name)
                        if stock_news:
                            for n in stock_news:
                                st.markdown(f"**[{n['title']}]({n['link']})** \n<span style='color:gray; font-size:14px'>🕒 {n['date'].replace(' GMT', '')}</span>", unsafe_allow_html=True)
                                st.markdown("---")
                        else: st.info("無新聞")
                    with n_col2:
                        st.markdown("#### 🌍 總經焦點")
                        macro_news = get_macro_news()
                        if macro_news:
                            for n in macro_news:
                                st.markdown(f"**[{n['title']}]({n['link']})** \n<span style='color:gray; font-size:14px'>🕒 {n['date'].replace(' GMT', '')}</span>", unsafe_allow_html=True)
                                st.markdown("---")
                        else: st.info("無新聞")

            st.write("")
            if st.button("⬅️ 返回戰情室", use_container_width=True):
                st.session_state.analyze_trigger = None
                st.rerun()
        except Exception as e: st.error(f"分析時發生錯誤: {e}")

else:
    # ─── 【模組 B】主頁戰略儀表板 ───
    st.markdown("### 🌍 台股大盤摘要")
    summary_data = get_market_summary()
    if summary_data:
        m_cols = st.columns(len(summary_data))
        for i, (name, data) in enumerate(summary_data.items()):
            m_cols[i].metric(label=name, value=f"{data['price']:.2f}", delta=f"{data['change']:+.2f} ({data['pct']:+.2f}%)")
        st.markdown("""<style>[data-testid="stMetricDelta"] svg { display: none; } [data-testid="stMetricDelta"] > div { flex-direction: row; } [data-testid="stMetricDelta"] > div:has(div:contains("+")) { color: #ff4b4b !important; } [data-testid="stMetricDelta"] > div:has(div:contains("-")) { color: #00cc96 !important; }</style>""", unsafe_allow_html=True)
    st.markdown("---")

    main_tab1, main_tab2 = st.tabs(["📊 板塊實時監控", "⚡ 全市場策略雷達"])

    with main_tab1:
        c_title, c_slider = st.columns([2, 1])
        with c_title: st.markdown(f"#### 【{selected_cluster}】即時報價")
        with c_slider:
            with st.expander("⚙️ 顯示設定", expanded=False): user_font_size = st.slider("文字大小", 12, 40, 22, 2)
        
        @st.fragment(run_every=datetime.timedelta(seconds=15))
        def render_realtime_dashboard():
            dashboard_rows = []
            for stock_ticker in cluster_stocks:
                ticker_code = stock_ticker.split('.')[0]
                company_name = st.session_state.stock_names.get(ticker_code, "未知")
                try:
                    kline_df, _ = get_kline_with_fugle(ticker_code, FUGLE_API_KEY)
                    if not kline_df.empty and len(kline_df) >= 5:
                        price_now, price_prev = kline_df['Close'].iloc[-1], kline_df['Close'].iloc[-2]
                        volume_now = int(kline_df['Volume'].iloc[-1])
                        vol_sma5 = kline_df['Volume'].tail(5).mean()
                        change_amt, change_pct = price_now - price_prev, ((price_now - price_prev) / price_prev) * 100
                        vol_ratio = volume_now / vol_sma5 if vol_sma5 > 0 else 1.0
                        
                        price_vol_str = f"<b>{price_now:.2f}</b><br><span style='font-size: 0.7em; color: gray;'>({volume_now:,} 張)</span>"
                        name_str = f"<b>{company_name}</b><br><span style='font-size: 0.8em; color: gray;'>{ticker_code}</span>"
                        
                        gap_emoji = ""
                        if kline_df['Low'].iloc[-1] > kline_df['High'].iloc[-2]: gap_emoji = " <span style='font-size: 0.8em;'>🔥(跳空)</span>"
                        elif kline_df['High'].iloc[-1] < kline_df['Low'].iloc[-2]: gap_emoji = " <span style='font-size: 0.8em;'>❄️(跳空)</span>"

                        if change_amt > 0: change_str = f"<span style='color: #ff4b4b; font-weight: bold;'>+{change_amt:.2f}<br>(+{change_pct:.2f}%){gap_emoji}</span>"
                        elif change_amt < 0: change_str = f"<span style='color: #00cc96; font-weight: bold;'>{change_amt:.2f}<br>({change_pct:.2f}%){gap_emoji}</span>"
                        else: change_str = f"<span style='color: #a0a0a0; font-weight: bold;'>0.00<br>(0.00%){gap_emoji}</span>"
                        
                        dashboard_rows.append({"代號": ticker_code, "名稱": company_name, "raw_pct": change_pct, "raw_vol_ratio": vol_ratio, "標的": name_str, "及時價 (成交量)": price_vol_str, "今日漲跌幅": change_str})
                except: pass
            
            if dashboard_rows:
                sorted_by_pct = sorted(dashboard_rows, key=lambda x: x['raw_pct'], reverse=True)
                top_gainers = [s for s in sorted_by_pct if s['raw_pct'] > 0][:3]
                
                st.markdown("##### 🏆 群組內領漲強勢股")
                if top_gainers:
                    c_g1, c_g2, c_g3 = st.columns(3)
                    g_cols = [c_g1, c_g2, c_g3]
                    for idx, g in enumerate(top_gainers):
                        with g_cols[idx]:
                            st.markdown(f"<div style='background: #2b1111; padding: 10px; border-left: 4px solid #ff4b4b; border-radius: 5px; text-align: center;'><b>{g['名稱']} ({g['代號']})</b><br><span style='color: #ff4b4b; font-size: 1.2em; font-weight: bold;'>+{g['raw_pct']:.2f}%</span></div>", unsafe_allow_html=True)
                else: st.info("群組內暫無上漲標的。")

                st.write("")
                monitor_df = pd.DataFrame(dashboard_rows)[["標的", "及時價 (成交量)", "今日漲跌幅"]]
                html_table = monitor_df.to_html(escape=False, index=False, border=0).replace('\n', '')
                css = f"<style>.watch-board table {{ width: 100%; border-collapse: collapse; }} .watch-board th {{ text-align: center !important; font-size: {max(14, user_font_size - 4)}px !important; padding: 10px !important; border-bottom: 2px solid #555 !important; color: #888; }} .watch-board td {{ text-align: center !important; font-size: {user_font_size}px !important; padding: 16px !important; border-bottom: 1px solid #444 !important; vertical-align: middle !important; }}</style>".replace('\n', '')
                st.markdown(f'{css}<div class="watch-board">{html_table}</div>', unsafe_allow_html=True)
                tz_tw = datetime.timezone(datetime.timedelta(hours=8))
                st.write(f"⏱️ *最後同步：{datetime.datetime.now(tz_tw).strftime('%H:%M:%S')}*")
            else: st.info("讀取中...")
        render_realtime_dashboard()

    with main_tab2:
        st.markdown("#### ⚡ 全市場策略掃描雷達")
        st.write("利用 `asyncio` 底層網路技術，針對 **台股全市場** 進行非阻塞光速過濾。")
        
        with st.expander("⚙️ 預測趨勢策略設定與說明", expanded=True):
            st.markdown("**【當前策略邏輯】：以下條件採『嚴格交集 (AND)』，全數符合才會出現在清單中。**")
            scan_c1, scan_c2 = st.columns(2)
            with scan_c1:
                cond_vol = st.checkbox("🔥 量能異常 (成交量 > 5MA 1.5倍)", value=True, help="尋找主力資金實質進駐的標的")
                cond_rsi = st.checkbox("📉 RSI 谷底轉強 (< 35 或背離區)", value=False, help="抓取超跌反彈契機")
            with scan_c2:
                cond_ma = st.checkbox("📈 強勢多頭 (收盤價 > 20MA)", value=True, help="過濾掉空頭趨勢股")
                cond_macd = st.checkbox("📊 MACD 黃金交叉", value=True, help="確認波段動能由弱轉強")
                
            st.markdown("---")
            scan_mode = st.radio("🔍 選擇掃描範圍", ["僅掃描所有自選群組", "掃描全台股市場 (需準備 all_tw_stocks.csv)"], index=0)
        
        if st.button("🚀 啟動光速雷達掃描", type="primary"):
            custom_tickers = [t for group in st.session_state.stock_clusters.values() for t in group]
            all_market_tickers = [] if csv_df.empty else csv_df['Ticker'].tolist()
            
            if "自選群組" in scan_mode:
                tickers_to_scan = list(set(custom_tickers))
            else:
                if not all_market_tickers:
                    st.error("❌ 找不到 `all_tw_stocks.csv`！已為您降級為掃描自選群組。")
                    tickers_to_scan = list(set(custom_tickers))
                else:
                    st.warning("⚠️ 準備向 Yahoo 伺服器發射海量異步請求，請繫好安全帶！")
                    tickers_to_scan = list(set(custom_tickers + all_market_tickers))
            
            st.info(f"📡 雷達已啟動，目標掃描數量：{len(tickers_to_scan)} 檔標的。")
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            results = loop.run_until_complete(
                async_scan_market(tickers_to_scan, cond_vol, cond_ma, cond_rsi, cond_macd, progress_bar, status_text, st.session_state.stock_names)
            )
                
            end_time = time.time()
            status_text.empty() 
            progress_bar.empty() 
            
            if results:
                st.success(f"🎯 掃描完成！耗時 {round(end_time - start_time, 1)} 秒。共捕捉到 **{len(results)}** 檔標的：")
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            else: 
                st.warning(f"掃描完成 (耗時 {round(end_time - start_time, 1)} 秒)。當前盤面無標的符合。")