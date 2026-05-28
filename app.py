import streamlit as st
import pandas as pd
import datetime
import time
import asyncio

from config import get_fugle_key, DEFAULT_CLUSTERS, DEFAULT_NAMES
from data_fetcher import (
    load_all_market_tickers, get_market_index_data, get_market_summary, 
    get_kline_with_fugle, get_stock_news, get_macro_news, async_scan_market
)

# 初始化 Session 狀態快取避免重刷洗掉使用者資料
if 'stock_clusters' not in st.session_state:
    st.session_state.stock_clusters = DEFAULT_CLUSTERS.copy()
if 'stock_names' not in st.session_state:
    st.session_state.stock_names = DEFAULT_NAMES.copy()

FUGLE_API_KEY = get_fugle_key()

# 預載入全市場 CSV 資料表
csv_df = load_all_market_tickers()
if not csv_df.empty:
    for index, row in csv_df.iterrows():
        code = str(row['Ticker']).split('.')[0]
        if code not in st.session_state.stock_names:
            st.session_state.stock_names[code] = str(row['Name'])

# --- 側邊欄控制面板 ---
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
        st.markdown("**動態建立/更新群組**")
        new_cluster_name = st.text_input("群組名稱 (如: 半導體飆股)")
        new_cluster_tickers = st.text_area("股票代號 (逗號分隔，如: 2330, 2454, 3034)")
        if st.button("➕ 儲存群組配置"):
            if new_cluster_name and new_cluster_tickers:
                raw_list = [t.strip().upper() for t in new_cluster_tickers.split(',')]
                processed_list = [t if ('.TW' in t or '.TWO' in t) else (f"{t}.TWO" if len(t)==4 and t.startswith(('3', '4', '5', '6', '8')) else f"{t}.TW") for t in raw_list]
                st.session_state.stock_clusters[new_cluster_name] = processed_list
                st.success(f"群組【{new_cluster_name}】配置成功！")
                st.rerun()

# --- 主路由邏輯分流 ---
st.markdown("##### 🔍 搜尋個股詳細技術診斷報告")
col1, col2 = st.columns([3, 1])
with col1: manual_ticker = st.text_input("輸入股票代號 (如: 3491, 2330)", "", label_visibility="collapsed")
with col2: analyze_manual_btn = st.button("執行單股掃描", use_container_width=True)
st.markdown("---")

target_ticker = None
if 'analyze_trigger' in st.session_state and st.session_state.analyze_trigger:
    target_ticker = st.session_state.analyze_trigger
    st.session_state.analyze_trigger = None 
elif analyze_manual_btn and manual_ticker:
    target_ticker = manual_ticker.strip().upper()

if target_ticker:
    # ─── 【模組 A】單股深度診斷模式 ───
    with st.spinner(f"正在調用量化矩陣分析 {target_ticker}..."):
        try:
            df, actual_symbol = get_kline_with_fugle(target_ticker, FUGLE_API_KEY)
            if df.empty or len(df) < 35: st.error("❌ 該標的數據深度不足，無法執行複雜演算法。")
            else:
                today, yesterday = df.iloc[-1], df.iloc[-2]
                vol_ratio = (today['Volume'] / today['Vol_SMA5']) if today['Vol_SMA5'] > 0 else 1.0
                p_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100
                
                st.subheader(f"🧬 {target_ticker} {st.session_state.stock_names.get(target_ticker, actual_symbol)} 深度量化診斷")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("當前收盤價", f"{today['Close']:.2f}", f"{p_change:+.2f}%")
                m2.metric("即時量比", f"{vol_ratio:.1f}x", f"今日成交 {int(today['Volume'])} 張", delta_color="off")
                m3.metric("AI 綜合多空評分", f"{int(today['Score'])} 分")
                m4.metric("大盤相對強度 (RS)", f"{today['RS_Index']*100:+.2f}%")
                
                st.markdown("---")
                t1, t2, t3, t4 = st.tabs(["🧱 策略測幅推演", "🔍 昨日預測對撞驗證", "🕵️‍♂️ 籌碼模擬擬合", "📰 專屬即時公告"])
                
                with t1:
                    c_l, c_r = st.columns(2)
                    with c_l:
                        st.markdown("#### 📐 關鍵結構水位")
                        st.write(f"- **前高關鍵阻力 (20日):** {today['Res_20']:.2f}")
                        st.write(f"- **前低關鍵支撐 (20日):** {today['Sup_20']:.2f}")
                        st.write(f"- **波動壓縮狀態 (Squeeze):** {'⚠️ 處於極度擠壓收斂期 (即將噴發)' if today['Squeeze_On'] else '🟢 波動度處於常態分佈'}")
                        st.write(f"- **大週期週線共振：** {'📈 週線處於波段多頭保護期' if today['Weekly_Trend_Up'] else '📉 週線空頭趨勢壓制'}")
                    with c_r:
                        st.markdown("#### 💡 操作劇本規劃")
                        box = today['Res_20'] - today['Sup_20']
                        st.write(f"**等距向上測幅目標：** {today['Res_20']+box:.2f}")
                        st.write(f"**波段防守支撐水位：** {today['Sup_20']:.2f}")
                        if today['Close'] > today['Res_20']: st.success("🚀 型態正式帶量向上突破！屬強勢追隨訊號，防守點位移至前高。")
                        elif today['Squeeze_On'] and today['Volume'] > today['Vol_SMA5']*1.3: st.warning("⚔️ Squeeze 臨界爆發點！若價格站穩布林上軌，大波段主升段將正式啟動。")
                        else: st.info("⏸️ 股價處於箱體內部結構震盪，採取下軌附近低吸、上軌調節之區間策略。")
                
                with t2:
                    st.markdown("### 🔍 前向測試 (Forward Testing) 策略回測對撞")
                    y_res, y_sup, y_atr = today['Res_20'], today['Sup_20'], yesterday['ATR_14']
                    st.info(f"**昨日盤後算定基準**：壓力位 **{y_res:.1f}** | 支撐位 **{y_sup:.1f}** | 測幅空間 **{y_res+y_atr:.1f}**")
                    st.warning(f"**今日實盤極值走勢**：最高價 **{today['High']:.1f}** | 最低價 **{today['Low']:.1f}** | 收盤現價 **{today['Close']:.1f}**")
                    if today['Close'] > y_res:
                        if today['High'] >= (y_res + y_atr): st.success("⭐⭐⭐ **超前達標**：今日精確突破昨日壓力，且最高價成功觸及等距測幅空間，多頭策略完美發動！")
                        else: st.success("⭐⭐ **突破確立**：收盤成功站穩突破臨界點，型態確認噴發。")
                    elif today['High'] >= y_res and today['Close'] <= y_res: st.warning("👀 **假突破 (主力誘多)**：盤中一度穿越壓力，但尾盤賣壓沉重未能站穩，多單應暫緩進場。")
                    else: st.write("⏸️ **結構未變**：今日高低點完全在預設結構箱體內波動，未脫離策略軌道。")
                
                with t3:
                    st.markdown("#### 🕵️‍♂️ 法人籌碼控盤度矩陣擬合")
                    st.progress(int(today['Score']), text=f"核心主力控盤集中度：{int(today['Score'])}%")
                    st.caption("基於 RS 指標與週線多時區共振加權推算。分數越高，代表法蘭與大戶資金沉澱度越高。")
                
                with t4:
                    nl, nr = st.columns(2)
                    with nl:
                        for n in get_stock_news(c_name)[:3]: st.markdown(f"**[{n['title']}]({n['link']})**\n`🕒 {n['date'].replace(' GMT','')}`")
                    with nr:
                        for n in get_macro_news()[:3]: st.markdown(f"**[{n['title']}]({n['link']})**\n`🕒 {n['date'].replace(' GMT','')}`")
            
            if st.button("⬅️ 返回戰情監控主頁", use_container_width=True):
                st.session_state.analyze_trigger = None
                st.rerun()
        except Exception as e: st.error(f"量化引擎運算碰撞錯誤: {e}")

else:
    # ─── 【模組 B】主頁戰情儀表板頁面 ───
    main_tab1, main_tab2, main_tab3 = st.tabs(["📊 板塊實時監控", "⚡ 全市場策略雷達", "🎯 多因子 AI 評分系統"])
    
    with main_tab1:
        c_title, c_slider = st.columns([2, 1])
        with c_title: st.markdown(f"#### 【{selected_cluster}】即時行情流")
        with c_slider:
            with st.expander("⚙️ 畫幅設定"): user_font_size = st.slider("表格文字大小", 12, 40, 22, 2)
            
        @st.fragment(run_every=datetime.timedelta(seconds=15))
        def render_realtime_dashboard():
            dashboard_rows = []
            for stock_ticker in cluster_stocks:
                ticker_code = stock_ticker.split('.')[0]
                try:
                    kline_df, _ = get_kline_with_fugle(ticker_code, FUGLE_API_KEY)
                    if not kline_df.empty and len(kline_df) >= 3:
                        c_day, p_day = kline_df.iloc[-1], kline_df.iloc[-2]
                        change_amt, change_pct = c_day['Close'] - p_day['Close'], ((c_day['Close'] - p_day['Close']) / p_day['Close']) * 100
                        gap = " <span style='color:#ff4b4b;font-size:0.7em;'>(跳空🔥)</span>" if c_day['Low'] > p_day['High'] else ""
                        
                        price_vol_str = f"<b>{c_day['Close']:.2f}</b><br><span style='font-size:0.7em;color:gray;'>({int(c_day['Volume']):,} 張)</span>"
                        name_str = f"<b>{st.session_state.stock_names.get(ticker_code, '市場個股')}</b><br><span style='font-size:0.8em;color:gray;'>{ticker_code}</span>"
                        change_str = f"<span style='color:#ff4b4b;font-weight:bold;'>+{change_amt:.2f}<br>(+{change_pct:.2f}%){gap}</span>" if change_amt > 0 else (f"<span style='color:#00cc96;font-weight:bold;'>{change_amt:.2f}<br>({change_pct:.2f}%){gap}</span>" if change_amt < 0 else f"<span style='color:gray;'>0.00<br>(0.00%)</span>")
                        
                        dashboard_rows.append({"標的": name_str, "及時價 (成交量)": price_vol_str, "今日漲跌幅": change_str, "raw_pct": change_pct})
                except: pass
            if dashboard_rows:
                sorted_df = sorted(dashboard_rows, key=lambda x: x['raw_pct'], reverse=True)
                st.markdown(f"<div style='background:#2b1111;padding:10px;border-left:4px solid #ff4b4b;border-radius:5px;margin-bottom:15px;'>🔥 <b>今日群組最強勢領頭羊：</b> {sorted_df[0]['標的'].split('<')[0].replace('<b>','')} ({sorted_df[0]['raw_pct']:+.2f}%)</div>", unsafe_allow_html=True)
                
                html_table = pd.DataFrame(dashboard_rows)[["標的", "及時價 (成交量)", "今日漲跌幅"]].to_html(escape=False, index=False, border=0).replace('\n', '')
                css = f"<style>.watch-board table {{ width:100%; }} .watch-board th {{ text-align:center !important; font-size:{max(14, user_font_size-4)}px; border-bottom:2px solid #555; color:#888; }} .watch-board td {{ text-align:center !important; font-size:{user_font_size}px; padding:12px !important; border-bottom:1px solid #444; vertical-align:middle !important; }}</style>".replace('\n', '')
                st.markdown(f'{css}<div class="watch-board">{html_table}</div>', unsafe_allow_html=True)
            else: st.info("同步流介接中...")
        render_realtime_dashboard()

    with main_tab2:
        st.markdown("#### ⚡ 全市場異步並行篩選雷達")
        with st.expander("⚙️ 演算法控制閥參數微調", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                cond_vol = st.checkbox("🔥 量能異常因子 (成交量 > 5MA 1.5倍)", value=True)
                cond_rsi = st.checkbox("📉 RSI 轉強因子 (RSI < 35 臨界超賣區間)", value=False)
            with c2:
                cond_ma = st.checkbox("📈 趨勢防護因子 (收盤價 > 20日MA月線)", value=True)
                cond_macd = st.checkbox("📊 MACD 黃金交叉動能因子", value=True)
            scan_mode = st.radio("選擇掃描射程範圍", ["僅限自選群組", "台股全市場上市上櫃池 (需準備 all_tw_stocks.csv)"])

        if st.button("🚀 啟動狂暴掃描矩陣", type="primary"):
            m_df = get_market_index_data()
            custom_list = [t for group in st.session_state.stock_clusters.values() for t in group]
            csv_list = csv_df['Ticker'].tolist() if not csv_df.empty else []
            tickers_to_scan = list(set(custom_list + csv_list)) if "全市場" in scan_mode else list(set(custom_list))
            
            if not tickers_to_scan: st.error("❌ 掃描池為空，請確認 CSV 檔案或自選設定。")
            else:
                p_bar, s_text = st.progress(0), st.empty()
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                res_list = loop.run_until_complete(async_scan_market(tickers_to_scan, cond_vol, cond_ma, cond_rsi, cond_macd, p_bar, s_text, st.session_state.stock_names, m_df))
                s_text.empty(); p_bar.empty()
                
                if res_list:
                    st.success(f"🎯 雷達掃描完成！符合當前篩選交集共 {len(res_list)} 檔標的：")
                    st.dataframe(pd.DataFrame(res_list)[["代號", "名稱", "現價", "今日漲跌", "量比", "RSI", "型態特特征"]], use_container_width=True)
                else: st.warning("當前盤面無任何標的通過此多重演算法過濾閥。")

    with main_tab3:
        # ✨ 新增：高階多因子評分系統分頁
        st.markdown("#### 🎯 多因子演算法綜合評分排行榜 (TOP 20)")
        st.caption("本系統採用權重計分制（最高 100 分），融合日/週雙時區共振、相對強度矩陣 (RS Index) 及布林/肯特納收斂臨界點進行全市場深度運算。")
        
        if st.button("🔮 執行全權重深度矩陣運算", key="factor_scoring_btn", type="primary"):
            m_df = get_market_index_data()
            custom_list = [t for group in st.session_state.stock_clusters.values() for t in group]
            csv_list = csv_df['Ticker'].tolist() if not csv_df.empty else []
            tickers_to_scan = list(set(custom_list + csv_list))
            
            p_bar, s_text = st.progress(0), st.empty()
            try: loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # 利用我們的高頻異步爬蟲，抓出全市場的完整 DF 並提取 Score
            connector = aiohttp.TCPConnector(limit=60)
            async def run_scoring():
                scored_results = []
                from data_fetcher import fetch_yahoo_history
                async with aiohttp.ClientSession(connector=connector) as session:
                    tasks = [fetch_yahoo_history(session, t) for t in tickers_to_scan]
                    comp = 0
                    for fut in asyncio.as_completed(tasks):
                        sym, df_single = await fut
                        comp += 1
                        if comp % 20 == 0 or comp == len(tasks):
                            p_bar.progress(comp / len(tasks))
                            s_text.text(f"🎯 矩陣運算中... 已完成 ({comp}/{len(tasks)})")
                        if df_single is not None:
                            from indicators import add_advanced_indicators
                            df_single = add_advanced_indicators(df_single, m_df)
                            if not df_single.empty:
                                last = df_single.iloc[-1]
                                code = sym.split('.')[0]
                                scored_results.append({
                                    "代號": code,
                                    "名稱": st.session_state.stock_names.get(code, "市場熱門"),
                                    "量化總分": int(last['Score']),
                                    "相對大盤強度": f"{last['RS_Index']*100:+.2f}%",
                                    "現價": f"{last['Close']:.2f}",
                                    "RSI": round(last['RSI'], 1),
                                    "週線趨勢": "🟢 多頭共振" if last['Weekly_Trend_Up'] else "🔴 趨勢壓制"
                                })
                return scored_results

            final_scores = loop.run_until_complete(run_scoring())
            s_text.empty(); p_bar.empty()
            
            if final_scores:
                score_df = pd.DataFrame(final_scores).sort_values(by="量化總分", ascending=False).head(20)
                st.success("🎉 全市場多因子矩陣權重演算法計算完成！")
                st.dataframe(
                    score_df,
                    column_config={
                        "量化總分": st.column_config.ProgressColumn("多空能量值", min_value=0, max_value=100, format="%d 分")
                    },
                    use_container_width=True,
                    index=False
                )
            else: st.warning("無法取得足夠數據進行矩陣運算。")