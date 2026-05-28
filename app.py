import streamlit as st
import pandas as pd
import datetime
import time

from config import get_fugle_key, DEFAULT_CLUSTERS, DEFAULT_NAMES
from data_fetcher import (
    load_all_market_tickers, get_market_index_data, get_market_summary, 
    get_kline_with_fugle, get_stock_news, get_macro_news, run_async_market_scan
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
# 📱 側邊欄控制與零延遲狀態顯示
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
    
    # 🌟 恢復零延遲系統狀態顯示
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
    # ─── 【模組 A】單股深度診斷模式 (全數復原) ───
    base_ticker = target_ticker.split('.')[0]
    c_name = st.session_state.stock_names.get(base_ticker, target_ticker)
    
    with st.spinner(f"矩陣分析 {target_ticker}..."):
        df, actual_symbol = get_kline_with_fugle(target_ticker, FUGLE_API_KEY)
        if df.empty or len(df) < 35: st.error("❌ 數據不足")
        else:
            today, yesterday = df.iloc[-1], df.iloc[-2]
            vol_ratio = (today['Volume'] / today['Vol_SMA5']) if today['Vol_SMA5'] > 0 else 1.0
            p_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100
            
            # 策略推演計算
            res_level, sup_level = today['Res_20'], today['Sup_20']
            box_height = res_level - sup_level
            breakout_status, target_proj, breakout_prob = "區間震盪 (未突破)", "無明確突破方向", "中立"
            if today['Close'] > res_level:
                breakout_status, target_proj, breakout_prob = "🚀 向上突破前高", f"目標上看 **{round(res_level + box_height, 1)}**", "強勢發動"
            elif today['Close'] < sup_level:
                breakout_status, target_proj, breakout_prob = "⚠️ 向下摜破前低", f"下看 **{round(sup_level - box_height, 1)}**", "弱勢探底"
            elif today['Close'] >= res_level * 0.98:
                breakout_status = "⚔️ 兵臨城下 (挑戰前高)"
                if vol_ratio > 1.3 and today['Close'] > yesterday['Close']: breakout_prob, target_proj = "高機率突破", f"目標上看 **{round(res_level + box_height, 1)}**"
                else: breakout_prob, target_proj = "機率中等 (量縮)", f"壓力位 {round(res_level, 1)} 附近震盪"
            elif today['Close'] <= sup_level * 1.02:
                breakout_status = "🛡️ 支撐保衛戰 (回測前低)"
                if vol_ratio > 1.3 and today['Close'] < yesterday['Close']: breakout_prob, target_proj = "高機率破底", f"下測 **{round(sup_level - box_height, 1)}**"
                else: breakout_prob, target_proj = "機率中等 (量縮)", f"支撐位 {round(sup_level, 1)} 防守戰"
            
            st.subheader(f"🧬 {target_ticker} {c_name} 深度量化診斷")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("當前現價", f"{today['Close']:.2f}", f"{p_change:+.2f}%")
            m2.metric("即時量比", f"{vol_ratio:.1f}x", f"成交 {int(today['Volume'])} 張", delta_color="off")
            m3.metric("AI 綜合評分", f"{int(today['Score'])} 分")
            m4.metric("大盤相對強度", f"{today['RS_Index']*100:+.2f}%")
            
            st.markdown("---")
            t1, t2, t3, t4 = st.tabs(["🧱 測幅與策略", "🔍 前向驗證", "🕵️‍♂️ 籌碼動向", "📰 新聞動態"])
            
            with t1:
                c_l, c_r = st.columns(2)
                with c_l:
                    st.markdown("#### 📐 關鍵結構水位")
                    st.write(f"- **前高關鍵阻力 (20日):** {res_level:.2f}")
                    st.write(f"- **前低關鍵支撐 (20日):** {sup_level:.2f}")
                    st.write(f"- **盤勢型態判定:** {breakout_status}")
                    st.write(f"- **波動壓縮狀態:** {'⚠️ 處於極度擠壓收斂期 (Squeeze)' if today['Squeeze_On'] else '🟢 波動度常態分佈'}")
                    st.write(f"- **大週期週線共振:** {'📈 週線波段多頭保護' if today['Weekly_Trend_Up'] else '📉 週線空頭趨勢壓制'}")
                with c_r:
                    st.markdown("#### 💡 操作劇本規劃")
                    limit_up = round(yesterday['Close'] * 1.10, 1)
                    limit_down = round(yesterday['Close'] * 0.90, 1)
                    st.write(f"**🔴 漲停極限:** {limit_up} | **🟢 跌停極限:** {limit_down}")
                    st.write(f"**等距測幅目標:** {target_proj}")
                    if "突破前高" in breakout_status: st.success("🚀 順勢偏多操作，停損防守點位移至前高。")
                    elif "挑戰前高" in breakout_status: st.warning("兵臨城下！若帶量突破布林上軌可嘗試建倉。")
                    elif "回測前低" in breakout_status: st.warning("測試底部支撐，若跌破前低應果斷停損。")
                    elif "摜破前低" in breakout_status: st.error("破底危機！嚴格執行停損，切勿攤平。")
                    else: st.info("⏸️ 箱體內部震盪，建議於支撐與壓力邊緣低買高賣。")
            
            with t2:
                st.markdown("### 🔍 昨日預測劇本 vs 今日實況對撞")
                y_res, y_sup, y_atr = today['Res_20'], today['Sup_20'], yesterday['ATR_14']
                y_target = y_res + y_atr
                col_r1, col_r2 = st.columns(2)
                with col_r1: st.info(f"**昨日盤後預測基準**\n- 壓力位: **{y_res:.1f}**\n- 支撐位: **{y_sup:.1f}**\n- 測幅目標: **{y_target:.1f}**")
                with col_r2: st.warning(f"**今日實況數據**\n- 最高價: **{today['High']:.1f}**\n- 最低價: **{today['Low']:.1f}**\n- 收盤現價: **{today['Close']:.1f}**")
                
                if today['Close'] > y_res:
                    if today['High'] >= y_target: st.success("⭐⭐⭐ **超前達標**：今日強勢突破壓力位，成功觸及測幅目標！")
                    else: st.success("⭐⭐ **突破確認**：今日收盤站上壓力位，多頭正式發動。")
                elif today['Close'] < y_sup: st.error("⚠️ **跌破防線**：今日收盤跌破支撐位，觸發停損機制。")
                elif today['High'] >= y_res and today['Close'] <= y_res: st.warning("👀 **假突破 / 壓力沉重**：盤中一度穿越壓力，但收盤未能站穩。")
                elif today['Low'] <= y_sup and today['Close'] >= y_sup: st.info("🛡️ **支撐有守 (破底翻)**：今日下探支撐，但獲得買盤承接拉回。")
                else: st.write("⏸️ **區間震盪**：走勢在預設箱體內震盪，符合觀望預期。")
            
            with t3:
                st.markdown("#### 🕵️‍♂️ 法人籌碼控盤度矩陣")
                st.progress(int(today['Score']), text=f"主力綜合控盤度：{int(today['Score'])}%")
                st.caption("基於 RS 指標與週線多時區共振加權推算。分數越高，代表法蘭與大戶資金沉澱度越高。")
            
            with t4:
                # 🌟 恢復新聞分類版面
                nl, nr = st.columns(2)
                with nl:
                    st.markdown("#### 🎯 個股專屬新聞")
                    news_s = get_stock_news(c_name)
                    if news_s:
                        for n in news_s[:4]: st.markdown(f"**[{n['title']}]({n['link']})**\n<span style='color:gray;font-size:14px;'>🕒 {n['date'].replace(' GMT','')}</span>", unsafe_allow_html=True)
                    else: st.info("無相關個股新聞")
                with nr:
                    st.markdown("#### 🌍 總經大盤焦點")
                    news_m = get_macro_news()
                    if news_m:
                        for n in news_m[:4]: st.markdown(f"**[{n['title']}]({n['link']})**\n<span style='color:gray;font-size:14px;'>🕒 {n['date'].replace(' GMT','')}</span>", unsafe_allow_html=True)
                    else: st.info("無大盤新聞")
                    
        if st.button("⬅️ 返回戰情室主頁", use_container_width=True): st.rerun()

else:
    # ─── 【模組 B】主頁儀表板 ───
    st.markdown("### 🌍 台股大盤摘要")
    summary = get_market_summary()
    if summary:
        cols = st.columns(len(summary))
        for i, (name, data) in enumerate(summary.items()): cols[i].metric(name, f"{data['price']:.2f}", f"{data['change']:+.2f} ({data['pct']:+.2f}%)")
        st.markdown("""<style>[data-testid="stMetricDelta"] svg { display: none; } [data-testid="stMetricDelta"] > div { flex-direction: row; } [data-testid="stMetricDelta"] > div:has(div:contains("+")) { color: #ff4b4b !important; } [data-testid="stMetricDelta"] > div:has(div:contains("-")) { color: #00cc96 !important; }</style>""", unsafe_allow_html=True)
    
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊 板塊實時監控", "⚡ 全市場策略雷達", "🎯 多因子 AI 評分系統"])
    
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
                
                # 🌟 恢復主頁頂部的強勢股三宮格紅框
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
                css = f"""
                <style>
                .watch-board {{ width: 100%; }}
                .watch-board table {{ width: 100% !important; border-collapse: collapse; }}
                .watch-board th {{ text-align: center !important; font-size: {max(14, user_font_size-4)}px !important; padding: 10px !important; border-bottom: 2px solid #555 !important; color: #888; }}
                .watch-board td {{ text-align: center !important; font-size: {user_font_size}px !important; padding: 16px !important; border-bottom: 1px solid #444 !important; vertical-align: middle !important; }}
                </style>
                """.replace('\n', '')
                st.markdown(f'{css}<div class="watch-board">{html_table}</div>', unsafe_allow_html=True)
            else: st.info("同步流介接中...")
        render_rt()

    with tab2:
        st.markdown("#### ⚡ 異步全市場並行篩選雷達")
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
            
            res = run_async_market_scan(tickers, conds, p_bar, s_text, st.session_state.stock_names, get_market_index_data(), "radar")
            s_text.empty(); p_bar.empty()
            if res: st.dataframe(pd.DataFrame(res), use_container_width=True)
            else: st.warning("無符合標的。")

    with tab3:
        st.markdown("#### 🎯 多因子演算法綜合評分排行榜 (TOP 20)")
        st.caption("融合 日/週雙時區共振、大盤相對強度矩陣 (RS) 及 Squeeze收斂臨界點 進行深度運算 (滿分100)。")
        
        if st.button("🔮 執行全權重深度矩陣運算", type="primary"):
            custom = [t for g in st.session_state.stock_clusters.values() for t in g]
            tickers = list(set(custom + (csv_df['Ticker'].tolist() if not csv_df.empty else [])))
            p_bar, s_text = st.progress(0), st.empty()
            
            res = run_async_market_scan(tickers, {}, p_bar, s_text, st.session_state.stock_names, get_market_index_data(), "score")
            s_text.empty(); p_bar.empty()
            if res: 
                df_res = pd.DataFrame(res).sort_values("量化總分", ascending=False).head(20)
                st.dataframe(df_res, column_config={"量化總分": st.column_config.ProgressColumn("多空能量值", min_value=0, max_value=100, format="%d 分")}, use_container_width=True, index=False)
            else: st.warning("運算失敗，無法取得數據。")