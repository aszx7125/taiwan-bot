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

# --- 側邊欄 ---
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
        new_cluster_name = st.text_input("群組名稱")
        new_cluster_tickers = st.text_area("股票代號 (逗號分隔)")
        if st.button("➕ 儲存配置"):
            if new_cluster_name and new_cluster_tickers:
                st.session_state.stock_clusters[new_cluster_name] = [t if '.TW' in t or '.TWO' in t else f"{t}.TW" for t in new_cluster_tickers.split(',')]
                st.success("配置成功！"); st.rerun()

# --- 主路由 ---
st.title("⚡ 台股戰情分析終端")
col1, col2 = st.columns([3, 1])
with col1: manual_ticker = st.text_input("輸入股票代號", "", label_visibility="collapsed")
with col2: analyze_manual_btn = st.button("單股掃描", use_container_width=True)
st.markdown("---")

target_ticker = st.session_state.pop('analyze_trigger', None) or (manual_ticker.strip().upper() if analyze_manual_btn else None)

if target_ticker:
    base_ticker = target_ticker.split('.')[0]
    c_name = st.session_state.stock_names.get(base_ticker, target_ticker)
    
    with st.spinner(f"矩陣分析 {target_ticker}..."):
        df, actual_symbol = get_kline_with_fugle(target_ticker, FUGLE_API_KEY)
        if df.empty or len(df) < 35: st.error("❌ 數據不足")
        else:
            today, yesterday = df.iloc[-1], df.iloc[-2]
            st.subheader(f"🧬 {target_ticker} {c_name} 量化診斷")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("現價", f"{today['Close']:.2f}", f"{((today['Close']-yesterday['Close'])/yesterday['Close'])*100:+.2f}%")
            m2.metric("量比", f"{today['Volume']/today['Vol_SMA5']:.1f}x")
            m3.metric("AI評分", f"{int(today['Score'])} 分")
            m4.metric("大盤相對強度", f"{today['RS_Index']*100:+.2f}%")
            
            st.markdown("---")
            t1, t2, t3, t4 = st.tabs(["🧱 測幅與策略", "🔍 前向驗證", "🕵️‍♂️ 籌碼動向", "📰 新聞動態"])
            with t1:
                st.write(f"**壓力:** {today['Res_20']:.2f} | **支撐:** {today['Sup_20']:.2f} | **型態:** {'💥 擠壓收斂中' if today['Squeeze_On'] else '常態'}")
                st.write(f"**週線共振:** {'📈 多頭保護' if today['Weekly_Trend_Up'] else '📉 空頭壓制'}")
            with t2:
                y_res, y_sup, y_atr = today['Res_20'], today['Sup_20'], yesterday['ATR_14']
                st.write(f"昨日預測壓力: **{y_res:.2f}** | 今日最高價: **{today['High']:.2f}**")
                if today['Close'] > y_res: st.success("突破確立！")
                else: st.info("區間震盪")
            with t3:
                st.progress(int(today['Score']), text=f"主力綜合控盤度：{int(today['Score'])}%")
            with t4:
                nl, nr = st.columns(2)
                with nl:
                    for n in get_stock_news(c_name)[:3]: st.markdown(f"[{n['title']}]({n['link']})")
                with nr:
                    for n in get_macro_news()[:3]: st.markdown(f"[{n['title']}]({n['link']})")
                    
        if st.button("⬅️ 返回主頁"): st.rerun()

else:
    st.markdown("### 🌍 大盤摘要")
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
                    # 🛡️ 雙重保險：強制剝離後綴餵給 API
                    df, _ = get_kline_with_fugle(t.split('.')[0], FUGLE_API_KEY)
                    if len(df) >= 3:
                        c, p = df.iloc[-1], df.iloc[-2]
                        change_amt = c['Close'] - p['Close']
                        change_pct = (change_amt / p['Close']) * 100
                        gap = " <span style='color:#ff4b4b;font-size:0.7em;'>(跳空🔥)</span>" if c['Low'] > p['High'] else ""
                        
                        price_vol = f"<b>{c['Close']:.2f}</b><br><span style='font-size:0.7em;color:gray;'>({int(c['Volume']):,} 張)</span>"
                        name_str = f"<b>{st.session_state.stock_names.get(t.split('.')[0], t)}</b><br><span style='font-size:0.8em;color:gray;'>{t.split('.')[0]}</span>"
                        change_str = f"<span style='color:#ff4b4b;font-weight:bold;'>+{change_amt:.2f}<br>(+{change_pct:.2f}%){gap}</span>" if change_amt > 0 else (f"<span style='color:#00cc96;font-weight:bold;'>{change_amt:.2f}<br>({change_pct:.2f}%){gap}</span>" if change_amt < 0 else "0.00")
                        
                        rows.append({"標的": name_str, "及時價 (成交量)": price_vol, "今日漲跌幅": change_str})
                except: pass
                
            if rows:
                html_table = pd.DataFrame(rows).to_html(escape=False, index=False, border=0).replace('\n', '')
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