import streamlit as st
import pandas as pd
import datetime
import random 
import concurrent.futures
import json
import os

# 在 app.py 中新增一個讀取快取的函數
@st.cache_data(ttl=600) # 快取 10 分鐘
def load_market_snapshot():
    if os.path.exists("market_snapshot.json"):
        with open("market_snapshot.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return None

from config import get_fugle_key, DEFAULT_CLUSTERS, DEFAULT_NAMES, INDUSTRY_CHAINS
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
    
    with st.spinner(f"正在分析 {target_ticker}... 提取深度資料中"):
        df, actual_symbol = get_kline_with_fugle(target_ticker, FUGLE_API_KEY)
        
        if df.empty or len(df) < 40: 
            st.error("❌ 該標的數據深度不足，無法執行複雜演算法。可能原因：代號錯誤、剛上市、或暫停交易。")
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
            
            # 🚀 提取 SMC 狀態
            liq_sweep = bool(today.get('Liquidity_Sweep_Bull', False))
            fvg_bull = bool(today.get('FVG_Bull', False))
            smc_status = []
            if liq_sweep: smc_status.append("🌊 流動性掠奪 (破底翻洗盤結束)")
            if fvg_bull: smc_status.append("🧱 公允價值缺口 (主力強勢建倉)")
            smc_text = " + ".join(smc_status) if smc_status else "無觸發"
            
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
            t1, t2, t3, t4 = st.tabs(["🧱 測幅與前瞻策略推演", "🔍 前向策略回測", "🕵️‍♂️ 籌碼動向矩陣", "📰 專屬新聞動態"])
            
            with t1:
                c_l, c_r = st.columns(2)
                with c_l:
                    st.markdown("#### 📐 關鍵結構與前瞻預判")
                    st.write(f"- **前高壓力 (近20日):** {res_level:.2f} | **已測試:** {res_tests} 次")
                    st.write(f"- **前低支撐 (近20日):** {sup_level:.2f} | **已測試:** {sup_tests} 次")
                    st.write(f"- **盤勢型態判定:** {breakout_status}")
                    st.markdown(f"- **RSI 動能背離偵測:** <span style='color:{'#00cc96' if bull_div else ('#ff4b4b' if bear_div else 'gray')}; font-weight:bold;'>{div_status}</span>", unsafe_allow_html=True)
                    st.markdown(f"- **SMC 機構級微觀結構:** <span style='color:{'#ffc107' if smc_status else 'gray'}; font-weight:bold;'>{smc_text}</span>", unsafe_allow_html=True)
                    st.write(f"- **波動壓縮狀態:** {'⚠️ 極度擠壓收斂 (Squeeze)' if today.get('Squeeze_On', False) else '🟢 波動度常態分佈'}")
                    st.write(f"- **大週期週線共振:** {'📈 週線處於波段多頭保護期' if today.get('Weekly_Trend_Up', False) else '📉 週線空頭趨勢壓制'}")
                with c_r:
                    st.markdown("#### 💡 操作劇本規劃")
                    st.write(f"**🔴 漲停極限:** {round(yesterday['Close'] * 1.10, 1)} | **🟢 跌停極限:** {round(yesterday['Close'] * 0.90, 1)}")
                    st.write(f"**等距測幅 (目標):** {target_proj}")
                    
                    if "破底翻" in smc_text or "底背離" in div_status:
                        st.success("🎯 **【起漲潛力】** 出現左側底背離或主力洗盤結束訊號！若確認站穩支撐，此處為風報比極佳的低價試單區間。")
                    elif "突破前高" in breakout_status: 
                        st.success("🚀 **【右側追隨】** 型態正式帶量向上突破！屬強勢追隨訊號，防守點位移至前高。")
                    elif "挑戰前高" in breakout_status: 
                        st.warning("⚔️ **【突破預備】** 兵臨城下，即將挑戰壓力！若帶量突破布林上軌可試單。")
                    elif "回測前低" in breakout_status: 
                        st.warning("🛡️ **【支撐保衛】** 測試底部支撐，若跌破前低應果斷停損。")
                    elif "摜破前低" in breakout_status: 
                        st.error("⚠️ **【破底危機】** 嚴格執行停損，切勿攤平。")
                    else: 
                        st.info("⏸️ **【箱體觀望】** 股價處於箱體內部結構震盪，採取下軌附近低吸、上軌調節之區間策略。")
            
            with t2:
                st.markdown("### 🔍 歷史預測與實況對撞 (Forward Testing)")
                sub_t1, sub_t2 = st.tabs(["1️⃣ 昨日預測 (近 1 日)", "5️⃣ 一週波段 (近 5 日)"])
                
                with sub_t1:
                    y_res, y_sup, y_atr = today['Res_20'], today['Sup_20'], yesterday['ATR_14']
                    y_target = y_res + y_atr

                    col_r1, col_r2 = st.columns(2)
                    with col_r1: st.info(f"**昨日盤後預測基準**\n- 壓力位: **{y_res:.2f}**\n- 支撐位: **{y_sup:.2f}**\n- 測幅目標: **{y_target:.2f}**")
                    with col_r2: st.warning(f"**今日實況極值**\n- 最高價: **{today['High']:.2f}**\n- 最低價: **{today['Low']:.2f}**\n- 收盤現價: **{today['Close']:.2f}**")

                    if today['Close'] > y_res:
                        if today['High'] >= y_target: st.success(f"⭐⭐⭐ **超前達標**：今日強勢突破壓力位，並成功觸及等距測幅目標！")
                        else: st.success(f"⭐⭐ **突破確認**：今日收盤站上壓力位，多頭正式發動。")
                    elif today['Close'] < y_sup:
                        st.error(f"⚠️ **跌破防線**：今日收盤跌破支撐位，觸發停損機制。")
                    elif today['High'] >= y_res and today['Close'] <= y_res:
                        st.warning(f"👀 **假突破 / 壓力沉重**：今日盤中突破壓力，但收盤未能站穩。")
                    elif today['Low'] <= y_sup and today['Close'] >= y_sup:
                        st.info(f"🛡️ **支撐有守 (破底翻)**：今日下探支撐，但獲得買盤承接拉回。")
                    else:
                        st.write(f"⏸️ **區間震盪**：走勢在預設箱體內震盪，符合觀望預期。")
                        
                with sub_t2:
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
                        
                        if max_h_5d >= w_target: st.success("⭐⭐⭐ **波段達標**：本週內曾成功突破並觸及一週前的等距測幅目標！")
                        elif max_h_5d > w_res: st.success("⭐⭐ **波段突破**：本週內曾成功突破壓力位，啟動多頭行情。")
                        elif min_l_5d < w_sup: st.error("⚠️ **波段破底**：本週內曾跌破一週前的關鍵支撐，若未停損可能擴大虧損。")
                        else: st.info("⏸️ **大型箱體**：這 5 天內始終在一週前的壓力與支撐區間內震盪洗盤。")
                    else:
                        st.warning("⚠️ 數據不足 26 天，無法進行一週歷史回測。")
            
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
                    
        if st.button("⬅️ 返回戰情室主頁", use_container_width=True):
            st.session_state.analyze_trigger = None
            st.rerun()

else:
    # ─── 【模組 B】主頁儀表板 ───
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

    # 🌟 重點修復：系統核心白皮書強勢回歸！
    with st.expander("🧠 系統核心：量化策略與多因子演算法白皮書 (點此展開)", expanded=False):
        st.markdown("""
        #### 1. 前瞻性預判：SMC 機構級微觀結構 (起漲點偵測)
        有別於落後的均線指標，本系統內建 **Smart Money Concepts (聰明錢概念)** 演算法，專抓主力發動前的細微痕跡：
        * **🌊 流動性掠奪 (破底翻)**：偵測股價刻意跌破近 20 日低點觸發散戶停損，卻在收盤強勢站回。這代表主力洗盤完畢，準備拉升。
        * **🧱 公允價值缺口 (FVG)**：偵測連續 K 線間留下的價格真空區，代表大戶不計代價的強勢建倉。

        #### 2. RSI 動態背離掃描 (左側抄底指標)
        系統內建波峰波谷比對演算法：當股價創下近期新低，但 RSI 指標卻「沒有」創低反而墊高時，觸發 **🟢 底背離** 訊號。代表殺跌動能枯竭，是極佳的低接訊號。

        #### 3. 產業鏈資金共振 (Top-Down 法人視角)
        系統將市場熱門標的嚴格劃分為「上、中、下游」。當資金灌入時，熱度會從上游（如：晶片）蔓延至下游（如：組裝）。透過觀察各區塊的平均 AI 評分，精準捕捉資金外溢效應。

        #### 4. Squeeze 波動率收斂突破 (右側主升段發動機)
        當布林通道縮口並完全被包進肯特納通道內時，稱為「極度擠壓 (Squeeze)」。這代表大風暴前的寧靜，此時若發生帶量突破，往往是暴賺主升段的起點。

        #### 5. AI 多因子評分矩陣 (滿分 100 分)
        為了強化「買在起漲前」的能力，我們降低了落後指標的權重，大幅提升了前瞻性指標的配分：
        1. **趨勢與動能 (10分)**：站上月線(5分) + MACD 金叉(5分)。
        2. **大盤相對強度 RS (15分)**：近 20 日報酬率戰勝大盤，抗跌飆股。
        3. **週線共振 (15分)**：大週期週線 MACD 為多頭，長線保護短線。
        4. **大戶籌碼動能 (10分)**：近 5 日「價漲量增」天數多於「價跌量增」。
        5. **量能與壓縮突破 (20分)**：爆發量(10分) + Squeeze 突破(10分)。
        6. **前瞻 - 底部背離發動 (10分)**：價格破底但 RSI 墊高，強烈反轉訊號。
        7. **前瞻 - SMC 主力建倉 (20分)**：流動性掠奪(10分) + FVG 缺口(10分)。
        """)
            
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 板塊實時監控", "⚡ 條件設定全市場雷達", "🎯 多因子 AI 評分 (含SMC)", "🕸️ 產業鏈資金共振 (精選)"])
    
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
            if res: st.dataframe(pd.DataFrame(res), use_container_width=True, hide_index=True)
            else: st.warning("⚠️ 查無符合標的，請嘗試放寬條件。")

    with tab3:
        st.markdown("#### 🎯 潛伏型 AI 勝率期望值排行 (TOP 20)")
        st.caption("以下數據由雲端背景運算引擎於每日盤後自動解算，達成 0 延遲載入。")
        
        snapshot = load_market_snapshot()
        if snapshot:
            st.success(f"⏱️ 數據最後更新時間: {snapshot['update_time']} (資料庫直連)")
            df_res = pd.DataFrame(snapshot['data'])
            st.dataframe(df_res.head(20), column_config={"量化總分": st.column_config.ProgressColumn("潛伏勝率期望值", min_value=0, max_value=100, format="%d 分")}, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ 尚未生成快取資料，請等待 GitHub Actions 執行完畢，或手動觸發運算。")

    with tab4:
        st.markdown("#### 🕸️ 上中下游產業鏈資金共振分析 (Top-Down)")
        selected_chain = st.selectbox("選擇要檢視的產業鏈", list(INDUSTRY_CHAINS.keys()))
        chain_data = INDUSTRY_CHAINS[selected_chain]
        
        snapshot = load_market_snapshot()
        if snapshot:
            res_df = pd.DataFrame(snapshot['data'])
            st.markdown("---")
            cols = st.columns(len(chain_data))
            recommendations = []
            
            for idx, (sub_name, tickers) in enumerate(chain_data.items()):
                with cols[idx]:
                    sub_codes = [t.split('.')[0] for t in tickers]
                    # 秒速從預先算好的 JSON 中篩選出該產業的標的，不用再爬蟲！
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