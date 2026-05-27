import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import datetime
import time
import urllib.parse
import xml.etree.ElementTree as ET

# --- 網頁全局設定 ---
st.set_page_config(page_title="台股分析終端", page_icon="📈", layout="wide")

# ==========================================
# ⚙️ 系統核心設定區
# ==========================================
try:
    FUGLE_API_KEY = st.secrets["FUGLE_API_KEY"]
except:
    FUGLE_API_KEY = "" # ⚠️ 本機測試若要用富果，請貼在此處；上傳 GitHub 前請清空

yf_session = requests.Session()
yf_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})

stock_clusters = {
    "低軌衛星": ["3491.TWO", "3138.TW", "6285.TW", "2383.TW", "2314.TW"],
    "矽光子": ["3363.TWO", "3450.TW", "6451.TW", "3081.TWO", "4979.TWO", "3163.TWO"],
    "面板": ["2409.TW", "3481.TW", "6116.TW"],
    "半導體": ["2330.TW", "3711.TW", "2454.TW", "2303.TW", "5347.TWO", "3034.TW"],
    "伺服器": ["2382.TW", "3231.TW", "6669.TW", "2376.TW", "3017.TW", "5274.TWO"],
    "金融股": ["2881.TW", "2882.TW", "2886.TW", "2891.TW", "2884.TW"],
    "傳統產業": ["1101.TW", "2002.TW", "2603.TW", "2609.TW", "2618.TW"],
    "ETF": ["0050.TW", "0056.TW", "00878.TW", "00919.TW", "00929.TW"]
}

stock_names = {
    "3491": "昇達科", "3138": "耀登", "6285": "啟碁", "2383": "華通", "2314": "台揚",
    "3363": "上詮", "3450": "聯鈞", "6451": "訊芯-KY", "3081": "聯亞", "4979": "華星光", "3163": "波若威",
    "2409": "友達", "3481": "群創", "6116": "彩晶",
    "2330": "台積電", "3711": "日月光投控", "2454": "聯發科", "2303": "聯電", "5347": "世界先進", "3034": "聯詠",
    "2382": "廣達", "3231": "緯創", "6669": "緯穎", "2376": "技嘉", "3017": "奇鋐", "5274": "信驊",
    "2881": "富邦金", "2882": "國泰金", "2886": "兆豐金", "2891": "中信金", "2884": "玉山金",
    "1101": "台泥", "2002": "中鋼", "2603": "長榮", "2609": "陽明", "2618": "長榮航",
    "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息", "00919": "群益精選高息", "00929": "復華科技優息"
}

TICKER_MAP = {t.split('.')[0]: t for tickers in stock_clusters.values() for t in tickers}

# --- 核心數據引擎 ---
@st.cache_data(ttl=60) 
def get_kline_with_fugle(ticker_code):
    symbols_to_try = []
    if ticker_code in TICKER_MAP:
        symbols_to_try.append(TICKER_MAP[ticker_code])
    else:
        symbols_to_try.extend([f"{ticker_code}.TW", f"{ticker_code}.TWO"])

    df = pd.DataFrame()
    actual_symbol = ""
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for symbol in symbols_to_try:
            for attempt in range(3):
                try:
                    temp_df = yf.Ticker(symbol, session=yf_session).history(period="6mo")
                    if not temp_df.empty:
                        df = temp_df
                        actual_symbol = symbol
                        break 
                except Exception as e:
                    if "Too Many Requests" in str(e) or "429" in str(e):
                        time.sleep(2 * (attempt + 1)) 
                    else:
                        break 
            if not df.empty:
                break 

    if df.empty or len(df) < 20:
        return df, actual_symbol 

    if FUGLE_API_KEY and FUGLE_API_KEY != "":
        try:
            time.sleep(1.2) 
            url = f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{ticker_code}"
            headers = {"X-API-KEY": FUGLE_API_KEY}
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                rt_price = data.get('closePrice') or data.get('lastTrade', {}).get('price')
                rt_vol = data.get('total', {}).get('tradeVolume', 0)
                
                tz_tw = datetime.timezone(datetime.timedelta(hours=8))
                today_date = datetime.datetime.now(tz_tw).date()
                last_candle_date = df.index[-1].date()
                
                if last_candle_date < today_date and rt_price:
                    new_row = df.iloc[-1].copy()
                    new_row.name = pd.Timestamp(today_date, tz=df.index.tz)
                    df = pd.concat([df, pd.DataFrame([new_row])])
                
                if rt_price:
                    df.iloc[-1, df.columns.get_loc('Close')] = rt_price
                    if data.get('highPrice'): df.iloc[-1, df.columns.get_loc('High')] = data['highPrice']
                    if data.get('lowPrice'): df.iloc[-1, df.columns.get_loc('Low')] = data['lowPrice']
                if rt_vol > 0:
                    df.iloc[-1, df.columns.get_loc('Volume')] = rt_vol
        except:
            pass
            
    return df, actual_symbol

# 🌟 新增：專屬個股新聞獲取引擎 (Google RSS 穩定版)
@st.cache_data(ttl=300)
def get_stock_news(keyword):
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(keyword)}+when:7d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        root = ET.fromstring(response.text)
        news_list = []
        for item in root.findall('./channel/item')[:5]:
            news_list.append({
                "title": item.find('title').text,
                "link": item.find('link').text,
                "date": item.find('pubDate').text
            })
        return news_list
    except:
        return []

# --- 總經新聞獲取引擎 ---
@st.cache_data(ttl=300)
def get_macro_news():
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote('台股 OR 聯準會 OR 財報')}+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        root = ET.fromstring(response.text)
        news_list = []
        for item in root.findall('./channel/item')[:6]:
            news_list.append({
                "title": item.find('title').text,
                "link": item.find('link').text,
                "date": item.find('pubDate').text
            })
        return news_list
    except:
        return []


# ==========================================
# 📱 側邊欄 (優化版：支援自動收合)
# ==========================================
with st.sidebar:
    st.header("📂 我的自選清單")
    
    if 'sidebar_state' not in st.session_state:
        st.session_state.sidebar_state = 'expanded'

    selected_cluster = st.selectbox("1. 選擇產業群組", list(stock_clusters.keys()))
    
    cluster_stocks = stock_clusters[selected_cluster]
    display_options = [f"{t.split('.')[0]} {stock_names.get(t.split('.')[0], '')}".strip() for t in cluster_stocks]
        
    selected_stock_display = st.selectbox("2. 選擇分析標的", display_options)
    sidebar_ticker = selected_stock_display.split(' ')[0]
    
    st.write("") 
    
    if st.button("📊 診斷此自選股", use_container_width=True, type="primary"):
        st.session_state.analyze_trigger = sidebar_ticker 
        st.rerun()

    st.markdown("---")
    if FUGLE_API_KEY:
        st.success("🟢 零延遲即時引擎已啟動")
    else:
        st.warning("🟡 目前使用 Yahoo 延遲報價")


# ==========================================
# 🖥️ 主畫面路由控制
# ==========================================
st.title("⚡ 台股分析終端")

# 搜尋區塊
st.markdown("##### 🔍 搜尋個股詳細診斷報告")
col1, col2 = st.columns([3, 1])
with col1:
    manual_ticker = st.text_input("輸入股票代號 (如: 3105, 2317)", "", label_visibility="collapsed")
with col2:
    analyze_manual_btn = st.button("執行搜尋", use_container_width=True)

st.markdown("---")

target_ticker = None

if 'analyze_trigger' in st.session_state and st.session_state.analyze_trigger:
    target_ticker = st.session_state.analyze_trigger
    st.session_state.analyze_trigger = None 

elif analyze_manual_btn and manual_ticker:
    target_ticker = manual_ticker.strip().upper()


# ==========================================
# ⚡ 模式分流：深度診斷模式 vs 主頁基本資訊看板
# ==========================================
if target_ticker:
    # ─── 進入詳細分析視圖 ───
    with st.spinner(f"正在擷取 {target_ticker} 的量價數據與基本面資料..."):
        try:
            tz_tw = datetime.timezone(datetime.timedelta(hours=8))
            now = datetime.datetime.now(tz_tw)
            is_market_open = now.weekday() < 5 and datetime.time(9, 0) <= now.time() <= datetime.time(13, 30)
            time_label = "今日盤中" if is_market_open else "明日"

            df, actual_symbol = get_kline_with_fugle(target_ticker)
            
            if df.empty or len(df) < 40:
                st.error(f"❌ 找不到代號 {target_ticker} 的有效資料。請確認該標的已經上市櫃。")
            else:
                df['SMA_5'] = df['Close'].rolling(window=5).mean()
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['SMA_60'] = df['Close'].rolling(window=60).mean()
                df['Vol_SMA5'] = df['Volume'].rolling(window=5).mean()
                df['TR'] = df[['High', 'Low', 'Close']].apply(lambda x: max(x['High'] - x['Low'], abs(x['High'] - df['Close'].shift(1).loc[x.name]), abs(x['Low'] - df['Close'].shift(1).loc[x.name])), axis=1)
                df['ATR_14'] = df['TR'].rolling(window=14).mean()

                df['Res_20'] = df['High'].shift(1).rolling(window=20).max()
                df['Sup_20'] = df['Low'].shift(1).rolling(window=20).min()

                today = df.iloc[-1]
                yesterday = df.iloc[-2]
                close_today = today['Close']
                open_today = today['Open']
                high_today = today['High']
                low_today = today['Low']
                yesterday_close = yesterday['Close']
                vol_today = today['Volume']
                vol_sma5 = today['Vol_SMA5']
                atr14 = today['ATR_14']
                
                vol_ratio = (vol_today / vol_sma5) if vol_sma5 > 0 else 1.0
                p_change = ((close_today - yesterday_close) / yesterday_close) * 100
                
                res_level = today['Res_20']
                sup_level = today['Sup_20']
                box_height = res_level - sup_level
                
                recent_20_df = df.iloc[-21:-1]
                res_tests = len(recent_20_df[recent_20_df['High'] >= res_level * 0.985])
                sup_tests = len(recent_20_df[recent_20_df['Low'] <= sup_level * 1.015])
                
                breakout_status = "區間震盪 (未突破)"
                target_proj = "無明確突破方向，等待表態"
                breakout_prob = "中立"
                
                if close_today > res_level:
                    breakout_status = "🚀 向上突破前高"
                    target_proj = f"突破確認！目標上看 **{round(res_level + box_height, 1)}**"
                    breakout_prob = "強勢發動"
                elif close_today < sup_level:
                    breakout_status = "⚠️ 向下摜破前低"
                    target_proj = f"破底危機！下看 **{round(sup_level - box_height, 1)}**"
                    breakout_prob = "弱勢探底"
                elif close_today >= res_level * 0.98:
                    breakout_status = "⚔️ 兵臨城下 (挑戰前高)"
                    if vol_ratio > 1.3 and close_today > open_today:
                        breakout_prob = "高機率突破 (帶量收紅，具備攻擊契機)"
                        target_proj = f"若成功突破 {round(res_level, 1)}，目標上看 **{round(res_level + box_height, 1)}**"
                    else:
                        breakout_prob = "機率中等 (量能不足，仍需補量)"
                        target_proj = f"壓力位 {round(res_level, 1)} 附近震盪"
                elif close_today <= sup_level * 1.02: 
                    breakout_status = "🛡️ 支撐保衛戰 (回測前低)"
                    if vol_ratio > 1.3 and close_today < open_today:
                        breakout_prob = "高機率破底 (放量下殺，賣壓沉重)"
                        target_proj = f"若失守 {round(sup_level, 1)}，下測 **{round(sup_level - box_height, 1)}**"
                    else:
                        breakout_prob = "機率中等 (量縮測試，觀察買盤承接)"
                        target_proj = f"支撐位 {round(sup_level, 1)} 防守戰"

                c_name = stock_names.get(target_ticker, "")
                try:
                    yf_ticker_obj = yf.Ticker(actual_symbol, session=yf_session)
                    info_tw = yf_ticker_obj.info
                    if not c_name: c_name = info_tw.get('shortName', '')
                    pe_ratio = info_tw.get('trailingPE', 'N/A')
                except:
                    pe_ratio = 'N/A'
                
                trend_status = "震盪整理"
                if pd.notna(today['SMA_60']):
                    if close_today > today['SMA_20'] and close_today > today['SMA_60']: trend_status = "多頭排列 📈"
                    elif close_today < today['SMA_20'] and close_today < today['SMA_60']: trend_status = "空頭弱勢 📉"

                limit_up = round(yesterday_close * 1.10, 1)
                limit_down = round(yesterday_close * 0.90, 1)

                st.subheader(f"🧬 {target_ticker} {c_name} 診斷報告")
                
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("當前現價", f"{close_today:.1f}", f"{p_change:+.2f}%")
                m_col2.metric("今日成交量 (量比)", f"{int(vol_today):,}", f"{vol_ratio:.1f}x", delta_color="off")
                m_col3.metric("技術趨勢", trend_status)
                pe_display = f"{round(pe_ratio, 1)} 倍" if isinstance(pe_ratio, (int, float)) else pe_ratio
                m_col4.metric("型態狀態", breakout_status.split(' ')[0])

                st.markdown("---")

                tab1, tab2, tab3 = st.tabs(["📊 量化診斷與策略推演", "📰 財報新聞與總經動態", "🔍 昨日預測 vs 今日實況"])
                
                with tab1:
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        st.markdown("### 🧱 型態與支撐壓力分析")
                        st.write(f"- **前高壓力 (近20日):** {round(res_level, 1)} | **已測試:** {res_tests} 次")
                        st.write(f"- **前低支撐 (近20日):** {round(sup_level, 1)} | **已測試:** {sup_tests} 次")
                        st.write(f"- **目前盤勢型態:** {breakout_status}")
                        st.write(f"- **突破機率評估:** {breakout_prob}")
                        st.write(f"- **等距測幅 (目標):** {target_proj}")
                        
                        st.markdown("### 📊 均線與波動度參數")
                        st.write(f"- **5日均線 (短):** {today['SMA_5']:.1f}")
                        st.write(f"- **20日均線 (月):** {today['SMA_20']:.1f}")
                        st.write(f"- **每日平均波動 (ATR):** {atr14:.1f} 元")

                    with d_col2:
                        st.markdown(f"### 🎯 {time_label}操作推演與防守極限")
                        st.markdown(f"**🔴 漲停板極限:** {limit_up} *(昨日收盤 +10%)*")
                        st.markdown(f"**🟢 跌停板極限:** {limit_down} *(昨日收盤 -10%)*")
                        
                        st.markdown("##### 💡 策略規劃")
                        if "突破前高" in breakout_status:
                            st.write("**🟢 順勢做多:** 型態已正式突破，上方無壓，可順勢切入。防守點設於前高壓力轉支撐處。")
                        elif "挑戰前高" in breakout_status:
                            st.write("**🟡 提前卡位 / 觀望:** 即將挑戰關鍵頸線，若量比持續大於 1.3 可小部位試單，否則建議等待確認突破後再追。")
                        elif "回測前低" in breakout_status:
                            st.write("**🟡 低接防守:** 正在測試底部支撐，測試次數越多支撐越強。可嘗試低接，但跌破前低必須果斷停損。")
                        elif "摜破前低" in breakout_status:
                            st.write("**🔴 嚴格停損:** 型態破底，空頭成形。多單請嚴格執行停損紀律，不建議進場攤平。")
                        else:
                            st.write("**🟡 區間操作:** 目前處於箱體中央，肉不多且容易被洗。建議耐心等待股價靠近上軌或下軌時再動作。")
                            
                    with st.expander("查看近期 K 線歷史原始數據"):
                        st.dataframe(df.tail(25).sort_index(ascending=False))

                with tab2:
                    n_col1, n_col2 = st.columns(2)
                    with n_col1:
                        st.markdown(f"### 🎯 {c_name} 最新專屬新聞與公告")
                        # 🌟 改用我們全新寫好的 Google RSS 引擎，徹底修復 1970 年的錯誤
                        stock_news = get_stock_news(c_name if c_name else target_ticker)
                        if stock_news:
                            for news_item in stock_news:
                                st.markdown(f"**[{news_item['title']}]({news_item['link']})**")
                                clean_date = news_item['date'].replace(" GMT", "")
                                st.caption(f"🕒 {clean_date}")
                                st.markdown("---")
                        else:
                            st.info("目前無最新個股專屬新聞。")

                    with n_col2:
                        st.markdown("### 🌍 全球大盤與總經焦點")
                        macro_news = get_macro_news()
                        if macro_news:
                            for news_item in macro_news:
                                st.markdown(f"**[{news_item['title']}]({news_item['link']})**")
                                clean_date = news_item['date'].replace(" GMT", "")
                                st.caption(f"🕒 {clean_date}")
                                st.markdown("---")
                        else:
                            st.info("目前無最新總經新聞。")

                with tab3:
                    st.markdown("### 🔍 昨日策略劇本與今日走勢驗證")
                    st.write("系統自動讀取**昨日盤後**設定的關鍵點位，對比**今日**的實際走勢，即時驗證策略是否如期發動。")

                    y_res = today['Res_20']  
                    y_sup = today['Sup_20']
                    y_atr = yesterday['ATR_14']
                    y_target = y_res + y_atr

                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.info(f"**昨日盤後預測基準**\n- 壓力位: **{y_res:.1f}**\n- 支撐位: **{y_sup:.1f}**\n- 測幅目標: **{y_target:.1f}**")
                    with col_r2:
                        st.warning(f"**今日實況數據**\n- 最高價: **{high_today:.1f}**\n- 最低價: **{low_today:.1f}**\n- 收盤/現價: **{close_today:.1f}**")

                    st.markdown("#### 💡 走勢驗證判定")
                    if close_today > y_res:
                        if high_today >= y_target:
                            st.success(f"⭐⭐⭐ **超前達標**：今日強勢突破壓力位 {y_res:.1f}，且最高來到 {high_today:.1f}，成功觸及昨日測幅目標 {y_target:.1f}！")
                        else:
                            st.success(f"⭐⭐ **突破確認**：今日收盤 {close_today:.1f} 成功站上壓力位 {y_res:.1f}，多頭正式發動，符合突破進場策略。")
                    elif close_today < y_sup:
                        st.error(f"⚠️ **跌破防線**：今日收盤 {close_today:.1f} 跌破支撐位 {y_sup:.1f}，趨勢明顯轉弱，觸發停損機制。")
                    elif high_today >= y_res and close_today <= y_res:
                        st.warning(f"👀 **假突破 / 壓力沉重**：今日盤中最高來到 {high_today:.1f}，一度穿越壓力位，但收盤未能站穩，顯示上方賣壓沉重。")
                    elif low_today <= y_sup and close_today >= y_sup:
                        st.info(f"🛡️ **支撐有守 (破底翻)**：今日最低下探 {low_today:.1f}，但在支撐位 {y_sup:.1f} 附近獲得買盤承接，收盤成功拉回箱體。")
                    else:
                        st.write(f"⏸️ **區間震盪**：今日走勢完全在預設的箱體 ({y_sup:.1f} ~ {y_res:.1f}) 內震盪，尚未表態，符合量縮觀望的預期。")
                            
            if st.button("⬅️ 返回分組監控主頁", use_container_width=True):
                st.session_state.analyze_trigger = None
                st.rerun()

        except Exception as e:
            st.error(f"分析運算時發生錯誤: {e}")

else:
    # ─── 🌟 無感刷新黑科技：極致量價實時看板 (單行防護版) ───
    st.markdown(f"### 📊 【{selected_cluster}】實時監控看板")
    st.caption("點選左側或利用上方搜尋框，進入個股深度診斷。")
    
    with st.expander("⚙️ 看板顯示設定", expanded=False):
        user_font_size = st.slider(
            "🔍 調整看板文字大小 (px)", 
            min_value=12, 
            max_value=40, 
            value=22, 
            step=2,
            help="向右拖曳可將看板數字放大，適合用手機看盤時使用。"
        )
    
    @st.fragment(run_every=datetime.timedelta(seconds=15))
    def render_realtime_dashboard():
        dashboard_rows = []
        for stock_ticker in cluster_stocks:
            ticker_code = stock_ticker.split('.')[0]
            company_name = stock_names.get(ticker_code, "未知個股")
            try:
                kline_df, _ = get_kline_with_fugle(ticker_code)
                if not kline_df.empty and len(kline_df) >= 3:
                    current_day = kline_df.iloc[-1]
                    prev_day = kline_df.iloc[-2]
                    prev2_day = kline_df.iloc[-3]
                    
                    price_now = current_day['Close']
                    price_prev = prev_day['Close']
                    volume_now = int(current_day['Volume'])
                    
                    change_amt = price_now - price_prev
                    change_pct = (change_amt / price_prev) * 100
                    
                    price_vol_str = f"<b>{price_now:.2f}</b><br><span style='font-size: 0.7em; color: gray;'>({volume_now:,} 張)</span>"
                    name_str = f"<b>{company_name}</b><br><span style='font-size: 0.8em; color: gray;'>{ticker_code}</span>"
                    
                    gap_emoji = ""
                    if current_day['Low'] > prev_day['High']:
                        gap_emoji = " <span style='font-size: 0.8em;'>🔥(跳空)</span>"
                    elif current_day['High'] < prev_day['Low']:
                        gap_emoji = " <span style='font-size: 0.8em;'>❄️(跳空)</span>"
                    elif (prev_day['Low'] > prev2_day['High'] and current_day['Low'] <= prev2_day['High']) or \
                         (prev_day['High'] < prev2_day['Low'] and current_day['High'] >= prev2_day['Low']):
                        gap_emoji = " <span style='font-size: 0.8em;'>✅(缺口補)</span>"

                    if change_amt > 0:
                        change_str = f"<span style='color: #ff4b4b; font-weight: bold;'>+{change_amt:.2f}<br>(+{change_pct:.2f}%){gap_emoji}</span>"
                    elif change_amt < 0:
                        change_str = f"<span style='color: #00cc96; font-weight: bold;'>{change_amt:.2f}<br>({change_pct:.2f}%){gap_emoji}</span>"
                    else:
                        change_str = f"<span style='color: #a0a0a0; font-weight: bold;'>0.00<br>(0.00%){gap_emoji}</span>"
                    
                    dashboard_rows.append({
                        "標的": name_str,
                        "及時價 (成交量)": price_vol_str,
                        "今日漲跌幅": change_str
                    })
            except: pass
                
        if dashboard_rows:
            monitor_df = pd.DataFrame(dashboard_rows)
            html_table = monitor_df.to_html(escape=False, index=False, border=0).replace('\n', '')
            css = f"<style>.watch-board table {{ width: 100%; border-collapse: collapse; }} .watch-board th {{ text-align: center !important; font-size: {max(14, user_font_size - 4)}px !important; padding: 10px !important; border-bottom: 2px solid #555 !important; color: #888; }} .watch-board td {{ text-align: center !important; font-size: {user_font_size}px !important; padding: 16px !important; border-bottom: 1px solid #444 !important; vertical-align: middle !important; }}</style>".replace('\n', '')
            final_html = f'{css}<div class="watch-board">{html_table}</div>'
            st.markdown(final_html, unsafe_allow_html=True)
            
            tz_tw = datetime.timezone(datetime.timedelta(hours=8))
            st.write(f"⏱️ *即時報價最後同步：{datetime.datetime.now(tz_tw).strftime('%H:%M:%S')}*")
        else:
            st.info("📡 讀取中...")

    render_realtime_dashboard()
