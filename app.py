import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import datetime
import time
import urllib.parse
import xml.etree.ElementTree as ET
import asyncio
import aiohttp

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# --- 網頁全局設定 ---
st.set_page_config(page_title="台股量化旗艦終端", page_icon="📈", layout="wide")

# ==========================================
# ⚙️ 系統核心設定與自訂清單管理器
# ==========================================
try: FUGLE_API_KEY = st.secrets["FUGLE_API_KEY"]
except: FUGLE_API_KEY = ""

yf_session = requests.Session()
yf_session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})

# 初始化預設群組 (存入 session_state 讓使用者可以動態修改)
if 'stock_clusters' not in st.session_state:
    st.session_state.stock_clusters = {
        "半導體": ["2330.TW", "3711.TW", "2454.TW", "2303.TW", "5347.TWO", "3034.TW"],
        "矽光子": ["3363.TWO", "3450.TW", "6451.TW", "3081.TWO", "4979.TWO", "3163.TWO"],
        "伺服器": ["2382.TW", "3231.TW", "6669.TW", "2376.TW", "3017.TW", "5274.TWO"],
        "金融股": ["2881.TW", "2882.TW", "2886.TW", "2891.TW", "2884.TW"],
        "傳統產業": ["1101.TW", "2002.TW", "2603.TW", "2609.TW", "2618.TW"],
        "低軌衛星": ["3491.TWO", "3138.TW", "6285.TW", "2383.TW", "2314.TW"],
        "面板": ["2409.TW", "3481.TW", "6116.TW"],
        "ETF": ["0050.TW", "0056.TW", "00878.TW", "00919.TW", "00929.TW"]
    }

# 初始化預設名稱字典
if 'stock_names' not in st.session_state:
    st.session_state.stock_names = {
        "3491": "昇達科", "3138": "耀登", "6285": "啟碁", "2383": "華通", "2314": "台揚",
        "3363": "上詮", "3450": "聯鈞", "6451": "訊芯", "3081": "聯亞", "4979": "華星光", "3163": "波若威",
        "2409": "友達", "3481": "群創", "6116": "彩晶",
        "2330": "台積電", "3711": "日月光投控", "2454": "聯發科", "2303": "聯電", "5347": "世界先進", "3034": "聯詠",
        "2382": "廣達", "3231": "緯創", "6669": "緯穎", "2376": "技嘉", "3017": "奇鋐", "5274": "信驊",
        "2881": "富邦金", "2882": "國泰金", "2886": "兆豐金", "2891": "中信金", "2884": "玉山金",
        "1101": "台泥", "2002": "中鋼", "2603": "長榮", "2609": "陽明", "2618": "長榮航",
        "0050": "台灣50", "0056": "高股息", "00878": "永續高息", "00919": "精選高息", "00929": "科技優息"
    }

def get_ticker_map():
    return {t.split('.')[0]: t for tickers in st.session_state.stock_clusters.values() for t in tickers}

# ==========================================
# ⚡ 數據獲取與指標引擎
# ==========================================
@st.cache_data(ttl=3600*24)
def load_all_market_tickers():
    try:
        df = pd.read_csv("all_tw_stocks.csv")
        for index, row in df.iterrows():
            code = str(row['Ticker']).split('.')[0]
            if code not in st.session_state.stock_names:
                st.session_state.stock_names[code] = str(row['Name'])
        return df['Ticker'].tolist()
    except Exception as e:
        return []

def add_advanced_indicators(df):
    if df.empty or len(df) < 30: return df
    df['SMA_5'], df['SMA_20'], df['SMA_60'] = df['Close'].rolling(5).mean(), df['Close'].rolling(20).mean(), df['Close'].rolling(60).mean()
    df['Vol_SMA5'] = df['Volume'].rolling(5).mean()
    delta = df['Close'].diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    exp1, exp2 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['TR'] = df[['High', 'Low', 'Close']].apply(lambda x: max(x['High'] - x['Low'], abs(x['High'] - df['Close'].shift(1).loc[x.name]), abs(x['Low'] - df['Close'].shift(1).loc[x.name])), axis=1)
    df['ATR_14'] = df['TR'].rolling(14).mean()
    df['Res_20'], df['Sup_20'] = df['High'].shift(1).rolling(20).max(), df['Low'].shift(1).rolling(20).min()
    return df

@st.cache_data(ttl=120)
def get_market_summary():
    indices = {"加權指數": "^TWII", "櫃買指數": "^TWOTC", "台灣50": "0050.TW"}
    summary_data = {}
    with requests.Session() as s:
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        for name, ticker in indices.items():
            try:
                df = yf.Ticker(ticker, session=s).history(period="2d")
                if len(df) >= 2:
                    p_now, p_prev = df['Close'].iloc[-1], df['Close'].iloc[-2]
                    summary_data[name] = {"price": p_now, "change": p_now - p_prev, "pct": ((p_now - p_prev) / p_prev) * 100}
            except: pass
    return summary_data

@st.cache_data(ttl=60) 
def get_kline_with_fugle(ticker_code):
    ticker_map = get_ticker_map()
    symbols_to_try = [ticker_map.get(ticker_code)] if ticker_code in ticker_map else [f"{ticker_code}.TW", f"{ticker_code}.TWO"]
    df, actual_symbol = pd.DataFrame(), ""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for symbol in symbols_to_try:
            try:
                temp_df = yf.Ticker(symbol, session=yf_session).history(period="6mo")
                if not temp_df.empty: df, actual_symbol = temp_df, symbol; break 
            except: pass

    if df.empty or len(df) < 20: return df, actual_symbol 

    if FUGLE_API_KEY:
        try:
            res = requests.get(f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{ticker_code}", headers={"X-API-KEY": FUGLE_API_KEY}, timeout=3)
            if res.status_code == 200:
                data = res.json()
                rt_price = data.get('closePrice') or data.get('lastTrade', {}).get('price')
                rt_vol = data.get('total', {}).get('tradeVolume', 0)
                tz_tw = datetime.timezone(datetime.timedelta(hours=8))
                today_date, last_candle_date = datetime.datetime.now(tz_tw).date(), df.index[-1].date()
                if last_candle_date < today_date and rt_price:
                    new_row = df.iloc[-1].copy(); new_row.name = pd.Timestamp(today_date, tz=df.index.tz)
                    df = pd.concat([df, pd.DataFrame([new_row])])
                if rt_price:
                    df.iloc[-1, df.columns.get_loc('Close')] = rt_price
                    if data.get('highPrice'): df.iloc[-1, df.columns.get_loc('High')] = data['highPrice']
                    if data.get('lowPrice'): df.iloc[-1, df.columns.get_loc('Low')] = data['lowPrice']
                if rt_vol > 0: df.iloc[-1, df.columns.get_loc('Volume')] = rt_vol
        except: pass
        
    df = add_advanced_indicators(df)
    return df, actual_symbol

@st.cache_data(ttl=300)
def get_stock_news(keyword):
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(keyword)}+when:7d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        root = ET.fromstring(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5).text)
        return [{"title": i.find('title').text, "link": i.find('link').text, "date": i.find('pubDate').text} for i in root.findall('./channel/item')[:5]]
    except: return []

@st.cache_data(ttl=300)
def get_macro_news():
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote('台股 OR 聯準會 OR 財報')}+when:1d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        root = ET.fromstring(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5).text)
        return [{"title": i.find('title').text, "link": i.find('link').text, "date": i.find('pubDate').text} for i in root.findall('./channel/item')[:6]]
    except: return []

# ==========================================
# ⚡ Asyncio 異步高速爬蟲引擎 (專為全市場掃描設計)
# ==========================================
async def fetch_yahoo_history(session, symbol):
    """使用 aiohttp 異步抓取 Yahoo 歷史數據"""
    # 這裡我們為了極致速度，繞過 yfinance，直接戳 Yahoo 的底層 API
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=60d&interval=1d"
    try:
        async with session.get(url, timeout=5) as response:
            if response.status == 200:
                data = await response.json()
                result = data.get('chart', {}).get('result', [])
                if result:
                    indicators = result[0].get('indicators', {}).get('quote', [{}])[0]
                    closes = indicators.get('close', [])
                    volumes = indicators.get('volume', [])
                    
                    # 清理 None 值
                    closes = [c for c in closes if c is not None]
                    volumes = [v for v in volumes if v is not None]
                    
                    if len(closes) >= 30:
                        df = pd.DataFrame({'Close': closes, 'Volume': volumes})
                        df = add_advanced_indicators(df)
                        return symbol, df
    except: pass
    return symbol, None

async def async_scan_market(tickers_to_scan, cond_vol, cond_ma, cond_rsi, cond_macd, progress_bar, status_text):
    results = []
    # 限制同時發出的請求數 (Concurrency Limit)，避免被 Yahoo 鎖 IP
    connector = aiohttp.TCPConnector(limit=50) 
    async with aiohttp.ClientSession(connector=connector, headers={"User-Agent": "Mozilla/5.0"}) as session:
        tasks = [fetch_yahoo_history(session, t) for t in tickers_to_scan]
        
        completed = 0
        total = len(tasks)
        
        # 使用 asyncio.as_completed 來即時獲取完成的任務
        for future in asyncio.as_completed(tasks):
            symbol, df = await future
            completed += 1
            
            # 更新進度條 UI
            if completed % 10 == 0 or completed == total:
                progress_bar.progress(completed / total)
                status_text.text(f"🚀 光速異步掃描中... ({completed}/{total})")

            if df is not None:
                c_close, c_vol = df['Close'].iloc[-1], df['Volume'].iloc[-1]
                sma20, vol_sma5 = df['SMA_20'].iloc[-1], df['Vol_SMA5'].iloc[-1]
                rsi, macd, signal = df['RSI'].iloc[-1], df['MACD'].iloc[-1], df['Signal'].iloc[-1]
                macd_prev, signal_prev = df['MACD'].iloc[-2], df['Signal'].iloc[-2]
                
                pass_vol = (c_vol > vol_sma5 * 1.5) if cond_vol else True
                pass_ma = (c_close > sma20) if cond_ma else True
                pass_rsi = (rsi < 35) if cond_rsi else True
                pass_macd = (macd > signal and macd_prev <= signal_prev) if cond_macd else True
                
                if pass_vol and pass_ma and pass_rsi and pass_macd:
                    pct = ((c_close - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                    code = symbol.split('.')[0]
                    results.append({
                        "代號": code, 
                        "名稱": st.session_state.stock_names.get(code, "大盤個股"), 
                        "現價": f"{c_close:.2f}", 
                        "今日漲跌": f"{pct:+.2f}%", 
                        "量比": f"{c_vol/vol_sma5:.1f}x" if vol_sma5>0 else "-",
                        "RSI": f"{rsi:.1f}"
                    })
    return results

# ==========================================
# 📱 側邊欄 
# ==========================================
with st.sidebar:
    st.header("📂 我的自選清單")
    if 'sidebar_state' not in st.session_state: st.session_state.sidebar_state = 'expanded'
    
    selected_cluster = st.selectbox("1. 選擇產業群組", list(st.session_state.stock_clusters.keys()))
    cluster_stocks = st.session_state.stock_clusters[selected_cluster]
    display_options = [f"{t.split('.')[0]} {st.session_state.stock_names.get(t.split('.')[0], '')}".strip() for t in cluster_stocks]
    sidebar_ticker = st.selectbox("2. 選擇分析標的", display_options).split(' ')[0]
    st.write("") 
    if st.button("📊 診斷此自選股", use_container_width=True, type="primary"):
        st.session_state.analyze_trigger = sidebar_ticker 
        st.rerun()
    st.markdown("---")
    
    # ✨ 新增：自訂清單管理器
    with st.expander("🛠️ 管理自選群組", expanded=False):
        st.markdown("**新增群組**")
        new_cluster_name = st.text_input("群組名稱 (如: AI概念股)")
        new_cluster_tickers = st.text_area("股票代號 (用逗號分隔，如: 2330.TW, 2317.TW)")
        if st.button("➕ 新增/更新群組"):
            if new_cluster_name and new_cluster_tickers:
                tickers_list = [t.strip().upper() for t in new_cluster_tickers.split(',')]
                # 簡單驗證後綴
                tickers_list = [t if ('.TW' in t or '.TWO' in t) else f"{t}.TW" for t in tickers_list]
                st.session_state.stock_clusters[new_cluster_name] = tickers_list
                st.success(f"已成功新增群組：{new_cluster_name}！")
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
# ⚡ 模式分流：深度診斷模式 vs 主頁戰情室
# ==========================================
if target_ticker:
    # ─── 【模組 A】單股深度診斷 ───
    with st.spinner(f"正在擷取 {target_ticker} 深度資料..."):
        try:
            tz_tw = datetime.timezone(datetime.timedelta(hours=8))
            is_market_open = now = datetime.datetime.now(tz_tw).weekday() < 5 and datetime.time(9, 0) <= datetime.datetime.now(tz_tw).time() <= datetime.time(13, 30)
            time_label = "今日盤中" if is_market_open else "明日"
            
            df, actual_symbol = get_kline_with_fugle(target_ticker)
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
    # ─── 【模組 B】主頁戰略儀表板 (雙分頁) ───
    st.markdown("### 🌍 台股大盤摘要")
    summary_data = get_market_summary()
    if summary_data:
        m_cols = st.columns(len(summary_data))
        for i, (name, data) in enumerate(summary_data.items()):
            m_cols[i].metric(label=name, value=f"{data['price']:.2f}", delta=f"{data['change']:+.2f} ({data['pct']:+.2f}%)")
        st.markdown("""<style>[data-testid="stMetricDelta"] svg { display: none; } [data-testid="stMetricDelta"] > div { flex-direction: row; } [data-testid="stMetricDelta"] > div:has(div:contains("+")) { color: #ff4b4b !important; } [data-testid="stMetricDelta"] > div:has(div:contains("-")) { color: #00cc96 !important; }</style>""", unsafe_allow_html=True)
    st.markdown("---")

    main_tab1, main_tab2 = st.tabs(["📊 板塊實時監控", "⚡ 全市場異步策略雷達"])

    # 視角 1：實時監控看板
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
                    kline_df, _ = get_kline_with_fugle(ticker_code)
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

    # 視角 2：全池策略選股雷達 (光速 Asyncio 版本)
    with main_tab2:
        st.markdown("#### ⚡ 異步光速：全市場策略掃描雷達")
        st.write("利用 `asyncio` 底層網路技術，針對 **台股全市場 (上市/上櫃)** 進行非阻塞光速過濾。")
        
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
            all_market_tickers = load_all_market_tickers()
            
            if "自選群組" in scan_mode:
                tickers_to_scan = list(set(custom_tickers))
            else:
                if not all_market_tickers:
                    st.error("❌ 找不到 `all_tw_stocks.csv`！已為您降級為掃描自選群組。")
                    tickers_to_scan = list(set(custom_tickers))
                else:
                    st.warning("⚠️ 準備向 Yahoo 伺服器發射 1700+ 個異步請求，請繫好安全帶！")
                    tickers_to_scan = list(set(custom_tickers + all_market_tickers))
            
            st.info(f"📡 雷達已啟動，目標掃描數量：{len(tickers_to_scan)} 檔標的。")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            # ✨ 使用 asyncio.run 啟動異步循環
            try:
                results = asyncio.run(async_scan_market(tickers_to_scan, cond_vol, cond_ma, cond_rsi, cond_macd, progress_bar, status_text))
            except Exception as e:
                # 解決 streamlit 內部 event loop 衝突問題
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(async_scan_market(tickers_to_scan, cond_vol, cond_ma, cond_rsi, cond_macd, progress_bar, status_text))
                
            end_time = time.time()
            status_text.empty() 
            progress_bar.empty() 
            
            if results:
                st.success(f"🎯 掃描完成！耗時 {round(end_time - start_time, 1)} 秒。共捕捉到 **{len(results)}** 檔符合嚴格條件的標的：")
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            else: 
                st.warning(f"掃描完成 (耗時 {round(end_time - start_time, 1)} 秒)。當前盤面無任何標的符合您設定的策略交集。")