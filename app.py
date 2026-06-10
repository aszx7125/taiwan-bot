"""
台股AI量化系統 - 完整雙向版 v2.0
支援多頭+空頭獨立預測
不依賴外部 config.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import concurrent.futures

# ========== 內嵌設定 (避免導入錯誤) ==========
def get_fugle_key():
    try: 
        return st.secrets["FUGLE_API_KEY"]
    except: 
        return ""

DEFAULT_CLUSTERS = {
    "半導體": ["2330.TW", "3711.TW", "2454.TW", "2303.TW", "5347.TWO", "3034.TW"],
    "矽光子": ["3363.TWO", "3450.TW", "6451.TW", "3081.TWO", "4979.TWO", "3163.TWO"],
    "伺服器": ["2382.TW", "3231.TW", "6669.TW", "2376.TW", "3017.TW", "5274.TWO"],
    "金融股": ["2881.TW", "2882.TW", "2886.TW", "2891.TW", "2884.TW"],
    "傳統產業": ["1101.TW", "2002.TW", "2603.TW", "2609.TW", "2618.TW"],
    "ETF": ["0050.TW", "0056.TW", "00878.TW", "00919.TW", "00929.TW"]
}

DEFAULT_NAMES = {
    "3491": "昇達科", "3138": "耀登", "6285": "啟碁", "2383": "華通", "2314": "台揚",
    "3363": "上詮", "3450": "聯鈞", "6451": "訊芯-KY", "3081": "聯亞", "4979": "華星光", "3163": "波若威",
    "2330": "台積電", "3711": "日月光投控", "2454": "聯發科", "2303": "聯電", "5347": "世界先進", "3034": "聯詠",
    "2382": "廣達", "3231": "緯創", "6669": "緯穎", "2376": "技嘉", "3017": "奇鋐", "5274": "信驊",
    "3443": "創意", "3661": "世芯-KY", "3228": "金麗科", "3324": "雙鴻", "3033": "威健", "3653": "健策", "2356": "英業達",
    "3234": "光環", "4908": "前鼎", "3596": "智易", "2345": "智邦", "5388": "中磊",
    "1503": "士電", "1513": "中興電", "1514": "亞力", "1519": "華城", "1609": "大亞", "1605": "華新", "1504": "東元", 
    "8996": "高力", "6806": "森崴能源", "6869": "雲豹能源", "3708": "上緯投控", "6443": "元晶",
    "2002": "中鋼", "2603": "長榮", "1101": "台泥", "2609": "陽明", "2618": "長榮航",
    "2881": "富邦金", "2882": "國泰金", "2886": "兆豐金", "2891": "中信金", "2884": "玉山金",
    "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息", "00919": "群益精選高息", "00929": "復華科技優息"
}

# ========== Imports ==========
from data_fetcher import load_all_market_tickers, get_market_summary, get_kline_with_fugle, get_stock_news
from data_pipeline import load_market_snapshot, get_snapshot_dict, get_realtime_quote, fetch_advanced_backtest, trigger_github_workflow, load_model_metrics
from ai_engine import DualCoreBrain

st.set_page_config(page_title="台股量化旗艦終端 v2.0", page_icon="📈", layout="wide")

FUGLE_API_KEY = get_fugle_key()
if 'stock_clusters' not in st.session_state:
    st.session_state.stock_clusters = DEFAULT_CLUSTERS.copy()
if 'stock_names' not in st.session_state:
    st.session_state.stock_names = DEFAULT_NAMES.copy()


@st.cache_resource
def load_brain():
    return DualCoreBrain()

brain = load_brain()


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_market_summary():
    return get_market_summary()


@st.cache_data
def get_stock_name_from_csv(ticker: str) -> str:
    try:
        clean_code = str(ticker).split('.')[0].strip()
        df_all = load_all_market_tickers()
        if not df_all.empty:
            df_all.columns = [col.lower() for col in df_all.columns]
            match = df_all[df_all['ticker'].astype(str).str.contains(clean_code)]
            if not match.empty:
                return str(match.iloc[0]['name'])
    except Exception:
        pass
    return ticker


def compute_fear_greed(twii_pct: float, snapshot_data: list) -> tuple[int, str, str]:
    twii_pct = np.nan_to_num(twii_pct, nan=0.0, posinf=5.0, neginf=-5.0)
    twii_pct = np.clip(twii_pct, -10, 10)
    
    pct_score = float(np.clip(50 + (twii_pct * 16.67), 0, 100))
    bull_ratio_score = 50.0
    rs_score = 50.0
    vol_score = 50.0

    if snapshot_data and len(snapshot_data) > 0:
        df = pd.DataFrame(snapshot_data)
        if 'rs_index' in df.columns:
            df['rs_index'] = pd.to_numeric(df['rs_index'], errors='coerce').fillna(0)
            df['rs_index'] = df['rs_index'].replace([np.inf, -np.inf], 0).clip(-100, 100)
            bull_ratio_score = float((df['rs_index'] > 0).mean()) * 100
            mean_rs = float(df['rs_index'].mean())
            rs_score = float(np.clip(50 + (mean_rs * 5), 0, 100))
        
        if 'vol_ratio' in df.columns:
            df['vol_ratio'] = pd.to_numeric(df['vol_ratio'], errors='coerce').fillna(1.0)
            df['vol_ratio'] = df['vol_ratio'].replace([np.inf, -np.inf], 1.0).clip(0.1, 10)
            mean_vol = float(df['vol_ratio'].mean())
            vol_score = float(np.clip(50 + (mean_vol - 1.0) * 50, 0, 100))

    final = pct_score * 0.35 + bull_ratio_score * 0.35 + rs_score * 0.20 + vol_score * 0.10
    index_val = int(np.clip(final, 0, 100))

    if index_val >= 75:   label, color = "極度貪婪", "#ff4b4b"
    elif index_val >= 60: label, color = "貪婪",     "#ffa500"
    elif index_val >= 45: label, color = "中性偏多", "#ffc107"
    elif index_val >= 30: label, color = "恐懼",     "#00cc96"
    else:                 label, color = "極度恐懼", "#00ccff"

    return index_val, label, color


def render_fear_greed_gauge(index_val: int, label: str, color: str):
    st.markdown(f"""
    <div style="background:#1e1e1e;border-radius:12px;padding:18px;text-align:center;border:1px solid #333;">
        <div style="color:#aaa;font-size:13px;margin-bottom:6px;">📊 台股恐懼貪婪指數</div>
        <div style="font-size:38px;font-weight:700;color:{color};">{index_val}</div>
        <div style="font-size:16px;color:{color};font-weight:600;margin-bottom:12px;">{label}</div>
        <div style="background:#333;border-radius:8px;height:10px;width:100%;overflow:hidden;">
            <div style="background:{color};width:{index_val}%;height:100%;border-radius:8px;transition:width 0.6s;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:11px;color:#666;margin-top:4px;">
            <span>極度恐懼</span><span>中性</span><span>極度貪婪</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_model_health_board(metrics):
    """四核心健康度看板"""
    st.markdown("### 🧪 四核心AI大腦：盲測勝率矩陣")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 多頭模型")
        wr_long = metrics.get('lgbm', {}).get('blind_win_rate', 0)
        color_l = "#00cc96" if wr_long > 0.55 else ("#ffc107" if wr_long > 0.50 else "#ff4b4b")
        st.markdown(f"""
        <div style='background:#1e1e1e; padding:12px; border-left:4px solid #00cc96; border-radius:5px; margin-bottom:8px;'>
            <span style='color:#aaa;font-size:12px;'>LightGBM 多頭</span><br>
            <span style='font-size:24px; color:{color_l}; font-weight:bold;'>{wr_long*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        wr_lstm_l = metrics.get('lstm', {}).get('blind_win_rate', 0)
        color_ll = "#00cc96" if wr_lstm_l > 0.53 else ("#ffc107" if wr_lstm_l > 0.50 else "#ff4b4b")
        st.markdown(f"""
        <div style='background:#1e1e1e; padding:12px; border-left:4px solid #00cc96; border-radius:5px;'>
            <span style='color:#aaa;font-size:12px;'>LSTM 多頭</span><br>
            <span style='font-size:24px; color:{color_ll}; font-weight:bold;'>{wr_lstm_l*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🔴 空頭模型")
        wr_short = metrics.get('short', {}).get('lgbm', {}).get('blind_win_rate', 0)
        if wr_short > 0:
            color_s = "#ff4b4b" if wr_short > 0.55 else ("#ff9966" if wr_short > 0.50 else "#888")
            st.markdown(f"""
            <div style='background:#1e1e1e; padding:12px; border-left:4px solid #ff4b4b; border-radius:5px; margin-bottom:8px;'>
                <span style='color:#aaa;font-size:12px;'>LightGBM 空頭</span><br>
                <span style='font-size:24px; color:{color_s}; font-weight:bold;'>{wr_short*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#1e1e1e; padding:12px; border-left:4px solid #666; border-radius:5px; margin-bottom:8px;'>
                <span style='color:#666;font-size:12px;'>LightGBM 空頭</span><br>
                <span style='font-size:18px; color:#666;'>未訓練</span>
            </div>
            """, unsafe_allow_html=True)
        
        wr_lstm_s = metrics.get('short', {}).get('lstm', {}).get('blind_win_rate', 0)
        if wr_lstm_s > 0:
            color_ls = "#ff4b4b" if wr_lstm_s > 0.53 else ("#ff9966" if wr_lstm_s > 0.50 else "#888")
            st.markdown(f"""
            <div style='background:#1e1e1e; padding:12px; border-left:4px solid #ff4b4b; border-radius:5px;'>
                <span style='color:#aaa;font-size:12px;'>LSTM 空頭</span><br>
                <span style='font-size:24px; color:{color_ls}; font-weight:bold;'>{wr_lstm_s*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#1e1e1e; padding:12px; border-left:4px solid #666; border-radius:5px;'>
                <span style='color:#666;font-size:12px;'>LSTM 空頭</span><br>
                <span style='font-size:18px; color:#666;'>未訓練</span>
            </div>
            """, unsafe_allow_html=True)


def render_top20_card(s, is_short=False):
    """渲染 TOP20 卡片"""
    color = s.get('box_color', '#00cc96' if not is_short else '#ff4b4b')
    prob = s.get('win_prob', 0) * 100 if not is_short else s.get('short_prob', 0) * 100
    
    st.markdown(f"""
    <div style="border: 2px solid {color}; border-radius: 10px; padding: 18px; background-color: #1e1e1e; margin-bottom: 12px;">
        <h4 style="color: {color}; margin-top: 0;">{'🔻' if is_short else '🎯'} {s['ticker']} {s['name']}</h4>
        <div style="display: flex; justify-content: space-between;">
            <div>
                <span style="color: gray; font-size: 13px;">{'空頭' if is_short else '多頭'}機率</span><br>
                <b style="font-size: 22px; color: {color};">{prob:.1f}%</b>
            </div>
            <div style="text-align: right;">
                <span style="color: gray; font-size: 13px;">進場價</span><br>
                <b style="font-size: 20px;">{s['entry_price']:.2f}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ========== 主程式 ==========
with st.sidebar:
    st.header("📂 我的自選清單")
    selected_cluster = st.selectbox("1. 選擇產業群組", list(st.session_state.stock_clusters.keys()))
    cluster_stocks = st.session_state.stock_clusters[selected_cluster]
    
    display_options = []
    for t in cluster_stocks:
        base = t.split('.')[0]
        name = st.session_state.stock_names.get(base)
        if not name or name == base:
            name = get_stock_name_from_csv(base)
            st.session_state.stock_names[base] = name
        display_options.append(f"{base} {name}".strip())
        
    sidebar_ticker = st.selectbox("2. 選擇分析標的", display_options).split(' ')[0]

    if st.button("📊 診斷此自選股", use_container_width=True, type="primary"):
        st.session_state.analyze_trigger = sidebar_ticker
        st.rerun()

    st.markdown("---")
    st.header("💰 實盤資金管理")
    user_capital = st.number_input("初始本金 (TWD)", min_value=1000, value=1000000, step=10000)
    user_max_pos = st.slider("最大持倉檔數", 1, 10, 5)

    st.markdown("---")
    st.header("🛠️ 遠端自動化控制")
    if st.button("🔥 啟動全市場 AI 掃描", use_container_width=True):
        success, msg = trigger_github_workflow("daily_scan.yml")
        if success: st.success(msg)
        else: st.error(msg)

    if st.button("🧠 啟動雙模型重新訓練", use_container_width=True):
        success, msg = trigger_github_workflow("train_ai.yml")
        if success: st.info(msg)
        else: st.error(msg)

    st.markdown("---")
    if brain.is_lstm_ready:  st.success("🔮 LSTM 多頭已連動")
    else:                    st.warning("⚪ LSTM 多頭未載入")
    if brain.is_lgbm_ready:  st.success("🌳 LGBM 多頭正常")
    else:                    st.error("🚨 LGBM 多頭缺失")
    
    if hasattr(brain, 'is_lgbm_short_ready') and brain.is_lgbm_short_ready:
        st.success("🔴 LGBM 空頭已連動")
    if hasattr(brain, 'is_lstm_short_ready') and brain.is_lstm_short_ready:
        st.success("🔴 LSTM 空頭已連動")


st.title("⚡ 台股戰情分析終端 v2.0")
st.caption("🟢 多頭 | 🔴 空頭 | ⚪ 盤整 | 四核心AI驅動")

col1, col2 = st.columns([3, 1])
with col1:
    manual_ticker = st.text_input("輸入股票代號", "", label_visibility="collapsed", placeholder="例如: 2330")
with col2:
    analyze_manual_btn = st.button("單股掃描", use_container_width=True)
st.markdown("---")

target_ticker = st.session_state.pop('analyze_trigger', None) or (
    manual_ticker.strip().upper() if analyze_manual_btn else None
)

if target_ticker:
    base_ticker = target_ticker.split('.')[0]
    c_name = st.session_state.stock_names.get(base_ticker)
    if not c_name or c_name == base_ticker:
        c_name = get_stock_name_from_csv(base_ticker)
        st.session_state.stock_names[base_ticker] = c_name

    with st.spinner(f"正在分析 {base_ticker} {c_name}..."):
        df_daily, df_hourly, actual_symbol = get_kline_with_fugle(target_ticker, FUGLE_API_KEY)
        if df_daily.empty:
            st.error("❌ 數據不足")
        else:
            news_s = get_stock_news(c_name)
            today = df_daily.iloc[-1]
            yesterday = df_daily.iloc[-2]
            entry_price = float(today.get('Close', 0.0))
            y_close = float(yesterday.get('Close', entry_price))
            rt_p, rt_v, _ = get_realtime_quote(base_ticker, FUGLE_API_KEY)
            if rt_p > 0: entry_price = rt_p

            if y_close > 0.01:
                p_change = ((entry_price - y_close) / max(y_close, 0.01)) * 100
                p_change = np.nan_to_num(p_change, nan=0.0, posinf=20.0, neginf=-20.0)
            else:
                p_change = 0.0
            
            res_level = float(today.get('Res_20', entry_price * 1.05))
            sup_level = float(today.get('Sup_20', entry_price * 0.95))
            atr_14 = float(yesterday.get('ATR_14', entry_price * 0.05))
            broker_conc = float(today.get('Broker_Concentration', 0.0))

            micro_status_text = "⚪ 1h 均線下弱勢震盪"
            if not df_hourly.empty and len(df_hourly) >= 2:
                last_hour = df_hourly.iloc[-1]
                if bool(last_hour.get('Micro_Sniper_Trigger', False)): micro_status_text = "🔥 帶量突破 1h 均線"
                elif bool(last_hour.get('MACD_Cross_Up', False)): micro_status_text = "📈 1h MACD 金叉發動"
                elif bool(last_hour.get('Vol_Surge_1h', False)): micro_status_text = "🌊 1h 微觀異常爆量"

            low_vol_pb = bool(today.get('Low_Vol_Pullback', False))
            smc_text = "量縮回踩" if low_vol_pb else "一般常態箱體震盪"
            stop_loss = round(min(entry_price - (1.5 * atr_14), sup_level * 0.985), 2)
            take_profit = round(res_level, 2) if low_vol_pb else round(res_level + (atr_14 * 1.0), 2)

            snapshot_dict = get_snapshot_dict(load_market_snapshot())
            feat_dict = brain.extract_features(base_ticker, entry_price, snapshot_dict, current_vol=rt_v, fallback_atr=atr_14, fallback_pattern=smc_text)
            
            # 使用雙向預測
            if hasattr(brain, 'predict_bidirectional'):
                result = brain.predict_bidirectional([feat_dict])
                final_prob = result['long_prob'][0]
                short_prob = result['short_prob'][0]
                neutral_prob = result['neutral_prob'][0]
                signal = result['signal'][0]
            else:
                final_prob = brain.predict_win_rates([feat_dict])[0]
                short_prob = 1.0 - final_prob
                neutral_prob = 0.0
                signal = "LONG" if final_prob > 0.55 else "SHORT" if short_prob > 0.55 else "NEUTRAL"
            
            final_prob = np.nan_to_num(final_prob, nan=0.5, posinf=0.99, neginf=0.01)
            short_prob = np.nan_to_num(short_prob, nan=0.5, posinf=0.99, neginf=0.01)
            neutral_prob = np.nan_to_num(neutral_prob, nan=0.0, posinf=0.5, neginf=0.0)

            st.subheader(f"🧬 {base_ticker} {c_name} 四核雙向分析")
            
            # 雙向儀表板
            col_l, col_n, col_s = st.columns(3)
            with col_l:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#003d2a,#00cc96);padding:15px;border-radius:8px;text-align:center;'>
                    <div style='color:rgba(255,255,255,0.8);font-size:12px;'>🟢 多頭</div>
                    <div style='color:#fff;font-size:28px;font-weight:bold;'>{final_prob*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_n:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#2a2a2a,#666);padding:15px;border-radius:8px;text-align:center;'>
                    <div style='color:rgba(255,255,255,0.8);font-size:12px;'>⚪ 盤整</div>
                    <div style='color:#fff;font-size:28px;font-weight:bold;'>{neutral_prob*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#3d0000,#ff4b4b);padding:15px;border-radius:8px;text-align:center;'>
                    <div style='color:rgba(255,255,255,0.8);font-size:12px;'>🔴 空頭</div>
                    <div style='color:#fff;font-size:28px;font-weight:bold;'>{short_prob*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # 訊號
            signal_map = {
                "STRONG_LONG": ("🟢 強勢做多", "#00ff00"),
                "LONG": ("🟢 偏多", "#00cc96"),
                "NEUTRAL": ("⚪ 中性", "#888"),
                "SHORT": ("🔴 偏空", "#ff9966"),
                "STRONG_SHORT": ("🔴 強勢放空", "#ff0000"),
                "HIGH_VOLATILITY": ("⚡ 高波動", "#ff00ff"),
                "WAIT": ("⏸ 觀望", "#666"),
            }
            sig_text, sig_color = signal_map.get(signal, ("未知", "#888"))
            
            st.markdown(f"""
            <div style='margin:15px 0;padding:15px;background:#1e1e1e;border-radius:8px;
                        border-left:4px solid {sig_color};text-align:center;'>
                <span style='font-size:20px;font-weight:bold;color:{sig_color};'>{sig_text}</span>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("當前現價", f"{entry_price:.2f}", f"{p_change:+.2f}%")
            m2.metric("SMC 結構", smc_text)
            m3.metric("1h 微觀", micro_status_text)
            m4.metric("機構集中", f"{broker_conc*100:.1f}%")

            if st.button("⬅️ 返回", use_container_width=True):
                st.rerun()

else:
    # 主頁
    st.markdown("### 🌍 大盤與情緒摘要")
    summary = get_cached_market_summary()
    snapshot = load_market_snapshot()
    snapshot_data = snapshot.get('data', []) if snapshot else []

    if summary:
        twii_data = summary.get("加權指數", {"pct": 0.0, "price": 0.0, "change": 0.0})
        twii_pct = float(twii_data.get('pct', 0.0))
        greed_val, greed_label, greed_color = compute_fear_greed(twii_pct, snapshot_data)

        c_idx, c_greed = st.columns([3, 1])
        with c_idx:
            cols = st.columns(len(summary))
            for i, (name, data) in enumerate(summary.items()):
                cols[i].metric(name, f"{data['price']:.2f}", f"{data['change']:+.2f} ({data['pct']:+.2f}%)")
        with c_greed: 
            render_fear_greed_gauge(greed_val, greed_label, greed_color)

    st.markdown("---")
    metrics = load_model_metrics()
    render_model_health_board(metrics)
    st.markdown("---")

    # 6個頁籤 (新增空頭)
    tab1, tab2, tab3, tab3s, tab5, tab6 = st.tabs([
        "📊 自選", "🔮 多空趨勢", "🎯 多頭TOP20", "🔻 空頭TOP20", "⚖️ 對撞", "🔬 回測"
    ])

    with tab1:
        st.markdown(f"#### 【{selected_cluster}】即時行情")
        st.info("💡 即時行情功能 - 請查看側邊欄選擇個股")

    with tab2:
        st.markdown("#### 🔮 全市場雙向分析")
        snap = load_market_snapshot()
        if snap and 'data' in snap and len(snap['data']) > 0:
            raw_list = snap['data']
            snap_dict = get_snapshot_dict(snap)
            bulk_features = []
            
            for item in raw_list[:100]:  # 限制100檔加速
                ticker = str(item.get('代號', '')).split('.')[0].strip()
                ep = float(item.get('現價', 0.0))
                if ep > 0:
                    bulk_features.append(brain.extract_features(ticker, ep, snap_dict, 
                                                               current_vol=float(item.get('成交量', 0.0))))
            
            if bulk_features:
                if hasattr(brain, 'predict_bidirectional'):
                    result = brain.predict_bidirectional(bulk_features)
                    long_probs = result['long_prob']
                    short_probs = result['short_prob']
                    signals = result['signal']
                else:
                    long_probs = brain.predict_win_rates(bulk_features)
                    short_probs = 1 - long_probs
                    signals = ['LONG' if p > 0.55 else 'SHORT' if p < 0.45 else 'NEUTRAL' for p in long_probs]
                
                avg_long = float(np.mean(long_probs)) * 100
                avg_short = float(np.mean(short_probs)) * 100
                
                c1, c2, c3 = st.columns(3)
                c1.metric("🟢 多頭均值", f"{avg_long:.1f}%")
                c2.metric("⚪ 中性", f"{100-avg_long-avg_short:.1f}%")
                c3.metric("🔴 空頭均值", f"{avg_short:.1f}%")
                
                strong_long = sum(1 for s in signals if 'LONG' in s)
                strong_short = sum(1 for s in signals if 'SHORT' in s)
                
                st.markdown(f"**市場狀態:** 多頭 {strong_long}檔 | 空頭 {strong_short}檔")

    with tab3:
        st.markdown("#### 🎯 多頭 TOP20")
        st.info("載入中...")
        # 簡化版，實際應加入完整邏輯

    with tab3s:
        st.markdown("#### 🔻 空頭 TOP20 (放空名單)")
        st.info("載入中...")
        # 簡化版

    with tab5:
        st.markdown("#### ⚖️ 實盤對撞")
        st.info("功能開發中")

    with tab6:
        st.markdown("#### 🔬 策略回測")
        if st.button("執行回測"):
            with st.spinner("回測中..."):
                res = fetch_advanced_backtest(initial_cap=user_capital, max_pos=user_max_pos)
                if res.get("status") == "ready":
                    st.success(f"勝率: {res['ai_strat']['wr']*100:.1f}% | 報酬: {res['account_pct']:.2f}%")
                else:
                    st.warning("回測資料不足")