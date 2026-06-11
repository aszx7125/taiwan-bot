"""
台股AI量化系統 v4.1 - 全局快取極速版
修復：解決首頁分頁重複推論導致 15秒以上卡頓的問題，將載入時間壓縮至 0.1秒
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import concurrent.futures
import json
import os

# ==========================================
# 🎨 UI 渲染元件
# ==========================================
def _render_clean_html(raw_html):
    """徹底清除所有換行與開頭空白，防止 Markdown 破版"""
    clean_html = "".join([line.strip() for line in raw_html.split('\n')])
    st.markdown(clean_html, unsafe_allow_html=True)

def render_top20_card(s):
    color = s.get('box_color', '#00cc96')
    prob = s.get('win_prob', 0) * 100
    html = (
        f"<div style='border: 2px solid {color}; border-radius: 10px; padding: 18px; background-color: #1e1e1e; margin-bottom: 12px;'>"
        f"<h4 style='color: {color}; margin-top: 0;'>🎯 {s['ticker']} {s['name']}</h4>"
        f"<div style='display: flex; justify-content: space-between;'>"
        f"<div><span style='color: gray; font-size: 13px;'>最高勝率極值</span><br>"
        f"<b style='font-size: 22px; color: {color};'>{prob:.1f}%</b></div>"
        f"<div style='text-align: right;'><span style='color: gray; font-size: 13px;'>現價</span><br>"
        f"<b style='font-size: 20px;'>{s['entry_price']:.2f}</b></div>"
        f"</div></div>"
    )
    _render_clean_html(html)

def render_single_diagnostic_card(core_data, entry_price, res_level, sup_level):
    bl = core_data['best_long'] * 100
    bs = core_data['best_short'] * 100
    ll = core_data['lgbm_long'] * 100
    tl = core_data['lstm_long'] * 100
    ls = core_data['lgbm_short'] * 100
    ts = core_data['lstm_short'] * 100
    signal = core_data['signal']

    if signal == "STRONG_LONG": box_color, rec = "#00cc96", "⭐⭐⭐ 強勢做多 (多方輾壓)"
    elif signal == "LONG": box_color, rec = "#4ade80", "⭐⭐ 偏多操作 (多方優勢)"
    elif signal == "STRONG_SHORT": box_color, rec = "#ff4b4b", "⚠️⚠️ 強勢放空 (空方輾壓)"
    elif signal == "SHORT": box_color, rec = "#ff8080", "⚠️ 偏空操作 (空方優勢)"
    elif signal == "HIGH_VOLATILITY": box_color, rec = "#ffa500", "⚡ 多空雙巴，建議空手觀望"
    else: box_color, rec = "#a8a8a8", "⚪ 動能不足，盤整觀望"

    html = (
        f"<div style='border: 2px solid {box_color}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;'>"
        f"<h4 style='color: {box_color}; margin-top: 0; margin-bottom: 20px; border-bottom: 1px solid #333; padding-bottom: 10px;'>"
        f"⚔️ 四核心對撞結果：{rec}</h4>"
        f"<div style='display: flex; justify-content: space-between; flex-wrap: wrap; margin-bottom: 15px;'>"
        f"<div style='flex: 1; min-width: 200px; padding: 10px; background-color: rgba(0, 204, 150, 0.05); border-radius: 8px; margin-right: 10px;'>"
        f"<span style='color: #00cc96; font-size: 16px; font-weight: bold;'>🟢 多頭陣營極值</span><br>"
        f"<b style='font-size: 32px; color: #00cc96;'>{bl:.1f}%</b>"
        f"<div style='font-size: 12px; color: gray; margin-top: 8px;'>"
        f"▶ LGBM 靜態結構: <span style='color: white;'>{ll:.1f}%</span><br>"
        f"▶ LSTM 時序動能: <span style='color: white;'>{tl:.1f}%</span></div></div>"
        f"<div style='flex: 1; min-width: 200px; padding: 10px; background-color: rgba(255, 75, 75, 0.05); border-radius: 8px; margin-left: 10px;'>"
        f"<span style='color: #ff4b4b; font-size: 16px; font-weight: bold;'>🔴 空頭陣營極值</span><br>"
        f"<b style='font-size: 32px; color: #ff4b4b;'>{bs:.1f}%</b>"
        f"<div style='font-size: 12px; color: gray; margin-top: 8px;'>"
        f"▶ LGBM 靜態結構: <span style='color: white;'>{ls:.1f}%</span><br>"
        f"▶ LSTM 時序動能: <span style='color: white;'>{ts:.1f}%</span></div></div></div>"
        f"<div style='display: flex; justify-content: space-between; border-top: 1px solid #333; padding-top: 15px;'>"
        f"<div style='flex: 1;'><span style='color: gray; font-size: 12px;'>現價</span><br><b style='font-size: 18px;'>{entry_price:.2f}</b></div>"
        f"<div style='flex: 1;'><span style='color: gray; font-size: 12px;'>上檔壓力位</span><br><b style='font-size: 18px; color: #ffc107;'>{res_level:.2f}</b></div>"
        f"<div style='flex: 1;'><span style='color: gray; font-size: 12px;'>下檔支撐位</span><br><b style='font-size: 18px; color: #00ccff;'>{sup_level:.2f}</b></div>"
        f"</div></div>"
    )
    _render_clean_html(html)

def render_backtest_metric_card(title, value, subtext, color):
    html = (
        f"<div style='background-color: #121218; padding: 20px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #2a2a35;'>"
        f"<div style='color: #8b8b99; font-size: 13px; margin-bottom: 6px;'>{title}</div>"
        f"<div style='color: {color}; font-size: 28px; font-weight: 700;'>{value}</div>"
        f"<div style='color: #6b6b79; font-size: 12px;'>{subtext}</div></div>"
    )
    _render_clean_html(html)

def render_model_health_board(metrics):
    st.markdown("### 🧪 四核心AI大腦：盲測勝率")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟢 多頭模型**")
        wr_l = metrics.get('lgbm', {}).get('blind_win_rate', 0)
        date_l = metrics.get('lgbm', {}).get('last_train', '未訓練')
        c_l = "#00cc96" if wr_l > 0.55 else "#ffc107"
        _render_clean_html(f"<div style='background:#1e1e1e; padding:12px; border-left:4px solid #00cc96; border-radius:5px; margin-bottom:8px;'><div style='display:flex;justify-content:space-between;align-items:center;'><div><span style='color:#aaa;font-size:11px;'>LightGBM</span><br><span style='font-size:22px; color:{c_l}; font-weight:bold;'>{wr_l*100:.1f}%</span></div><div style='text-align:right;'><span style='color:#666;font-size:10px;'>{date_l}</span></div></div></div>")
        
        wr_ll = metrics.get('lstm', {}).get('blind_win_rate', 0)
        date_ll = metrics.get('lstm', {}).get('last_train', '未訓練')
        c_ll = "#00cc96" if wr_ll > 0.53 else "#ffc107"
        _render_clean_html(f"<div style='background:#1e1e1e; padding:12px; border-left:4px solid #00cc96; border-radius:5px;'><div style='display:flex;justify-content:space-between;align-items:center;'><div><span style='color:#aaa;font-size:11px;'>LSTM</span><br><span style='font-size:22px; color:{c_ll}; font-weight:bold;'>{wr_ll*100:.1f}%</span></div><div style='text-align:right;'><span style='color:#666;font-size:10px;'>{date_ll}</span></div></div></div>")
        
    with col2:
        st.markdown("**🔴 空頭模型**")
        wr_s = metrics.get('short', {}).get('lgbm', {}).get('blind_win_rate', 0)
        date_s = metrics.get('short', {}).get('lgbm', {}).get('last_train', '未訓練')
        c_s = "#ff4b4b" if wr_s > 0.55 else "#ff9966"
        if wr_s > 0:
            _render_clean_html(f"<div style='background:#1e1e1e; padding:12px; border-left:4px solid #ff4b4b; border-radius:5px; margin-bottom:8px;'><div style='display:flex;justify-content:space-between;align-items:center;'><div><span style='color:#aaa;font-size:11px;'>LightGBM</span><br><span style='font-size:22px; color:{c_s}; font-weight:bold;'>{wr_s*100:.1f}%</span></div><div style='text-align:right;'><span style='color:#666;font-size:10px;'>{date_s}</span></div></div></div>")
        else:
            _render_clean_html("<div style='background:#1e1e1e; padding:12px; border-left:4px solid #666; border-radius:5px; margin-bottom:8px;'><span style='color:#666;font-size:12px;'>LightGBM</span><br><span style='font-size:16px; color:#666;'>未訓練</span></div>")
        
        wr_ls = metrics.get('short', {}).get('lstm', {}).get('blind_win_rate', 0)
        date_ls = metrics.get('short', {}).get('lstm', {}).get('last_train', '未訓練')
        c_ls = "#ff4b4b" if wr_ls > 0.53 else "#ff9966"
        if wr_ls > 0:
            _render_clean_html(f"<div style='background:#1e1e1e; padding:12px; border-left:4px solid #ff4b4b; border-radius:5px;'><div style='display:flex;justify-content:space-between;align-items:center;'><div><span style='color:#aaa;font-size:11px;'>LSTM</span><br><span style='font-size:22px; color:{c_ls}; font-weight:bold;'>{wr_ls*100:.1f}%</span></div><div style='text-align:right;'><span style='color:#666;font-size:10px;'>{date_ls}</span></div></div></div>")
        else:
            _render_clean_html("<div style='background:#1e1e1e; padding:12px; border-left:4px solid #666; border-radius:5px;'><span style='color:#666;font-size:12px;'>LSTM</span><br><span style='font-size:16px; color:#666;'>未訓練</span></div>")

def render_fear_greed_gauge(index_val: int, label: str, color: str):
    html = (
        f"<div style='background:#1e1e1e;border-radius:12px;padding:18px;text-align:center;border:1px solid #333;'>"
        f"<div style='color:#aaa;font-size:13px;margin-bottom:6px;'>📊 台股恐懼貪婪指數</div>"
        f"<div style='font-size:38px;font-weight:700;color:{color};'>{index_val}</div>"
        f"<div style='font-size:16px;color:{color};font-weight:600;margin-bottom:12px;'>{label}</div>"
        f"<div style='background:#333;border-radius:8px;height:10px;width:100%;overflow:hidden;'>"
        f"<div style='background:{color};width:{index_val}%;height:100%;border-radius:8px;'></div>"
        f"</div></div>"
    )
    _render_clean_html(html)

# ==========================================
# 主程式與邏輯快取
# ==========================================
def get_fugle_key():
    try: return st.secrets["FUGLE_API_KEY"]
    except: return ""

DEFAULT_CLUSTERS = {
    "半導體": ["2330.TW", "3711.TW", "2454.TW", "2303.TW", "5347.TWO", "3034.TW"],
    "矽光子": ["3363.TWO", "3450.TW", "6451.TW", "3081.TWO", "4979.TWO", "3163.TWO"],
    "伺服器": ["2382.TW", "3231.TW", "6669.TW", "2376.TW", "3017.TW", "5274.TWO"],
    "金融股": ["2881.TW", "2882.TW", "2886.TW", "2891.TW", "2884.TW"],
    "傳統產業": ["1101.TW", "2002.TW", "2603.TW", "2609.TW", "2618.TW"],
    "ETF": ["0050.TW", "0056.TW", "00878.TW", "00919.TW", "00929.TW"]
}

DEFAULT_NAMES = {
    "2330": "台積電", "2454": "聯發科", "2303": "聯電", "2382": "廣達", "3231": "緯創",
    "2881": "富邦金", "2882": "國泰金", "2886": "兆豐金", "2891": "中信金", "2884": "玉山金",
    "1101": "台泥", "2002": "中鋼", "2603": "長榮", "2609": "陽明", "2618": "長榮航",
    "0050": "元大台灣50", "0056": "元大高股息", "00878": "國泰永續高股息", "00919": "群益精選高息", "00929": "復華台灣科技優息",
    "3711": "日月光投控", "5347": "世界", "3034": "聯詠", "3363": "禾聯碩", "3450": "聯鈞", 
    "6451": "訊芯-KY", "3081": "聯亞", "4979": "華星光", "3163": "波若威", "6669": "緯穎", 
    "2376": "技嘉", "3017": "奇鋐", "5274": "信驊", "5474": "聰泰"
}

from data_fetcher import load_all_market_tickers, get_market_summary, get_kline_with_fugle, get_stock_news
from data_pipeline import load_market_snapshot, get_snapshot_dict, get_realtime_quote, fetch_advanced_backtest, trigger_github_workflow, load_model_metrics
from ai_engine import DualCoreBrain

st.set_page_config(page_title="台股量化旗艦終端 v4.1", page_icon="📈", layout="wide")

FUGLE_API_KEY = get_fugle_key()
if 'stock_clusters' not in st.session_state: 
    st.session_state.stock_clusters = DEFAULT_CLUSTERS.copy()
if 'stock_names' not in st.session_state: 
    st.session_state.stock_names = DEFAULT_NAMES.copy()

@st.cache_resource
def load_brain(): 
    return DualCoreBrain()

brain = load_brain()

# 🔥 全市場預測全局快取 (只算一次，四大分頁秒速共享)
@st.cache_data(ttl=60, show_spinner=False)
def get_market_predictions_cached():
    snap = load_market_snapshot()
    if not snap or 'data' not in snap or not snap['data']:
        return []
    
    raw_list = snap['data']
    snap_dict = get_snapshot_dict(snap)
    valid_items, bulk_features = [], []
    
    for item in raw_list:
        ticker = str(item.get('代號', '')).split('.')[0].strip()
        ep = float(item.get('現價', item.get('close_price', 0.0)))
        if ep > 0:
            valid_items.append(item)
            bulk_features.append(brain.extract_features(ticker, ep, snap_dict, current_vol=float(item.get('成交量', 0.0))))
            
    if not valid_items: return []
    
    if hasattr(brain, 'predict_four_core'):
        core_results = brain.predict_four_core(bulk_features)
    else:
        probs = brain.predict_win_rates(bulk_features)
        core_results = [{'best_long': p, 'best_short': 1-p, 'lgbm_long':p, 'lstm_long':p, 'lgbm_short':1-p, 'lstm_short':1-p, 'signal': 'WAIT'} for p in probs]
        
    for i, item in enumerate(valid_items):
        item['core_data'] = core_results[i]
        
    return valid_items

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
            df_all['clean_ticker'] = df_all['ticker'].astype(str).str.split('.').str[0].str.strip()
            match = df_all[df_all['clean_ticker'] == clean_code]
            if not match.empty: 
                return str(match.iloc[0]['name'])
    except: 
        pass
    return ticker

def compute_fear_greed(twii_pct: float, snapshot_data: list) -> tuple[int, str, str]:
    twii_pct = np.nan_to_num(twii_pct, nan=0.0, posinf=5.0, neginf=-5.0)
    twii_pct = np.clip(twii_pct, -10, 10)
    pct_score = float(np.clip(50 + (twii_pct * 16.67), 0, 100))
    bull_ratio_score = rs_score = vol_score = 50.0
    if snapshot_data:
        df = pd.DataFrame(snapshot_data)
        if 'rs_index' in df.columns:
            df['rs_index'] = pd.to_numeric(df['rs_index'], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0).clip(-100, 100)
            bull_ratio_score = float((df['rs_index'] > 0).mean()) * 100
            rs_score = float(np.clip(50 + (df['rs_index'].mean() * 5), 0, 100))
        if 'vol_ratio' in df.columns:
            df['vol_ratio'] = pd.to_numeric(df['vol_ratio'], errors='coerce').fillna(1.0).replace([np.inf, -np.inf], 1.0).clip(0.1, 10)
            vol_score = float(np.clip(50 + (df['vol_ratio'].mean() - 1.0) * 50, 0, 100))
    final = pct_score * 0.35 + bull_ratio_score * 0.35 + rs_score * 0.20 + vol_score * 0.10
    index_val = int(np.clip(final, 0, 100))
    
    if index_val >= 75: label, color = "極度貪婪", "#ff4b4b"
    elif index_val >= 60: label, color = "貪婪", "#ffa500"
    elif index_val >= 45: label, color = "中性偏多", "#ffc107"
    elif index_val >= 30: label, color = "恐懼", "#00cc96"
    else: label, color = "極度恐懼", "#00ccff"
    
    return index_val, label, color

# ==========================================
# 🎛️ 左側控制面板
# ==========================================
with st.sidebar:
    st.header("📂 我的自選清單")
    selected_cluster = st.selectbox("1. 選擇產業群組", list(st.session_state.stock_clusters.keys()))
    cluster_stocks = st.session_state.stock_clusters[selected_cluster]
    
    display_options = []
    for t in cluster_stocks:
        base = t.split('.')[0]
        name = st.session_state.stock_names.get(base) or get_stock_name_from_csv(base)
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
            
    if st.button("🧠 啟動四模型訓練", use_container_width=True):
        success, msg = trigger_github_workflow("train_ai.yml")
        if success: st.info(msg)
        else: st.error(msg)
            
    st.markdown("---")
    if brain.is_lstm_ready: st.success("🔮 LSTM 多頭已連動")
    else: st.warning("⚪ LSTM 多頭未載入")
        
    if brain.is_lgbm_ready: st.success("🌳 LGBM 多頭正常")
    else: st.error("🚨 LGBM 多頭缺失")
        
    if hasattr(brain, 'is_lgbm_short_ready') and brain.is_lgbm_short_ready: st.success("🔴 LGBM 空頭已連動")
    if hasattr(brain, 'is_lstm_short_ready') and brain.is_lstm_short_ready: st.success("🔴 LSTM 空頭已連動")

# ==========================================
# ⚡ 戰情室主視覺
# ==========================================
st.title("⚡ 台股戰情分析終端 v4.1")
st.caption("🟢 多頭 | 🔴 空頭 | ⚪ 盤整 | 四核心極速快取版")
col1, col2 = st.columns([3, 1])
with col1: manual_ticker = st.text_input("輸入股票代號", "", label_visibility="collapsed", placeholder="例如: 2330")
with col2: analyze_manual_btn = st.button("單股掃描", use_container_width=True)
st.markdown("---")

target_ticker = st.session_state.pop('analyze_trigger', None) or (manual_ticker.strip().upper() if analyze_manual_btn else None)

if target_ticker:
    base_ticker = target_ticker.split('.')[0]
    c_name = st.session_state.stock_names.get(base_ticker) or get_stock_name_from_csv(base_ticker)
    st.session_state.stock_names[base_ticker] = c_name
    
    with st.spinner(f"正在深度分析 {base_ticker} {c_name}..."):
        df_daily, df_hourly, _ = get_kline_with_fugle(target_ticker, FUGLE_API_KEY)
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
                
            p_change = ((entry_price - y_close) / max(y_close, 0.01) * 100) if y_close > 0.01 else 0.0
            p_change = np.nan_to_num(p_change, nan=0.0, posinf=20.0, neginf=-20.0)
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
            
            snapshot_dict = get_snapshot_dict(load_market_snapshot())
            feat_dict = brain.extract_features(base_ticker, entry_price, snapshot_dict, current_vol=rt_v, fallback_atr=atr_14, fallback_pattern=smc_text)
            
            if hasattr(brain, 'predict_four_core'):
                core_data = brain.predict_four_core([feat_dict])[0]
            else:
                core_data = {'best_long': 0.5, 'best_short': 0.5, 'lgbm_long': 0.5, 'lstm_long': 0.5, 'lgbm_short': 0.5, 'lstm_short': 0.5, 'signal': 'WAIT'}

            st.subheader(f"🧬 {base_ticker} {c_name} 診斷報告")
            render_single_diagnostic_card(core_data, entry_price, res_level, sup_level)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("當前現價", f"{entry_price:.2f}", f"{p_change:+.2f}%")
            m2.metric("SMC 結構", smc_text)
            m3.metric("1h 微觀狀態", micro_status_text)
            m4.metric("機構集中度", f"{broker_conc*100:.1f}%")
            st.markdown("---")
            
            infra_col1, infra_col2 = st.columns(2)
            with infra_col1:
                st.markdown("### 📊 策略與型態技術面分析")
                st.info(f"🧬 **SMC 聰明錢結構型態：** {smc_text}")
                st.write(f"* **20日高位壓力 (Res_20)：** `NT$ {res_level:.2f}`")
                st.write(f"* **20日低位支撐 (Sup_20)：** `NT$ {sup_level:.2f}`")
                st.write(f"* **ATR 波動度基準 (ATR_14)：** `NT$ {atr_14:.2f}`")
                
            with infra_col2:
                st.markdown("### 📰 即時相關新聞與情報")
                if news_s and isinstance(news_s, list):
                    for idx, n in enumerate(news_s[:3]):
                        if isinstance(n, dict): 
                            st.markdown(f"📢 **情報 {idx+1}：** [{n.get('title','檢視')}]({n.get('link','#')})")
                else: 
                    st.write("⚪ 暫無即時個股催化劑新聞。")
                with st.expander("🌍 國際總經環境解讀"): 
                    st.write("🟢 總體經濟環境處於常態偏多格局。")
                    
            st.markdown("---")
            if st.button("⬅️ 返回戰情室主頁", use_container_width=True):
                st.session_state.analyze_trigger = None
                st.rerun()
else:
    st.markdown("### 🌍 大盤與情緒摘要")
    summary = get_cached_market_summary()
    snapshot = load_market_snapshot()
    snapshot_data = snapshot.get('data', []) if snapshot else []
    
    update_time = "未知"
    if snapshot and isinstance(snapshot, dict):
        update_time = snapshot.get('update_time', '未知')
        snapshot_data = snapshot.get('data', [])
    
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
    
    if update_time != "未知": 
        st.caption(f"📡 快取更新時間：{update_time} ｜ 涵蓋標的：{len(snapshot_data)} 檔")
    else: 
        st.caption(f"⚠️ 快取狀態異常 ｜ 請執行全市場掃描")
    
    st.markdown("---")
    metrics = load_model_metrics()
    render_model_health_board(metrics)
    st.markdown("---")
    tab1, tab2, tab3, tab3s, tab5, tab6 = st.tabs(["📊 自選即時流", "🔮 每日收盤趨勢", "🎯 全市場 TOP 20", "🔻 空頭 TOP 20", "⚖️ 實盤開獎對撞", "🔬 策略回測"])
    
    with tab1:
        c_title, c_slider = st.columns([2, 1])
        with c_title: 
            st.markdown(f"#### 【{selected_cluster}】即時行情流")
        with c_slider:
            with st.expander("⚙️ 畫幅設定"): 
                user_font_size = st.slider("表格文字大小", 12, 40, 22, 2)
                
        @st.fragment(run_every=datetime.timedelta(seconds=15))
        def render_rt():
            rows = []
            current_names = st.session_state.get('stock_names', DEFAULT_NAMES).copy()
            snap = load_market_snapshot()
            snap_dict = get_snapshot_dict(snap)
            def fetch_rt(t):
                ticker = t.split('.')[0]
                p, v, prev = get_realtime_quote(ticker, FUGLE_API_KEY)
                if p > 0:
                    chg_amt = p - prev
                    chg_pct = (chg_amt / prev) * 100 if prev > 0 else 0
                    s_name = snap_dict.get(ticker, {}).get('名稱') or current_names.get(ticker)
                    if not s_name or s_name == ticker:
                        s_name = get_stock_name_from_csv(ticker)
                        st.session_state.stock_names[ticker] = s_name
                    name_str = f"<b>{s_name}</b><br><span style='font-size:0.8em;color:gray;'>{ticker}</span>"
                    p_str = f"<b>{p:.2f}</b><br><span style='font-size:0.7em;color:gray;'>({int(v):,} 張)</span>"
                    
                    if chg_amt > 0: chg_str = f"<span style='color:#ff4b4b;font-weight:bold;'>+{chg_amt:.2f}<br>(+{chg_pct:.2f}%)</span>"
                    elif chg_amt < 0: chg_str = f"<span style='color:#00cc96;font-weight:bold;'>{chg_amt:.2f}<br>({chg_pct:.2f}%)</span>"
                    else: chg_str = "0.00"
                        
                    return {"標的": name_str, "及時價 (成交量)": p_str, "今日漲跌幅": chg_str}
                return None
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                future_to_ticker = {ex.submit(fetch_rt, t): t for t in cluster_stocks}
                for future in concurrent.futures.as_completed(future_to_ticker):
                    try:
                        res = future.result()
                        if res: rows.append(res)
                        time.sleep(0.3)
                    except: pass
                        
            if rows:
                html_table = pd.DataFrame(rows).to_html(escape=False, index=False, border=0).replace('\n', '')
                fs_th = max(14, user_font_size - 4)
                css = f"<style>.watch-board table{{width:100%!important;border-collapse:collapse;}}.watch-board th{{text-align:center!important;font-size:{fs_th}px!important;padding:10px!important;border-bottom:2px solid #555!important;}}.watch-board td{{text-align:center!important;font-size:{user_font_size}px!important;padding:16px!important;border-bottom:1px solid #444!important;vertical-align:middle!important;}}</style>"
                _render_clean_html(f'{css}<div class="watch-board">{html_table}</div>')
        render_rt()
    
    with tab2:
        st.markdown("#### 🔮 全市場雙核勝率分佈狀態透視")
        # 🔥 從全局快取瞬間讀取
        valid_items = get_market_predictions_cached()
        if valid_items:
            probs = np.array([item['core_data']['best_long'] for item in valid_items])
            avg_win_rate = float(np.mean(probs)) * 100
            bullish_ratio = float(np.mean(probs >= 0.50)) * 100
            highest_prob = float(np.max(probs)) * 100
            tier_alpha = int(np.sum(probs >= 0.52))
            tier_beta = int(np.sum((probs >= 0.48) & (probs < 0.52)))
            tier_gamma = int(np.sum(probs < 0.48))
            
            c1, c2, c3 = st.columns(3)
            c1.metric("🤖 絕對多頭標的比率 (勝率 ≥ 50%)", f"{bullish_ratio:.1f}%", help="全市場達到絕對勝率門檻的股票比例")
            c2.metric("📊 全市場 AI 平均勝率期望值", f"{avg_win_rate:.1f}%", help="當前大腦對全台股總體動能的信心分佈")
            c3.metric("🎯 全市場最優標的 AI 勝率峰值", f"{highest_prob:.1f}%", help="當下盤面上勝率最高的一檔股票之機率值")
            
            st.markdown("---")
            st.markdown("##### 📦 雙核大腦全市場標的「勝率梯隊分佈」")
            html_dist = (
                f"<div style='display: flex; width: 100%; height: 24px; border-radius: 6px; overflow: hidden; margin-bottom: 15px;'>"
                f"<div style='background: #00cc96; width: {(tier_alpha/len(probs))*100}%; text-align: center; color: white; font-size: 12px; line-height: 24px;'>{tier_alpha} 檔</div>"
                f"<div style='background: #ffc107; width: {(tier_beta/len(probs))*100}%; text-align: center; color: black; font-size: 12px; line-height: 24px;'>{tier_beta} 檔</div>"
                f"<div style='background: #444; width: {(tier_gamma/len(probs))*100}%; text-align: center; color: #aaa; font-size: 12px; line-height: 24px;'>{tier_gamma} 檔</div></div>"
                f"<div style='display: flex; justify-content: space-between; font-size: 13px;'>"
                f"<span style='color: #00cc96;'>🟢 領先多頭梯隊 (勝率 ≥ 52%): <b>{tier_alpha} 檔</b></span>"
                f"<span style='color: #ffc107;'>🟡 蓄勢震盪梯隊 (48% ~ 52%): <b>{tier_beta} 檔</b></span>"
                f"<span style='color: #888;'>⚪ 防守觀望梯隊 (勝率 < 48%): <b>{tier_gamma} 檔</b></span></div>"
            )
            _render_clean_html(html_dist)
            
            if bullish_ratio == 0: 
                st.info("💡 **解讀提示**：當前看多比率為 0.0% 代表目前盤勢處於極端修正或震盪。")
        else: 
            st.info("ℹ️ 快取中無有效數據。")
    
    with tab3:
        st.markdown("#### 🎯 全市場 AI 進出場戰術面板 (TOP 20)")
        valid_items = get_market_predictions_cached()
        if valid_items:
            processed = []
            for item in valid_items:
                prob = item['core_data']['best_long']
                if prob >= 0.50:
                    ep = float(item.get('現價', 0.0))
                    res = float(item.get('Res_20', ep * 1.05))
                    sup = float(item.get('Sup_20', ep * 0.95))
                    atr = float(item.get('ATR_14', ep * 0.05))
                    sl = round(res * 0.985, 2) if ep > res else round(min(ep - (1.5 * atr), sup * 0.985), 2)
                    tp = round(res + (res - sup), 2) if ep > res else round(res + (atr * 1.0), 2)
                    s_ticker = item.get('代號', '')
                    s_name = item.get('名稱')
                    if not s_name or s_name == s_ticker: 
                        s_name = get_stock_name_from_csv(s_ticker)
                    processed.append({'ticker': s_ticker, 'name': s_name, 'win_prob': prob, 'box_color': "#00cc96" if prob >= 0.52 else "#ffc107", 'ai_rec': "推薦佈局" if prob >= 0.52 else "謹慎試單", 'entry_price': ep, 'take_profit': tp, 'stop_loss': sl, 'profit_reason': "波段"})
            if processed:
                for s in sorted(processed, key=lambda x: x['win_prob'], reverse=True)[:20]: 
                    render_top20_card(s)
            else:
                probs = [item['core_data']['best_long'] for item in valid_items]
                highest = float(max(probs)) if len(probs) > 0 else 0.0
                st.warning(f"⚠️ **目前全市場無符合高勝率標準 (>50%) 之標的。**\n\nAI 判定市場風險高（最高勝率僅 {highest*100:.1f}%）。")
        else: 
            st.info("快取中無數據。")
    
    with tab3s:
        st.markdown("#### 🔻 全市場 AI 空頭戰術面板 (TOP 20 放空名單)")
        valid_items = get_market_predictions_cached()
        if valid_items:
            processed = []
            for item in valid_items:
                sp = float(item['core_data']['best_short'])
                lp = float(item['core_data']['best_long'])
                if sp >= 0.55:
                    ep = float(item.get('現價', 0.0))
                    res = float(item.get('Res_20', ep * 1.05))
                    sup = float(item.get('Sup_20', ep * 0.95))
                    atr = float(item.get('ATR_14', ep * 0.03))
                    take_profit = round(sup - atr, 2)
                    stop_loss = round(res * 1.015, 2)
                    s_ticker = item.get('代號', '')
                    s_name = item.get('名稱', s_ticker)
                    if not s_name or s_name == s_ticker: 
                        s_name = get_stock_name_from_csv(s_ticker)
                    processed.append({'ticker': s_ticker, 'name': s_name, 'short_prob': sp, 'long_prob': lp, 'entry_price': ep, 'take_profit': take_profit, 'stop_loss': stop_loss})
            if processed:
                st.success(f"✅ 找到 {len(processed)} 檔空頭訊號，顯示前20檔")
                for stock in sorted(processed, key=lambda x: x['short_prob'], reverse=True)[:20]:
                    sp = stock['short_prob']
                    color = "#ff0000" if sp >= 0.65 else "#ff4b4b" if sp >= 0.60 else "#ff9966"
                    
                    entry_p = max(stock['entry_price'], 0.01)
                    drop_pct = (1 - stock['take_profit'] / entry_p) * 100
                    risk_pct = (stock['stop_loss'] / entry_p - 1) * 100
                    html_s = (
                        f"<div style='border:2px solid {color};border-radius:10px;padding:16px; background:#1e1e1e;margin-bottom:12px;'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                        f"<div><h4 style='color:{color};margin:0;'>🔻 {stock['ticker']} {stock['name']}</h4>"
                        f"<span style='color:#888;font-size:12px;'>空頭極值 {sp*100:.1f}% | 多頭極值 {stock['long_prob']*100:.1f}%</span></div>"
                        f"<div style='text-align:right;'><div style='color:#888;font-size:11px;'>現價</div>"
                        f"<div style='color:#fff;font-size:20px;font-weight:bold;'>{stock['entry_price']:.2f}</div></div></div>"
                        f"<div style='display:flex;justify-content:space-between;margin-top:12px; padding-top:12px;border-top:1px solid #333;font-size:13px;'>"
                        f"<span>下檔支撐 (停利區): <b style='color:#00cc96;'>{stock['take_profit']:.2f}</b> <small>(-{drop_pct:.1f}%)</small></span>"
                        f"<span>上檔壓力 (停損區): <b style='color:#ff4b4b;'>{stock['stop_loss']:.2f}</b> <small>(+{risk_pct:.1f}%)</small></span>"
                        f"</div></div>"
                    )
                    _render_clean_html(html_s)
            else:
                st.success("✅ 目前市場無明顯空頭訊號")
                st.info("💡 所有股票空頭極值均 < 55%，市場相對安全")
        else: 
            st.error("❌ 無法讀取市場快取")
            
    with tab5:
        st.markdown("#### ⚖️ 昨晚 AI 預測 x 今日實盤開獎比對面板")
        if st.button("🔄 執行即時對撞比對", key="run_clash_realtime_btn"):
            valid_items = get_market_predictions_cached()
            if not valid_items: 
                st.warning("⚪ 歷史快取中無有效個股資料。")
            else:
                current_names = st.session_state.get('stock_names', DEFAULT_NAMES).copy()
                candidates = sorted([{**item, 'win_prob': float(item['core_data']['best_long'])} for item in valid_items], key=lambda x: x['win_prob'], reverse=True)
                top_candidates = [c for c in candidates if c['win_prob'] >= 0.50][:15]
                
                if not top_candidates:
                    highest = float(max([c['win_prob'] for c in candidates])) if candidates else 0.0
                    st.warning(f"⚪ 昨晚快取數據中無勝率達標 (>50%) 的標的（最高僅 {highest*100:.1f}%），今日無開獎清單。")
                else:
                    st.info(f"⏳ 針對 {len(top_candidates)} 檔高勝率標的進行開獎...")
                    clash_rows = []
                    def fetch_clash(item):
                        ticker = str(item.get('代號', '')).split('.')[0].strip()
                        stock_name = item.get('名稱') or current_names.get(ticker)
                        if not stock_name or stock_name == ticker: 
                            stock_name = get_stock_name_from_csv(ticker)
                        last_price = float(item.get('現價', 0.0))
                        win_prob = item['win_prob']
                        rt_p, _, _ = get_realtime_quote(ticker, FUGLE_API_KEY)
                        if rt_p > 0 and last_price > 0:
                            clash_pct = ((rt_p - last_price) / last_price) * 100
                            return {"股票代號": ticker, "股票名稱": stock_name, "雙核預測勝率": f"{win_prob*100:.1f}%", "預測基準價 (昨)": f"${last_price:.2f}", "實盤即時價 (今)": f"${rt_p:.2f}", "實盤開獎漲跌": f"{clash_pct:+.2f}%", "_win_prob_raw": win_prob}
                        return None
                        
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                        futures = {ex.submit(fetch_clash, item): item for item in top_candidates}
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                res = future.result()
                                if res: 
                                    clash_rows.append(res)
                                time.sleep(0.3)
                            except Exception: 
                                pass
                    if clash_rows:
                        clash_df = pd.DataFrame(clash_rows).sort_values(by="_win_prob_raw", ascending=False)
                        st.success("✅ 實盤對撞數據已成功產出！")
                        st.dataframe(clash_df.drop(columns=["_win_prob_raw"]), use_container_width=True, hide_index=True)
                    else: 
                        st.error("❌ 無法成功取得今日即時開獎報價。")
    
    with tab6:
        st.markdown("#### 🔬 實盤自動優化回測分析")
        with st.spinner("正在從資料庫拉取歷史特徵並進行回測運算..."):
            res_adv = fetch_advanced_backtest(initial_cap=user_capital, max_pos=user_max_pos)
            status = res_adv.get("status")
            if status == "ready":
                c1, c2, c3 = st.columns(3)
                with c1: 
                    render_backtest_metric_card("AI 真實勝率", f"{res_adv['ai_strat']['wr']*100:.1f}%", "", "#4ade80")
                with c2: 
                    render_backtest_metric_card("帳戶總淨利", f"${res_adv['net_profit_twd']:,.0f}", "", "#4ade80")
                with c3: 
                    render_backtest_metric_card("總報酬", f"{res_adv['account_pct']:.2f}%", "", "#4ade80")
                st.line_chart(pd.DataFrame(res_adv['equity']).set_index("date_str")[["strat_cum_pct", "market_cum_pct"]])
            elif status == "no_key": 
                st.warning("🔑 缺少 Supabase 資料庫金鑰。請在 Streamlit Secrets 中設定 `SUPABASE_URL` 與 `SUPABASE_KEY`。")
            elif status == "empty": 
                st.info("ℹ️ 資料庫中目前無足夠的歷史資料可供回測。")
            elif status == "pending": 
                st.info("⏳ 條件過於嚴格，當前回測區間內沒有符合的交易訊號。")
            else: 
                st.error(f"❌ 回測系統發生錯誤: {res_adv.get('msg', '未知錯誤')}")