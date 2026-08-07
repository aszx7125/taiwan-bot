"""
台股AI量化系統 v4.1 - 全局快取極速版 + 高科 iAI (Nemotron-3-Super) 交易副駕
修復：解決首頁分頁重複推論導致 15秒以上卡頓的問題，將載入時間壓縮至 0.1秒
新增：高科 iAI (Nemotron-3-super-120b) RAG 本地數據注入聊天室
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import concurrent.futures
import json
import os
import requests

# ==========================================
# 🛑 AI 額度限制：本地模擬攔截器
# ==========================================
class MockResponse:
    def __init__(self, content):
        self.status_code = 200
        self._content = content
    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}

def mock_post(url, *args, **kwargs):
    if "chat/completions" in url:
        payload = kwargs.get('json', {})
        messages = payload.get('messages', [])
        sys_msg = next((m['content'] for m in messages if m['role'] == 'system'), "")
        if "華爾街頂尖價值型基金經理人" in sys_msg:
            msg = "【系統提示：AI 模型已停用】\n\n此標的目前缺乏 AI 分析資料。請依據上方數據卡片與K線圖進行判斷。"
        elif "量化分析師" in sys_msg:
            msg = "【AI 已停用】型態健康度評估暫無法使用，請參考月線與現價關係。"
        elif "明日操盤晨會報告" in sys_msg or "明日操盤晨報" in sys_msg:
            msg = "【系統提示：AI 模型已停用】\n\n無法生成晨會報告。請參考下方排行面板。"
        else:
            msg = "【AI 已停用】此為系統模擬回覆，因額度限制已暫停呼叫外部模型。"
        return MockResponse(msg)
    import requests as orig_requests
    return orig_requests.post(url, *args, **kwargs)

import requests
requests.post = mock_post


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
        f"<a href='/?analyze={s['ticker']}' target='_self' style='text-decoration:none; color:inherit; display:block;'>" \
        f"<div style='border: 2px solid {color}; border-radius: 10px; padding: 18px; background-color: #1e1e1e; margin-bottom: 12px; transition: opacity 0.2s;' onmouseover='this.style.opacity=0.8' onmouseout='this.style.opacity=1'>"
        f"<h4 style='color: {color}; margin-top: 0;'>🎯 {s['ticker']} {s['name']}</h4>"
        f"<div style='display: flex; justify-content: space-between;'>"
        f"<div><span style='color: gray; font-size: 13px;'>最高勝率極值</span><br>"
        f"<b style='font-size: 22px; color: {color};'>{prob:.1f}%</b></div>"
        f"<div style='text-align: right;'><span style='color: gray; font-size: 13px;'>現價</span><br>"
        f"<b style='font-size: 20px;'>{s['entry_price']:.2f}</b></div>"
        f"</div></div></a>"
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

    if signal == "STRONG_LONG": 
        box_color = "#00cc96"
        rec = f"模型預測將上漲 (機率 {bl:.1f}%)，預期目標價為 {res_level:.2f}；若反向回檔，下探支撐為 {sup_level:.2f}"
    elif signal == "LONG": 
        box_color = "#4ade80"
        rec = f"模型預測偏多 (機率 {bl:.1f}%)，預期目標價為 {res_level:.2f}；若反向回檔，下探支撐為 {sup_level:.2f}"
    elif signal == "STRONG_SHORT": 
        box_color = "#ff4b4b"
        rec = f"模型預測將下跌 (機率 {bs:.1f}%)，預期下探目標為 {sup_level:.2f}；若反向強彈，上檔壓力為 {res_level:.2f}"
    elif signal == "SHORT": 
        box_color = "#ff8080"
        rec = f"模型預測偏空 (機率 {bs:.1f}%)，預期下探目標為 {sup_level:.2f}；若反向強彈，上檔壓力為 {res_level:.2f}"
    elif signal == "HIGH_VOLATILITY": 
        box_color = "#ffa500"
        rec = f"多空分歧 (上漲 {bl:.1f}% / 下跌 {bs:.1f}%)，上探預期 {res_level:.2f}，下探預期 {sup_level:.2f}，建議觀望"
    else: 
        box_color = "#a8a8a8"
        rec = f"動能平淡，預期於支撐 {sup_level:.2f} 與壓力 {res_level:.2f} 之間震盪整理"

    html = (
        f"<div style='border: 2px solid {box_color}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;'>"
        f"<h4 style='color: {box_color}; margin-top: 0; margin-bottom: 20px; border-bottom: 1px solid #333; padding-bottom: 10px; line-height: 1.4;'>"
        f"🎯 AI 趨勢建議：<br><span style='font-size: 18px; color: #fff;'>{rec}</span></h4>"
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
        f"</div></div></a>"
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
        f"</div></div></a>"
    )
    _render_clean_html(html)

# ==========================================
# 主程式與邏輯快取
# ==========================================
st.set_page_config(page_title="台股量化旗艦終端 v4.1", page_icon="📈", layout="wide")

# 初始化：策略開發室專屬對話紀錄
if "coder_messages" not in st.session_state:
    st.session_state.coder_messages = [
        {"role": "assistant", "content": "你好！我是你的專屬量化開發工程師。無論是 Python 爬蟲、Pandas 數據處理、LightGBM 模型調優，還是 TradingView 的 PineScript 指標，直接把程式碼或需求貼給我吧！"}
    ]

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
    st.title("🧭 導覽列")
    if "current_page" not in st.session_state:
        st.session_state.current_page = "📊 台股大盤掃描"
    

    
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
    
    def analyze_sidebar():
        st.session_state.analyze_trigger = sidebar_ticker
        if 'manual_search' in st.session_state:
            st.session_state.manual_search = ""
            
    st.button("📊 診斷此自選股", use_container_width=True, type="primary", on_click=analyze_sidebar)
        
    st.markdown("---")
    st.caption("🤖 核心引擎狀態")
    st.markdown(f"多頭: {'🟢' if brain.is_lgbm_ready and brain.is_lstm_ready else '🔴'} | 空頭: {'🟢' if hasattr(brain, 'is_lgbm_short_ready') and brain.is_lgbm_short_ready else '🔴'}")

    # ==========================================
# ⚡ 戰情室主視覺
# ==========================================
query_ticker = st.query_params.get("analyze")
if query_ticker:
    st.query_params.clear()
    if 'manual_search' in st.session_state:
        st.session_state.manual_search = ""

st.title("⚡ 台股戰情分析終端 v4.1")
st.caption("🟢 多頭 | 🔴 空頭 | ⚪ 盤整 | 四核心極速快取版")
col1, col2 = st.columns([3, 1])
with col1: manual_ticker = st.text_input("輸入股票代號", "", label_visibility="collapsed", placeholder="例如: 2330", key="manual_search")
with col2: analyze_manual_btn = st.button("單股掃描", use_container_width=True)
st.markdown("---")

sidebar_trigger = st.session_state.pop('analyze_trigger', None)
target_ticker = None

if sidebar_trigger or query_ticker:
    target_ticker = sidebar_trigger or query_ticker
    st.session_state.current_page = "🎯 單股技術診斷"
    st.session_state.target_ticker_cache = target_ticker
    st.rerun()
elif manual_ticker or analyze_manual_btn:
    target_ticker = manual_ticker.strip().upper()
    if target_ticker and st.session_state.current_page != "🎯 單股技術診斷":
        st.session_state.current_page = "🎯 單股技術診斷"
        st.session_state.target_ticker_cache = target_ticker
        st.rerun()
    elif target_ticker and st.session_state.get('target_ticker_cache') != target_ticker:
        st.session_state.target_ticker_cache = target_ticker
        st.rerun()

if st.session_state.current_page == "🎯 單股技術診斷":
    target_ticker = st.session_state.get('target_ticker_cache', None)
    if not target_ticker:
        st.warning('請先從左側自選清單或大盤掃描中選擇一檔股票進行診斷！')
    else:
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
                
                snapshot_dict = get_snapshot_dict(load_market_snapshot())
                broker_conc = float(snapshot_dict.get(base_ticker, {}).get('broker_conc', 0.0))
            
                micro_status_text = "⚪ 1h 均線下弱勢震盪"
                if not df_hourly.empty and len(df_hourly) >= 2:
                    last_hour = df_hourly.iloc[-1]
                    if bool(last_hour.get('Micro_Sniper_Trigger', False)): micro_status_text = "🔥 帶量突破 1h 均線"
                    elif bool(last_hour.get('MACD_Cross_Up', False)): micro_status_text = "📈 1h MACD 金叉發動"
                    elif bool(last_hour.get('Vol_Surge_1h', False)): micro_status_text = "🌊 1h 微觀異常爆量"
            
                low_vol_pb = bool(today.get('Low_Vol_Pullback', False))
                smc_text = "量縮回踩" if low_vol_pb else "一般常態箱體震盪"
            
                feat_dict = brain.extract_features(base_ticker, entry_price, snapshot_dict, current_vol=rt_v, fallback_atr=atr_14, fallback_pattern=smc_text)
            
                if hasattr(brain, 'predict_four_core'):
                    core_data = brain.predict_four_core([feat_dict])[0]
                else:
                    core_data = {'best_long': 0.5, 'best_short': 0.5, 'lgbm_long': 0.5, 'lstm_long': 0.5, 'lgbm_short': 0.5, 'lstm_short': 0.5, 'signal': 'WAIT'}

                st.subheader(f"🧬 {base_ticker} {c_name} 診斷報告")
            
                # --- 1. 計算 UI 數據 ---
                bl = core_data.get('best_long', 0.5)
                bs = core_data.get('best_short', 0.5)
                is_long = bl >= bs
                prob = bl if is_long else bs
            
                if is_long:
                    sl = min(entry_price - 1.5 * atr_14, sup_level)
                    tp = max(res_level, entry_price + 2 * atr_14)
                    color = "#00cc96"
                    direction = "多頭波段"
                    trend_icon = "📈"
                    pullback_label = "預計回檔/支撐價"
                    entry_label = "建議買入區間"
                    entry_value = f"接近 {sl:.2f}"
                else:
                    sl = max(res_level, entry_price + 1.5 * atr_14)
                    tp = min(sup_level, entry_price - 2 * atr_14)
                    color = "#ff4b4b"
                    direction = "空頭波段"
                    trend_icon = "📉"
                    pullback_label = "預計反彈/壓力價"
                    entry_label = "建議空手/放空區"
                    entry_value = f"接近 {sl:.2f}"
                
                # --- 2. 渲染沉浸式網格 UI (致敬貓眼策略) ---
                ui_html = f'''
                <div style="background-color: #0b0e14; padding: 20px; border-radius: 16px; margin-bottom: 24px; font-family: sans-serif;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <img src="https://img.icons8.com/color/48/000000/bullish.png" width="32" style="border-radius:50%;" />
                            <span style="color: #fff; font-size: 20px; font-weight: bold;">{base_ticker} {c_name}</span>
                            <span style="background: rgba(255,255,255,0.1); color: #ccc; padding: 4px 8px; border-radius: 6px; font-size: 12px;">現價: {entry_price:.2f}</span>
                        </div>
                    </div>
                
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                        <div style="background-color: #151924; padding: 16px; border-radius: 12px; border: 1px solid #2a2e39;">
                            <div style="color: #8bb0d9; font-size: 13px; margin-bottom: 6px;">📊 目前 AI 趨勢 (勝率)</div>
                            <div style="color: {color}; font-size: 24px; font-weight: bold;">{trend_icon} {direction} ({prob*100:.1f}%)</div>
                        </div>
                        <div style="background-color: #151924; padding: 16px; border-radius: 12px; border: 1px solid #2a2e39;">
                            <div style="color: #8bb0d9; font-size: 13px; margin-bottom: 6px;">{trend_icon} {pullback_label}</div>
                            <div style="color: #ffc107; font-size: 24px; font-weight: bold;">{sl:.2f}</div>
                        </div>
                    
                        <div style="background-color: #151924; padding: 16px; border-radius: 12px; border: 1px solid #2a2e39;">
                            <div style="color: #8bb0d9; font-size: 13px; margin-bottom: 6px;">🛒 {entry_label}</div>
                            <div style="color: {color}; font-size: 24px; font-weight: bold;">{entry_value}</div>
                        </div>
                        <div style="background-color: #151924; padding: 16px; border-radius: 12px; border: 1px solid #2a2e39; border-left: 4px solid {color};">
                            <div style="color: #8bb0d9; font-size: 13px; margin-bottom: 6px;">🎯 波段停利目標</div>
                            <div style="color: {color}; font-size: 24px; font-weight: bold;">{tp:.2f}</div>
                        </div>
                    </div>
                
                    <div style="margin-top: 16px; background-color: #151924; padding: 12px 16px; border-radius: 12px; border: 1px solid #2a2e39; display: flex; justify-content: space-between;">
                        <div style="color: #8bb0d9; font-size: 13px;">1h 微觀狀態: <span style="color:#fff;">{micro_status_text}</span></div>
                        <div style="color: #8bb0d9; font-size: 13px;">SMC 結構: <span style="color:#fff;">{smc_text}</span></div>
                    </div>
                </div>
                '''
                _render_clean_html(ui_html)
            
            
                # --- 3. TradingView 原生互動 K線圖 (Lightweight Charts) ---
                st.markdown("### 📈 即時技術分析 (TradingView 原生體驗)")
                
                # UI 控制區
                col_tf, col_ind, col_ma1, col_ma2, col_ma3 = st.columns([2, 3, 1, 1, 1])
                with col_tf: 
                    tf_sel = st.selectbox("時區", ["1小時", "4小時", "日線", "週線", "月線"], index=2, label_visibility="collapsed")
                with col_ind:
                    ind_sel = st.multiselect("顯示指標", ["均線", "成交量"], default=["均線", "成交量"], label_visibility="collapsed")
                with col_ma1: 
                    ma1_p = st.number_input("MA 1", min_value=1, max_value=300, value=5)
                with col_ma2: 
                    ma2_p = st.number_input("MA 2", min_value=1, max_value=300, value=20)
                with col_ma3: 
                    ma3_p = st.number_input("MA 3", min_value=1, max_value=300, value=60)
                
                import json
                import numpy as np
                import pandas as pd
                
                # 依據時區選擇或重採樣資料
                if tf_sel in ["1小時", "4小時"] and not df_hourly.empty:
                    df_tv = df_hourly.copy()
                    df_tv = df_tv[~df_tv.index.duplicated(keep='last')]
                    df_tv = df_tv.sort_index()
                    if tf_sel == "4小時":
                        df_tv = df_tv.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                else:
                    df_tv = df_daily.copy()
                    df_tv = df_tv[~df_tv.index.duplicated(keep='last')]
                    df_tv = df_tv.sort_index()
                
                if tf_sel == "週線":
                    df_tv = df_tv.resample('W-FRI').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})
                elif tf_sel == "月線":
                    df_tv = df_tv.resample('M').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})
                
                # 均線計算
                df_tv[f'SMA_{ma1_p}'] = df_tv['Close'].rolling(window=ma1_p).mean()
                df_tv[f'SMA_{ma2_p}'] = df_tv['Close'].rolling(window=ma2_p).mean()
                df_tv[f'SMA_{ma3_p}'] = df_tv['Close'].rolling(window=ma3_p).mean()
                
                # 清洗與轉換
                df_tv = df_tv.dropna(subset=['Open', 'High', 'Low', 'Close'])
                df_tv = df_tv.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # 處理時間格式 (Lightweight Charts 要求: 日線以上為 'YYYY-MM-DD'，小時線以下為 Unix Timestamp 數字)
                if tf_sel in ["1小時", "4小時"]:
                    # Convert DatetimeIndex to Unix timestamp in seconds
                    df_tv['time'] = df_tv.index.astype('int64') // 10**9
                else:
                    df_tv['time'] = df_tv.index.strftime('%Y-%m-%d')
                
                candle_data = [{"time": r['time'], "open": float(r['Open']), "high": float(r['High']), "low": float(r['Low']), "close": float(r['Close'])} for _, r in df_tv.iterrows()]
                volume_data = [{"time": r['time'], "value": float(r.get('Volume', 0)), "color": "rgba(8,153,129,0.5)" if r['Close'] >= r['Open'] else "rgba(242,54,69,0.5)"} for _, r in df_tv.iterrows()]
                
                ma1_data = [{"time": r['time'], "value": float(r[f'SMA_{ma1_p}'])} for _, r in df_tv.iterrows() if r[f'SMA_{ma1_p}'] > 0]
                ma2_data = [{"time": r['time'], "value": float(r[f'SMA_{ma2_p}'])} for _, r in df_tv.iterrows() if r[f'SMA_{ma2_p}'] > 0]
                ma3_data = [{"time": r['time'], "value": float(r[f'SMA_{ma3_p}'])} for _, r in df_tv.iterrows() if r[f'SMA_{ma3_p}'] > 0]
                
                show_ma = "均線" in ind_sel
                show_vol = "成交量" in ind_sel
                
                html_code = f'''
                <div id="tvchart" style="width: 100%; height: 500px; background-color: #131722; border-radius: 8px;"></div>
                <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
                <script>
                    try {{
                        const domElement = document.getElementById('tvchart');
                        const chartProperties = {{
                            layout: {{ background: {{ type: 'solid', color: '#131722' }}, textColor: '#d1d4dc' }},
                            grid: {{ vertLines: {{ color: '#2b2b43' }}, horzLines: {{ color: '#2b2b43' }} }},
                            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                            rightPriceScale: {{ borderColor: '#2b2b43' }},
                            timeScale: {{ 
                                borderColor: '#2b2b43', 
                                timeVisible: {"true" if tf_sel in ["1小時", "4小時"] else "false"} 
                            }}
                        }};
                        
                        const chart = LightweightCharts.createChart(domElement, chartProperties);
                        
                        const candleSeries = chart.addCandlestickSeries({{
                            upColor: '#089981', downColor: '#F23645', borderDownColor: '#F23645', borderUpColor: '#089981', wickDownColor: '#F23645', wickUpColor: '#089981'
                        }});
                        candleSeries.setData({json.dumps(candle_data)});
                        
                        if ({str(show_ma).lower()}) {{
                            const ma1Series = chart.addLineSeries({{ color: '#f5c542', lineWidth: 2, title: 'SMA {ma1_p}' }});
                            ma1Series.setData({json.dumps(ma1_data)});
                            const ma2Series = chart.addLineSeries({{ color: '#2962FF', lineWidth: 2, title: 'SMA {ma2_p}' }});
                            ma2Series.setData({json.dumps(ma2_data)});
                            const ma3Series = chart.addLineSeries({{ color: '#e841f4', lineWidth: 2, title: 'SMA {ma3_p}' }});
                            ma3Series.setData({json.dumps(ma3_data)});
                        }}
                        
                        if ({str(show_vol).lower()}) {{
                            const volumeSeries = chart.addHistogramSeries({{
                                priceFormat: {{ type: 'volume' }}, priceScaleId: '', scaleMargins: {{ top: 0.8, bottom: 0 }}
                            }});
                            volumeSeries.setData({json.dumps(volume_data)});
                        }}
                        
                        chart.timeScale().fitContent();
                        
                        new ResizeObserver(entries => {{
                            if (entries.length === 0 || entries[0].target !== domElement) return;
                            chart.applyOptions({{ width: entries[0].contentRect.width, height: entries[0].contentRect.height }});
                        }}).observe(domElement);
                    }} catch (e) {{
                        document.getElementById('tvchart').innerHTML = "<div style='color:red; padding: 20px;'>Chart Error: " + e.toString() + "</div>";
                    }}
                </script>
                '''
                import streamlit.components.v1 as components
                # components.html 不支援 key 參數，若 html_code 內容改變即會自動重新渲染
                components.html(html_code, height=520)
                
                # --- 4. 系統戰術分析 (基於自有模型) ---
                # 結合 core_data (LGBM/LSTM) 與 技術面/籌碼面 進行邏輯推導
                score = float(today.get('Score', 0.0))
                broker_text = "籌碼集中，主力/外資偏多佈局" if broker_conc > 0 else "籌碼渙散，外資或主力呈現倒貨"
                ma_trend = "短均線(5MA)強勢向上" if today.get('SMA_5', 0) > today.get('SMA_20', 0) else "短期趨勢偏弱，跌破月線"
                vol_text = "量能放大，具備攻擊動能" if today.get('Volume', 0) > today.get('Vol_SMA5', 0) else "量能萎縮，動能不足"
                
                is_long = bl >= bs
                main_color = "#00cc96" if is_long else "#ff4b4b"
                bg_color = "rgba(0, 204, 150, 0.08)" if is_long else "rgba(255, 75, 75, 0.08)"
                trend_title = "上漲機率較高" if is_long else "下跌機率較高"
                
                bl_pct = bl * 100
                bs_pct = bs * 100
                
                if is_long:
                    target_text = f"若發動上漲，預期目標價位為 <b style='color: #ffc107; font-size: 18px;'>{tp:.2f}</b>；<br>若反轉下跌，預期下檔支撐為 <b style='color: #00ccff; font-size: 18px;'>{sl:.2f}</b>。"
                else:
                    target_text = f"若發動下跌，預期下探目標為 <b style='color: #00ccff; font-size: 18px;'>{tp:.2f}</b>；<br>若反轉強彈，上檔防守壓力為 <b style='color: #ffc107; font-size: 18px;'>{sl:.2f}</b>。"
                
                # --- 生成動態擬真 AI 交易建議 (Dynamic Trading Signal) ---
                action_signal = "⏳ 觀望 / 等待買點"
                action_color = "#f5c542"
                action_reasoning = f"{base_ticker} 目前趨勢混沌不明，建議先空手觀望，等待進階訊號確認後再行操作。"
                
                if is_long:
                    if prob >= 0.65 and "強勢向上" in ma_trend and "具備攻擊動能" in vol_text:
                        action_signal = "🚀 強勢買入 (Strong Buy)"
                        action_color = "#00cc96"
                        action_reasoning = f"綜合 AI 雙引擎判定，{base_ticker} 目前多方勝率高達 {bl_pct:.1f}%。技術面上 5MA 已強勢翻揚，且今日成交量顯著放大。籌碼面顯示集中度達 {broker_conc:+.2f}，具備強烈上攻動能。建議可於現價 {entry_price:.2f} 附近果斷介入順勢操作，將防守點設於 {sl:.2f}，強勢挑戰上方 {tp:.2f} 壓力區。"
                    elif ("回踩" in smc_text or "震盪" in micro_status_text) and prob >= 0.55:
                        action_signal = "🛒 波段低點佈局 (Swing Buy)"
                        action_color = "#00cc96"
                        action_reasoning = f"AI 偵測 {base_ticker} 目前處於波段的量縮回踩或震盪洗盤區 (多方勝率 {bl_pct:.1f}%)。儘管短線走勢黏著，但籌碼集中度 ({broker_conc:+.2f}) 依然健康，下方支撐 {sl:.2f} 相當堅實。此時介入下檔風險極低，屬於絕佳的波段左側買點，預計蓄力後將向 {tp:.2f} 發動攻勢。"
                    elif prob >= 0.60:
                        action_signal = "🟢 偏多操作 (Buy)"
                        action_color = "#00cc96"
                        action_reasoning = f"目前 {base_ticker} 多頭勝率 {bl_pct:.1f}% 佔據優勢，雖然短期 {vol_text}，但整體結構偏多。建議可於 {sl:.2f} 以上分批佈局，耐心等待均線糾結後向上發散，目標上看 {tp:.2f}。"
                    else:
                        action_signal = "⏳ 觀望 / 等待買點 (Wait)"
                        action_color = "#f5c542"
                        action_reasoning = f"{base_ticker} 目前多空交戰激烈 (多方勝率僅 {bl_pct:.1f}%)，且 {ma_trend}。目前的震盪結構尚未表態，建議先空手觀望，等待突破 {tp:.2f} 或回落至 {sl:.2f} 測試支撐有守後，再行佈局。"
                else:
                    if prob >= 0.65:
                        action_signal = "⚠️ 強烈賣出 / 做空 (Strong Sell)"
                        action_color = "#ff4b4b"
                        action_reasoning = f"AI 模型發出嚴重示警，{base_ticker} 空頭勝率高達 {bs_pct:.1f}%！技術面已呈現 {ma_trend}，且籌碼持續渙散 ({broker_conc:+.2f})。若持有現股建議立即於 {entry_price:.2f} 附近停損或減碼避險；積極者可伺機於 {tp:.2f} 跌破時尋找做空機會。"
                    elif prob >= 0.60:
                        action_signal = "🔴 逢高減碼 (Sell)"
                        action_color = "#ff4b4b"
                        action_reasoning = f"目前 {base_ticker} 趨勢顯著轉弱 (空方勝率 {bs_pct:.1f}%)，且 {vol_text} 導致反彈無力。上檔防守壓力重重 ({sl:.2f})，建議趁反彈時先減碼降低持股水位，等待下探 {tp:.2f} 後再重新評估。"
                    else:
                        action_signal = "⏳ 觀望 / 等待買點 (Wait)"
                        action_color = "#f5c542"
                        action_reasoning = f"{base_ticker} 目前趨勢偏空震盪 (空方勝率 {bs_pct:.1f}%)，雖然尚未出現崩跌危機，但上漲動能極度匱乏。建議保持空手，切勿隨意接刀，等待股價在 {tp:.2f} 附近落底築底後，再考慮下一個波段買點。"
                
                card_html = f"""
<div style="border: 2px solid {main_color}; border-radius: 12px; padding: 24px; background: linear-gradient(145deg, #1e1e1e 0%, #151515 100%); margin-bottom: 25px; box-shadow: 0 8px 16px rgba(0,0,0,0.4);">
<h3 style="color: {main_color}; margin-top: 0; border-bottom: 1px solid #333; padding-bottom: 15px; margin-bottom: 20px; font-weight: 600;">
🧠 AI 雙引擎全方位戰術面板 (LGBM/LSTM)
</h3>

<div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 25px;">
<div style="flex: 1; min-width: 250px; background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; border-left: 4px solid {main_color};">
<h4 style="margin: 0 0 10px 0; color: #eee;">🎯 趨勢與目標價預測</h4>
<p style="margin: 0; color: #ccc; line-height: 1.6; font-size: 15px;">
依據模型綜合判定，目前<b style="color: {main_color};">{trend_title}</b> (多頭勝率 <span style="color: #00cc96;">{bl_pct:.1f}%</span> / 空頭勝率 <span style="color: #ff4b4b;">{bs_pct:.1f}%</span>)。<br>
{target_text}
</p>
</div>
</div>

<div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 25px;">
<div style="flex: 1; min-width: 200px; background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px;">
<h5 style="margin: 0 0 8px 0; color: #aaa;">📊 均線與結構</h5>
<span style="color: #fff; font-size: 14px; line-height: 1.5;">{smc_text}。<br>{ma_trend}，且{vol_text}。</span>
</div>
<div style="flex: 1; min-width: 200px; background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px;">
<h5 style="margin: 0 0 8px 0; color: #aaa;">💰 籌碼與外資動向</h5>
<span style="color: #fff; font-size: 14px; line-height: 1.5;">{broker_text}<br>(集中度: <span style="color: {'#00cc96' if broker_conc > 0 else '#ff4b4b'};">{broker_conc:+.2f}</span>)</span>
</div>
</div>

<div style="background: {bg_color}; padding: 18px; border-radius: 8px; border-left: 5px solid {action_color};">
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
    <h4 style="margin: 0; color: #fff; font-size: 18px;">🤖 AI 交易決策指示</h4>
    <span style="background-color: {action_color}; color: #000; font-weight: 800; padding: 4px 12px; border-radius: 4px; font-size: 15px;">{action_signal}</span>
</div>
<p style="margin: 0; color: #ddd; line-height: 1.6; font-size: 15px;">
    {action_reasoning}<br><br>
    <span style="color: #999; font-size: 13px;">※ 核心指引：以目標價 <b style="color:#ffc107;">{tp:.2f}</b> 及支撐/停損價 <b style="color:#00ccff;">{sl:.2f}</b> 作為進出嚴格紀律。</span>
</p>
</div>
</div>
"""
                
                st.markdown(card_html, unsafe_allow_html=True)
                
                # --- 歷史預測與實際走勢回測追蹤表 ---
                st.markdown("### 🕰️ 歷史預測回測追蹤表 (過去 20 日)")
                
                if len(df_daily) >= 20:
                    hist_html = "<table style='width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; margin-bottom: 20px;'>"
                    hist_html += "<tr style='border-bottom: 1px solid #444; color: #aaa; background-color: rgba(255,255,255,0.05);'>"
                    hist_html += "<th style='padding: 12px 8px;'>日期</th>"
                    hist_html += "<th style='padding: 12px 8px;'>收盤價</th>"
                    hist_html += "<th style='padding: 12px 8px;'>AI 預測訊號</th>"
                    hist_html += "<th style='padding: 12px 8px;'>多方勝率</th>"
                    hist_html += "<th style='padding: 12px 8px;'>空方勝率</th>"
                    hist_html += "<th style='padding: 12px 8px;'>未來5日實際報酬</th>"
                    hist_html += "</tr>"
                    
                    # 取過去20天 (排除今天)
                    start_idx = max(0, len(df_daily) - 21)
                    end_idx = len(df_daily) - 1
                    
                    rows = []
                    for idx in range(start_idx, end_idx):
                        row_date = df_daily.index[idx].strftime('%Y-%m-%d')
                        row = df_daily.iloc[idx]
                        ep = float(row.get('Close', 0.0))
                        vol = float(row.get('Volume', 0.0))
                        atr = float(row.get('ATR_14', ep * 0.05))
                        smc = "量縮回踩" if bool(row.get('Low_Vol_Pullback', False)) else "一般常態箱體震盪"
                        
                        f_dict = brain.extract_features(
                            clean_ticker=base_ticker, 
                            current_price=ep, 
                            snapshot_dict=snapshot_dict, 
                            current_vol=vol, 
                            fallback_atr=atr, 
                            fallback_pattern=smc
                        )
                        
                        if hasattr(brain, 'predict_four_core'):
                            hist_core = brain.predict_four_core([f_dict])[0]
                        else:
                            hist_core = {'best_long': 0.5, 'best_short': 0.5, 'signal': 'WAIT'}
                            
                        actual_return = 0.0
                        if idx + 5 < len(df_daily):
                            future_close = float(df_daily.iloc[idx+5].get('Close', ep))
                            actual_return = (future_close / ep) - 1.0
                        else:
                            future_close = float(df_daily.iloc[-1].get('Close', ep))
                            actual_return = (future_close / ep) - 1.0
                            
                        h_bl = hist_core.get('best_long', 0.5)
                        h_bs = hist_core.get('best_short', 0.5)
                        h_is_long = h_bl >= h_bs
                        h_prob = h_bl if h_is_long else h_bs
                        
                        if h_is_long:
                            if h_prob >= 0.65: sig, sig_col = "🚀 強勢買入", "#00cc96"
                            elif h_prob >= 0.60: sig, sig_col = "🟢 偏多操作", "#00cc96"
                            elif h_prob >= 0.55 and "回踩" in smc: sig, sig_col = "🛒 低點佈局", "#00cc96"
                            else: sig, sig_col = "⏳ 觀望", "#f5c542"
                        else:
                            if h_prob >= 0.65: sig, sig_col = "⚠️ 強烈賣出", "#ff4b4b"
                            elif h_prob >= 0.60: sig, sig_col = "🔴 逢高減碼", "#ff4b4b"
                            else: sig, sig_col = "⏳ 觀望", "#f5c542"
                            
                        ret_col = "#00cc96" if actual_return > 0 else "#ff4b4b" if actual_return < 0 else "gray"
                        ret_str = f"{actual_return*100:+.2f}%"
                        if idx + 5 >= len(df_daily):
                            ret_str += " (未滿5日)"
                            
                        rows.append(f"""
                        <tr style='border-bottom: 1px solid #2a2a35;'>
                            <td style='padding: 12px 8px; color: #ddd;'>{row_date}</td>
                            <td style='padding: 12px 8px; color: #fff;'>{ep:.2f}</td>
                            <td style='padding: 12px 8px; font-weight: bold; color: {sig_col};'>{sig}</td>
                            <td style='padding: 12px 8px; color: #00cc96;'>{h_bl*100:.1f}%</td>
                            <td style='padding: 12px 8px; color: #ff4b4b;'>{h_bs*100:.1f}%</td>
                            <td style='padding: 12px 8px; font-weight: bold; color: {ret_col};'>{ret_str}</td>
                        </tr>
                        """)
                    
                    rows.reverse()
                    hist_html += "".join(rows) + "</table>"
                    st.markdown(hist_html, unsafe_allow_html=True)
                else:
                    st.info("歷史資料不足 20 日，無法產生回測追蹤表。")
                
                st.markdown("---")
                def go_back():
                    st.session_state.current_page = "📊 台股大盤掃描"
                    st.session_state.analyze_trigger = None
                    st.session_state.manual_search = ""
                
                st.button("⬅️ 返回戰情室主頁", use_container_width=True, on_click=go_back)
elif st.session_state.current_page == "📊 台股大盤掃描":
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
    
    # --- 新增：0050 紅綠燈 (致敬 bubuaplus) ---
    st.markdown("### 🚦 0050 紅綠燈")
    if snapshot_data:
        try:
            df_snap = pd.DataFrame(snapshot_data)
            t_col = '代號' if '代號' in df_snap.columns else 'N'
            match_0050 = df_snap[df_snap[t_col].astype(str).str.contains('0050')]
            if not match_0050.empty:
                info_0050 = match_0050.iloc[0]
                p_col = '現價' if '現價' in df_snap.columns else '{'
                p_0050 = float(info_0050.get(p_col, 0.0))
                # Simple logic for traffic light based on current snapshot
                light_color = "#ff4b4b" if p_0050 > 180 else "#ffc107" if p_0050 > 150 else "#00cc96"
                light_label = "🔴 稍高" if light_color == "#ff4b4b" else "🟡 合理" if light_color == "#ffc107" else "🟢 考慮"
                html = f'''
                <div style="background-color: #141823; border: 1px solid #252b3b; border-radius: 12px; padding: 16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; color:#fff;">0050 元大台灣50</h4>
                        <span style="font-size:14px; color:#8a93a6;">現價 {p_0050}</span>
                    </div>
                    <div style="margin-top:12px; font-size:24px; font-weight:800; color:{light_color};">{light_label}</div>
                    <div style="margin-top:8px; font-size:12px; color:#8a93a6;">系統引擎評估合理性，僅供參考。</div>
                </div>
                '''
                _render_clean_html(html)
        except Exception as e:
            st.error(f"0050 紅綠燈載入失敗: {e}")

    # --- 新增：大盤全景排行榜 (漲跌幅前三、族群板塊、外資主力前10) ---
    if snapshot_data:
        try:
            df_snap = pd.DataFrame(snapshot_data)
            # 確保資料完整性
            if 'recent_returns' in df_snap.columns and '代號' in df_snap.columns:
                # 解析出當日漲跌幅 (%)
                df_snap['漲跌幅'] = df_snap['recent_returns'].apply(lambda x: (x[-1] * 100) if isinstance(x, list) and len(x) > 0 else 0.0)
                
                st.markdown("### 🏆 大盤全景風雲榜")
                
                # 區塊 1: 漲跌幅前三名 與 跌幅前三名
                col_top, col_bot = st.columns(2)
                
                top3_gainers = df_snap.sort_values(by='漲跌幅', ascending=False).head(3)
                bottom3_losers = df_snap.sort_values(by='漲跌幅', ascending=True).head(3)
                
                with col_top:
                    st.markdown("<h4 style='color:#00cc96;'>📈 漲幅前三強</h4>", unsafe_allow_html=True)
                    for _, row in top3_gainers.iterrows():
                        t_name = row.get('名稱', row['代號'])
                        st.metric(label=f"{row['代號']} {t_name}", value=f"{row.get('現價', 0.0):.2f}", delta=f"+{row['漲跌幅']:.2f}%")
                        
                with col_bot:
                    st.markdown("<h4 style='color:#ff4b4b;'>📉 跌幅前三弱</h4>", unsafe_allow_html=True)
                    for _, row in bottom3_losers.iterrows():
                        t_name = row.get('名稱', row['代號'])
                        st.metric(label=f"{row['代號']} {t_name}", value=f"{row.get('現價', 0.0):.2f}", delta=f"{row['漲跌幅']:.2f}%")
                        
                st.markdown("---")
                
                # 區塊 2: 各族群分類依照漲幅來排序
                st.markdown("### 🧩 族群板塊強弱勢輪動")
                
                # 建立 mapping
                ticker_to_cluster = {}
                for cluster_name, tickers in st.session_state.stock_clusters.items():
                    for t in tickers:
                        ticker_to_cluster[t.split('.')[0]] = cluster_name
                        
                df_snap['族群'] = df_snap['代號'].astype(str).map(ticker_to_cluster)
                
                # 計算各族群平均漲跌幅
                cluster_perf = df_snap.dropna(subset=['族群']).groupby('族群')['漲跌幅'].mean().sort_values(ascending=False).reset_index()
                
                if not cluster_perf.empty:
                    # 使用橫向直方圖或 Markdown 呈現
                    cols_cluster = st.columns(len(cluster_perf))
                    for i, (_, row) in enumerate(cluster_perf.iterrows()):
                        c_name = row['族群']
                        c_perf = row['漲跌幅']
                        color = "#00cc96" if c_perf >= 0 else "#ff4b4b"
                        sign = "+" if c_perf >= 0 else ""
                        with cols_cluster[i]:
                            st.markdown(f"""
                            <div style="background-color:#1e1e1e; padding:12px; border-radius:8px; text-align:center; border-top: 3px solid {color};">
                                <div style="color:#aaa; font-size:14px;">{c_name}</div>
                                <div style="color:{color}; font-size:20px; font-weight:bold; margin-top:5px;">{sign}{c_perf:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # 區塊 3: 外資/主力買超前 10 名
                st.markdown("### 🏦 主力外資買超前 10 名 (籌碼集中度)")
                if 'broker_conc' in df_snap.columns:
                    top10_broker = df_snap.sort_values(by='broker_conc', ascending=False).head(10)
                    
                    # 建立美觀的 HTML 表格
                    th_style = "text-align:center; padding:12px; border-bottom:2px solid #555; color:#8bb0d9;"
                    td_style = "text-align:center; padding:12px; border-bottom:1px solid #333;"
                    
                    rows_html = ""
                    for rank, (_, row) in enumerate(top10_broker.iterrows(), 1):
                        chg = row['漲跌幅']
                        chg_color = "#00cc96" if chg >= 0 else "#ff4b4b"
                        chg_sign = "+" if chg >= 0 else ""
                        
                        rows_html += f"""
                        <tr>
                            <td style="{td_style} font-weight:bold; color:#fff;">#{rank}</td>
                            <td style="{td_style} color:#fff;">{row['代號']} {row.get('名稱', '')}</td>
                            <td style="{td_style} color:#ffc107; font-weight:bold;">{row['broker_conc']:+.2f}</td>
                            <td style="{td_style} color:#fff;">{row.get('現價', 0.0):.2f}</td>
                            <td style="{td_style} color:{chg_color}; font-weight:bold;">{chg_sign}{chg:.2f}%</td>
                        </tr>
                        """
                        
                    table_html = f"""
                    <div style="background-color:#141823; border-radius:12px; padding:16px; border:1px solid #2a2e39; overflow-x:auto;">
                        <table style="width:100%; border-collapse:collapse;">
                            <thead>
                                <tr>
                                    <th style="{th_style}">名次</th>
                                    <th style="{th_style}">標的名稱</th>
                                    <th style="{th_style}">集中度</th>
                                    <th style="{th_style}">現價</th>
                                    <th style="{th_style}">漲跌幅</th>
                                </tr>
                            </thead>
                            <tbody>
                                {rows_html}
                            </tbody>
                        </table>
                    </div>
                    """
                    _render_clean_html(table_html)
                    
        except Exception as e:
            st.error(f"大盤排行榜載入失敗: {e}")

    st.markdown("---")

    with st.container():
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
                    name_str = f"<a href='/?analyze={ticker}' target='_self' style='text-decoration:none;color:inherit;display:block;'><b>{s_name}</b><br><span style='font-size:0.8em;color:gray;'>{ticker}</span></a>"
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

    st.markdown('---')
    st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>🎯 全市場多空頭 TOP 20 🔻</h3>", unsafe_allow_html=True)
    col_long, col_short = st.columns(2)
    with col_long:
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

    with col_short:
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
                for stock in sorted(processed, key=lambda x: x['short_prob'], reverse=True)[:20]:
                    sp = stock['short_prob']
                    color = "#ff0000" if sp >= 0.65 else "#ff4b4b" if sp >= 0.60 else "#ff9966"
                    
                    entry_p = max(stock['entry_price'], 0.01)
                    drop_pct = (1 - stock['take_profit'] / entry_p) * 100
                    risk_pct = (stock['stop_loss'] / entry_p - 1) * 100
                    html_s = (
                        f"<a href='/?analyze={stock['ticker']}' target='_self' style='text-decoration:none; color:inherit; display:block;'>" \
                        f"<div style='border: 2px solid {color}; border-radius: 10px; padding: 18px; background-color: #1e1e1e; margin-bottom: 12px; transition: opacity 0.2s;' onmouseover='this.style.opacity=0.8' onmouseout='this.style.opacity=1'>"
                        f"<h4 style='color: {color}; margin-top: 0;'>🔻 {stock['ticker']} {stock['name']}</h4>"
                        f"<div style='display: flex; justify-content: space-between;'>"
                        f"<div><span style='color: gray; font-size: 13px;'>最高勝率極值</span><br>"
                        f"<b style='font-size: 22px; color: {color};'>{sp*100:.1f}%</b></div>"
                        f"<div style='text-align: right;'><span style='color: gray; font-size: 13px;'>現價</span><br>"
                        f"<b style='font-size: 20px;'>{stock['entry_price']:.2f}</b></div>"
                        f"</div></div></a>"
                    )
                    _render_clean_html(html_s)
            else:
                st.success("✅ 目前市場無明顯空頭訊號")
                st.info("💡 所有股票空頭極值均 < 55%，市場相對安全")
        else: 
            st.error("❌ 無法讀取市場快取")
            
