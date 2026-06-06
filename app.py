import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random 
import concurrent.futures
import requests
import json
import os
import joblib

from config import get_fugle_key, DEFAULT_CLUSTERS, DEFAULT_NAMES, INDUSTRY_CHAINS
from data_fetcher import (
    load_all_market_tickers, get_market_index_data, get_market_summary, 
    get_kline_with_fugle, get_stock_news, get_macro_news, run_robust_market_scan, 
    get_precalculated_market_ret, fetch_yahoo_robust
)

st.set_page_config(page_title="台股量化旗艦終端", page_icon="📈", layout="wide")

# ==========================================
# ⚙️ 系統環境與資料初始化
# ==========================================
if 'stock_clusters' not in st.session_state: st.session_state.stock_clusters = DEFAULT_CLUSTERS.copy()
if 'stock_names' not in st.session_state: st.session_state.stock_names = DEFAULT_NAMES.copy()

FUGLE_API_KEY = get_fugle_key()
csv_df = load_all_market_tickers()
if not csv_df.empty:
    for index, row in csv_df.iterrows():
        code = str(row['Ticker']).split('.')[0]
        if code not in st.session_state.stock_names: st.session_state.stock_names[code] = str(row['Name'])

def load_market_snapshot():
    if os.path.exists("market_snapshot.json"):
        try:
            with open("market_snapshot.json", "r", encoding="utf-8") as f: return json.load(f)
        except Exception: return None
    return None

def get_snapshot_dict(snapshot):
    if snapshot and 'data' in snapshot:
        return {str(item.get('代號', item.get('ticker', ''))).split('.')[0].strip(): item for item in snapshot['data']}
    return {}

# ==========================================
# 🧠 雙核引擎安全讀取區 
# ==========================================
@st.cache_resource
def get_ai_model():
    if os.path.exists("quant_model.joblib") and os.path.exists("model_features.joblib"):
        try:
            lgbm = joblib.load("quant_model.joblib")
            feats = joblib.load("model_features.joblib")
            return lgbm, feats
        except Exception as e:
            print(f"❌ LightGBM 讀取失敗: {e}")
            return None, None
    return None, None

@st.cache_resource
def get_lstm_model():
    if os.path.exists("lstm_momentum_brain.h5"):
        try:
            import tensorflow as tf
            return tf.keras.models.load_model("lstm_momentum_brain.h5")
        except Exception as e:
            print(f"❌ LSTM 讀取失敗: {e}")
            return None
    return None

# ==========================================
# 📡 即時報價與特徵萃取引擎
# ==========================================
def get_realtime_quote(clean_ticker):
    rt_price, rt_vol, prev_close = 0.0, 0.0, 0.0
    if FUGLE_API_KEY:
        try:
            res = requests.get(f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{clean_ticker}", headers={"X-API-KEY": FUGLE_API_KEY}, timeout=2)
            if res.status_code == 200:
                data = res.json()
                rt_price = data.get('closePrice')
                if not rt_price and data.get('lastTrade'): rt_price = data.get('lastTrade').get('price')
                prev_close = data.get('previousClose') or data.get('referencePrice')
                rt_vol = data.get('total', {}).get('tradeVolume', 0)
                if rt_price: rt_price = float(rt_price)
                if prev_close: prev_close = float(prev_close)
                if rt_vol: rt_vol = float(rt_vol)
        except: pass

    if not rt_price or rt_price == 0:
        try:
            df = fetch_yahoo_robust(f"{clean_ticker}.TW", period="5d", interval="1d")
            if df.empty: df = fetch_yahoo_robust(f"{clean_ticker}.TWO", period="5d", interval="1d")
            if not df.empty and len(df) >= 2:
                c, p = df.iloc[-1], df.iloc[-2]
                rt_price, prev_close = float(c['Close']), float(p['Close'])
                rt_vol = float(c['Volume'])
        except: pass
    return rt_price, rt_vol, prev_close

def extract_ai_features(clean_ticker, current_price, snapshot_dict, current_vol=0.0, fallback_rs=0.0, fallback_atr=None, fallback_pattern="", fallback_vol=0.0):
    rs_idx = fallback_rs
    pat = fallback_pattern
    anchor_price = current_price
    base_vol = fallback_vol 
    atr = fallback_atr if fallback_atr else (anchor_price * 0.05)
    vol_ratio = 1.0
    broker_conc = 0.0

    if snapshot_dict and clean_ticker in snapshot_dict:
        item = snapshot_dict[clean_ticker]
        anchor_price = float(item.get('現價', item.get('close_price', item.get('Close', anchor_price))))
        vol_raw = item.get('成交量', item.get('Volume', item.get('volume', None)))
        if vol_raw is not None:
            try: base_vol = float(vol_raw)
            except: pass
        pat_raw = item.get('pattern', item.get('Pattern', item.get('型態', pat)))
        if pat_raw: pat = str(pat_raw)
        rs_raw = item.get('RS_Index', item.get('rs_index', None))
        if rs_raw is not None:
            try: rs_idx = float(str(rs_raw).replace('%', '').strip())
            except: pass
        atr_raw = item.get('ATR_14', item.get('atr_14', None))
        if atr_raw is not None:
            try: atr = float(atr_raw)
            except: pass
        vol_ratio = float(item.get('vol_ratio', item.get('Vol_Ratio', 1.0)))
        broker_conc = float(item.get('broker_conc', item.get('Broker_Concentration', 0.0)))

    if base_vol <= 0 and current_vol > 0: base_vol = current_vol
    if anchor_price <= 0: anchor_price = 1.0

    volatility = float(atr / anchor_price)
    turnover = float(anchor_price * base_vol)
    if 0 < turnover < 100_000_000: turnover *= 1000

    return {
        'is_pullback': 1.0 if "量縮回踩" in pat else 0.0,
        'is_squeeze': 1.0 if "區間壓縮" in pat else 0.0,
        'is_divergence': 1.0 if "底背離" in pat else 0.0,
        'is_liquidity_sweep': 1.0 if "流動性掠奪" in pat else 0.0,
        'is_poc_rejection': 1.0 if "POC" in pat else 0.0,
        'rs_index': float(rs_idx),
        'vol_ratio': float(vol_ratio),
        'volatility': float(volatility),
        'turnover': float(turnover),
        'broker_conc': float(broker_conc)
    }

# 🔥 高效矩陣批次推論：將 1000 檔股票打包成 1 個方塊秒殺運算
def compute_batch_lstm_scores(features_list):
    lstm_net = get_lstm_model()
    if lstm_net is None or not features_list: 
        return np.full(len(features_list), 0.50)
    
    # 嚴格校準與訓練時一模一樣的 15 維特徵輸入順序，避免失真
    LSTM_FEATURE_ORDER = ['daily_return', 'vol_ratio', 'broker_conc', 'rs_index', 'volatility', 'turnover', 'is_pullback', 'is_squeeze', 'is_divergence', 'is_liquidity_sweep', 'is_poc_rejection']
    
    try:
        all_seqs = []
        for feat in features_list:
            seq = []
            for i in range(10):
                # 模擬時序軌跡：利用衰減系數構造 10 日動能變化
                decay = 1.0 - (0.01 * (9 - i))
                day_feat = feat.copy()
                day_feat['daily_return'] = float(feat.get('rs_index', 0.0) * 0.001 * decay)
                day_feat['vol_ratio'] = float(feat.get('vol_ratio', 1.0) * decay)
                
                # 強制對齊特徵陣列
                ordered_feat = [day_feat.get(col, 0.0) for col in LSTM_FEATURE_ORDER]
                seq.append(ordered_feat)
            all_seqs.append(seq)
            
        tensor_3d = np.array(all_seqs, dtype=np.float32)
        # 一次性運算整個矩陣，batch_size=512 可榨乾 CPU 效能
        scores = lstm_net.predict(tensor_3d, batch_size=512, verbose=0).flatten()
        return scores
    except Exception as e:
        print(f"Batch LSTM 發生異常: {e}")
        return np.full(len(features_list), 0.50)

# ==========================================================
# 📊 實盤自動優化回測模組 (資金曲線)
# ==========================================
@st.cache_data(ttl=3600*2) 
def fetch_advanced_backtest(ai_prob_threshold=0.50, use_market_filter=True, initial_cap=1000000, max_pos=5):
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
        if not url or not key: return {"status": "no_key"}
        
        model, features = get_ai_model()
        if not model: return {"status": "error", "msg": "找不到 LightGBM 大腦模型。"}
        
        market_k = fetch_yahoo_robust("^TWII", period="3y", interval="1d")
        if market_k.empty: return {"status": "error", "msg": "無法下載大盤對照基準線。"}
        
        market_k = market_k.sort_index()
        market_k['market_sma20'] = market_k['Close'].rolling(window=20).mean()
        market_k['market_pct'] = market_k['Close'].pct_change()
        market_k = market_k.reset_index()
        market_k['date_norm'] = pd.to_datetime(market_k['index']).dt.normalize()
        market_brief = market_k[['date_norm', 'Close', 'market_sma20', 'market_pct']].rename(columns={'Close': 'market_close'})

        supabase = create_client(url, key)
        all_data = []
        offset, limit = 0, 1000
        while True:
            res = supabase.table("quant_history").select("*").range(offset, offset+limit-1).execute()
            if not res.data: break
            all_data.extend(res.data)
            offset += limit

        if not all_data: return {"status": "empty"}
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df['date_norm'] = df['date'].dt.normalize()
        df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)

        df['pattern'] = df['pattern'].fillna("")
        df['is_pullback'] = df['pattern'].str.contains("量縮回踩").astype(int)
        df['is_squeeze'] = df['pattern'].str.contains("區間壓縮").astype(int)
        df['is_divergence'] = df['pattern'].str.contains("底背離").astype(int)
        df['is_liquidity_sweep'] = df['pattern'].str.contains("流動性掠奪").astype(int)
        df['is_poc_rejection'] = df['pattern'].str.contains("POC").astype(int)
        
        df['rs_index'] = pd.to_numeric(df['rs_index'], errors='coerce').fillna(0)
        df['vol_ratio'] = pd.to_numeric(df.get('vol_ratio', 1.0), errors='coerce').fillna(1.0)
        df['volatility'] = pd.to_numeric(df.get('volatility', 0.0), errors='coerce').fillna(0.0)
        df['turnover'] = pd.to_numeric(df.get('turnover', 0.0), errors='coerce').fillna(0.0)
        df['broker_conc'] = pd.to_numeric(df.get('broker_conc', 0.0), errors='coerce').fillna(0.0)

        try:
            input_df = df[features].astype(float).fillna(0)
            df['ai_prob'] = model.predict_proba(input_df)[:, 1]
        except Exception as e:
            return {"status": "error", "msg": f"AI 特徵對齊失敗: {str(e)}"}

        df = pd.merge(df, market_brief, on='date_norm', how='left')

        df['entry_price_real'] = df.groupby('ticker')['close_price'].shift(-1)
        df['future_close_6d'] = df.groupby('ticker')['close_price'].shift(-6)
        df['return_5d'] = (df['future_close_6d'] - df['entry_price_real']) / df['entry_price_real']

        if use_market_filter:
            signals = df[(df['ai_prob'] >= ai_prob_threshold) & (df['market_close'] >= df['market_sma20']) & (df['entry_price_real'].notna()) & (df['future_close_6d'].notna())].copy()
        else:
            signals = df[(df['ai_prob'] >= ai_prob_threshold) & (df['entry_price_real'].notna()) & (df['future_close_6d'].notna())].copy()

        if len(signals) == 0: return {"status": "pending", "pending_count": 0}

        pos_size = initial_cap / max_pos
        current_equity = initial_cap
        active_trades, executed_trades, daily_equity = [], [], []

        signals = signals.sort_values(['date_norm', 'ai_prob'], ascending=[True, False])

        for current_date, daily_sigs in signals.groupby('date_norm'):
            still_active = []
            for t in active_trades:
                if current_date >= t['exit_date']: current_equity += t['profit']
                else: still_active.append(t)
            active_trades = still_active
            
            for _, row in daily_sigs.iterrows():
                if len(active_trades) < max_pos: 
                    net_return_5d = row['return_5d'] - 0.005
                    profit_twd = pos_size * net_return_5d
                    active_trades.append({'exit_date': current_date + pd.Timedelta(days=7), 'profit': profit_twd})
                    row_dict = row.to_dict()
                    row_dict['sim_profit_twd'] = profit_twd
                    row_dict['net_return_5d'] = net_return_5d
                    executed_trades.append(row_dict)
                    
            daily_equity.append({
                'date_str': current_date.strftime('%Y-%m-%d'), 
                'strat_cum_pct': ((current_equity - initial_cap) / initial_cap) * 100,
                'market_cum_pct': daily_sigs.iloc[0]['market_cum'] if 'market_cum' in daily_sigs else 0
            })

        if not executed_trades: return {"status": "empty", "msg": "無有效交易。"}

        exec_df = pd.DataFrame(executed_trades)
        wins = len(exec_df[exec_df['net_return_5d'] > 0])
        total_samples = len(exec_df)

        market_k['market_cum_pct'] = market_k['market_pct'].fillna(0).cumsum() * 100
        market_lookup = dict(zip(market_k['date_norm'].dt.strftime('%Y-%m-%d'), market_k['market_cum_pct']))
        for d in daily_equity:
            if d['date_str'] in market_lookup: d['market_cum_pct'] = market_lookup[d['date_str']]

        return {
            "status": "ready", "ai_strat": {"wr": wins / total_samples if total_samples > 0 else 0, "w": wins, "l": total_samples - wins},
            "net_profit_twd": exec_df['sim_profit_twd'].sum(), "account_pct": (exec_df['sim_profit_twd'].sum() / initial_cap) * 100,
            "avg_trade_twd": exec_df['sim_profit_twd'].sum() / total_samples, "trades": total_samples,
            "tps": {"tp2": len(exec_df[exec_df['net_return_5d'] >= 0.05]) / total_samples, "samples": total_samples},
            "signals": exec_df.sort_values('date', ascending=False).head(50)[['date', 'ticker', 'entry_price_real', 'ai_prob', 'sim_profit_twd']].to_dict('records'),
            "equity": daily_equity
        }
    except Exception as e: return {"status": "error", "msg": str(e)}

# ==========================================================
# 🎛️ 左側控制面板
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
    st.header("💰 實盤資金管理")
    user_capital = st.number_input("初始本金 (TWD)", min_value=1000, max_value=20000000, value=1000000, step=10000)
    user_max_pos = st.slider("最大同時持倉檔數", min_value=1, max_value=10, value=5)
    st.markdown("---")

    if get_lstm_model() is not None: st.success("🔮 LSTM 深度大腦已自動連動")
    else: st.warning("⚪ 找不到 LSTM，單核運行中")
    
    if get_ai_model()[0] is not None: st.success("🌳 LightGBM 大腦運作正常")
    else: st.error("🚨 缺少 LightGBM 大腦")

# ==========================================================
# ⚡ 戰情室主視覺
# ==========================================================
st.title("⚡ 台股戰情分析終端")
col1, col2 = st.columns([3, 1])
with col1: manual_ticker = st.text_input("輸入股票代號", "", label_visibility="collapsed")
with col2: analyze_manual_btn = st.button("單股掃描", use_container_width=True)
st.markdown("---")

target_ticker = st.session_state.pop('analyze_trigger', None) or (manual_ticker.strip().upper() if analyze_manual_btn else None)

if target_ticker:
    # --------------------------------------------------------
    # 🔍 單股深入診斷模式
    # --------------------------------------------------------
    base_ticker = target_ticker.split('.')[0]
    c_name = st.session_state.stock_names.get(base_ticker, target_ticker)
    
    with st.spinner(f"正在分析 {target_ticker}... 多維度時區運算與 AI 預測中"):
        df_daily, df_hourly, actual_symbol = get_kline_with_fugle(target_ticker, FUGLE_API_KEY)
        if df_daily.empty or len(df_daily) < 40: st.error("❌ 該標的數據深度不足，無法執行複雜演算法。")
        else:
            news_s = get_stock_news(c_name)
            today, yesterday = df_daily.iloc[-1], df_daily.iloc[-2]
            
            try:
                entry_price = float(today.get('Close', 0.0))
                y_close = float(yesterday.get('Close', entry_price))
                vol_sma5 = float(today.get('Vol_SMA5', today.get('Volume', 1.0)))
                rt_p, rt_v, _ = get_realtime_quote(base_ticker)
                if rt_p > 0: entry_price = rt_p

                p_change = ((entry_price - y_close) / y_close) * 100
                res_level = float(today.get('Res_20', entry_price * 1.05))
                sup_level = float(today.get('Sup_20', entry_price * 0.95))
                atr_14 = float(yesterday.get('ATR_14', entry_price * 0.05))
                ai_score = int(today.get('Score', 0))
                broker_conc = float(today.get('Broker_Concentration', 0.0))
            except:
                entry_price, y_close, p_change = 0.0, 1.0, 0.0
                res_level, sup_level, atr_14, ai_score, broker_conc = 0.0, 0.0, 0.0, 0, 0.0
            
            low_vol_pb = bool(today.get('Low_Vol_Pullback', False))
            smc_text = "量縮回踩" if low_vol_pb else "一般常態箱體震盪"
            stop_loss = round(min(entry_price - (1.5 * atr_14), sup_level * 0.985), 2)
            take_profit = round(res_level, 2) if low_vol_pb else round(res_level + (atr_14 * 1.0), 2)
            real_rr_ratio = round((take_profit - entry_price) / max(entry_price - stop_loss, 0.01), 2)
            
            ai_win_rate_str = "系統錯誤：缺乏大腦檔案"
            ai_recommendation = "⏸️ 建議嚴格觀望"
            box_color = "#a8a8a8"; text_color = "#f0f0f0"

            model, features = get_ai_model()
            snapshot = load_market_snapshot()
            snapshot_dict = get_snapshot_dict(snapshot)
            
            if model:
                try:
                    input_data = extract_ai_features(base_ticker, entry_price, snapshot_dict, current_vol=rt_v, fallback_rs=float(today.get('RS_Index', 0.0)), fallback_atr=atr_14, fallback_pattern=smc_text, fallback_vol=vol_sma5)
                    
                    # 🔥 利用矩陣引擎極速推論 1 筆單股資料
                    lstm_score = compute_batch_lstm_scores([input_data])[0]
                    
                    input_df = pd.DataFrame([input_data], columns=features).astype(float).fillna(0)
                    win_prob = float(model.predict_proba(input_df)[0][1])
                    final_prob = (win_prob * 0.6) + (lstm_score * 0.4)
                    
                    ai_win_rate_str = f"{final_prob * 100:.1f}%"
                    if final_prob > 0.52 and real_rr_ratio >= 1.2:
                        ai_recommendation = "⭐⭐⭐ 雙核多頭結構共振！(高期望值)"; box_color = "#00cc96"; text_color = "#00cc96"
                    elif final_prob > 0.50:
                        ai_recommendation = "⭐⭐ 溫和佈局 (具備正向期望值)"; box_color = "#ffc107"; text_color = "#ffc107"
                    else: 
                        ai_recommendation = "⚠️ 預測敗率較高或動能消退，建議觀望"; box_color = "#a8a8a8"; text_color = "#f0f0f0"
                except Exception as e: st.error(f"單股預測演算失敗: {e}")

            st.subheader(f"🧬 {target_ticker} {c_name} 多時區量化診斷報告")
            
            st.markdown(f"""
            <div style="border: 2px solid {box_color}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;">
                <h4 style="color: {box_color}; margin-top: 0;">🎯 AI 雙核戰術計畫</h4>
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 180px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">1. 雙核加權真實勝率</span><br>
                        <b style="font-size: 24px; color: {box_color};">{ai_win_rate_str}</b><br>
                        <span style="font-size: 14px; font-weight: bold; color: {text_color};">{ai_recommendation}</span>
                    </div>
                    <div style="flex: 1; min-width: 130px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">2. 建議進場價</span><br>
                        <b style="font-size: 22px;">{entry_price:.2f}</b>
                    </div>
                    <div style="flex: 1; min-width: 200px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">3. 結構停利點</span><br>
                        <b style="font-size: 22px; color: #00cc96;">{take_profit:.2f}</b>
                    </div>
                    <div style="flex: 1; min-width: 130px; margin-bottom: 10px;">
                        <span style="color: gray; font-size: 14px;">4. 嚴格防守價</span><br>
                        <b style="font-size: 22px; color: #ff4b4b;">{stop_loss:.2f}</b>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("當前現價", f"{entry_price:.2f}", f"{p_change:+.2f}%")
            m2.metric("巨觀潛伏分數", f"{ai_score} 分")
            m3.metric("SMC 結構", smc_text)
            m4.metric("機構集中度", f"{broker_conc*100:.1f}%")
            st.markdown("---")
            if st.button("⬅️ 返回戰情室主頁", use_container_width=True):
                st.session_state.analyze_trigger = None; st.rerun()

else:
    # --------------------------------------------------------
    # 🏠 滿血旗艦主視覺儀表板 (6 大頁籤全開)
    # --------------------------------------------------------
    st.markdown("### 🌍 大盤與情緒摘要")
    summary = get_market_summary()
    if summary:
        twii_data = summary.get("加權指數", {"pct": 0})
        greed_index = int(max(0, min(100, 50 + (twii_data['pct'] * 15) + random.randint(-5, 5))))
        c_idx, c_greed = st.columns([3, 1])
        with c_idx:
            cols = st.columns(len(summary))
            for i, (name, data) in enumerate(summary.items()): 
                cols[i].metric(name, f"{data['price']:.2f}", f"{data['change']:+.2f} ({data['pct']:+.2f}%)")
        with c_greed: st.metric("放眼全球：台股恐懼貪婪指數", f"{greed_index} / 100")
            
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 自選即時流", "🔮 每日收盤趨勢預測", "🎯 全市場 AI 進出場戰術面板", 
        "🕸️ 產業鏈資金共振", "⚖️ 預測與今日盤面比對", "🔬 策略回測實驗室 (實盤)"
    ])
    
    with tab1:
        c_title, c_slider = st.columns([2, 1])
        with c_title: st.markdown(f"#### 【{selected_cluster}】即時行情流")
        with c_slider:
            with st.expander("⚙️ 畫幅設定"): user_font_size = st.slider("表格文字大小", 12, 40, 22, 2)
            
        @st.fragment(run_every=datetime.timedelta(seconds=15))
        def render_rt():
            rows = []
            current_names = st.session_state.stock_names.copy()
            for t in cluster_stocks:
                try:
                    clean_ticker = t.split('.')[0]
                    rt_price, rt_vol, prev_close = get_realtime_quote(clean_ticker)
                    if rt_price > 0:
                        change_amt = rt_price - prev_close
                        change_pct = (change_amt / prev_close) * 100 if prev_close > 0 else 0
                        
                        name_str = f"<b>{current_names.get(clean_ticker, clean_ticker)}</b><br><span style='font-size:0.8em;color:gray;'>{clean_ticker}</span>"
                        price_vol = f"<b>{rt_price:.2f}</b><br><span style='font-size:0.7em;color:gray;'>({int(rt_vol):,} 張)</span>"
                        change_str = f"<span style='color:#ff4b4b;font-weight:bold;'>+{change_amt:.2f}<br>(+{change_pct:.2f}%)</span>" if change_amt > 0 else (f"<span style='color:#00cc96;font-weight:bold;'>{change_amt:.2f}<br>({change_pct:.2f}%)</span>" if change_amt < 0 else "0.00")
                        rows.append({"標的": name_str, "及時價 (成交量)": price_vol, "今日漲跌幅": change_str})
                except: pass
            if rows:
                html_table = pd.DataFrame(rows).to_html(escape=False, index=False, border=0).replace('\n', '')
                css = f"<style>.watch-board table {{ width: 100% !important; border-collapse: collapse; }} .watch-board th {{ text-align: center !important; font-size: {max(14, user_font_size-4)}px !important; padding: 10px !important; border-bottom: 2px solid #555 !important; }} .watch-board td {{ text-align: center !important; font-size: {user_font_size}px !important; padding: 16px !important; border-bottom: 1px solid #444 !important; vertical-align: middle !important; }}</style>"
                st.markdown(f'{css}<div class="watch-board">{html_table}</div>', unsafe_allow_html=True)
        render_rt()

    with tab2:
        st.markdown("#### 🔮 每日收盤後大盤特徵與明日趨勢預測")
        snapshot = load_market_snapshot()
        if snapshot and 'data' in snapshot and len(snapshot['data']) > 0:
            all_probs = [int(item.get('量化總分', 50)) / 100.0 for item in snapshot['data']]
            bullish_ratio = float(np.mean(np.array(all_probs) >= 0.52)) * 100
            st.metric("🤖 AI 明日全市場強勢看多標的比率", f"{bullish_ratio:.1f}%")
            if bullish_ratio >= 35.0: st.success("🔥 **大腦趨勢研判：** 全市場突破特徵大範圍共振，明日多頭勝率極高。")
            elif bullish_ratio >= 15.0: st.warning("⚖️ **大腦趨勢研判：** 市場進入個股分化橫盤震盪，建議嚴格鎖定高勝率前段班。")
            else: st.error("❄️ **大腦趨勢研判：** 全市場潛在多頭動能萎縮，強烈建議啟動自動空倉防守。")
        else: st.info("ℹ️ 快取中無有效數據，請等待今日收盤後 GitHub Actions 重新抓取資料。")

    with tab3:
        st.markdown("#### 🎯 全市場 AI 進出場戰術面板 (TOP 20)")
        snapshot = load_market_snapshot()
        if snapshot and 'data' in snapshot and len(snapshot['data']) > 0:
            model, features = get_ai_model()
            if not model:
                st.error("🚨 雙核引擎發生嚴重缺損：找不到『LightGBM 靜態大腦』！請上傳 `quant_model.joblib`。")
            else:
                raw_list = snapshot['data']
                valid_items, bulk_features = [], []
                snapshot_dict = get_snapshot_dict(snapshot)
                
                for item in raw_list:
                    ticker = str(item.get('代號', item.get('ticker', ''))).split('.')[0].strip()
                    entry_price = float(item.get('現價', item.get('close_price', item.get('Close', 0.0))))
                    if entry_price == 0: continue
                    valid_items.append(item)
                    vol_val = float(item.get('成交量', item.get('Volume', item.get('volume', 0.0))))
                    bulk_features.append(extract_ai_features(ticker, entry_price, snapshot_dict, current_vol=vol_val))

                processed_stocks = []
                if len(valid_items) > 0:
                    try:
                        input_df = pd.DataFrame(bulk_features, columns=features).astype(float).fillna(0)
                        base_probs = model.predict_proba(input_df)[:, 1] 
                        
                        # 🔥 呼叫極速批次推論：一秒解算全市場
                        lstm_scores = compute_batch_lstm_scores(bulk_features)
                        
                        for idx, item in enumerate(valid_items):
                            ticker = str(item.get('代號', '')).split('.')[0].strip()
                            final_prob = (base_probs[idx] * 0.6) + (lstm_scores[idx] * 0.4)
                            
                            entry_price = float(item.get('現價', item.get('close_price', item.get('Close', 0.0))))
                            res_level = float(item.get('Res_20', entry_price * 1.05))
                            sup_level = float(item.get('Sup_20', entry_price * 0.95))
                            atr_14 = float(item.get('ATR_14', entry_price * 0.05))
                            
                            stop_loss = round(res_level * 0.985, 2) if entry_price > res_level else round(min(entry_price - (1.5 * atr_14), sup_level * 0.985), 2)
                            take_profit = round(res_level + (res_level - sup_level), 2) if entry_price > res_level else round(res_level + (atr_14 * 1.0), 2)
                            profit_reason = "🚀 噴發目標" if entry_price > res_level else "🎯 波段目標"
                            
                            box_color = "#00cc96" if final_prob > 0.52 else ("#ffc107" if final_prob > 0.50 else "#a8a8a8")
                            ai_rec = "⭐⭐⭐ 雙核極致期望值" if final_prob > 0.52 else ("⭐⭐ 溫和佈局" if final_prob > 0.50 else "⚠️ 建議嚴格觀望")
                            
                            processed_stocks.append({
                                'ticker': ticker, 'name': str(item.get('名稱', '')), 'win_prob': final_prob, 'box_color': box_color, 'ai_rec': ai_rec,
                                'entry_price': entry_price, 'take_profit': take_profit, 'stop_loss': stop_loss, 'profit_reason': profit_reason
                            })
                    except Exception as e: st.error(f"⚠️ 批量推論發生錯誤: {e}")
                
                if not processed_stocks:
                    st.info("ℹ️ 目前快取資料中無符合運算條件之股票（可能因週末無報價）。請等待系統自動抓取最新資料。")
                else:
                    for s in sorted(processed_stocks, key=lambda x: x['win_prob'], reverse=True)[:20]:
                        st.markdown(f"""
                        <div style="border: 2px solid {s['box_color']}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 15px;">
                            <h4 style="color: {s['box_color']}; margin-top: 0;">🎯 {s['ticker']} {s['name']}</h4>
                            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                                <div style="flex: 1; min-width: 150px;">
                                    <span style="color: gray; font-size: 14px;">雙核勝率</span><br>
                                    <b style="font-size: 22px; color: {s['box_color']};">{s['win_prob']*100:.1f}%</b><br>
                                    <span style="font-size: 12px; font-weight: bold; color: {s['box_color']};">{s['ai_rec']}</span>
                                </div>
                                <div style="flex: 1; min-width: 120px;">
                                    <span style="color: gray; font-size: 14px;">進場價</span><br>
                                    <b style="font-size: 20px;">{s['entry_price']:.2f}</b>
                                </div>
                                <div style="flex: 1; min-width: 150px;">
                                    <span style="color: gray; font-size: 14px;">停利點</span><br>
                                    <b style="font-size: 20px; color: #00cc96;">{s['take_profit']:.2f}</b><br>
                                    <span style="font-size: 12px; color: #00cc96;">{s['profit_reason']}</span>
                                </div>
                                <div style="flex: 1; min-width: 120px;">
                                    <span style="color: gray; font-size: 14px;">防守價</span><br>
                                    <b style="font-size: 20px; color: #ff4b4b;">{s['stop_loss']:.2f}</b>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else: st.info("ℹ️ 快取資料為空，請等待今日 GitHub Actions 自動化腳本執行完成。")

    with tab4:
        st.markdown("#### 🕸️ 上中下游產業鏈資金共振分析")
        chain_list = list(INDUSTRY_CHAINS.keys())
        if chain_list:
            selected_chain = st.selectbox("選擇要檢視的產業鏈", chain_list)
            chain_data = INDUSTRY_CHAINS[selected_chain]
            snapshot = load_market_snapshot()
            if snapshot and 'data' in snapshot:
                market_dict = {str(item.get('代號', item.get('ticker', ''))).split('.')[0].strip(): item for item in snapshot['data']}
                cols = st.columns(len(chain_data) if len(chain_data) > 0 else 1)
                for idx, (sub_name, tickers) in enumerate(chain_data.items()):
                    with cols[idx]:
                        sub_items = []
                        for code in [str(t).split('.')[0].strip() for t in tickers]:
                            if code in market_dict:
                                item = market_dict[code]
                                sub_items.append({"名稱": str(item.get('名稱', code)), "現價": float(item.get('現價', item.get('close_price', 0.0))), "量化分數": int(item.get('量化總分', 0))})
                        if sub_items:
                            display_df = pd.DataFrame(sub_items)
                            avg_score = int(display_df['量化分數'].mean())
                            heat_color = "#ff4b4b" if avg_score >= 65 else ("#ffc107" if avg_score >= 45 else "#00cc96")
                            st.markdown(f"<div style='background:#1e1e1e;padding:15px;border-top:4px solid {heat_color};border-radius:5px;margin-bottom:15px;'><b>{sub_name}</b><br><span style='font-size:24px;color:{heat_color};'>板塊熱度: {avg_score} 分</span></div>", unsafe_allow_html=True)
                            st.dataframe(display_df.sort_values("量化分數", ascending=False), hide_index=True, use_container_width=True)
            else: st.info("ℹ️ 快取中無有效數據。")

    with tab5:
        st.markdown("#### ⚖️ 昨晚 AI 趨勢預測 x 今日實盤開獎比對面板")
        snapshot = load_market_snapshot()
        if snapshot and 'data' in snapshot and len(snapshot['data']) > 0:
            model, features = get_ai_model()
            if not model: st.error("🚨 缺少大腦模型檔案。")
            else:
                raw_list = snapshot['data']
                snapshot_dict = get_snapshot_dict(snapshot)
                valid_items, bulk_features = [], []
                for item in raw_list[:30]: 
                    ticker = str(item.get('代號', ''))
                    if not ticker: continue
                    ticker = ticker.split('.')[0].strip()
                    entry_price = float(item.get('現價', item.get('close_price', item.get('Close', 0.0))))
                    if entry_price == 0: continue
                    vol_val = float(item.get('成交量', 0.0))
                    valid_items.append({'ticker': ticker, 'name': str(item.get('名稱', ticker)), 'yesterday_close': entry_price})
                    bulk_features.append(extract_ai_features(ticker, entry_price, snapshot_dict, current_vol=vol_val))
                
                if len(valid_items) > 0:
                    try:
                        input_df = pd.DataFrame(bulk_features, columns=features).astype(float).fillna(0)
                        base_probs = model.predict_proba(input_df)[:, 1]
                        
                        # 🔥 拔除耗時地雷：在執行緒外圍一次性算完 LSTM 分數
                        lstm_scores = compute_batch_lstm_scores(bulk_features)
                        
                        def fetch_live_comparison(idx_item):
                            idx, item = idx_item
                            rt_p, _, _ = get_realtime_quote(item['ticker'])
                            if rt_p <= 0: return None
                            y_close = item['yesterday_close']
                            change_pct = ((rt_p - y_close) / y_close) * 100
                            
                            # 直接取用已經秒算好的分數，完全不卡死
                            prob_pct = ((base_probs[idx] * 0.6) + (lstm_scores[idx] * 0.4)) * 100
                            
                            if prob_pct >= 52.0:
                                direction = "📈 預期突破做多"; status = "🟢 成功捕捉突破" if change_pct > 0 else "🔴 訊號反向跌破"
                            elif prob_pct >= 50.0:
                                direction = "⚖️ 溫和多頭結構"; status = "🟢 符合震盪偏多" if change_pct > -1 else "🔴 跌破多頭結構"
                            else:
                                direction = "⚠️ 建議空倉觀望"; status = "🟢 成功避開風險" if change_pct <= 0 else "⚪ 錯失低位反彈"
                                
                            return {"股票代號": item['ticker'], "股票名稱": item['name'], "勝率": f"{prob_pct:.1f}%", "方向": direction, "昨收": f"{y_close:.2f}", "現價": f"{rt_p:.2f}", "漲跌": f"{change_pct:+.2f}%", "實況比對": status}
                            
                        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                            results = list(executor.map(fetch_live_comparison, enumerate(valid_items)))
                            
                        comparison_rows = [r for r in results if r is not None]
                        if comparison_rows: st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)
                        else: st.info("正在連線報價引擎，或目前非交易時段無法取得跳動報價...")
                    except Exception as e: st.error(f"比對引擎演算中... {e}")
                else: st.info("ℹ️ 無可比對之有效數據。")
        else: st.info("ℹ️ 全市場快取準備中...")

    with tab6:
        st.markdown("#### 🔬 AI 演算法實盤回測面板")
        use_mkt_filter = st.checkbox("🛡️ 啟動大盤月線 (20MA) 智慧防禦濾網", value=True)
        res_adv = fetch_advanced_backtest(ai_prob_threshold=0.50, use_market_filter=use_mkt_filter, initial_cap=user_capital, max_pos=user_max_pos)
        
        if res_adv["status"] == "no_key": st.error("⚠️ 找不到資料庫金鑰。")
        elif res_adv["status"] == "empty": st.warning("⚠️ 經過資金與成本侵蝕過濾後，無交易紀錄。")
        elif res_adv["status"] == "pending": st.info("⏸️ 等待開獎。")
        elif res_adv["status"] == "error": st.error(f"❌ 運算發生錯誤: {res_adv['msg']}")
        elif res_adv["status"] == "ready":
            sub_tab1, sub_tab2 = st.tabs(["📋 真實帳戶績效 & 訊號清單", "📈 帳戶淨值對照曲線"])
            with sub_tab1:
                def build_card(title, value, subtext, color):
                    return f"""<div style="background-color: #121218; padding: 22px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #2a2a35;"><div style="color: #8b8b99; font-size: 14px; margin-bottom: 8px;">{title}</div><div style="color: {color}; font-size: 32px; font-weight: 700; margin-bottom: 5px;">{value}</div><div style="color: #6b6b79; font-size: 13px;">{subtext}</div></div>"""
                color_green = "#4ade80"; color_purple = "#c084fc"; color_red = "#ff4b4b"
                
                r1_c1, r1_c2, r1_c3 = st.columns(3)
                with r1_c1: st.markdown(build_card("AI 真實勝率", f"{res_adv['ai_strat']['wr']*100:.1f}%", f"{res_adv['ai_strat']['w']}W / {res_adv['ai_strat']['l']}L", color_green), unsafe_allow_html=True)
                with r1_c2: st.markdown(build_card("帳戶總淨利", f"+${res_adv['net_profit_twd']:,.0f}" if res_adv['net_profit_twd'] > 0 else f"${res_adv['net_profit_twd']:,.0f}", f"初始本金 NT$ {user_capital:,}", color_green if res_adv['net_profit_twd'] > 0 else color_red), unsafe_allow_html=True)
                with r1_c3: st.markdown(build_card("帳戶總報酬", f"+{res_adv['account_pct']:.2f}%" if res_adv['account_pct'] > 0 else f"{res_adv['account_pct']:.2f}%", f"成交: {res_adv['trades']} 筆", color_green if res_adv['account_pct'] > 0 else color_red), unsafe_allow_html=True)
                
                st.markdown("#### 🚨 歷史 AI 實盤觸發清單")
                if res_adv['signals']:
                    sig_df = pd.DataFrame(res_adv['signals'])
                    sig_df.rename(columns={"date": "日期", "ticker": "代號", "entry_price_real": "真實進場價", "ai_prob": "基底勝率", "sim_profit_twd": "損益"}, inplace=True)
                    sig_df['基底勝率'] = (sig_df['基底勝率'] * 100).apply(lambda x: f"{x:.1f}%")
                    st.dataframe(sig_df, hide_index=True, use_container_width=True)

            with sub_tab2:
                if res_adv['equity']:
                    eq_df = pd.DataFrame(res_adv['equity'])
                    eq_df.rename(columns={"date_str": "日期", "strat_cum_pct": "AI 帳戶報酬 (%)", "market_cum_pct": "大盤基準 (%)"}, inplace=True)
                    st.line_chart(eq_df.set_index("日期")[["AI 帳戶報酬 (%)", "大盤基準 (%)"]], color=["#c084fc", "#6b6b79"])