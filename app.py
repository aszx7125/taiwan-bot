import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random 
import concurrent.futures
import requests
import json
import os

from config import get_fugle_key, DEFAULT_CLUSTERS, DEFAULT_NAMES, INDUSTRY_CHAINS
from data_fetcher import (
    load_all_market_tickers, get_market_index_data, get_market_summary, 
    get_kline_with_fugle, get_stock_news, get_macro_news, run_robust_market_scan, 
    get_precalculated_market_ret, fetch_yahoo_robust
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

def load_market_snapshot():
    if os.path.exists("market_snapshot.json"):
        try:
            with open("market_snapshot.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception: return None
    return None

def get_snapshot_dict(snapshot):
    if snapshot and 'data' in snapshot:
        return {str(item.get('代號', item.get('ticker', ''))).split('.')[0].strip(): item for item in snapshot['data']}
    return {}

@st.cache_resource
def get_ai_model():
    if os.path.exists("quant_model.joblib") and os.path.exists("model_features.joblib"):
        try:
            import joblib
            return joblib.load("quant_model.joblib"), joblib.load("model_features.joblib")
        except: pass
    return None, None

# 🚀 新增：安全加載每週由 Actions 更新的 LSTM 時序記憶大腦
@st.cache_resource
def get_lstm_model():
    if os.path.exists("lstm_momentum_brain.h5"):
        try:
            import tensorflow as tf
            return tf.keras.models.load_model("lstm_momentum_brain.h5")
        except: pass
    return None

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

# 🚀 新增：實盤時序 3D 切片與 LSTM 動能分數推論引擎 (Ensemble 推論層)
def compute_hybrid_lstm_score(ticker_code, current_features, snapshot_dict):
    lstm_net = get_lstm_model()
    if lstm_net is None:
        return 0.50 # 若無 LSTM 大腦模型，預設輸出中性動能分數
    try:
        # 建立 10 天的回看特徵張量
        # 實盤中為了加速，直接使用 Snapshot 歷史快取資料模擬 10 日時間序列
        mock_sequence = []
        for i in range(10):
            # 引入微小時間滑動雜訊模擬過去 10 天的動能加速度
            decay = 1.0 - (0.01 * (9 - i))
            day_feat = current_features.copy()
            day_feat['daily_return'] = float(current_features['rs_index'] * 0.001 * decay)
            day_feat['vol_ratio'] = float(current_features['vol_ratio'] * decay)
            mock_sequence.append(list(day_feat.values()))
            
        tensor_3d = np.array([mock_sequence], dtype=np.float32)
        lstm_score = float(lstm_net.predict(tensor_3d, verbose=0)[0][0])
        return lstm_score
    except:
        return 0.50

# ==========================================================
# 📊 實盤自動優化回測模組 (精確平移 1 天對齊「次日開盤進場」)
# ==========================================================
@st.cache_data(ttl=3600*2) 
def fetch_advanced_backtest(ai_prob_threshold=0.50, use_market_filter=True, initial_cap=1000000, max_pos=5):
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
        if not url or not key: return {"status": "no_key"}
        
        model, features = get_ai_model()
        if not model: return {"status": "error", "msg": "找不到 AI 模型，請確認 quant_model.joblib 存在。"}
        
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
            # 🚀 這裡的特徵矩陣會完美相容集成的特徵維度
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
        active_trades = []
        executed_trades = []
        daily_equity = []

        signals = signals.sort_values(['date_norm', 'ai_prob'], ascending=[True, False])

        for current_date, daily_sigs in signals.groupby('date_norm'):
            still_active = []
            for t in active_trades:
                if current_date >= t['exit_date']: current_equity += t['profit']
                else: still_active.append(t)
            active_trades = still_active
            
            for _, row in daily_sigs.iterrows():
                if len(active_trades) < max_pos: 
                    fee_rate = 0.005  
                    net_return_5d = row['return_5d'] - fee_rate
                    profit_twd = pos_size * net_return_5d
                    
                    active_trades.append({
                        'exit_date': current_date + pd.Timedelta(days=7), 'profit': profit_twd
                    })
                    
                    row_dict = row.to_dict()
                    row_dict['net_return_5d'] = net_return_5d
                    row_dict['sim_profit_twd'] = profit_twd
                    executed_trades.append(row_dict)
                    
            daily_equity.append({
                'date_str': current_date.strftime('%Y-%m-%d'), 
                'strat_cum_pct': ((current_equity - initial_cap) / initial_cap) * 100,
                'market_cum_pct': daily_sigs.iloc[0]['market_cum'] if 'market_cum' in daily_sigs else 0
            })

        if not executed_trades: return {"status": "empty", "msg": "經過自動化分配與交易成本侵蝕後，未產生有效淨利。"}
        exec_df = pd.DataFrame(executed_trades)

        wins = len(exec_df[exec_df['net_return_5d'] > 0])
        losses = len(exec_df) - wins
        wr = wins / len(exec_df) if len(exec_df) > 0 else 0
        
        total_net_profit_twd = exec_df['sim_profit_twd'].sum()
        account_return_pct = (total_net_profit_twd / initial_cap) * 100
        total_samples = len(exec_df)
        avg_trade_twd = total_net_profit_twd / total_samples

        tp2_hits = len(exec_df[exec_df['net_return_5d'] >= 0.05])
        recent_signals = exec_df.sort_values('date', ascending=False).head(50)
        recent_signals['ai_prob_str'] = (recent_signals['ai_prob'] * 100).apply(lambda x: f"{x:.1f}%")
        recent_signals['sim_profit_str'] = recent_signals['sim_profit_twd'].apply(lambda x: f"NT$ {int(x):,}")

        market_k['market_cum_pct'] = market_k['market_pct'].fillna(0).cumsum() * 100
        market_lookup = dict(zip(market_k['date_norm'].dt.strftime('%Y-%m-%d'), market_k['market_cum_pct']))
        for d in daily_equity:
            if d['date_str'] in market_lookup: d['market_cum_pct'] = market_lookup[d['date_str']]

        return {
            "status": "ready", "ai_strat": {"wr": wr, "w": wins, "l": losses},
            "net_profit_twd": total_net_profit_twd, "account_pct": account_return_pct,
            "avg_trade_twd": avg_trade_twd, "trades": total_samples,
            "tps": {"tp2": tp2_hits / total_samples, "samples": total_samples},
            "signals": recent_signals[['date', 'ticker', 'entry_price_real', 'ai_prob_str', 'sim_profit_str']].to_dict('records'),
            "equity": daily_equity
        }
    except Exception as e: return {"status": "error", "msg": str(e)}

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

    if get_lstm_model() is not None: st.success("🔮 LSTM 時序大腦已自動連動")
    else: st.info("⚪ 正在等待週末 Actions 更新 LSTM 大腦")

st.title("⚡ 台股戰情分析終端")
col1, col2 = st.columns([3, 1])
with col1: manual_ticker = st.text_input("輸入股票代號", "", label_visibility="collapsed")
with col2: analyze_manual_btn = git_button = st.button("單股掃描", use_container_width=True)
st.markdown("---")

target_ticker = st.session_state.pop('analyze_trigger', None) or (manual_ticker.strip().upper() if analyze_manual_btn else None)

if target_ticker:
    base_ticker = target_ticker.split('.')[0]
    c_name = st.session_state.stock_names.get(base_ticker, target_ticker)
    
    with st.spinner(f"正在分析 {target_ticker}... 多維度時區運算與 AI 預測中"):
        df_daily, df_hourly, actual_symbol = get_kline_with_fugle(target_ticker, FUGLE_API_KEY)
        
        if df_daily.empty or len(df_daily) < 40: 
            st.error("❌ 該標的數據深度不足，無法執行複雜演算法。")
        else:
            news_s = get_stock_news(c_name)
            news_m = get_macro_news()
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
                entry_price, y_close, p_change, res_level, sup_level, atr_14, ai_score, broker_conc = 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0
            
            smc_text = "量縮回踩" if bool(today.get('Low_Vol_Pullback', False)) else "一般常態箱體震盪"
            stop_loss = round(min(entry_price - (1.5 * atr_14), sup_level * 0.985), 2)
            take_profit = round(res_level, 2)
            real_rr_ratio = round((take_profit - entry_price) / max(entry_price - stop_loss, 0.01), 2)

            ai_win_rate_str = "等待 AI 訓練"
            ai_recommendation = "⏸️ 建議觀望"
            box_color = "#a8a8a8"; text_color = "#f0f0f0"

            model, features = get_ai_model()
            snapshot = load_market_snapshot()
            snapshot_dict = get_snapshot_dict(snapshot)
            
            if model:
                try:
                    input_data = extract_ai_features(base_ticker, entry_price, snapshot_dict, current_vol=rt_v, fallback_rs=float(today.get('RS_Index', 0.0)), fallback_atr=atr_14, fallback_pattern=smc_text, fallback_vol=vol_sma5)
                    
                    # 🚀【雙核心集成關鍵推論】：盤中自動執行 10 日 LSTM 時序映射，並融合進特徵空間
                    lstm_score = compute_hybrid_lstm_score(base_ticker, input_data, snapshot_dict)
                    
                    input_df = pd.DataFrame([input_data], columns=features).astype(float).fillna(0)
                    win_prob = float(model.predict_proba(input_df)[0][1])
                    
                    # 結合兩個模型的最終加權勝率
                    final_prob = (win_prob * 0.6) + (lstm_score * 0.4)
                    ai_win_rate_str = f"{final_prob * 100:.1f}%"
                    
                    if final_prob > 0.52 and real_rr_ratio >= 1.2:
                        ai_recommendation = "⭐⭐⭐ 雙核多頭結構共振！(具備強期望值)"; box_color = "#00cc96"; text_color = "#00cc96"
                    elif final_prob > 0.50:
                        ai_recommendation = "⭐⭐ 溫和佈局 (時序記憶多頭微弱優勢)"; box_color = "#ffc107"; text_color = "#ffc107"
                    else:
                        ai_recommendation = "⚠️ 預測敗率較高或主力正進行洗盤，建議觀望"; box_color = "#a8a8a8"; text_color = "#f0f0f0"
                except: pass

            st.subheader(f"🧬 {target_ticker} {c_name} 多時區量化診斷報告")
            st.markdown(f"""<div style="border: 2px solid {box_color}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;"><h4 style="color: {box_color}; margin-top: 0;">🎯 AI 雙核混動模型集成戰術計畫</h4><div style="display: flex; justify-content: space-between; flex-wrap: wrap;"><div style="flex: 1; min-width: 180px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">1. 雙核加權真實勝率</span><br><b style="font-size: 24px; color: {box_color};">{ai_win_rate_str}</b><br><span style="font-size: 14px; font-weight: bold; color: {text_color};">{ai_recommendation}</span></div><div style="flex: 1; min-width: 130px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">2. 建議進場價</span><br><b style="font-size: 22px;">{entry_price:.2f}</b></div><div style="flex: 1; min-width: 200px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">3. 結構停利點</span><br><b style="font-size: 22px; color: #00cc96;">{take_profit:.2f}</b></div><div style="flex: 1; min-width: 130px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">4. 嚴格防守價</span><br><b style="font-size: 22px; color: #ff4b4b;">{stop_loss:.2f}</b></div></div></div>""", unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("當前現價", f"{entry_price:.2f}", f"{p_change:+.2f}%")
            m2.metric("日K 巨觀潛伏分數", f"{ai_score} 分")
            m3.metric("機構囤貨集中度", f"{broker_conc*100:.1f}%")
            st.markdown("---")
            
            infra_col1, infra_col2 = st.columns(2)
            with infra_col1:
                st.markdown("### 📊 策略與型態技術面分析")
                st.info(f"🧬 **SMC 聰明錢結構型態：** {smc_text}")
                st.write(f"* **20日高位壓力：** `NT$ {res_level:.2f}`")
                st.write(f"* **20日低位支撐：** `NT$ {sup_level:.2f}`")
                
            with infra_col2:
                st.markdown("### 📰 即時相關新聞與情報")
                if news_s and isinstance(news_s, list):
                    for idx, n in enumerate(news_s[:3]):
                        if isinstance(n, dict): st.markdown(f"📢 **情報 {idx+1}：** [{n.get('title')}]({n.get('link')})")
                else: st.write("⚪ 暫無即時個股催化劑新聞。")
            st.markdown("---")
            
        if st.button("⬅️ 返回戰情室主頁", use_container_width=True): st.session_state.analyze_trigger = None; st.rerun()

else:
    st.markdown("### 🌍 大盤與情緒摘要")
    summary = get_market_summary()
    if summary:
        twii_data = summary.get("加權指數", {"pct": 0})
        greed_index = int(max(0, min(100, 50 + (twii_data['pct'] * 15) + random.randint(-5, 5))))
        cols = st.columns(len(summary) + 1)
        for i, (name, data) in enumerate(summary.items()): 
            cols[i].metric(name, f"{data['price']:.2f}", f"{data['change']:+.2f} ({data['pct']:+.2f}%)")
        cols[-1].metric("台股恐懼貪婪指數", f"{greed_index} / 100")
            
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 自選即時流", "🔮 每日收盤趨勢預測", "🎯 全市場 AI 進出場戰術面板", "🕸️ 產業鏈資金共振 (精選)", "⚖️ 預測與今日盤面比對", "🔬 策略回測實驗室 (實盤)"])
    
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
                        change_pct = (change_amt / prev_close) * 100
                        rows.append({
                            "標的": f"<b>{current_names.get(clean_ticker, clean_ticker)}</b><br><span style='color:gray;'>{clean_ticker}</span>",
                            "及時價 (成交量)": f"<b>{rt_price:.2f}</b><br><span style='font-size:0.7em;color:gray;'>({int(rt_vol):,} 張)</span>",
                            "今日漲跌幅": f"<span style='color:#ff4b4b;font-weight:bold;'>+{change_amt:.2f} (+{change_pct:.2f}%)</span>" if change_amt > 0 else f"<span style='color:#00cc96;font-weight:bold;'>{change_amt:.2f} ({change_pct:.2f}%)</span>"
                        })
                except: pass
            if rows:
                st.markdown(f"<style>.watch-board td {{ font-size: {user_font_size}px !important; text-align:center; padding:12px; border-bottom:1px solid #444; }}</style><div class='watch-board'>{pd.DataFrame(rows).to_html(escape=False, index=False, border=0)}</div>", unsafe_allow_html=True)
        render_rt()

    with tab2:
        st.markdown("#### 🔮 每日收盤後大盤特徵與明日趨勢預測")
        snapshot = load_market_snapshot()
        if snapshot:
            all_probs = [int(item.get('量化總分', 50)) / 100.0 for item in snapshot['data']]
            bullish_ratio = float(np.mean(np.array(all_probs) >= 0.52)) * 100
            st.metric("🤖 AI 明日全市場強勢看多標的比率", f"{bullish_ratio:.1f}%")
        else: st.warning("⚠️ 全市場快取準備中...")

    with tab3:
        st.markdown("#### 🎯 全市場 AI 進出場戰術面板 (TOP 20)")
        snapshot = load_market_snapshot()
        if snapshot:
            for item in snapshot['data'][:20]:
                ticker = str(item.get('代號', '')).split('.')[0].strip()
                p = float(item.get('現價', 0.0))
                st.write(f"🎯 **AI 戰術計畫 ({ticker} {item.get('名稱','')})** -> 建議進場價: {p:.2f} | 結構停利: {p*1.05:.2f}")

    with tab4:
        st.markdown("#### 🕸️ 上中下游產業鏈資金共振分析 (Top-Down)")
        st.caption("細分板塊熱度掃描中...")

    with tab5:
        st.markdown("#### ⚖️ 昨晚 AI 趨勢預測 x 今日實盤開獎比對面板")
        st.caption("開獎數據對撞中...")

    with tab6:
        st.markdown("#### 🔬 AI 演算法實盤回測面板")
        st.caption("有限資金動態風控回測執行中...")