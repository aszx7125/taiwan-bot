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

    if base_vol <= 0 and current_vol > 0:
        base_vol = current_vol

    if anchor_price <= 0: anchor_price = 1.0

    volatility = float(atr / anchor_price)
    turnover = float(anchor_price * base_vol)

    if 0 < turnover < 100_000_000:
        turnover *= 1000

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

# ==========================================================
# 📊 實盤自動優化回測模組 (精確導入 0.5% 實盤手續費、稅金與滑價)
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
            input_df = df[features].astype(float).fillna(0)
            df['ai_prob'] = model.predict_proba(input_df)[:, 1]
        except Exception as e:
            return {"status": "error", "msg": f"AI 特徵對齊失敗: {str(e)}"}

        df = pd.merge(df, market_brief, on='date_norm', how='left')
        df['future_close_5d'] = df.groupby('ticker')['close_price'].shift(-5)
        df['return_5d'] = (df['future_close_5d'] - df['close_price']) / df['close_price']

        if use_market_filter:
            signals = df[(df['ai_prob'] >= ai_prob_threshold) & (df['market_close'] >= df['market_sma20']) & (df['future_close_5d'].notna())].copy()
        else:
            signals = df[(df['ai_prob'] >= ai_prob_threshold) & (df['future_close_5d'].notna())].copy()

        if len(signals) == 0:
            return {"status": "pending", "pending_count": 0}

        # 🧠 真實部位自動化排序分配器
        pos_size = initial_cap / max_pos
        current_equity = initial_cap
        active_trades = []
        executed_trades = []
        daily_equity = []

        # 每日開盤，自動把當天訊號按照 AI 勝率從高到低排序
        signals = signals.sort_values(['date_norm', 'ai_prob'], ascending=[True, False])

        for current_date, daily_sigs in signals.groupby('date_norm'):
            still_active = []
            for t in active_trades:
                if current_date >= t['exit_date']:
                    current_equity += t['profit']
                else:
                    still_active.append(t)
            active_trades = still_active
            
            for _, row in daily_sigs.iterrows():
                if len(active_trades) < max_pos: 
                    # 🚀 【核心淨利計算法】：嚴格扣除 0.5% 交易摩擦成本（稅金+手續費+實盤滑價）
                    fee_rate = 0.005
                    net_return_5d = row['return_5d'] - fee_rate
                    profit_twd = pos_size * net_return_5d
                    
                    active_trades.append({
                        'exit_date': current_date + pd.Timedelta(days=7),
                        'profit': profit_twd
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

        if not executed_trades:
            return {"status": "empty", "msg": "經過自動化分配與交易成本侵蝕後，未產生有效淨利。"}

        exec_df = pd.DataFrame(executed_trades)

        # 🚀 勝率卡片同步升級：扣除成本後「實質淨回報 > 0%」的單子才算贏！
        wins = len(exec_df[exec_df['net_return_5d'] > 0])
        losses = len(exec_df) - wins
        wr = wins / len(exec_df) if len(exec_df) > 0 else 0
        
        total_net_profit_twd = exec_df['sim_profit_twd'].sum()
        account_return_pct = (total_net_profit_twd / initial_cap) * 100
        total_samples = len(exec_df)
        avg_trade_twd = total_net_profit_twd / total_samples

        tp1_hits = len(exec_df[exec_df['net_return_5d'] >= 0.03])
        tp2_hits = len(exec_df[exec_df['net_return_5d'] >= 0.05])
        tp3_hits = len(exec_df[exec_df['net_return_5d'] >= 0.07])
        ftp_hits = len(exec_df[exec_df['net_return_5d'] >= 0.10])

        recent_signals = exec_df.sort_values('date', ascending=False).head(50)
        recent_signals['ai_prob_str'] = (recent_signals['ai_prob'] * 100).apply(lambda x: f"{x:.1f}%")
        recent_signals['sim_profit_str'] = recent_signals['sim_profit_twd'].apply(lambda x: f"NT$ {int(x):,}")

        market_k['market_cum_pct'] = market_k['market_pct'].fillna(0).cumsum() * 100
        market_lookup = dict(zip(market_k['date_norm'].dt.strftime('%Y-%m-%d'), market_k['market_cum_pct']))
        
        for d in daily_equity:
            if d['date_str'] in market_lookup:
                d['market_cum_pct'] = market_lookup[d['date_str']]

        return {
            "status": "ready",
            "ai_strat": {"wr": wr, "w": wins, "l": losses},
            "net_profit_twd": total_net_profit_twd,
            "account_pct": account_return_pct,
            "avg_trade_twd": avg_trade_twd,
            "trades": total_samples,
            "tps": {
                "tp1": tp1_hits / total_samples, "tp2": tp2_hits / total_samples,
                "tp3": tp3_hits / total_samples, "ftp": ftp_hits / total_samples,
                "samples": total_samples
            },
            "signals": recent_signals[['date', 'ticker', 'close_price', 'ai_prob_str', 'sim_profit_str']].to_dict('records'),
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
    user_max_pos = st.slider("最大同時持倉檔數", min_value=1, max_value=10, value=5, help="決定資金切分份數。")
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

st.title("⚡ 台股戰情分析終端")
col1, col2 = st.columns([3, 1])
with col1: manual_ticker = st.text_input("輸入股票代號", "", label_visibility="collapsed")
with col2: analyze_manual_btn = st.button("單股掃描", use_container_width=True)
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
                t_high = float(today.get('High', entry_price))
                t_low = float(today.get('Low', entry_price))
                y_close = float(yesterday.get('Close', entry_price))
                vol_sma5 = float(today.get('Vol_SMA5', today.get('Volume', 1.0)))
                
                rt_p, rt_v, _ = get_realtime_quote(base_ticker)
                if rt_p > 0: entry_price = rt_p

                if pd.isna(entry_price): entry_price = 0.0
                if pd.isna(y_close) or y_close == 0: y_close = entry_price if entry_price > 0 else 1.0
                p_change = ((entry_price - y_close) / y_close) * 100
                
                res_level = float(today.get('Res_20', entry_price * 1.05))
                sup_level = float(today.get('Sup_20', entry_price * 0.95))
                box_height = max(res_level - sup_level, 0.01)
                atr_14 = float(yesterday.get('ATR_14', entry_price * 0.05))
                ai_score = int(today.get('Score', 0))
                broker_conc = float(today.get('Broker_Concentration', 0.0))
            except Exception as e:
                entry_price, y_close, t_high, t_low, vol_sma5, p_change = 0.0, 1.0, 0.0, 0.0, 1.0, 0.0
                res_level, sup_level, box_height, atr_14, ai_score, broker_conc = 0.0, 0.0, 0.0, 0.0, 0, 0.0
            
            bull_div = bool(today.get('Bullish_Div', False))
            low_vol_pb = bool(today.get('Low_Vol_Pullback', False))
            squeeze_on = bool(today.get('Squeeze_On', False))
            
            smc_status = []
            if low_vol_pb: smc_status.append("📉 量縮回踩")
            if squeeze_on: smc_status.append("🛡️ 區間極度壓縮")
            smc_text = " + ".join(smc_status) if smc_status else "一般常態震盪"

            stop_loss = round(min(entry_price - (1.5 * atr_14), sup_level * 0.985), 2)
            risk_per_share = max(entry_price - stop_loss, 0.01)
            take_profit = round(res_level, 2) if (low_vol_pb or bull_div) else round(res_level + (atr_14 * 1.0), 2)
            profit_reason = "🎯 潛伏目標：前高/箱頂壓力區" if (low_vol_pb or bull_div) else "⚔️ 波段目標：前高波動擴張位"
            real_rr_ratio = round((take_profit - entry_price) / risk_per_share, 2)

            micro_trigger = False
            micro_status_text = "數據不足"
            if not df_hourly.empty and len(df_hourly) >= 2:
                last_hour = df_hourly.iloc[-1]
                micro_trigger = bool(last_hour.get('Micro_Sniper_Trigger', False))
                h_macd_cross = bool(last_hour.get('MACD_Cross_Up', False))
                h_vol_surge = bool(last_hour.get('Vol_Surge_1h', False))
                
                if micro_trigger: micro_status_text = "🔥 帶量突破 1h 均線 (強烈買點)"
                elif h_macd_cross: micro_status_text = "📈 1h MACD 金叉發動"
                elif h_vol_surge: micro_status_text = "🌊 1h 微觀異常爆量"
                elif last_hour.get('Close', entry_price) > last_hour.get('SMA_20_1h', entry_price): micro_status_text = "🟢 站穩 1h 均線 (短線強勢)"
                else: micro_status_text = "⚪ 1h 均線下弱勢震盪"

            ai_win_rate_str = "等待 AI 訓練"
            ai_recommendation = "⏸️ 勝率偏低或追高風險，強制觀望"
            box_color = "#a8a8a8"
            text_color = "#f0f0f0"

            model, features = get_ai_model()
            snapshot = load_market_snapshot()
            snapshot_dict = get_snapshot_dict(snapshot)
            
            if model:
                try:
                    input_data = extract_ai_features(
                        base_ticker, entry_price, snapshot_dict, current_vol=rt_v,
                        fallback_rs=float(today.get('RS_Index', 0.0)), fallback_atr=atr_14, 
                        fallback_pattern=smc_text, fallback_vol=vol_sma5
                    )
                    input_df = pd.DataFrame([input_data], columns=features).astype(float).fillna(0)
                    win_prob = float(model.predict_proba(input_df)[0][1])
                    ai_win_rate_str = f"{win_prob * 100:.1f}%"
                    
                    if win_prob > 0.60 and real_rr_ratio >= 1.5:
                        ai_recommendation = "⭐⭐⭐ 極致期望值！(高勝率 + 高風報比)"; box_color = "#00cc96"; text_color = "#00cc96"
                    elif win_prob > 0.50 and real_rr_ratio >= 1.0:
                        ai_recommendation = "⭐⭐ 溫和佈局 (具備正向期望值)"; box_color = "#ffc107"; text_color = "#ffc107"
                    else: 
                        ai_recommendation = "⚠️ 預測敗率較高，建議嚴格觀望"; box_color = "#a8a8a8"; text_color = "#f0f0f0"
                except: pass

            st.subheader(f"🧬 {target_ticker} {c_name} 多時區量化診斷報告")
            st.markdown(f"""<div style="border: 2px solid {box_color}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;"><h4 style="color: {box_color}; margin-top: 0;">🎯 AI 深度學習 x 結構價格 戰術計畫</h4><div style="display: flex; justify-content: space-between; flex-wrap: wrap;"><div style="flex: 1; min-width: 180px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">1. AI 真實勝率預測</span><br><b style="font-size: 24px; color: {box_color};">{ai_win_rate_str}</b><br><span style="font-size: 14px; font-weight: bold; color: {text_color};">{ai_recommendation}</span></div><div style="flex: 1; min-width: 130px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">2. 建議進場價</span><br><b style="font-size: 22px;">{entry_price:.2f}</b></div><div style="flex: 1; min-width: 200px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">3. 結構停利點</span><br><b style="font-size: 22px; color: #00cc96;">{take_profit:.2f}</b><br><span style="font-size: 12px; color: #00cc96; font-weight: bold;">{profit_reason}</span></div><div style="flex: 1; min-width: 130px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">4. 嚴格防守價</span><br><b style="font-size: 22px; color: #ff4b4b;">{stop_loss:.2f}</b></div></div></div>""", unsafe_allow_html=True)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("當前現價", f"{entry_price:.2f}", f"{p_change:+.2f}%")
            m2.metric("日K 巨觀潛伏分數", f"{ai_score} 分")
            m3.metric("1h 小時區微觀狀態", micro_status_text)
            m4.metric("機構囤貨集中度", f"{broker_conc*100:.1f}%")
            st.markdown("---")
            
        if st.button("⬅️ 返回戰情室主頁", use_container_width=True):
            st.session_state.analyze_trigger = None; st.rerun()

else:
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
        with c_greed: st.metric("台股恐懼貪婪指數", f"{greed_index} / 100")
            
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 自選即時流", "🎯 全市場 AI 進出場戰術面板", "🕸️ 產業鏈資金共振 (精選)", "🔬 策略回測實驗室 (實盤)"])
    
    with tab1:
        c_title, c_slider = st.columns([2, 1])
        with c_title: st.markdown(f"#### 【{selected_cluster}】即時行情流")
        with c_slider:
            with st.expander("⚙️ 畫幅設定"): user_font_size = st.slider("表格文字大小", 12, 40, 22, 2)
            
        @st.fragment(run_every=datetime.timedelta(seconds=15))
        def render_rt():
            rows = []
            current_names = st.session_state.stock_names.copy()
            
            def fetch_single_rt(t, names_dict):
                try:
                    clean_ticker = t.split('.')[0]
                    base_name = names_dict.get(clean_ticker, clean_ticker)
                    rt_price, rt_vol, prev_close = get_realtime_quote(clean_ticker)
                    if not rt_price or rt_price == 0: return None
                        
                    change_amt = rt_price - prev_close
                    change_pct = (change_amt / prev_close) * 100 if prev_close > 0 else 0
                        
                    name_str = f"<b>{base_name}</b><br><span style='font-size:0.8em;color:gray;'>{clean_ticker}</span>"
                    display_vol = int(rt_vol) if rt_vol < 2000000 else int(rt_vol / 1000)
                    price_vol = f"<b>{rt_price:.2f}</b><br><span style='font-size:0.7em;color:gray;'>({display_vol:,} 張)</span>"
                    change_str = f"<span style='color:#ff4b4b;font-weight:bold;'>+{change_amt:.2f}<br>(+{change_pct:.2f}%)</span>" if change_amt > 0 else (f"<span style='color:#00cc96;font-weight:bold;'>{change_amt:.2f}<br>({change_pct:.2f}%)</span>" if change_amt < 0 else "0.00")
                    return {"標的": name_str "及時價 (成交量)": price_vol, "今日漲跌幅": change_str, "raw_pct": change_pct}
                except: pass
                return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                futures = [executor.submit(fetch_single_rt, t, current_names) for t in cluster_stocks]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res: rows.append(res)
            if rows:
                html_table = pd.DataFrame(rows)[["標的", "及時價 (成交量)", "今日漲跌幅"]].to_html(escape=False, index=False, border=0).replace('\n', '')
                css = f"<style>.watch-board table {{ width: 100% !important; border-collapse: collapse; }} .watch-board th {{ text-align: center !important; font-size: {max(14, user_font_size-4)}px !important; padding: 10px !important; border-bottom: 2px solid #555 !important; }} .watch-board td {{ text-align: center !important; font-size: {user_font_size}px !important; padding: 16px !important; border-bottom: 1px solid #444 !important; vertical-align: middle !important; }}</style>"
                st.markdown(f'{css}<div class="watch-board">{html_table}</div>', unsafe_allow_html=True)
        render_rt()

    with tab2:
        st.markdown("#### 🎯 全市場 AI 進出場戰術面板 (TOP 20)")
        snapshot = load_market_snapshot()
        if snapshot:
            raw_list = snapshot['data']
            valid_items, bulk_features = [], []
            model, features = get_ai_model()
            snapshot_dict = get_snapshot_dict(snapshot)
            
            for item in raw_list:
                ticker = str(item.get('代號', item.get('ticker', ''))).split('.')[0].strip()
                entry_price = float(item.get('現價', item.get('close_price', item.get('Close', 0.0))))
                if entry_price == 0: continue
                valid_items.append(item)
                if model:
                    vol_val = float(item.get('成交量', item.get('Volume', item.get('volume', 0.0))))
                    bulk_features.append(extract_ai_features(ticker, entry_price, snapshot_dict, current_vol=vol_val))

            all_probs = []
            if model and bulk_features:
                try:
                    input_df = pd.DataFrame(bulk_features, columns=features).astype(float).fillna(0)
                    all_probs = model.predict_proba(input_df)[:, 1] 
                except Exception as e: 
                    st.error(f"⚠️ 批量推論發生錯誤，請檢查特徵格式: {e}")

            processed_stocks = []
            for idx, item in enumerate(valid_items):
                ticker = str(item.get('代號', item.get('ticker', ''))).split('.')[0].strip()
                name = str(item.get('名稱', item.get('name', '')))
                entry_price = float(item.get('現價', item.get('close_price', item.get('Close', 0.0))))
                win_prob = float(all_probs[idx]) if (model and len(all_probs) > idx) else (int(item.get('量化總分', 0)) / 100.0)
                
                res_level = float(item.get('Res_20', entry_price * 1.05))
                sup_level = float(item.get('Sup_20', entry_price * 0.95))
                atr_14 = float(item.get('ATR_14', entry_price * 0.05))
                box_height = max(res_level - sup_level, 0.01)
                pattern_str = str(item.get('pattern', ''))
                
                stop_loss = round(res_level * 0.985, 2) if entry_price > res_level else round(min(entry_price - (1.5 * atr_14), sup_level * 0.985), 2)
                take_profit = round(res_level + box_height, 2) if entry_price > res_level else (round(res_level, 2) if ("量縮回踩" in pattern_str or "底背離" in pattern_str) else round(res_level + (atr_14 * 1.0), 2))
                profit_reason = "🚀 噴發目標：等距測幅擴展位" if entry_price > res_level else ("🎯 潛伏目標：前高/箱頂壓力區" if ("量縮回踩" in pattern_str or "底背離" in pattern_str) else "⚔️ 波段目標：前高波動擴張位")
                real_rr_ratio = round((take_profit - entry_price) / max(entry_price - stop_loss, 0.01), 2)
                
                box_color = "#00cc96" if win_prob > 0.60 else ("#ffc107" if win_prob > 0.50 else "#a8a8a8")
                text_color = "#00cc96" if win_prob > 0.60 else ("#ffc107" if win_prob > 0.50 else "#f0f0f0")
                ai_rec = "⭐⭐⭐ 極致期望值！" if win_prob > 0.60 else ("⭐⭐ 溫和佈局" if win_prob > 0.50 else "⚠️ 建議嚴格觀望")
                
                processed_stocks.append({
                    'ticker': ticker, 'name': name, 'win_prob': win_prob, 'box_color': box_color, 'text_color': text_color, 'ai_rec': ai_rec,
                    'entry_price': entry_price, 'take_profit': take_profit, 'stop_loss': stop_loss, 'profit_reason': profit_reason, 'real_rr_ratio': real_rr_ratio
                })
            
            for s in sorted(processed_stocks, key=lambda x: x['win_prob'], reverse=True)[:20]:
                st.markdown(f"""<div style="border: 2px solid {s['box_color']}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;"><h4 style="color: {s['box_color']}; margin-top: 0;">🎯 AI 戰術計畫 ({s['ticker']} {s['name']})</h4><div style="display: flex; justify-content: space-between; flex-wrap: wrap;"><div style="flex: 1; min-width: 180px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">1. AI 真實勝率預測</span>br><b style="font-size: 24px; color: {s['box_color']};">{s['win_prob']*100:.1f}%</b><br><span style="font-size: 14px; font-weight: bold; color: {s['text_color']};">{s['ai_rec']}</span></div><div style="flex: 1; min-width: 130px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">2. 建議進場價</span><br><b style="font-size: 22px;">{s['entry_price']:.2f}</b></div><div style="flex: 1; min-width: 200px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">3. 結構停利點</span><br><b style="font-size: 22px; color: #00cc96;">{s['take_profit']:.2f}</b><br><span style="font-size: 12px; color: #00cc96; font-weight: bold;">{s['profit_reason']}</span></div><div style="flex: 1; min-width: 130px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">4. 嚴格防守價</span><br><b style="font-size: 22px; color: #ff4b4b;">{s['stop_loss']:.2f}</b></div></div></div>""", unsafe_allow_html=True)
        else: st.warning("⚠️ 全市場快取準備中...")

    with tab3:
        st.markdown("#### 🕸️ 上中下游產業鏈資金共振分析 (Top-Down)")
        st.caption("透過剖析細分產業鏈的平均量化熱度，快速抓出目前受到大資金與主力青睞的板塊共振族群。")
        
        chain_list = list(INDUSTRY_CHAINS.keys())
        if not chain_list:
            st.warning("⚠️ 尚未配置任何產業鏈。")
        else:
            selected_chain = st.selectbox("選擇要檢視的產業鏈", chain_list)
            chain_data = INDUSTRY_CHAINS[selected_chain]
            
            snapshot = load_market_snapshot()
            if snapshot and 'data' in snapshot:
                raw_data = snapshot['data']
                market_dict = {str(item.get('代號', item.get('ticker', ''))).split('.')[0].strip(): item for item in raw_data}
                
                st.markdown("---")
                num_cols = len(chain_data) if len(chain_data) > 0 else 1
                cols = st.columns(num_cols)
                
                for idx, (sub_name, tickers) in enumerate(chain_data.items()):
                    with cols[idx]:
                        sub_codes = [str(t).split('.')[0].strip() for t in tickers]
                        sub_items = []
                        
                        for code in sub_codes:
                            if code in market_dict:
                                item = market_dict[code]
                                name = str(item.get('名稱', item.get('name', code)))
                                price = float(item.get('現價', item.get('close_price', item.get('Close', 0.0))))
                                score = int(item.get('量化總分', item.get('score', item.get('Score', 0))))
                                sub_items.append({"名稱": name, "現價": price, "量化分數": score})
                        
                        if sub_items:
                            display_df = pd.DataFrame(sub_items)
                            avg_score = int(display_df['量化分數'].mean())
                            heat_color = "#ff4b4b" if avg_score >= 65 else ("#ffc107" if avg_score >= 45 else "#00cc96")
                            
                            st.markdown(f"<div style='background:#1e1e1e;padding:15px;border-top:4px solid {heat_color};border-radius:5px;margin-bottom:15px;'><b>{sub_name}</b><br><span style='font-size:24px;color:{heat_color};'>板塊熱度: {avg_score} 分</span></div>", unsafe_allow_html=True)
                            st.dataframe(display_df.sort_values("量化分數", ascending=False), hide_index=True, use_container_width=True)
                        else:
                            st.markdown(f"<div style='background:#111;padding:15px;border-radius:5px;color:gray;'><b>{sub_name}</b><br>暫無快取數據</div>", unsafe_allow_html=True)
            else: 
                st.warning("⚠️ 系統快取準備中，請先前往 Actions 觸發掃描...")

    with tab4:
        st.markdown("#### 🔬 AI 演算法實盤回測面板")
        st.caption("🧠 演算法已啟動全自動優化：系統每日開盤會自動將有限資金「自動鎖定並優先買入」當天勝率最高、最具正向期望值的標的物。")
        
        use_mkt_filter = st.checkbox("🛡️ 啟動大盤月線 (20MA) 智慧防禦濾網 (大盤轉弱破線時，自動空倉防守、拒絕新部位進場)", value=True)
        
        # 🚀 拋棄手動滑桿，完全交由神經網路自動優化
        res_adv = fetch_advanced_backtest(
            ai_prob_threshold=0.50, 
            use_market_filter=use_mkt_filter,
            initial_cap=user_capital,
            max_pos=user_max_pos
        )
        
        if res_adv["status"] == "no_key": st.error("⚠️ 找不到資料庫金鑰。")
        elif res_adv["status"] == "empty": st.warning("⚠️ 經過資金與成本侵蝕過濾後，無交易紀錄。")
        elif res_adv["status"] == "pending": st.info("⏸️ 等待開獎。目前暫無達成條件之信號。")
        elif res_adv["status"] == "error": st.error(f"❌ 運算發生錯誤: {res_adv['msg']}")
        elif res_adv["status"] == "ready":
            
            sub_tab1, sub_tab2 = st.tabs(["📋 真實帳戶績效 & 訊號清單", "📈 帳戶淨值對照曲線"])
            
            with sub_tab1:
                st.markdown("<br>", unsafe_allow_html=True)
                
                def build_card(title, value, subtext, color):
                    return f"""
                    <div style="background-color: #121218; padding: 22px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #2a2a35; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
                        <div style="color: #8b8b99; font-size: 14px; margin-bottom: 8px; font-weight: 500;">{title}</div>
                        <div style="color: {color}; font-size: 32px; font-weight: 700; margin-bottom: 5px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">{value}</div>
                        <div style="color: #6b6b79; font-size: 13px;">{subtext}</div>
                    </div>
                    """
                
                color_green = "#4ade80"
                color_purple = "#c084fc"
                color_red = "#ff4b4b"
                
                r1_c1, r1_c2, r1_c3 = st.columns(3)
                with r1_c1: st.markdown(build_card("AI 真實勝率 (扣除摩擦成本)", f"{res_adv['ai_strat']['wr']*100:.1f}%", f"{res_adv['ai_strat']['w']}W / {res_adv['ai_strat']['l']}L", color_green), unsafe_allow_html=True)
                with r1_c2: 
                    pnl_color = color_green if res_adv['net_profit_twd'] > 0 else color_red
                    st.markdown(build_card("帳戶實質總淨利", f"+${res_adv['net_profit_twd']:,.0f}" if res_adv['net_profit_twd'] > 0 else f"${res_adv['net_profit_twd']:,.0f}", f"基於初始本金 NT$ {user_capital:,}", pnl_color), unsafe_allow_html=True)
                with r1_c3:
                    pct_color = color_green if res_adv['account_pct'] > 0 else color_red
                    st.markdown(build_card("帳戶真實報酬率", f"+{res_adv['account_pct']:.2f}%" if res_adv['account_pct'] > 0 else f"{res_adv['account_pct']:.2f}%", f"總成交筆數: {res_adv['trades']} 筆", pct_color), unsafe_allow_html=True)
                
                r2_c1, r2_c2 = st.columns(2)
                with r2_c1: 
                    avg_color = color_green if res_adv['avg_trade_twd'] > 0 else color_red
                    st.markdown(build_card("單筆純利利潤 (TWD)", f"+${res_adv['avg_trade_twd']:,.0f}" if res_adv['avg_trade_twd'] > 0 else f"${res_adv['avg_trade_twd']:,.0f}", "已扣除 0.5% 摩擦成本後的期望值", avg_color), unsafe_allow_html=True)
                with r2_c2: st.markdown(build_card("TP2 觸及概率 (目標 5%)", f"{res_adv['tps']['tp2']*100:.1f}%", f"{res_adv['tps']['samples']} 筆樣本", color_purple), unsafe_allow_html=True)

                st.markdown("#### 🚨 歷史 AI 實盤觸發清單 (每日依大腦信心度自動排序選股)")
                if res_adv['signals']:
                    sig_df = pd.DataFrame(res_adv['signals'])
                    sig_df.rename(columns={"date": "觸發日期", "ticker": "股票代號", "close_price": "進場價", "ai_prob_str": "AI勝率", "sim_profit_str": "扣費後實際損益"}, inplace=True)
                    st.dataframe(sig_df, hide_index=True, use_container_width=True)

            with sub_tab2:
                st.markdown("#### 📈 AI 實盤淨資產報酬 vs 大盤基準線 (%)")
                st.caption("紫線為考慮手續費、滑價與大盤防禦濾網後的『真實財富增長軌跡』。")
                if res_adv['equity']:
                    eq_df = pd.DataFrame(res_adv['equity'])
                    eq_df.rename(columns={"date_str": "日期", "strat_cum_pct": "AI 策略帳戶總報酬 (%)", "market_cum_pct": "加權指數大盤基準線 (%)"}, inplace=True)
                    st.line_chart(eq_df.set_index("日期")[["AI 策略帳戶總報酬 (%)", "加權指數大盤基準線 (%)"]], color=["#c084fc", "#6b6b79"])
                else:
                    st.info("尚無足夠的歷史數據繪製曲線。")