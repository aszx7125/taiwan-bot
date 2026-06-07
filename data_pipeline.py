import os
import json
import requests
import pandas as pd
import streamlit as st
from data_fetcher import fetch_yahoo_robust

def load_market_snapshot():
    """讀取每日全市場快取資料"""
    if os.path.exists("market_snapshot.json"):
        try:
            with open("market_snapshot.json", "r", encoding="utf-8") as f: return json.load(f)
        except Exception: return None
    return None

def get_snapshot_dict(snapshot):
    """將清單轉換為以股票代號為 Key 的字典"""
    if snapshot and 'data' in snapshot:
        return {str(item.get('代號', item.get('ticker', ''))).split('.')[0].strip(): item for item in snapshot['data']}
    return {}

def get_realtime_quote(clean_ticker, api_key):
    """向 Fugle 或 Yahoo 抓取即時報價 (強固防崩潰版)"""
    rt_price, rt_vol, prev_close = 0.0, 0.0, 0.0
    
    # 1. 優先 Fugle API
    if api_key:
        try:
            res = requests.get(f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{clean_ticker}", headers={"X-API-KEY": api_key}, timeout=2)
            if res.status_code == 200:
                data = res.json()
                rt_price = data.get('closePrice') or data.get('lastTrade', {}).get('price', 0.0)
                prev_close = data.get('previousClose') or data.get('referencePrice', 0.0)
                rt_vol = data.get('total', {}).get('tradeVolume', 0.0)
                if rt_price: rt_price = float(rt_price)
                if prev_close: prev_close = float(prev_close)
                if rt_vol: rt_vol = float(rt_vol)
        except Exception: 
            pass

    # 2. 備用方案：Yahoo Finance (配合 data_fetcher 新防摔安全氣囊)
    if not rt_price or rt_price == 0:
        try:
            df = fetch_yahoo_robust(f"{clean_ticker}.TW", period="5d", interval="1d")
            if df.empty: 
                df = fetch_yahoo_robust(f"{clean_ticker}.TWO", period="5d", interval="1d")
                
            if not df.empty and len(df) >= 2:
                c, p = df.iloc[-1], df.iloc[-2]
                rt_price, prev_close = float(c['Close']), float(p['Close'])
                rt_vol = float(c.get('Volume', 0.0))
        except Exception: 
            pass
            
    return rt_price, rt_vol, prev_close

def trigger_github_workflow(workflow_filename):
    """手動觸發 GitHub Action 進行訓練或掃描"""
    token = st.secrets.get("GH_PAT")
    repo = "aszx7125/taiwan-bot"
    if not token:
        return False, "缺少 GitHub PAT 金鑰設定 (請在 Streamlit Secrets 設定 GH_PAT)"
    
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_filename}/dispatches"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"ref": "main"}
    
    res = requests.post(url, headers=headers, json=data)
    if res.status_code == 204:
        return True, "🚀 指令已成功發送至 GitHub 虛擬工廠！請等待幾分鐘執行。"
    else:
        return False, f"發送失敗: {res.text}"

def load_model_metrics():
    """讀取訓練時產生的盲測勝率報告"""
    if os.path.exists("model_metrics.json"):
        try:
            with open("model_metrics.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "lgbm": {"blind_win_rate": 0.585, "last_train": "等待排程更新"},
        "lstm": {"blind_win_rate": 0.542, "last_train": "等待排程更新"}
    }

@st.cache_data(ttl=7200, show_spinner=False) 
def fetch_advanced_backtest(ai_prob_threshold=0.50, use_market_filter=True, initial_cap=1000000, max_pos=5):
    """執行實盤自動優化回測運算"""
    try:
        from supabase import create_client
        import joblib
        url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
        if not url or not key: return {"status": "no_key"}
        
        if not os.path.exists("quant_model.joblib"): return {"status": "error", "msg": "找不到 LightGBM 大腦模型。"}
        model = joblib.load("quant_model.joblib")
        features = joblib.load("model_features.joblib")
        
        market_k = fetch_yahoo_robust("^TWII", period="3y", interval="1d")
        if market_k.empty: return {"status": "error", "msg": "無法下載大盤。"}
        
        market_k = market_k.sort_index()
        market_k['market_sma20'] = market_k['Close'].rolling(window=20).mean()
        market_k['market_pct'] = market_k['Close'].pct_change()
        market_k = market_k.reset_index()
        market_k['date_norm'] = pd.to_datetime(market_k['index']).dt.normalize()
        market_brief = market_k[['date_norm', 'Close', 'market_sma20', 'market_pct']].rename(columns={'Close': 'market_close'})

        supabase = create_client(url, key)
        all_data, offset, limit = [], 0, 1000
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
        
        for col in ['rs_index', 'volatility', 'turnover', 'broker_conc']:
            df[col] = pd.to_numeric(df.get(col, 0.0), errors='coerce').fillna(0.0)
        df['vol_ratio'] = pd.to_numeric(df.get('vol_ratio', 1.0), errors='coerce').fillna(1.0)

        input_df = df[features].astype(float).fillna(0)
        df['ai_prob'] = model.predict_proba(input_df)[:, 1]
        df = pd.merge(df, market_brief, on='date_norm', how='left')
        df['entry_price_real'] = df.groupby('ticker')['close_price'].shift(-1)
        df['future_close_6d'] = df.groupby('ticker')['close_price'].shift(-6)
        df['return_5d'] = (df['future_close_6d'] - df['entry_price_real']) / df['entry_price_real']

        if use_market_filter: signals = df[(df['ai_prob'] >= ai_prob_threshold) & (df['market_close'] >= df['market_sma20']) & (df['entry_price_real'].notna()) & (df['future_close_6d'].notna())].copy()
        else: signals = df[(df['ai_prob'] >= ai_prob_threshold) & (df['entry_price_real'].notna()) & (df['future_close_6d'].notna())].copy()

        if len(signals) == 0: return {"status": "pending"}

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

        if not executed_trades: return {"status": "empty"}

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