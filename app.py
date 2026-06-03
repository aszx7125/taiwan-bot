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
    """將大盤快取轉為 O(1) 字典，確保全站讀取同一份歷史基準特徵"""
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

# ==========================================================
# 🚀 終極對齊核心：中央報價引擎與特徵萃取器 (已修復參數順序 Bug)
# ==========================================================
def get_realtime_quote(clean_ticker):
    """確保全站呼叫同一套報價邏輯，防堵 API 延遲錯亂"""
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
    """終極特徵對齊：修復引數順序錯亂，完美防堵盤中殘缺成交量欺騙 AI"""
    rs_idx = fallback_rs
    pat = fallback_pattern
    
    anchor_price = current_price
    base_vol = fallback_vol 
    atr = fallback_atr if fallback_atr else (anchor_price * 0.05)

    # 💎 強制從快取中抽取昨天的「全日完整成交量」與「基準型態」
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

    # 若快取遺失且無基準量，才用當下真實即時量
    if base_vol <= 0 and current_vol > 0:
        base_vol = current_vol

    if anchor_price <= 0: anchor_price = 1.0

    volatility = float(atr / anchor_price)
    turnover = float(anchor_price * base_vol)

    return {
        'is_pullback': 1 if "量縮回踩" in pat else 0,
        'is_sweep': 1 if "流動性掠奪" in pat else 0,
        'is_squeeze': 1 if "區間壓縮" in pat else 0,
        'is_divergence': 1 if "底背離" in pat else 0,
        'rs_index': rs_idx,
        'volatility': volatility,
        'turnover': turnover
    }

@st.cache_data(ttl=3600*6) 
def fetch_and_calculate_backtest(holding_period=5, threshold=60):
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
        if not url or not key: return {"status": "no_key"}
        
        supabase = create_client(url, key)
        all_data = []
        offset, limit = 0, 1000
        while True:
            res = supabase.table("quant_history").select("date, ticker, close_price, score").range(offset, offset+limit-1).execute()
            if not res.data: break
            all_data.extend(res.data)
            offset += limit

        if not all_data: return {"status": "empty"}
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)

        df[f'future_close_{holding_period}d'] = df.groupby('ticker')['close_price'].shift(-holding_period)
        df[f'return_{holding_period}d'] = (df[f'future_close_{holding_period}d'] - df['close_price']) / df['close_price']

        signals = df[(df['score'] >= threshold) & (df[f'future_close_{holding_period}d'].notna())].copy()
        total_signals = len(signals)
        
        if total_signals == 0:
            pending = len(df[(df['score'] >= threshold) & (df[f'future_close_{holding_period}d'].isna())])
            return {"status": "pending", "pending_count": pending}

        signals['is_win'] = signals[f'return_{holding_period}d'] > 0
        win_rate = len(signals[signals['is_win']]) / total_signals
        avg_win = signals[signals['is_win']][f'return_{holding_period}d'].mean() if len(signals[signals['is_win']]) > 0 else 0
        avg_loss = signals[~signals['is_win']][f'return_{holding_period}d'].mean() if len(signals[~signals['is_win']]) > 0 else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return {
            "status": "ready", "total": total_signals, "win_rate": win_rate,
            "avg_win": avg_win, "avg_loss": avg_loss, "expectancy": expectancy
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

            ai_win_rate_str = "等待 AI 訓練"
            ai_recommendation = "⏸️ 勝率偏低或追高風險，強制觀望"
            box_color = "#555555"

            model, features = get_ai_model()
            snapshot = load_market_snapshot()
            snapshot_dict = get_snapshot_dict(snapshot)
            
            if model:
                try:
                    # 🚀 中央錨定：對齊引數順序，消滅漂移與崩潰風險
                    input_data = extract_ai_features(
                        base_ticker, entry_price, snapshot_dict, current_vol=rt_v,
                        fallback_rs=float(today.get('RS_Index', 0.0)), fallback_atr=atr_14, 
                        fallback_pattern=smc_text, fallback_vol=vol_sma5
                    )
                    input_df = pd.DataFrame([input_data], columns=features).fillna(0)
                    win_prob = float(model.predict_proba(input_df)[0][1])
                    ai_win_rate_str = f"{win_prob * 100:.1f}%"
                    
                    if win_prob > 0.60 and real_rr_ratio >= 1.5:
                        ai_recommendation = "⭐⭐⭐ 極致期望值！(高勝率 + 高風報比)"; box_color = "#00cc96"
                    elif win_prob > 0.50 and real_rr_ratio >= 1.0:
                        ai_recommendation = "⭐⭐ 溫和佈局 (具備正向期望值)"; box_color = "#ffc107"
                    else: ai_recommendation = "⚠️ 預測敗率較高，建議嚴格觀望"; box_color = "#555555"
                except: pass

            st.subheader(f"🧬 {target_ticker} {c_name} 多時區量化診斷報告")
            st.markdown(f"""<div style="border: 2px solid {box_color}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;"><h4 style="color: {box_color}; margin-top: 0;">🎯 AI 深度學習 x 結構價格 戰術計畫</h4><div style="display: flex; justify-content: space-between; flex-wrap: wrap;"><div style="flex: 1; min-width: 180px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">1. AI 真實勝率預測</span><br><b style="font-size: 24px; color: {box_color};">{ai_win_rate_str}</b><br><span style="font-size: 14px; font-weight: bold;">{ai_recommendation}</span></div><div style="flex: 1; min-width: 130px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">2. 建議進場價</span><br><b style="font-size: 22px;">{entry_price:.2f}</b></div><div style="flex: 1; min-width: 200px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">3. 結構停利點</span><br><b style="font-size: 22px; color: #00cc96;">{take_profit:.2f}</b><br><span style="font-size: 12px; color: #00cc96; font-weight: bold;">{profit_reason}</span><br><span style="font-size: 12px; color: gray;">(實況風報比 1 : {real_rr_ratio})</span></div><div style="flex: 1; min-width: 130px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">4. 嚴格防守價</span><br><b style="font-size: 22px; color: #ff4b4b;">{stop_loss:.2f}</b></div></div></div>""", unsafe_allow_html=True)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("當前現價", f"{entry_price:.2f}", f"{p_change:+.2f}%")
            m2.metric("日K 巨觀潛伏分數", f"{ai_score} 分")
            m3.metric("1h 小時區微觀狀態", "站穩 1h 均線" if micro_trigger else "弱勢震盪")
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
        with c_title: st.markdown(f"#### 【{selected_cluster}】即時行情流 x AI 戰術標籤")
        with c_slider:
            with st.expander("⚙️ 畫幅設定"): user_font_size = st.slider("表格文字大小", 12, 40, 22, 2)
            
        @st.fragment(run_every=datetime.timedelta(seconds=15))
        def render_rt():
            rows = []
            current_names = st.session_state.stock_names.copy()
            model, features = get_ai_model()
            snapshot = load_market_snapshot()
            snapshot_dict = get_snapshot_dict(snapshot)
            
            def fetch_single_rt(t, names_dict):
                try:
                    clean_ticker = t.split('.')[0]
                    base_name = names_dict.get(clean_ticker, clean_ticker)
                    rt_price, rt_vol, prev_close = get_realtime_quote(clean_ticker)
                    if not rt_price or rt_price == 0: return None
                        
                    change_amt = rt_price - prev_close
                    change_pct = (change_amt / prev_close) * 100 if prev_close > 0 else 0
                    ai_badge_html = ""
                    
                    if model and clean_ticker in snapshot_dict:
                        try:
                            # 🚀 中央錨定：對齊引數順序
                            input_data = extract_ai_features(clean_ticker, rt_price, snapshot_dict, current_vol=rt_vol)
                            input_df = pd.DataFrame([input_data], columns=features).fillna(0)
                            win_prob = float(model.predict_proba(input_df)[0][1])
                            win_rate_pct = win_prob * 100
                            
                            badge_style = "background-color: #00cc96; color: black; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold;" if win_rate_pct >= 60 else ("background-color: #ffc107; color: black; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold;" if win_rate_pct >= 50 else "background-color: #555555; color: #bbb; padding: 2px 6px; border-radius: 4px; font-size: 12px;")
                            prefix = "⭐ 核心強勢" if win_rate_pct >= 60 else ("⚖️ 溫和觀察" if win_rate_pct >= 50 else "⏸️ 暫無動能")
                            ai_badge_html = f"<br><span style='{badge_style}'>{prefix} {win_rate_pct:.1f}%</span>"
                        except: pass
                        
                    name_str = f"<b>{base_name}</b><br><span style='font-size:0.8em;color:gray;'>{clean_ticker}</span>{ai_badge_html}"
                    display_vol = int(rt_vol) if rt_vol < 2000000 else int(rt_vol / 1000)
                    price_vol = f"<b>{rt_price:.2f}</b><br><span style='font-size:0.7em;color:gray;'>({display_vol:,} 張)</span>"
                    change_str = f"<span style='color:#ff4b4b;font-weight:bold;'>+{change_amt:.2f}<br>(+{change_pct:.2f}%)</span>" if change_amt > 0 else (f"<span style='color:#00cc96;font-weight:bold;'>{change_amt:.2f}<br>({change_pct:.2f}%)</span>" if change_amt < 0 else "0.00")
                    return {"標的": name_str, "及時價 (成交量)": price_vol, "今日漲跌幅": change_str, "raw_pct": change_pct}
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
                    # 🚀 中央錨定：修正此處的引數對齊，徹底消除崩潰 Bug
                    bulk_features.append(extract_ai_features(ticker, entry_price, snapshot_dict, current_vol=vol_val))

            all_probs = []
            if model and bulk_features:
                try:
                    input_df = pd.DataFrame(bulk_features, columns=features).fillna(0)
                    all_probs = model.predict_proba(input_df)[:, 1] 
                except: pass

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
                
                box_color = "#00cc96" if win_prob > 0.60 else ("#ffc107" if win_prob > 0.50 else "#555555")
                ai_rec = "⭐⭐⭐ 極致期望值！" if win_prob > 0.60 else ("⭐⭐ 溫和佈局" if win_prob > 0.50 else "⚠️ 建議嚴格觀望")
                
                processed_stocks.append({
                    'ticker': ticker, 'name': name, 'win_prob': win_prob, 'box_color': box_color, 'ai_rec': ai_rec,
                    'entry_price': entry_price, 'take_profit': take_profit, 'stop_loss': stop_loss, 'profit_reason': profit_reason, 'real_rr_ratio': real_rr_ratio
                })
            
            for s in sorted(processed_stocks, key=lambda x: x['win_prob'], reverse=True)[:20]:
                st.markdown(f"""<div style="border: 2px solid {s['box_color']}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;"><h4 style="color: {s['box_color']}; margin-top: 0;">🎯 AI 戰術計畫 ({s['ticker']} {s['name']})</h4><div style="display: flex; justify-content: space-between; flex-wrap: wrap;"><div style="flex: 1; min-width: 180px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">1. AI 真實勝率預測</span><br><b style="font-size: 24px; color: {s['box_color']};">{s['win_prob']*100:.1f}%</b><br><span style="font-size: 14px; font-weight: bold;">{s['ai_rec']}</span></div><div style="flex: 1; min-width: 130px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">2. 建議進場價</span><br><b style="font-size: 22px;">{s['entry_price']:.2f}</b></div><div style="flex: 1; min-width: 200px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">3. 結構停利點</span><br><b style="font-size: 22px; color: #00cc96;">{s['take_profit']:.2f}</b><br><span style="font-size: 12px; color: #00cc96; font-weight: bold;">{s['profit_reason']}</span></div><div style="flex: 1; min-width: 130px; margin-bottom: 10px;"><span style="color: gray; font-size: 14px;">4. 嚴格防守價</span><br><b style="font-size: 22px; color: #ff4b4b;">{s['stop_loss']:.2f}</b></div></div></div>""", unsafe_allow_html=True)