import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random 
import time
import concurrent.futures

# 📦 導入模組
from config import get_fugle_key, DEFAULT_CLUSTERS, DEFAULT_NAMES, INDUSTRY_CHAINS
from data_fetcher import load_all_market_tickers, get_market_summary, get_kline_with_fugle, get_stock_news
from data_pipeline import load_market_snapshot, get_snapshot_dict, get_realtime_quote, fetch_advanced_backtest, trigger_github_workflow, load_model_metrics
from ai_engine import DualCoreBrain
from ui_components import render_top20_card, render_single_diagnostic_card, render_backtest_metric_card, render_model_health_board

st.set_page_config(page_title="台股量化旗艦終端", page_icon="📈", layout="wide")

FUGLE_API_KEY = get_fugle_key()
if 'stock_clusters' not in st.session_state: st.session_state.stock_clusters = DEFAULT_CLUSTERS.copy()
if 'stock_names' not in st.session_state: st.session_state.stock_names = DEFAULT_NAMES.copy()

@st.cache_resource
def load_brain():
    return DualCoreBrain()

brain = load_brain()

# ==========================================================
# 🎛️ 左側控制面板
# ==========================================================
with st.sidebar:
    st.header("📂 我的自選清單")
    selected_cluster = st.selectbox("1. 選擇產業群組", list(st.session_state.stock_clusters.keys()))
    cluster_stocks = st.session_state.stock_clusters[selected_cluster]
    display_options = [f"{t.split('.')[0]} {st.session_state.stock_names.get(t.split('.')[0], '')}".strip() for t in cluster_stocks]
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
    if brain.is_lstm_ready: st.success("🔮 LSTM 大腦已連動")
    else: st.warning("⚪ 找不到 LSTM")
    if brain.is_lgbm_ready: st.success("🌳 LightGBM 大腦正常")
    else: st.error("🚨 缺少 LightGBM")

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
    # 🔍 單股深入診斷模式
    base_ticker = target_ticker.split('.')[0]
    c_name = st.session_state.stock_names.get(base_ticker, target_ticker)
    
    with st.spinner(f"正在分析 {target_ticker}..."):
        df_daily, df_hourly, actual_symbol = get_kline_with_fugle(target_ticker, FUGLE_API_KEY)
        if df_daily.empty: st.error("❌ 數據不足")
        else:
            news_s = get_stock_news(c_name)
            today, yesterday = df_daily.iloc[-1], df_daily.iloc[-2]
            entry_price = float(today.get('Close', 0.0))
            y_close = float(yesterday.get('Close', entry_price))
            rt_p, rt_v, _ = get_realtime_quote(base_ticker, FUGLE_API_KEY)
            if rt_p > 0: entry_price = rt_p
            
            p_change = ((entry_price - y_close) / y_close) * 100 if y_close > 0 else 0.0
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
            
            final_prob = brain.predict_win_rates([feat_dict])[0]
            
            box_color = "#00cc96" if final_prob >= 0.52 else ("#ffc107" if final_prob >= 0.50 else "#a8a8a8")
            ai_rec = "⭐⭐⭐ 高期望值" if final_prob >= 0.52 else ("⭐⭐ 溫和佈局" if final_prob >= 0.50 else "⚠️ 建議觀望")
            
            st.subheader(f"🧬 {target_ticker} {c_name} 雙核量化報告")
            render_single_diagnostic_card(f"{final_prob*100:.1f}%", ai_rec, entry_price, take_profit, stop_loss, box_color, box_color)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("當前現價", f"{entry_price:.2f}", f"{p_change:+.2f}%")
            m2.metric("SMC 結構", smc_text)
            m3.metric("1h 微觀狀態", micro_status_text)
            m4.metric("機構集中度", f"{broker_conc*100:.1f}%")
            st.markdown("---")
            
            if st.button("⬅️ 返回戰情室主頁", use_container_width=True):
                st.session_state.analyze_trigger = None; st.rerun()
else:
    # 🏠 滿血旗艦主視覺儀表板
    st.markdown("### 🌍 大盤與情緒摘要")
    summary = get_market_summary()
    if summary:
        twii_data = summary.get("加權指數", {"pct": 0})
        greed_index = int(max(0, min(100, 50 + (twii_data['pct'] * 15) + random.randint(-5, 5))))
        c_idx, c_greed = st.columns([3, 1])
        with c_idx:
            cols = st.columns(len(summary))
            for i, (name, data) in enumerate(summary.items()): cols[i].metric(name, f"{data['price']:.2f}", f"{data['change']:+.2f} ({data['pct']:+.2f}%)")
        with c_greed: st.metric("放眼全球：台股恐懼貪婪指數", f"{greed_index} / 100")
    
    st.markdown("---")
    
    metrics = load_model_metrics()
    render_model_health_board(metrics)
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 自選即時流", "🔮 每日收盤趨勢", "🎯 全市場 TOP 20", "🕸️ 產業鏈資金共振", "⚖️ 實盤開獎對撞", "🔬 策略回測"])
    
    with tab1:
        c_title, c_slider = st.columns([2, 1])
        with c_title: st.markdown(f"#### 【{selected_cluster}】即時行情流")
        with c_slider:
            with st.expander("⚙️ 畫幅設定"): user_font_size = st.slider("表格文字大小", 12, 40, 22, 2)
            
        @st.fragment(run_every=datetime.timedelta(seconds=15))
        def render_rt():
            rows = []
            current_names = st.session_state.stock_names.copy() 
            snapshot = load_market_snapshot()
            snapshot_dict = get_snapshot_dict(snapshot)
            
            def fetch_rt(t):
                ticker = t.split('.')[0]
                p, v, prev = get_realtime_quote(ticker, FUGLE_API_KEY)
                if p > 0: 
                    chg_amt = p - prev
                    chg_pct = (chg_amt/prev)*100 if prev > 0 else 0
                    stock_name = snapshot_dict.get(ticker, {}).get('名稱', current_names.get(ticker, ticker))
                    name_str = f"<b>{stock_name}</b><br><span style='font-size:0.8em;color:gray;'>{ticker}</span>"
                    p_str = f"<b>{p:.2f}</b><br><span style='font-size:0.7em;color:gray;'>({int(v):,} 張)</span>"
                    chg_str = f"<span style='color:#ff4b4b;font-weight:bold;'>+{chg_amt:.2f}<br>(+{chg_pct:.2f}%)</span>" if chg_amt > 0 else (f"<span style='color:#00cc96;font-weight:bold;'>{chg_amt:.2f}<br>({chg_pct:.2f}%)</span>" if chg_amt < 0 else "0.00")
                    return {"標的": name_str, "及時價 (成交量)": p_str, "今日漲跌幅": chg_str}
                return None
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
                future_to_ticker = {ex.submit(fetch_rt, t): t for t in cluster_stocks}
                for future in concurrent.futures.as_completed(future_to_ticker):
                    try:
                        res = future.result()
                        if res: rows.append(res)
                        time.sleep(0.05)
                    except Exception: pass
                    
            if rows: 
                html_table = pd.DataFrame(rows).to_html(escape=False, index=False, border=0).replace('\n', '')
                css = f"<style>.watch-board table {{ width: 100% !important; border-collapse: collapse; }} .watch-board th {{ text-align: center !important; font-size: {max(14, user_font_size-4)}px !important; padding: 10px !important; border-bottom: 2px solid #555 !important; }} .watch-board td {{ text-align: center !important; font-size: {user_font_size}px !important; padding: 16px !important; border-bottom: 1px solid #444 !important; vertical-align: middle !important; }}</style>"
                st.markdown(f'{css}<div class="watch-board">{html_table}</div>', unsafe_allow_html=True)
        render_rt()

    with tab2:
        st.markdown("#### 🔮 每日收盤後大盤特徵與明日趨勢預測")
        snapshot = load_market_snapshot()
        if snapshot and 'data' in snapshot and len(snapshot['data']) > 0:
            raw_list = snapshot['data']
            snapshot_dict = get_snapshot_dict(snapshot)
            bulk_features = []
            for item in raw_list:
                ticker = str(item.get('代號', '')).split('.')[0].strip()
                ep = float(item.get('現價', item.get('close_price', 0.0)))
                if ep > 0:
                    bulk_features.append(brain.extract_features(ticker, ep, snapshot_dict, current_vol=float(item.get('成交量', 0.0))))
            
            if bulk_features:
                # 徹底拋棄量化總分，計算全市場真實雙核勝率
                probs = brain.predict_win_rates(bulk_features)
                bullish_ratio = float(np.mean(np.array(probs) >= 0.50)) * 100
                st.metric("🤖 AI 明日全市場強勢看多標的比率 (勝率>50%)", f"{bullish_ratio:.1f}%")
        else: st.info("ℹ️ 快取中無有效數據。")

    with tab3:
        st.markdown("#### 🎯 全市場 AI 進出場戰術面板 (TOP 20)")
        snapshot = load_market_snapshot()
        if snapshot and 'data' in snapshot:
            raw_list = snapshot['data']
            snapshot_dict = get_snapshot_dict(snapshot)
            valid_items, bulk_features = [], []
            
            for item in raw_list:
                ticker = str(item.get('代號', '')).split('.')[0].strip()
                ep = float(item.get('現價', item.get('close_price', 0.0)))
                if ep > 0: 
                    valid_items.append(item)
                    bulk_features.append(brain.extract_features(ticker, ep, snapshot_dict, current_vol=float(item.get('成交量', 0.0))))
                
            if valid_items:
                probs = brain.predict_win_rates(bulk_features)
                processed = []
                for idx, item in enumerate(valid_items):
                    prob = probs[idx]
                    # 🔥 極度嚴格的防禦網：只允許勝率大於等於 50% 的進入候選池！
                    if prob >= 0.50:
                        ep = float(item.get('現價', 0.0))
                        res = float(item.get('Res_20', ep*1.05))
                        sup = float(item.get('Sup_20', ep*0.95))
                        atr = float(item.get('ATR_14', ep*0.05))
                        
                        sl = round(res * 0.985, 2) if ep > res else round(min(ep - (1.5 * atr), sup * 0.985), 2)
                        tp = round(res + (res - sup), 2) if ep > res else round(res + (atr * 1.0), 2)
                        
                        processed.append({
                            'ticker': item.get('代號'), 'name': item.get('名稱', ''), 'win_prob': prob, 
                            'box_color': "#00cc96" if prob >= 0.52 else "#ffc107", 
                            'ai_rec': "推薦佈局" if prob >= 0.52 else "謹慎試單",
                            'entry_price': ep, 'take_profit': tp, 'stop_loss': sl, 'profit_reason': "波段"
                        })
                
                if processed:
                    for s in sorted(processed, key=lambda x: x['win_prob'], reverse=True)[:20]:
                        render_top20_card(s)
                else:
                    highest = max(probs) if len(probs)>0 else 0
                    st.warning(f"⚠️ **目前全市場無符合高勝率標準 (>50%) 之標的。**\n\nAI 雙核大腦判定當前市場風險極高（全市場最高勝率僅 {highest*100:.1f}%）。\n\n💡 **系統建議：** 強烈建議保持空手，等待大盤出現明確企穩訊號，或點擊側邊欄【啟動全市場 AI 掃描】更新快取。")
        else: st.info("快取中無數據。")

    with tab4:
        st.markdown("#### 🕸️ 上中下游產業鏈資金共振分析")
        chain_list = list(INDUSTRY_CHAINS.keys())
        if chain_list:
            selected_chain = st.selectbox("選擇要檢視的產業鏈", chain_list)
            chain_data = INDUSTRY_CHAINS[selected_chain]
            snapshot = load_market_snapshot()
            if snapshot and 'data' in snapshot:
                market_dict = {str(item.get('代號', '')).split('.')[0].strip(): item for item in snapshot['data']}
                cols = st.columns(len(chain_data) if len(chain_data) > 0 else 1)
                for idx, (sub_name, tickers) in enumerate(chain_data.items()):
                    with cols[idx]:
                        sub_items, sub_feats = [], []
                        for code in [str(t).split('.')[0].strip() for t in tickers]:
                            if code in market_dict:
                                item = market_dict[code]
                                ep = float(item.get('現價', 0.0))
                                sub_items.append({"代號": code, "名稱": str(item.get('名稱', code)), "現價": ep})
                                sub_feats.append(brain.extract_features(code, ep, snapshot_dict, current_vol=float(item.get('成交量', 0.0))))
                        
                        if sub_items:
                            # 產業鏈也全面導入真實勝率計算
                            probs = brain.predict_win_rates(sub_feats)
                            for i in range(len(sub_items)):
                                sub_items[i]['雙核勝率'] = f"{probs[i]*100:.1f}%"
                                sub_items[i]['_prob_raw'] = probs[i]

                            display_df = pd.DataFrame(sub_items)
                            avg_prob = display_df['_prob_raw'].mean()
                            heat_color = "#ff4b4b" if avg_prob >= 0.51 else ("#ffc107" if avg_prob >= 0.49 else "#00cc96")
                            st.markdown(f"<div style='background:#1e1e1e;padding:15px;border-top:4px solid {heat_color};border-radius:5px;margin-bottom:15px;'><b>{sub_name}</b><br><span style='font-size:20px;color:{heat_color};'>平均勝率: {avg_prob*100:.1f}%</span></div>", unsafe_allow_html=True)
                            
                            st.dataframe(display_df.drop(columns=['_prob_raw']).sort_values("雙核勝率", ascending=False), hide_index=True)

    with tab5:
        st.markdown("#### ⚖️ 昨晚 AI 預測 x 今日實盤開獎比對面板")
        if st.button("🔄 執行即時對撞比對", key="run_clash_realtime_btn"):
            snapshot = load_market_snapshot()
            if snapshot and 'data' in snapshot:
                raw_list = snapshot['data']
                snapshot_dict = get_snapshot_dict(snapshot)
                current_names = st.session_state.stock_names.copy()
                
                # 1. 抓出有效標的並算勝率 (抹除舊評分)
                valid_items, bulk_features = [], []
                for item in raw_list:
                    ticker = str(item.get('代號', '')).split('.')[0].strip()
                    ep = float(item.get('現價', item.get('close_price', 0.0)))
                    if ep > 0:
                        valid_items.append(item)
                        bulk_features.append(brain.extract_features(ticker, ep, snapshot_dict, current_vol=float(item.get('成交量', 0.0))))
                
                if not valid_items:
                    st.warning("⚪ 歷史快取中無有效個股資料。")
                else:
                    probs = brain.predict_win_rates(bulk_features)
                    candidates = []
                    for idx, item in enumerate(valid_items):
                        item['win_prob'] = probs[idx]
                        candidates.append(item)
                        
                    # 🔥 2. 只讓勝率大於等於 50% 的高優質標的參與對撞開獎
                    candidates = sorted(candidates, key=lambda x: x['win_prob'], reverse=True)
                    top_candidates = [c for c in candidates if c['win_prob'] >= 0.50][:15]
                    
                    if not top_candidates:
                        highest = max(probs) if len(probs)>0 else 0
                        st.warning(f"⚪ 昨晚快取數據中無勝率達標 (>50%) 的標的（最高僅 {highest*100:.1f}%），AI 強烈建議空手觀望，今日無開獎清單。")
                    else:
                        st.info(f"⏳ 正在調度高併發線路，針對 {len(top_candidates)} 檔高勝率標的進行開獎...")
                        clash_rows = []
                        
                        def fetch_clash(item):
                            ticker = str(item.get('代號', '')).split('.')[0].strip()
                            stock_name = snapshot_dict.get(ticker, {}).get('名稱', current_names.get(ticker, ticker))
                            last_price = float(item.get('現價', 0.0))
                            win_prob = item['win_prob']
                            
                            rt_p, _, _ = get_realtime_quote(ticker, FUGLE_API_KEY)
                            if rt_p > 0 and last_price > 0:
                                clash_pct = ((rt_p - last_price) / last_price) * 100
                                return {
                                    "股票代號": ticker,
                                    "股票名稱": stock_name,
                                    "雙核預測勝率": f"{win_prob*100:.1f}%",
                                    "預測基準價 (昨)": f"${last_price:.2f}",
                                    "實盤即時價 (今)": f"${rt_p:.2f}",
                                    "實盤開獎漲跌": f"{clash_pct:+.2f}%",
                                    "_win_prob_raw": win_prob
                                }
                            return None

                        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
                            future_to_item = {ex.submit(fetch_clash, item): item for item in top_candidates}
                            for future in concurrent.futures.as_completed(future_to_item):
                                try:
                                    res = future.result()
                                    if res: clash_rows.append(res)
                                    time.sleep(0.05)
                                except Exception: pass
                        
                        if clash_rows:
                            clash_df = pd.DataFrame(clash_rows).sort_values(by="_win_prob_raw", ascending=False)
                            st.success("✅ 實盤對撞數據已成功產出！")
                            st.dataframe(clash_df.drop(columns=["_win_prob_raw"]), use_container_width=True, hide_index=True)
                        else:
                            st.error("❌ 無法成功取得今日即時開獎報價。")
            else:
                st.warning("ℹ️ 快取檔案異常或無數據。")
            
    with tab6:
        res_adv = fetch_advanced_backtest(initial_cap=user_capital, max_pos=user_max_pos)
        if res_adv.get("status") == "ready":
            c1, c2, c3 = st.columns(3)
            with c1: render_backtest_metric_card("AI 真實勝率", f"{res_adv['ai_strat']['wr']*100:.1f}%", "", "#4ade80")
            with c2: render_backtest_metric_card("帳戶總淨利", f"${res_adv['net_profit_twd']:,.0f}", "", "#4ade80")
            with c3: render_backtest_metric_card("總報酬", f"{res_adv['account_pct']:.2f}%", "", "#4ade80")
            st.line_chart(pd.DataFrame(res_adv['equity']).set_index("date_str")[["strat_cum_pct", "market_cum_pct"]])