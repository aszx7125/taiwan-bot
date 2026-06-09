import pandas as pd
import numpy as np
import json
import datetime
import os
import time
import yfinance as yf
from supabase import create_client, Client

from data_fetcher import load_all_market_tickers, get_precalculated_market_ret, _normalize_ticker
from indicators import add_advanced_indicators
from config import DEFAULT_NAMES


def safe_float(val, default=0.0):
    """🔥 新增：安全浮點轉換"""
    try:
        f = float(val)
        return np.nan_to_num(f, nan=default, posinf=default, neginf=default)
    except:
        return default


def run_backend_update():
    tz_tw = datetime.timezone(datetime.timedelta(hours=8))
    now_tw = datetime.datetime.now(tz_tw)
    today_str = now_tw.strftime("%Y-%m-%d")
    print(f"[{now_tw.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 啟動極速批量雲端量化引擎 (Yahoo Batch Mode)...")

    csv_df = load_all_market_tickers()
    if csv_df.empty:
        print("❌ 無法讀取全市場代碼表 (all_tw_stocks.csv)。")
        return

    market_ret_20 = get_precalculated_market_ret()
    names_dict = DEFAULT_NAMES.copy()
    
    valid_tickers = []
    for _, row in csv_df.iterrows():
        raw_ticker = str(row['Ticker']).strip()
        clean, tw, two = _normalize_ticker(raw_ticker)
        if clean not in names_dict:
            names_dict[clean] = str(row['Name'])
        
        y_ticker = two if raw_ticker.endswith(".TWO") else tw
        valid_tickers.append((y_ticker, clean))

    print(f"📊 預計批量掃描 {len(valid_tickers)} 檔標的...")
    
    final_data = []
    chunk_size = 300
    inf_count = 0  # 🔥 新增：inf計數器
    
    for i in range(0, len(valid_tickers), chunk_size):
        chunk = valid_tickers[i:i+chunk_size]
        y_tickers_chunk = [t[0] for t in chunk]
        ticker_map = {t[0]: t[1] for t in chunk}
        
        print(f"   ⏳ 正在高速下載第 {i+1} ~ {min(i+chunk_size, len(valid_tickers))} 檔...")
        
        try:
            data = yf.download(
                y_tickers_chunk, 
                period="2mo", 
                interval="1d", 
                group_by='ticker', 
                auto_adjust=True, 
                threads=True, 
                progress=False
            )
            
            if data.empty:
                continue
            
            for y_ticker in y_tickers_chunk:
                clean_ticker = ticker_map[y_ticker]
                
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if y_ticker in data.columns.get_level_values(0):
                            df_stock = data[y_ticker].copy()
                        else:
                            continue
                    else:
                        df_stock = data.copy()

                    df_stock = df_stock.dropna(subset=['Close', 'Volume'])
                    if df_stock.empty or len(df_stock) < 20:
                        continue
                        
                    df_stock = add_advanced_indicators(df_stock, market_ret_20)
                    
                    c = safe_float(df_stock['Close'].iloc[-1])
                    v = safe_float(df_stock['Volume'].iloc[-1])
                    if c <= 0.01 or v <= 0:  # 🔥 修復：提高閾值
                        continue
                        
                    bull_div = bool(df_stock['Bullish_Div'].iloc[-1])
                    liq_sweep = bool(df_stock['Liquidity_Sweep_Bull'].iloc[-1])
                    low_vol_pb = bool(df_stock['Low_Vol_Pullback'].iloc[-1])
                    squeeze_on = bool(df_stock['Squeeze_On'].iloc[-1])

                    pattern_list = []
                    if low_vol_pb: pattern_list.append("📉 量縮回踩")
                    if squeeze_on: pattern_list.append("🛡️ 區間壓縮")
                    if liq_sweep: pattern_list.append("🌊 流動性掠奪")
                    if bull_div: pattern_list.append("🟢 RSI底背離")
                    pattern_str = " + ".join(pattern_list) if pattern_list else "常態震盪"

                    # 🔥 核心修復：所有數值強制清理
                    atr_val = safe_float(df_stock['ATR_14'].iloc[-1])
                    rs_idx = safe_float(df_stock['RS_Index'].iloc[-1])
                    vol_ratio = safe_float(df_stock['Vol_Ratio'].iloc[-1], 1.0)
                    broker_conc = safe_float(df_stock['Broker_Concentration'].iloc[-1])
                    
                    # 🔥 修復#4: volatility除零保護
                    volatility = safe_float(atr_val / max(c, 0.01), 0.05)
                    
                    # 🔥 修復#5: recent_returns清理
                    returns = df_stock['Close'].pct_change().replace([np.inf, -np.inf], 0).fillna(0).tail(10)
                    returns_clean = np.nan_to_num(returns.values, nan=0.0, posinf=0.2, neginf=-0.2).tolist()
                    
                    # 檢查inf
                    if np.isinf(volatility) or np.isnan(volatility):
                        inf_count += 1
                        volatility = 0.05

                    final_data.append({
                        "代號": clean_ticker,
                        "名稱": names_dict.get(clean_ticker, clean_ticker),
                        "現價": c,
                        "成交量": v,
                        "Res_20": safe_float(df_stock['Res_20'].iloc[-1], c * 1.05),
                        "Sup_20": safe_float(df_stock['Sup_20'].iloc[-1], c * 0.95),
                        "ATR_14": atr_val,
                        "rs_index": rs_idx,
                        "vol_ratio": vol_ratio,
                        "volatility": volatility,
                        "turnover": safe_float(c * v),
                        "broker_conc": broker_conc,
                        "pattern": pattern_str,
                        "recent_returns": returns_clean,
                        "score": int(np.clip(safe_float(df_stock['Score'].iloc[-1], 50), 0, 100))
                    })
                except Exception as e:
                    continue
                    
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ 批次下載發生異常: {e}")
            time.sleep(2)

    # 🔥 新增：數據品質報告
    if inf_count > 0:
        print(f"⚠️ 警告：清理了 {inf_count} 筆inf數據")

    is_fresh_data = True
    if len(final_data) < 50:
        is_fresh_data = False
        print("⚠️ 抓取到的有效標的過少。嘗試保留舊有快取資料...")
        if os.path.exists("market_snapshot.json"):
            try:
                with open("market_snapshot.json", "r", encoding="utf-8") as f:
                    final_data = json.load(f).get("data", [])
            except Exception:
                pass

    if not final_data:
        print("❌ 無法取得任何資料，系統終止。")
        return

    with open("market_snapshot.json", "w", encoding="utf-8") as f:
        json.dump({
            "update_time": now_tw.strftime("%Y-%m-%d %H:%M:%S"), 
            "data": final_data
        }, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ 快取更新成功！共完美萃取 {len(final_data)} 檔標的。")

    if not is_fresh_data:
        print("🛑 因為使用舊快取，跳過資料庫寫入。")
        return

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if supabase_url and supabase_key:
        try:
            print("🔗 正在封裝數據至 Supabase...")
            supabase: Client = create_client(supabase_url, supabase_key)
            db_records = []

            for item in final_data:
                db_records.append({
                    "date": today_str,
                    "ticker": str(item.get("代號")),
                    "name": str(item.get("名稱", "")),
                    "pattern": str(item.get("pattern", "")),
                    "close_price": safe_float(item.get("現價", 0)),
                    "rs_index": safe_float(item.get("rs_index", 0.0)),
                    "volatility": safe_float(item.get("volatility", 0.0)),
                    "turnover": safe_float(item.get("turnover", 0.0)),
                    "vol_ratio": safe_float(item.get("vol_ratio", 1.0)),
                    "broker_conc": safe_float(item.get("broker_conc", 0.0)),
                    "score": int(item.get("score", 0))
                })

            if db_records:
                for i in range(0, len(db_records), 500):
                    supabase.table("quant_history").insert(db_records[i:i+500]).execute()
                print(f"🎉 歷史記憶已成功封裝！今日共寫入 {len(db_records)} 筆。")
        except Exception as e:
            print(f"❌ 寫入 Supabase 失敗：{e}")


if __name__ == "__main__":
    run_backend_update()