# backend_updater.py — 終極極速批量版 (Batch Mode)
import pandas as pd
import numpy as np
import json
import datetime
import os
import time
import yfinance as yf
from supabase import create_client, Client

# 從 data_fetcher 引入基本工具
from data_fetcher import load_all_market_tickers, get_precalculated_market_ret, _normalize_ticker
from indicators import add_advanced_indicators
from config import DEFAULT_NAMES

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
    # 1. 整理出有效的 Yahoo Ticker 列表
    for _, row in csv_df.iterrows():
        raw_ticker = str(row['Ticker']).strip()
        clean, tw, two = _normalize_ticker(raw_ticker)
        if clean not in names_dict:
            names_dict[clean] = str(row['Name'])
        
        # 依照 CSV 原本的後綴來決定 Yahoo 查詢代碼
        y_ticker = two if raw_ticker.endswith(".TWO") else tw
        valid_tickers.append((y_ticker, clean))

    print(f"📊 預計批量掃描 {len(valid_tickers)} 檔標的 (自動過濾下市與 ETF)...")
    
    final_data = []
    chunk_size = 300 # 將全市場切分成每批 300 檔，避免 URL 過長引發錯誤
    
    for i in range(0, len(valid_tickers), chunk_size):
        chunk = valid_tickers[i:i+chunk_size]
        y_tickers_chunk = [t[0] for t in chunk]
        ticker_map = {t[0]: t[1] for t in chunk}
        
        print(f"   ⏳ 正在高速下載第 {i+1} ~ {min(i+chunk_size, len(valid_tickers))} 檔...")
        
        try:
            # 🔥 核心修正：使用批量下載，yf.download 在底層會打包請求，完全避開 429 限流
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
                    # 處理 YFinance 批量回傳的 MultiIndex 結構
                    if isinstance(data.columns, pd.MultiIndex):
                        if y_ticker in data.columns.get_level_values(0):
                            df_stock = data[y_ticker].copy()
                        else:
                            continue
                    else:
                        # 只有單檔股票成功時的 Flat Dataframe 防護
                        df_stock = data.copy()

                    df_stock = df_stock.dropna(subset=['Close', 'Volume'])
                    if df_stock.empty or len(df_stock) < 20:
                        continue
                        
                    # 執行指標運算
                    df_stock = add_advanced_indicators(df_stock, market_ret_20)
                    
                    c = float(df_stock['Close'].iloc[-1])
                    v = float(df_stock['Volume'].iloc[-1])
                    if c <= 0 or v <= 0:
                        continue
                        
                    bull_div   = bool(df_stock['Bullish_Div'].iloc[-1])
                    liq_sweep  = bool(df_stock['Liquidity_Sweep_Bull'].iloc[-1])
                    low_vol_pb = bool(df_stock['Low_Vol_Pullback'].iloc[-1])
                    squeeze_on = bool(df_stock['Squeeze_On'].iloc[-1])

                    pattern_list = []
                    if low_vol_pb: pattern_list.append("📉 量縮回踩")
                    if squeeze_on: pattern_list.append("🛡️ 區間壓縮")
                    if liq_sweep:  pattern_list.append("🌊 流動性掠奪")
                    if bull_div:   pattern_list.append("🟢 RSI底背離")
                    pattern_str = " + ".join(pattern_list) if pattern_list else "常態震盪"

                    final_data.append({
                        "代號":         clean_ticker,
                        "名稱":         names_dict.get(clean_ticker, clean_ticker),
                        "現價":         c,
                        "成交量":       v,
                        "Res_20":       float(df_stock['Res_20'].iloc[-1]),
                        "Sup_20":       float(df_stock['Sup_20'].iloc[-1]),
                        "ATR_14":       float(df_stock['ATR_14'].iloc[-1]),
                        "rs_index":     float(df_stock['RS_Index'].iloc[-1]),
                        "vol_ratio":    float(df_stock['Vol_Ratio'].iloc[-1]),
                        "volatility":   float(df_stock['ATR_14'].iloc[-1] / c),
                        "turnover":     float(c * v),
                        "broker_conc":  float(df_stock['Broker_Concentration'].iloc[-1]),
                        "pattern":      pattern_str,
                        "recent_returns": df_stock['Close'].pct_change().fillna(0).tail(10).tolist(),
                        "score":        int(df_stock['Score'].iloc[-1])
                    })
                except Exception as e:
                    # 單檔若出錯，不影響其他股票
                    continue
                    
            # 批次之間微幅暫停
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ 批次下載發生異常: {e}")
            time.sleep(2)

    is_fresh_data = True
    if len(final_data) < 50:
        is_fresh_data = False
        print("⚠️ 抓取到的有效標的過少。嘗試保留舊有快取資料以供網頁顯示...")
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
        print("🛑 因為使用舊快取，跳過資料庫寫入，保護機器學習記憶的純潔性。")
        return

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if supabase_url and supabase_key:
        try:
            print("🔗 正在封裝數據至 Supabase 歷史記憶庫...")
            supabase: Client = create_client(supabase_url, supabase_key)
            db_records = []

            for item in final_data:
                db_records.append({
                    "date":        today_str,
                    "ticker":      str(item.get("代號")),
                    "name":        str(item.get("名稱", "")),
                    "pattern":     str(item.get("pattern", "")),
                    "close_price": float(item.get("現價", 0)),
                    "rs_index":    float(item.get("rs_index", 0.0)),
                    "volatility":  float(item.get("volatility", 0.0)),
                    "turnover":    float(item.get("turnover", 0.0)),
                    "vol_ratio":   float(item.get("vol_ratio", 1.0)),
                    "broker_conc": float(item.get("broker_conc", 0.0)),
                    "score":       int(item.get("score", 0))
                })

            if db_records:
                for i in range(0, len(db_records), 500):
                    supabase.table("quant_history").insert(db_records[i:i+500]).execute()
                print(f"🎉 歷史記憶已成功封裝！今日共寫入 {len(db_records)} 筆強勢特徵。")
        except Exception as e:
            print(f"❌ 寫入 Supabase 失敗：{e}")

if __name__ == "__main__":
    run_backend_update()