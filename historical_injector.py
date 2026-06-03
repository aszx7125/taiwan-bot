import pandas as pd
import numpy as np
import json
import datetime
import os
import concurrent.futures
from supabase import create_client, Client
from data_fetcher import fetch_yahoo_robust, get_precalculated_market_ret
from indicators import add_advanced_indicators
from config import DEFAULT_CLUSTERS, INDUSTRY_CHAINS, DEFAULT_NAMES

SUPABASE_URL = os.environ.get("SUPABASE_URL", "請填入您的_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "請填入您的_SUPABASE_KEY")

def process_single_ticker(t, days_back, market_ret_20, names_dict, start_date):
    clean_ticker = t.split('.')[0]
    name = names_dict.get(clean_ticker, f"代號 {clean_ticker}")
    db_records = []
    
    try:
        df = fetch_yahoo_robust(f"{clean_ticker}.TW", period="3y", interval="1d")
        if df.empty:
            df = fetch_yahoo_robust(f"{clean_ticker}.TWO", period="3y", interval="1d")
            
        if df.empty or len(df) < 100:
            return clean_ticker, f"⚠️ {clean_ticker} {name} 數據不足，跳過。"

        df = add_advanced_indicators(df, market_ret_20)
        df = df[df.index >= start_date]
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        for date_timestamp, row in df.iterrows():
            date_str = date_timestamp.strftime("%Y-%m-%d")
            
            try: 
                rs_val = float(str(row.get("RS_Index", "0")).replace("%", ""))
                if pd.isna(rs_val) or rs_val == float('inf') or rs_val == float('-inf'): rs_val = 0.0
            except: rs_val = 0.0
            
            # 🚀 提取新特徵：波動率 (Volatility) 與 成交金額 (Turnover/規模)
            close_val = float(row.get("Close", 0))
            vol_val = float(row.get("Volume", 0))
            atr_val = float(row.get("ATR_14", 0))
            
            volatility = round(atr_val / close_val, 4) if close_val > 0 else 0.0
            turnover = float(close_val * vol_val)
            
            bull_div = bool(row.get('Bullish_Div', False))
            liq_sweep = bool(row.get('Liquidity_Sweep_Bull', False))
            low_vol_pb = bool(row.get('Low_Vol_Pullback', False))
            squeeze_on = bool(row.get('Squeeze_On', False))
            
            pattern_list = []
            if low_vol_pb: pattern_list.append("📉 量縮回踩")
            if squeeze_on: pattern_list.append("🛡️ 區間壓縮")
            if liq_sweep: pattern_list.append("🌊 流動性掠奪")
            if bull_div: pattern_list.append("🟢 RSI底背離")
            pattern_str = " + ".join(pattern_list) if pattern_list else "常態震盪"
            
            db_records.append({
                "date": date_str,
                "ticker": clean_ticker,
                "name": name,
                "score": int(row.get("Score", 0)),
                "pattern": pattern_str,
                "close_price": close_val,
                "rs_index": rs_val,
                "volatility": volatility,  # 🆕 新增存入資料庫
                "turnover": turnover       # 🆕 新增存入資料庫
            })
            
        return clean_ticker, db_records
        
    except Exception as e:
        return clean_ticker, f"❌ {clean_ticker} 運算失敗: {str(e)}"

def inject_history_data(target_tickers, days_back=730):
    if SUPABASE_URL.startswith("請填入") or SUPABASE_KEY.startswith("請填入"):
        print("❌ 請先填入正確的 Supabase URL 與 Key！")
        return

    print(f"🔗 正在連線至 Supabase 量化記憶中樞...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    market_ret_20 = get_precalculated_market_ret()
    names_dict = DEFAULT_NAMES.copy()
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)
    all_db_records = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_ticker, t, days_back, market_ret_20, names_dict, start_date): t for t in target_tickers}
        for future in concurrent.futures.as_completed(futures):
            ticker, result = future.result()
            if isinstance(result, str): print(result) 
            else:
                all_db_records.extend(result)
                print(f"   ✅ {ticker} 歷史運算完成，產出 {len(result)} 天記憶。")

    if not all_db_records: return
    
    try:
        batch_size = 500
        for i in range(0, len(all_db_records), batch_size):
            batch = all_db_records[i:i + batch_size]
            supabase.table("quant_history").insert(batch).execute()
            print(f"   ⬆️ 成功注入進度：{min(i+batch_size, len(all_db_records))} / {len(all_db_records)}")
        print("🎉 歷史記憶大回補 完美成功！(包含全新股性特徵)")
    except Exception as e: print(f"❌ 寫入資料庫失敗: {e}")

if __name__ == "__main__":
    auto_tickers = set()
    
    # 🚀 強制讓 AI 見世面：硬塞入台灣市值前 50 大權值股作為學習基準
    top_50 = ['2330.TW', '2317.TW', '2454.TW', '2382.TW', '2308.TW', '2881.TW', '2882.TW', '2891.TW', '3231.TW', '2303.TW', '2886.TW', '2884.TW', '2885.TW', '1216.TW', '2002.TW', '2892.TW', '2880.TW', '2883.TW', '2887.TW', '2912.TW', '2356.TW', '2379.TW', '2301.TW', '3045.TW', '2345.TW', '2395.TW', '2412.TW', '2890.TW', '2603.TW', '2609.TW', '2615.TW', '2207.TW', '3711.TW', '5871.TW', '4938.TW', '5880.TW', '6669.TW', '2324.TW', '3008.TW', '3034.TW', '3481.TW', '2409.TW', '2801.TW', '2812.TW', '8046.TW', '2888.TW', '2353.TW', '2352.TW', '1101.TW', '1102.TW']
    for t in top_50: auto_tickers.add(t)
        
    for cluster_name, tickers in DEFAULT_CLUSTERS.items():
        for t in tickers: auto_tickers.add(t.strip().upper())
            
    for chain_name, sub_chains in INDUSTRY_CHAINS.items():
        for sub_name, tickers in sub_chains.items():
            for t in tickers:
                formatted_t = t.strip().upper()
                if not (formatted_t.endswith('.TW') or formatted_t.endswith('.TWO')): formatted_t = f"{formatted_t}.TW"
                auto_tickers.add(formatted_t)
                
    final_watchlist = sorted(list(auto_tickers))
    
    # ⚡ 這裡請在本地端手動執行一次，讓它回補兩年資料。
    # 之後 GitHub Actions 每天跑的時候，只要把這裡改成 days_back=1 就可以了！
    inject_history_data(final_watchlist, days_back=730)