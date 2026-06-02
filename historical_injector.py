import pandas as pd
import json
import datetime
import os
import concurrent.futures
from supabase import create_client, Client
from data_fetcher import fetch_yahoo_robust, get_precalculated_market_ret
from indicators import add_advanced_indicators
# 🚀 引入 config 內的所有標的配置
from config import DEFAULT_CLUSTERS, INDUSTRY_CHAINS, DEFAULT_NAMES

# 🔒 雲端安全版：由 GitHub Actions 保險箱自動動態注入密碼
SUPABASE_URL = os.environ.get("SUPABASE_URL", "請填入您的_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "請填入您的_SUPABASE_KEY")

def process_single_ticker(t, days_back, market_ret_20, names_dict, start_date):
    """獨立函數：處理單一檔股票的歷史數據運算"""
    clean_ticker = t.split('.')[0]
    name = names_dict.get(clean_ticker, f"代號 {clean_ticker}")
    db_records = []
    
    try:
        # 1. 抓取長期歷史日 K 線 (多抓幾天以確保指標能順利計算)
        df = fetch_yahoo_robust(f"{clean_ticker}.TW", period="3y", interval="1d")
        if df.empty:
            df = fetch_yahoo_robust(f"{clean_ticker}.TWO", period="3y", interval="1d")
            
        if df.empty or len(df) < 100:
            return clean_ticker, f"⚠️ {clean_ticker} {name} 數據不足或市面上找不到，跳過。"

        # 2. 核心步驟：計算所有的 AI 潛伏分數與微觀結構特徵
        df = add_advanced_indicators(df, market_ret_20)
        
        # 3. 篩選出我們需要回補的目標天數範圍
        df = df[df.index >= start_date]
        
        # 4. 將每一天的特徵打包成資料庫格式
        for date_timestamp, row in df.iterrows():
            date_str = date_timestamp.strftime("%Y-%m-%d")
            
            # 安全轉換 RS 數值
            try: rs_val = float(str(row.get("RS_Index", "0")).replace("%", ""))
            except: rs_val = 0.0
            
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
                "close_price": float(row.get("Close", 0)),
                "rs_index": rs_val
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
    
    # 設定時間範圍
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)
    
    print(f"📅 開始回補歷史區間：{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print(f"🎯 自動解析 Config 完畢！總計目標股票數量: {len(target_tickers)} 檔")
    
    all_db_records = []
    
    # 🚀 使用多執行緒加速下載與運算
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_ticker, t, days_back, market_ret_20, names_dict, start_date): t for t in target_tickers}
        
        for future in concurrent.futures.as_completed(futures):
            ticker, result = future.result()
            if isinstance(result, str):
                print(result) 
            else:
                all_db_records.extend(result)
                print(f"   ✅ {ticker} 歷史運算完成，產出 {len(result)} 天記憶。")

    if not all_db_records:
        print("⚠️ 沒有產出任何可寫入的歷史資料。")
        return

    print(f"\n======================================")
    print(f"🚀 運算完畢！準備將 {len(all_db_records)} 筆龐大歷史記憶注入雲端資料庫...")
    print(f"======================================")
    
    try:
        batch_size = 500
        for i in range(0, len(all_db_records), batch_size):
            batch = all_db_records[i:i + batch_size]
            supabase.table("quant_history").insert(batch).execute()
            print(f"   ⬆️ 成功注入進度：{min(i+batch_size, len(all_db_records))} / {len(all_db_records)}")
            
        print("🎉 歷史記憶大回補 完美成功！")
    except Exception as e:
        print(f"❌ 寫入資料庫失敗: {e}")

if __name__ == "__main__":
    # 🚀 自動化特徵提取邏輯：動態掃描並合併所有 config 內的標的
    auto_tickers = set()
    
    # 1. 抽取自選群組 (DEFAULT_CLUSTERS)
    for cluster_name, tickers in DEFAULT_CLUSTERS.items():
        for t in tickers:
            auto_tickers.add(t.strip().upper())
            
    # 2. 抽取產業鏈 (INDUSTRY_CHAINS)
    for chain_name, sub_chains in INDUSTRY_CHAINS.items():
        for sub_name, tickers in sub_chains.items():
            for t in tickers:
                # 確保後綴格式正確 (.TW / .TWO)
                formatted_t = t.strip().upper()
                if not (formatted_t.endswith('.TW') or formatted_t.endswith('.TWO')):
                    formatted_t = f"{formatted_t}.TW"
                auto_tickers.add(formatted_t)
                
    # 轉換回排序後的清單
    final_watchlist = sorted(list(auto_tickers))
    
    # 執行回補：預設回補過去兩年 (730天)
    inject_history_data(final_watchlist, days_back=730)