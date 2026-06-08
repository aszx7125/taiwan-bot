import pandas as pd
import numpy as np
import datetime
import asyncio
import aiohttp
import os
from supabase import create_client, Client
from data_fetcher import get_historical_twii_series, fetch_kline_robust_async, _normalize_ticker
from indicators import add_advanced_indicators
from config import DEFAULT_CLUSTERS, INDUSTRY_CHAINS, DEFAULT_NAMES, get_fugle_key

SUPABASE_URL = os.environ.get("SUPABASE_URL", "請填入您的_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "請填入您的_SUPABASE_KEY")
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")

async def process_single_ticker_async(session, t, days_back, market_ret_series, names_dict, start_date, sem, fugle_key):
    clean_ticker = _normalize_ticker(t)[0]
    name = names_dict.get(clean_ticker, f"代號 {clean_ticker}")
    db_records = []
    
    async with sem:
        await asyncio.sleep(0.3) # 歷史抓取較大，間隔 0.3 秒保平安
        df = await fetch_kline_robust_async(session, t, FINMIND_TOKEN, fugle_key, days_back=days_back)
            
        if df.empty or len(df) < 100:
            return clean_ticker, f"⚠️ {clean_ticker} {name} 數據不足，跳過。"

        df = add_advanced_indicators(df, market_ret_series)
        
        df['Vol_SMA5'] = df['Volume'].rolling(window=5).mean()
        df['Vol_Ratio'] = np.where(df['Vol_SMA5'] > 0, df['Volume'] / df['Vol_SMA5'], 1.0)
        
        df = df[df.index >= start_date]
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        for date_timestamp, row in df.iterrows():
            date_str = date_timestamp.strftime("%Y-%m-%d")
            
            close_val = float(row.get("Close", 0))
            vol_val = float(row.get("Volume", 0))
            atr_val = float(row.get("ATR_14", 0))
            
            volatility = round(atr_val / close_val, 4) if close_val > 0 else 0.0
            turnover = float(close_val * vol_val)
            
            vol_ratio = float(row.get('Vol_Ratio', 1.0))
            
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
                "rs_index": float(row.get("RS_Index", 0.0)),
                "volatility": volatility,
                "turnover": turnover,
                "vol_ratio": round(vol_ratio, 2),
                "broker_conc": 0.0
            })
            
        return clean_ticker, db_records

async def run_history_injection(target_tickers, days_back, market_ret_series, names_dict, start_date):
    fugle_key = get_fugle_key()
    sem = asyncio.Semaphore(5)
    all_db_records = []
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_single_ticker_async(session, t, days_back, market_ret_series, names_dict, start_date, sem, fugle_key) for t in target_tickers]
        for future in asyncio.as_completed(tasks):
            try:
                ticker, result = await future
                if isinstance(result, str): print(result) 
                else:
                    all_db_records.extend(result)
                    print(f"   ✅ {ticker} 歷史運算完成，產出 {len(result)} 天記憶。")
            except Exception as e:
                print(f"❌ 未知錯誤: {e}")
                
    return all_db_records

def inject_history_data(target_tickers, days_back=730):
    if SUPABASE_URL.startswith("請填入") or SUPABASE_KEY.startswith("請填入"):
        print("❌ 請先填入正確的 Supabase URL 與 Key！")
        return

    print(f"🔗 正在連線至 Supabase 量化記憶中樞...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    market_ret_series = get_historical_twii_series()
    names_dict = DEFAULT_NAMES.copy()
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)
    
    print(f"🎯 啟動全市場非同步歷史回補！共計 {len(target_tickers)} 檔代表性標的...")
    all_db_records = asyncio.run(run_history_injection(target_tickers, days_back, market_ret_series, names_dict, start_date))

    if not all_db_records: return
    
    print(f"\n======================================")
    print(f"🚀 運算完畢！準備將 {len(all_db_records)} 筆記憶注入資料庫...")
    
    try:
        batch_size = 500
        for i in range(0, len(all_db_records), batch_size):
            batch = all_db_records[i:i + batch_size]
            supabase.table("quant_history").insert(batch).execute()
            print(f"   ⬆️ 成功注入進度：{min(i+batch_size, len(all_db_records))} / {len(all_db_records)}")
        print("🎉 歷史記憶大回補 完美成功！")
    except Exception as e: print(f"❌ 寫入資料庫失敗: {e}")

if __name__ == "__main__":
    auto_tickers = set()
    top_50 = ['2330', '2317', '2454', '2382', '2308', '2881', '2882', '2891', '3231', '2303', '2886', '2884', '2885', '1216', '2002', '2892', '2880', '2883', '2887', '2912', '2356', '2379', '2301', '3045', '2345', '2395', '2412', '2890', '2603', '2609', '2615', '2207', '3711', '5871', '4938', '5880', '6669', '2324', '3008', '3034', '3481', '2409', '2801', '2812', '8046', '2888', '2353', '2352', '1101', '1102']
    mid_50 = ['2368', '2376', '2377', '2383', '3037', '2618', '2610', '2313', '2354', '2449', '2373', '2385', '2392', '2408', '2458', '2606', '2809', '2834', '2845', '2889', '2903', '2915', '3044', '3443', '3532', '3661', '3702', '4904', '4915', '5347', '5483', '6176', '6239', '6271', '8016', '8081', '8112', '8464', '9904', '9910', '9914', '9921', '9941', '9945']
    otc_50 = ['3105', '3293', '3324', '3529', '5425', '6147', '6274', '8069', '8299', '3131', '3141', '3227', '3260', '3264', '3314', '3328', '3362', '3374', '3483', '3491', '3552', '3556', '3587', '3680', '4105', '4114', '4123', '4128', '4162', '4743', '4947', '4953', '4979', '5289', '5351', '5478', '5490', '5536', '6104', '6121', '6138', '6182', '6188', '6223', '6245', '6279', '8044', '8050']

    for t in top_50 + mid_50 + otc_50: auto_tickers.add(t)
    for cluster_name, tickers in DEFAULT_CLUSTERS.items():
        for t in tickers: auto_tickers.add(t)
    for chain_name, sub_chains in INDUSTRY_CHAINS.items():
        for sub_name, tickers in sub_chains.items():
            for t in tickers: auto_tickers.add(t)
                
    final_watchlist = sorted(list(auto_tickers))
    inject_history_data(final_watchlist, days_back=730)