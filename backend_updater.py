import pandas as pd
import json
import datetime
import asyncio
import aiohttp
import os
from supabase import create_client, Client
from data_fetcher import load_all_market_tickers, get_precalculated_market_ret, _fetch_and_score_async
from config import DEFAULT_NAMES, get_fugle_key

async def run_async_scraping(tickers, market_ret_20):
    finmind_token = os.environ.get("FINMIND_TOKEN", "") # 可選的 FinMind Token
    fugle_key = get_fugle_key()
    
    # 限制並發數為 5，確保每秒請求數不超過 FinMind 與 Fugle 的免費天花板
    sem = asyncio.Semaphore(5) 
    results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_and_score_async(session, t, market_ret_20, finmind_token, fugle_key, sem) for t in tickers]
        for future in asyncio.as_completed(tasks):
            try:
                res = await future
                if res: results.append(res)
            except Exception: pass
            
    return results

def run_backend_update():
    tz_tw = datetime.timezone(datetime.timedelta(hours=8))
    now_tw = datetime.datetime.now(tz_tw)
    today_str = now_tw.strftime("%Y-%m-%d")
    print(f"[{now_tw.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 啟動非同步雲端量化引擎...")

    csv_df = load_all_market_tickers()
    if csv_df.empty: return

    tickers = csv_df['Ticker'].tolist()
    names_dict = DEFAULT_NAMES.copy()
    for _, row in csv_df.iterrows():
        code = str(row['Ticker']).split('.')[0]
        if code not in names_dict: names_dict[code] = str(row['Name'])

    market_ret_20 = get_precalculated_market_ret()
    
    print(f"📊 預計掃描 {len(tickers)} 檔標的 (FinMind -> Fugle -> Yahoo 三層備援)...")
    
    # 🔥 核心：驅動非同步迴圈
    results = asyncio.run(run_async_scraping(tickers, market_ret_20))

    is_fresh_data = True
    final_data = []

    if len(results) < 50:
        is_fresh_data = False
        print("⚠️ 抓取到的有效標的過少。嘗試保留舊有快取資料以供網頁顯示...")
        if os.path.exists("market_snapshot.json"):
            try:
                with open("market_snapshot.json", "r", encoding="utf-8") as f:
                    final_data = json.load(f).get("data", [])
            except Exception: pass
    else:
        df_res = pd.DataFrame(results)
        final_data = df_res.to_dict(orient="records")

    if not final_data: return

    with open("market_snapshot.json", "w", encoding="utf-8") as f:
        json.dump({"update_time": now_tw.strftime("%Y-%m-%d %H:%M:%S"), "data": final_data}, f, ensure_ascii=False, indent=2)
    print(f"✅ 快取更新成功！共萃取 {len(final_data)} 檔標的。")

    if not is_fresh_data:
        print("🛑 因為使用舊快取，跳過資料庫寫入，保護機器學習記憶的純潔性。")
        return

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if supabase_url and supabase_key:
        try:
            supabase: Client = create_client(supabase_url, supabase_key)
            db_records = []

            for item in final_data:
                ticker_code = str(item.get("代號"))
                db_records.append({
                    "date": today_str,
                    "ticker": ticker_code,
                    "name": str(item.get("名稱", names_dict.get(ticker_code.split('.')[0], ""))),
                    "pattern": str(item.get("pattern", "")),
                    "close_price": float(item.get("現價", 0)),
                    "rs_index": float(item.get("rs_index", 0.0)),
                    "volatility": float(item.get("volatility", 0.0)),
                    "turnover": float(item.get("turnover", 0.0)),
                    "vol_ratio": float(item.get("vol_ratio", 1.0)),
                    "broker_conc": float(item.get("broker_conc", 0.0)),
                    "score": int(item.get("score", 0)), # 直接取用一站式算好的 Score
                })

            if db_records:
                for i in range(0, len(db_records), 500):
                    supabase.table("quant_history").insert(db_records[i:i+500]).execute()
                print(f"✅ 歷史記憶已成功封裝入 Supabase！共寫入 {len(db_records)} 筆。")
        except Exception as e:
            print(f"❌ 寫入 Supabase 失敗：{e}")

if __name__ == "__main__":
    run_backend_update()