import pandas as pd
import json
import datetime
import concurrent.futures
from data_fetcher import load_all_market_tickers, get_precalculated_market_ret, _fetch_and_score_sync
from config import DEFAULT_NAMES

def run_backend_update():
    print(f"[{datetime.datetime.now()}] 🚀 啟動背景雲端量化引擎...")
    
    # 1. 載入全市場清單與輕量化大盤參數
    csv_df = load_all_market_tickers()
    if csv_df.empty:
        print("❌ 找不到全市場股票清單。")
        return
        
    tickers = csv_df['Ticker'].tolist()
    # 您也可以在這裡加入您的自選股清單確保它們被掃描到
    
    names_dict = DEFAULT_NAMES.copy()
    for index, row in csv_df.iterrows():
        code = str(row['Ticker']).split('.')[0]
        if code not in names_dict:
            names_dict[code] = str(row['Name'])

    market_ret_20 = get_precalculated_market_ret()
    results = []
    
    # 2. 火力全開：使用 60 條執行緒在雲端背景暴力解算
    total = len(tickers)
    completed = 0
    print(f"📊 預計掃描 {total} 檔標的...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=60) as executor:
        future_to_ticker = {executor.submit(_fetch_and_score_sync, t, market_ret_20, {}, names_dict, "score"): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            completed += 1
            if completed % 100 == 0:
                print(f"⏳ 運算進度: {completed} / {total}")
            try:
                res = future.result()
                if res: results.append(res)
            except Exception as e:
                pass

   # 3. 將龐大的運算結果過濾、排序，並儲存為極輕量的 JSON 快取檔
    if results:
        df_res = pd.DataFrame(results)
        
        # 🚀 修復：移除 50 分過濾限制，保留全市場資料，確保產業鏈共振能完整對應！
        df_res = df_res.sort_values("量化總分", ascending=False)
        
        output_data = {
            "update_time": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
            "data": df_res.to_dict(orient="records")
        }
        
        with open("market_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ 更新成功！共萃取 {len(df_res)} 檔標的，已儲存至 market_snapshot.json")
    else:
        print("⚠️ 運算完成，但無有效結果。")

if __name__ == "__main__":
    run_backend_update()