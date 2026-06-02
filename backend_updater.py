import pandas as pd
import json
import datetime
import concurrent.futures
import os
from data_fetcher import load_all_market_tickers, get_precalculated_market_ret, _fetch_and_score_sync
from config import DEFAULT_NAMES

def run_backend_update():
    print(f"[{datetime.datetime.now()}] 🚀 啟動背景雲端量化引擎...")
    
    csv_df = load_all_market_tickers()
    if csv_df.empty:
        print("❌ 找不到全市場股票清單 (all_tw_stocks.csv)，跳過運算。")
        return
        
    tickers = csv_df['Ticker'].tolist()
    names_dict = DEFAULT_NAMES.copy()
    for index, row in csv_df.iterrows():
        code = str(row['Ticker']).split('.')[0]
        if code not in names_dict:
            names_dict[code] = str(row['Name'])

    market_ret_20 = get_precalculated_market_ret()
    results = []
    total = len(tickers)
    completed = 0
    print(f"📊 預計掃描 {total} 檔標的...")
    
    # 🚀 將併發數降至 10，避免觸發 Yahoo 的 DDoS 防火牆而導致全盤斷線
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
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

    final_data = []
    # 🛡️ 容錯防護：如果抓取到的資料太少（例如小於 50 檔），直接沿用前一天的快取
    if len(results) < 50:
        print("⚠️ 抓取到的有效標的過少 (可能遭 Yahoo 防火牆短暫限流)。嘗試保留舊有快取資料...")
        if os.path.exists("market_snapshot.json"):
            try:
                with open("market_snapshot.json", "r", encoding="utf-8") as f:
                    old_snapshot = json.load(f)
                    final_data = old_snapshot.get("data", [])
            except: pass
    else:
        df_res = pd.DataFrame(results).sort_values("量化總分", ascending=False)
        final_data = df_res.to_dict(orient="records")

    if not final_data:
        print("❌ 無法生成任何資料，結束程式。")
        return

    output_data = {
        "update_time": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
        "data": final_data
    }
    
    with open("market_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 更新成功！共萃取 {len(final_data)} 檔標的，已儲存至 market_snapshot.json")

if __name__ == "__main__":
    run_backend_update()