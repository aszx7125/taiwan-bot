import pandas as pd
import json
import datetime
import concurrent.futures
import os
from supabase import create_client, Client
from data_fetcher import load_all_market_tickers, get_precalculated_market_ret, _fetch_and_score_sync
from config import DEFAULT_NAMES

def run_backend_update():
    tz_tw = datetime.timezone(datetime.timedelta(hours=8))
    now_tw = datetime.datetime.now(tz_tw)
    today_str = now_tw.strftime("%Y-%m-%d")
    print(f"[{now_tw.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 啟動背景雲端量化引擎...")
    
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
    
    # 🚀 將併發數降至 10，避免觸發 Yahoo 防火牆
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
    if len(results) < 50:
        print("⚠️ 抓取到的有效標的過少。嘗試保留舊有快取資料...")
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

    # 1. 寫入本地 JSON 供 Streamlit 網頁極速讀取
    output_data = {
        "update_time": now_tw.strftime("%Y-%m-%d %H:%M:%S"),
        "data": final_data
    }
    with open("market_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 更新成功！共萃取 {len(final_data)} 檔標的，已儲存至 market_snapshot.json")

    # ==========================================
    # 🧠 機器學習記憶中樞：將數據推送到 Supabase
    # ==========================================
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    if supabase_url and supabase_key:
        print(f"🔗 偵測到資料庫金鑰 (URL: {supabase_url[:15]}...)，開始連線與上傳...")
        try:
            supabase: Client = create_client(supabase_url, supabase_key)
            
            db_records = []
            for item in final_data:
                try: rs_val = float(str(item.get("大盤相對強度", "0")).replace("%", ""))
                except: rs_val = 0.0
                
                db_records.append({
                    "date": today_str,
                    "ticker": str(item.get("代號")),
                    "name": str(item.get("名稱")),
                    "score": int(item.get("量化總分", 0)),
                    "pattern": str(item.get("機構籌碼/型態", "")),
                    "close_price": float(item.get("現價", 0)),
                    "rs_index": rs_val
                })
            
            if db_records:
                print(f"   ⬆️ 嘗試寫入首批資料...")
                response = supabase.table("quant_history").insert(db_records[:10]).execute()
                
                if len(db_records) > 10:
                    batch_size = 500
                    for i in range(10, len(db_records), batch_size):
                        batch = db_records[i:i + batch_size]
                        supabase.table("quant_history").insert(batch).execute()
                
                print(f"✅ 歷史記憶已成功封裝入 Supabase 資料庫！共寫入 {len(db_records)} 筆。")
            else:
                print("⚠️ 沒有有效資料可供寫入。")
                
        except Exception as e:
            print(f"❌ 嚴重錯誤：寫入 Supabase 失敗！詳細原因：{str(e)}")
            raise e 
    else:
        print("⚠️ 未設定 SUPABASE_URL 或 SUPABASE_KEY，跳過資料庫寫入。")

if __name__ == "__main__":
    run_backend_update()