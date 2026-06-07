# backend_updater.py — 修正版
# 修正點：
#   1. 寫入 Supabase 時 score 欄位改為寫入 indicators.py 算出的真實 Score，
#      而非硬編碼 0，讓 backtester 的 ai_prob 門檻判斷有意義。
#   2. 為了算出 Score，在 _fetch_and_score_sync 結果之外，
#      額外呼叫 add_advanced_indicators 取得 Score 欄位。
#   3. 其餘快取毒化防護邏輯保持不變。

import pandas as pd
import json
import datetime
import concurrent.futures
import os
from supabase import create_client, Client
from data_fetcher import (
    load_all_market_tickers,
    get_precalculated_market_ret,
    _fetch_and_score_sync,
    fetch_yahoo_robust,
)
from indicators import add_advanced_indicators
from config import DEFAULT_NAMES


def _compute_score_for_ticker(ticker, market_ret_20):
    """
    用 indicators.add_advanced_indicators 算出最新一天的 Score。
    回傳 int，失敗時回傳 0。
    """
    try:
        df = fetch_yahoo_robust(f"{ticker}.TW", period="2mo", interval="1d")
        if df.empty:
            df = fetch_yahoo_robust(f"{ticker}.TWO", period="2mo", interval="1d")
        if df.empty or len(df) < 40:
            return 0
        df = add_advanced_indicators(df, market_ret_20)
        score_val = df['Score'].iloc[-1]
        if pd.isna(score_val):
            return 0
        return int(score_val)
    except Exception:
        return 0


def run_backend_update():
    tz_tw   = datetime.timezone(datetime.timedelta(hours=8))
    now_tw  = datetime.datetime.now(tz_tw)
    today_str = now_tw.strftime("%Y-%m-%d")
    print(f"[{now_tw.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 啟動背景雲端量化引擎...")

    csv_df = load_all_market_tickers()
    if csv_df.empty:
        return

    tickers    = csv_df['Ticker'].tolist()
    names_dict = DEFAULT_NAMES.copy()
    for _, row in csv_df.iterrows():
        code = str(row['Ticker']).split('.')[0]
        if code not in names_dict:
            names_dict[code] = str(row['Name'])

    market_ret_20 = get_precalculated_market_ret()
    results = []

    print(f"📊 預計掃描 {len(tickers)} 檔標的...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(_fetch_and_score_sync, t, market_ret_20): t
            for t in tickers
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception:
                pass

    # ── 快取毒化防護 ──────────────────────────────────────────────────────
    is_fresh_data = True
    final_data    = []

    if len(results) < 50:
        is_fresh_data = False
        print("⚠️ 抓取到的有效標的過少。嘗試保留舊有快取資料以供網頁顯示...")
        if os.path.exists("market_snapshot.json"):
            try:
                with open("market_snapshot.json", "r", encoding="utf-8") as f:
                    final_data = json.load(f).get("data", [])
            except Exception:
                pass
    else:
        df_res     = pd.DataFrame(results)
        final_data = df_res.to_dict(orient="records")

    if not final_data:
        return

    with open("market_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(
            {"update_time": now_tw.strftime("%Y-%m-%d %H:%M:%S"), "data": final_data},
            f, ensure_ascii=False, indent=2
        )
    print(f"✅ 快取更新成功！共萃取 {len(final_data)} 檔標的。")

    if not is_fresh_data:
        print("🛑 因為使用舊快取，跳過資料庫寫入，保護機器學習記憶的純潔性。")
        return

    # ── Supabase 寫入 ─────────────────────────────────────────────────────
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if supabase_url and supabase_key:
        try:
            supabase: Client = create_client(supabase_url, supabase_key)
            db_records = []

            # 批次計算每檔的真實 Score（使用 ThreadPool 加速）
            ticker_list = [str(item.get("代號")) for item in final_data]
            score_map   = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_score = {
                    executor.submit(_compute_score_for_ticker, t, market_ret_20): t
                    for t in ticker_list
                }
                for future in concurrent.futures.as_completed(future_score):
                    t = future_score[future]
                    try:
                        score_map[t] = future.result()
                    except Exception:
                        score_map[t] = 0

            for item in final_data:
                ticker_code = str(item.get("代號"))
                db_records.append({
                    "date":        today_str,
                    "ticker":      ticker_code,
                    "name":        str(item.get("名稱", names_dict.get(ticker_code.split('.')[0], ""))),
                    "pattern":     str(item.get("pattern", "")),
                    "close_price": float(item.get("現價", 0)),
                    "rs_index":    float(item.get("rs_index",   0.0)),
                    "volatility":  float(item.get("volatility", 0.0)),
                    "turnover":    float(item.get("turnover",   0.0)),
                    "vol_ratio":   float(item.get("vol_ratio",  1.0)),
                    "broker_conc": float(item.get("broker_conc", 0.0)),
                    # 修正：寫入真實 Score，不再硬編碼 0
                    "score":       score_map.get(ticker_code, 0),
                })

            if db_records:
                for i in range(0, len(db_records), 500):
                    supabase.table("quant_history").insert(db_records[i:i+500]).execute()
                print(f"✅ 歷史記憶已成功封裝入 Supabase！共寫入 {len(db_records)} 筆。")

        except Exception as e:
            print(f"❌ 寫入 Supabase 失敗：{e}")


if __name__ == "__main__":
    run_backend_update()