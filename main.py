from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import requests

app = FastAPI(title="台股四核量化 API (生產環境版)", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 延遲載入 AI 引擎，避免啟動時記憶體直接塞爆
from ai_engine import DualCoreBrain
brain = DualCoreBrain()

# 💡 請務必將下方的設定換成你真實的 GitHub 帳號與儲存庫名稱
GITHUB_USER = "aszx7125"
GITHUB_REPO = "taiwan-bot"
GITHUB_BRANCH = "main"

def load_market_snapshot():
    """
    從 GitHub Raw 抓取最新爬蟲產出的快取檔案
    """
    github_raw_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/market_snapshot.json"
    
    try:
        response = requests.get(github_raw_url, timeout=5)
        # 🔥 修正點：Python 的屬性是 status_code，不是 statusCode！
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                return data
            else:
                print("⚠️ GitHub 檔案存在，但內部的 'data' 陣列為空")
        else:
            print(f"⚠️ GitHub 請求失敗，狀態碼: {response.status_code}")
    except Exception as e:
        print(f"⚠️ 遠端直連 GitHub 發生異常: {e}")
        
    # 本地檔案保底（僅供本機開發無網路時使用）
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "market_snapshot.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "data" in data and len(data["data"]) > 0:
                    return data
    except:
        pass
        
    return {"data": []}

@app.get("/")
def read_root():
    return {"status": "success", "message": "API 服務正常運作中"}

# 🛠️ 新增自我診斷端點：讓你直接用網頁檢查爬蟲資料有沒有接對
@app.get("/api/v1/debug")
def debug_status():
    github_raw_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/market_snapshot.json"
    status_report = {"target_url": github_raw_url, "fetch_status": "unknown"}
    try:
        res = requests.get(github_raw_url, timeout=5)
        status_report["http_status_code"] = res.status_code
        if res.status_code == 200:
            data = res.json()
            status_report["fetch_status"] = "success"
            status_report["is_data_key_exist"] = "data" in data
            if "data" in data:
                status_report["total_records_found"] = len(data["data"])
                if len(data["data"]) > 0:
                    status_report["first_record_sample"] = data["data"][0]
        else:
            status_report["fetch_status"] = "failed"
            status_report["server_response"] = res.text[:200]
    except Exception as e:
        status_report["fetch_status"] = "exception"
        status_report["error_message"] = str(e)
    return status_report

@app.get("/api/v1/scan/{ticker}")
def scan_stock(ticker: str):
    snap = load_market_snapshot()
    if not snap or "data" not in snap or len(snap["data"]) == 0:
        raise HTTPException(status_code=500, detail="雲端資料庫目前沒有任何爬蟲快取數據，請檢查後端日誌。")
        
    snap_dict = {str(item.get('代號', '')).split('.')[0].strip(): item for item in snap['data']}
    
    if ticker not in snap_dict:
        raise HTTPException(status_code=404, detail=f"在最新的爬蟲清單中找不到代號 {ticker}")
        
    item = snap_dict[ticker]
    entry_price = float(item.get('現價', item.get('close_price', 0.0)))
    vol = float(item.get('成交量', 0.0))
    atr_14 = float(item.get('ATR_14', entry_price * 0.05))
    res_level = round(float(item.get('Res_20', entry_price * 1.05)), 2)
    sub_level = round(float(item.get('Sup_20', entry_price * 0.95)), 2)
    
    feat = brain.extract_features(ticker, entry_price, snap_dict, current_vol=vol, fallback_atr=atr_14)
    core_data = brain.predict_four_core([feat])[0]
    
    volatility_status = "區間壓縮 (醞釀表態)" if feat['volatility'] < 0.03 else "波動放大 (趨勢延伸)"
    rs_status = "強於大盤 (動能充沛)" if feat['rs_index'] > 55 else "弱於大盤 (動能轉弱)" if feat['rs_index'] < 45 else "與大盤同步"
    
    return {
        "ticker": ticker,
        "name": item.get('名稱', '未知'),
        "current_price": entry_price,
        "res_level": res_level,
        "sup_level": sub_level,
        "ai_analysis": {
            "signal": core_data['signal'],
            "best_long_prob": round(core_data['best_long'] * 100, 1),
            "best_short_prob": round(core_data['best_short'] * 100, 1),
            "details": {
                "lgbm_long": round(core_data['lgbm_long'] * 100, 1),
                "lstm_long": round(core_data['lstm_long'] * 100, 1),
                "lgbm_short": round(core_data['lgbm_short'] * 100, 1),
                "lstm_short": round(core_data['lstm_short'] * 100, 1),
            }
        },
        "strategy_analysis": {
            "rs_status": rs_status,
            "volatility_status": volatility_status,
            "is_pullback": bool(feat['is_pullback']),
            "is_liquidity_sweep": bool(feat['is_liquidity_sweep']),
            "is_poc_rejection": bool(feat['is_poc_rejection']),
            "raw_pattern": str(item.get('pattern', '無特定形態'))
        }
    }

@app.get("/api/v1/market/watchlist")
def get_watchlist_quotes(tickers: str):
    snap = load_market_snapshot()
    if not snap or 'data' not in snap:
        return {"watchlist": []}
        
    snap_dict = {str(item.get('代號', '')).split('.')[0].strip(): item for item in snap['data']}
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    
    result = []
    for ticker in ticker_list:
        if ticker in snap_dict:
            item = snap_dict[ticker]
            result.append({
                "ticker": ticker,
                "name": item.get('名稱', '未知'),
                "price": float(item.get('現價', item.get('close_price', 0.0))),
                "pattern": str(item.get('pattern', '無特定形態'))
            })
    return {"watchlist": result}

@app.get("/api/v1/market/top20")
def get_top_20():
    snap = load_market_snapshot()
    if not snap or 'data' not in snap or len(snap['data']) == 0:
        return {"top20": []}
    valid_items, features_list = [], []
    snap_dict = {str(item.get('代號', '')).split('.')[0].strip(): item for item in snap['data']}
    for ticker, item in snap_dict.items():
        ep = float(item.get('現價', item.get('close_price', 0.0)))
        vol = float(item.get('成交量', 0.0))
        if ep > 0:
            valid_items.append(item)
            atr_14 = float(item.get('ATR_14', ep * 0.05))
            features_list.append(brain.extract_features(ticker, ep, snap_dict, current_vol=vol, fallback_atr=atr_14))
    if not features_list: return {"top20": []}
    core_results = brain.predict_four_core(features_list)
    processed = []
    for i, item in enumerate(valid_items):
        processed.append({
            "ticker": str(item.get('代號', '')).split('.')[0].strip(),
            "name": item.get('名稱', '未知'),
            "current_price": float(item.get('現價', item.get('close_price', 0.0))),
            "win_prob": round(core_results[i]['best_long'] * 100, 1),
            "signal": core_results[i]['signal']
        })
    processed.sort(key=lambda x: x['win_prob'], reverse=True)
    return {"top20": processed[:20]}