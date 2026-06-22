from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import requests  # 🔥 新增這行，用來發送網路請求
from ai_engine import DualCoreBrain

app = FastAPI(title="台股四核量化 API (生產環境版)", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

brain = DualCoreBrain()

def load_market_snapshot():
    """
    🔥 終極解法：直接從 GitHub Raw 抓取最新爬蟲產出的快取檔案。
    這樣 Render 永遠都能拿到最新報價，完全不依賴本地檔案！
    """
    # ⚠️ 請把下面網址中的 "你的GitHub帳號" 換成你真實的帳號名稱！
    # 如果你的專案名稱不是 taiwan-bot，也請一併替換。
    github_raw_url = "https://raw.githubusercontent.com/你的GitHub帳號/taiwan-bot/main/market_snapshot.json"

    try:
        # 1. 優先從 GitHub 雲端抓取最新的爬蟲資料
        response = requests.get(github_raw_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                print("✅ 成功從 GitHub 獲取最新爬蟲資料！")
                return data
    except Exception as e:
        print(f"⚠️ 從 GitHub 獲取資料失敗: {e}")

    # 2. 如果沒網路或 GitHub 擋連線，才退回讀取本地檔案（供你在自己電腦開發時使用）
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "market_snapshot.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "data" in data and len(data["data"]) > 0:
                    return data
    except Exception:
        pass

    # 3. 如果連本地都空了，回傳空陣列，讓前端顯示「無資料」而不是顯示假數據
    return {"data": []}

@app.get("/")
def read_root():
    return {"status": "success", "message": "API 服務正常運作中"}

# 🔥 升級 1：單股深度診斷，輸出真實 SMC 與動能指標
@app.get("/api/v1/scan/{ticker}")
def scan_stock(ticker: str):
    snap = load_market_snapshot()
    snap_dict = {str(item.get('代號', '')).split('.')[0].strip(): item for item in snap['data']}
    
    if ticker not in snap_dict:
        raise HTTPException(status_code=404, detail=f"找不到代號 {ticker} 的市場快取資料")
        
    item = snap_dict[ticker]
    entry_price = float(item.get('現價', item.get('close_price', 0.0)))
    vol = float(item.get('成交量', 0.0))
    atr_14 = float(item.get('ATR_14', entry_price * 0.05))
    res_level = round(float(item.get('Res_20', entry_price * 1.05)), 2)
    sub_level = round(float(item.get('Sup_20', entry_price * 0.95)), 2)
    
    # 透過 AI 引擎萃取真實特徵
    feat = brain.extract_features(ticker, entry_price, snap_dict, current_vol=vol, fallback_atr=atr_14)
    core_data = brain.predict_four_core([feat])[0]
    
    # 根據特徵反推給前端看盤的具體文字描述
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
        # 這裡是傳送給前端的真實策略分析資料
        "strategy_analysis": {
            "rs_status": rs_status,
            "volatility_status": volatility_status,
            "is_pullback": bool(feat['is_pullback']),
            "is_liquidity_sweep": bool(feat['is_liquidity_sweep']),
            "is_poc_rejection": bool(feat['is_poc_rejection']),
            "raw_pattern": str(item.get('pattern', '無特定形態'))
        }
    }

# 🔥 升級 2：自選群組動態行情接口
@app.get("/api/v1/market/watchlist")
def get_watchlist_quotes(tickers: str):
    """
    接收逗號分隔的代號字串 (例如: "2330,0056,00878,2317")，批次回傳即時報價
    """
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
    if not snap or 'data' not in snap:
        raise HTTPException(status_code=500, detail="無快取資料")
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