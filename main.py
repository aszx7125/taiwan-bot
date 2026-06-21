from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from ai_engine import DualCoreBrain

app = FastAPI(title="台股四核量化 API (Mock版)", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
brain = DualCoreBrain()

def load_market_snapshot():
    if os.path.exists("market_snapshot.json"):
        try:
            with open("market_snapshot.json", "r", encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    return {"data": [{"代號": "2330.TW", "名稱": "台積電", "現價": 1000.0, "成交量": 50000, "ATR_14": 20.0, "Res_20": 1050.0, "Sup_20": 950.0}]}

@app.get("/")
def read_root():
    return {"status": "success", "message": "台股四核量化 API (本地Mock版) 正常運作中"}

@app.get("/api/v1/scan/{ticker}")
def scan_stock(ticker: str):
    snap = load_market_snapshot()
    snap_dict = {str(item.get('代號', '')).split('.')[0].strip(): item for item in snap['data']}
    if ticker not in snap_dict:
        snap_dict[ticker] = {"名稱": "測試標的", "現價": 100.0, "ATR_14": 5.0, "Res_20": 110.0, "Sup_20": 90.0}
    item = snap_dict[ticker]
    entry_price = float(item.get('現價', item.get('close_price', 0.0)))
    vol = float(item.get('成交量', 0.0))
    atr_14 = float(item.get('ATR_14', entry_price * 0.05))
    res_level = round(float(item.get('Res_20', entry_price * 1.05)), 2)
    sup_level = round(float(item.get('Sup_20', entry_price * 0.95)), 2)
    feat_dict = brain.extract_features(ticker, entry_price, snap_dict, current_vol=vol, fallback_atr=atr_14)
    core_data = brain.predict_four_core([feat_dict])[0]
    return {"ticker": ticker, "name": item.get('名稱', ticker), "current_price": entry_price, "res_level": res_level, "sup_level": sup_level, "ai_analysis": {"signal": core_data['signal'], "best_long_prob": round(core_data['best_long'] * 100, 1), "best_short_prob": round(core_data['best_short'] * 100, 1), "details": {"lgbm_long": round(core_data['lgbm_long'] * 100, 1), "lstm_long": round(core_data['lstm_long'] * 100, 1), "lgbm_short": round(core_data['lgbm_short'] * 100, 1), "lstm_short": round(core_data['lstm_short'] * 100, 1)}}}

@app.get("/api/v1/market/top20")
def get_top_20():
    """
    一次性掃描全市場快取，回傳勝率最高的前 20 檔標的
    """
    snap = load_market_snapshot()
    if not snap or 'data' not in snap:
        raise HTTPException(status_code=500, detail="伺服器無市場快取資料")
        
    valid_items = []
    features_list = []
    snap_dict = {str(item.get('代號', '')).split('.')[0].strip(): item for item in snap['data']}
    
    # 準備全市場的特徵批次
    for ticker, item in snap_dict.items():
        ep = float(item.get('現價', item.get('close_price', 0.0)))
        vol = float(item.get('成交量', 0.0))
        if ep > 0:
            valid_items.append(item)
            atr_14 = float(item.get('ATR_14', ep * 0.05))
            features_list.append(brain.extract_features(ticker, ep, snap_dict, current_vol=vol, fallback_atr=atr_14))
            
    if not features_list:
        return {"top20": []}
        
    # 丟給張量引擎進行極速批次推論
    core_results = brain.predict_four_core(features_list)
    
    # 組合結果
    processed = []
    for i, item in enumerate(valid_items):
        prob = core_results[i]['best_long']
        processed.append({
            "ticker": str(item.get('代號', '')).split('.')[0].strip(),
            "name": item.get('名稱', '未知'),
            "current_price": float(item.get('現價', item.get('close_price', 0.0))),
            "win_prob": round(prob * 100, 1),
            "signal": core_results[i]['signal']
        })
            
    # 依照勝率由高到低排序，並切出前 20 名
    processed.sort(key=lambda x: x['win_prob'], reverse=True)
    top_20 = processed[:20]
    
    return {"top20": top_20}
