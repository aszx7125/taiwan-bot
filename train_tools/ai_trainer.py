import os
import pandas as pd
import numpy as np
from supabase import create_client, Client
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# 🔑 Supabase 金鑰設定
SUPABASE_URL = os.environ.get("SUPABASE_URL", "請填入您的_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "請填入您的_SUPABASE_KEY")

def fetch_training_data():
    """從 Supabase 撈取核心教材"""
    print("🔗 正在連線至大腦記憶庫撈取全市場均衡教材...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    all_data = []
    offset = 0
    limit = 1000
    
    while True:
        response = supabase.table("quant_history").select("*").range(offset, offset + limit - 1).execute()
        if not response.data: break
        all_data.extend(response.data)
        offset += limit
        
    df = pd.DataFrame(all_data)
    if df.empty:
        raise ValueError("❌ 資料庫是空的，請先執行歷史大回補！")
        
    df['date'] = pd.to_datetime(df['date'])
    # 確保嚴格按照時間排序，這是避免未來函數的第一步
    df = df.sort_values(by=['date', 'ticker']).reset_index(drop=True)
    return df

def prepare_features_and_labels(df, holding_period=5):
    """特徵清洗與高維度矩陣擴充 (完美對齊當前 DB Schema)"""
    print(f"🧹 正在清洗數據與擴充高維度特徵矩陣 (目標預測 {holding_period} 天後突破)...")
    
    # 1. 嚴格標註未來解答 (設定 1.5% 的實質獲利門檻，濾除盤整雜訊)
    df[f'future_close_{holding_period}d'] = df.groupby('ticker')['close_price'].shift(-holding_period)
    df = df.dropna(subset=[f'future_close_{holding_period}d']).copy()
    
    # 計算真實報酬率 R = (Close_future - Close_current) / Close_current
    df['future_return'] = (df[f'future_close_{holding_period}d'] - df['close_price']) / df['close_price']
    df['is_win'] = (df['future_return'] > 0.015).astype(int) # 漲幅 > 1.5% 才算有效突破
    
    # 2. SMC 特徵解耦與型態萃取
    df['pattern'] = df['pattern'].fillna("")
    df['is_pullback'] = df['pattern'].str.contains("量縮回踩").astype(int)
    df['is_squeeze'] = df['pattern'].str.contains("區間壓縮").astype(int)
    df['is_divergence'] = df['pattern'].str.contains("底背離").astype(int)
    
    # 精確分離流動性掠奪與 POC，避免神經網路混淆市場結構
    df['is_liquidity_sweep'] = df['pattern'].str.contains("流動性掠奪").astype(int)
    df['is_poc_rejection'] = df['pattern'].str.contains("POC").astype(int)
    
    # 3. 基礎量化指標對齊
    df['rs_index'] = pd.to_numeric(df['rs_index'], errors='coerce').fillna(0)
    df['vol_ratio'] = pd.to_numeric(df['vol_ratio'], errors='coerce').fillna(1.0) # 成交量放大倍數
    
    # 4. 防禦武器與微觀籌碼
    df['volatility'] = pd.to_numeric(df['volatility'], errors='coerce').fillna(0)
    df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce').fillna(0)
    df['broker_conc'] = pd.to_numeric(df['broker_conc'], errors='coerce').fillna(0)
    
    # 對齊所有高維度特徵 (共 10 大特徵)
    feature_columns = [
        'is_pullback', 'is_squeeze', 'is_divergence', 
        'is_liquidity_sweep', 'is_poc_rejection', 
        'rs_index', 'vol_ratio', 'volatility', 'turnover',
        'broker_conc'
    ]
    
    # 5. 🛠️ 嚴格時間序列切分 (消滅未來函數)
    split_index = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    X_train = train_df[feature_columns]
    y_train = train_df['is_win']
    X_test = test_df[feature_columns]
    y_test = test_df['is_win']
    
    print(f"📊 切分完成: 訓練集 {len(X_train)} 筆 (過去), 盲測集 {len(X_test)} 筆 (近期)")
    return X_train, X_test, y_train, y_test, feature_columns

def train_ai_model():
    df = fetch_training_data()
    
    try:
        X_train, X_test, y_train, y_test, feature_cols = prepare_features_and_labels(df, holding_period=5)
    except Exception as e:
        print(f"⚠️ 特徵處理失敗: {e}")
        return
        
    if len(X_train) < 100:
        print("⚠️ 有效樣本數不足，請確保歷史回補成功。")
        return
    
    print("🤖 啟動 LightGBM 殘差鏈式學習引擎 (具備早停機制)...")
    
    model = LGBMClassifier(
        n_estimators=500,        
        learning_rate=0.03,      
        max_depth=5,             
        num_leaves=20,           
        subsample=0.8,           
        colsample_bytree=0.8,    
        random_state=42,
        class_weight='balanced', 
        n_jobs=-1 
    )
    
    # 深度訓練與早停監控
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[
            early_stopping(stopping_rounds=30), 
            log_evaluation(period=50)           
        ]
    )
    
    # 嚴格盲測評估
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n✅ LightGBM 訓練完成！")
    print(f"   - 盲測準確率 (Accuracy): {acc * 100:.2f}%")
    print(f"   - 模型鑑別力 (AUC Score): {auc:.4f}")
    
    print("\n🔬 LightGBM 萃取出的全市場股性特徵權重分布：")
    importances = model.feature_importances_
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        if imp > 0:
            print(f"   - {col}: {imp} 次有效分裂裂變")
        
    # 封裝檔案
    joblib.dump(model, "quant_model.joblib")
    joblib.dump(feature_cols, "model_features.joblib")
    print("\n💾 旗艦版 LightGBM 大腦已成功封裝！隨時可供前線即時推論使用。")

if __name__ == "__main__":
    train_ai_model()