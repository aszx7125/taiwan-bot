import os
import pandas as pd
import numpy as np
from supabase import create_client, Client
from lightgbm import LGBMClassifier  # 🚀 換上微軟最強 Tabular 機器學習引擎
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 🔑 請替換為您的 Supabase 金鑰 (測試完可改回 os.environ)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "請填入您的_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "請填入您的_SUPABASE_KEY")

def fetch_training_data():
    """從 Supabase 撈取全新含有波動率與市值規模的核心教材"""
    print("🔗 正在連線至大腦記憶庫撈取全市場均衡教材...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    all_data = []
    offset = 0
    limit = 1000
    
    while True:
        response = supabase.table("quant_history").select("*").range(offset, offset + limit - 1).execute()
        if not response.data:
            break
        all_data.extend(response.data)
        offset += limit
        
    df = pd.DataFrame(all_data)
    if df.empty:
        raise ValueError("❌ 資料庫是空的，無法進行 AI 訓練！請先手動執行全新歷史大回補 Actions")
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    return df

def prepare_features_and_labels(df, holding_period=5):
    """特徵清洗：將大、中、小型股的數據矩陣化，並注入 7 大黃金特徵"""
    print(f"🧹 正在清洗數據與萃取 7 大維度特徵矩陣 (目標預測 {holding_period} 天後漲跌)...")
    
    # 1. 標註未來解答 (Labels - y)
    df[f'future_close_{holding_period}d'] = df.groupby('ticker')['close_price'].shift(-holding_period)
    df = df.dropna(subset=[f'future_close_{holding_period}d']).copy()
    df['is_win'] = (df[f'future_close_{holding_period}d'] > df['close_price']).astype(int)
    
    # 2. 萃取型態特徵
    df['pattern'] = df['pattern'].fillna("")
    df['is_pullback'] = df['pattern'].str.contains("量縮回踩").astype(int)
    df['is_sweep'] = df['pattern'].str.contains("流動性掠奪").astype(int)
    df['is_squeeze'] = df['pattern'].str.contains("區間壓縮").astype(int)
    df['is_divergence'] = df['pattern'].str.contains("底背離").astype(int)
    
    df['rs_index'] = pd.to_numeric(df['rs_index'], errors='coerce').fillna(0)
    
    # 🚀 注入核心防禦武器：股性波動率與市值規模
    df['volatility'] = pd.to_numeric(df['volatility'], errors='coerce').fillna(0)
    df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce').fillna(0)
    
    # 7 大特徵完全對齊
    feature_columns = ['is_pullback', 'is_sweep', 'is_squeeze', 'is_divergence', 'rs_index', 'volatility', 'turnover']
    
    X = df[feature_columns]
    y = df['is_win']
    
    print(f"📊 成功萃取 {len(X)} 筆涵蓋大、中、小型股的有效訓練樣本！")
    return X, y, feature_columns

def train_ai_model():
    # 1. 取得清洗後的資料
    df = fetch_training_data()
    
    # 2. 準備特徵與解答
    X, y, feature_cols = prepare_features_and_labels(df, holding_period=5)
    
    if len(X) < 100:
        print("⚠️ 有效樣本數不足，AI 無法交叉比對股性，請確保歷史回補成功。")
        return
    
    # 3. 切分訓練集 (80%) 與 測試集 (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. 🧠 建立 LightGBM 梯度提升樹模型
    print("🤖 啟動 LightGBM 殘差鏈式學習引擎...")
    
    # 超參數設定說明：
    # n_estimators=300: 預計最多建 300 棵樹修正殘差
    # learning_rate=0.05: 穩健的收斂步長，防止學得太快產生過擬合
    # max_depth=6, num_leaves=31: 限制大腦複雜度，逼迫 AI 學習「通則」而非記住「個案」
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1  # 雲端全核心平行加速
    )
    
    # 5. 深度訓練
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[] # 基礎版本保持 Scikit-learn 簡潔
    )
    
    # 6. 盲測評估 AI 戰力
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ LightGBM 訓練完成！模型盲測準確率 (Accuracy): {acc * 100:.2f}%")
    
    # 7. 揭曉梯度提升樹自動推演出的「黃金特徵權重 (Feature Importance)」
    print("\n🔬 LightGBM 萃取出的全市場股性特徵權重分布：")
    importances = model.feature_importances_
    # 梯度提升樹算出來的權重是特徵分裂的次數 (Gain/Split)，會比隨機森林更精確
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"   - {col}: {imp} 次有效分裂裂變")
        
    # 8. 封裝封印大腦檔案
    joblib.dump(model, "quant_model.joblib")
    joblib.dump(feature_cols, "model_features.joblib")
    print("\n💾 旗艦版 LightGBM 大腦已成功封裝！隨時可供前線 1700 檔即時推論使用。")

if __name__ == "__main__":
    train_ai_model()