import os
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 🔑 請替換為您的 Supabase 金鑰 (測試完可改回 os.environ)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "請填入您的_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "請填入您的_SUPABASE_KEY")

def fetch_training_data():
    """從 Supabase 撈取歷史資料作為 AI 的訓練教材"""
    print("🔗 正在連線至大腦記憶庫撈取教材...")
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
        raise ValueError("❌ 資料庫是空的，無法進行 AI 訓練！請先執行 historical_injector.py")
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    return df

def prepare_features_and_labels(df, holding_period=5):
    """資料清洗：將文字轉為 AI 看得懂的特徵矩陣 (X)，並計算未來的真實勝負 (y)"""
    print(f"🧹 正在清洗數據與萃取特徵 (目標預測 {holding_period} 天後漲跌)...")
    
    # 1. 標註解答 (Labels - y)：N 天後收盤價是否大於今天收盤價
    df[f'future_close_{holding_period}d'] = df.groupby('ticker')['close_price'].shift(-holding_period)
    df = df.dropna(subset=[f'future_close_{holding_period}d']).copy() # 刪除還沒開獎的資料
    df['is_win'] = (df[f'future_close_{holding_period}d'] > df['close_price']).astype(int)
    
    # 2. 萃取特徵 (Features - X)：把文字型態拆解成二元矩陣 (0 與 1)
    # AI 看不懂 "📉 量縮回踩"，我們必須把它變成欄位 is_pullback = 1
    df['pattern'] = df['pattern'].fillna("")
    df['is_pullback'] = df['pattern'].str.contains("量縮回踩").astype(int)
    df['is_sweep'] = df['pattern'].str.contains("流動性掠奪").astype(int)
    df['is_squeeze'] = df['pattern'].str.contains("區間壓縮").astype(int)
    df['is_divergence'] = df['pattern'].str.contains("底背離").astype(int)
    
    # 加入連續數值特徵
    df['rs_index'] = pd.to_numeric(df['rs_index'], errors='coerce').fillna(0)
    
    # 定義 AI 要學習的特徵欄位
    feature_columns = ['is_pullback', 'is_sweep', 'is_squeeze', 'is_divergence', 'rs_index']
    
    X = df[feature_columns]
    y = df['is_win']
    
    print(f"📊 成功萃取 {len(X)} 筆有效訓練樣本！")
    return X, y, feature_columns

def train_ai_model():
    # 1. 取得資料
    df = fetch_training_data()
    
    # 2. 準備特徵與解答
    X, y, feature_cols = prepare_features_and_labels(df, holding_period=5)
    
    if len(X) < 100:
        print("⚠️ 有效樣本數低於 100 筆，AI 可能無法有效學習，建議補齊更多歷史數據。")
    
    # 3. 切分訓練集 (80%) 與測試集 (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. 🧠 建立與訓練 AI 模型 (隨機森林)
    print("🤖 AI 演算法開始深度學習歷史勝敗規律...")
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 5. 評估 AI 智商 (準確率測試)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ 訓練完成！模型盲測準確率 (Accuracy): {acc * 100:.2f}%")
    
    # 6. 揭曉 AI 自動推演出的「特徵權重 (Feature Importance)」
    print("\n🔬 AI 學習到的特徵重要性 (推翻人類規則的結果)：")
    importances = model.feature_importances_
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"   - {col}: {imp*100:.1f}%")
        
    # 7. 儲存大腦
    joblib.dump(model, "quant_model.joblib")
    joblib.dump(feature_cols, "model_features.joblib")
    print("\n💾 AI 大腦已封裝為 quant_model.joblib，隨時可供前線每日掃描使用！")

if __name__ == "__main__":
    train_ai_model()