# ==========================================================
# 🪐 GitHub Actions 專用：台股輕量化 LSTM 週末自動訓練引擎 (train_lstm.py)
# ==========================================================
import os
import numpy as np
import pandas as pd
from supabase import create_client
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. 初始化 Supabase 連線 (從環境變數讀取安全金鑰)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ 錯誤：找不到 Supabase 環境變數金鑰，終止訓練。")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. 拉取全歷史資料庫
print("📡 正在從 Supabase 拔取 quant_history 全量資料...")
all_data = []
offset = 0
limit = 1000
while True:
    res = supabase.table("quant_history").select("*").range(offset, offset+limit-1).execute()
    if not res.data:
        break
    all_data.extend(res.data)
    offset += limit

df = pd.DataFrame(all_data)
df['date'] = pd.to_datetime(df['date'])
df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
print(f"✅ 資料拉取完成，共 {len(df)} 筆樣本。")

# 3. 特徵工程數位化
print("🛠️ 執行特徵空間數位化手術...")
df['pattern'] = df['pattern'].fillna("")
df['is_pullback'] = df['pattern'].str.contains("量縮回踩").astype(float)
df['is_squeeze'] = df['pattern'].str.contains("區間壓縮").astype(float)
df['is_divergence'] = df['pattern'].str.contains("底背離").astype(float)
df['is_liquidity_sweep'] = df['pattern'].str.contains("流動性掠奪").astype(float)
df['is_poc_rejection'] = df['pattern'].str.contains("POC").astype(float)

df['daily_return'] = df.groupby('ticker')['close_price'].pct_change().fillna(0)

numeric_cols = ['daily_return', 'vol_ratio', 'broker_conc', 'rs_index', 'volatility', 'turnover']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

feature_cols = numeric_cols + ['is_pullback', 'is_squeeze', 'is_divergence', 'is_liquidity_sweep', 'is_poc_rejection']

# 4. 滾動切割 3D 張量 (10天回看，5天開獎)
TIME_STEPS = 10
FUTURE_DAYS = 5
X, y = [], []

print(f"🔪 正在按標的切碎時間序列 (Time Steps: {TIME_STEPS})...")
for ticker, group in df.groupby('ticker'):
    group = group.sort_values('date').reset_index(drop=True)
    group['future_return'] = group['close_price'].shift(-FUTURE_DAYS) / group['close_price'].shift(-1) - 1
    group['target_label'] = (group['future_return'] > 0.02).astype(int)
    
    for i in range(len(group) - TIME_STEPS - FUTURE_DAYS):
        X.append(group.iloc[i : i + TIME_STEPS][feature_cols].values)
        y.append(group.iloc[i + TIME_STEPS - 1]['target_label'])

X_train = np.array(X, dtype=np.float32)
y_train = np.array(y, dtype=np.float32).reshape(-1, 1)

if len(X_train) == 0:
    print("❌ 樣本深度不足，無法訓練。")
    exit(1)

# 5. 鎖定極輕量結構 (保證 CPU 15分鐘內煮完，檔案 <2MB)
model = Sequential([
    LSTM(64, input_shape=(TIME_STEPS, len(feature_cols)), return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

print("🔥 GitHub Actions 免費 CPU 煉丹開爐 (限制 100 Epochs)...")
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 6. 覆蓋落庫大腦檔
model_path = "lstm_momentum_brain.h5"
model.save(model_path)
print(f"🎉 週末大腦自動優化完畢！新大腦體積: {os.path.getsize(model_path)/(1024*1024):.2f} MB")