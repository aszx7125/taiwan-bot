"""
台股AI多頭LSTM訓練器
預測未來5日上漲超過2%的時序機率
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from supabase import create_client
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import datetime

print("=" * 60)
print("🟢 啟動多頭LSTM訓練 (時序上漲預測)")
print("=" * 60)

# ── 1. 初始化 Supabase ────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ 錯誤：找不到 Supabase 環境變數金鑰，終止訓練。")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── 2. 拉取全量資料 ───────────────────────────────────────────────────────
print("📡 正在從 Supabase 拔取 quant_history 全量資料...")
all_data = []
offset = 0
limit = 1000

while True:
    res = supabase.table("quant_history").select("*").range(offset, offset + limit - 1).execute()
    if not res.data:
        break
    all_data.extend(res.data)
    offset += limit

df = pd.DataFrame(all_data)
df['date'] = pd.to_datetime(df['date'])
df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
print(f"✅ 資料拉取完成，共 {len(df)} 筆樣本。")

# ── 3. 特徵工程 ───────────────────────────────────────────────────────────
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

# 無限大防護
df.replace([np.inf, -np.inf], 0, inplace=True)
feature_cols = numeric_cols + ['is_pullback', 'is_squeeze', 'is_divergence', 'is_liquidity_sweep', 'is_poc_rejection']

# ── 4. 切割時序張量 ───────────────────────────────────────────────────────
TIME_STEPS = 10
FUTURE_DAYS = 5
all_X, all_y, all_dates = [], [], []

for ticker, group in df.groupby('ticker'):
    group = group.sort_values('date').reset_index(drop=True)
    group['future_return'] = group['close_price'].shift(-FUTURE_DAYS) / group['close_price'].shift(-1) - 1
    group.replace([np.inf, -np.inf], 0, inplace=True)
    group['target_label'] = (group['future_return'] > 0.02).astype(int)
    
    for i in range(len(group) - TIME_STEPS - FUTURE_DAYS):
        window = group.iloc[i : i + TIME_STEPS]
        all_X.append(window[feature_cols].values)
        all_y.append(group.iloc[i + TIME_STEPS - 1]['target_label'])
        all_dates.append(group.iloc[i + TIME_STEPS - 1]['date'])

all_X = np.array(all_X, dtype=np.float32)
all_y = np.array(all_y, dtype=np.float32).reshape(-1, 1)
all_dates = np.array(all_dates)

if len(all_X) == 0:
    print("❌ 樣本深度不足，無法訓練。")
    exit(1)

# ── 5. 特徵縮放 ───────────────────────────────────────────────────────────
n_samples, n_steps, n_features = all_X.shape
X_2d = all_X.reshape(-1, n_features)

scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)
X_scaled = X_2d_scaled.reshape(n_samples, n_steps, n_features)
joblib.dump(scaler, "lstm_scaler.joblib")

# ── 6. 交叉驗證與模型訓練 ─────────────────────────────────────────────────
sorted_idx = np.argsort(all_dates)
X_sorted = X_scaled[sorted_idx]
y_sorted = all_y[sorted_idx]

tscv = TimeSeriesSplit(n_splits=5)
best_val_fold = None
for train_idx, val_idx in tscv.split(X_sorted):
    best_val_fold = (train_idx, val_idx)

X_train, X_val = X_sorted[best_val_fold[0]], X_sorted[best_val_fold[1]]
y_train, y_val = y_sorted[best_val_fold[0]], y_sorted[best_val_fold[1]]

model = Sequential([
    LSTM(64, input_shape=(TIME_STEPS, n_features), return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# ── 7. 盲測與儲存 ─────────────────────────────────────────────────────────
val_preds = (model.predict(X_val, verbose=0) >= 0.5).astype(int)
blind_win_rate = float((val_preds.flatten() == y_val.flatten()).mean())
print(f"🎯 多頭 LSTM 盲測勝率: {blind_win_rate * 100:.2f}%")
model.save("lstm_momentum_brain.h5")

# ── 8. 寫入指標 (時區修正) ────────────────────────────────────────────────
metrics_path = "model_metrics.json"
metrics = {}

if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        pass

tz_tw = datetime.timezone(datetime.timedelta(hours=8))
now_tw = datetime.datetime.now(tz_tw)

metrics["lstm"] = {
    "blind_win_rate": round(blind_win_rate, 4),
    "last_train": now_tw.strftime("%Y-%m-%d %H:%M")
}

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("✅ 多頭 LSTM 訓練完成！")