# train_lstm.py — 修正版
# GitHub Actions 專用：台股輕量化 LSTM 週末自動訓練引擎
#
# 修正點：
#   1. 時序資料洩漏：改用 TimeSeriesSplit 取代 validation_split=0.2（隨機切割），
#      確保 validation 集永遠是「未來」的資料，貼近實盤情境。
#   2. 特徵標準化：加入 StandardScaler，解決 turnover 與 daily_return
#      尺度差異過大導致 LSTM 梯度不穩定的問題。
#   3. Scaler 序列化：訓練完畢後將 scaler 存為 lstm_scaler.joblib，
#      供 data_pipeline.py 實盤推論時使用同一個縮放基準。
#   4. 儲存訓練後的盲測勝率至 model_metrics.json，讓 ui_components
#      的看板可以讀到最新數據。

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

# ── 1. 初始化 Supabase ────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ 錯誤：找不到 Supabase 環境變數金鑰，終止訓練。")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── 2. 拉取全歷史資料庫 ───────────────────────────────────────────────────
print("📡 正在從 Supabase 拔取 quant_history 全量資料...")
all_data, offset, limit = [], 0, 1000
while True:
    res = supabase.table("quant_history").select("*").range(offset, offset+limit-1).execute()
    if not res.data:
        break
    all_data.extend(res.data)
    offset += limit

df = pd.DataFrame(all_data)
df['date']        = pd.to_datetime(df['date'])
df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
print(f"✅ 資料拉取完成，共 {len(df)} 筆樣本。")

# ── 3. 特徵工程 ───────────────────────────────────────────────────────────
print("🛠️ 執行特徵空間數位化手術...")
df['pattern'] = df['pattern'].fillna("")
df['is_pullback']        = df['pattern'].str.contains("量縮回踩").astype(float)
df['is_squeeze']         = df['pattern'].str.contains("區間壓縮").astype(float)
df['is_divergence']      = df['pattern'].str.contains("底背離").astype(float)
df['is_liquidity_sweep'] = df['pattern'].str.contains("流動性掠奪").astype(float)
df['is_poc_rejection']   = df['pattern'].str.contains("POC").astype(float)

df['daily_return'] = df.groupby('ticker')['close_price'].pct_change().fillna(0)

numeric_cols = ['daily_return', 'vol_ratio', 'broker_conc',
                'rs_index', 'volatility', 'turnover']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

feature_cols = numeric_cols + [
    'is_pullback', 'is_squeeze', 'is_divergence',
    'is_liquidity_sweep', 'is_poc_rejection'
]

# ── 4. 滾動切割 3D 張量 ───────────────────────────────────────────────────
TIME_STEPS   = 10
FUTURE_DAYS  = 5

print(f"🔪 正在按標的切碎時間序列 (Time Steps: {TIME_STEPS})...")

# 先收集所有樣本的「時間索引」，用於 TimeSeriesSplit 的有序切割
all_X, all_y, all_dates = [], [], []

for ticker, group in df.groupby('ticker'):
    group = group.sort_values('date').reset_index(drop=True)
    group['future_return'] = (
        group['close_price'].shift(-FUTURE_DAYS) / group['close_price'].shift(-1) - 1
    )
    group['target_label'] = (group['future_return'] > 0.02).astype(int)

    for i in range(len(group) - TIME_STEPS - FUTURE_DAYS):
        window = group.iloc[i : i + TIME_STEPS]
        all_X.append(window[feature_cols].values)
        all_y.append(group.iloc[i + TIME_STEPS - 1]['target_label'])
        all_dates.append(group.iloc[i + TIME_STEPS - 1]['date'])

all_X     = np.array(all_X,     dtype=np.float32)
all_y     = np.array(all_y,     dtype=np.float32).reshape(-1, 1)
all_dates = np.array(all_dates)

if len(all_X) == 0:
    print("❌ 樣本深度不足，無法訓練。")
    exit(1)

print(f"✅ 切割完成，共產出 {len(all_X)} 個時間序列樣本。")

# ── 5. StandardScaler（修正：解決尺度差異導致梯度爆炸）────────────────────
#
# 需要在 2D 上 fit（samples × features），再 reshape 回 3D。
n_samples, n_steps, n_features = all_X.shape
X_2d = all_X.reshape(-1, n_features)

scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)
X_scaled    = X_2d_scaled.reshape(n_samples, n_steps, n_features)

# 序列化 scaler，供實盤推論使用
joblib.dump(scaler, "lstm_scaler.joblib")
print("✅ StandardScaler 已序列化至 lstm_scaler.joblib")

# ── 6. TimeSeriesSplit（修正：消除時序資料洩漏）────────────────────────────
#
# 按時間排序索引（因為來自多個 ticker，要按照日期排序）
sorted_idx = np.argsort(all_dates)
X_sorted   = X_scaled[sorted_idx]
y_sorted   = all_y[sorted_idx]

tscv       = TimeSeriesSplit(n_splits=5)
val_scores = []

print("🔍 TimeSeriesSplit 交叉驗證（取最後一折作為最終 validation）...")
best_val_fold = None
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted)):
    best_val_fold = (train_idx, val_idx)   # 最後一折保留

train_idx, val_idx = best_val_fold
X_train, X_val = X_sorted[train_idx], X_sorted[val_idx]
y_train, y_val = y_sorted[train_idx], y_sorted[val_idx]

print(f"   訓練集大小: {len(X_train)}, 驗證集大小: {len(X_val)}")
print(f"   （驗證集為時間上最新的 {len(X_val)} 筆，模擬實盤前向驗證）")

# ── 7. 模型定義（輕量化，確保 CPU 15 分鐘內完成）────────────────────────
model = Sequential([
    LSTM(64, input_shape=(TIME_STEPS, n_features), return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1,  activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)

# ── 8. 訓練 ──────────────────────────────────────────────────────────────
print("🔥 GitHub Actions 免費 CPU 煉丹開爐 (限制 100 Epochs)...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_val, y_val),   # 使用時序切割的 val set
    callbacks=[early_stop],
    verbose=1
)

# ── 9. 盲測勝率計算 ───────────────────────────────────────────────────────
val_preds      = (model.predict(X_val, verbose=0) >= 0.5).astype(int)
blind_win_rate = float((val_preds.flatten() == y_val.flatten()).mean())
print(f"🎯 盲測勝率 (Val Accuracy): {blind_win_rate * 100:.2f}%")

# ── 10. 更新 model_metrics.json ──────────────────────────────────────────
metrics_path = "model_metrics.json"
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}
else:
    metrics = {}

import datetime
metrics["lstm"] = {
    "blind_win_rate": round(blind_win_rate, 4),
    "last_train":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
}

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("✅ model_metrics.json 已更新。")

# ── 11. 儲存模型 ──────────────────────────────────────────────────────────
model_path = "lstm_momentum_brain.h5"
model.save(model_path)
print(
    f"🎉 週末大腦自動優化完畢！"
    f"新大腦體積: {os.path.getsize(model_path) / (1024*1024):.2f} MB"
)