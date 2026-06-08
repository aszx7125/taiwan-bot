# ai_trainer.py — 雙核同步滿血版
# 負責訓練 LightGBM 靜態大腦，並與 LSTM 共享相同的特徵工程與評估標準

import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from supabase import create_client

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

# ── 3. 特徵工程 (與 LSTM 嚴格對齊) ──────────────────────────────────────────
print("🛠️ 執行特徵空間數位化手術...")
df['pattern'] = df['pattern'].fillna("")
df['is_pullback']        = df['pattern'].str.contains("量縮回踩").astype(float)
df['is_squeeze']         = df['pattern'].str.contains("區間壓縮").astype(float)
df['is_divergence']      = df['pattern'].str.contains("底背離").astype(float)
df['is_liquidity_sweep'] = df['pattern'].str.contains("流動性掠奪").astype(float)
df['is_poc_rejection']   = df['pattern'].str.contains("POC").astype(float)

df['daily_return'] = df.groupby('ticker')['close_price'].pct_change().fillna(0)

numeric_cols = ['daily_return', 'vol_ratio', 'broker_conc', 'rs_index', 'volatility', 'turnover']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 🔥 核心防護網：徹底抹除無限大 (Infinity) 數值
df.replace([np.inf, -np.inf], 0, inplace=True)

feature_cols = numeric_cols + [
    'is_pullback', 'is_squeeze', 'is_divergence',
    'is_liquidity_sweep', 'is_poc_rejection'
]

# ── 4. 標籤建立與資料清理 ─────────────────────────────────────────────────
FUTURE_DAYS = 5
df['future_return'] = df.groupby('ticker')['close_price'].shift(-FUTURE_DAYS) / df.groupby('ticker')['close_price'].shift(-1) - 1
df.replace([np.inf, -np.inf], 0, inplace=True)

# 刪除尚未開獎的最新幾天資料
df_clean = df.dropna(subset=['future_return']).copy()
df_clean['target_label'] = (df_clean['future_return'] > 0.02).astype(int)

# 依時間排序，確保不發生未來數據洩漏
df_clean = df_clean.sort_values('date').reset_index(drop=True)

X = df_clean[feature_cols]
y = df_clean['target_label']

if len(X) < 100:
    print("❌ 樣本過少，無法訓練 LightGBM。")
    exit(1)

print(f"✅ 特徵萃取完成，有效樣本數: {len(X)}")

# ── 5. TimeSeriesSplit 交叉驗證與訓練 ──────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
best_val_fold = None
for train_idx, val_idx in tscv.split(X):
    best_val_fold = (train_idx, val_idx)

train_idx, val_idx = best_val_fold
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

print(f"🌳 啟動 LightGBM 靜態大腦煉丹 (Train: {len(X_train)}, Val: {len(X_val)})...")

model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    importance_type='gain',
    verbose=-1
)

# 訓練並加入 Early Stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)

# ── 6. 盲測勝率計算 ───────────────────────────────────────────────────────
val_preds = model.predict(X_val)
blind_win_rate = float(np.mean(val_preds == y_val))
print(f"🎯 盲測勝率 (Val Accuracy): {blind_win_rate * 100:.2f}%")

# ── 7. 儲存模型與特徵清單 ─────────────────────────────────────────────────
joblib.dump(model, "quant_model.joblib")
joblib.dump(feature_cols, "model_features.joblib")
print("✅ LightGBM 模型與特徵清單已儲存 (quant_model.joblib, model_features.joblib)")

# ── 8. 寫入 model_metrics.json (核心補齊邏輯) ─────────────────────────────
metrics_path = "model_metrics.json"
metrics = {}

# 先讀取舊有的數據 (保留 LSTM 剛寫入的紀錄)
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        pass

# 更新 LightGBM 的數據
metrics["lgbm"] = {
    "blind_win_rate": round(blind_win_rate, 4),
    "last_train": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
}

# 寫回檔案
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("✅ model_metrics.json 已成功更新 LightGBM 勝率！")