"""
台股AI多頭靜態模型訓練器
預測未來5日上漲超過2%的結構機率
"""
import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from supabase import create_client

print("=" * 60)
print("🟢 啟動多頭靜態大腦訓練 (預測上漲>2%)")
print("=" * 60)

# ── 1. 初始化 ─────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ 錯誤：找不到 Supabase 環境變數")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── 2. 拉取資料 ───────────────────────────────────────────────────────────
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

df.replace([np.inf, -np.inf], 0, inplace=True)
feature_cols = numeric_cols + ['is_pullback', 'is_squeeze', 'is_divergence', 'is_liquidity_sweep', 'is_poc_rejection']

# ── 4. 標籤建立 ───────────────────────────────────────────────────────────
FUTURE_DAYS = 5
df['future_return'] = df.groupby('ticker')['close_price'].shift(-FUTURE_DAYS) / df.groupby('ticker')['close_price'].shift(-1) - 1
df.replace([np.inf, -np.inf], 0, inplace=True)

df_clean = df.dropna(subset=['future_return']).copy()
df_clean['target_label'] = (df_clean['future_return'] > 0.02).astype(int)
df_clean = df_clean.sort_values('date').reset_index(drop=True)

X = df_clean[feature_cols]
y = df_clean['target_label']

if len(X) < 100:
    print("❌ 樣本過少，無法訓練。")
    exit(1)

# ── 5. 模型訓練 ───────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
best_val_fold = None
for train_idx, val_idx in tscv.split(X):
    best_val_fold = (train_idx, val_idx)

X_train, X_val = X.iloc[best_val_fold[0]], X.iloc[best_val_fold[1]]
y_train, y_val = y.iloc[best_val_fold[0]], y.iloc[best_val_fold[1]]

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

model.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)], 
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)

# ── 6. 評估與儲存 ─────────────────────────────────────────────────────────
val_preds = model.predict(X_val)
blind_win_rate = float(np.mean(val_preds == y_val))
print(f"🎯 多頭 LGBM 盲測準確率: {blind_win_rate * 100:.2f}%")

joblib.dump(model, "quant_model.joblib")
joblib.dump(feature_cols, "model_features.joblib")

# ── 7. 更新 metrics ───────────────────────────────────────────────────────
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

metrics["lgbm"] = {
    "blind_win_rate": round(blind_win_rate, 4),
    "last_train": now_tw.strftime("%Y-%m-%d %H:%M")
}

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("✅ 多頭 LightGBM 訓練完成！")