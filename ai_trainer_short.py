"""
台股AI空頭模型訓練器
預測未來5日下跌超過2%的機率
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
print("🔴 啟動空頭大腦訓練 (預測下跌>2%)")
print("=" * 60)

# ── 1. 初始化 Supabase ────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ 錯誤：找不到 Supabase 環境變數")
    exit(1)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── 2. 拉取資料 ───────────────────────────────────────────────────────────
print("📡 正在從 Supabase 拉取全量資料...")
all_data, offset, limit = [], 0, 1000
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
print(f"✅ 資料拉取完成，共 {len(df)} 筆")

# ── 3. 特徵工程 ───────────────────────────────────────────────────────────
print("🛠️ 執行特徵工程...")
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

feature_cols = numeric_cols + [
    'is_pullback', 'is_squeeze', 'is_divergence',
    'is_liquidity_sweep', 'is_poc_rejection'
]

# ── 4. 🔥 關鍵差異：空頭標籤 ──────────────────────────────────────────────
FUTURE_DAYS = 5
df['future_return'] = df.groupby('ticker')['close_price'].shift(-FUTURE_DAYS) / df.groupby('ticker')['close_price'].shift(-1) - 1
df.replace([np.inf, -np.inf], 0, inplace=True)

df_clean = df.dropna(subset=['future_return']).copy()

# 🔥 空頭標籤：下跌超過2%
df_clean['target_short'] = (df_clean['future_return'] < -0.02).astype(int)

df_clean = df_clean.sort_values('date').reset_index(drop=True)
X = df_clean[feature_cols]
y = df_clean['target_short']

print(f"📊 標籤分佈：")
print(f"   空頭樣本 (跌>2%): {y.sum()} 筆 ({y.mean()*100:.1f}%)")
print(f"   非空頭樣本: {len(y) - y.sum()} 筆 ({(1-y.mean())*100:.1f}%)")

if len(X) < 100:
    print("❌ 樣本過少")
    exit(1)

# ── 5. 訓練 ───────────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
best_val_fold = None
for train_idx, val_idx in tscv.split(X):
    best_val_fold = (train_idx, val_idx)

train_idx, val_idx = best_val_fold
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

print(f"🌳 訓練空頭LightGBM (Train: {len(X_train)}, Val: {len(X_val)})...")

model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    importance_type='gain',
    verbose=-1,
    # 🔥 空頭模型參數調整：更敏感
    scale_pos_weight=(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)

# ── 6. 評估 ───────────────────────────────────────────────────────────────
val_preds = model.predict(X_val)
val_probs = model.predict_proba(X_val)[:, 1]
blind_win_rate = float(np.mean(val_preds == y_val))

# 計算空頭捕捉率
short_samples = y_val == 1
if short_samples.sum() > 0:
    short_capture_rate = float((val_preds[short_samples] == 1).mean())
else:
    short_capture_rate = 0.0

print(f"🎯 空頭盲測準確率: {blind_win_rate * 100:.2f}%")
print(f"🎯 空頭捕捉率: {short_capture_rate * 100:.2f}%")

# ── 7. 儲存 ───────────────────────────────────────────────────────────────
joblib.dump(model, "quant_model_short.joblib")
joblib.dump(feature_cols, "model_features_short.joblib")
print("✅ 空頭模型已儲存")

# ── 8. 更新 metrics ───────────────────────────────────────────────────────
metrics_path = "model_metrics.json"
metrics = {}
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except:
        pass

if "short" not in metrics:
    metrics["short"] = {}

metrics["short"]["lgbm"] = {
    "blind_win_rate": round(blind_win_rate, 4),
    "short_capture_rate": round(short_capture_rate, 4),
    "last_train": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
}

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("✅ model_metrics.json 已更新空頭數據")
print("=" * 60)
print("🎉 空頭大腦訓練完成！")
print("=" * 60)