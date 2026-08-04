import os
import json
import datetime
import traceback
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from supabase import create_client

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

os.environ["TF_USE_LEGACY_KERAS"] = "1"

print("=" * 60)
print("🧠 啟動 AI 每日自我學習 (Continuous Self-Learning)")
print("=" * 60)

# ── 1. 初始化 ─────────────────────────────────────────────────────────────
import streamlit as st

raw_url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
raw_key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")

SUPABASE_URL = raw_url.strip().strip("\"'") if raw_url else ""
SUPABASE_KEY = raw_key.strip().strip("\"'") if raw_key else ""

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ 錯誤：找不到 Supabase 環境變數")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
base_dir = os.path.dirname(os.path.abspath(__file__))

# ── 2. 獲取上次訓練時間 ───────────────────────────────────────────────────
metrics_path = os.path.join(base_dir, "model_metrics.json")
last_train_date_str = None

if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            # 取出 lgbm 的 last_train 做為參考 (格式: YYYY-MM-DD HH:MM)
            lgbm_last = metrics.get("lgbm", {}).get("last_train")
            if lgbm_last and lgbm_last != "尚未訓練":
                # 只取日期部分 YYYY-MM-DD
                last_train_date_str = lgbm_last.split(" ")[0]
    except Exception as e:
        print(f"⚠️ 無法解析 model_metrics.json: {e}")

# 如果找不到時間，預設抓最近 14 天的資料 (避免拉取失敗)
if not last_train_date_str:
    last_train_date_str = (datetime.datetime.now() - datetime.timedelta(days=14)).strftime("%Y-%m-%d")

print(f"📅 尋找日期大於 {last_train_date_str} 的新資料...")

# ── 3. 拉取增量資料 ───────────────────────────────────────────────────────
# 為了避免未來5天的 target 計算失敗，我們需要把資料往前多抓幾天
# 但因為是增量學習，我們抓取 recent data 即可
fetch_date_start = (datetime.datetime.strptime(last_train_date_str, "%Y-%m-%d") - datetime.timedelta(days=15)).strftime("%Y-%m-%d")

all_data = []
offset = 0
limit = 1000

while True:
    res = supabase.table("quant_history").select("*").gte("date", fetch_date_start).range(offset, offset + limit - 1).execute()
    if not res.data:
        break
    all_data.extend(res.data)
    offset += limit

df = pd.DataFrame(all_data)
if df.empty:
    print("✅ 沒有歷史資料可以學習，結束任務。")
    exit(0)

df['date'] = pd.to_datetime(df['date'])
df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)

# ── 4. 特徵工程 ───────────────────────────────────────────────────────────
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

# 建立標籤
FUTURE_DAYS = 5
df['future_return'] = df.groupby('ticker')['close_price'].shift(-FUTURE_DAYS) / df.groupby('ticker')['close_price'].shift(-1) - 1
df.replace([np.inf, -np.inf], 0, inplace=True)

df_clean = df.dropna(subset=['future_return']).copy()
df_clean['target_long'] = (df_clean['future_return'] > 0.02).astype(int)
df_clean['target_short'] = (df_clean['future_return'] < -0.02).astype(int)

# 過濾出真正需要學習的新資料 (>= last_train_date)
df_new = df_clean[df_clean['date'] >= pd.to_datetime(last_train_date_str)].sort_values('date').reset_index(drop=True)

if len(df_new) < 50:
    print(f"✅ 新增有效樣本過少 (只有 {len(df_new)} 筆)，暫不觸發學習，結束任務。")
    exit(0)

print(f"📈 取得 {len(df_new)} 筆新樣本進行微調訓練。")

X_new = df_new[feature_cols]
y_long_new = df_new['target_long']
y_short_new = df_new['target_short']


def update_lgbm_model(model_path, X_new, y_new, name=""):
    print(f"🌳 開始微調 LightGBM ({name})...")
    if not os.path.exists(model_path):
        print(f"⚠️ 找不到舊模型 {model_path}，跳過。")
        return None, 0.0
    
    old_model = joblib.load(model_path)
    
    # 評估舊模型在新資料上的表現 (作為退回機制的基準)
    old_preds = old_model.predict(X_new)
    old_acc = accuracy_score(y_new, old_preds)
    
    # 建立新的分類器，並將舊模型當作 init_model 傳入
    # 設定學習率較低，並限制新增的樹數量，避免 Overfitting
    new_model = lgb.LGBMClassifier(
        n_estimators=30,  # 增量只長 30 棵新樹
        learning_rate=0.01,
        max_depth=4,
        random_state=42,
        importance_type='gain'
    )
    
    # 執行增量訓練
    try:
        new_model.fit(X_new, y_new, init_model=old_model.booster_)
    except Exception as e:
        print(f"⚠️ {name} 增量訓練失敗 (可能類別全為0或1): {e}")
        return old_model, old_acc

    new_preds = new_model.predict(X_new)
    new_acc = accuracy_score(y_new, new_preds)
    
    print(f"  - 舊模型勝率: {old_acc*100:.2f}% | 新模型勝率: {new_acc*100:.2f}%")
    if new_acc >= old_acc:
        print(f"  ✅ 表現提升或持平，採用新權重！")
        return new_model, new_acc
    else:
        print(f"  ❌ 表現退步，觸發 Rollback，保留舊權重。")
        return old_model, old_acc

lgbm_long_model, lgbm_long_acc = update_lgbm_model(os.path.join(base_dir, "quant_model.joblib"), X_new, y_long_new, "多頭")
lgbm_short_model, lgbm_short_acc = update_lgbm_model(os.path.join(base_dir, "quant_model_short.joblib"), X_new, y_short_new, "空頭")

if lgbm_long_model: joblib.dump(lgbm_long_model, os.path.join(base_dir, "quant_model.joblib"))
if lgbm_short_model: joblib.dump(lgbm_short_model, os.path.join(base_dir, "quant_model_short.joblib"))


# ── 5. LSTM 增量訓練 (動能) ──────────────────────────────────────────────
TIME_STEPS = 10
def create_lstm_tensors(df_group, feature_cols, target_col):
    all_X, all_y = [], []
    for ticker, group in df_group.groupby('ticker'):
        group = group.sort_values('date').reset_index(drop=True)
        if len(group) <= TIME_STEPS + FUTURE_DAYS:
            continue
        
        for i in range(len(group) - TIME_STEPS - FUTURE_DAYS):
            # 只取新資料的區段 (如果最後一個時間點 >= last_train_date)
            date_val = group.iloc[i + TIME_STEPS - 1]['date']
            if date_val >= pd.to_datetime(last_train_date_str):
                window = group.iloc[i : i + TIME_STEPS]
                all_X.append(window[feature_cols].values)
                all_y.append(group.iloc[i + TIME_STEPS - 1][target_col])
                
    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32).reshape(-1, 1)

# 注意這裡傳入的是 df_clean (包含前面的資料，以利湊齊 TIME_STEPS)
X_lstm, y_lstm_long = create_lstm_tensors(df_clean, feature_cols, 'target_long')
_, y_lstm_short = create_lstm_tensors(df_clean, feature_cols, 'target_short')

def update_lstm_model(model_path, scaler_path, X_raw, y_new, name=""):
    print(f"🔮 開始微調 LSTM ({name})...")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or len(X_raw) == 0:
        print(f"⚠️ 找不到舊模型或無資料 ({name})，跳過。")
        return 0.0
        
    old_model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    
    n_samples, n_steps, n_features = X_raw.shape
    
    # 增量更新 Scaler
    X_2d = X_raw.reshape(-1, n_features)
    scaler.partial_fit(X_2d)
    
    # 標準化
    X_scaled = scaler.transform(X_2d).reshape(n_samples, n_steps, n_features)
    
    # 評估舊模型
    old_preds_prob = old_model.predict(X_scaled, verbose=0)
    old_preds = (old_preds_prob >= 0.5).astype(int)
    old_acc = accuracy_score(y_new, old_preds)
    
    # 編譯並用小 Learning Rate 訓練 (避免災難性遺忘)
    old_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    # 訓練 5 個 epoch
    old_model.fit(X_scaled, y_new, epochs=5, batch_size=64, verbose=0, validation_split=0.1)
    
    new_preds_prob = old_model.predict(X_scaled, verbose=0)
    new_preds = (new_preds_prob >= 0.5).astype(int)
    new_acc = accuracy_score(y_new, new_preds)
    
    print(f"  - 舊模型勝率: {old_acc*100:.2f}% | 新模型勝率: {new_acc*100:.2f}%")
    if new_acc >= old_acc:
        print(f"  ✅ 表現提升或持平，採用新權重！")
        old_model.save(model_path)
        joblib.dump(scaler, scaler_path)
        return new_acc
    else:
        print(f"  ❌ 表現退步，觸發 Rollback，保留舊權重。")
        return old_acc

lstm_long_acc = update_lstm_model(
    os.path.join(base_dir, "lstm_momentum_brain.h5"), 
    os.path.join(base_dir, "lstm_scaler.joblib"), 
    X_lstm, y_lstm_long, "多頭"
)

lstm_short_acc = update_lstm_model(
    os.path.join(base_dir, "lstm_short_brain.h5"), 
    os.path.join(base_dir, "lstm_scaler_short.joblib"), 
    X_lstm, y_lstm_short, "空頭"
)


# ── 6. 寫入指標 (時區修正) ────────────────────────────────────────────────
tz_tw = datetime.timezone(datetime.timedelta(hours=8))
now_tw = datetime.datetime.now(tz_tw)
now_str = now_tw.strftime("%Y-%m-%d %H:%M")

if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
else:
    metrics = {}

if "lgbm" not in metrics: metrics["lgbm"] = {}
metrics["lgbm"]["last_train"] = now_str
if lgbm_long_acc > 0: metrics["lgbm"]["blind_win_rate"] = round(float(lgbm_long_acc), 4)

if "lstm" not in metrics: metrics["lstm"] = {}
metrics["lstm"]["last_train"] = now_str
if lstm_long_acc > 0: metrics["lstm"]["blind_win_rate"] = round(float(lstm_long_acc), 4)

if "short" not in metrics: metrics["short"] = {"lgbm": {}, "lstm": {}}
if "lgbm" not in metrics["short"]: metrics["short"]["lgbm"] = {}
if "lstm" not in metrics["short"]: metrics["short"]["lstm"] = {}

metrics["short"]["lgbm"]["last_train"] = now_str
if lgbm_short_acc > 0: metrics["short"]["lgbm"]["blind_win_rate"] = round(float(lgbm_short_acc), 4)

metrics["short"]["lstm"]["last_train"] = now_str
if lstm_short_acc > 0: metrics["short"]["lstm"]["blind_win_rate"] = round(float(lstm_short_acc), 4)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("🎉 每日增量自我學習完成！")
