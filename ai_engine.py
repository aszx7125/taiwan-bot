"""
台股AI四核引擎 - 終極極值對撞版
負責載入四個模型，精準對齊特徵，並找出多空雙方最強烈的訊號
"""
import os
import numpy as np
import pandas as pd
import joblib

class DualCoreBrain:
    def __init__(self, lgbm_path="quant_model.joblib", feats_path="model_features.joblib", 
                 lstm_path="lstm_momentum_brain.h5"):
        self.lgbm_model, self.features_list = None, None
        self.lstm_model, self.lstm_scaler = None, None
        self.is_lgbm_ready, self.is_lstm_ready = False, False
        
        self.lgbm_short_model, self.features_short_list = None, None
        self.lstm_short_model, self.lstm_short_scaler = None, None
        self.is_lgbm_short_ready, self.is_lstm_short_ready = False, False
        
        self._load_models(lgbm_path, feats_path, lstm_path)
        self._load_short_models()
        
    def _load_models(self, lgbm_path, feats_path, lstm_path):
        if os.path.exists(lgbm_path) and os.path.exists(feats_path):
            try:
                self.lgbm_model = joblib.load(lgbm_path)
                self.features_list = joblib.load(feats_path)
                self.is_lgbm_ready = True
            except: pass
            
        if os.path.exists(lstm_path) and os.path.exists("lstm_scaler.joblib"):
            try:
                import tensorflow as tf
                self.lstm_model = tf.keras.models.load_model(lstm_path)
                self.lstm_scaler = joblib.load("lstm_scaler.joblib")
                self.is_lstm_ready = True
            except: pass

    def _load_short_models(self):
        if os.path.exists("quant_model_short.joblib") and os.path.exists("model_features_short.joblib"):
            try:
                self.lgbm_short_model = joblib.load("quant_model_short.joblib")
                self.features_short_list = joblib.load("model_features_short.joblib")
                self.is_lgbm_short_ready = True
            except: pass
            
        if os.path.exists("lstm_short_brain.h5") and os.path.exists("lstm_scaler_short.joblib"):
            try:
                import tensorflow as tf
                self.lstm_short_model = tf.keras.models.load_model("lstm_short_brain.h5")
                self.lstm_short_scaler = joblib.load("lstm_scaler_short.joblib")
                self.is_lstm_short_ready = True
            except: pass

    def extract_features(self, clean_ticker, current_price, snapshot_dict, current_vol=0.0, fallback_atr=0.0, fallback_pattern=""):
        item = snapshot_dict.get(clean_ticker, {})
        feat_dict = {}
        
        # 🔥 蒐集多頭與空頭模型所需的所有特徵聯集
        target_features = set()
        if self.features_list: target_features.update(self.features_list)
        if self.features_short_list: target_features.update(self.features_short_list)
        
        # 如果模型都還沒載入，給予預設特徵清單防止報錯
        if not target_features:
            target_features = [
                'daily_return', 'vol_ratio', 'broker_conc', 'rs_index', 'volatility', 'turnover',
                'is_pullback', 'is_squeeze', 'is_divergence', 'is_liquidity_sweep', 'is_poc_rejection'
            ]
        
        for col in target_features:
            if col == 'is_pullback': feat_dict[col] = 1.0 if "回踩" in fallback_pattern else 0.0
            elif col == 'is_squeeze': feat_dict[col] = 1.0 if "壓縮" in fallback_pattern else 0.0
            elif col == 'is_divergence': feat_dict[col] = 1.0 if "背離" in fallback_pattern else 0.0
            elif col == 'is_liquidity_sweep': feat_dict[col] = 1.0 if "掠奪" in fallback_pattern else 0.0
            elif col == 'is_poc_rejection': feat_dict[col] = 1.0 if "POC" in fallback_pattern else 0.0
            else: feat_dict[col] = float(item.get(col, 0.0))
            
        return feat_dict

    def predict_four_core(self, features_list):
        """
        🔥 四核心極值對撞引擎
        分別將特徵精準餵入四個大腦，並進行多空對撞
        """
        if not features_list: return []
        df = pd.DataFrame(features_list)
        
        # 1. LGBM 多頭
        df_lgbm = df[self.features_list].astype(float).fillna(0) if self.features_list else pd.DataFrame()
        lgbm_long = self.lgbm_model.predict_proba(df_lgbm)[:, 1] if self.is_lgbm_ready and not df_lgbm.empty else np.full(len(df), 0.5)
        
        # 2. LGBM 空頭
        df_lgbm_s = df[self.features_short_list].astype(float).fillna(0) if self.features_short_list else pd.DataFrame()
        lgbm_short = self.lgbm_short_model.predict_proba(df_lgbm_s)[:, 1] if self.is_lgbm_short_ready and not df_lgbm_s.empty else np.full(len(df), 0.5)

        # 3. LSTM 多頭與空頭
        lstm_long = np.full(len(df), 0.5)
        lstm_short = np.full(len(df), 0.5)
        
        for i in range(len(df)):
            # 多頭 LSTM (精準選取多頭特徵)
            if self.is_lstm_ready and self.features_list:
                row_vals_l = df[self.features_list].iloc[i].values
                scaled_l = self.lstm_scaler.transform([row_vals_l])
                seq_l = np.tile(scaled_l, (10, 1)).reshape(1, 10, -1)
                lstm_long[i] = self.lstm_model.predict(seq_l, verbose=0)[0][0]
                
            # 空頭 LSTM (精準選取空頭特徵)
            if self.is_lstm_short_ready and self.features_short_list:
                row_vals_s = df[self.features_short_list].iloc[i].values
                scaled_s = self.lstm_short_scaler.transform([row_vals_s])
                seq_s = np.tile(scaled_s, (10, 1)).reshape(1, 10, -1)
                lstm_short[i] = self.lstm_short_model.predict(seq_s, verbose=0)[0][0]

        results = []
        for i in range(len(df)):
            ll, tl = float(lgbm_long[i]), float(lstm_long[i])
            ls, ts = float(lgbm_short[i]), float(lstm_short[i])
            
            # 🔥 取多方與空方各自的極值
            best_long = max(ll, tl)
            best_short = max(ls, ts)
            
            # ⚔️ 極值對撞決策樹
            if best_long > 0.60 and best_long > best_short * 1.2:
                signal = "STRONG_LONG"
            elif best_long > 0.52 and best_long > best_short:
                signal = "LONG"
            elif best_short > 0.60 and best_short > best_long * 1.2:
                signal = "STRONG_SHORT"
            elif best_short > 0.52 and best_short > best_long:
                signal = "SHORT"
            elif best_long > 0.55 and best_short > 0.55:
                signal = "HIGH_VOLATILITY" # 多空都很強，容易雙巴
            else:
                signal = "WAIT" # 動能不足

            results.append({
                'lgbm_long': ll, 'lstm_long': tl,
                'lgbm_short': ls, 'lstm_short': ts,
                'best_long': best_long, 'best_short': best_short,
                'signal': signal
            })
            
        return results

    # 確保與舊版分頁相容
    def predict_win_rates(self, features_list):
        if not hasattr(self, 'predict_four_core'): return [0.5] * len(features_list)
        return [res['best_long'] for res in self.predict_four_core(features_list)]