"""
台股AI雙向四核引擎 - 極速批次推論版 (解決卡頓)
修復：將單筆 predict 迴圈改為 Tensor Batch 批次平行推論，載入時間從數分鐘縮短為 1 秒。
"""
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

class DualCoreBrain:
    def __init__(self, lgbm_path="quant_model.joblib", feats_path="model_features.joblib", 
                 lstm_path="lstm_momentum_brain.h5"):
        self.lgbm_model, self.features_list, self.lstm_model = None, None, None
        self.lstm_scaler = None
        self.is_lgbm_ready, self.is_lstm_ready = False, False
        
        self.lgbm_short_model = None
        self.lstm_short_model = None
        self.lstm_short_scaler = None
        self.is_lgbm_short_ready = False
        self.is_lstm_short_ready = False
        
        self._load_models(lgbm_path, feats_path, lstm_path)
        self._load_short_models()
    
    def _load_models(self, lgbm_path, feats_path, lstm_path):
        if os.path.exists(lgbm_path) and os.path.exists(feats_path):
            try:
                self.lgbm_model = joblib.load(lgbm_path)
                self.features_list = joblib.load(feats_path)
                self.is_lgbm_ready = True
            except: pass
            
        if os.path.exists(lstm_path):
            try:
                self.lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
                scaler_path = "lstm_scaler.joblib"
                if os.path.exists(scaler_path):
                    self.lstm_scaler = joblib.load(scaler_path)
                    self.is_lstm_ready = True
            except: pass
    
    def _load_short_models(self):
        if os.path.exists("quant_model_short.joblib"):
            try:
                self.lgbm_short_model = joblib.load("quant_model_short.joblib")
                if os.path.exists("model_features_short.joblib"):
                    self.features_short_list = joblib.load("model_features_short.joblib")
                else:
                    self.features_short_list = self.features_list
                self.is_lgbm_short_ready = True
            except: pass
        
        if os.path.exists("lstm_short_brain.h5"):
            try:
                self.lstm_short_model = tf.keras.models.load_model("lstm_short_brain.h5", compile=False)
                if os.path.exists("lstm_scaler_short.joblib"):
                    self.lstm_short_scaler = joblib.load("lstm_scaler_short.joblib")
                    self.is_lstm_short_ready = True
            except: pass
    
    def extract_features(self, clean_ticker, current_price, snapshot_dict, 
                        current_vol=0.0, fallback_rs=0.0, fallback_atr=None, 
                        fallback_pattern="", fallback_vol=0.0):
        pat = fallback_pattern
        rs = fallback_rs
        atr = fallback_atr if fallback_atr else (current_price * 0.05)
        vol = current_vol if current_vol > 0 else fallback_vol
        vol_ratio = 1.0
        broker_conc = 0.0
        recent_returns = [0.0] * 10
        
        if snapshot_dict and clean_ticker in snapshot_dict:
            item = snapshot_dict[clean_ticker]
            pat = str(item.get('pattern', pat))
            rs = float(item.get('rs_index', rs))
            vol_ratio = float(item.get('vol_ratio', vol_ratio))
            atr_raw = item.get('ATR_14', item.get('atr_14'))
            if atr_raw is not None: atr = float(atr_raw)
            vol = float(item.get('成交量', vol))
            broker_conc = float(item.get('broker_conc', broker_conc))
            recent_returns = item.get('recent_returns', [0.0] * 10)
        
        current_price = max(float(current_price), 0.01)
        vol = max(float(vol), 0.0)
        
        feat = {}
        feat['is_pullback'] = 1.0 if "量縮回踩" in pat else 0.0
        feat['is_squeeze'] = 1.0 if "區間壓縮" in pat else 0.0
        feat['is_divergence'] = 1.0 if "底背離" in pat else 0.0
        feat['is_liquidity_sweep'] = 1.0 if "流動性掠奪" in pat else 0.0
        feat['is_poc_rejection'] = 1.0 if "POC" in pat else 0.0
        feat['rs_index'] = float(np.nan_to_num(rs, nan=0.0, posinf=100.0, neginf=0.0))
        feat['vol_ratio'] = float(np.nan_to_num(vol_ratio, nan=1.0, posinf=10.0, neginf=0.1))
        feat['volatility'] = float(np.nan_to_num(atr / current_price, nan=0.05, posinf=0.5, neginf=0.0))
        feat['turnover'] = float(np.nan_to_num(current_price * vol, nan=0.0, posinf=1e12, neginf=0.0))
        feat['broker_conc'] = float(np.nan_to_num(broker_conc, nan=0.0, posinf=1.0, neginf=0.0))
        
        recent_returns = np.array(recent_returns[-10:], dtype=np.float32)
        recent_returns = np.nan_to_num(recent_returns, nan=0.0, posinf=0.2, neginf=-0.2)
        feat['recent_returns'] = recent_returns.tolist()
        
        return feat
    
    def _compute_lstm_batch(self, features_list, model=None, scaler=None):
        """核心修復：使用 3D 張量陣列一次性餵給 GPU/CPU 平行推論"""
        if not features_list or model is None or scaler is None:
            return np.full(len(features_list), 0.50)
        
        LSTM_ORDER = ['daily_return', 'vol_ratio', 'broker_conc', 'rs_index', 
                     'volatility', 'turnover', 'is_pullback', 'is_squeeze', 
                     'is_divergence', 'is_liquidity_sweep', 'is_poc_rejection']
        try:
            all_seqs = []
            for feat in features_list:
                seq = []
                ret_list = feat.get('recent_returns', [0.0]*10)
                if len(ret_list) < 10: 
                    ret_list = [0.0]*(10-len(ret_list)) + ret_list
                ret_list = ret_list[-10:] 
                
                # 重建過去 10 日的時序特徵
                for i in range(10):
                    day_feat = feat.copy()
                    day_feat['daily_return'] = float(ret_list[i])
                    row = [np.nan_to_num(day_feat.get(col, 0.0), nan=0.0, posinf=1.0, neginf=-1.0) 
                           for col in LSTM_ORDER]
                    seq.append(row)
                all_seqs.append(seq)
                
            tensor_3d = np.array(all_seqs, dtype=np.float32)
            tensor_3d = np.nan_to_num(tensor_3d, nan=0.0, posinf=1.0, neginf=-1.0)
            tensor_3d = np.clip(tensor_3d, -10, 10)
            
            n_samples, n_steps, n_features = tensor_3d.shape
            tensor_2d = tensor_3d.reshape(-1, n_features)
            tensor_2d_scaled = scaler.transform(tensor_2d)
            tensor_2d_scaled = np.nan_to_num(tensor_2d_scaled, nan=0.0, posinf=5.0, neginf=-5.0)
            tensor_3d_scaled = tensor_2d_scaled.reshape(n_samples, n_steps, n_features)
            
            # 🔥 極速推論：把2000檔股票打包成一個 Batch 直接算完
            preds = model.predict(tensor_3d_scaled, batch_size=512, verbose=0).flatten()
            return np.clip(preds, 0.0, 1.0)
            
        except Exception as e: 
            print(f"LSTM 批次預測異常: {e}")
            return np.full(len(features_list), 0.50)

    def predict_four_core(self, features_list):
        if not features_list: return []
        
        # 移除 recent_returns 避免 LightGBM 無法解析串列 (List)
        clean_features = [{k: v for k, v in f.items() if k != 'recent_returns'} for f in features_list]
        df = pd.DataFrame(clean_features)
        
        # 1. LGBM 多頭
        if self.is_lgbm_ready and self.features_list:
            for col in self.features_list:
                if col not in df.columns: df[col] = 0.0
            df_lgbm = df[self.features_list].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
            try:
                lgbm_long = self.lgbm_model.predict_proba(df_lgbm)[:, 1]
            except:
                lgbm_long = np.full(len(df), 0.5)
        else:
            lgbm_long = np.full(len(df), 0.5)
        
        # 2. LGBM 空頭
        if self.is_lgbm_short_ready and hasattr(self, 'features_short_list'):
            for col in self.features_short_list:
                if col not in df.columns: df[col] = 0.0
            df_lgbm_s = df[self.features_short_list].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
            try:
                lgbm_short = self.lgbm_short_model.predict_proba(df_lgbm_s)[:, 1]
            except:
                lgbm_short = np.full(len(df), 0.5)
        else:
            lgbm_short = np.full(len(df), 0.5)

        # 3. LSTM 多頭與空頭 (🔥 使用批次推論瞬間完成)
        if self.is_lstm_ready:
            lstm_long = self._compute_lstm_batch(features_list, self.lstm_model, self.lstm_scaler)
        else:
            lstm_long = np.full(len(df), 0.5)

        if self.is_lstm_short_ready:
            lstm_short = self._compute_lstm_batch(features_list, self.lstm_short_model, self.lstm_short_scaler)
        else:
            lstm_short = np.full(len(df), 0.5)

        results = []
        for i in range(len(df)):
            ll, tl = float(lgbm_long[i]), float(lstm_long[i])
            ls, ts = float(lgbm_short[i]), float(lstm_short[i])
            
            best_long = max(ll, tl)
            best_short = max(ls, ts)
            
            if best_long > 0.60 and best_long > best_short * 1.2: signal = "STRONG_LONG"
            elif best_long > 0.52 and best_long > best_short: signal = "LONG"
            elif best_short > 0.60 and best_short > best_long * 1.2: signal = "STRONG_SHORT"
            elif best_short > 0.52 and best_short > best_long: signal = "SHORT"
            elif best_long > 0.55 and best_short > 0.55: signal = "HIGH_VOLATILITY"
            else: signal = "WAIT"

            results.append({
                'lgbm_long': ll, 'lstm_long': tl,
                'lgbm_short': ls, 'lstm_short': ts,
                'best_long': best_long, 'best_short': best_short,
                'signal': signal
            })
            
        return results

    def predict_win_rates(self, features_list):
        if not hasattr(self, 'predict_four_core'): return [0.5] * len(features_list)
        return [res['best_long'] for res in self.predict_four_core(features_list)]