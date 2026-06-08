import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

class DualCoreBrain:
    def __init__(self, lgbm_path="quant_model.joblib", feats_path="model_features.joblib", lstm_path="lstm_momentum_brain.h5"):
        self.lgbm_model, self.features_list, self.lstm_model = None, None, None
        self.lstm_scaler = None  # 🔥 新增 Scaler 接收器
        self.is_lgbm_ready, self.is_lstm_ready = False, False
        self._load_models(lgbm_path, feats_path, lstm_path)

    def _load_models(self, lgbm_path, feats_path, lstm_path):
        if os.path.exists(lgbm_path) and os.path.exists(feats_path):
            try:
                self.lgbm_model = joblib.load(lgbm_path)
                self.features_list = joblib.load(feats_path)
                self.is_lgbm_ready = True
            except: pass
            
        if os.path.exists(lstm_path):
            try:
                self.lstm_model = tf.keras.models.load_model(lstm_path)
                # 🔥 同步載入訓練好的縮放器
                scaler_path = "lstm_scaler.joblib"
                if os.path.exists(scaler_path):
                    self.lstm_scaler = joblib.load(scaler_path)
                self.is_lstm_ready = True
            except: pass

    def extract_features(self, clean_ticker, current_price, snapshot_dict, current_vol=0.0, fallback_rs=0.0, fallback_atr=None, fallback_pattern="", fallback_vol=0.0):
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
            
        feat = {}
        feat['is_pullback'] = 1.0 if "量縮回踩" in pat else 0.0
        feat['is_squeeze'] = 1.0 if "區間壓縮" in pat else 0.0
        feat['is_divergence'] = 1.0 if "底背離" in pat else 0.0
        feat['is_liquidity_sweep'] = 1.0 if "流動性掠奪" in pat else 0.0
        feat['is_poc_rejection'] = 1.0 if "POC" in pat else 0.0
        feat['rs_index'] = float(rs)
        feat['vol_ratio'] = float(vol_ratio)
        feat['volatility'] = float(atr / current_price if current_price > 0 else 0.0)
        feat['turnover'] = float(current_price * vol)
        feat['broker_conc'] = float(broker_conc)
        feat['recent_returns'] = recent_returns
        return feat

    def _compute_lstm_batch(self, features_list):
        if not self.is_lstm_ready or self.lstm_scaler is None or not features_list: 
            return np.full(len(features_list), 0.50)
        
        LSTM_ORDER = ['daily_return', 'vol_ratio', 'broker_conc', 'rs_index', 'volatility', 'turnover', 'is_pullback', 'is_squeeze', 'is_divergence', 'is_liquidity_sweep', 'is_poc_rejection']
        try:
            all_seqs = []
            for feat in features_list:
                seq = []
                ret_list = feat.get('recent_returns', [0.0]*10)
                if len(ret_list) < 10: ret_list = [0.0]*(10-len(ret_list)) + ret_list
                ret_list = ret_list[-10:] 
                
                for i in range(10):
                    day_feat = feat.copy()
                    day_feat['daily_return'] = float(ret_list[i])
                    seq.append([day_feat.get(col, 0.0) for col in LSTM_ORDER])
                all_seqs.append(seq)
                
            tensor_3d = np.array(all_seqs, dtype=np.float32)
            
            # 🔥 核心修復：在餵給 LSTM 之前，務必執行 2D 降維縮放再還原回 3D
            n_samples, n_steps, n_features = tensor_3d.shape
            tensor_2d = tensor_3d.reshape(-1, n_features)
            tensor_2d_scaled = self.lstm_scaler.transform(tensor_2d)
            tensor_3d_scaled = tensor_2d_scaled.reshape(n_samples, n_steps, n_features)
            
            return self.lstm_model.predict(tensor_3d_scaled, batch_size=512, verbose=0).flatten()
        except Exception as e: 
            print(f"LSTM 矩陣運算異常: {e}")
            return np.full(len(features_list), 0.50)

    def predict_win_rates(self, features_list):
        if not self.is_lgbm_ready: return np.full(len(features_list), 0.0)
        input_df = pd.DataFrame(features_list)[self.features_list]
        base_probs = self.lgbm_model.predict_proba(input_df)[:, 1]
        lstm_scores = self._compute_lstm_batch(features_list)
        return (base_probs * 0.6) + (lstm_scores * 0.4)