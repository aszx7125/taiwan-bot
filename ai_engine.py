"""
台股AI雙向四核引擎 - 直接覆蓋版
完全相容原有 DualCoreBrain 介面
新增：predict_bidirectional() 方法
"""
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime


class DualCoreBrain:
    """
    雙核大腦 (升級為四核但保持介面相容)
    - 支援原有 predict_win_rates()
    - 新增 predict_bidirectional()
    """
    
    def __init__(self, lgbm_path="quant_model.joblib", feats_path="model_features.joblib", 
                 lstm_path="lstm_momentum_brain.h5"):
        # 原有屬性 (保持相容)
        self.lgbm_model, self.features_list, self.lstm_model = None, None, None
        self.lstm_scaler = None
        self.is_lgbm_ready, self.is_lstm_ready = False, False
        
        # 新增：空頭模型
        self.lgbm_short_model = None
        self.lstm_short_model = None
        self.lstm_short_scaler = None
        self.is_lgbm_short_ready = False
        self.is_lstm_short_ready = False
        
        self._load_models(lgbm_path, feats_path, lstm_path)
        self._load_short_models()
    
    def _load_models(self, lgbm_path, feats_path, lstm_path):
        """載入原有多頭模型 (保持原有邏輯)"""
        if os.path.exists(lgbm_path) and os.path.exists(feats_path):
            try:
                self.lgbm_model = joblib.load(lgbm_path)
                self.features_list = joblib.load(feats_path)
                self.is_lgbm_ready = True
                print(f"[DualCore] LightGBM Long 載入成功")
            except Exception as e:
                print(f"[DualCore] LightGBM Long 載入失敗: {e}")
            
        if os.path.exists(lstm_path):
            try:
                self.lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
                scaler_path = "lstm_scaler.joblib"
                if os.path.exists(scaler_path):
                    self.lstm_scaler = joblib.load(scaler_path)
                    self.is_lstm_ready = True
                    print(f"[DualCore] LSTM Long 載入成功")
            except Exception as e:
                print(f"[DualCore] LSTM Long 載入失敗: {e}")
    
    def _load_short_models(self):
        """載入空頭模型 (新增)"""
        # LightGBM Short
        if os.path.exists("quant_model_short.joblib"):
            try:
                self.lgbm_short_model = joblib.load("quant_model_short.joblib")
                self.is_lgbm_short_ready = True
                print(f"[DualCore] LightGBM Short 載入成功")
            except:
                pass
        
        # LSTM Short
        if os.path.exists("lstm_short_brain.h5"):
            try:
                self.lstm_short_model = tf.keras.models.load_model(
                    "lstm_short_brain.h5", compile=False
                )
                if os.path.exists("lstm_scaler_short.joblib"):
                    self.lstm_short_scaler = joblib.load("lstm_scaler_short.joblib")
                    self.is_lstm_short_ready = True
                    print(f"[DualCore] LSTM Short 載入成功")
            except:
                pass
    
    def extract_features(self, clean_ticker, current_price, snapshot_dict, 
                        current_vol=0.0, fallback_rs=0.0, fallback_atr=None, 
                        fallback_pattern="", fallback_vol=0.0):
        """特徵萃取 (與原版相同)"""
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
            if atr_raw is not None: 
                atr = float(atr_raw)
                
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
        """通用LSTM批次預測"""
        if model is None:
            model = self.lstm_model
            scaler = self.lstm_scaler
            is_ready = self.is_lstm_ready
        else:
            is_ready = True
        
        if not is_ready or scaler is None or not features_list: 
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
            
            preds = model.predict(tensor_3d_scaled, batch_size=512, verbose=0).flatten()
            return np.clip(preds, 0.0, 1.0)
            
        except Exception as e: 
            print(f"LSTM 預測異常: {e}")
            return np.full(len(features_list), 0.50)
    
    def predict_win_rates(self, features_list):
        """
        原有方法 (保持完全相容)
        返回多頭機率 array
        """
        if not self.is_lgbm_ready or not features_list: 
            return np.full(len(features_list), 0.0)
        
        # 清理特徵
        clean_features = []
        for feat in features_list:
            clean_feat = {k: v for k, v in feat.items() if k != 'recent_returns'}
            clean_features.append(clean_feat)
        
        input_df = pd.DataFrame(clean_features)
        for col in self.features_list:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[self.features_list]
        input_df = input_df.replace([np.inf, -np.inf], 0).fillna(0).clip(-1e6, 1e6)
        
        try:
            base_probs = self.lgbm_model.predict_proba(input_df)[:, 1]
            base_probs = np.clip(base_probs, 0.0, 1.0)
        except:
            base_probs = np.full(len(features_list), 0.5)
        
        lstm_scores = self._compute_lstm_batch(features_list)
        final_probs = (base_probs * 0.6) + (lstm_scores * 0.4)
        final_probs = np.nan_to_num(final_probs, nan=0.5, posinf=0.99, neginf=0.01)
        
        return np.clip(final_probs, 0.01, 0.99)
    
    def predict_bidirectional(self, features_list):
        """
        🔥 新增方法：雙向預測
        
        Returns:
            dict: {
                'long_prob': array,
                'short_prob': array,
                'neutral_prob': array,
                'signal': list
            }
        """
        if not features_list:
            return {
                'long_prob': np.array([]),
                'short_prob': np.array([]),
                'neutral_prob': np.array([]),
                'signal': []
            }
        
        # 多頭預測 (使用原有方法)
        long_prob = self.predict_win_rates(features_list)
        
        # 空頭預測
        n = len(features_list)
        short_prob = np.full(n, 0.5)
        
        if self.is_lgbm_short_ready:
            # 清理特徵
            clean_features = []
            for feat in features_list:
                clean_feat = {k: v for k, v in feat.items() if k != 'recent_returns'}
                clean_features.append(clean_feat)
            
            input_df = pd.DataFrame(clean_features)
            for col in self.features_list:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            input_df = input_df[self.features_list]
            input_df = input_df.replace([np.inf, -np.inf], 0).fillna(0).clip(-1e6, 1e6)
            
            try:
                short_lgbm = self.lgbm_short_model.predict_proba(input_df)[:, 1]
                short_lgbm = np.clip(short_lgbm, 0, 1)
            except:
                short_lgbm = 1.0 - long_prob
            
            # LSTM Short
            if self.is_lstm_short_ready:
                short_lstm = self._compute_lstm_batch(
                    features_list, 
                    self.lstm_short_model, 
                    self.lstm_short_scaler
                )
            else:
                short_lstm = 1.0 - self._compute_lstm_batch(features_list)
            
            short_prob = (short_lgbm * 0.6) + (short_lstm * 0.4)
        else:
            # Fallback: 用 1 - 多頭 估算
            short_prob = 1.0 - long_prob
        
        short_prob = np.nan_to_num(short_prob, nan=0.5, posinf=0.99, neginf=0.01)
        short_prob = np.clip(short_prob, 0.01, 0.99)
        
        # 計算中性
        neutral_prob = np.maximum(0, 1.0 - long_prob - short_prob)
        total = long_prob + short_prob + neutral_prob
        long_prob = long_prob / total
        short_prob = short_prob / total
        neutral_prob = neutral_prob / total
        
        # 產生訊號
        signals = []
        for i in range(n):
            lp, sp, np_ = long_prob[i], short_prob[i], neutral_prob[i]
            
            if lp > 0.6 and lp > sp * 1.5:
                signal = "STRONG_LONG"
            elif lp > 0.55 and lp > sp:
                signal = "LONG"
            elif sp > 0.6 and sp > lp * 1.5:
                signal = "STRONG_SHORT"
            elif sp > 0.55 and sp > lp:
                signal = "SHORT"
            elif np_ > 0.5:
                signal = "NEUTRAL"
            elif lp > 0.5 and sp > 0.5:
                signal = "HIGH_VOLATILITY"
            else:
                signal = "WAIT"
            
            signals.append(signal)
        
        return {
            'long_prob': long_prob,
            'short_prob': short_prob,
            'neutral_prob': neutral_prob,
            'signal': signals,
            'confidence': np.maximum(long_prob, short_prob)
        }