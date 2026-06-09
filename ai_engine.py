import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime

class DualCoreBrain:
    def __init__(self, lgbm_path="quant_model.joblib", feats_path="model_features.joblib", 
                 lstm_path="lstm_momentum_brain.h5", dynamic_weight_path="fusion_weights.json"):
        self.lgbm_model, self.features_list, self.lstm_model = None, None, None
        self.lstm_scaler = None
        self.is_lgbm_ready, self.is_lstm_ready = False, False
        self.dynamic_weight_path = dynamic_weight_path
        self._load_models(lgbm_path, feats_path, lstm_path)

    def _load_models(self, lgbm_path, feats_path, lstm_path):
        """載入模型 - 增加詳細錯誤日誌"""
        if os.path.exists(lgbm_path) and os.path.exists(feats_path):
            try:
                self.lgbm_model = joblib.load(lgbm_path)
                self.features_list = joblib.load(feats_path)
                self.is_lgbm_ready = True
                print(f"[DualCore] LightGBM載入成功 | 特徵數: {len(self.features_list)}")
            except Exception as e:
                print(f"[DualCore] LightGBM載入失敗: {e}")
            
        if os.path.exists(lstm_path):
            try:
                self.lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
                scaler_path = "lstm_scaler.joblib"
                if os.path.exists(scaler_path):
                    self.lstm_scaler = joblib.load(scaler_path)
                    self.is_lstm_ready = True
                    print(f"[DualCore] LSTM載入成功")
                else:
                    print(f"[DualCore] 警告: 找不到 lstm_scaler.joblib")
            except Exception as e:
                print(f"[DualCore] LSTM載入失敗: {e}")

    def extract_features(self, clean_ticker, current_price, snapshot_dict, 
                        current_vol=0.0, fallback_rs=0.0, fallback_atr=None, 
                        fallback_pattern="", fallback_vol=0.0):
        """特徵萃取 - 嚴格遵守Infinity抹除鐵律"""
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
        
        # 🔥 鐵律1: 所有除法前先防呆
        current_price = max(float(current_price), 0.01)  # 避免除零
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
        
        # 🔥 關鍵修復: recent_returns必須清理inf
        recent_returns = np.array(recent_returns[-10:], dtype=np.float32)
        recent_returns = np.nan_to_num(recent_returns, nan=0.0, posinf=0.2, neginf=-0.2)
        feat['recent_returns'] = recent_returns.tolist()
        
        return feat

    def _compute_lstm_batch(self, features_list):
        """LSTM批次預測 - 完整Infinity抹除"""
        if not self.is_lstm_ready or self.lstm_scaler is None or not features_list: 
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
                    # 🔥 鐵律1: 逐行清理
                    row = [np.nan_to_num(day_feat.get(col, 0.0), nan=0.0, posinf=1.0, neginf=-1.0) 
                           for col in LSTM_ORDER]
                    seq.append(row)
                all_seqs.append(seq)
                
            tensor_3d = np.array(all_seqs, dtype=np.float32)
            
            # 🔥 鐵律1: 餵給scaler前強制抹除
            tensor_3d = np.nan_to_num(tensor_3d, nan=0.0, posinf=1.0, neginf=-1.0)
            tensor_3d = np.clip(tensor_3d, -10, 10)  # 極端值裁剪
            
            n_samples, n_steps, n_features = tensor_3d.shape
            tensor_2d = tensor_3d.reshape(-1, n_features)
            tensor_2d_scaled = self.lstm_scaler.transform(tensor_2d)
            
            # 🔥 再次清理scaler輸出
            tensor_2d_scaled = np.nan_to_num(tensor_2d_scaled, nan=0.0, posinf=5.0, neginf=-5.0)
            tensor_3d_scaled = tensor_2d_scaled.reshape(n_samples, n_steps, n_features)
            
            preds = self.lstm_model.predict(tensor_3d_scaled, batch_size=512, verbose=0).flatten()
            return np.clip(preds, 0.0, 1.0)  # 確保機率範圍
            
        except Exception as e: 
            print(f"[LSTM] 矩陣運算異常: {e}")
            import traceback
            traceback.print_exc()
            return np.full(len(features_list), 0.50)

    def _get_dynamic_weights(self):
        """🔥 新增: 動態權重機制 - 根據近期實盤表現自動調整"""
        # 預設權重
        w_lgbm, w_lstm = 0.6, 0.4
        
        try:
            if os.path.exists(self.dynamic_weight_path):
                import json
                with open(self.dynamic_weight_path, 'r') as f:
                    weights = json.load(f)
                    # 根據最近20日實盤勝率調整
                    lgbm_win = weights.get('lgbm_20d_winrate', 0.55)
                    lstm_win = weights.get('lstm_20d_winrate', 0.55)
                    
                    total = lgbm_win + lstm_win
                    if total > 0:
                        w_lgbm = lgbm_win / total
                        w_lstm = lstm_win / total
                        # 平滑處理，避免極端權重
                        w_lgbm = np.clip(w_lgbm, 0.3, 0.7)
                        w_lstm = 1.0 - w_lgbm
        except:
            pass
        
        return w_lgbm, w_lstm

    def predict_win_rates(self, features_list):
        """雙核融合預測 - 修復特徵污染問題"""
        if not self.is_lgbm_ready or not features_list: 
            return np.full(len(features_list), 0.0)
        
        # 🔥 鐵律2: 關鍵修復 - 移除recent_returns避免污染LightGBM
        clean_features = []
        for feat in features_list:
            clean_feat = {k: v for k, v in feat.items() if k != 'recent_returns'}
            clean_features.append(clean_feat)
        
        # 建立DataFrame並嚴格對齊特徵
        input_df = pd.DataFrame(clean_features)
        
        # 確保所有特徵都存在
        for col in self.features_list:
            if col not in input_df.columns:
                input_df[col] = 0.0
        
        input_df = input_df[self.features_list]
        
        # 🔥 鐵律1: 餵給LightGBM前強制抹除
        input_df = input_df.replace([np.inf, -np.inf], 0)
        input_df = input_df.fillna(0)
        input_df = input_df.clip(-1e6, 1e6)  # 極端值保護
        
        try:
            base_probs = self.lgbm_model.predict_proba(input_df)[:, 1]
            base_probs = np.clip(base_probs, 0.0, 1.0)
        except Exception as e:
            print(f"[LGBM] 預測失敗: {e}")
            base_probs = np.full(len(features_list), 0.5)
        
        lstm_scores = self._compute_lstm_batch(features_list)
        
        # 🔥 動態權重融合
        w_lgbm, w_lstm = self._get_dynamic_weights()
        final_probs = (base_probs * w_lgbm) + (lstm_scores * w_lstm)
        
        # 最終清理
        final_probs = np.nan_to_num(final_probs, nan=0.5, posinf=1.0, neginf=0.0)
        final_probs = np.clip(final_probs, 0.01, 0.99)  # 避免極端0/1
        
        print(f"[Fusion] LGBM:{w_lgbm:.2f} LSTM:{w_lstm:.2f} | 均值:{final_probs.mean():.3f}")
        
        return final_probs