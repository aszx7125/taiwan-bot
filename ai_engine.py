import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

class DualCoreBrain:
    """
    台股量化雙核運算引擎 (OOP 封裝版)
    結合 LightGBM (靜態特徵) 與 LSTM (時序動能) 進行混合決策推論
    """
    
    def __init__(self, lgbm_path="quant_model.joblib", feats_path="model_features.joblib", lstm_path="lstm_momentum_brain.h5"):
        # 實例化時，立刻載入模型並保存在物件的記憶體中 (self)
        self.lgbm_model = None
        self.features_list = None
        self.lstm_model = None
        
        self.is_lgbm_ready = False
        self.is_lstm_ready = False
        
        self._load_models(lgbm_path, feats_path, lstm_path)

    def _load_models(self, lgbm_path, feats_path, lstm_path):
        """內部方法：負責安全載入模型檔"""
        if os.path.exists(lgbm_path) and os.path.exists(feats_path):
            try:
                self.lgbm_model = joblib.load(lgbm_path)
                self.features_list = joblib.load(feats_path)
                self.is_lgbm_ready = True
                print("🌳 LightGBM 靜態大腦載入成功")
            except Exception as e:
                print(f"❌ LightGBM 載入失敗: {e}")

        if os.path.exists(lstm_path):
            try:
                self.lstm_model = tf.keras.models.load_model(lstm_path)
                self.is_lstm_ready = True
                print("🔮 LSTM 深度大腦載入成功")
            except Exception as e:
                print(f"❌ LSTM 載入失敗: {e}")

    def extract_features(self, clean_ticker, current_price, snapshot_dict, current_vol=0.0, fallback_rs=0.0, fallback_atr=None, fallback_pattern="", fallback_vol=0.0):
        """將原始市場資料轉換為 AI 看得懂的特徵字典"""
        rs_idx = fallback_rs
        pat = fallback_pattern
        anchor_price = current_price
        base_vol = fallback_vol 
        atr = fallback_atr if fallback_atr else (anchor_price * 0.05)
        vol_ratio = 1.0
        broker_conc = 0.0

        if snapshot_dict and clean_ticker in snapshot_dict:
            item = snapshot_dict[clean_ticker]
            anchor_price = float(item.get('現價', item.get('close_price', item.get('Close', anchor_price))))
            
            vol_raw = item.get('成交量', item.get('Volume', item.get('volume', None)))
            if vol_raw is not None:
                try: base_vol = float(vol_raw)
                except: pass
                
            pat_raw = item.get('pattern', item.get('Pattern', item.get('型態', pat)))
            if pat_raw: pat = str(pat_raw)
            
            rs_raw = item.get('RS_Index', item.get('rs_index', None))
            if rs_raw is not None:
                try: rs_idx = float(str(rs_raw).replace('%', '').strip())
                except: pass
                
            atr_raw = item.get('ATR_14', item.get('atr_14', None))
            if atr_raw is not None:
                try: atr = float(atr_raw)
                except: pass
                
            vol_ratio = float(item.get('vol_ratio', item.get('Vol_Ratio', 1.0)))
            broker_conc = float(item.get('broker_conc', item.get('Broker_Concentration', 0.0)))

        if base_vol <= 0 and current_vol > 0: base_vol = current_vol
        if anchor_price <= 0: anchor_price = 1.0

        volatility = float(atr / anchor_price)
        turnover = float(anchor_price * base_vol)
        if 0 < turnover < 100_000_000: turnover *= 1000

        return {
            'is_pullback': 1.0 if "量縮回踩" in pat else 0.0,
            'is_squeeze': 1.0 if "區間壓縮" in pat else 0.0,
            'is_divergence': 1.0 if "底背離" in pat else 0.0,
            'is_liquidity_sweep': 1.0 if "流動性掠奪" in pat else 0.0,
            'is_poc_rejection': 1.0 if "POC" in pat else 0.0,
            'rs_index': float(rs_idx),
            'vol_ratio': float(vol_ratio),
            'volatility': float(volatility),
            'turnover': float(turnover),
            'broker_conc': float(broker_conc)
        }

    def _compute_lstm_batch(self, features_list):
        """內部方法：LSTM 極速矩陣批次運算"""
        if not self.is_lstm_ready or not features_list: 
            return np.full(len(features_list), 0.50)
            
        LSTM_FEATURE_ORDER = ['daily_return', 'vol_ratio', 'broker_conc', 'rs_index', 'volatility', 'turnover', 'is_pullback', 'is_squeeze', 'is_divergence', 'is_liquidity_sweep', 'is_poc_rejection']
        
        try:
            all_seqs = []
            for feat in features_list:
                seq = []
                for i in range(10):
                    decay = 1.0 - (0.01 * (9 - i))
                    day_feat = feat.copy()
                    day_feat['daily_return'] = float(feat.get('rs_index', 0.0) * 0.001 * decay)
                    day_feat['vol_ratio'] = float(feat.get('vol_ratio', 1.0) * decay)
                    ordered_feat = [day_feat.get(col, 0.0) for col in LSTM_FEATURE_ORDER]
                    seq.append(ordered_feat)
                all_seqs.append(seq)
                
            tensor_3d = np.array(all_seqs, dtype=np.float32)
            scores = self.lstm_model.predict(tensor_3d, batch_size=512, verbose=0).flatten()
            return scores
        except Exception as e: 
            print(f"LSTM 矩陣運算異常: {e}")
            return np.full(len(features_list), 0.50)

    def predict_win_rates(self, features_list):
        """
        核心對外接口：輸入特徵陣列，直接輸出雙核加權後的最終勝率陣列！
        """
        if not self.is_lgbm_ready:
            return np.full(len(features_list), 0.0)
            
        # 1. 喚醒靜態大腦
        input_df = pd.DataFrame(features_list, columns=self.features_list).astype(float).fillna(0)
        base_probs = self.lgbm_model.predict_proba(input_df)[:, 1]
        
        # 2. 喚醒動態大腦
        lstm_scores = self._compute_lstm_batch(features_list)
        
        # 3. 雙核矩陣融合 (0.6 : 0.4)
        final_probs = (base_probs * 0.6) + (lstm_scores * 0.4)
        return final_probs