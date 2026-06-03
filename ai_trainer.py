def prepare_features_and_labels(df, holding_period=5):
    print(f"🧹 正在清洗數據與萃取特徵 (目標預測 {holding_period} 天後漲跌)...")
    
    df[f'future_close_{holding_period}d'] = df.groupby('ticker')['close_price'].shift(-holding_period)
    df = df.dropna(subset=[f'future_close_{holding_period}d']).copy()
    df['is_win'] = (df[f'future_close_{holding_period}d'] > df['close_price']).astype(int)
    
    df['pattern'] = df['pattern'].fillna("")
    df['is_pullback'] = df['pattern'].str.contains("量縮回踩").astype(int)
    df['is_sweep'] = df['pattern'].str.contains("流動性掠奪").astype(int)
    df['is_squeeze'] = df['pattern'].str.contains("區間壓縮").astype(int)
    df['is_divergence'] = df['pattern'].str.contains("底背離").astype(int)
    
    df['rs_index'] = pd.to_numeric(df['rs_index'], errors='coerce').fillna(0)
    
    # 🚀 將新特徵納入訓練矩陣
    df['volatility'] = pd.to_numeric(df['volatility'], errors='coerce').fillna(0)
    df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce').fillna(0)
    
    # 🆕 告訴 AI：以後判斷股票，除了型態，還要看波動率和市值規模！
    feature_columns = ['is_pullback', 'is_sweep', 'is_squeeze', 'is_divergence', 'rs_index', 'volatility', 'turnover']
    
    X = df[feature_columns]
    y = df['is_win']
    
    print(f"📊 成功萃取 {len(X)} 筆有效訓練樣本！")
    return X, y, feature_columns