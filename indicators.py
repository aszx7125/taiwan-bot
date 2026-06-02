import pandas as pd
import numpy as np

def add_advanced_indicators(df, market_ret_20=None):
    df = df.dropna(subset=['Close', 'Volume']).copy()
    if df.empty or len(df) < 40: 
        return df
    
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_60'] = df['Close'].rolling(60).mean()
    df['Vol_SMA5'] = df['Volume'].rolling(5).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    prev_close = df['Close'].shift(1)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum((df['High'] - prev_close).abs(), (df['Low'] - prev_close).abs()))
    df['ATR_14'] = df['TR'].rolling(14).mean()

    # 1. 波動率收斂突破 (Squeeze) - 獎勵「正在壓縮」，而非已經突破
    df['BB_Mid'] = df['SMA_20']
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    df['KC_Upper'] = df['BB_Mid'] + (1.5 * df['ATR_14'])
    df['KC_Lower'] = df['BB_Mid'] - (1.5 * df['ATR_14'])
    df['Squeeze_On'] = (df['BB_Upper'] < df['KC_Upper']) & (df['BB_Lower'] > df['KC_Lower'])

    # 2. RSI 動態波段背離掃描 (尋找跌無可跌的左側)
    df['Price_Min_5'] = df['Close'].rolling(window=5).min()
    df['RSI_Min_5'] = df['RSI'].rolling(window=5).min()
    df['Prev_Price_Min'] = df['Price_Min_5'].shift(5).rolling(window=15).min()
    df['Prev_RSI_Min'] = df['RSI_Min_5'].shift(5).rolling(window=15).min()
    df['Bullish_Div'] = (df['Price_Min_5'] < df['Prev_Price_Min']) & (df['RSI_Min_5'] > df['Prev_RSI_Min']) & (df['RSI_Min_5'] < 45)
    
    df['Price_Max_5'] = df['Close'].rolling(window=5).max()
    df['RSI_Max_5'] = df['RSI'].rolling(window=5).max()
    df['Prev_Price_Max'] = df['Price_Max_5'].shift(5).rolling(window=15).max()
    df['Prev_RSI_Max'] = df['RSI_Max_5'].shift(5).rolling(window=15).max()
    df['Bearish_Div'] = (df['Price_Max_5'] > df['Prev_Price_Max']) & (df['RSI_Max_5'] < df['Prev_RSI_Max']) & (df['RSI_Max_5'] > 55)

    # 3. SMC 機構微觀結構
    df['Swing_Low_20'] = df['Low'].shift(1).rolling(20).min()
    df['Liquidity_Sweep_Bull'] = (df['Low'] < df['Swing_Low_20']) & (df['Close'] > df['Swing_Low_20'])
    df['FVG_Bull'] = (df['Low'] > df['High'].shift(2)) & (df['Close'] > df['Open'])
    
    # 🌟 4. 新增：量縮回踩支撐 (Low Volume Pullback) - 最完美的安全進場點
    # 邏輯：維持在月線之上(趨勢偏多)，今日價格下跌，且成交量顯著萎縮(小於5日均量80%)
    df['Price_Drop'] = df['Close'] < prev_close
    df['Low_Vol_Pullback'] = (df['Close'] > df['SMA_20']) & df['Price_Drop'] & (df['Volume'] < df['Vol_SMA5'] * 0.8)

    # 5. 籌碼代理模型
    df['Vol_Ratio'] = np.where(df['Vol_SMA5'] > 0, df['Volume'] / df['Vol_SMA5'], 1)
    df['Buying_Pressure'] = np.where(df['High'] > df['Low'], (df['Close'] - df['Low']) / (df['High'] - df['Low']), 0.5)
    df['Block_Trade_Inflow'] = (df['Vol_Ratio'] > 1.8) & (df['Buying_Pressure'] > 0.7) & (df['Close'] > df['Open'])
    df['Net_Money_Flow'] = df['Volume'] * np.where(df['Close'] > df['Open'], df['Buying_Pressure'], - (1 - df['Buying_Pressure']))
    df['Broker_Concentration'] = df['Net_Money_Flow'].rolling(5).sum() / (df['Vol_SMA5'] * 5 + 1e-10)

    if market_ret_20 is not None and not market_ret_20.empty:
        stock_ret = df['Close'].pct_change(20)
        df['RS_Index'] = stock_ret - market_ret_20.reindex(df.index, method='ffill')
    else: 
        df['RS_Index'] = 0.0

    w_ema12 = df['Close'].ewm(span=60, adjust=False).mean()
    w_ema26 = df['Close'].ewm(span=130, adjust=False).mean()
    w_macd = w_ema12 - w_ema26
    df['Weekly_Trend_Up'] = w_macd > w_macd.ewm(span=45, adjust=False).mean()

    # ==========================================
    # 🎯 終極優化：潛伏型 AI 評分矩陣 (滿分 100)
    # 取消大漲與爆量追高的分數，將分數配給「潛伏壓縮」與「量縮回踩」
    # ==========================================
    df['Score'] = 0
    # A. 基礎防護 (20分)
    df.loc[df['Close'] > df['SMA_20'], 'Score'] += 10                 # 站穩月線 (長線保護)
    df.loc[df['RS_Index'] > 0.0, 'Score'] += 10                       # 跑贏大盤 (相對強勢)
    
    # B. 主力暗中建倉痕跡 (30分)
    df.loc[df['Broker_Concentration'] > 0.2, 'Score'] += 15           # 贏家分點持續囤貨
    df.loc[df['Squeeze_On'] == True, 'Score'] += 15                   # 🌟 正在極度壓縮中 (尚未爆發，適合潛伏)
    
    # C. 完美左側/回測進場點 (50分) - 這決定了勝率期待值
    df.loc[df['Low_Vol_Pullback'] == True, 'Score'] += 20             # 🌟 量縮回踩 (最好、最安全的進場點)
    df.loc[df['Bullish_Div'] == True, 'Score'] += 15                  # RSI底背離 (跌無可跌)
    df.loc[df['Liquidity_Sweep_Bull'] == True, 'Score'] += 15         # 破底翻 (主力剛洗盤完畢)

    # 🚨 懲罰機制 (過熱降溫)
    # 如果今天已經噴出大於 5%，或者成交量大於均量 2.5 倍，強制扣 15 分，防止追高
    df.loc[(df['Close'] - prev_close) / prev_close > 0.05, 'Score'] -= 15
    df.loc[df['Volume'] > df['Vol_SMA5'] * 2.5, 'Score'] -= 15

    # 確保分數在 0~100 之間
    df['Score'] = df['Score'].clip(0, 100)

    df['Res_20'] = df['High'].shift(1).rolling(20).max()
    df['Sup_20'] = df['Low'].shift(1).rolling(20).min()
    
    return df