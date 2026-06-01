import pandas as pd
import numpy as np

def add_advanced_indicators(df, market_df=None):
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
    
    df['TR'] = df[['High', 'Low', 'Close']].apply(
        lambda x: max(x['High'] - x['Low'], 
                      abs(x['High'] - df['Close'].shift(1).get(x.name, x['High'])), 
                      abs(x['Low'] - df['Close'].shift(1).get(x.name, x['Low']))), axis=1
    )
    df['ATR_14'] = df['TR'].rolling(14).mean()

    # 1. 波動率收斂突破 (Squeeze)
    df['BB_Mid'] = df['SMA_20']
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    df['KC_Upper'] = df['BB_Mid'] + (1.5 * df['ATR_14'])
    df['KC_Lower'] = df['BB_Mid'] - (1.5 * df['ATR_14'])
    df['Squeeze_On'] = (df['BB_Upper'] < df['KC_Upper']) & (df['BB_Lower'] > df['KC_Lower'])

    # 2. RSI 動態波段背離掃描 (Divergence Scanner)
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

    # 3. SMC 機構級微觀結構分析
    df['Swing_Low_20'] = df['Low'].shift(1).rolling(20).min()
    df['Liquidity_Sweep_Bull'] = (df['Low'] < df['Swing_Low_20']) & (df['Close'] > df['Swing_Low_20'])
    df['FVG_Bull'] = (df['Low'] > df['High'].shift(2)) & (df['Close'] > df['Open'])

    # 4. 🌟 新增：機構特大單流 (Block Trade Flow) 代理模型
    # 邏輯：當日爆量且收紅，且收盤價逼近最高價，視為大單淨流入
    df['Vol_Ratio'] = np.where(df['Vol_SMA5'] > 0, df['Volume'] / df['Vol_SMA5'], 1)
    df['Buying_Pressure'] = np.where(df['High'] > df['Low'], (df['Close'] - df['Low']) / (df['High'] - df['Low']), 0.5)
    df['Block_Trade_Inflow'] = (df['Vol_Ratio'] > 1.8) & (df['Buying_Pressure'] > 0.7) & (df['Close'] > df['Open'])

    # 5. 🌟 新增：贏家分點集中度 (Broker Concentration) 代理模型
    # 邏輯：近 5 日內，若大單買進的累積量遠大於均量，視為特定分點正在囤貨
    df['Net_Money_Flow'] = df['Volume'] * np.where(df['Close'] > df['Open'], df['Buying_Pressure'], - (1 - df['Buying_Pressure']))
    df['Broker_Concentration'] = df['Net_Money_Flow'].rolling(5).sum() / (df['Vol_SMA5'] * 5 + 1e-10)

    # 6. 大盤相對強度 (RS Index)
    if market_df is not None and not market_df.empty:
        market_df = market_df.copy()
        market_df.index = pd.to_datetime(market_df.index).tz_localize(None).normalize()
        market_df = market_df[~market_df.index.duplicated(keep='last')]
        stock_ret = df['Close'].pct_change(20)
        mkt_ret = market_df['Close'].reindex(df.index, method='ffill').pct_change(20)
        df['RS_Index'] = stock_ret - mkt_ret
    else: df['RS_Index'] = 0.0

    # 7. 多時區週線共振
    w_ema12 = df['Close'].ewm(span=60, adjust=False).mean()
    w_ema26 = df['Close'].ewm(span=130, adjust=False).mean()
    w_macd = w_ema12 - w_ema26
    df['Weekly_Trend_Up'] = w_macd > w_macd.ewm(span=45, adjust=False).mean()

    # 8. 🏆 終極版 AI 評分系統 (滿分 100) - 加入大單與分點權重
    df['Score'] = 0
    df.loc[df['Close'] > df['SMA_20'], 'Score'] += 5                  # 趨勢防護 (5)
    df.loc[df['MACD'] > df['Signal'], 'Score'] += 5                   # 動能金叉 (5)
    df.loc[df['RS_Index'] > 0.02, 'Score'] += 10                      # 領漲抗跌 (10)
    df.loc[df['Weekly_Trend_Up'] == True, 'Score'] += 10              # 週線共振 (10)
    df.loc[(df['Squeeze_On'].shift(1) == True) & (df['Close'] > df['BB_Upper']), 'Score'] += 10 # 壓縮突破 (10)
    df.loc[df['Bullish_Div'] == True, 'Score'] += 10                  # 左側背離 (10)
    df.loc[df['Liquidity_Sweep_Bull'] == True, 'Score'] += 10         # SMC破底翻 (10)
    df.loc[df['FVG_Bull'] == True, 'Score'] += 10                     # SMC缺口 (10)
    # 🌟 新增機構級籌碼權重
    df.loc[df['Block_Trade_Inflow'] == True, 'Score'] += 10           # 大單暴量敲進 (10)
    df.loc[df['Broker_Concentration'] > 0.3, 'Score'] += 10           # 分點集中囤貨 (10)

    df['Res_20'] = df['High'].shift(1).rolling(20).max()
    df['Sup_20'] = df['Low'].shift(1).rolling(20).min()
    
    return df