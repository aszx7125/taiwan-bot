import pandas as pd
import numpy as np

def add_advanced_indicators(df, market_df=None):
    if df.empty or len(df) < 35: 
        return df
        
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

    df['BB_Mid'] = df['SMA_20']
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    df['KC_Upper'] = df['BB_Mid'] + (1.5 * df['ATR_14'])
    df['KC_Lower'] = df['BB_Mid'] - (1.5 * df['ATR_14'])
    df['Squeeze_On'] = (df['BB_Upper'] < df['KC_Upper']) & (df['BB_Lower'] > df['KC_Lower'])

    if market_df is not None and not market_df.empty:
        stock_ret = df['Close'].pct_change(20)
        mkt_ret = market_df['Close'].reindex(df.index, method='ffill').pct_change(20)
        df['RS_Index'] = stock_ret - mkt_ret
    else:
        df['RS_Index'] = pd.Series(0, index=df.index)

    w_ema12 = df['Close'].ewm(span=60, adjust=False).mean()
    w_ema26 = df['Close'].ewm(span=130, adjust=False).mean()
    w_macd = w_ema12 - w_ema26
    w_signal = w_macd.ewm(span=45, adjust=False).mean()
    df['Weekly_Trend_Up'] = w_macd > w_signal

    # 評分系統
    df['Score'] = 0
    df.loc[df['Close'] > df['SMA_20'], 'Score'] += 15
    df.loc[df['Volume'] > df['Vol_SMA5'] * 1.5, 'Score'] += 20
    df.loc[df['MACD'] > df['Signal'], 'Score'] += 20
    df.loc[df['RS_Index'] > 0.02, 'Score'] += 15
    df.loc[df['Weekly_Trend_Up'] == True, 'Score'] += 15
    df.loc[(df['Squeeze_On'].shift(1) == True) & (df['Close'] > df['BB_Upper']), 'Score'] += 15

    df['Res_20'] = df['High'].shift(1).rolling(20).max()
    df['Sup_20'] = df['Low'].shift(1).rolling(20).min()
    
    return df