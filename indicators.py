# indicators.py
import pandas as pd

def add_advanced_indicators(df):
    if df.empty or len(df) < 30: return df
    
    # 移動平均線
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['Vol_SMA5'] = df['Volume'].rolling(5).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    
    # MACD
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 支撐壓力
    df['Res_20'] = df['High'].shift(1).rolling(20).max()
    df['Sup_20'] = df['Low'].shift(1).rolling(20).min()
    
    return df