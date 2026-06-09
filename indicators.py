import pandas as pd
import numpy as np


def safe_divide(numerator, denominator, default=0.0):
    """🔥 新增：安全除法，徹底杜絕inf"""
    denominator = np.where(denominator == 0, 1e-10, denominator)
    result = numerator / denominator
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)


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
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs()
        )
    )
    df['ATR_14'] = df['TR'].rolling(14).mean().fillna(df['Close'] * 0.02)  # 🔥 修復：填充NaN

    # ── 1. 波動率收斂突破 (Squeeze) ───────────────────────────────────────
    df['BB_Mid'] = df['SMA_20']
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    df['KC_Upper'] = df['BB_Mid'] + (1.5 * df['ATR_14'])
    df['KC_Lower'] = df['BB_Mid'] - (1.5 * df['ATR_14'])
    df['Squeeze_On'] = (df['BB_Upper'] < df['KC_Upper']) & (df['BB_Lower'] > df['KC_Lower'])

    # ── 2. RSI 動態波段背離 ───────────────────────────────────────────────
    df['Price_Min_5'] = df['Close'].rolling(window=5).min()
    df['RSI_Min_5'] = df['RSI'].rolling(window=5).min()
    df['Prev_Price_Min'] = df['Price_Min_5'].shift(5).rolling(window=15).min()
    df['Prev_RSI_Min'] = df['RSI_Min_5'].shift(5).rolling(window=15).min()
    df['Bullish_Div'] = (
        (df['Price_Min_5'] < df['Prev_Price_Min']) &
        (df['RSI_Min_5'] > df['Prev_RSI_Min']) &
        (df['RSI_Min_5'] < 45)
    )

    df['Price_Max_5'] = df['Close'].rolling(window=5).max()
    df['RSI_Max_5'] = df['RSI'].rolling(window=5).max()
    df['Prev_Price_Max'] = df['Price_Max_5'].shift(5).rolling(window=15).max()
    df['Prev_RSI_Max'] = df['RSI_Max_5'].shift(5).rolling(window=15).max()
    df['Bearish_Div'] = (
        (df['Price_Max_5'] > df['Prev_Price_Max']) &
        (df['RSI_Max_5'] < df['Prev_RSI_Max']) &
        (df['RSI_Max_5'] > 55)
    )

    # ── 3. SMC 機構微觀結構 ───────────────────────────────────────────────
    df['Swing_Low_20'] = df['Low'].shift(1).rolling(20).min()
    df['Liquidity_Sweep_Bull'] = (
        (df['Low'] < df['Swing_Low_20']) &
        (df['Close'] > df['Swing_Low_20'])
    )
    df['FVG_Bull'] = (df['Low'] > df['High'].shift(2)) & (df['Close'] > df['Open'])

    # ── 4. 量縮回踩支撐 ───────────────────────────────────────────────────
    df['Price_Drop'] = df['Close'] < prev_close
    df['Low_Vol_Pullback'] = (
        (df['Close'] > df['SMA_20']) &
        df['Price_Drop'] &
        (df['Volume'] < df['Vol_SMA5'] * 0.8)
    )

    # ── 5. 籌碼代理模型 ───────────────────────────────────────────────────
    df['Vol_Ratio'] = np.where(df['Vol_SMA5'] > 0, df['Volume'] / df['Vol_SMA5'], 1)
    df['Buying_Pressure'] = np.where(
        df['High'] > df['Low'],
        (df['Close'] - df['Low']) / (df['High'] - df['Low']),
        0.5
    )
    df['Block_Trade_Inflow'] = (
        (df['Vol_Ratio'] > 1.8) &
        (df['Buying_Pressure'] > 0.7) &
        (df['Close'] > df['Open'])
    )
    df['Net_Money_Flow'] = df['Volume'] * np.where(
        df['Close'] > df['Open'],
        df['Buying_Pressure'],
        -(1 - df['Buying_Pressure'])
    )
    df['Broker_Concentration'] = df['Net_Money_Flow'].rolling(5).sum() / (df['Vol_SMA5'] * 5 + 1e-10)

    # ── 6. 相對強弱 (RS_Index) ────────────────────────────────────────────
    stock_ret = df['Close'].pct_change(20)
    if market_ret_20 is not None:
        if hasattr(market_ret_20, 'empty') and not market_ret_20.empty:
            df['RS_Index'] = (stock_ret - market_ret_20.reindex(df.index, method='ffill')) * 100
        elif isinstance(market_ret_20, (int, float)):
            df['RS_Index'] = (stock_ret - market_ret_20) * 100
        else:
            df['RS_Index'] = 0.0
    else:
        df['RS_Index'] = 0.0

    # ── 7. 週線趨勢代理 ───────────────────────────────────────────────────
    w_ema12 = df['Close'].ewm(span=60, adjust=False).mean()
    w_ema26 = df['Close'].ewm(span=130, adjust=False).mean()
    w_macd = w_ema12 - w_ema26
    df['Weekly_Trend_Up'] = w_macd > w_macd.ewm(span=45, adjust=False).mean()

    # ── 8. 潛伏型 AI 評分矩陣 ────────────────────────
    df['Score'] = 0

    # 正向加分
    df.loc[df['Close'] > df['SMA_20'], 'Score'] += 10
    df.loc[df['RS_Index'] > 0.0, 'Score'] += 10
    df.loc[df['Broker_Concentration'] > 0.2, 'Score'] += 15
    df.loc[df['Squeeze_On'] == True, 'Score'] += 15
    df.loc[df['Low_Vol_Pullback'] == True, 'Score'] += 20
    df.loc[df['Bullish_Div'] == True, 'Score'] += 15
    df.loc[df['Liquidity_Sweep_Bull'] == True, 'Score'] += 15

    # 🔥 核心修復：懲罰追高（使用安全除法）
    prev_close_safe = prev_close.fillna(df['Close']).replace(0, np.nan).fillna(method='bfill')
    price_change_pct = safe_divide(df['Close'] - prev_close_safe, prev_close_safe, 0.0)
    
    is_chase = (
        (price_change_pct > 0.05) &
        (~df['Liquidity_Sweep_Bull'])
    )
    df.loc[is_chase, 'Score'] -= 15

    # 懲罰爆量
    is_vol_spike_bad = (
        (df['Volume'] > df['Vol_SMA5'] * 2.5) &
        (~df['Liquidity_Sweep_Bull'])
    )
    df.loc[is_vol_spike_bad, 'Score'] -= 15

    df['Score'] = df['Score'].clip(0, 100)

    df['Res_20'] = df['High'].shift(1).rolling(20).max()
    df['Sup_20'] = df['Low'].shift(1).rolling(20).min()

    # 🔥 最終防護：全表清理
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    return df


def add_intraday_indicators(df):
    df = df.dropna(subset=['Close', 'Volume']).copy()
    if df.empty or len(df) < 10:
        return df

    df['SMA_20_1h'] = df['Close'].rolling(20).mean()
    df['Vol_SMA5_1h'] = df['Volume'].rolling(5).mean()

    prev_c = df['Close'].shift(1)
    tr_h = np.maximum(
        df['High'] - df['Low'],
        np.maximum((df['High'] - prev_c).abs(), (df['Low'] - prev_c).abs())
    )
    df['ATR_14_1h'] = tr_h.rolling(14).mean().fillna(df['Close'] * 0.02)

    df['MACD_1h'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_1h'] = df['MACD_1h'].ewm(span=9, adjust=False).mean()

    df['MACD_Cross_Up'] = (df['MACD_1h'] > df['Signal_1h']) & (df['MACD_1h'].shift(1) <= df['Signal_1h'].shift(1))
    df['Vol_Surge_1h'] = df['Volume'] > (df['Vol_SMA5_1h'] * 1.5)
    df['Micro_Sniper_Trigger'] = (
        (df['Close'] > df['SMA_20_1h']) &
        (df['MACD_Cross_Up'] | df['Vol_Surge_1h'])
    )

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df