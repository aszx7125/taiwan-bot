# data_fetcher.py — 修正版
# 修正點：
#   1. broker_conc 改為固定 0.0，移除 random.uniform 雜訊，
#      避免模型學到假相關，實盤預測不穩定。
#   2. get_kline_with_fugle 的 ATR_14 改為動態計算，移除寫死的 Close * 0.03。
#   3. _fetch_and_score_sync 保持原有 recent_returns 打包邏輯不變。

import yfinance as yf
import pandas as pd
import numpy as np


def fetch_yahoo_robust(ticker, period="5d", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty or 'Close' not in df.columns:
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"⚠️ Yahoo API 行情限流攔截器安全介入 [{ticker}]: {e}")
        return pd.DataFrame()


def load_all_market_tickers():
    try:
        return pd.read_csv("all_tw_stocks.csv")
    except Exception:
        return pd.DataFrame(columns=['ticker', 'name'])


def get_market_summary():
    summary = {}
    try:
        df = fetch_yahoo_robust("^TWII", period="5d", interval="1d")
        if not df.empty:
            df = df.dropna(subset=['Close'])
            if len(df) >= 2:
                c = float(df.iloc[-1]['Close'])
                p = float(df.iloc[-2]['Close'])
                summary["加權指數"] = {
                    "price":  c,
                    "change": c - p,
                    "pct":    ((c - p) / p) * 100
                }
    except Exception:
        pass
    if not summary:
        summary["加權指數"] = {"price": 22000.0, "change": 0.0, "pct": 0.0}
    return summary


def get_kline_with_fugle(ticker, api_key):
    """
    修正點：ATR_14 改為動態計算 (TR rolling 14)，
    移除原版 df_daily['ATR_14'] = df_daily['Close'] * 0.03 寫死值。
    """
    clean_ticker = ticker.split('.')[0].strip()
    df_daily, df_hourly = pd.DataFrame(), pd.DataFrame()

    try:
        df_daily = fetch_yahoo_robust(f"{clean_ticker}.TW", period="3mo", interval="1d")
        if df_daily.empty:
            df_daily = fetch_yahoo_robust(f"{clean_ticker}.TWO", period="3mo", interval="1d")

        if not df_daily.empty:
            df_daily = df_daily.sort_index()
            df_daily['Res_20'] = df_daily['Close'].rolling(20).max()
            df_daily['Sup_20'] = df_daily['Close'].rolling(20).min()

            # 動態 ATR_14（修正：不再寫死為 Close * 0.03）
            prev_close = df_daily['Close'].shift(1)
            tr = np.maximum(
                df_daily['High'] - df_daily['Low'],
                np.maximum(
                    (df_daily['High'] - prev_close).abs(),
                    (df_daily['Low']  - prev_close).abs()
                )
            )
            df_daily['ATR_14'] = tr.rolling(14).mean()

            df_daily['Score']                = 50
            df_daily['Broker_Concentration'] = 0.0   # 無真實籌碼數據時固定 0.0
            df_daily['Low_Vol_Pullback']      = False

        df_hourly = fetch_yahoo_robust(f"{clean_ticker}.TW", period="1mo", interval="1h")

    except Exception:
        pass

    return df_daily, df_hourly, f"{clean_ticker}.TW"


def get_stock_news(company_name):
    return [
        {"title": f"【量化追蹤】{company_name} 法人籌碼結構性吸籌顯著", "link": "#"},
        {"title": f"【產業動態】{company_name} 供應鏈動能迎來長週期復甦",  "link": "#"}
    ]


def get_precalculated_market_ret():
    try:
        df = fetch_yahoo_robust("^TWII", period="2mo", interval="1d")
        if not df.empty and len(df) >= 20:
            return (
                float(df.iloc[-1]['Close']) - float(df.iloc[-20]['Close'])
            ) / float(df.iloc[-20]['Close'])
    except Exception:
        pass
    return 0.0


def _fetch_and_score_sync(ticker, market_ret):
    """
    背景排程專用。
    修正點：
      - broker_conc 固定為 0.0，移除 random.uniform 雜訊。
      - volatility / turnover / ATR_14 動態計算保持不變。
      - recent_returns 10 日真實序列保留供 LSTM 使用。
    """
    try:
        df = fetch_yahoo_robust(f"{ticker}.TW",  period="2mo", interval="1d")
        if df.empty:
            df = fetch_yahoo_robust(f"{ticker}.TWO", period="2mo", interval="1d")
        if df.empty or len(df) < 20:
            return None

        c = float(df.iloc[-1]['Close'])
        v = float(df.iloc[-1].get('Volume', 0.0))
        if c <= 0 or v <= 0:
            return None

        # 10 日真實報酬率序列（供 LSTM 實盤用）
        df['daily_return'] = df['Close'].pct_change().fillna(0)
        recent_returns = df['daily_return'].tail(10).tolist()

        res_20 = float(df['Close'].rolling(20).max().iloc[-1])
        sup_20 = float(df['Close'].rolling(20).min().iloc[-1])

        # 動態 ATR_14
        prev_close = df['Close'].shift(1)
        tr = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                (df['High'] - prev_close).abs(),
                (df['Low']  - prev_close).abs()
            )
        )
        atr_14 = float(tr.rolling(14).mean().iloc[-1])

        c_20     = float(df.iloc[-20]['Close'])
        rs_index = ((c - c_20) / c_20 - market_ret) * 100 if c_20 > 0 else 0

        vol_5a   = float(df['Volume'].rolling(5).mean().iloc[-1])
        vol_ratio = v / vol_5a if vol_5a > 0 else 1.0

        volatility = atr_14 / c if c > 0 else 0.0
        turnover   = c * v

        # 形態識別
        pattern = ""
        if vol_ratio < 0.75 and c > sup_20:
            pattern += "量縮回踩 "
        if res_20 > 0 and (res_20 - sup_20) / sup_20 < 0.08:
            pattern += "區間壓縮 "
        if c < df['Close'].rolling(20).mean().iloc[-1] and rs_index > 5:
            pattern += "底背離 "
        if v > vol_5a * 2.5:
            pattern += "流動性掠奪 "
        if not pattern:
            pattern = "一般常態箱體震盪"

        return {
            "代號":         ticker,
            "現價":         c,
            "成交量":       v,
            "Res_20":       res_20,
            "Sup_20":       sup_20,
            "ATR_14":       atr_14,
            "rs_index":     rs_index,
            "vol_ratio":    vol_ratio,
            "volatility":   volatility,
            "turnover":     turnover,
            "broker_conc":  0.0,       # 修正：固定 0.0，移除隨機雜訊
            "pattern":      pattern.strip(),
            "recent_returns": recent_returns
        }
    except Exception:
        return None