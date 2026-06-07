# data_fetcher.py — 限流修正版
# 修正點：
#   1. fetch_yahoo_robust 加入 exponential backoff 重試（最多 3 次），
#      遇到 429 Rate Limit 時自動等待後重試，不直接放棄。
#   2. _fetch_and_score_sync 修正 Ticker 格式問題：
#      若 CSV 的 Ticker 欄位已含 .TW/.TWO 後綴，避免重複拼接變成
#      8923.TWO.TW 或 8923.TWO.TWO 的錯誤格式。
#   3. 新增 _normalize_ticker() 工具函數，統一處理所有 ticker 格式。
#   4. 新增模組層級的 _REQUEST_SEMAPHORE，將並發請求從 10 個
#      限制為最多 3 個同時進行，從源頭降低被限流機率。

import time
import random
import threading
import yfinance as yf
import pandas as pd
import numpy as np

# ── 並發節流器：最多同時 3 個請求，避免觸發 Yahoo 429 ─────────────────────
_REQUEST_SEMAPHORE = threading.Semaphore(3)

# ── Ticker 格式正規化 ──────────────────────────────────────────────────────

def _normalize_ticker(raw_ticker: str) -> tuple[str, str, str]:
    """
    將任意格式的 ticker 正規化，回傳 (clean_code, tw_ticker, two_ticker)。

    處理的格式：
      - "2330"        → clean="2330",  tw="2330.TW",   two="2330.TWO"
      - "2330.TW"     → clean="2330",  tw="2330.TW",   two="2330.TWO"
      - "3491.TWO"    → clean="3491",  tw="3491.TW",   two="3491.TWO"
      - "3491.TWO.TW" → clean="3491",  tw="3491.TW",   two="3491.TWO"  ← 修正重複後綴
    """
    s = str(raw_ticker).strip().upper()

    # 移除重複後綴（如 8923.TWO.TWO 或 8923.TWO.TW）
    for bad in [".TWO.TWO", ".TWO.TW", ".TW.TWO", ".TW.TW"]:
        if s.endswith(bad):
            s = s[: -len(bad)]
            break

    # 移除合法後綴，取得純代號
    if s.endswith(".TWO"):
        clean = s[:-4]
    elif s.endswith(".TW"):
        clean = s[:-3]
    else:
        clean = s

    return clean, f"{clean}.TW", f"{clean}.TWO"


# ── 強固抓取（含 exponential backoff）────────────────────────────────────

def fetch_yahoo_robust(ticker: str, period: str = "5d",
                       interval: str = "1d", max_retries: int = 3) -> pd.DataFrame:
    """
    帶退避重試的 Yahoo Finance 抓取函數。
    遇到 429 / Too Many Requests 時，等待 2^attempt 秒後重試，最多 3 次。
    使用 Semaphore 控制最大並發數為 3。
    """
    with _REQUEST_SEMAPHORE:
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                df    = stock.history(period=period, interval=interval, auto_adjust=True)
                if df is None or df.empty or 'Close' not in df.columns:
                    return pd.DataFrame()
                return df

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = (
                    "too many requests" in err_str or
                    "rate limit"        in err_str or
                    "429"               in err_str
                )

                if is_rate_limit and attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                    print(f"⏳ 限流退避 [{ticker}]：等待 {wait:.1f}s 後重試 "
                          f"(第 {attempt + 1}/{max_retries - 1} 次)...")
                    time.sleep(wait)
                else:
                    if is_rate_limit:
                        print(f"⚠️ Yahoo API 限流放棄 [{ticker}]：已重試 {max_retries} 次。")
                    else:
                        print(f"⚠️ Yahoo API 錯誤 [{ticker}]: {e}")
                    return pd.DataFrame()

        return pd.DataFrame()


# ── 其他工具函數 ───────────────────────────────────────────────────────────

def load_all_market_tickers() -> pd.DataFrame:
    try:
        return pd.read_csv("all_tw_stocks.csv")
    except Exception:
        return pd.DataFrame(columns=['Ticker', 'Name'])


def get_market_summary() -> dict:
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


def get_kline_with_fugle(ticker: str, api_key: str):
    """
    修正：使用 _normalize_ticker 確保格式正確，ATR_14 動態計算。
    """
    clean, tw_ticker, two_ticker = _normalize_ticker(ticker)
    df_daily, df_hourly = pd.DataFrame(), pd.DataFrame()

    try:
        df_daily = fetch_yahoo_robust(tw_ticker, period="3mo", interval="1d")
        if df_daily.empty:
            df_daily = fetch_yahoo_robust(two_ticker, period="3mo", interval="1d")

        if not df_daily.empty:
            df_daily = df_daily.sort_index()
            df_daily['Res_20'] = df_daily['Close'].rolling(20).max()
            df_daily['Sup_20'] = df_daily['Close'].rolling(20).min()

            prev_close = df_daily['Close'].shift(1)
            tr = np.maximum(
                df_daily['High'] - df_daily['Low'],
                np.maximum(
                    (df_daily['High'] - prev_close).abs(),
                    (df_daily['Low']  - prev_close).abs()
                )
            )
            df_daily['ATR_14']               = tr.rolling(14).mean()
            df_daily['Score']                = 50
            df_daily['Broker_Concentration'] = 0.0
            df_daily['Low_Vol_Pullback']      = False

        df_hourly = fetch_yahoo_robust(tw_ticker, period="1mo", interval="1h")

    except Exception:
        pass

    return df_daily, df_hourly, tw_ticker


def get_stock_news(company_name: str) -> list:
    return [
        {"title": f"【量化追蹤】{company_name} 法人籌碼結構性吸籌顯著", "link": "#"},
        {"title": f"【產業動態】{company_name} 供應鏈動能迎來長週期復甦",  "link": "#"}
    ]


def get_precalculated_market_ret() -> float:
    try:
        df = fetch_yahoo_robust("^TWII", period="2mo", interval="1d")
        if not df.empty and len(df) >= 20:
            return (
                float(df.iloc[-1]['Close']) - float(df.iloc[-20]['Close'])
            ) / float(df.iloc[-20]['Close'])
    except Exception:
        pass
    return 0.0


def _fetch_and_score_sync(ticker: str, market_ret: float) -> dict | None:
    """
    背景排程專用。
    修正：使用 _normalize_ticker 解決 8923.TWO.TW / 8923.TWO.TWO 雙重後綴問題。
    """
    try:
        clean, tw_ticker, two_ticker = _normalize_ticker(ticker)

        df = fetch_yahoo_robust(tw_ticker, period="2mo", interval="1d")
        if df.empty:
            df = fetch_yahoo_robust(two_ticker, period="2mo", interval="1d")
        if df.empty or len(df) < 20:
            return None

        c = float(df.iloc[-1]['Close'])
        v = float(df.iloc[-1].get('Volume', 0.0))
        if c <= 0 or v <= 0:
            return None

        df['daily_return'] = df['Close'].pct_change().fillna(0)
        recent_returns     = df['daily_return'].tail(10).tolist()

        res_20 = float(df['Close'].rolling(20).max().iloc[-1])
        sup_20 = float(df['Close'].rolling(20).min().iloc[-1])

        prev_close = df['Close'].shift(1)
        tr = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                (df['High'] - prev_close).abs(),
                (df['Low']  - prev_close).abs()
            )
        )
        atr_14 = float(tr.rolling(14).mean().iloc[-1])

        c_20      = float(df.iloc[-20]['Close'])
        rs_index  = ((c - c_20) / c_20 - market_ret) * 100 if c_20 > 0 else 0

        vol_5a    = float(df['Volume'].rolling(5).mean().iloc[-1])
        vol_ratio = v / vol_5a if vol_5a > 0 else 1.0

        volatility = atr_14 / c if c > 0 else 0.0
        turnover   = c * v

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
            "代號":           clean,       # 回傳純代號，不含後綴
            "現價":           c,
            "成交量":         v,
            "Res_20":         res_20,
            "Sup_20":         sup_20,
            "ATR_14":         atr_14,
            "rs_index":       rs_index,
            "vol_ratio":      vol_ratio,
            "volatility":     volatility,
            "turnover":       turnover,
            "broker_conc":    0.0,
            "pattern":        pattern.strip(),
            "recent_returns": recent_returns
        }

    except Exception:
        return None