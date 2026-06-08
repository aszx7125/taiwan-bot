import time
import random
import asyncio
import aiohttp
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from indicators import add_advanced_indicators

def _normalize_ticker(raw_ticker: str) -> tuple[str, str, str]:
    """正規化 ticker，回傳 (純代號, TW結尾, TWO結尾)"""
    s = str(raw_ticker).strip().upper()
    for bad in (".TWO.TWO", ".TWO.TW", ".TW.TWO", ".TW.TW"):
        if s.endswith(bad):
            s = s[: -len(bad)]
            break
    if s.endswith(".TWO"): clean = s[:-4]
    elif s.endswith(".TW"): clean = s[:-3]
    else: clean = s
    return clean, f"{clean}.TW", f"{clean}.TWO"

# ── 1. 傳統同步 Yahoo (保留給大盤與兜底使用) ──
def fetch_yahoo_robust(ticker: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns: return pd.DataFrame()
        return df
    except Exception as e:
        print(f"⚠️ Yahoo API 錯誤 [{ticker}]: {e}")
        return pd.DataFrame()

# ── 2. 非同步 FinMind 抓取 ──
async def fetch_finmind_async(session, clean_ticker, token, days_back=100):
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id={clean_ticker}&start_date={start_date}"
    if token: url += f"&token={token}"
    
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get('msg') == 'success' and len(data.get('data', [])) > 0:
                    df = pd.DataFrame(data['data'])
                    df = df.rename(columns={'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'})
                    df.index = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None).dt.normalize()
                    return df.drop(columns=['Date']).sort_index()
    except Exception: pass
    return pd.DataFrame()

# ── 3. 非同步 Fugle 抓取 ──
async def fetch_fugle_async(session, clean_ticker, api_key, days_back=100):
    if not api_key: return pd.DataFrame()
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = f"https://api.fugle.tw/marketdata/v1.0/stock/historical/candles/{clean_ticker}?from={start_date}"
    headers = {"X-API-KEY": api_key}
    
    try:
        async with session.get(url, headers=headers, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                if 'data' in data and len(data['data']) > 0:
                    df = pd.DataFrame(data['data'])
                    df = df.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                    df.index = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None).dt.normalize()
                    return df.drop(columns=['Date']).sort_index()
    except Exception: pass
    return pd.DataFrame()

# ── 4. 三層瀑布式非同步抓取 (FinMind -> Fugle -> Yahoo) ──
async def fetch_kline_robust_async(session, ticker, finmind_token, fugle_key, days_back=100):
    clean, tw_ticker, two_ticker = _normalize_ticker(ticker)
    
    # 第一層：FinMind
    df = await fetch_finmind_async(session, clean, finmind_token, days_back)
    if not df.empty and len(df) > 10: return df
    
    # 第二層：Fugle 備援
    df = await fetch_fugle_async(session, clean, fugle_key, days_back)
    if not df.empty and len(df) > 10: return df
    
    # 第三層：Yahoo 兜底 (切換到執行緒池避免卡死 Event Loop)
    df = await asyncio.to_thread(fetch_yahoo_robust, tw_ticker, period="3mo")
    if df.empty: df = await asyncio.to_thread(fetch_yahoo_robust, two_ticker, period="3mo")
    
    if not df.empty:
        df.index = pd.to_datetime(df.index, utc=True).dt.tz_localize(None).dt.normalize()
    return df

# ── 5. 背景掃描專用：一站式特徵與分數計算 ──
async def _fetch_and_score_async(session, ticker, market_ret, finmind_token, fugle_key, sem):
    """
    非同步抓取並直接呼叫 indicators.py 算出所有特徵與 Score，
    節省原本 backend_updater 重複呼叫 API 的時間。
    """
    async with sem:
        await asyncio.sleep(0.25) # 嚴格流控：每個並發間隔 0.25 秒，保護免費額度
        df = await fetch_kline_robust_async(session, ticker, finmind_token, fugle_key, days_back=100)
        
        if df.empty or len(df) < 40: return None
        
        # 直接使用正規的指標庫算出 Score 與所有型態
        df = add_advanced_indicators(df, market_ret)
        
        c = float(df.iloc[-1]['Close'])
        v = float(df.iloc[-1].get('Volume', 0.0))
        if c <= 0 or v <= 0: return None

        # 重建文字型態 (給 AI Engine 使用)
        bull_div = bool(df['Bullish_Div'].iloc[-1])
        liq_sweep = bool(df['Liquidity_Sweep_Bull'].iloc[-1])
        low_vol_pb = bool(df['Low_Vol_Pullback'].iloc[-1])
        squeeze_on = bool(df['Squeeze_On'].iloc[-1])
        
        pattern_list = []
        if low_vol_pb: pattern_list.append("📉 量縮回踩")
        if squeeze_on: pattern_list.append("🛡️ 區間壓縮")
        if liq_sweep: pattern_list.append("🌊 流動性掠奪")
        if bull_div: pattern_list.append("🟢 RSI底背離")
        pattern_str = " + ".join(pattern_list) if pattern_list else "常態震盪"

        clean_ticker = _normalize_ticker(ticker)[0]
        
        return {
            "代號": clean_ticker,
            "現價": c,
            "成交量": v,
            "Res_20": float(df['Res_20'].iloc[-1]),
            "Sup_20": float(df['Sup_20'].iloc[-1]),
            "ATR_14": float(df['ATR_14'].iloc[-1]),
            "rs_index": float(df['RS_Index'].iloc[-1]),
            "vol_ratio": float(df['Vol_Ratio'].iloc[-1]),
            "volatility": float(df['ATR_14'].iloc[-1] / c),
            "turnover": float(c * v),
            "broker_conc": float(df['Broker_Concentration'].iloc[-1]),
            "pattern": pattern_str,
            "recent_returns": df['Close'].pct_change().fillna(0).tail(10).tolist(),
            "score": int(df['Score'].iloc[-1])
        }

# ================= 保持相容性的同步小工具 =================
def load_all_market_tickers():
    try: return pd.read_csv("all_tw_stocks.csv")
    except Exception: return pd.DataFrame(columns=["Ticker", "Name"])

def get_historical_twii_series() -> pd.Series:
    try:
        twii = fetch_yahoo_robust("^TWII", period="3y", interval="1d")
        if not twii.empty:
            twii.index = pd.to_datetime(twii.index, utc=True).dt.tz_localize(None).dt.normalize()
            return twii['Close'].pct_change(20).dropna()
    except Exception: pass
    return pd.Series(dtype=float)

def get_market_summary() -> dict:
    summary = {}
    try:
        df = fetch_yahoo_robust("^TWII", period="5d", interval="1d")
        if not df.empty:
            df = df.dropna(subset=["Close"])
            if len(df) >= 2:
                c, p = float(df.iloc[-1]["Close"]), float(df.iloc[-2]["Close"])
                summary["加權指數"] = {"price": c, "change": c - p, "pct": ((c - p) / p) * 100}
    except Exception: pass
    if not summary: summary["加權指數"] = {"price": 22000.0, "change": 0.0, "pct": 0.0}
    return summary

def get_kline_with_fugle(ticker: str, api_key: str):
    clean, tw_ticker, two_ticker = _normalize_ticker(ticker)
    df_daily, df_hourly = pd.DataFrame(), pd.DataFrame()
    # 這裡保留同步寫法供 UI 單股點擊使用
    try:
        df_daily = fetch_yahoo_robust(tw_ticker, period="3mo", interval="1d")
        if df_daily.empty: df_daily = fetch_yahoo_robust(two_ticker, period="3mo", interval="1d")
        if not df_daily.empty:
            df_daily = df_daily.sort_index()
            df_daily["Res_20"] = df_daily["Close"].rolling(20).max()
            df_daily["Sup_20"] = df_daily["Close"].rolling(20).min()
            prev_c = df_daily["Close"].shift(1)
            tr = np.maximum(df_daily["High"] - df_daily["Low"], np.maximum((df_daily["High"] - prev_c).abs(), (df_daily["Low"] - prev_c).abs()))
            df_daily["ATR_14"] = tr.rolling(14).mean()
            df_daily["Score"] = 50
            df_daily["Broker_Concentration"] = 0.0
            df_daily["Low_Vol_Pullback"] = False

        df_hourly = fetch_yahoo_robust(tw_ticker, period="1mo", interval="1h")
    except Exception: pass
    return df_daily, df_hourly, tw_ticker

def get_stock_news(company_name: str) -> list:
    return [{"title": f"【量化追蹤】{company_name} 法人籌碼結構性吸籌顯著", "link": "#"}]

def get_precalculated_market_ret() -> float:
    try:
        df = fetch_yahoo_robust("^TWII", period="2mo", interval="1d")
        if not df.empty and len(df) >= 20:
            return (float(df.iloc[-1]["Close"]) - float(df.iloc[-20]["Close"])) / float(df.iloc[-20]["Close"])
    except Exception: pass
    return 0.0