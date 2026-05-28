# data_fetcher.py
import streamlit as st
import yfinance as yf
import pandas as pd
import aiohttp
import asyncio
from indicators import add_advanced_indicators  # 引入我們的數學模組
from config import load_api_keys, get_ticker_map # 引入設定

# 建立獨立的 session 提高效能
yf_session = requests.Session()

@st.cache_data(ttl=60) 
def get_kline_data(ticker_code):
    """標準獲取單股歷史資料"""
    ticker_map = get_ticker_map()
    symbol = ticker_map.get(ticker_code, f"{ticker_code}.TW")
    
    try:
        df = yf.Ticker(symbol, session=yf_session).history(period="6mo")
        df = add_advanced_indicators(df) # 計算指標
        return df, symbol
    except:
        return pd.DataFrame(), ""

# (這裡可以放入您原本寫好的 async_scan_market 等複雜的異步爬蟲邏輯)