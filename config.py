# config.py
import streamlit as st

def load_api_keys():
    try: return st.secrets["FUGLE_API_KEY"]
    except: return ""

STOCK_CLUSTERS = {
    "半導體": ["2330.TW", "3711.TW", "2454.TW", "2303.TW", "5347.TWO", "3034.TW"],
    "伺服器": ["2382.TW", "3231.TW", "6669.TW", "2376.TW", "3017.TW", "5274.TWO"],
    "低軌衛星": ["3491.TWO", "3138.TW", "6285.TW", "2383.TW", "2314.TW"]
}

STOCK_NAMES = {
    "2330": "台積電", "2382": "廣達", "3491": "昇達科", 
    "3711": "日月光投控", "3231": "緯創", "3138": "耀登"
}

def get_ticker_map():
    return {t.split('.')[0]: t for tickers in STOCK_CLUSTERS.values() for t in tickers}