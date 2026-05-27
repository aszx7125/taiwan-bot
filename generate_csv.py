import requests
import pandas as pd

print("⏳ 正在從證交所與櫃買中心獲取最新台股名單...")

try:
    # 1. 抓取上市 (TWSE) 官方資料
    twse_url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    twse_data = requests.get(twse_url, timeout=10).json()
    twse_df = pd.DataFrame(twse_data)
    twse_df['Ticker'] = twse_df['Code'] + '.TW'
    twse_df = twse_df[['Ticker', 'Name']]

    # 2. 抓取上櫃 (TPEx) 官方資料
    tpex_url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
    tpex_data = requests.get(tpex_url, timeout=10).json()
    tpex_df = pd.DataFrame(tpex_data)
    tpex_df['Ticker'] = tpex_df['SecuritiesCompanyCode'] + '.TWO'
    tpex_df['Name'] = tpex_df['CompanyName']
    tpex_df = tpex_df[['Ticker', 'Name']]

    # 3. 合併並存檔
    all_stocks = pd.concat([twse_df, tpex_df], ignore_index=True)
    all_stocks.to_csv('all_tw_stocks.csv', index=False, encoding='utf-8-sig')
    
    print(f"✅ 成功生成 all_tw_stocks.csv！共包含 {len(all_stocks)} 檔股票。")
    print("👉 請將這個 CSV 檔案與您的 app.py 一起上傳到 GitHub！")

except Exception as e:
    print(f"❌ 發生錯誤: {e}")