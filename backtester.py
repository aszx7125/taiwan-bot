import os
import pandas as pd
import numpy as np
from supabase import create_client, Client

# 🔑 請確保您的電腦環境變數有這兩把鑰匙，或者直接在這裡替換為您的字串
SUPABASE_URL = os.environ.get("SUPABASE_URL", "請填入您的_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "請填入您的_SUPABASE_KEY")

def fetch_all_memory():
    """從 Supabase 撈取所有歷史訊號記憶"""
    if SUPABASE_URL.startswith("請填入"):
        print("❌ 請先設定 Supabase URL 與 Key！")
        return pd.DataFrame()

    print("🔗 正在連線至大腦記憶庫 (Supabase)...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # 由於 Supabase API 預設單次最多回傳 1000 筆，我們使用分頁抓取
    all_data = []
    offset = 0
    limit = 1000
    
    while True:
        response = supabase.table("quant_history").select("*").range(offset, offset + limit - 1).execute()
        data = response.data
        if not data:
            break
        all_data.extend(data)
        offset += limit
        
    df = pd.DataFrame(all_data)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    return df

def run_forward_evaluation(df, signal_threshold=70, holding_period=5):
    """
    執行前向測試與期望值運算
    :param df: 資料庫撈出來的 DataFrame
    :param signal_threshold: AI 潛伏分數的進場門檻
    :param holding_period: 假設持有天數 (預設 5 天)
    """
    if df.empty:
        print("⚠️ 資料庫目前無任何記憶，請等待每日 Actions 收集數據。")
        return

    print(f"\n🔬 [啟動策略深度覆盤] 進場條件: AI 分數 >= {signal_threshold} / 持倉週期: {holding_period} 天")
    
    # 1. 為了計算未來收益，我們要把 N 天後的收盤價「往回拉」對齊當天
    df[f'future_close_{holding_period}d'] = df.groupby('ticker')['close_price'].shift(-holding_period)
    
    # 2. 計算真實報酬率
    df[f'return_{holding_period}d'] = (df[f'future_close_{holding_period}d'] - df['close_price']) / df['close_price']
    
    # 3. 濾出「當年觸發進場訊號，且已經有未來 N 天收盤價」的有效樣本
    signals = df[(df['score'] >= signal_threshold) & (df[f'future_close_{holding_period}d'].notna())].copy()
    
    total_signals = len(signals)
    if total_signals == 0:
        print(f"⏸️ 目前尚無完成 {holding_period} 天週期的有效交易樣本，系統持續學習中...")
        # 顯示目前累積了多少訊號正在「等待開獎」
        pending = len(df[(df['score'] >= signal_threshold) & (df[f'future_close_{holding_period}d'].isna())])
        print(f"👀 目前有 {pending} 筆訊號正在等待 {holding_period} 天後的真實市場驗證。")
        return

    # 4. 核心統計學解算
    signals['is_win'] = signals[f'return_{holding_period}d'] > 0
    winning_trades = signals[signals['is_win']]
    losing_trades = signals[~signals['is_win']]
    
    win_rate = len(winning_trades) / total_signals
    avg_win = winning_trades[f'return_{holding_period}d'].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades[f'return_{holding_period}d'].mean() if not losing_trades.empty else 0
    
    # 盈虧比 (Risk-Reward Ratio) 與 期望值 (Expectancy)
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # 5. 印出量化診斷報告
    print("="*50)
    print("📈 AI 演算法真實戰績報告 (Out-of-Sample)")
    print("="*50)
    print(f"總交易次數 (樣本數) : {total_signals} 次")
    print(f"策略真實勝率       : {win_rate * 100:.2f}%")
    print(f"平均獲利 (Win)     : +{avg_win * 100:.2f}%")
    print(f"平均虧損 (Loss)    : {avg_loss * 100:.2f}%")
    print(f"盈虧比 (R/R)       : {rr_ratio:.2f}")
    print(f"每筆交易期望值     : {expectancy * 100:+.2f}%")
    print("-" * 50)
    
    if expectancy > 0.005:
        print("⭐⭐⭐ 狀態：極優異！策略具備正向期望值，演算法邏輯正確。")
    elif expectancy > 0:
        print("⭐⭐ 狀態：微幅獲利。可考慮微調 SMC 指標權重以提高勝率。")
    else:
        print("⚠️ 狀態：出現耗損。請檢查是否遭遇系統性風險(如大盤連續重挫)，或需加嚴進場濾網。")

if __name__ == "__main__":
    memory_df = fetch_all_memory()
    print(f"✅ 成功載入 {len(memory_df)} 筆市場特徵記憶。")
    
    # 執行一週 (5天) 短波段的績效回測
    run_forward_evaluation(memory_df, signal_threshold=70, holding_period=5)
    
    # 執行一個月 (20天) 長波段的績效回測
    run_forward_evaluation(memory_df, signal_threshold=70, holding_period=20)