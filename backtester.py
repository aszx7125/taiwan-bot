# backtester.py — 修正版
# 修正點：
#   1. run_forward_evaluation 改用雙重門檻：
#      score >= signal_threshold（現在寫入的是真實分數）
#      OR ai_prob >= 0.5（若資料庫已有 ai_prob 欄位則優先使用）。
#   2. 加入自動偵測：若 score 欄位全為 0（舊資料），
#      自動 fallback 改用 ai_prob 欄位，並印出警告。
#   3. 新增 show_top_signals() 輸出最近觸發訊號的詳細清單，
#      方便人工覆核。

import os
import pandas as pd
import numpy as np
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "請填入您的_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "請填入您的_SUPABASE_KEY")


def fetch_all_memory() -> pd.DataFrame:
    """從 Supabase 撈取所有歷史訊號記憶"""
    if SUPABASE_URL.startswith("請填入"):
        print("❌ 請先設定 Supabase URL 與 Key！")
        return pd.DataFrame()

    print("🔗 正在連線至大腦記憶庫 (Supabase)...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    all_data, offset, limit = [], 0, 1000
    while True:
        response = (
            supabase.table("quant_history")
            .select("*")
            .range(offset, offset + limit - 1)
            .execute()
        )
        if not response.data:
            break
        all_data.extend(response.data)
        offset += limit

    df = pd.DataFrame(all_data)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)

        # 數值欄位強制轉型
        for col in ['close_price', 'rs_index', 'volatility', 'turnover',
                    'vol_ratio', 'broker_conc', 'score']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        if 'ai_prob' in df.columns:
            df['ai_prob'] = pd.to_numeric(df['ai_prob'], errors='coerce').fillna(0.5)

    return df


def _detect_signal_column(df: pd.DataFrame, signal_threshold: float) -> tuple[pd.Series, str]:
    """
    自動偵測應使用哪個欄位作為進場訊號。
    優先使用真實 score；若 score 全為 0（舊資料），fallback 到 ai_prob。
    回傳 (signal_mask, 使用的欄位名稱)。
    """
    if 'score' in df.columns and df['score'].max() > 0:
        mask = df['score'] >= signal_threshold
        return mask, f"score >= {signal_threshold}"

    if 'ai_prob' in df.columns:
        prob_threshold = signal_threshold / 100.0   # 把 0-100 分數對應到 0-1 機率
        print(f"⚠️  score 欄位全為 0（舊資料），自動 fallback 改用 ai_prob >= {prob_threshold:.2f} 作為進場條件。")
        mask = df['ai_prob'] >= prob_threshold
        return mask, f"ai_prob >= {prob_threshold:.2f}"

    print("⚠️  找不到有效進場訊號欄位，無法執行回測。")
    return pd.Series(False, index=df.index), "無有效欄位"


def run_forward_evaluation(df: pd.DataFrame,
                           signal_threshold: int = 70,
                           holding_period:   int = 5) -> None:
    """
    執行前向測試與期望值運算。
    :param df:               資料庫撈出來的 DataFrame
    :param signal_threshold: 進場分數門檻（0-100）
    :param holding_period:   假設持有天數
    """
    if df.empty:
        print("⚠️ 資料庫目前無任何記憶，請等待每日 Actions 收集數據。")
        return

    print(f"\n🔬 [啟動策略深度覆盤] 持倉週期: {holding_period} 天")

    # 自動偵測訊號欄位
    entry_mask, signal_desc = _detect_signal_column(df, signal_threshold)
    print(f"   進場條件: {signal_desc}")

    # 計算未來收益
    df = df.copy()
    df[f'future_close_{holding_period}d'] = df.groupby('ticker')['close_price'].shift(-holding_period)
    df[f'return_{holding_period}d']       = (
        (df[f'future_close_{holding_period}d'] - df['close_price']) / df['close_price']
    )

    # 篩出已完成週期的有效樣本
    valid_future = df[f'future_close_{holding_period}d'].notna()
    signals      = df[entry_mask & valid_future].copy()

    total_signals = len(signals)
    if total_signals == 0:
        pending = len(df[entry_mask & ~valid_future])
        print(f"⏸️  目前尚無完成 {holding_period} 天週期的有效交易樣本，系統持續學習中...")
        print(f"👀 目前有 {pending} 筆訊號正在等待 {holding_period} 天後的真實市場驗證。")
        return

    # 核心統計
    signals['is_win']  = signals[f'return_{holding_period}d'] > 0
    winning_trades     = signals[signals['is_win']]
    losing_trades      = signals[~signals['is_win']]

    win_rate = len(winning_trades) / total_signals
    avg_win  = winning_trades[f'return_{holding_period}d'].mean() if not winning_trades.empty else 0.0
    avg_loss = losing_trades[f'return_{holding_period}d'].mean()  if not losing_trades.empty  else 0.0

    rr_ratio   = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # 報告
    print("=" * 50)
    print("📈 AI 演算法真實戰績報告 (Out-of-Sample)")
    print("=" * 50)
    print(f"總交易次數 (樣本數) : {total_signals} 次")
    print(f"策略真實勝率       : {win_rate * 100:.2f}%")
    print(f"平均獲利 (Win)     : +{avg_win  * 100:.2f}%")
    print(f"平均虧損 (Loss)    : {avg_loss  * 100:.2f}%")
    print(f"盈虧比 (R/R)       : {rr_ratio:.2f}")
    print(f"每筆交易期望值     : {expectancy * 100:+.2f}%")
    print("-" * 50)

    if expectancy > 0.005:
        print("⭐⭐⭐ 狀態：極優異！策略具備正向期望值，演算法邏輯正確。")
    elif expectancy > 0:
        print("⭐⭐ 狀態：微幅獲利。可考慮微調 SMC 指標權重以提高勝率。")
    else:
        print("⚠️  狀態：出現耗損。請檢查是否遭遇系統性風險，或需加嚴進場濾網。")


def show_top_signals(df: pd.DataFrame, signal_threshold: int = 70, top_n: int = 20) -> None:
    """
    顯示最近觸發進場訊號的標的清單，供人工覆核。
    """
    if df.empty:
        return

    entry_mask, signal_desc = _detect_signal_column(df, signal_threshold)
    recent = (
        df[entry_mask]
        .sort_values('date', ascending=False)
        .head(top_n)
    )

    print(f"\n📋 最近 {top_n} 筆觸發訊號（{signal_desc}）")
    print("=" * 70)
    cols = ['date', 'ticker', 'close_price', 'score', 'pattern']
    if 'ai_prob' in recent.columns:
        cols.append('ai_prob')
    print(recent[cols].to_string(index=False))


if __name__ == "__main__":
    memory_df = fetch_all_memory()
    print(f"✅ 成功載入 {len(memory_df)} 筆市場特徵記憶。")

    # 顯示最近觸發訊號
    show_top_signals(memory_df, signal_threshold=70, top_n=20)

    # 5 天短波段回測
    run_forward_evaluation(memory_df, signal_threshold=70, holding_period=5)

    # 20 天長波段回測
    run_forward_evaluation(memory_df, signal_threshold=70, holding_period=20)