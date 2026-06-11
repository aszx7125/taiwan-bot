"""
UI組件 - 終極防縮排 Bug 版
修復 HTML 被 Streamlit 誤判為程式碼區塊 (Code Block) 的問題
"""
import streamlit as st

def render_top20_card(s):
    color = s.get('box_color', '#00cc96')
    prob = s.get('win_prob', 0) * 100
    html = f"""
    <div style="border: 2px solid {color}; border-radius: 10px; padding: 18px; background-color: #1e1e1e; margin-bottom: 12px;">
        <h4 style="color: {color}; margin-top: 0;">🎯 {s['ticker']} {s['name']}</h4>
        <div style="display: flex; justify-content: space-between;">
            <div><span style="color: gray; font-size: 13px;">最高勝率極值</span><br>
                <b style="font-size: 22px; color: {color};">{prob:.1f}%</b></div>
            <div style="text-align: right;"><span style="color: gray; font-size: 13px;">現價</span><br>
                <b style="font-size: 20px;">{s['entry_price']:.2f}</b></div>
        </div>
    </div>
    """
    st.markdown(html.replace('\n', ''), unsafe_allow_html=True)


def render_single_diagnostic_card(core_data, entry_price, res_level, sup_level):
    bl = core_data['best_long'] * 100
    bs = core_data['best_short'] * 100
    ll = core_data['lgbm_long'] * 100
    tl = core_data['lstm_long'] * 100
    ls = core_data['lgbm_short'] * 100
    ts = core_data['lstm_short'] * 100
    signal = core_data['signal']

    if signal == "STRONG_LONG": box_color, rec = "#00cc96", "⭐⭐⭐ 強勢做多 (多方輾壓)"
    elif signal == "LONG": box_color, rec = "#4ade80", "⭐⭐ 偏多操作 (多方優勢)"
    elif signal == "STRONG_SHORT": box_color, rec = "#ff4b4b", "⚠️⚠️ 強勢放空 (空方輾壓)"
    elif signal == "SHORT": box_color, rec = "#ff8080", "⚠️ 偏空操作 (空方優勢)"
    elif signal == "HIGH_VOLATILITY": box_color, rec = "#ffa500", "⚡ 多空雙巴，建議空手觀望"
    else: box_color, rec = "#a8a8a8", "⚪ 動能不足，盤整觀望"

    html = f"""
    <div style="border: 2px solid {box_color}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;">
        <h4 style="color: {box_color}; margin-top: 0; margin-bottom: 20px; border-bottom: 1px solid #333; padding-bottom: 10px;">
            ⚔️ 四核心對撞結果：{rec}
        </h4>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; margin-bottom: 15px;">
            <div style="flex: 1; min-width: 200px; padding: 10px; background-color: rgba(0, 204, 150, 0.05); border-radius: 8px; margin-right: 10px;">
                <span style="color: #00cc96; font-size: 16px; font-weight: bold;">🟢 多頭陣營極值</span><br>
                <b style="font-size: 32px; color: #00cc96;">{bl:.1f}%</b>
                <div style="font-size: 12px; color: gray; margin-top: 8px;">
                    ▶ LGBM 靜態結構: <span style="color: white;">{ll:.1f}%</span><br>
                    ▶ LSTM 時序動能: <span style="color: white;">{tl:.1f}%</span>
                </div>
            </div>
            <div style="flex: 1; min-width: 200px; padding: 10px; background-color: rgba(255, 75, 75, 0.05); border-radius: 8px; margin-left: 10px;">
                <span style="color: #ff4b4b; font-size: 16px; font-weight: bold;">🔴 空頭陣營極值</span><br>
                <b style="font-size: 32px; color: #ff4b4b;">{bs:.1f}%</b>
                <div style="font-size: 12px; color: gray; margin-top: 8px;">
                    ▶ LGBM 靜態結構: <span style="color: white;">{ls:.1f}%</span><br>
                    ▶ LSTM 時序動能: <span style="color: white;">{ts:.1f}%</span>
                </div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; border-top: 1px solid #333; padding-top: 15px;">
            <div style="flex: 1;"><span style="color: gray; font-size: 12px;">現價</span><br><b style="font-size: 18px;">{entry_price:.2f}</b></div>
            <div style="flex: 1;"><span style="color: gray; font-size: 12px;">上檔壓力位</span><br><b style="font-size: 18px; color: #ffc107;">{res_level:.2f}</b></div>
            <div style="flex: 1;"><span style="color: gray; font-size: 12px;">下檔支撐位</span><br><b style="font-size: 18px; color: #00ccff;">{sup_level:.2f}</b></div>
        </div>
    </div>
    """
    # 🔥 關鍵：抹除換行符號，防止 Streamlit 解析為程式碼區塊
    st.markdown(html.replace('\n', ''), unsafe_allow_html=True)


def render_backtest_metric_card(title, value, subtext, color):
    html = f"""
    <div style="background-color: #121218; padding: 20px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #2a2a35;">
        <div style="color: #8b8b99; font-size: 13px; margin-bottom: 6px;">{title}</div>
        <div style="color: {color}; font-size: 28px; font-weight: 700;">{value}</div>
        <div style="color: #6b6b79; font-size: 12px;">{subtext}</div>
    </div>
    """
    st.markdown(html.replace('\n', ''), unsafe_allow_html=True)


def render_model_health_board(metrics):
    st.markdown("### 🧪 四核心AI大腦：盲測勝率")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟢 多頭模型**")
        wr_l = metrics.get('lgbm', {}).get('blind_win_rate', 0)
        date_l = metrics.get('lgbm', {}).get('last_train', '未訓練')
        c_l = "#00cc96" if wr_l > 0.55 else "#ffc107"
        st.markdown(f"<div style='background:#1e1e1e; padding:12px; border-left:4px solid #00cc96; border-radius:5px; margin-bottom:8px;'><div style='display:flex;justify-content:space-between;align-items:center;'><div><span style='color:#aaa;font-size:11px;'>LightGBM</span><br><span style='font-size:22px; color:{c_l}; font-weight:bold;'>{wr_l*100:.1f}%</span></div><div style='text-align:right;'><span style='color:#666;font-size:10px;'>{date_l}</span></div></div></div>", unsafe_allow_html=True)
        
        wr_ll = metrics.get('lstm', {}).get('blind_win_rate', 0)
        date_ll = metrics.get('lstm', {}).get('last_train', '未訓練')
        c_ll = "#00cc96" if wr_ll > 0.53 else "#ffc107"
        st.markdown(f"<div style='background:#1e1e1e; padding:12px; border-left:4px solid #00cc96; border-radius:5px;'><div style='display:flex;justify-content:space-between;align-items:center;'><div><span style='color:#aaa;font-size:11px;'>LSTM</span><br><span style='font-size:22px; color:{c_ll}; font-weight:bold;'>{wr_ll*100:.1f}%</span></div><div style='text-align:right;'><span style='color:#666;font-size:10px;'>{date_ll}</span></div></div></div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("**🔴 空頭模型**")
        wr_s = metrics.get('short', {}).get('lgbm', {}).get('blind_win_rate', 0)
        date_s = metrics.get('short', {}).get('lgbm', {}).get('last_train', '未訓練')
        c_s = "#ff4b4b" if wr_s > 0.55 else "#ff9966"
        if wr_s > 0:
            st.markdown(f"<div style='background:#1e1e1e; padding:12px; border-left:4px solid #ff4b4b; border-radius:5px; margin-bottom:8px;'><div style='display:flex;justify-content:space-between;align-items:center;'><div><span style='color:#aaa;font-size:11px;'>LightGBM</span><br><span style='font-size:22px; color:{c_s}; font-weight:bold;'>{wr_s*100:.1f}%</span></div><div style='text-align:right;'><span style='color:#666;font-size:10px;'>{date_s}</span></div></div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#1e1e1e; padding:12px; border-left:4px solid #666; border-radius:5px; margin-bottom:8px;'><span style='color:#666;font-size:12px;'>LightGBM</span><br><span style='font-size:16px; color:#666;'>未訓練</span></div>", unsafe_allow_html=True)
        
        wr_ls = metrics.get('short', {}).get('lstm', {}).get('blind_win_rate', 0)
        date_ls = metrics.get('short', {}).get('lstm', {}).get('last_train', '未訓練')
        c_ls = "#ff4b4b" if wr_ls > 0.53 else "#ff9966"
        if wr_ls > 0:
            st.markdown(f"<div style='background:#1e1e1e; padding:12px; border-left:4px solid #ff4b4b; border-radius:5px;'><div style='display:flex;justify-content:space-between;align-items:center;'><div><span style='color:#aaa;font-size:11px;'>LSTM</span><br><span style='font-size:22px; color:{c_ls}; font-weight:bold;'>{wr_ls*100:.1f}%</span></div><div style='text-align:right;'><span style='color:#666;font-size:10px;'>{date_ls}</span></div></div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#1e1e1e; padding:12px; border-left:4px solid #666; border-radius:5px;'><span style='color:#666;font-size:12px;'>LSTM</span><br><span style='font-size:16px; color:#666;'>未訓練</span></div>", unsafe_allow_html=True)