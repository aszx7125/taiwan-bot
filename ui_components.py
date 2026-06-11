"""
UI組件 - 完整版 v2.0
支援四核心顯示與訓練日期
"""
import streamlit as st


def render_top20_card(s):
    """渲染 TOP20 卡片"""
    color = s.get('box_color', '#00cc96')
    prob = s.get('win_prob', 0) * 100
    
    st.markdown(f"""
    <div style="border: 2px solid {color}; border-radius: 10px; padding: 18px; 
                background-color: #1e1e1e; margin-bottom: 12px;">
        <h4 style="color: {color}; margin-top: 0;">🎯 {s['ticker']} {s['name']}</h4>
        <div style="display: flex; justify-content: space-between;">
            <div>
                <span style="color: gray; font-size: 13px;">勝率</span><br>
                <b style="font-size: 22px; color: {color};">{prob:.1f}%</b>
            </div>
            <div style="text-align: right;">
                <span style="color: gray; font-size: 13px;">進場</span><br>
                <b style="font-size: 20px;">{s['entry_price']:.2f}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_single_diagnostic_card(long_prob, short_prob, signal, entry_price, res_level, sup_level):
    """四核心單股診斷卡 (多空雙向綜合評估)"""
    
    # 根據 AI 訊號決定顏色與建議文字
    if signal in ["STRONG_LONG", "LONG"]:
        box_color = "#00cc96"
        rec = "⭐⭐⭐ 強勢做多" if signal == "STRONG_LONG" else "⭐⭐ 偏多佈局"
    elif signal in ["STRONG_SHORT", "SHORT"]:
        box_color = "#ff4b4b"
        rec = "⚠️⚠️ 強勢放空" if signal == "STRONG_SHORT" else "⚠️ 偏空操作"
    elif signal == "HIGH_VOLATILITY":
        box_color = "#ffa500"
        rec = "⚡ 多空雙巴 (建議空手)"
    else:
        box_color = "#a8a8a8"
        rec = "⚪ 動能不足 (建議觀望)"

    st.markdown(f"""
    <div style="border: 2px solid {box_color}; border-radius: 10px; padding: 20px; 
                background-color: #1e1e1e; margin-bottom: 20px;">
        <h4 style="color: {box_color}; margin-top: 0;">🎯 四核心戰術計畫：{rec}</h4>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 150px;">
                <span style="color: gray; font-size: 14px;">多頭勝率 (做多)</span><br>
                <b style="font-size: 24px; color: #00cc96;">{long_prob*100:.1f}%</b>
            </div>
            <div style="flex: 1; min-width: 150px;">
                <span style="color: gray; font-size: 14px;">空頭勝率 (做空)</span><br>
                <b style="font-size: 24px; color: #ff4b4b;">{short_prob*100:.1f}%</b>
            </div>
            <div style="flex: 1; min-width: 120px;">
                <span style="color: gray; font-size: 14px;">當前現價</span><br>
                <b style="font-size: 20px;">{entry_price:.2f}</b>
            </div>
            <div style="flex: 1; min-width: 140px;">
                <span style="color: gray; font-size: 14px;">上檔壓力(停利/空單停損)</span><br>
                <b style="font-size: 20px; color: #ffc107;">{res_level:.2f}</b>
            </div>
            <div style="flex: 1; min-width: 140px;">
                <span style="color: gray; font-size: 14px;">下檔支撐(停損/空單停利)</span><br>
                <b style="font-size: 20px; color: #00ccff;">{sup_level:.2f}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_backtest_metric_card(title, value, subtext, color):
    """回測指標卡"""
    st.markdown(f"""
    <div style="background-color: #121218; padding: 20px; border-radius: 10px; 
                margin-bottom: 15px; border: 1px solid #2a2a35;">
        <div style="color: #8b8b99; font-size: 13px; margin-bottom: 6px;">{title}</div>
        <div style="color: {color}; font-size: 28px; font-weight: 700;">{value}</div>
        <div style="color: #6b6b79; font-size: 12px;">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)


def render_model_health_board(metrics):
    """四核心健康度 - 含訓練日期"""
    st.markdown("### 🧪 四核心AI大腦：盲測勝率")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 多頭模型**")
        # LightGBM Long
        wr_l = metrics.get('lgbm', {}).get('blind_win_rate', 0)
        date_l = metrics.get('lgbm', {}).get('last_train', '未訓練')
        c_l = "#00cc96" if wr_l > 0.55 else "#ffc107"
        st.markdown(f"""
        <div style='background:#1e1e1e; padding:12px; border-left:4px solid #00cc96; border-radius:5px; margin-bottom:8px;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <span style='color:#aaa;font-size:11px;'>LightGBM</span><br>
                    <span style='font-size:22px; color:{c_l}; font-weight:bold;'>{wr_l*100:.1f}%</span>
                </div>
                <div style='text-align:right;'>
                    <span style='color:#666;font-size:10px;'>{date_l}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # LSTM Long
        wr_ll = metrics.get('lstm', {}).get('blind_win_rate', 0)
        date_ll = metrics.get('lstm', {}).get('last_train', '未訓練')
        c_ll = "#00cc96" if wr_ll > 0.53 else "#ffc107"
        st.markdown(f"""
        <div style='background:#1e1e1e; padding:12px; border-left:4px solid #00cc96; border-radius:5px;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <span style='color:#aaa;font-size:11px;'>LSTM</span><br>
                    <span style='font-size:22px; color:{c_ll}; font-weight:bold;'>{wr_ll*100:.1f}%</span>
                </div>
                <div style='text-align:right;'>
                    <span style='color:#666;font-size:10px;'>{date_ll}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🔴 空頭模型**")
        # LightGBM Short
        wr_s = metrics.get('short', {}).get('lgbm', {}).get('blind_win_rate', 0)
        date_s = metrics.get('short', {}).get('lgbm', {}).get('last_train', '未訓練')
        if wr_s > 0:
            c_s = "#ff4b4b" if wr_s > 0.55 else "#ff9966"
            st.markdown(f"""
            <div style='background:#1e1e1e; padding:12px; border-left:4px solid #ff4b4b; border-radius:5px; margin-bottom:8px;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <span style='color:#aaa;font-size:11px;'>LightGBM</span><br>
                        <span style='font-size:22px; color:{c_s}; font-weight:bold;'>{wr_s*100:.1f}%</span>
                    </div>
                    <div style='text-align:right;'>
                        <span style='color:#666;font-size:10px;'>{date_s}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#1e1e1e; padding:12px; border-left:4px solid #666; border-radius:5px; margin-bottom:8px;'>
                <span style='color:#666;font-size:12px;'>LightGBM</span><br>
                <span style='font-size:16px; color:#666;'>未訓練</span>
            </div>
            """, unsafe_allow_html=True)
        
        # LSTM Short
        wr_ls = metrics.get('short', {}).get('lstm', {}).get('blind_win_rate', 0)
        date_ls = metrics.get('short', {}).get('lstm', {}).get('last_train', '未訓練')
        if wr_ls > 0:
            c_ls = "#ff4b4b" if wr_ls > 0.53 else "#ff9966"
            st.markdown(f"""
            <div style='background:#1e1e1e; padding:12px; border-left:4px solid #ff4b4b; border-radius:5px;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <span style='color:#aaa;font-size:11px;'>LSTM</span><br>
                        <span style='font-size:22px; color:{c_ls}; font-weight:bold;'>{wr_ls*100:.1f}%</span>
                    </div>
                    <div style='text-align:right;'>
                        <span style='color:#666;font-size:10px;'>{date_ls}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#1e1e1e; padding:12px; border-left:4px solid #666; border-radius:5px;'>
                <span style='color:#666;font-size:12px;'>LSTM</span><br>
                <span style='font-size:16px; color:#666;'>未訓練</span>
            </div>
            """, unsafe_allow_html=True)