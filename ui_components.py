"""
UI組件 - 完整版
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


def render_single_diagnostic_card(win_rate_str, recommendation, entry_price, take_profit, stop_loss, box_color, text_color):
    """單股診斷卡"""
    st.markdown(f"""
    <div style="border: 2px solid {box_color}; border-radius: 10px; padding: 20px; 
                background-color: #1e1e1e; margin-bottom: 20px;">
        <h4 style="color: {box_color}; margin-top: 0;">🎯 AI 戰術計畫</h4>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 150px;">
                <span style="color: gray; font-size: 14px;">勝率</span><br>
                <b style="font-size: 24px; color: {box_color};">{win_rate_str}</b><br>
                <span style="font-size: 13px; color: {text_color};">{recommendation}</span>
            </div>
            <div style="flex: 1; min-width: 120px;">
                <span style="color: gray; font-size: 14px;">進場</span><br>
                <b style="font-size: 20px;">{entry_price:.2f}</b>
            </div>
            <div style="flex: 1; min-width: 120px;">
                <span style="color: gray; font-size: 14px;">停利</span><br>
                <b style="font-size: 20px; color: #00cc96;">{take_profit:.2f}</b>
            </div>
            <div style="flex: 1; min-width: 120px;">
                <span style="color: gray; font-size: 14px;">停損</span><br>
                <b style="font-size: 20px; color: #ff4b4b;">{stop_loss:.2f}</b>
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
    """四核心健康度"""
    st.markdown("### 🧪 四核心AI大腦")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 多頭**")
        wr_l = metrics.get('lgbm', {}).get('blind_win_rate', 0)
        c_l = "#00cc96" if wr_l > 0.55 else "#ffc107"
        st.markdown(f"<div style='background:#1e1e1e;padding:10px;border-left:3px solid #00cc96;'>
                     LGBM: <b style='color:{c_l}'>{wr_l*100:.1f}%</b></div>", unsafe_allow_html=True)
        
        wr_ll = metrics.get('lstm', {}).get('blind_win_rate', 0)
        c_ll = "#00cc96" if wr_ll > 0.53 else "#ffc107"
        st.markdown(f"<div style='background:#1e1e1e;padding:10px;border-left:3px solid #00cc96;margin-top:5px;'>
                     LSTM: <b style='color:{c_ll}'>{wr_ll*100:.1f}%</b></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🔴 空頭**")
        wr_s = metrics.get('short', {}).get('lgbm', {}).get('blind_win_rate', 0)
        if wr_s > 0:
            c_s = "#ff4b4b" if wr_s > 0.55 else "#ff9966"
            st.markdown(f"<div style='background:#1e1e1e;padding:10px;border-left:3px solid #ff4b4b;'>
                         LGBM: <b style='color:{c_s}'>{wr_s*100:.1f}%</b></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#1e1e1e;padding:10px;border-left:3px solid #666; color:#666;'>
                         LGBM: 未訓練</div>", unsafe_allow_html=True)
        
        wr_ls = metrics.get('short', {}).get('lstm', {}).get('blind_win_rate', 0)
        if wr_ls > 0:
            c_ls = "#ff4b4b" if wr_ls > 0.53 else "#ff9966"
            st.markdown(f"<div style='background:#1e1e1e;padding:10px;border-left:3px solid #ff4b4b;margin-top:5px;'>
                         LSTM: <b style='color:{c_ls}'>{wr_ls*100:.1f}%</b></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#1e1e1e;padding:10px;border-left:3px solid #666;margin-top:5px;color:#666;'>
                         LSTM: 未訓練</div>", unsafe_allow_html=True)