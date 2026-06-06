import streamlit as st

def render_top20_card(s):
    """渲染全市場 Top 20 戰術卡片"""
    st.markdown(f"""
    <div style="border: 2px solid {s['box_color']}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 15px;">
        <h4 style="color: {s['box_color']}; margin-top: 0;">🎯 {s['ticker']} {s['name']}</h4>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 150px;">
                <span style="color: gray; font-size: 14px;">雙核勝率</span><br>
                <b style="font-size: 22px; color: {s['box_color']};">{s['win_prob']*100:.1f}%</b><br>
                <span style="font-size: 12px; font-weight: bold; color: {s['box_color']};">{s['ai_rec']}</span>
            </div>
            <div style="flex: 1; min-width: 120px;">
                <span style="color: gray; font-size: 14px;">進場價</span><br>
                <b style="font-size: 20px;">{s['entry_price']:.2f}</b>
            </div>
            <div style="flex: 1; min-width: 150px;">
                <span style="color: gray; font-size: 14px;">停利點</span><br>
                <b style="font-size: 20px; color: #00cc96;">{s['take_profit']:.2f}</b><br>
                <span style="font-size: 12px; color: #00cc96;">{s['profit_reason']}</span>
            </div>
            <div style="flex: 1; min-width: 120px;">
                <span style="color: gray; font-size: 14px;">防守價</span><br>
                <b style="font-size: 20px; color: #ff4b4b;">{s['stop_loss']:.2f}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_single_diagnostic_card(win_rate_str, recommendation, entry_price, take_profit, stop_loss, box_color, text_color):
    """渲染單股專注模式的雙核卡片"""
    st.markdown(f"""
    <div style="border: 2px solid {box_color}; border-radius: 10px; padding: 20px; background-color: #1e1e1e; margin-bottom: 20px;">
        <h4 style="color: {box_color}; margin-top: 0;">🎯 AI 雙核戰術計畫</h4>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 180px; margin-bottom: 10px;">
                <span style="color: gray; font-size: 14px;">1. 雙核加權真實勝率</span><br>
                <b style="font-size: 24px; color: {box_color};">{win_rate_str}</b><br>
                <span style="font-size: 14px; font-weight: bold; color: {text_color};">{recommendation}</span>
            </div>
            <div style="flex: 1; min-width: 130px; margin-bottom: 10px;">
                <span style="color: gray; font-size: 14px;">2. 建議進場價</span><br>
                <b style="font-size: 22px;">{entry_price:.2f}</b>
            </div>
            <div style="flex: 1; min-width: 200px; margin-bottom: 10px;">
                <span style="color: gray; font-size: 14px;">3. 結構停利點</span><br>
                <b style="font-size: 22px; color: #00cc96;">{take_profit:.2f}</b>
            </div>
            <div style="flex: 1; min-width: 130px; margin-bottom: 10px;">
                <span style="color: gray; font-size: 14px;">4. 嚴格防守價</span><br>
                <b style="font-size: 22px; color: #ff4b4b;">{stop_loss:.2f}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_backtest_metric_card(title, value, subtext, color):
    """渲染回測專用的績效方塊"""
    st.markdown(f"""
    <div style="background-color: #121218; padding: 22px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #2a2a35;">
        <div style="color: #8b8b99; font-size: 14px; margin-bottom: 8px;">{title}</div>
        <div style="color: {color}; font-size: 32px; font-weight: 700; margin-bottom: 5px;">{value}</div>
        <div style="color: #6b6b79; font-size: 13px;">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)