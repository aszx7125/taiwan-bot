import re
import os

file_path = "c:/Users/aszx7/Desktop/taiwan-bot/taiwan-bot/app.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Sidebar to add navigation
sidebar_marker = """with st.sidebar:\n    st.header("📂 我的自選清單")"""
new_sidebar = """with st.sidebar:
    st.title("🧭 導覽列")
    if "current_page" not in st.session_state:
        st.session_state.current_page = "📊 台股大盤掃描"
    
    def nav_callback():
        st.session_state.analyze_trigger = None
        st.session_state.current_page = st.session_state.nav_radio
        
    st.radio("功能切換", ["📊 台股大盤掃描", "👨‍⚕️ AI 自選健檢", "💎 價值深度報告", "🎯 狙擊標的診斷"], 
                            index=["📊 台股大盤掃描", "👨‍⚕️ AI 自選健檢", "💎 價值深度報告", "🎯 狙擊標的診斷"].index(st.session_state.current_page),
                            key="nav_radio", on_change=nav_callback)
    st.markdown("---")
    
    st.header("📂 我的自選清單")"""

content = content.replace(sidebar_marker, new_sidebar)

# 2. Update routing
# Find where the main content routing starts:
# st.title("⚡ 台股戰情分析終端 v4.1")
# ...
# if target_ticker:
# ...
# else:
# ... (ends at the end of the file)

start_marker = 'st.title("⚡ 台股戰情分析終端 v4.1")'

start_idx = content.find(start_marker)
if start_idx == -1:
    print("Could not find start marker")
    exit(1)

# Extract everything from start_marker to the end
tail = content[start_idx:]

# Let's break the tail into the "if target_ticker:" part and the "else:" part
target_ticker_start = tail.find('if target_ticker:')
else_start = tail.find('\nelse:\n')

if target_ticker_start == -1 or else_start == -1:
    print("Could not find routing blocks")
    exit(1)

sniper_block = tail[target_ticker_start + len('if target_ticker:'):else_start]
market_block = tail[else_start + len('\nelse:\n'):]

# Fix indentations
def unindent(text):
    return '\n'.join(line[4:] if line.startswith('    ') else line for line in text.split('\n'))
def indent(text):
    return '\n'.join('    ' + line if line else line for line in text.split('\n'))

sniper_block_unindented = unindent(sniper_block).strip()
market_block_unindented = unindent(market_block).strip()

# Replace the "⬅️ 返回戰情室主頁" logic in sniper block
old_return = """if st.button("⬅️ 返回戰情室主頁", use_container_width=True):
    st.session_state.analyze_trigger = None
    st.rerun()"""
new_return = """if st.button("⬅️ 返回戰情室主頁", use_container_width=True):
    st.session_state.current_page = "📊 台股大盤掃描"
    st.rerun()"""
sniper_block_unindented = sniper_block_unindented.replace(
    'if st.button("⬅️ 返回戰情室主頁", use_container_width=True):\n    st.session_state.analyze_trigger = None\n    st.rerun()', 
    new_return
)

# Replace target_ticker usage inside sniper_block
sniper_block_unindented = sniper_block_unindented.replace(
    "base_ticker = target_ticker.split('.')[0]",
    "target_ticker = st.session_state.get('target_ticker_cache', None)\nif not target_ticker:\n    st.warning('請先從左側自選清單或大盤掃描中選擇一檔股票進行診斷！')\nelse:\n    base_ticker = target_ticker.split('.')[0]"
)
# We need to indent everything after the else: in sniper_block
lines = sniper_block_unindented.split('\n')
new_lines = []
indenting = False
for line in lines:
    if line.startswith("base_ticker = target_ticker.split('.')[0]"):
        new_lines.append("target_ticker = st.session_state.get('target_ticker_cache', None)")
        new_lines.append("if not target_ticker:")
        new_lines.append("    st.warning('請先從左側自選清單或大盤掃描中選擇一檔股票進行診斷！')")
        new_lines.append("else:")
        new_lines.append("    base_ticker = target_ticker.split('.')[0]")
        indenting = True
    elif indenting:
        new_lines.append("    " + line)
    else:
        new_lines.append(line)

sniper_block_indented = indent('\n'.join(new_lines))


# Market Overview Additions
# Top gainers/losers
market_additions = """
    # --- 新增：大盤掃描排行 (致敬 bubuaplus) ---
    st.markdown("### 📊 大盤掃描排行")
    if snapshot_data:
        df_snap = pd.DataFrame(snapshot_data)
        # Handle columns
        t_col = '代號' if '代號' in df_snap.columns else 'N'
        n_col = '名稱' if '名稱' in df_snap.columns else 'W'
        p_col = '現價' if '現價' in df_snap.columns else '{'
        v_col = '成交量' if '成交量' in df_snap.columns else 'q'
        
        if 'recent_returns' in df_snap.columns and t_col in df_snap.columns:
            df_snap['漲跌幅'] = df_snap['recent_returns'] * 100
            df_snap['現價'] = df_snap[p_col]
            df_snap['名稱'] = df_snap[n_col]
            df_snap['成交量'] = df_snap[v_col]
            
            top_gainers = df_snap.sort_values(by='漲跌幅', ascending=False).head(5)
            top_losers = df_snap.sort_values(by='漲跌幅', ascending=True).head(5)
            top_vol = df_snap.sort_values(by='成交量', ascending=False).head(5)
            
            col_g, col_l = st.columns(2)
            with col_g:
                st.markdown("#### 🚀 漲幅排行 (Top 5)")
                _render_clean_html("<div style='background:#141823;border-radius:10px;padding:10px;'>")
                st.dataframe(top_gainers[[t_col, '名稱', '現價', '漲跌幅']].style.format({'現價': '{:.2f}', '漲跌幅': '{:.2f}%'}), use_container_width=True, hide_index=True)
                _render_clean_html("</div>")
            with col_l:
                st.markdown("#### 📉 跌幅排行 (Top 5)")
                _render_clean_html("<div style='background:#141823;border-radius:10px;padding:10px;'>")
                st.dataframe(top_losers[[t_col, '名稱', '現價', '漲跌幅']].style.format({'現價': '{:.2f}', '漲跌幅': '{:.2f}%'}), use_container_width=True, hide_index=True)
                _render_clean_html("</div>")
                
            st.markdown("#### 💰 成交量排行 (人氣股)")
            _render_clean_html("<div style='background:#141823;border-radius:10px;padding:10px;'>")
            st.dataframe(top_vol[[t_col, '名稱', '現價', '漲跌幅', '成交量']].style.format({'現價': '{:.2f}', '漲跌幅': '{:.2f}%', '成交量': '{:.0f}'}), use_container_width=True, hide_index=True)
            _render_clean_html("</div>")
"""

market_block_indented = indent(market_block_unindented + market_additions)


# AI Health Check Block
health_check_block = indent("""
st.markdown("### 👨‍⚕️ AI 自選健檢")
st.caption("輸入股票代號，Nemotron-120B 即時評估該標的之形態健康度。")
hc_ticker = st.text_input("輸入股票代號", placeholder="例如: 2330", key="hc_input")
if st.button("🔄 開始健檢", type="primary"):
    if hc_ticker:
        base_ticker = hc_ticker.split('.')[0]
        c_name = st.session_state.stock_names.get(base_ticker) or get_stock_name_from_csv(base_ticker)
        with st.spinner(f"正在對 {base_ticker} {c_name} 進行 AI 健檢..."):
            df_daily, _, _ = get_kline_with_fugle(hc_ticker, FUGLE_API_KEY)
            if df_daily.empty:
                st.error("❌ 查無該標的歷史數據")
            else:
                today = df_daily.iloc[-1]
                ep = float(today.get('Close', 0.0))
                sma = float(today.get('SMA_20', 0.0))
                rsi = float(today.get('RSI', 50.0))
                
                payload = {
                    "model": "nemotron-3-super-120b",
                    "messages": [
                        {"role": "system", "content": "你是一位量化分析師。請根據現價、月線與RSI，簡潔給出『趨勢狀態』(如多頭排列/跌破月線)與『強勢度評估』(1-10分)，並給出一句總結，總字數50字內。"},
                        {"role": "user", "content": f"標的:{c_name}，現價:{ep}，月線SMA20:{sma:.2f}，RSI:{rsi:.2f}"}
                    ],
                    "temperature": 0.3
                }
                res = requests.post("https://www.iai.nkust.edu.tw/aihub/v1/chat/completions", headers={"Authorization": "Bearer sk-DPOkK719wRKLm7VIzNxFjw"}, json=payload, timeout=30)
                if res.status_code == 200:
                    ai_health = res.json()["choices"][0]["message"]["content"]
                    html = f'''
                    <div style="background-color: #141823; border: 1px solid #252b3b; border-radius: 12px; padding: 16px; margin-top:14px;">
                        <h3 style="margin-top:0;">{base_ticker} {c_name} <span style="font-size:14px; color:#8a93a6; font-weight:normal;">現價 {ep}</span></h3>
                        <div style="font-size:15px; line-height:1.6; color:#e6e9f0;">{ai_health}</div>
                    </div>
                    '''
                    _render_clean_html(html)
                else:
                    st.warning("AI 回應失敗")
""")

# Value Deep Dive Block
value_report_block = indent("""
st.markdown("### 💎 價值投資深度報告")
st.caption("透過 120B 大腦，一鍵生成媲美機構水準的深度研究報告。")
vr_ticker = st.text_input("輸入欲產出報告之代號", placeholder="例如: 2330", key="vr_input")
if st.button("✨ 生成深度報告", type="primary"):
    if vr_ticker:
        base_ticker = vr_ticker.split('.')[0]
        c_name = st.session_state.stock_names.get(base_ticker) or get_stock_name_from_csv(base_ticker)
        with st.spinner(f"Nemotron-120B 正在撰寫 {base_ticker} {c_name} 的深度報告..."):
            df_daily, _, _ = get_kline_with_fugle(vr_ticker, FUGLE_API_KEY)
            if df_daily.empty:
                st.error("❌ 查無該標的歷史數據")
            else:
                today = df_daily.iloc[-1]
                ep = float(today.get('Close', 0.0))
                sma = float(today.get('SMA_20', 0.0))
                vol = float(today.get('Volume', 0.0))
                
                prompt = f"請針對 {base_ticker} {c_name} (現價:{ep}, 月線:{sma:.2f}, 成交量:{vol}) 撰寫價值投資掃描報告。請包含以下三大結構，不要使用Markdown粗體，直接輸出文字：1.催化因素 (Catalyst Breakdown) 2.財務健康與估值 (Fundamentals) 3.風險報酬對稱性 (Asymmetry, 列出Base/Bull/Bear情境)。總字數300字。"
                
                payload = {
                    "model": "nemotron-3-super-120b",
                    "messages": [
                        {"role": "system", "content": "你是一位華爾街頂尖價值型基金經理人。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.5
                }
                res = requests.post("https://www.iai.nkust.edu.tw/aihub/v1/chat/completions", headers={"Authorization": "Bearer sk-DPOkK719wRKLm7VIzNxFjw"}, json=payload, timeout=60)
                if res.status_code == 200:
                    ai_report = res.json()["choices"][0]["message"]["content"]
                    report_html = f'''
                    <div style="background-color: #141823; border: 1px solid #252b3b; border-radius: 12px; margin-top:14px; overflow:hidden;">
                        <div style="padding: 16px; border-bottom: 1px dashed #252b3b; cursor: pointer;">
                            <div style="display:flex; justify-content:space-between;">
                                <div style="font-size:18px; font-weight:800; color:#fff;">{base_ticker} {c_name}</div>
                                <span style="background:rgba(88,101,242,.18); color:#8ab4ff; padding: 2px 8px; border-radius:4px; font-size:12px;">深度掃描完成</span>
                            </div>
                            <div style="color:#8a93a6; font-size:13px; margin-top:8px;">即時價: {ep:.2f}</div>
                        </div>
                        <div style="padding: 16px; color:#e6e9f0; line-height: 1.8; font-size:14px; white-space:pre-wrap;">{ai_report}</div>
                    </div>
                    '''
                    _render_clean_html(report_html)
                else:
                    st.warning("AI 回應失敗")
""")

new_routing = f"""st.title("⚡ 台股戰情分析終端 v4.1")
st.caption("🟢 多頭 | 🔴 空頭 | ⚪ 盤整 | 四核心極速快取版")
col1, col2 = st.columns([3, 1])
with col1: manual_ticker = st.text_input("輸入股票代號", "", label_visibility="collapsed", placeholder="例如: 2330", key="manual_search")
with col2: analyze_manual_btn = st.button("單股掃描", use_container_width=True)
st.markdown("---")

query_ticker = st.query_params.get("analyze")
if query_ticker:
    st.query_params.clear()

target_ticker = query_ticker or st.session_state.pop('analyze_trigger', None) or (manual_ticker.strip().upper() if analyze_manual_btn else None)

if target_ticker:
    st.session_state.current_page = "🎯 狙擊標的診斷"
    st.session_state.target_ticker_cache = target_ticker
    st.rerun()

if st.session_state.current_page == "🎯 狙擊標的診斷":
{sniper_block_indented}
elif st.session_state.current_page == "📊 台股大盤掃描":
{market_block_indented}
elif st.session_state.current_page == "👨‍⚕️ AI 自選健檢":
{health_check_block}
elif st.session_state.current_page == "💎 價值深度報告":
{value_report_block}
"""

content = content[:start_idx] + new_routing

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Routing update completed successfully.")
