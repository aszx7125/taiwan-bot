import os
import re

file_path = "c:/Users/aszx7/Desktop/taiwan-bot/taiwan-bot/app.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

start_marker = "# --- 3. TradingView 原生互動 K線圖 (Lightweight Charts) ---"
end_marker = "# --- 4. 系統戰術分析 (基於自有模型) ---"

idx1 = content.find(start_marker)
idx2 = content.find(end_marker)

if idx1 == -1 or idx2 == -1:
    print("Could not find markers")
    exit(1)

tv_code = """# --- 3. TradingView 原生互動 K線圖 (Lightweight Charts) ---
                st.markdown("### 📈 即時技術分析 (TradingView 原生體驗)")
                
                # UI 控制區
                col_tf, col_ind, col_ma1, col_ma2, col_ma3 = st.columns([2, 3, 1, 1, 1])
                with col_tf: 
                    tf_sel = st.selectbox("時區", ["日線", "小時圖", "週線", "月線"], index=0, label_visibility="collapsed")
                with col_ind:
                    ind_sel = st.multiselect("顯示指標", ["均線", "成交量"], default=["均線", "成交量"], label_visibility="collapsed")
                with col_ma1: 
                    ma1_p = st.number_input("MA 1", min_value=1, max_value=300, value=5)
                with col_ma2: 
                    ma2_p = st.number_input("MA 2", min_value=1, max_value=300, value=20)
                with col_ma3: 
                    ma3_p = st.number_input("MA 3", min_value=1, max_value=300, value=60)
                
                import json
                import numpy as np
                import pandas as pd
                
                # 依據時區選擇或重採樣資料
                if tf_sel == "小時圖" and not df_hourly.empty:
                    df_tv = df_hourly.copy()
                else:
                    df_tv = df_daily.copy()
                    
                df_tv = df_tv[~df_tv.index.duplicated(keep='last')]
                df_tv = df_tv.sort_index()
                
                if tf_sel == "週線":
                    df_tv = df_tv.resample('W-FRI').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})
                elif tf_sel == "月線":
                    df_tv = df_tv.resample('M').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})
                
                # 均線計算
                df_tv[f'SMA_{ma1_p}'] = df_tv['Close'].rolling(window=ma1_p).mean()
                df_tv[f'SMA_{ma2_p}'] = df_tv['Close'].rolling(window=ma2_p).mean()
                df_tv[f'SMA_{ma3_p}'] = df_tv['Close'].rolling(window=ma3_p).mean()
                
                # 清洗與轉換
                df_tv = df_tv.dropna(subset=['Open', 'High', 'Low', 'Close'])
                df_tv = df_tv.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # 處理時間格式 (Lightweight Charts 要求: 日線以上為 'YYYY-MM-DD'，小時線以下為 Unix Timestamp 數字)
                if tf_sel == "小時圖":
                    # Convert DatetimeIndex to Unix timestamp in seconds
                    df_tv['time'] = df_tv.index.astype('int64') // 10**9
                else:
                    df_tv['time'] = df_tv.index.strftime('%Y-%m-%d')
                
                candle_data = [{"time": r['time'], "open": float(r['Open']), "high": float(r['High']), "low": float(r['Low']), "close": float(r['Close'])} for _, r in df_tv.iterrows()]
                volume_data = [{"time": r['time'], "value": float(r.get('Volume', 0)), "color": "rgba(8,153,129,0.5)" if r['Close'] >= r['Open'] else "rgba(242,54,69,0.5)"} for _, r in df_tv.iterrows()]
                
                ma1_data = [{"time": r['time'], "value": float(r[f'SMA_{ma1_p}'])} for _, r in df_tv.iterrows() if r[f'SMA_{ma1_p}'] > 0]
                ma2_data = [{"time": r['time'], "value": float(r[f'SMA_{ma2_p}'])} for _, r in df_tv.iterrows() if r[f'SMA_{ma2_p}'] > 0]
                ma3_data = [{"time": r['time'], "value": float(r[f'SMA_{ma3_p}'])} for _, r in df_tv.iterrows() if r[f'SMA_{ma3_p}'] > 0]
                
                show_ma = "均線" in ind_sel
                show_vol = "成交量" in ind_sel
                
                html_code = f'''
                <div id="tvchart" style="width: 100%; height: 500px; background-color: #131722; border-radius: 8px;"></div>
                <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
                <script>
                    try {{
                        const domElement = document.getElementById('tvchart');
                        const chartProperties = {{
                            layout: {{ background: {{ type: 'solid', color: '#131722' }}, textColor: '#d1d4dc' }},
                            grid: {{ vertLines: {{ color: '#2b2b43' }}, horzLines: {{ color: '#2b2b43' }} }},
                            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                            rightPriceScale: {{ borderColor: '#2b2b43' }},
                            timeScale: {{ 
                                borderColor: '#2b2b43', 
                                timeVisible: {"true" if tf_sel == "小時圖" else "false"} 
                            }}
                        }};
                        
                        const chart = LightweightCharts.createChart(domElement, chartProperties);
                        
                        const candleSeries = chart.addCandlestickSeries({{
                            upColor: '#089981', downColor: '#F23645', borderDownColor: '#F23645', borderUpColor: '#089981', wickDownColor: '#F23645', wickUpColor: '#089981'
                        }});
                        candleSeries.setData({json.dumps(candle_data)});
                        
                        if ({str(show_ma).lower()}) {{
                            const ma1Series = chart.addLineSeries({{ color: '#f5c542', lineWidth: 2, title: 'SMA {ma1_p}' }});
                            ma1Series.setData({json.dumps(ma1_data)});
                            const ma2Series = chart.addLineSeries({{ color: '#2962FF', lineWidth: 2, title: 'SMA {ma2_p}' }});
                            ma2Series.setData({json.dumps(ma2_data)});
                            const ma3Series = chart.addLineSeries({{ color: '#e841f4', lineWidth: 2, title: 'SMA {ma3_p}' }});
                            ma3Series.setData({json.dumps(ma3_data)});
                        }}
                        
                        if ({str(show_vol).lower()}) {{
                            const volumeSeries = chart.addHistogramSeries({{
                                priceFormat: {{ type: 'volume' }}, priceScaleId: '', scaleMargins: {{ top: 0.8, bottom: 0 }}
                            }});
                            volumeSeries.setData({json.dumps(volume_data)});
                        }}
                        
                        new ResizeObserver(entries => {{
                            if (entries.length === 0 || entries[0].target !== domElement) return;
                            chart.applyOptions({{ width: entries[0].contentRect.width, height: entries[0].contentRect.height }});
                        }}).observe(domElement);
                    }} catch (e) {{
                        document.getElementById('tvchart').innerHTML = "<div style='color:red; padding: 20px;'>Chart Error: " + e.toString() + "</div>";
                    }}
                </script>
                '''
                import streamlit.components.v1 as components
                components.html(html_code, height=520)
                
                """

content = content[:idx1] + tv_code + content[idx2:]

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("TV chart features updated successfully.")
