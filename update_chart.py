import os
import re

file_path = "c:/Users/aszx7/Desktop/taiwan-bot/taiwan-bot/app.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# We need to find the block starting from `# --- 3. 原生 Plotly K線圖 (致敬 TradingView) ---`
# down to the end of the AI section (before `st.markdown("---")` and `⬅️ 返回戰情室主頁`)

start_marker = "# --- 3. 原生 Plotly K線圖 (致敬 TradingView) ---"
end_marker = 'if st.button("⬅️ 返回戰情室主頁", use_container_width=True):'

idx1 = content.find(start_marker)
idx2 = content.find(end_marker)

if idx1 == -1 or idx2 == -1:
    print("Could not find markers.")
    exit(1)

# We will replace this block with the Lightweight Charts implementation
tv_code = """
            # --- 3. TradingView 原生互動 K線圖 (Lightweight Charts) ---
            st.markdown("### 📈 即時技術分析 (TradingView 原生體驗)")
            
            # Prepare data for Lightweight Charts
            # df_daily needs: time, open, high, low, close, value (volume)
            # MA needs: time, value
            import json
            
            df_tv = df_daily.copy()
            df_tv['time'] = df_tv.index.strftime('%Y-%m-%d')
            
            # Calculate MAs if not exist
            if 'SMA_5' not in df_tv.columns:
                df_tv['SMA_5'] = df_tv['Close'].rolling(window=5).mean()
            if 'SMA_20' not in df_tv.columns:
                df_tv['SMA_20'] = df_tv['Close'].rolling(window=20).mean()
            if 'SMA_60' not in df_tv.columns:
                df_tv['SMA_60'] = df_tv['Close'].rolling(window=60).mean()
                
            df_tv = df_tv.dropna(subset=['Close']) # clean up
            
            # Build JSON structures
            candle_data = [{"time": r['time'], "open": r['Open'], "high": r['High'], "low": r['Low'], "close": r['Close']} for _, r in df_tv.iterrows()]
            volume_data = [{"time": r['time'], "value": r.get('Volume', 0), "color": "rgba(8,153,129,0.5)" if r['Close'] >= r['Open'] else "rgba(242,54,69,0.5)"} for _, r in df_tv.iterrows()]
            
            ma5_data = [{"time": r['time'], "value": r['SMA_5']} for _, r in df_tv.iterrows() if not pd.isna(r['SMA_5'])]
            ma20_data = [{"time": r['time'], "value": r['SMA_20']} for _, r in df_tv.iterrows() if not pd.isna(r['SMA_20'])]
            ma60_data = [{"time": r['time'], "value": r['SMA_60']} for _, r in df_tv.iterrows() if not pd.isna(r['SMA_60'])]
            
            html_code = f'''
            <div id="tvchart" style="width: 100%; height: 500px; background-color: #131722; border-radius: 8px;"></div>
            <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
            <script>
                const chartProperties = {{
                    layout: {{ background: {{ type: 'solid', color: '#131722' }}, textColor: '#d1d4dc' }},
                    grid: {{ vertLines: {{ color: '#2b2b43' }}, horzLines: {{ color: '#2b2b43' }} }},
                    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                    rightPriceScale: {{ borderColor: '#2b2b43' }},
                    timeScale: {{ borderColor: '#2b2b43', timeVisible: true }}
                }};
                
                const domElement = document.getElementById('tvchart');
                const chart = LightweightCharts.createChart(domElement, chartProperties);
                
                // --- Candlestick Series ---
                const candleSeries = chart.addCandlestickSeries({{
                    upColor: '#089981', downColor: '#F23645',
                    borderDownColor: '#F23645', borderUpColor: '#089981',
                    wickDownColor: '#F23645', wickUpColor: '#089981'
                }});
                candleSeries.setData({json.dumps(candle_data)});
                
                // --- Moving Averages ---
                const ma5Series = chart.addLineSeries({{ color: '#f5c542', lineWidth: 2, title: 'SMA 5' }});
                ma5Series.setData({json.dumps(ma5_data)});
                const ma20Series = chart.addLineSeries({{ color: '#2962FF', lineWidth: 2, title: 'SMA 20' }});
                ma20Series.setData({json.dumps(ma20_data)});
                const ma60Series = chart.addLineSeries({{ color: '#e841f4', lineWidth: 2, title: 'SMA 60' }});
                ma60Series.setData({json.dumps(ma60_data)});
                
                // --- Volume Series ---
                const volumeSeries = chart.addHistogramSeries({{
                    priceFormat: {{ type: 'volume' }},
                    priceScaleId: '', // Overlay on chart
                    scaleMargins: {{ top: 0.8, bottom: 0 }}
                }});
                volumeSeries.setData({json.dumps(volume_data)});
                
                // Auto adjust size
                new ResizeObserver(entries => {{
                    if (entries.length === 0 || entries[0].target !== domElement) return;
                    const newRect = entries[0].contentRect;
                    chart.applyOptions({{ width: newRect.width, height: newRect.height }});
                }}).observe(domElement);
            </script>
            '''
            
            import streamlit.components.v1 as components
            components.html(html_code, height=520)
            
            st.markdown("---")
            """

content = content[:idx1] + tv_code + "            " + content[idx2:]

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Plotly replaced with TradingView Lightweight Charts successfully.")
