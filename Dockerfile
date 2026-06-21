FROM python:3.10-slim

WORKDIR /app

# 安裝系統依賴項目（TensorFlow 與 LightGBM 運作所需）
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 綁定雲端分配的 PORT 環境變數
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
