import os
import re

file_path = "c:/Users/aszx7/Desktop/taiwan-bot/taiwan-bot/app.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the requests.post with a dummy mock class for the LLM
# We will just redefine requests.post to a local function at the top of the file.
# But wait, it's better to just inject a mock function right after `import requests`

mock_code = """
# ==========================================
# 🛑 AI 額度限制：本地模擬攔截器
# ==========================================
class MockResponse:
    def __init__(self, content):
        self.status_code = 200
        self._content = content
    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}

def mock_post(url, *args, **kwargs):
    if "chat/completions" in url:
        payload = kwargs.get('json', {})
        messages = payload.get('messages', [])
        sys_msg = next((m['content'] for m in messages if m['role'] == 'system'), "")
        if "華爾街頂尖價值型基金經理人" in sys_msg:
            msg = "【系統提示：AI 模型已停用】\\n\\n此標的目前缺乏 AI 分析資料。請依據上方數據卡片與K線圖進行判斷。"
        elif "量化分析師" in sys_msg:
            msg = "【AI 已停用】型態健康度評估暫無法使用，請參考月線與現價關係。"
        elif "明日操盤晨會報告" in sys_msg or "明日操盤晨報" in sys_msg:
            msg = "【系統提示：AI 模型已停用】\\n\\n無法生成晨會報告。請參考下方排行面板。"
        else:
            msg = "【AI 已停用】此為系統模擬回覆，因額度限制已暫停呼叫外部模型。"
        return MockResponse(msg)
    import requests as orig_requests
    return orig_requests.post(url, *args, **kwargs)

import requests
requests.post = mock_post
"""

if "# 🛑 AI 額度限制：本地模擬攔截器" not in content:
    content = content.replace("import requests  # 🔥 呼叫高科 API 必備套件", "import requests\n" + mock_code)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Mock applied successfully.")
else:
    print("Mock already applied.")
