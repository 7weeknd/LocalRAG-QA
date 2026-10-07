"""大模型调用模块。

只依赖 requests 调 OpenAI 兼容的 /chat/completions 接口，
因此可无缝接入 DeepSeek / 豆包 / 通义 / 本地 Ollama 等任何兼容服务。

设计原则：
- 未配置 Key 时走「抽取式回答」兜底，保证脱机也能演示；
- 统一超时与异常，调用失败不影响主流程。
"""
import requests


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 30, max_tokens: int = 1024):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def generate(self, system: str, user: str) -> str:
        """发起一次对话补全，返回模型文本。

        Ollama 走原生 /api/chat（支持 think=False 关闭思考模式）；
        其余 OpenAI 兼容服务走 /v1/chat/completions。
        """
        if ":11434" in self.base_url or "localhost" in self.base_url:
            return self._generate_ollama(system, user)
        return self._generate_openai(system, user)

    def _generate_openai(self, system: str, user: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _generate_ollama(self, system: str, user: str) -> str:
        # 原生端点在 {host}/api/chat，不带 /v1 前缀
        host = self.base_url.replace("/v1", "").rstrip("/")
        url = f"{host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # 关键：关闭思考模式，否则 qwen3/deepseek-r1 等思考版模型
            # 会把 reasoning 塞满 num_predict 上限，最终 content 为空
            "think": False,
            "options": {"num_predict": self.max_tokens, "temperature": 0.2},
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()