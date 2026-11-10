"""LLM 客户端 (OpenAI 兼容协议)

未配置 LLM_BASE_URL 时进入离线演示模式: 直接基于检索到的记忆拼装回答,
保证 demo 在无任何模型服务时也能完整验证记忆链路.
"""

import json
import logging
import re
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_NAME_QUESTION_RE = re.compile(r"(名字|叫什么|我是谁)")


class LLMClient:
    """统一的对话与结构化抽取入口"""

    async def chat(self, messages: list[dict], context: dict[str, Any] | None = None) -> str:
        if not settings.LLM_BASE_URL:
            return self._offline_reply(messages, context or {})

        payload = {"model": settings.LLM_MODEL, "messages": messages, "temperature": 0.3}
        headers = {}
        if settings.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"
        url = settings.LLM_BASE_URL.rstrip("/") + "/chat/completions"
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def extract_json(self, system: str, user: str) -> list | None:
        """调用 LLM 抽取结构化 JSON; 失败返回 None, 由调用方降级到规则抽取."""
        if not settings.LLM_BASE_URL:
            return None
        try:
            text = await self.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
            start, end = text.find("["), text.rfind("]")
            if start == -1 or end == -1:
                return None
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, list) else None
        except Exception as exc:  # 降级不抛出
            logger.warning("LLM 抽取失败, 降级到规则抽取: %s", exc)
            return None

    def _offline_reply(self, messages: list[dict], context: dict[str, Any]) -> str:
        """离线演示模式: 基于注入的记忆直接作答."""
        user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        facts: list[dict] = context.get("facts", [])

        name = next((f["value"] for f in facts if f["key"] == "identity.name"), None)
        others = [
            f"{self._key_label(f['key'])}：{f['value']}"
            for f in facts
            if f["key"] != "identity.name"
        ]

        if not facts:
            return "（离线演示模式）我目前还没有关于你的记忆，先告诉我一些你的信息吧。"

        if name and _NAME_QUESTION_RE.search(user_msg):
            reply = f"根据我的长期记忆，你叫{name}。"
        else:
            reply = "（离线演示模式：未配置 LLM，以下基于检索到的记忆作答）\n"

        if others:
            reply += "我还记得：" + "；".join(others) + "。"
        elif not _NAME_QUESTION_RE.search(user_msg):
            reply += f"关于你的记忆：你的名字是{name}。"
        reply += "\n（配置 .env 中的 LLM_BASE_URL 后即可接入真实模型）"
        return reply

    @staticmethod
    def _key_label(key: str) -> str:
        labels = {
            "work.current_focus": "当前工作重点",
            "preference.like": "偏好",
            "preference.style": "回复风格偏好",
        }
        return labels.get(key, key)
