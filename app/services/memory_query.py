"""记忆检索 (读路径): 画像 / 事实混合评分 / 近期对话"""

import math
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.chat_message import ChatMessage
from app.models.memory_fact import MemoryFact
from app.models.user_profile import UserProfile

# 查询改写: 问题关键词 -> 记忆 key 别名 (演示版"查询改写"步骤)
_KEY_ALIASES = {
    "identity.name": ("名字", "叫什么", "我是谁", "姓名"),
    "work.current_focus": ("负责", "工作", "项目", "在做什么", "做什么"),
    "preference.style": ("回复", "回答风格", "怎么回"),
    "preference.like": ("喜欢", "偏好", "习惯"),
}


def _bigrams(text: str) -> set[str]:
    text = text.strip().lower()
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _key_alias_boost(fact_key: str, query: str) -> float:
    boost = 0.0
    for key, words in _KEY_ALIASES.items():
        if fact_key.startswith(key) or fact_key.startswith(key.split(".")[0]):
            if any(w in query for w in words):
                boost = max(boost, 1.2)
    return boost


def _recency_decay(dt: datetime | None) -> float:
    if dt is None:
        return 0.0
    days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400)
    return 1.0 - min(days / 180.0, 0.5)  # 180 天内线性衰减, 下限 0.5


async def get_profile(session: AsyncSession, user_id: str) -> UserProfile | None:
    return await session.get(UserProfile, user_id)


async def search_facts(
    session: AsyncSession, user_id: str, query: str, limit: int | None = None
) -> list[dict]:
    """混合评分检索: 关键词重叠 + key 别名 + 置信度 + 时间衰减."""
    limit = limit or settings.MAX_FACTS_INJECTED
    res = await session.execute(
        select(MemoryFact)
        .where(
            MemoryFact.user_id == user_id,
            MemoryFact.status == "active",
        )
        .order_by(MemoryFact.updated_at.desc())
    )
    facts = res.scalars().all()

    q_tokens = _bigrams(query)
    scored = []
    for f in facts:
        f_tokens = _bigrams(f"{f.key} {f.value}")
        overlap = (
            len(q_tokens & f_tokens) / max(1, math.sqrt(len(q_tokens)))
            if q_tokens
            else 0.0
        )
        score = (
            0.1  # 基线: 保证冷启动时仍有记忆可注入
            + f.confidence * 0.3
            + _recency_decay(f.last_confirmed_at or f.updated_at)
            + min(overlap, 1.0) * 1.5
            + _key_alias_boost(f.key, query)
        )
        scored.append(
            {
                "id": f.id,
                "key": f.key,
                "value": f.value,
                "confidence": round(f.confidence, 2),
                "updated_at": f.updated_at.isoformat() if f.updated_at else None,
                "score": round(score, 3),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


async def recent_messages(
    session: AsyncSession, user_id: str, limit: int | None = None
) -> list[ChatMessage]:
    limit = limit or settings.RECENT_MESSAGE_LIMIT
    res = await session.execute(
        select(ChatMessage)
        .where(ChatMessage.user_id == user_id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
    )
    return list(reversed(res.scalars().all()))


async def list_facts(
    session: AsyncSession, user_id: str, include_superseded: bool = False
) -> list[MemoryFact]:
    stmt = select(MemoryFact).where(MemoryFact.user_id == user_id)
    if not include_superseded:
        stmt = stmt.where(MemoryFact.status == "active")
    stmt = stmt.order_by(MemoryFact.key, MemoryFact.id.desc())
    res = await session.execute(stmt)
    return list(res.scalars().all())
