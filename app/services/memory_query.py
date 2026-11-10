"""记忆检索 (读路径): 画像 / 事实混合评分 / 近期对话 / 情景记忆时间召回"""

import math
import re
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


# ---- 情景记忆时间召回: 修复"结构化事实检索覆盖不到原始消息历史"的盲区 ----
# 时间意图正则: 问"第一次/最早"类问题需要按时间正序取原始消息, 而非按相关性评分
# (相关性评分带时间衰减, 越旧的越靠后, 与"第一次"的意图正好相反)
_FIRST_RE = re.compile(r"第一次|第1次|最早|最开始|头一回|初次|开天辟地")
_PAST_RE = re.compile(r"上次|之前|以前|历史|问过|聊过|说过|谈过|讨论过|记录")


def detect_temporal_intent(query: str) -> str | None:
    """检测时间意图: 'first'=问最早的, 'past'=问之前聊过的, None=非时间类问题."""
    if _FIRST_RE.search(query):
        return "first"
    if _PAST_RE.search(query):
        return "past"
    return None


async def recall_first_messages(
    session: AsyncSession, user_id: str, limit: int = 4
) -> list[dict]:
    """时间正序取最早的若干条用户消息 (情景记忆的'锚点'召回)."""
    res = await session.execute(
        select(ChatMessage)
        .where(ChatMessage.user_id == user_id, ChatMessage.role == "user")
        .order_by(ChatMessage.id.asc())
        .limit(limit)
    )
    return [
        {
            "role": m.role,
            "content": m.content,
            "id": m.id,
            "created_at": m.created_at.isoformat(),
        }
        for m in res.scalars().all()
    ]


async def search_all_messages(
    session: AsyncSession, user_id: str, query: str, limit: int = 6
) -> list[dict]:
    """全量原始消息检索 (大表应换向量库, 演示规模直接全扫+bigram评分)."""
    res = await session.execute(
        select(ChatMessage)
        .where(ChatMessage.user_id == user_id)
        .order_by(ChatMessage.id.desc())
        .limit(2000)  # 演示保护: 最多回扫 2000 条
    )
    msgs = res.scalars().all()

    q_tokens = _bigrams(query)
    scored = []
    for m in msgs:
        m_tokens = _bigrams(m.content)
        overlap = (
            len(q_tokens & m_tokens) / max(1, math.sqrt(len(q_tokens)))
            if q_tokens
            else 0.0
        )
        if overlap <= 0:
            continue
        scored.append(
            {
                "role": m.role,
                "content": m.content,
                "id": m.id,
                "created_at": m.created_at.isoformat(),
                "score": round(overlap, 3),
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


async def recall_episodic(session: AsyncSession, user_id: str, query: str) -> list[dict]:
    """情景记忆召回入口: 按时间意图路由到对应策略, 返回按时间正序的原始消息."""
    intent = detect_temporal_intent(query)
    if intent == "first":
        hits = await recall_first_messages(session, user_id)
        return sorted(hits, key=lambda x: x["id"])
    if intent == "past":
        hits = await search_all_messages(session, user_id, query)
        return sorted(hits, key=lambda x: x["id"])
    return []


async def list_facts(
    session: AsyncSession, user_id: str, include_superseded: bool = False
) -> list[MemoryFact]:
    stmt = select(MemoryFact).where(MemoryFact.user_id == user_id)
    if not include_superseded:
        stmt = stmt.where(MemoryFact.status == "active")
    stmt = stmt.order_by(MemoryFact.key, MemoryFact.id.desc())
    res = await session.execute(stmt)
    return list(res.scalars().all())
