"""记忆检索 (读路径): 画像 / 事实混合评分 / 近期对话 / 情景记忆时间召回"""

import math
import re
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.chat_message import ChatMessage
from app.models.memory_fact import MemoryFact
from app.models.user_profile import UserProfile

try:  # 用户口中的"昨天/前天"按中国时区换算, 而不是数据库的 UTC
    from zoneinfo import ZoneInfo

    _LOCAL_TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # pragma: no cover
    _LOCAL_TZ = timezone.utc

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


# ---- 日期意图: "昨天/前天/今天/上周/X月X日" 问的是时间窗, 不是主题词 ----
# 这类问题没有可检索的关键词, 必须按 created_at 区间直查 (2026-11-11 实测踩坑:
# "昨天我跟你说了什么"不命中任何关键词, LLM 只看到近10条就答"查不到昨天")
_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"昨天|昨日"), "day-1"),
    (re.compile(r"前天|前日"), "day-2"),
    (re.compile(r"今天|今日|刚才|刚刚|早些时候"), "today"),
    (re.compile(r"上周|过去一周|最近一周|近几天|这几天|上周[一二三四五六日天]"), "week"),
]
_MD_RE = re.compile(r"(\d{1,2})月(\d{1,2})[日号]")


def detect_date_intent(query: str) -> tuple[str, datetime, datetime] | None:
    """检测日期意图, 返回 (标签, start_utc, end_utc); 边界按中国时区换算为 UTC."""
    now = datetime.now(_LOCAL_TZ)

    def day_start(d: datetime) -> datetime:
        return d.replace(hour=0, minute=0, second=0, microsecond=0)

    label = next((key for pat, key in _DATE_PATTERNS if pat.search(query)), None)
    if label is None:
        m = _MD_RE.search(query)
        if not m:
            return None
        try:  # "3月5日" -> 今年当天窗口 (跨年判断从简, 演示够用)
            d = day_start(now).replace(month=int(m.group(1)), day=int(m.group(2)))
        except ValueError:
            return None
        label = f"{m.group(1)}月{m.group(2)}日"
        return (label, d.astimezone(timezone.utc), (d + timedelta(days=1)).astimezone(timezone.utc))

    if label == "day-1":
        start, end, name = day_start(now) - timedelta(days=1), day_start(now), "昨天"
    elif label == "day-2":
        start, end, name = day_start(now) - timedelta(days=2), day_start(now) - timedelta(days=1), "前天"
    elif label == "today":
        start, end, name = day_start(now), now, "今天"
    else:  # week
        start, end, name = day_start(now) - timedelta(days=7), now, "最近一周"
    return (f"{name}({start:%m-%d})", start.astimezone(timezone.utc), end.astimezone(timezone.utc))


class EpisodicResult(NamedTuple):
    """情景召回结果: msgs=时间正序原始消息; window_label/note 供 prompt 与前端标注."""

    msgs: list[dict]
    intent: str | None = None
    window_label: str | None = None
    note: str | None = None


# ---- 指代消解: "前三个问题/上面那些/刚才说的" 指的是上一轮回答里的清单条目 ----
# 不做消解的话, 这类追问无日期词无主题词, 什么召回都命不中, LLM 只能拿近10条核对,
# 早前的消息不在窗口里就会被误判为"不存在" (2026-11-11 实测: 追问归属引发编造道歉)
_ANAPHORA_RE = re.compile(r"前[一二三四五六七八九十\d]+个|第[一二三四五六七八九十\d]+个|上面|以上|刚才|刚刚|那些|这些问题|前面提到|清单")
_CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+[\.、)）]|[-*•]\s*)\s*(.+)$")


def extract_list_items(text: str, limit: int = 5) -> list[str]:
    """从上一轮回答中提取编号条目 (指代消解的展开源).

    LLM 的清单常是行内编号(" 1. xxx 2. xxx"), 不能按行拆——按编号标记全文切分;
    条目截断到 40 字符只作检索词用, bigram 召回不受截断影响.
    分隔符只认 "数字." / "数字、" (避免 "11-11）" 这类日期被误切), 导语段(以冒号结尾)跳过.
    """
    items: list[str] = []
    segments = re.split(r"\s*\d+[\.、]\s*", text)
    for seg in segments[1:]:
        s = seg.split("\n")[0]
        s = s.replace("**", "").replace("「", "").replace("」", "").strip(" 。，,？?？.．-—")
        s = re.sub(r"\s+", " ", s)[:40]
        if not s or s.endswith(("：", ":")):
            continue
        items.append(s)
        if len(items) >= limit:
            break
    return items


def anaphora_item_count(query: str) -> int:
    """解析"前三个/前两个"中的数量, 供指代展开截取."""
    m = re.search(r"前([一二两三四五六七八九十\d]+)个", query)
    if not m:
        return 5
    token = m.group(1)
    if token.isdigit():
        return min(int(token), 10)
    return min(_CN_NUM.get(token, 5), 10)


async def recall_for_items(session: AsyncSession, user_id: str, items: list[str]) -> tuple[list[dict], str]:
    """逐条目回查对话日志, 返回 (去重后时间正序消息, 指代消解标注)."""
    msgs_by_id: dict[int, dict] = {}
    parts: list[str] = []
    for it in items:
        hits = await search_all_messages(session, user_id, it, limit=2)
        for h in hits:
            msgs_by_id.setdefault(h["id"], h)
        day = hits[0]["created_at"][:10] if hits else None
        parts.append(f"「{it}」→ " + (f"命中日志 {len(hits)} 条(最早 {day})" if hits else "日志中未检索到"))
    note = "指代消解：" + "；".join(parts) + "。以下原始消息均来自当前 user_id 的日志（按用户隔离，不存在其他用户的消息）。"
    msgs = sorted(msgs_by_id.values(), key=lambda x: x["id"])[:12]
    return msgs, note


async def recall_messages_in_window(
    session: AsyncSession, user_id: str, start: datetime, end: datetime, limit: int = 12
) -> list[dict]:
    """按时间窗直查原始消息 (user+assistant 双角色, 时间正序)."""
    res = await session.execute(
        select(ChatMessage)
        .where(
            ChatMessage.user_id == user_id,
            ChatMessage.created_at >= start,
            ChatMessage.created_at < end,
        )
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


async def recall_episodic(
    session: AsyncSession, user_id: str, query: str, prev_assistant_text: str | None = None
) -> EpisodicResult:
    """情景记忆召回入口: 日期意图 → 指代消解 → 锚点/关键词, 返回时间正序原始消息."""
    date_hit = detect_date_intent(query)
    if date_hit:
        label, start, end = date_hit
        msgs = await recall_messages_in_window(session, user_id, start, end)
        return EpisodicResult(msgs, "date", label)
    if _ANAPHORA_RE.search(query) and prev_assistant_text:
        items = extract_list_items(prev_assistant_text, anaphora_item_count(query))
        if items:
            msgs, note = await recall_for_items(session, user_id, items)
            return EpisodicResult(msgs, "anaphora", note=note)
    intent = detect_temporal_intent(query)
    if intent == "first":
        hits = await recall_first_messages(session, user_id)
        return EpisodicResult(sorted(hits, key=lambda x: x["id"]), "first")
    if intent == "past":
        hits = await search_all_messages(session, user_id, query)
        return EpisodicResult(sorted(hits, key=lambda x: x["id"]), "past")
    return EpisodicResult([])


async def list_facts(
    session: AsyncSession, user_id: str, include_superseded: bool = False
) -> list[MemoryFact]:
    stmt = select(MemoryFact).where(MemoryFact.user_id == user_id)
    if not include_superseded:
        stmt = stmt.where(MemoryFact.status == "active")
    stmt = stmt.order_by(MemoryFact.key, MemoryFact.id.desc())
    res = await session.execute(stmt)
    return list(res.scalars().all())
