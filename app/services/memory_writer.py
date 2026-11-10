"""记忆写入管线: 消息落盘 -> 事实抽取(LLM优先/规则降级) -> 冲突消解 -> 双写"""

import logging
import re
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.llm.llm_client import LLMClient
from app.models.chat_message import ChatMessage
from app.models.memory_fact import MemoryFact
from app.models.user_profile import UserProfile

logger = logging.getLogger(__name__)

# ---- 规则抽取器 (LLM 未配置或抽取失败时的降级方案) ----
NAME_RE = re.compile(r"(?:我叫|我的名字叫|叫我|以后叫我)\s*([一-龥a-zA-Z0-9·]{1,12})")
FOCUS_RE = re.compile(r"(?:我负责|我管理|我在做|在负责|负责)\s*([^，。；,;！!？?]{2,20})")
LIKE_RE = re.compile(r"(?:我喜欢|我偏好|我习惯|回复请用|回答请用|请用)\s*([^，。；,;！!？?]{1,20})")

# 疑问句不抽取; 疑问代词出现在抽取值中也直接拒绝 (防止 "我叫什么名字?" 污染记忆)
_SENT_RE = re.compile(r"[^。！？!?；;\n]*[。！？!?；;\n]?")
_BAD_VALUE_RE = re.compile(r"什么|谁|啥|哪|怎么|如何|多少|吗|吧|呢|[？?]")


def _iter_sentences(text: str):
    for m in _SENT_RE.finditer(text):
        s = m.group(0).strip()
        if s:
            yield s

# identity.* / work.* 属于"单值事实": 新值取代旧值(保留替代谱系); 其余为多值事实, 并存
REPLACE_KEY_PREFIXES = ("identity.", "work.")

EXTRACT_SYSTEM_PROMPT = (
    "你是记忆抽取器。从对话中抽取两类稳定信息, 不抽取临时闲聊:\n"
    "A. 用户属性: 身份/工作/偏好/长期承诺, key 建议 identity.name, work.current_focus, "
    "preference.like, preference.style, goal.*。\n"
    "B. 经验结论(对话中沉淀的可复用知识, 优先抽取): 某问题的解决方案(exp.solution.*)、"
    "技术或业务决策及其原因(exp.decision.*)、踩坑与教训(exp.lesson.*)、验证过的结论(exp.finding.*)。"
    "key 的最后一段用简短主题命名, 如 exp.solution.导入超时, exp.decision.队列选型。value 要自包含"
    "(脱离上下文也能看懂), 包含'问题→结论'或'决策→原因'。\n"
    "entity 字段: 用户属性填 user; 经验结论填 exp。\n"
    "只输出 JSON 数组, 元素形如 "
    '[{"entity":"user","key":"identity.name","value":"李明","confidence":0.95}, '
    '{"entity":"exp","key":"exp.lesson.pg超时","value":"TSV大批量导入超时: 改用20并发+分批commit后稳定","confidence":0.85}], '
    "没有可抽取内容时输出 []。不要输出任何其他文字。"
)


async def append_message(
    session: AsyncSession, user_id: str, role: str, content: str
) -> ChatMessage:
    msg = ChatMessage(user_id=user_id, role=role, content=content)
    session.add(msg)
    await session.commit()
    await session.refresh(msg)
    return msg


def rule_extract(text: str) -> list[dict]:
    """规则降级抽取: 覆盖 demo 中最常见的三类事实.

    防污染纪律: 疑问句整体跳过; 抽取值命中疑问代词也拒绝 (宁严勿滥).
    """
    drafts: list[dict] = []
    for sent in _iter_sentences(text):
        if sent[-1] in "？??":
            continue

        def _ok(v: str) -> bool:
            return not _BAD_VALUE_RE.search(v)

        if m := NAME_RE.search(sent):
            if _ok(m.group(1)):
                drafts.append(
                    {"entity": "user", "key": "identity.name", "value": m.group(1), "confidence": 0.95}
                )
        if m := FOCUS_RE.search(sent):
            if _ok(m.group(1)):
                drafts.append(
                    {"entity": "user", "key": "work.current_focus", "value": m.group(1), "confidence": 0.75}
                )
        if m := LIKE_RE.search(sent):
            if _ok(m.group(1)):
                drafts.append(
                    {"entity": "user", "key": "preference.like", "value": m.group(1), "confidence": 0.6}
                )
    return drafts


async def _upsert_profile_name(session: AsyncSession, user_id: str, name: str) -> None:
    profile = await session.get(UserProfile, user_id)
    if profile is None:
        session.add(UserProfile(user_id=user_id, display_name=name))
    else:
        profile.display_name = name


async def extract_and_store(
    session: AsyncSession,
    llm: LLMClient,
    user_id: str,
    user_text: str,
    assistant_text: str,
    source_message_id: int | None = None,
) -> list[dict]:
    """抽取事实并落库, 返回 memory_changes 供前端展示."""
    drafts: list[dict] | None = None
    if settings.LLM_BASE_URL:
        raw = await llm.extract_json(
            EXTRACT_SYSTEM_PROMPT,
            f"用户消息：{user_text}\n助手回复：{assistant_text}",
        )
        if isinstance(raw, list):
            drafts = [
                d
                for d in raw
                if isinstance(d, dict)
                and isinstance(d.get("key"), str)
                and isinstance(d.get("value"), str)
                and d["value"].strip()
            ]
    if not drafts:
        drafts = rule_extract(user_text)

    changes: list[dict] = []
    now = datetime.now(timezone.utc)

    for d in drafts:
        key = d["key"].strip()
        value = d["value"].strip()
        replace = key.startswith(REPLACE_KEY_PREFIXES)
        confidence = min(1.0, float(d.get("confidence", 0.8) or 0.8))

        res = await session.execute(
            select(MemoryFact).where(
                MemoryFact.user_id == user_id,
                MemoryFact.key == key,
                MemoryFact.status == "active",
            )
        )
        actives = res.scalars().all()

        same = next((o for o in actives if o.value == value), None)
        if same is not None:
            same.confidence = min(1.0, same.confidence + 0.05)
            same.last_confirmed_at = now
            changes.append({"action": "reinforced", "key": key, "value": value, "fact_id": same.id})
            continue

        action = "added"
        if actives and replace:
            for old in actives:  # 冲突消解: 新替旧, 旧记录降级保留替代谱系, 不物理删除
                old.status = "superseded"
                old.valid_to = now
            action = "superseded"

        fact = MemoryFact(
            user_id=user_id,
            entity=d.get("entity", "user") or "user",
            key=key,
            value=value,
            confidence=confidence,
            source_message_id=source_message_id,
        )
        session.add(fact)
        await session.flush()
        changes.append({"action": action, "key": key, "value": value, "fact_id": fact.id})

        if key == "identity.name":
            await _upsert_profile_name(session, user_id, value)

    await session.commit()
    return changes
