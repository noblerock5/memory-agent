"""上下文组装引擎 (无会话架构的核心): 画像常驻 + 相关事实按需 + 近期对话, 组装系统提示"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.user_profile import UserProfile
from app.services import knowledge
from app.services.memory_query import (
    get_profile,
    recent_messages,
    recall_episodic,
    search_facts,
)


@dataclass
class ContextBundle:
    system_prompt: str
    profile: UserProfile | None
    facts: list[dict] = field(default_factory=list)
    recent: list[dict] = field(default_factory=list)
    episodic: list[dict] = field(default_factory=list)
    kb_hits: list[dict] = field(default_factory=list)


def _fmt_dt(dt_str: str | None) -> str:
    if not dt_str:
        return "未知时间"
    try:
        dt = datetime.fromisoformat(dt_str).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return "未知时间"


def build_system_prompt(
    profile: UserProfile | None,
    facts: list[dict],
    episodic: list[dict] | None = None,
    kb_hits: list[dict] | None = None,
) -> str:
    lines = [
        "你是用户的长期个人助手。以下是关于用户的持久记忆（跨对话有效）。",
        "",
    ]
    if profile and profile.display_name:
        lines.append(f"【用户画像】称呼：{profile.display_name}")
        lines.append("")

    # L2 事实分两组: 用户属性 / 经验结论(exp.* 可复用的解决方案、决策、教训)
    user_facts = [f for f in facts if not f["key"].startswith("exp.")]
    exp_facts = [f for f in facts if f["key"].startswith("exp.")]

    lines.append("【长期记忆·用户属性】")
    if user_facts:
        for f in user_facts:
            lines.append(
                f"- {f['key']}：{f['value']}（置信度 {f['confidence']}，更新于 {_fmt_dt(f['updated_at'])}）"
            )
    else:
        lines.append("- （暂无记忆）")
    lines.append("")

    lines.append("【长期记忆·经验结论】（过往对话沉淀的可复用知识，同类问题直接复用，不要让用户从头再讲）")
    if exp_facts:
        for f in exp_facts:
            lines.append(
                f"- {f['key']}：{f['value']}（置信度 {f['confidence']}，更新于 {_fmt_dt(f['updated_at'])}）"
            )
    else:
        lines.append("- （暂无，随对话积累）")
    lines.append("")

    if episodic:
        lines.append("【历史对话记录】（从完整对话日志中召回的原始消息，按时间正序，可引用原文回答）")
        for m in episodic:
            lines.append(f"- {_fmt_dt(m['created_at'])} {m['role']}：{m['content'][:200]}")
        lines.append("")

    if kb_hits:
        lines.append("【知识库片段】（从共享企业知识库语义检索得到，引用时注明文档名）")
        for h in kb_hits:
            lines.append(f"- （{h['doc_name']} #{h['chunk_index']}，相关度 {h['score']}）{h['content'][:300]}")
        lines.append("")

    lines.append("【使用规则】")
    lines.append("1. 回答时优先使用上述记忆，无需用户重复自我介绍。")
    lines.append("2. 若记忆与用户当前消息冲突，以当前消息为准，并在回复末尾提示记忆已更新。")
    lines.append("3. 记忆中找不到的信息如实说明，不要编造。")
    lines.append("4. 若【历史对话记录】中有与问题相关的内容，直接引用原文回答。")
    return "\n".join(lines)


async def assemble_context(session: AsyncSession, user_id: str, query: str) -> ContextBundle:
    profile = await get_profile(session, user_id)
    facts = await search_facts(session, user_id, query)
    recent = await recent_messages(session, user_id)
    episodic = await recall_episodic(session, user_id, query)
    # 知识库语义检索 (Milvus 不可达时内部降级为空列表; embedding 计算较重, 用线程池避免阻塞事件循环)
    kb_hits: list[dict] = []
    if settings.MILVUS_URI and settings.MILVUS_URI.startswith("http"):
        import anyio

        kb_hits = await anyio.to_thread.run_sync(knowledge.search, user_id, query)

    recent_dicts = [
        {"role": m.role, "content": m.content, "id": m.id, "created_at": m.created_at.isoformat()}
        for m in recent
    ]
    return ContextBundle(
        system_prompt=build_system_prompt(profile, facts, episodic, kb_hits),
        profile=profile,
        facts=facts,
        recent=recent_dicts,
        episodic=episodic,
        kb_hits=kb_hits,
    )
