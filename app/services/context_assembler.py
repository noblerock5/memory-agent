"""上下文组装引擎 (无会话架构的核心): 画像常驻 + 相关事实按需 + 近期对话, 组装系统提示"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user_profile import UserProfile
from app.services.memory_query import get_profile, recent_messages, search_facts


@dataclass
class ContextBundle:
    system_prompt: str
    profile: UserProfile | None
    facts: list[dict] = field(default_factory=list)
    recent: list[dict] = field(default_factory=list)


def _fmt_dt(dt_str: str | None) -> str:
    if not dt_str:
        return "未知时间"
    try:
        dt = datetime.fromisoformat(dt_str).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return "未知时间"


def build_system_prompt(profile: UserProfile | None, facts: list[dict]) -> str:
    lines = [
        "你是用户的长期个人助手。以下是关于用户的持久记忆（跨对话有效）。",
        "",
    ]
    if profile and profile.display_name:
        lines.append(f"【用户画像】称呼：{profile.display_name}")
        lines.append("")

    lines.append("【长期记忆】")
    if facts:
        for f in facts:
            lines.append(
                f"- {f['key']}：{f['value']}（置信度 {f['confidence']}，更新于 {_fmt_dt(f['updated_at'])}）"
            )
    else:
        lines.append("- （暂无记忆）")
    lines.append("")
    lines.append("【使用规则】")
    lines.append("1. 回答时优先使用上述记忆，无需用户重复自我介绍。")
    lines.append("2. 若记忆与用户当前消息冲突，以当前消息为准，并在回复末尾提示记忆已更新。")
    lines.append("3. 记忆中找不到的信息如实说明，不要编造。")
    return "\n".join(lines)


async def assemble_context(session: AsyncSession, user_id: str, query: str) -> ContextBundle:
    profile = await get_profile(session, user_id)
    facts = await search_facts(session, user_id, query)
    recent = await recent_messages(session, user_id)

    recent_dicts = [
        {"role": m.role, "content": m.content, "id": m.id, "created_at": m.created_at.isoformat()}
        for m in recent
    ]
    return ContextBundle(
        system_prompt=build_system_prompt(profile, facts),
        profile=profile,
        facts=facts,
        recent=recent_dicts,
    )
