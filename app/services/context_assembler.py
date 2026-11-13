"""上下文组装引擎 (无会话架构的核心): 画像常驻 + 相关事实按需 + 近期对话, 组装系统提示"""

import re
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

try:  # 时间标签统一用北京时间——用户说"昨天"指北京日历日
    from zoneinfo import ZoneInfo

    _LOCAL_TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # pragma: no cover
    _LOCAL_TZ = timezone.utc

# 盘点类意图: 问知识库里"有什么文件/哪些文档"是聚合问题, 向量检索无法回答,
# 必须走文件清单层 (knowledge.list_docs) 而不是语义检索 top-k
_LIST_INTENT_RE = re.compile(
    r"(有什么|有哪些|都有什么|都有哪些|列出|列举|盘点|统计).{0,4}(文件|文档|资料|知识)"
    r"|(文件|文档|资料)(列表|清单|都有哪些|有哪些)"
    r"|知识库(里|中)?(都)?有(什么|哪些)"
)


@dataclass
class ContextBundle:
    system_prompt: str
    profile: UserProfile | None
    facts: list[dict] = field(default_factory=list)
    recent: list[dict] = field(default_factory=list)
    episodic: list[dict] = field(default_factory=list)
    episodic_note: str | None = None  # 日期类召回的时间窗标注(含"无记录"提示)
    kb_hits: list[dict] = field(default_factory=list)
    kb_docs: list[dict] = field(default_factory=list)  # 盘点类问题时注入的文件清单


def _fmt_dt(dt_str: str | None) -> str:
    """北京时间短标签 (月-日 时:分)——归日必须精确到时刻, 只给日期不够."""
    if not dt_str:
        return "未知时间"
    try:
        dt = datetime.fromisoformat(dt_str).astimezone(_LOCAL_TZ)
        return dt.strftime("%m-%d %H:%M")
    except ValueError:
        return "未知时间"


def build_system_prompt(
    profile: UserProfile | None,
    facts: list[dict],
    episodic: list[dict] | None = None,
    kb_hits: list[dict] | None = None,
    kb_docs: list[dict] | None = None,
    episodic_note: str | None = None,
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

    if episodic or episodic_note:
        lines.append("【历史对话记录】（从完整对话日志中召回的原始消息，可引用原文回答）")
        if episodic_note:
            lines.append(f"- 时间范围：{episodic_note}")
        for m in episodic:
            lines.append(f"- {_fmt_dt(m['created_at'])} {m['role']}：{m['content'][:200]}")
        lines.append("")

    if kb_hits:
        lines.append("【知识库片段】（从共享企业知识库语义检索得到，引用时注明文档名）")
        for h in kb_hits:
            lines.append(f"- （{h['doc_name']} #{h['chunk_index']}，相关度 {h['score']}）{h['content'][:300]}")
        lines.append("")

    if kb_docs:
        lines.append(
            f"【知识库文件清单】（当前知识库共 {len(kb_docs)} 个文档，问\"有什么文件/哪些文档\"时按此如实盘点）"
        )
        for d in kb_docs:
            owner_label = "共享" if d["owner"] == knowledge.SHARED_OWNER else "私有"
            lines.append(
                f"- {d['doc_name']}（{owner_label}，{d['chunks']} 块）— {d['preview']}"
            )
        lines.append("")

    lines.append("【使用规则】")
    lines.append("1. 回答时优先使用上述记忆，无需用户重复自我介绍。")
    lines.append("2. 若记忆与用户当前消息冲突，以当前消息为准，并在回复末尾提示记忆已更新。")
    lines.append("3. 记忆中找不到的信息如实说明，不要编造。")
    lines.append("4. 若【历史对话记录】中有与问题相关的内容，直接引用原文回答。")
    lines.append("5. 问\"有什么文件/哪些文档\"时，严格按【知识库文件清单】逐个盘点，不要遗漏或编造文件名。")
    lines.append("6. 用户问某段时间（昨天/前天/今天等）说了什么时，严格以【历史对话记录】标注的时间范围为准；")
    lines.append("   该时间范围内若无消息记录，就如实回答查不到，绝不能拿近期其他时间的消息冒充。")
    lines.append("6b. 近期对话的每条消息都带 [月-日 时:分] 北京时间标签——按时间戳归日：窗口里的旧消息")
    lines.append("   （如标注为昨天/上周）不得算进\"今天\"或\"本轮\"；时间戳就是归日的唯一依据。")
    lines.append("7. 所有记忆检索均按当前 user_id 过滤——你上下文中的消息与事实都属于当前用户，")
    lines.append("   其他用户的消息不可能出现；回答\"是谁问的\"时以【历史对话记录】中的时间戳为依据，")
    lines.append("   未检索到的条目如实说明，不要推断为\"误加\"或\"不存在\"。")
    lines.append("8. 输出一律使用规范 Markdown（前端会渲染表格/列表）：直接结论放第一行；")
    lines.append("   多条并列信息用列表（- 或 1.）；≥3 组对比信息（多项问题/多条记录/多选项）优先用表格；")
    lines.append("   不要把多个要点挤在同一行。长度仍遵守用户偏好——极简＝少而整洁，不是挤成一段。")
    return "\n".join(lines)


async def assemble_context(session: AsyncSession, user_id: str, query: str) -> ContextBundle:
    profile = await get_profile(session, user_id)
    facts = await search_facts(session, user_id, query)
    recent = await recent_messages(session, user_id)
    # 指代消解的展开源: 上一轮助手回答 (此时本轮用户消息尚未落盘, recent[-1] 即上一轮回复)
    prev_assistant = next(
        (m.content for m in reversed(recent) if m.role == "assistant"), None
    )
    ep = await recall_episodic(session, user_id, query, prev_assistant_text=prev_assistant)
    episodic = ep.msgs
    episodic_note = ep.note
    if ep.intent == "date":
        episodic_note = (
            f"{ep.window_label}，共召回 {len(episodic)} 条原始消息"
            if episodic
            else f"{ep.window_label}时间窗内没有任何对话记录（用户当天未与助手对话），请如实告知查不到"
        )
    # 知识库 (Milvus 不可达时内部降级为空列表; embedding 计算较重, 用线程池避免阻塞事件循环)
    kb_hits: list[dict] = []
    kb_docs: list[dict] = []
    if settings.MILVUS_URI and settings.MILVUS_URI.startswith("http"):
        import anyio

        if _LIST_INTENT_RE.search(query):
            # 盘点类问题: 注入文件清单 (带 TTL 缓存), 不做语义检索——
            # top-k 片段对"列出所有文件"毫无帮助, 只会误导 LLM
            kb_docs = await anyio.to_thread.run_sync(knowledge.list_docs, user_id)
        else:
            kb_hits = await anyio.to_thread.run_sync(knowledge.search, user_id, query)

    recent_dicts = [
        {"role": m.role, "content": m.content, "id": m.id, "created_at": m.created_at.isoformat()}
        for m in recent
    ]
    return ContextBundle(
        system_prompt=build_system_prompt(profile, facts, episodic, kb_hits, kb_docs, episodic_note),
        profile=profile,
        facts=facts,
        recent=recent_dicts,
        episodic=episodic,
        episodic_note=episodic_note,
        kb_hits=kb_hits,
        kb_docs=kb_docs,
    )
