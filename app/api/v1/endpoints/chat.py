"""对话端点: 无 session 概念, 只凭 user_id 完成跨对话记忆"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import get_db, get_llm
from app.llm.llm_client import LLMClient
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    EpisodicMsg,
    FactInjected,
    KbDocItem,
    KbHit,
    MemoryDebug,
    MemoryChange,
)
from app.services.context_assembler import assemble_context
from app.services.memory_writer import append_message, extract_and_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
) -> ChatResponse:
    # 1. 组装上下文 (在落盘用户消息之前, 近期对话不含本条)
    bundle = await assemble_context(db, req.user_id, req.message)

    # 2. 用户消息写入情景记忆 (append-only 事件日志)
    await append_message(db, req.user_id, "user", req.message)

    # 3. 调用 LLM (记忆 + 近期对话 + 当前消息)
    messages = [{"role": "system", "content": bundle.system_prompt}]
    messages += [{"role": m["role"], "content": m["content"]} for m in bundle.recent]
    messages.append({"role": "user", "content": req.message})
    reply = await llm.chat(messages, context={"facts": bundle.facts})

    # 4. 助手回复写入事件日志
    assistant_msg = await append_message(db, req.user_id, "assistant", reply)

    # 5. 异步语义: 抽取事实 -> 冲突消解 -> 双写 (演示中同步执行, 生产可挪到消息队列)
    changes = await extract_and_store(
        db, llm, req.user_id, req.message, reply, source_message_id=assistant_msg.id
    )

    return ChatResponse(
        reply=reply,
        memory=MemoryDebug(
            facts_injected=[
                FactInjected(
                    id=f["id"],
                    key=f["key"],
                    value=f["value"],
                    confidence=f["confidence"],
                    score=f["score"],
                    updated_at=f.get("updated_at"),
                )
                for f in bundle.facts
            ],
            recent_messages_used=len(bundle.recent),
            episodic_used=len(bundle.episodic),
            episodic=[
                EpisodicMsg(role=m["role"], content=m["content"], created_at=m.get("created_at"))
                for m in bundle.episodic
            ],
            episodic_note=bundle.episodic_note,
            kb_used=len(bundle.kb_hits),
            kb=[
                KbHit(
                    doc_name=h["doc_name"],
                    chunk_index=h["chunk_index"],
                    content=h["content"],
                    score=h["score"],
                )
                for h in bundle.kb_hits
            ],
            kb_docs=[KbDocItem(**d) for d in bundle.kb_docs],
            display_name=bundle.profile.display_name if bundle.profile else None,
            changes=[MemoryChange(**c) for c in changes],
        ),
    )
