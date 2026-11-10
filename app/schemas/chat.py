"""API 请求/响应模型"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64, description="用户唯一标识(无会话概念, 仅凭 user_id 跨对话记忆)")
    message: str = Field(min_length=1, max_length=4000)


class MemoryChange(BaseModel):
    action: str
    key: str
    value: str
    fact_id: int | None = None


class FactInjected(BaseModel):
    id: int
    key: str
    value: str
    confidence: float
    score: float
    updated_at: str | None = None


class EpisodicMsg(BaseModel):
    role: str
    content: str
    created_at: str | None = None


class KbHit(BaseModel):
    doc_name: str
    chunk_index: int
    content: str
    score: float


class MemoryDebug(BaseModel):
    facts_injected: list[FactInjected]
    recent_messages_used: int
    episodic_used: int = 0
    episodic: list[EpisodicMsg] = []
    kb_used: int = 0
    kb: list[KbHit] = []
    display_name: str | None = None
    changes: list[MemoryChange]


class ChatResponse(BaseModel):
    reply: str
    memory: MemoryDebug


class FactOut(BaseModel):
    id: int
    user_id: str
    key: str
    value: str
    confidence: float
    status: str
    source_message_id: int | None
    valid_from: str | None = None
    valid_to: str | None = None
    last_confirmed_at: str | None = None
    updated_at: str | None = None


class FactListResponse(BaseModel):
    count: int
    items: list[FactOut]


class ProfileOut(BaseModel):
    user_id: str
    display_name: str | None
    updated_at: str | None = None


class DeleteResponse(BaseModel):
    deleted: bool
    fact_id: int
