"""L2 语义记忆: 结构化事实 (带生命周期与来源追溯, 冲突时 superseded 不物理删除)"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class MemoryFact(Base):
    __tablename__ = "memory_facts"
    __table_args__ = (Index("ix_fact_user_key_status", "user_id", "key", "status"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    entity: Mapped[str] = mapped_column(String(64), default="user")
    key: Mapped[str] = mapped_column(String(128))  # 如 identity.name / work.current_focus / preference.*
    value: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | superseded
    source_message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
