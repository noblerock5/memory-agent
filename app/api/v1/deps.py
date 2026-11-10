"""依赖注入"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import SessionLocal
from app.llm.llm_client import LLMClient

_llm = LLMClient()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


def get_llm() -> LLMClient:
    return _llm
