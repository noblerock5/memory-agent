"""记忆管理端点: 查看 / 遗忘 (透明可审计)"""

from datetime import timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import get_db
from app.models.memory_fact import MemoryFact
from app.models.user_profile import UserProfile
from app.schemas.chat import DeleteResponse, FactListResponse, FactOut, ProfileOut
from app.services.memory_query import list_facts

router = APIRouter(prefix="/memory", tags=["memory"])


def _iso(dt) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def _to_fact_out(f: MemoryFact) -> FactOut:
    return FactOut(
        id=f.id,
        user_id=f.user_id,
        key=f.key,
        value=f.value,
        confidence=round(f.confidence, 2),
        status=f.status,
        source_message_id=f.source_message_id,
        valid_from=_iso(f.valid_from),
        valid_to=_iso(f.valid_to),
        last_confirmed_at=_iso(f.last_confirmed_at),
        updated_at=_iso(f.updated_at),
    )


@router.get("/facts", response_model=FactListResponse)
async def get_facts(
    user_id: str,
    include_superseded: bool = False,
    db: AsyncSession = Depends(get_db),
) -> FactListResponse:
    facts = await list_facts(db, user_id, include_superseded=include_superseded)
    return FactListResponse(count=len(facts), items=[_to_fact_out(f) for f in facts])


@router.delete("/facts/{fact_id}", response_model=DeleteResponse)
async def delete_fact(fact_id: int, db: AsyncSession = Depends(get_db)) -> DeleteResponse:
    fact = await db.get(MemoryFact, fact_id)
    if fact is None:
        raise HTTPException(status_code=404, detail="记忆不存在")
    await db.delete(fact)  # 遗忘: 单条硬删除, 用户明确指定的记忆
    await db.commit()
    return DeleteResponse(deleted=True, fact_id=fact_id)


@router.get("/profile/{user_id}", response_model=ProfileOut)
async def get_profile(user_id: str, db: AsyncSession = Depends(get_db)) -> ProfileOut:
    profile = await db.get(UserProfile, user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="画像不存在")
    return ProfileOut(
        user_id=profile.user_id,
        display_name=profile.display_name,
        updated_at=_iso(profile.updated_at),
    )
