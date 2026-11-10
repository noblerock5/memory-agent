from app.api.v1.endpoints import chat, knowledge, memory
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(memory.router)
api_router.include_router(knowledge.router)
