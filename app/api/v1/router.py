from app.api.v1.endpoints import chat, memory
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(memory.router)
