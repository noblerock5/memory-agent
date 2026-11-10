"""知识库端点: 文档上传入库 / 语义检索 / 健康状态"""

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from app.services import knowledge

router = APIRouter(prefix="/kb", tags=["knowledge-base"])


class UploadRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    doc_name: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2_000_000)


@router.post("/upload")
async def upload_text(req: UploadRequest) -> dict:
    """纯文本/Markdown 直接入库."""
    result = knowledge.ingest_text(req.user_id, req.doc_name, req.text)
    return {"ok": result.get("stored", 0) > 0, **result}


@router.post("/upload-file")
async def upload_file(user_id: str = Form(...), file: UploadFile = File(...)) -> dict:
    """上传 .txt/.md 文件入库 (演示范围; PDF/DOCX 后续接入解析器)."""
    name = file.filename or "untitled.txt"
    if not name.lower().endswith((".txt", ".md", ".markdown")):
        return {"ok": False, "error": "当前仅支持 .txt/.md/.markdown 文件"}
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("gb18030", errors="ignore")
    result = knowledge.ingest_text(user_id, name, text)
    return {"ok": result.get("stored", 0) > 0, **result}


@router.get("/search")
async def search(user_id: str, q: str, top_k: int | None = None) -> dict:
    hits = knowledge.search(user_id, q, top_k)
    return {"count": len(hits), "items": hits}


@router.get("/status")
async def status() -> dict:
    return knowledge.status()


@router.post("/reconnect")
async def reconnect() -> dict:
    """手动重试 Milvus 连接 (服务器重启 Milvus 后调用)."""
    knowledge.reset_connection()
    return knowledge.status()
