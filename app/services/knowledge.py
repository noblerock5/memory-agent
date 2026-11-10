"""知识库服务: 个人文件 -> 分块 -> embedding -> Milvus 向量检索

设计要点:
- 懒加载: embedding 模型和 Milvus 连接在首次使用时才初始化
- 优雅降级: Milvus 不可达时 search/ingest 返回安全默认值并记日志, 主对话链路不受影响
- 按用户隔离: 所有检索都带 user_id 过滤 (布尔表达式, 不依赖 partition)
"""

import logging
import threading

from app.core.config import settings

logger = logging.getLogger(__name__)

COLLECTION = "kb_chunks"

_state = {
    "embed_model": None,      # fastembed.TextEmbedding
    "milvus_client": None,    # pymilvus.MilvusClient
    "milvus_ok": False,
    "last_error": None,
    "dim": None,
}
_lock = threading.Lock()


def _get_embed_model():
    """懒加载本地 embedding 模型 (fastembed/ONNX, 首次会下载模型文件)."""
    if _state["embed_model"] is None:
        from fastembed import TextEmbedding

        logger.info("加载 embedding 模型: %s", settings.EMBED_MODEL)
        model = TextEmbedding(model_name=settings.EMBED_MODEL)
        _state["embed_model"] = model
        dim = len(next(model.embed(["dim probe"])))
        _state["dim"] = dim
        logger.info("embedding 模型就绪, dim=%d", dim)
    return _state["embed_model"]


def _get_milvus():
    """懒加载 Milvus 连接并确保集合存在; 失败时标记不可用."""
    if _state["milvus_client"] is not None and _state["milvus_ok"]:
        return _state["milvus_client"]
    with _lock:
        if _state["milvus_client"] is not None and _state["milvus_ok"]:
            return _state["milvus_client"]
        try:
            from pymilvus import DataType, MilvusClient

            client = MilvusClient(uri=settings.MILVUS_URI)
            if not client.has_collection(COLLECTION):
                schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
                schema.add_field("id", DataType.INT64, is_primary=True)
                schema.add_field("user_id", DataType.VARCHAR, max_length=64)
                schema.add_field("doc_name", DataType.VARCHAR, max_length=256)
                schema.add_field("chunk_index", DataType.INT64)
                schema.add_field("content", DataType.VARCHAR, max_length=8192)
                schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=_state["dim"])

                index_params = client.prepare_index_params()
                index_params.add_index(
                    field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE"
                )
                client.create_collection(COLLECTION, schema=schema, index_params=index_params)
                logger.info("已创建 Milvus 集合 %s (dim=%d)", COLLECTION, _state["dim"])
            _state["milvus_client"] = client
            _state["milvus_ok"] = True
            _state["last_error"] = None
            return client
        except Exception as e:  # noqa: BLE001 - 降级路径, 需要吞掉所有连接类异常
            _state["milvus_ok"] = False
            _state["last_error"] = str(e)
            logger.warning("Milvus 不可达, 知识库降级为空结果: %s", e)
            return None


def reset_connection() -> None:
    """供状态接口/运维手动重连."""
    with _lock:
        _state["milvus_client"] = None
        _state["milvus_ok"] = False


# ---- 分块 ----


def chunk_text(text: str) -> list[str]:
    """按段落聚合分块, 目标每块约 KB_CHUNK_SIZE 字符, 块间保留重叠."""
    size, overlap = settings.KB_CHUNK_SIZE, settings.KB_CHUNK_OVERLAP
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        # 超长段落硬切
        while len(p) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(p[:size])
            p = p[size - overlap :]
        if buf and len(buf) + len(p) + 2 > size:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


# ---- 对外接口 ----


def ingest_text(user_id: str, doc_name: str, text: str) -> dict:
    """文档入库: 分块 -> 向量化 -> 写 Milvus. 返回 {chunks, stored, error?}."""
    chunks = chunk_text(text)
    if not chunks:
        return {"chunks": 0, "stored": 0}
    client = _get_milvus()
    if client is None:
        return {"chunks": len(chunks), "stored": 0, "error": _state["last_error"]}
    try:
        model = _get_embed_model()
        vectors = [list(map(float, v)) for v in model.embed(chunks)]
        rows = [
            {
                "user_id": user_id[:64],
                "doc_name": doc_name[:256],
                "chunk_index": i,
                "content": c[:8192],
                "embedding": vectors[i],
            }
            for i, c in enumerate(chunks)
        ]
        client.insert(COLLECTION, rows)
        return {"chunks": len(chunks), "stored": len(rows)}
    except Exception as e:  # noqa: BLE001
        _state["milvus_ok"] = False
        _state["last_error"] = str(e)
        logger.warning("知识库入库失败: %s", e)
        return {"chunks": len(chunks), "stored": 0, "error": str(e)}


def search(user_id: str, query: str, top_k: int | None = None) -> list[dict]:
    """语义检索用户的知识库; Milvus 不可达时返回空列表 (主链路降级)."""
    top_k = top_k or settings.KB_TOP_K
    client = _get_milvus()
    if client is None:
        return []
    try:
        model = _get_embed_model()
        q_vec = list(map(float, next(model.embed([query]))))
        hits = client.search(
            COLLECTION,
            data=[q_vec],
            limit=top_k,
            filter=f'user_id == "{user_id}"',
            output_fields=["doc_name", "chunk_index", "content"],
            search_params={"metric_type": "COSINE"},
        )[0]
        return [
            {
                "doc_name": h["entity"]["doc_name"],
                "chunk_index": h["entity"]["chunk_index"],
                "content": h["entity"]["content"],
                "score": round(h["distance"], 3),
            }
            for h in hits
        ]
    except Exception as e:  # noqa: BLE001
        _state["milvus_ok"] = False
        _state["last_error"] = str(e)
        logger.warning("知识库检索失败: %s", e)
        return []


def status() -> dict:
    """知识库健康状态 (供 /health 与前端面板展示)."""
    s = {
        "enabled": bool(settings.MILVUS_URI),
        "uri": settings.MILVUS_URI,
        "model": settings.EMBED_MODEL,
        "milvus": "unknown",
    }
    client = _get_milvus()
    if client is not None:
        try:
            stats = client.get_collection_stats(COLLECTION)
            s["milvus"] = "ok"
            s["chunks"] = int(stats.get("row_count", stats.get("num_rows", 0)))
        except Exception as e:  # noqa: BLE001
            s["milvus"] = "down"
            s["error"] = str(e)
    else:
        s["milvus"] = "down"
        s["error"] = _state["last_error"]
    return s
