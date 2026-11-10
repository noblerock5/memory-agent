"""应用入口"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.db import engine
from app.models import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)


# 健康检查 (必须先于静态挂载注册, 否则会被 "/" mount 拦截)
@app.get("/health", tags=["system"])
async def health():
    db_ok = True
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    return {"status": "ok", "db": db_ok, "version": settings.APP_VERSION}


# 静态测试页面 (已注册的 /api/*、/health 与 "/" 路由优先于 mount)
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
import os  # noqa: E402

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):

    # 首页禁用缓存, 保证页面更新立即生效 (必须在 mount 之前注册)
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(
            os.path.join(_static_dir, "index.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
