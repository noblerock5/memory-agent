"""应用配置管理 (pydantic-settings, 与 KLD-License-api 风格对齐)"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from dotenv import load_dotenv

    for _p in (".env.dev", ".env", os.path.join(os.getcwd(), ".env.dev"), os.path.join(os.getcwd(), ".env")):
        if os.path.exists(_p):
            load_dotenv(_p, override=False)
            break
except ImportError:  # pragma: no cover
    pass


class Settings(BaseSettings):
    """应用配置类"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_TITLE: str = "Memory Agent API"
    APP_DESCRIPTION: str = "无会话记忆智能体 Demo API"
    APP_VERSION: str = "0.1.0"

    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # 数据库配置
    DB_HOST: str = "43.143.233.242"
    DB_PORT: int = 5432
    DB_USER: str = "admin"
    DB_PASSWORD: str = ""
    DB_NAME: str = "memory_agent"

    # LLM 配置 (OpenAI 兼容; LLM_BASE_URL 为空时进入离线演示模式)
    LLM_BASE_URL: str = ""
    LLM_API_KEY: str = ""
    LLM_MODEL: str = ""
    LLM_TIMEOUT: float = 30.0

    # 记忆策略
    RECENT_MESSAGE_LIMIT: int = 10
    MAX_FACTS_INJECTED: int = 8

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


settings = Settings()
