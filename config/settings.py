"""
智能办公平台配置，使用 Pydantic Settings 管理所有配置项，支持从环境变量读取
应用名称、LLM、数据库、向量数据库、工具配置
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """应用配置类"""
    
    # ========== 应用基础配置 ==========
    APP_NAME: str = "Enterprise Intelligent Office Assistant"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True
    
    # ========== LLM 配置 ==========
    # 通义千问 API 配置（可通过环境变量 DASHSCOPE_API_KEY 设置）
    DASHSCOPE_API_KEY: Optional[str] = None
    LLM_MODEL_NAME: str = "qwen-plus"  # 可选: qwen-turbo, qwen-plus, qwen-max
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2000
    
    # 备用本地模型路径（如果不用通义千问，可切换到本地模型）
    LOCAL_EMBEDDING_MODEL: Path = PROJECT_ROOT / "models" / "bge_small_zh"
    
    # ========== 数据库配置 ==========
    # MySQL
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "password"
    MYSQL_DATABASE: str = "office_assistant"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # ========== 向量数据库配置 ==========
    CHROMA_PERSIST_DIR: Path = PROJECT_ROOT / "data" / "chroma_db"
    EMBEDDING_MODEL_NAME: Path = PROJECT_ROOT / "models" / "bge_small_zh"  # 中文Embedding模型
    
    # ========== 工具配置 ==========
    # 数据分析工具
    DEFAULT_DATA_PATH: Path = PROJECT_ROOT / "data" / "sales_data.xlsx"
    VISUALIZATION_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "visualizations"
    
    # 文档生成工具
    DOCUMENT_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "documents"
    
    # API 工具 (模拟企业内部系统)
    OA_API_BASE_URL: str = "http://localhost:8000/api/oa"
    CRM_API_BASE_URL: str = "http://localhost:8000/api/crm"
    
    # 知识库
    KNOWLEDGE_BASE_DIR: Path = PROJECT_ROOT / "data" / "knowledge_base"
    
    # ========== 记忆模块配置 ==========
    SHORT_TERM_MEMORY_MAX_LENGTH: int = 10  # 短期记忆最大保留条数
    LONG_TERM_MEMORY_PREFIX: str = "user_memory:"  # Redis key前缀
    
    # ========== 反思模块配置 ==========
    MAX_RETRY_COUNT: int = 3  # 最大重试次数
    
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def llm_configured(self) -> bool:
        """是否已配置可用的 LLM API Key。"""
        return bool(self.DASHSCOPE_API_KEY)

    def ensure_runtime_dirs(self) -> None:
        """创建运行时目录；由应用启动阶段显式调用。"""
        for directory in (
            self.CHROMA_PERSIST_DIR,
            self.VISUALIZATION_OUTPUT_DIR,
            self.DOCUMENT_OUTPUT_DIR,
            self.KNOWLEDGE_BASE_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内共享的配置实例。"""
    return Settings()


settings = get_settings()
