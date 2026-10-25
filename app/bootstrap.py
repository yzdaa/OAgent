"""统一组装应用组件，避免各个入口重复注册工具。"""

from app.agent import OfficeAgent
from app.tools import (
    APITool,
    DataAnalyzerTool,
    DocumentGeneratorTool,
    KnowledgeQATool,
)
from config.settings import settings


def create_office_agent() -> OfficeAgent:
    """创建配置完整的办公 Agent。"""
    settings.ensure_runtime_dirs()
    agent = OfficeAgent()
    agent.register_tools(
        [
            DataAnalyzerTool(),
            DocumentGeneratorTool(),
            APITool(),
            KnowledgeQATool(),
        ]
    )
    return agent
