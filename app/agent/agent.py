"""Agent 对外服务入口。"""
import logging
import os
import uuid
from typing import Optional

from langchain_openai import ChatOpenAI

from config.settings import settings
from .executor import TaskExecutor
from .graph import AgentWorkflow
from .memory_manager import MemoryManager
from .planner import TaskPlanner
from .reflection import Reflection
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class OfficeAgent:
    """企业办公 Agent 的统一外部入口。"""

    def __init__(self, llm: Optional[ChatOpenAI] = None, enable_planner: bool = True):
        # Agent 只负责组装组件，不在这里实现计划、执行或反思细节。
        self.llm = llm or self._create_llm()
        self.memory_manager = MemoryManager()
        self.tool_registry = ToolRegistry()
        # Reflection 当前保留为扩展点，待接入 graph 的失败分支和重试流程。
        self.reflection = Reflection(self.llm)
        self.enable_planner = enable_planner and self.llm is not None
        self.workflow = AgentWorkflow(
            TaskPlanner(self.llm),
            TaskExecutor(self.memory_manager, self.tool_registry, self.llm),
            self.reflection,
        )
        logger.info("OfficeAgent 初始化完成，Planner模式: %s", "开启" if self.enable_planner else "关闭")

    def _create_llm(self) -> Optional[ChatOpenAI]:
        if not settings.llm_configured:
            logger.warning("未配置 API Key，Agent 将无法正常工作")
            return None
        os.environ["DASHSCOPE_API_KEY"] = settings.DASHSCOPE_API_KEY
        return ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

    def register_tool(self, tool):
        self.tool_registry.register(tool)

    def register_tools(self, tools):
        self.tool_registry.register_many(tools)

    def run(self, user_input: str, session_id: Optional[str] = None,
            user_id: Optional[str] = None, use_planner: Optional[bool] = None):
        session_id = session_id or str(uuid.uuid4())
        if not self.llm:
            return {"output": "LLM未初始化，请配置 API Key 后重试。",
                    "intermediate_steps": [], "session_id": session_id}
        if use_planner is None:
            use_planner = self.enable_planner
        # 所有请求统一从 Workflow 进入，避免入口层绕过 Planner 或 Executor。
        return self.workflow.run(user_input, session_id, user_id, use_planner)

    def clear_session(self, session_id: str):
        self.memory_manager.clear_session(session_id)
