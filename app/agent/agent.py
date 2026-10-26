"""Agent 对外服务入口。"""
import logging
import os
import uuid
from typing import Optional

from langchain_openai import ChatOpenAI

from config.settings import settings
from .executor import TaskExecutor
from .graph import AgentGraph
from .memory_manager import MemoryManager
from .planner import TaskPlanner
from .reflection import Reflection
from .tool_registry import ToolRegistry
from app.observability import record_event

logger = logging.getLogger(__name__)


class OfficeAgent:
    """企业办公 Agent 的统一外部入口。"""

    def __init__(self, llm: Optional[ChatOpenAI] = None, enable_planner: bool = True):
        """组装 Agent 的基础组件和 LangGraph 工作流。"""
        # Agent 只负责组装组件，不在这里实现计划、执行或反思细节。
        self.llm = llm or self._create_llm()  # 统一的模型客户端；None 表示未配置模型。
        self.memory_manager = MemoryManager()  # 会话历史和用户偏好存储。
        self.tool_registry = ToolRegistry()  # Planner 与直接 Agent 共享的工具目录。
        # Reflection 负责 LangGraph 执行失败后的恢复决策。
        self.reflection = Reflection(self.llm)
        self.enable_planner = enable_planner and self.llm is not None  # 是否允许自动规划。
        self.graph = AgentGraph(
            TaskPlanner(self.llm),
            TaskExecutor(self.memory_manager, self.tool_registry, self.llm),
            self.reflection,
        )
        logger.info("OfficeAgent 初始化完成，Planner模式: %s", "开启" if self.enable_planner else "关闭")

    def _create_llm(self) -> Optional[ChatOpenAI]:
        """根据配置创建 LLM 客户端；缺少 API Key 时返回 None。"""
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
        """注册一个工具，供 Planner 和直接 Agent 使用。"""
        self.tool_registry.register(tool)

    def register_tools(self, tools):
        """批量注册工具；工具名由 ToolRegistry 管理。"""
        self.tool_registry.register_many(tools)

    def invoke(self, user_request: str, session_id: Optional[str] = None,
            user_id: Optional[str] = None, use_planner: Optional[bool] = None):
        """接收统一命名的用户请求，并交给 LangGraph 执行。"""
        session_id = session_id or str(uuid.uuid4())
        record_event("request.received", session_id=session_id, user_id=user_id,
                     user_request=user_request, use_planner=use_planner)
        if not self.llm:
            return {"output": "LLM未初始化，请配置 API Key 后重试。",
                    "step_results": [], "session_id": session_id}
        if use_planner is None:
            # 未显式指定时交给 Graph 根据请求内容判断；关闭 Planner 时才强制走直接 Agent。
            use_planner = None if self.enable_planner else False
        # 所有请求统一从 Graph 进入；Graph 内部只为复杂任务启用规划流程。
        # Agent 层只负责对外入口；具体路由、规划、执行和恢复由 Graph 负责。
        try:
            return self.graph.invoke(user_request, session_id, user_id, use_planner)
        except Exception as exc:
            record_event("request.failed", session_id=session_id, user_id=user_id,
                         user_request=user_request, error=str(exc))
            raise

    def clear_session(self, session_id: str):
        """清除指定会话的短期记忆和相关缓存。"""
        self.memory_manager.clear_session(session_id)
