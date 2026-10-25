"""
核心Agent封装
基于LangChain ReAct Agent，集成记忆、工具调用、反思功能
支持复杂任务的自动规划与多步骤执行
"""
import os
import uuid
from typing import Optional, List, Dict, Any
import logging

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config.settings import settings
from .memory_manager import MemoryManager
from .tool_registry import ToolRegistry
from .reflection import Reflection
from .planner import TaskPlanner

logger = logging.getLogger(__name__)


class OfficeAgent:
    REACT_TEMPLATE = """你是一个企业智能办公助手，能够使用各种工具帮助用户完成办公任务。

你可以使用的工具有：
{tools}

工具名称列表：{tool_names}

请严格按照以下格式回答：

Question: 用户的问题
Thought: 思考需要做什么
Action: 要使用的工具名称（必须是 [{tool_names}] 中的一个）
Action Input: 工具的输入参数（JSON格式）
Observation: 工具返回的结果
... (这个 Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 对用户的最终回复

开始！

历史对话：
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""

    def __init__(self, llm: Optional[ChatOpenAI] = None, enable_planner: bool = True):
        if llm:
            self.llm = llm
        else:
            self._init_llm()

        self.memory_manager = MemoryManager()
        self.tool_registry = ToolRegistry()
        self.reflection = Reflection(self.llm)
        self.agent_executor = None
        self.enable_planner = enable_planner and (self.llm is not None)
        if self.enable_planner:
            self.task_planner = TaskPlanner(self.llm)
        logger.info(f"OfficeAgent初始化完成，Planner模式: {'开启' if self.enable_planner else '关闭'}")

    def _init_llm(self):
        if settings.DASHSCOPE_API_KEY:
            os.environ["DASHSCOPE_API_KEY"] = settings.DASHSCOPE_API_KEY
            self.llm = ChatOpenAI(
                model=settings.LLM_MODEL_NAME,
                api_key=settings.DASHSCOPE_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            logger.info(f"LLM初始化成功，模型: {settings.LLM_MODEL_NAME}")
        else:
            logger.warning("未配置API Key，Agent将无法正常工作")
            self.llm = None

    def register_tool(self, tool):
        self.tool_registry.register(tool)

    def register_tools(self, tools: List):
        self.tool_registry.register_many(tools)

    def _create_agent(self, session_id: str):
        tools = self.tool_registry.list_tools()
        if not tools:
            raise ValueError("没有注册任何工具，请先注册工具")

        # LangChain 1.x 原生 Agent。状态由底层 LangGraph 管理，不再使用
        # AgentExecutor、ConversationBufferWindowMemory 或手写 ReAct prompt。
        history = self.memory_manager.get_messages(session_id)
        messages = []
        for msg in history[-settings.SHORT_TERM_MEMORY_MAX_LENGTH:]:
            messages.append(("human" if msg["role"] == "user" else "ai", msg["content"]))
        return create_agent(
            model=self.llm,
            tools=tools,
            system_prompt="你是一个企业智能办公助手，使用工具完成办公任务；回答要准确、清晰，并说明关键结果。",
        ), messages

    def _is_complex_task(self, user_input: str) -> bool:
        complex_keywords = ["分析", "报告", "生成", "图表", "可视化", "工作总结", "会议纪要", "审批"]
        return any(kw in user_input for kw in complex_keywords)

    def _execute_planned_task(self, user_input: str, session_id: str,
                               user_id: Optional[str] = None) -> Dict[str, Any]:
        """多步骤任务执行（使用规划器）"""
        available_tools = self.tool_registry.get_tool_names()
        task_steps = self.task_planner.plan(user_input, available_tools)
        if not task_steps:
            return self._execute_single_step(user_input, session_id, user_id)

        logger.info(f"任务规划成功，共 {len(task_steps)} 个步骤")
        all_steps = []
        final_output = ""

        for i, step in enumerate(task_steps):
            tool_name = step.get("tool")
            params = step.get("params", {})
            logger.info(f"执行步骤 {i+1}/{len(task_steps)}: {tool_name}")

            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                error_msg = f"错误：未找到工具 {tool_name}"
                all_steps.append({"step": i+1, "tool": tool_name, "error": error_msg})
                final_output += f"\n[步骤{i+1} 失败] {error_msg}"
                continue

            try:
                # 修复：使用标准调用 tool.invoke(params)
                observation = tool.invoke(params)
                all_steps.append({
                    "tool": tool_name,
                    "tool_input": params,
                    "observation": observation
                })
                final_output += f"\n[步骤{i+1} 完成] {tool_name}: {observation[:200]}"
                self.memory_manager.mysql.log_tool_call(
                    session_id=session_id,
                    user_id=user_id or "anonymous",
                    tool_name=tool_name,
                    params=params,
                    result=str(observation)[:1000]
                )
            except Exception as e:
                error_msg = f"执行出错: {str(e)}"
                logger.error(f"步骤 {i+1} 执行失败: {e}")
                all_steps.append({"step": i+1, "tool": tool_name, "error": error_msg})
                final_output += f"\n[步骤{i+1} 出错] {error_msg}"

        return {
            "output": final_output.strip() or "任务执行完成",
            "intermediate_steps": all_steps,
            "session_id": session_id,
            "planned": True,
            "steps_count": len(task_steps)
        }

    def _execute_single_step(self, user_input: str, session_id: str,
                             user_id: Optional[str] = None) -> Dict[str, Any]:
        """单步任务（ReAct Agent）"""
        try:
            executor, history = self._create_agent(session_id)
            user_context = ""
            if user_id:
                prefs = self.memory_manager.get_all_preferences(user_id)
                if prefs:
                    user_context = f"\n用户偏好信息：{prefs}"

            request = user_input + user_context
            result = executor.invoke({"messages": history + [("human", request)]})
            messages = result.get("messages", [])
            output = messages[-1].content if messages else "执行完成"
            self.memory_manager.add_message(session_id, "user", request)
            self.memory_manager.add_message(session_id, "assistant", output)
            steps = []
            for message in messages:
                if getattr(message, "type", None) == "tool":
                    self.memory_manager.mysql.log_tool_call(
                        session_id=session_id,
                        user_id=user_id or "anonymous",
                        tool_name=getattr(message, "name", "unknown"),
                        params=getattr(message, "additional_kwargs", {}),
                        result=str(message.content)[:1000]
                    )
            return {
                "output": output,
                "intermediate_steps": steps,
                "session_id": session_id,
                "planned": False
            }
        except Exception as e:
            logger.error(f"单步执行失败: {e}", exc_info=True)
            return {
                "output": f"执行失败: {str(e)}",
                "intermediate_steps": [],
                "session_id": session_id,
                "error": str(e)
            }

    def run(self, user_input: str, session_id: Optional[str] = None,
            user_id: Optional[str] = None, use_planner: bool = None) -> Dict[str, Any]:
        if session_id is None:
            session_id = str(uuid.uuid4())
        if not self.llm:
            return {
                "output": "LLM未初始化，请配置API Key后重试。",
                "intermediate_steps": [],
                "session_id": session_id
            }
        if use_planner is None:
            use_planner = self.enable_planner and self._is_complex_task(user_input)
        if use_planner:
            logger.info("使用任务规划器处理请求")
            return self._execute_planned_task(user_input, session_id, user_id)
        else:
            logger.info("使用ReAct Agent处理请求")
            return self._execute_single_step(user_input, session_id, user_id)

    def clear_session(self, session_id: str):
        self.memory_manager.clear_session(session_id)
