"""Agent 任务执行器。"""
from typing import Any, Dict, List, Optional
from langchain.agents import create_agent
from config.settings import settings
from app.observability import record_event
from .schemas import ToolExecutionResult


class TaskExecutor:
    """负责执行规划步骤和单次 LangChain Agent 请求。"""
    def __init__(self, memory_manager, tool_registry, llm):
        """保存记忆、工具注册表和模型依赖，不负责创建这些组件。"""
        self.memory_manager = memory_manager  # 提供历史消息和用户偏好。
        self.tool_registry = tool_registry  # 按 tool_name 查找可执行工具。
        self.llm = llm  # 直接 Agent 使用的模型客户端。

    def execute_plan(self, plan_steps: List[Dict[str, Any]], session_id: str,
                     user_id: Optional[str] = None) -> Dict[str, Any]:
        """按统一计划协议执行步骤，并返回逐步 tool_result。"""
        # plan_steps 是待执行步骤；step_results 是执行后的记录，二者不能混用。
        step_results, response_text = [], ""
        for index, step in enumerate(plan_steps, 1):
            # 每个计划步骤都通过注册表查找工具，避免 Executor 直接依赖具体工具实现。
            # tool_name 标识工具；tool_arguments 仅表示本次调用参数。
            tool_name = step.get("tool_name")
            tool_arguments = step.get("tool_arguments", {})
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                error = f"错误：未找到工具 {tool_name}"
                step_results.append({"step": index, "tool_name": tool_name, "tool_arguments": tool_arguments, "error": error})
                response_text += f"\n[步骤{index} 失败] {error}"
                continue
            try:
                # 工具统一使用 invoke，保证手工规划和 LangChain 工具调用遵循同一入口。
                tool_result = tool.invoke(tool_arguments)
                result_text = str(tool_result)
                # 现有工具暂时返回错误文本而非抛异常，这里集中识别，避免误记为成功。
                failed_text = result_text.startswith(("工具执行出错", "执行出错", "错误：", "文件不存在", "未知操作", "请先"))
                execution = ToolExecutionResult(
                    success=not failed_text, tool_name=tool_name,
                    tool_arguments=tool_arguments, content=tool_result,
                    error=result_text if failed_text else None,
                )
                record_event("tool.completed" if execution.success else "tool.failed",
                             session_id=session_id, user_id=user_id,
                             step=index, tool_name=tool_name, tool_arguments=tool_arguments,
                             tool_result=tool_result, success=execution.success)
                step_results.append({"step": index, "tool_name": tool_name, "tool_arguments": tool_arguments,
                                "tool_result": tool_result, "error": execution.error} if not execution.success else
                                {"step": index, "tool_name": tool_name, "tool_arguments": tool_arguments,
                                 "tool_result": tool_result})
                response_text += f"\n[步骤{index} 完成] {tool_name}: {str(tool_result)[:200]}"
                self.memory_manager.mysql.log_tool_call(
                    session_id=session_id, user_id=user_id or "anonymous",
                    tool_name=tool_name, params=tool_arguments, result=str(tool_result)[:1000])
            except Exception as exc:
                # 单步失败不会中断整个计划，先记录错误，后续由 Reflection 接入重试策略。
                error = f"执行出错: {exc}"
                step_results.append({"step": index, "tool_name": tool_name, "error": error})
                record_event("tool.failed", session_id=session_id, user_id=user_id,
                             step=index, tool_name=tool_name, tool_arguments=tool_arguments,
                             error=error)
                response_text += f"\n[步骤{index} 出错] {error}"
        return {"output": response_text.strip() or "任务执行完成", "step_results": step_results,
                "session_id": session_id, "planned": True, "steps_count": len(plan_steps)}

    def invoke_direct_agent(self, user_request: str, session_id: str,
                     user_id: Optional[str] = None) -> Dict[str, Any]:
        """让 LangChain Agent 自主决定是否调用工具，区别于显式计划执行。"""
        # 记忆层只负责提供历史消息；Agent 本身负责当前请求中的工具决策。
        history = self.memory_manager.get_messages(session_id)
        messages = [("human" if item["role"] == "user" else "ai", item["content"])
                    for item in history[-settings.SHORT_TERM_MEMORY_MAX_LENGTH:]]
        if user_id:
            preferences = self.memory_manager.get_all_preferences(user_id)
            if preferences:
                user_request += f"\n用户偏好信息：{preferences}"
        agent = create_agent(
            model=self.llm, tools=self.tool_registry.list_tools(),
            system_prompt="你是一个企业智能办公助手，使用工具完成办公任务；回答要准确、清晰，并说明关键结果。")
        # LangChain 返回的是包含 messages 的状态对象，不直接称为最终 response。
        model_state = agent.invoke({"messages": messages + [("human", user_request)]})
        # LangChain 1.x 返回消息状态列表，最后一条 AI 消息就是本次对话结果。
        output = model_state.get("messages", [])[-1].content if model_state.get("messages") else "执行完成"
        # 记录 LangChain Agent 内部的工具消息，保证日志能还原直接 Agent 的完整路径。
        for message in model_state.get("messages", []):
            if getattr(message, "type", None) == "tool":
                record_event("tool.completed", session_id=session_id, user_id=user_id,
                             tool_name=getattr(message, "name", "unknown"),
                             tool_result=str(message.content))
        record_event("llm.completed", session_id=session_id, user_id=user_id,
                     user_request=user_request, llm_answer=output)
        self.memory_manager.add_message(session_id, "user", user_request)
        self.memory_manager.add_message(session_id, "assistant", output)
        return {"output": output, "step_results": [], "session_id": session_id, "planned": False}
