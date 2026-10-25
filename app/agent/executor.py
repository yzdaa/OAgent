"""Agent 任务执行器。"""
from typing import Any, Dict, List, Optional
from langchain.agents import create_agent
from config.settings import settings


class TaskExecutor:
    """负责执行规划步骤和单次 LangChain Agent 请求。"""
    def __init__(self, memory_manager, tool_registry, llm):
        self.memory_manager, self.tool_registry, self.llm = memory_manager, tool_registry, llm

    def execute_plan(self, steps: List[Dict[str, Any]], session_id: str,
                     user_id: Optional[str] = None) -> Dict[str, Any]:
        results, output = [], ""
        for index, step in enumerate(steps, 1):
            # 每个计划步骤都通过注册表查找工具，避免 Executor 直接依赖具体工具实现。
            name, params = step.get("tool"), step.get("params", {})
            tool = self.tool_registry.get_tool(name)
            if not tool:
                error = f"错误：未找到工具 {name}"
                results.append({"step": index, "tool": name, "tool_input": params, "error": error})
                output += f"\n[步骤{index} 失败] {error}"
                continue
            try:
                # 工具统一使用 invoke，保证手工规划和 LangChain 工具调用遵循同一入口。
                observation = tool.invoke(params)
                results.append({"step": index, "tool": name, "tool_input": params,
                                "observation": observation})
                output += f"\n[步骤{index} 完成] {name}: {str(observation)[:200]}"
                self.memory_manager.mysql.log_tool_call(
                    session_id=session_id, user_id=user_id or "anonymous",
                    tool_name=name, params=params, result=str(observation)[:1000])
            except Exception as exc:
                # 单步失败不会中断整个计划，先记录错误，后续由 Reflection 接入重试策略。
                error = f"执行出错: {exc}"
                results.append({"step": index, "tool": name, "error": error})
                output += f"\n[步骤{index} 出错] {error}"
        return {"output": output.strip() or "任务执行完成", "intermediate_steps": results,
                "session_id": session_id, "planned": True, "steps_count": len(steps)}

    def invoke_agent(self, user_input: str, session_id: str,
                     user_id: Optional[str] = None) -> Dict[str, Any]:
        # 记忆层只负责提供历史消息；Agent 本身负责当前请求中的工具决策。
        history = self.memory_manager.get_messages(session_id)
        messages = [("human" if item["role"] == "user" else "ai", item["content"])
                    for item in history[-settings.SHORT_TERM_MEMORY_MAX_LENGTH:]]
        if user_id:
            preferences = self.memory_manager.get_all_preferences(user_id)
            if preferences:
                user_input += f"\n用户偏好信息：{preferences}"
        agent = create_agent(
            model=self.llm, tools=self.tool_registry.list_tools(),
            system_prompt="你是一个企业智能办公助手，使用工具完成办公任务；回答要准确、清晰，并说明关键结果。")
        result = agent.invoke({"messages": messages + [("human", user_input)]})
        # LangChain 1.x 返回消息状态列表，最后一条 AI 消息就是本次对话结果。
        output = result.get("messages", [])[-1].content if result.get("messages") else "执行完成"
        self.memory_manager.add_message(session_id, "user", user_input)
        self.memory_manager.add_message(session_id, "assistant", output)
        return {"output": output, "intermediate_steps": [], "session_id": session_id, "planned": False}
