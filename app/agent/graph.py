"""Agent 流程编排。"""
from typing import Any, Dict, Optional


class AgentWorkflow:
    """决定请求进入计划执行还是单步 Agent。"""
    COMPLEX_KEYWORDS = ("分析", "报告", "生成", "图表", "可视化", "工作总结", "会议纪要", "审批")

    def __init__(self, planner, executor, reflection=None):
        self.planner, self.executor, self.reflection = planner, executor, reflection

    def run(self, user_input: str, session_id: str, user_id: Optional[str] = None,
            use_planner: Optional[bool] = None) -> Dict[str, Any]:
        # 未显式指定时，仅将明显的复合任务交给 Planner，普通问答直接进入 Agent。
        # 这样可以避免简单问题先经过一次额外的规划调用，降低延迟和模型消耗。
        use_planner = any(k in user_input for k in self.COMPLEX_KEYWORDS) if use_planner is None else use_planner
        if use_planner:
            # Planner 只生成步骤，不直接执行工具；执行权统一交给 TaskExecutor。
            steps = self.planner.plan(user_input, self.executor.tool_registry.get_tool_names())
            if steps:
                result = self.executor.execute_plan(steps, session_id, user_id)
                failed = [s for s in result["intermediate_steps"] if "error" in s]
                if failed and self.reflection:
                    decision = self.reflection.decide({
                        "user_input": user_input,
                        "plan": steps,
                        "failed_steps": failed,
                        "results": result["intermediate_steps"],
                    })
                    result["reflection"] = decision.model_dump()
                    if decision.action == "retry":
                        retry_step = {"tool": failed[0]["tool"], "params": failed[0].get("tool_input", {})}
                        result["retry_result"] = self.executor.execute_plan(
                            [retry_step], session_id, user_id)
                    elif decision.action == "update_step" and decision.updated_params:
                        retry_step = {"tool": failed[0]["tool"], "params": decision.updated_params}
                        result["retry_result"] = self.executor.execute_plan(
                            [retry_step], session_id, user_id)
                    elif decision.action == "replan":
                        revised = self.planner.plan(
                            f"{user_input}\n请根据失败原因调整计划：{decision.reason}",
                            self.executor.tool_registry.get_tool_names(),
                        )
                        if revised:
                            result["replanned_result"] = self.executor.execute_plan(
                                revised, session_id, user_id)
                return result
        # 计划为空或任务不复杂时，使用 LangChain Agent 自主决定工具调用。
        return self.executor.execute_agent(user_input, session_id, user_id)
