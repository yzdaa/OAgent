"""基于 LangGraph 的 Agent 工作流。"""

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict, total=False):
    """工作流中各节点共享的最小状态。"""

    user_input: str
    session_id: str
    user_id: Optional[str]
    use_planner: bool
    plan: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    failed_step: Dict[str, Any]
    retry_count: int
    decision: Dict[str, Any]
    output: str
    response: Dict[str, Any]


class AgentWorkflow:
    """使用 StateGraph 编排计划、执行和恢复流程。"""

    COMPLEX_KEYWORDS = ("分析", "报告", "生成", "图表", "可视化", "工作总结", "会议纪要", "审批")

    def __init__(self, planner, executor, reflection=None):
        self.planner = planner
        self.executor = executor
        self.reflection = reflection
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("route", self._route)
        workflow.add_node("plan", self._plan)
        workflow.add_node("execute_plan", self._execute_plan)
        workflow.add_node("execute_agent", self._execute_agent)
        workflow.add_node("recover", self._recover)
        workflow.add_node("retry", self._retry)
        workflow.add_node("replan", self._replan)

        workflow.add_edge(START, "route")
        workflow.add_conditional_edges(
            "route", self._route_next,
            {"plan": "plan", "agent": "execute_agent"},
        )
        workflow.add_edge("plan", "execute_plan")
        workflow.add_conditional_edges(
            "execute_plan", self._after_execute,
            {"recover": "recover", "end": END},
        )
        workflow.add_conditional_edges(
            "recover", self._after_recover,
            {"retry": "retry", "replan": "replan", "end": END},
        )
        workflow.add_conditional_edges(
            "retry", self._after_execute,
            {"recover": "recover", "end": END},
        )
        workflow.add_edge("replan", "execute_plan")
        workflow.add_edge("execute_agent", END)
        return workflow.compile()

    def _route(self, state: AgentState) -> AgentState:
        if "use_planner" not in state:
            state["use_planner"] = any(
                keyword in state["user_input"] for keyword in self.COMPLEX_KEYWORDS
            )
        return state

    @staticmethod
    def _route_next(state: AgentState) -> str:
        return "plan" if state.get("use_planner") else "agent"

    def _plan(self, state: AgentState) -> AgentState:
        state["plan"] = self.planner.plan(
            state["user_input"], self.executor.tool_registry.get_tool_names()
        )
        return state

    def _execute_plan(self, state: AgentState) -> AgentState:
        # 每次执行新计划前清除上一次失败标记，避免成功重规划后再次进入恢复节点。
        state.pop("failed_step", None)
        result = self.executor.execute_plan(
            state.get("plan", []), state["session_id"], state.get("user_id")
        )
        state["response"] = result
        state["results"] = result.get("intermediate_steps", [])
        failed = [item for item in state["results"] if "error" in item]
        if failed:
            state["failed_step"] = failed[0]
        return state

    def _execute_agent(self, state: AgentState) -> AgentState:
        state["response"] = self.executor.execute_agent(
            state["user_input"], state["session_id"], state.get("user_id")
        )
        return state

    def _recover(self, state: AgentState) -> AgentState:
        if not self.reflection:
            state["decision"] = {"action": "abort", "reason": "未配置恢复策略"}
            return state
        decision = self.reflection.recover(
            {
                "user_input": state["user_input"],
                "plan": state.get("plan", []),
                "failed_step": state.get("failed_step", {}),
                "results": state.get("results", []),
            },
            state.get("retry_count", 0),
        )
        state["decision"] = decision.model_dump()
        return state

    @staticmethod
    def _after_execute(state: AgentState) -> str:
        return "recover" if state.get("failed_step") else "end"

    @staticmethod
    def _after_recover(state: AgentState) -> str:
        action = state.get("decision", {}).get("action")
        if action == "retry":
            return "retry"
        if action in {"update_step", "replan"}:
            return "replan"
        return "end"

    def _retry(self, state: AgentState) -> AgentState:
        state["retry_count"] = state.get("retry_count", 0) + 1
        failed = state.get("failed_step", {})
        state["plan"] = [{"tool": failed.get("tool"), "params": failed.get("tool_input", {})}]
        return self._execute_plan(state)

    def _replan(self, state: AgentState) -> AgentState:
        reason = state.get("decision", {}).get("reason", "执行失败")
        state["retry_count"] = state.get("retry_count", 0) + 1
        state["plan"] = self.planner.plan(
            f"{state['user_input']}\n请根据失败原因调整计划：{reason}",
            self.executor.tool_registry.get_tool_names(),
        )
        return state

    def run(self, user_input: str, session_id: str, user_id: Optional[str] = None,
            use_planner: Optional[bool] = None) -> Dict[str, Any]:
        initial: AgentState = {
            "user_input": user_input,
            "session_id": session_id,
            "user_id": user_id,
            "retry_count": 0,
        }
        if use_planner is not None:
            initial["use_planner"] = use_planner
        final_state = self.graph.invoke(initial)
        response = final_state.get("response", {})
        if final_state.get("decision"):
            response["recovery"] = final_state["decision"]
        return response
