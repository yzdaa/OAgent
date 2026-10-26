"""基于 LangGraph 的 Agent 工作流。"""

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from app.observability import record_event


class AgentState(TypedDict, total=False):
    """工作流中各节点共享的最小状态。"""

    # user_request 是用户的原始任务描述，不再使用容易与问题查询混淆的 user_input。
    user_request: str
    session_id: str
    user_id: Optional[str]
    use_planner: bool
    # plan_steps 是 Planner 生成的完整步骤列表；step_results 是 Executor 的逐步执行结果。
    plan_steps: List[Dict[str, Any]]
    step_results: List[Dict[str, Any]]
    failed_step: Dict[str, Any]
    retry_count: int
    decision: Dict[str, Any]
    # agent_response 是最终结构化响应，output 是其中面向用户展示的文本字段。
    agent_response: Dict[str, Any]


class AgentGraph:
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
        workflow.add_node("create_plan", self._create_plan)
        workflow.add_node("execute_plan", self._execute_plan)
        workflow.add_node("invoke_direct_agent", self._invoke_direct_agent)
        workflow.add_node("recover", self._recover)
        workflow.add_node("retry", self._retry)
        workflow.add_node("replan", self._replan)

        workflow.add_edge(START, "route")
        workflow.add_conditional_edges(
            "route", self._route_next,
            {"plan": "create_plan", "agent": "invoke_direct_agent"},
        )
        workflow.add_edge("create_plan", "execute_plan")
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
        workflow.add_edge("invoke_direct_agent", END)
        return workflow.compile()

    def _route(self, state: AgentState) -> AgentState:
        if "use_planner" not in state:
            state["use_planner"] = any(
                keyword in state["user_request"] for keyword in self.COMPLEX_KEYWORDS
            )
        record_event("graph.route", session_id=state.get("session_id"), user_id=state.get("user_id"),
                     use_planner=state.get("use_planner"), user_request=state.get("user_request"))
        return state

    @staticmethod
    def _route_next(state: AgentState) -> str:
        return "plan" if state.get("use_planner") else "agent"

    def _create_plan(self, state: AgentState) -> AgentState:
        # create_plan 只生成计划，不执行工具；执行职责属于 execute_plan 节点。
        state["plan_steps"] = self.planner.plan(
            state["user_request"], self.executor.tool_registry.get_tool_names()
        )
        record_event("graph.plan_created", session_id=state["session_id"], user_id=state.get("user_id"),
                     plan_steps=state["plan_steps"])
        return state

    def _execute_plan(self, state: AgentState) -> AgentState:
        # Executor 使用统一的 tool_name/tool_arguments 协议逐步调用工具。
        # 每次执行新计划前清除上一次失败标记，避免成功重规划后再次进入恢复节点。
        state.pop("failed_step", None)
        execution_result = self.executor.execute_plan(
            state.get("plan_steps", []), state["session_id"], state.get("user_id")
        )
        state["agent_response"] = execution_result
        state["step_results"] = execution_result.get("step_results", [])
        failed = [item for item in state["step_results"] if "error" in item]
        if failed:
            state["failed_step"] = failed[0]
        record_event("graph.plan_executed", session_id=state["session_id"], user_id=state.get("user_id"),
                     step_results=state["step_results"], failed_step=state.get("failed_step"))
        return state

    def _invoke_direct_agent(self, state: AgentState) -> AgentState:
        # 简单请求走直接 Agent；复杂请求走上面的显式规划路径。
        state["agent_response"] = self.executor.invoke_direct_agent(
            state["user_request"], state["session_id"], state.get("user_id")
        )
        record_event("graph.direct_agent_completed", session_id=state["session_id"], user_id=state.get("user_id"),
                     llm_answer=state["agent_response"].get("output", ""))
        return state

    def _recover(self, state: AgentState) -> AgentState:
        if not self.reflection:
            state["decision"] = {"action": "abort", "reason": "未配置恢复策略"}
            return state
        decision = self.reflection.recover(
            {
                "user_request": state["user_request"],
                "plan_steps": state.get("plan_steps", []),
                "failed_step": state.get("failed_step", {}),
                "step_results": state.get("step_results", []),
            },
            state.get("retry_count", 0),
        )
        state["decision"] = decision.model_dump()
        record_event("graph.recovery_decided", session_id=state["session_id"], user_id=state.get("user_id"),
                     decision=state["decision"])
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
        state["plan_steps"] = [{"tool_name": failed.get("tool_name"), "tool_arguments": failed.get("tool_arguments", {})}]
        record_event("graph.retry", session_id=state["session_id"], user_id=state.get("user_id"),
                     retry_count=state["retry_count"], plan_steps=state["plan_steps"])
        return self._execute_plan(state)

    def _replan(self, state: AgentState) -> AgentState:
        reason = state.get("decision", {}).get("reason", "执行失败")
        state["retry_count"] = state.get("retry_count", 0) + 1
        state["plan_steps"] = self.planner.plan(
            f"{state['user_request']}\n请根据失败原因调整计划：{reason}",
            self.executor.tool_registry.get_tool_names(),
        )
        record_event("graph.replanned", session_id=state["session_id"], user_id=state.get("user_id"),
                     retry_count=state["retry_count"], plan_steps=state["plan_steps"])
        return state

    def invoke(self, user_request: str, session_id: str, user_id: Optional[str] = None,
            use_planner: Optional[bool] = None) -> Dict[str, Any]:
        initial: AgentState = {
            "user_request": user_request,
            "session_id": session_id,
            "user_id": user_id,
            "retry_count": 0,
        }
        if use_planner is not None:
            initial["use_planner"] = use_planner
        final_state = self.graph.invoke(initial)
        # response 是对外返回的 Agent 结果；agent_response 是 Graph 内部状态字段。
        response = final_state.get("agent_response", {})
        if final_state.get("decision"):
            response["recovery"] = final_state["decision"]
        record_event("graph.completed", session_id=session_id, user_id=user_id,
                     user_request=user_request, llm_answer=response.get("output", ""),
                     use_planner=final_state.get("use_planner"),
                     step_results=final_state.get("step_results", []))
        return response
