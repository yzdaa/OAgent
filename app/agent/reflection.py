"""任务反思决策器：根据执行上下文决定下一步动作。"""
import json
import logging
from typing import Any, Dict, Literal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ErrorPolicy:
    """确定性错误策略，优先处理无需 LLM 的常见故障。"""
    RETRYABLE = ("timeout", "timed out", "connection", "network", "超时", "连接", "网络")

    def __init__(self, max_retries: int = 3):
        """设置确定性错误策略允许的最大恢复次数。"""
        self.max_retries = max_retries  # 超过此次数后直接终止，不再重试。

    def classify(self, failed_step: Dict[str, Any], retry_count: int = 0) -> "ReflectionDecision":
        """先按错误特征分类，判断是否可以直接重试或必须终止。"""
        if retry_count >= self.max_retries:
            return ReflectionDecision(action="abort", reason="已达到最大重试次数")
        error = str(failed_step.get("error", "")).lower()
        if any(marker in error for marker in self.RETRYABLE):
            return ReflectionDecision(action="retry", reason="检测到可恢复的临时错误")
        if "未找到工具" in error or "tool not found" in error:
            return ReflectionDecision(action="abort", reason="工具未注册，无法自动恢复")
        return ReflectionDecision(action="needs_reflection", reason="错误原因需要结合任务上下文判断")


class ReflectionDecision(BaseModel):
    """限制 LLM 只能返回受控的工作流动作。"""
    action: Literal["retry", "update_step", "replan", "abort", "needs_reflection"]  # 下一步动作。
    reason: str = Field(description="决策原因")  # 给日志和用户看的决策说明。
    updated_tool_arguments: Dict[str, Any] = Field(default_factory=dict)  # update_step 时覆盖的工具参数。


class Reflection:
    """统一的失败恢复模块：规则优先，LLM 反思兜底。"""
    def __init__(self, llm=None, max_retries: int = 3):
        """创建反思器；规则判断优先，复杂错误才调用 LLM。"""
        self.llm = llm  # 复杂错误的结构化决策模型。
        self.max_retries = max_retries  # 整个恢复流程的最大次数。
        self.error_policy = ErrorPolicy(max_retries)  # 无需 LLM 的确定性策略。

    def recover(self, context: Dict[str, Any], retry_count: int = 0) -> ReflectionDecision:
        """统一恢复入口，先执行 ErrorPolicy，再按需调用 LLM。"""
        """先执行确定性判断，只有复杂错误才交给 LLM。"""
        decision = self.error_policy.classify(context.get("failed_step", {}), retry_count)
        return self.decide(context, retry_count) if decision.action == "needs_reflection" else decision

    def decide(self, context: Dict[str, Any], retry_count: int = 0) -> ReflectionDecision:
        """让 LLM 在受控动作集合中选择恢复策略。"""
        if retry_count >= self.max_retries:
            return ReflectionDecision(action="abort", reason="已达到最大重试次数")
        if not self.llm:
            return ReflectionDecision(action="abort", reason="未配置 LLM，无法进行反思")
        # 只传递失败步骤和短摘要，避免把大段工具输出塞进反思上下文导致模型截断。
        compact_context = {
            "user_request": context.get("user_request", ""),
            "plan_steps": context.get("plan_steps", []),
            "failed_step": context.get("failed_step", {}),
            "step_results": context.get("step_results", [])[-3:],
        }
        prompt = f"""请分析企业办公任务的执行失败，并选择下一步动作：
retry=直接重试，update_step=修改当前步骤参数，replan=重新规划，abort=终止。
只能返回结构化决策，不要执行工具。
执行上下文：{json.dumps(compact_context, ensure_ascii=False, default=str)[:8000]}"""
        try:
            return self.llm.with_structured_output(ReflectionDecision).invoke(prompt)
        except Exception as exc:
            logger.error("反思决策失败: %s", exc)
            return ReflectionDecision(action="abort", reason=f"反思失败: {exc}")
