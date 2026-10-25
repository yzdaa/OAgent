"""确定性错误策略：优先处理无需 LLM 的常见故障。"""

from typing import Any, Dict, Literal

from pydantic import BaseModel


class ErrorDecision(BaseModel):
    action: Literal["retry", "needs_reflection", "abort"]
    reason: str


class ErrorPolicy:
    """将明确的执行错误转换为确定性决策。"""

    RETRYABLE = ("timeout", "timed out", "connection", "network", "超时", "连接", "网络")

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def classify(self, failed_step: Dict[str, Any], retry_count: int = 0) -> ErrorDecision:
        if retry_count >= self.max_retries:
            return ErrorDecision(action="abort", reason="已达到最大重试次数")

        error = str(failed_step.get("error", "")).lower()
        if any(marker in error for marker in self.RETRYABLE):
            return ErrorDecision(action="retry", reason="检测到可恢复的临时错误")
        if "未找到工具" in error or "tool not found" in error:
            return ErrorDecision(action="abort", reason="工具未注册，无法自动恢复")
        return ErrorDecision(action="needs_reflection", reason="错误原因需要结合任务上下文判断")
