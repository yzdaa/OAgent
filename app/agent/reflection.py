"""任务反思决策器：根据执行上下文决定下一步动作。"""
import json
import logging
from typing import Any, Dict, Literal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReflectionDecision(BaseModel):
    """限制 LLM 只能返回受控的工作流动作。"""
    action: Literal["retry", "update_step", "replan", "abort"]
    reason: str = Field(description="决策原因")
    updated_params: Dict[str, Any] = Field(default_factory=dict)


class Reflection:
    """把失败上下文转换为一个结构化流程决策。"""
    def __init__(self, llm=None, max_retries: int = 3):
        self.llm, self.max_retries = llm, max_retries

    def decide(self, context: Dict[str, Any], retry_count: int = 0) -> ReflectionDecision:
        if retry_count >= self.max_retries:
            return ReflectionDecision(action="abort", reason="已达到最大重试次数")
        if not self.llm:
            return ReflectionDecision(action="abort", reason="未配置 LLM，无法进行反思")
        prompt = f"""请分析企业办公任务的执行失败，并选择下一步动作：
retry=直接重试，update_step=修改当前步骤参数，replan=重新规划，abort=终止。
只能返回结构化决策，不要执行工具。
执行上下文：{json.dumps(context, ensure_ascii=False, default=str)}"""
        try:
            return self.llm.with_structured_output(ReflectionDecision).invoke(prompt)
        except Exception as exc:
            logger.error("反思决策失败: %s", exc)
            return ReflectionDecision(action="abort", reason=f"反思失败: {exc}")
