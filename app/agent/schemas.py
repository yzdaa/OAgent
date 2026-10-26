"""Agent 内部共享的数据协议，避免 Planner 和 Executor 各自约定字段。"""
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """一个可执行计划步骤。"""
    tool_name: str = Field(min_length=1)
    tool_arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult(BaseModel):
    """工具执行的标准结果；success 明确区分成功和业务错误文本。"""
    success: bool
    tool_name: str
    tool_arguments: Dict[str, Any] = Field(default_factory=dict)
    content: Optional[Any] = None
    error: Optional[str] = None
