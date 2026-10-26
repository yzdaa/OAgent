"""FastAPI 请求与响应模型。"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    user_request: str = Field(min_length=1, description="用户请求")
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class TaskResponse(BaseModel):
    output: str
    session_id: str
    step_results: List[Dict[str, Any]] = Field(default_factory=list)


class MemoryResponse(BaseModel):
    history: List[Dict[str, Any]]


class PreferenceRequest(BaseModel):
    key: str = Field(min_length=1)
    value: str


class PreferenceResponse(BaseModel):
    success: bool


class KnowledgeRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)


class ClearMemoryResponse(BaseModel):
    message: str


class ToolSummary(BaseModel):
    name: str
    description: str
