"""FastAPI HTTP 接口：路由负责协议转换，业务逻辑由应用组件实现。"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from app import create_office_agent
from app.agent import OfficeAgent
from app.knowledge import UniversalDocumentLoader, get_vector_store
from app.tools import KnowledgeQATool
from config.settings import settings
from .dependencies import get_agent
from .schemas import (
    ClearMemoryResponse, KnowledgeRequest, MemoryResponse, PreferenceRequest,
    PreferenceResponse, TaskRequest, TaskResponse, ToolSummary,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时创建共享 Agent，避免模块导入阶段初始化外部资源。"""
    settings.ensure_runtime_dirs()
    app.state.agent = create_office_agent()
    logger.info("FastAPI Agent 初始化完成")
    yield
    app.state.agent = None


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="企业智能办公助手 API，支持任务规划、工具调用和知识问答",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """健康检查，不触发模型调用。"""
    return {"status": "healthy", "service": settings.APP_NAME}


@app.post("/api/chat", response_model=TaskResponse)
def chat(request: TaskRequest, agent: OfficeAgent = Depends(get_agent)) -> TaskResponse:
    """处理对话请求；同步 Agent 使用同步路由，避免阻塞异步事件循环。"""
    try:
        result = agent.invoke(request.user_request, request.session_id, request.user_id)
        return TaskResponse(output=result.get("output", ""), session_id=result["session_id"],
                            step_results=result.get("step_results", []))
    except Exception as exc:
        logger.error("处理对话请求失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="对话处理失败，请查看服务日志") from exc


@app.get("/api/memory/{session_id}", response_model=MemoryResponse)
def get_memory(session_id: str, agent: OfficeAgent = Depends(get_agent)) -> MemoryResponse:
    return MemoryResponse(history=agent.memory_manager.get_messages(session_id))


@app.post("/api/memory/{session_id}/clear", response_model=ClearMemoryResponse)
def clear_memory(session_id: str, agent: OfficeAgent = Depends(get_agent)) -> ClearMemoryResponse:
    agent.clear_session(session_id)
    return ClearMemoryResponse(message=f"会话 {session_id} 记忆已清除")


@app.get("/api/user/{user_id}/preferences")
def get_user_preferences(user_id: str, agent: OfficeAgent = Depends(get_agent)) -> Dict[str, Any]:
    return {"preferences": agent.memory_manager.get_all_preferences(user_id)}


@app.post("/api/user/{user_id}/preferences", response_model=PreferenceResponse)
def set_user_preference(user_id: str, request: PreferenceRequest,
                        agent: OfficeAgent = Depends(get_agent)) -> PreferenceResponse:
    success = agent.memory_manager.set_preference(user_id, request.key, request.value)
    return PreferenceResponse(success=success)


@app.post("/api/knowledge/qa")
def knowledge_qa(request: KnowledgeRequest) -> Dict[str, str]:
    """独立知识问答接口，不经过 Agent 路由。"""
    answer = KnowledgeQATool().invoke({"question": request.question, "top_k": request.top_k})
    return {"answer": answer}


@app.post("/api/knowledge/upload")
def upload_document(file: UploadFile = File(...)) -> Dict[str, str]:
    """上传并建立索引，只使用安全文件名防止路径穿越。"""
    safe_name = Path(file.filename or "").name
    allowed_extensions = {".pdf", ".docx", ".txt", ".xlsx", ".xls", ".csv"}
    if not safe_name or Path(safe_name).suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=400, detail="不支持的文件类型")
    file_path = settings.KNOWLEDGE_BASE_DIR / safe_name
    try:
        with file_path.open("wb") as output_file:
            output_file.write(file.file.read())
        documents = UniversalDocumentLoader().load_file(str(file_path))
        if not documents:
            return {"message": "文档解析失败或无内容"}
        if not get_vector_store().add_documents(documents):
            raise HTTPException(status_code=500, detail="知识库索引建立失败")
        return {"message": f"成功添加文档 {safe_name}，共 {len(documents)} 个片段"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("知识库上传失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="文档上传处理失败") from exc


@app.get("/api/tools", response_model=list[ToolSummary])
def list_tools(agent: OfficeAgent = Depends(get_agent)) -> list[ToolSummary]:
    return [ToolSummary(name=tool.name, description=tool.description)
            for tool in agent.tool_registry.list_tools()]


@app.post("/api/callback/oa/approval")
async def oa_approval_callback(data: Dict[str, Any]) -> Dict[str, str]:
    logger.info("收到 OA 审批回调: %s", data)
    return {"status": "received"}
