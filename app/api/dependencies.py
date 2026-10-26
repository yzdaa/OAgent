"""API 依赖：从应用生命周期中读取共享 Agent。"""
from fastapi import Request
from app.agent import OfficeAgent


def get_agent(request: Request) -> OfficeAgent:
    """获取启动阶段创建的共享 Agent，避免路由重复初始化。"""
    return request.app.state.agent
