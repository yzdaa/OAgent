"""
这个目录下，
agent：核心目录
api：FastAPI接口层，不承载具体业务逻辑。处理HTTP 路由、请求参数校验、Agent调用、文件上传、返回响应、健康检查。
integrations：负责处理第三方组件相关操作，目前只有MySQL和Redis
knowledge：构建知识库。文档上传解析、切片、向量化。
tools：将业务能力包装成 LangChain Tool。包括数据分析、文档生成、OA/CRM 操作、知识库问答。
web：前端展示。管理页面布局、用户输入、会话状态、展示 Agent 输出、文件上传交互。

"""

from .bootstrap import create_office_agent

__all__ = ["create_office_agent"]
