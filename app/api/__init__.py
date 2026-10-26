"""
FastAPI接口层，不承载具体业务逻辑。处理HTTP 路由、请求参数校验、Agent调用、文件上传、返回响应、健康检查。
dependencies：共享agent，避免路由重复初始化
fastapi_app：路由不同来源的HTTP请求、处理请求参数、调用Agent、返回响应、健康检查
schemas：请求与响应模型配置
"""