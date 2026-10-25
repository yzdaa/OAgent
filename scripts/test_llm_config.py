"""诊断项目当前 LLM 配置和 API 访问状态。

运行：
    conda activate OAgentbz
    python scripts/test_llm_config.py

脚本不会打印完整 API Key，也不会修改项目配置。
"""

import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from langchain_openai import ChatOpenAI

# 支持从项目根目录直接执行：python scripts/test_llm_config.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402


def mask_key(value: str | None) -> str:
    if not value:
        return "未配置"
    if len(value) <= 10:
        return "已配置（长度过短，未显示）"
    return f"{value[:6]}...{value[-4:]}"


def check_configuration() -> tuple[str, str, str]:
    print("[1/4] 检查项目实际读取的配置")
    base_url = str(settings.LLM_BASE_URL).strip().rstrip("/")
    model = str(settings.LLM_MODEL_NAME).strip()
    api_key = settings.DASHSCOPE_API_KEY

    print(f"  LLM_BASE_URL: {base_url}")
    print(f"  LLM_MODEL_NAME: {model}")
    print(f"  DASHSCOPE_API_KEY: {mask_key(api_key)}")

    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置，请检查项目根目录 .env")
    if not model:
        raise RuntimeError("LLM_MODEL_NAME 为空")

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"LLM_BASE_URL 格式无效: {base_url}")
    if "/v1" not in parsed.path:
        print("  警告：URL 路径中没有 /v1，请确认是否为 OpenAI 兼容接口地址")

    print("  配置格式：通过")
    return base_url, model, api_key


def check_network(base_url: str) -> None:
    print("[2/4] 检查地址和网络连通性")
    hostname = urlparse(base_url).hostname
    if not hostname:
        raise RuntimeError("无法从 LLM_BASE_URL 解析主机名")
    try:
        addresses = socket.getaddrinfo(hostname, 443)
        print(f"  DNS：通过（解析到 {len(addresses)} 个地址）")
    except OSError as exc:
        raise RuntimeError(f"DNS 解析失败: {exc}") from exc

    try:
        response = httpx.get(f"{base_url}/models", timeout=10)
        print(f"  HTTP：已收到响应（状态码 {response.status_code}）")
        if response.status_code in {401, 403}:
            print("  提示：/models 被拒绝不一定表示配置错误，部分服务商不开放该接口。")
    except httpx.HTTPError as exc:
        raise RuntimeError(f"HTTP 连接失败: {exc}") from exc


def check_model_call(base_url: str, model: str, api_key: str) -> None:
    print("[3/4] 发起最小模型调用")
    try:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            max_tokens=16,
            timeout=20,
            max_retries=0,
        )
        response = llm.invoke("只回复：OK")
        print(f"  模型调用：通过（返回内容：{str(response.content)[:80]}）")
    except Exception as exc:
        message = str(exc)
        print(f"  模型调用：失败")
        print(f"  错误类型：{type(exc).__name__}")
        print(f"  错误信息：{message[:1000]}")
        if "AllocationQuota.FreeTierOnly" in message or "free quota exhausted" in message.lower():
            print("  诊断结论：API 地址和请求已到达服务商，但账号免费额度已耗尽或被限制为仅免费额度。")
        elif "401" in message or "invalid" in message.lower() and "key" in message.lower():
            print("  诊断结论：API Key 无效、过期，或与当前 API 地址/区域不匹配。")
        elif "404" in message:
            print("  诊断结论：模型地址、接口路径或模型名称可能不正确。")
        raise


def main() -> int:
    print("=" * 60)
    print("OAgentbz LLM 配置诊断")
    print("=" * 60)
    try:
        base_url, model, api_key = check_configuration()
        check_network(base_url)
        check_model_call(base_url, model, api_key)
        print("[4/4] 诊断结果：LLM 配置和访问正常")
        return 0
    except Exception:
        print("[4/4] 诊断结果：失败，请根据上面的具体结论修正 .env")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
