"""Agent 执行追踪日志。

每个事件都使用 session_id 串联，写入 JSONL，便于按会话还原完整处理路径。
日志属于运行时数据，默认位于 data/logs/agent_events.jsonl，不提交到仓库。
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)
_write_lock = Lock()


def record_event(event: str, session_id: Optional[str] = None,
                 user_id: Optional[str] = None, **details: Any) -> None:
    """记录一次可检索的 Agent 生命周期事件。"""
    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "session_id": session_id,
        "user_id": user_id,
        **details,
    }
    log_path = PROJECT_ROOT / "data" / "logs" / "agent_events.jsonl"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        # 追踪日志不能影响正常问答，只记录日志写入失败。
        logger.warning("Agent 追踪日志写入失败: %s", exc)

