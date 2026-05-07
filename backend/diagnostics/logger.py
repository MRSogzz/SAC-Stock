"""
diagnostics/logger.py
======================
Log 機制：
  - DebugLogger：每次實例化建立新的獨立 log 檔（深度診斷用）
  - _registry_log：Append 到固定的 diagnostics.log（Registry 展示工具用）
  - new_logger()：建立 DebugLogger 的工廠函數
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path


# ── 路徑設定（延遲 import configs 避免循環依賴）────────────────────────────────

def _get_debug_log_dir() -> str:
    try:
        from configs.base_config import HISTORY_DIR
        return os.path.join(HISTORY_DIR, "debug_logs")
    except ImportError:
        return "storage/history/debug_logs"


# ═══════════════════════════════════════════════════════════════════════════════
# DebugLogger：每次新建獨立 log 檔（深度診斷用）
# ═══════════════════════════════════════════════════════════════════════════════

class DebugLogger:
    """
    每次實例化時建立一個新的 log 檔案。
    同時輸出到 stdout（讓 server 仍能看到訊息）和 log 檔案。

    格式：storage/history/debug_logs/debug_TAG_YYYY-MM-DD_HH-MM-SS.log

    用法：
        with new_logger(tag="runA_w1") as logger:
            logger.log("訓練開始")
        # 離開 with 區塊時自動 close()
    """

    def __init__(self, tag: str = ""):
        log_dir = _get_debug_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix   = f"_{tag}" if tag else ""
        filename = f"debug{suffix}_{ts}.log"
        self.path = os.path.join(log_dir, filename)
        self._f   = open(self.path, "w", encoding="utf-8", buffering=1)
        self._write_header(ts, tag)

    def _write_header(self, ts: str, tag: str):
        self.log("=" * 70)
        self.log(f"  偵錯 Log  {ts}  {tag}")
        self.log("=" * 70)

    def log(self, msg: str = ""):
        """同時寫入 log 檔案和 stdout。"""
        print(msg)
        self._f.write(msg + "\n")
        self._f.flush()

    def close(self):
        self.log("\n" + "=" * 70)
        self.log(f"  Log 已儲存：{self.path}")
        self.log("=" * 70)
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def new_logger(tag: str = "") -> DebugLogger:
    """建立新的 DebugLogger，每次呼叫產生新的 log 檔案。"""
    return DebugLogger(tag=tag)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry 固定 log（inspector.py 用）
# ═══════════════════════════════════════════════════════════════════════════════

_REGISTRY_LOG_PATH = Path("diagnostics/diagnostics.log")


def registry_log(message: str) -> None:
    """Append 一行訊息到固定 log 檔（含時間戳）。"""
    _REGISTRY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _REGISTRY_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")