"""
diagnostics/inspector.py
=========================
Registry / IO Map 展示工具（靜態，不依賴訓練實例）。

用法：
    from diagnostics import show_all, show, summary
    show_all()       # 印出完整 IO Map
    show("Agent")    # 只看 Agent 模組
    summary()        # call_count 統計，找出從未執行的函數
"""

from __future__ import annotations

import time

from .Registry import _REGISTRY, IOSignature
from .logger import registry_log


# ── 內部輔助 ──────────────────────────────────────────────────────────────────

def _registered_modules() -> str:
    modules = sorted({r.module for r in _REGISTRY.values()})
    return ", ".join(modules) if modules else "（無）"


def _render_table(rows: list[IOSignature], title: str) -> None:
    """印出對齊格式的 IO 表格。"""
    col_w = 60
    print(f"\n{'═' * col_w}")
    print(f"  {title}")
    print(f"{'═' * col_w}")
    for r in rows:
        inp = ", ".join(f"{k}: {v}" for k, v in r.inputs.items())
        out = ", ".join(v for v in r.outputs.values())
        print(f"\n  [{r.module}] {r.name}  (calls: {r.call_count})")
        print(f"    IN : {inp}")
        print(f"    OUT: {out}")
        if r.notes:
            print(f"    ▸  {r.notes}")
    print(f"\n{'═' * col_w}\n")


# ── 公開介面 ──────────────────────────────────────────────────────────────────

def show_all() -> None:
    """印出完整 IO Map 表格（所有已登錄模組），並寫入 log。"""
    rows = list(_REGISTRY.values())
    _render_table(rows, title="IO Map — All Modules")
    registry_log(f"show_all() called — {len(rows)} entries")


def show(module: str) -> None:
    """
    只顯示指定模組的 IO 記錄。

    Args:
        module: 模組名稱，例如 "Data"、"Proc"、"Env"、"Agent"、"Model"、"Engine"
    """
    rows = [r for r in _REGISTRY.values() if r.module == module]
    if not rows:
        msg = f"（找不到模組 '{module}'，目前已登錄：{_registered_modules()}）"
        print(msg)
        registry_log(f"show('{module}') — not found")
        return
    _render_table(rows, title=f"IO Map — Module: {module}")
    registry_log(f"show('{module}') called — {len(rows)} entries")


def summary() -> None:
    """
    印出 call_count 統計表，並標示從未執行的函數（Dead code 候選）。
    """
    rows = list(_REGISTRY.values())
    if not rows:
        print("（登錄表為空）")
        return

    rows_sorted  = sorted(rows, key=lambda r: r.call_count, reverse=True)
    never_called = []

    print(f"\n{'─' * 60}")
    print(f"  {'MODULE':<10} {'FUNCTION':<28} {'CALLS':>6}  STATUS")
    print(f"{'─' * 60}")

    for r in rows_sorted:
        status = "⚠️  never called" if r.call_count == 0 else "✅"
        last   = (
            time.strftime("%H:%M:%S", time.localtime(r.last_called))
            if r.last_called else "—"
        )
        print(f"  {r.module:<10} {r.name:<28} {r.call_count:>6}  {status}  (last: {last})")
        if r.call_count == 0:
            never_called.append(f"{r.module}.{r.name}")

    print(f"{'─' * 60}")
    print(f"  總計 {len(rows)} 個函數 | 從未呼叫: {len(never_called)}")
    if never_called:
        print(f"  ⚠️  Dead code 候選: {', '.join(never_called)}")
    print(f"{'─' * 60}\n")

    registry_log(
        f"summary() called — {len(rows)} entries, "
        f"{len(never_called)} never called: {never_called}"
    )