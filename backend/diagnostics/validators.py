"""
diagnostics/validators.py
=========================
輸入型別驗證 + NaN 自動填補裝飾器。

使用方式（疊加順序固定）：
    @nan_guard()
    @validate_input()
    def run(df: pd.DataFrame, n: int) -> torch.Tensor:
        ...

裝飾器執行順序（由外而內）：
    nan_guard  → 先修正 NaN
    validate_input → 再驗證型別
    原始函數

NaN 處理策略：
    pd.DataFrame  → ffill() 後再 fillna(0)
    np.ndarray    → np.nan_to_num(0)
    torch.Tensor  → torch.nan_to_num(0)
    其他型別      → 略過
"""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Any, Callable, get_type_hints

import numpy as np

# pandas / torch 為選配，避免在未安裝環境下直接 import 報錯
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ── 內部工具 ─────────────────────────────────────────────────────────────────

def _type_name(obj: Any) -> str:
    """回傳物件的型別名稱字串，用於警告訊息。"""
    return type(obj).__name__


def _has_nan(obj: Any) -> bool:
    """偵測 DataFrame / ndarray / Tensor 中是否含有 NaN。"""
    if _HAS_PANDAS and isinstance(obj, pd.DataFrame):
        return bool(obj.isnull().any().any())
    if _HAS_PANDAS and isinstance(obj, pd.Series):
        return bool(obj.isnull().any())
    if isinstance(obj, np.ndarray):
        return bool(np.isnan(obj).any())
    if _HAS_TORCH and isinstance(obj, torch.Tensor):
        return bool(torch.isnan(obj).any())
    return False


def _fix_nan(obj: Any, param_name: str, func_name: str) -> Any:
    """
    填補 NaN 並印出警告。

    DataFrame → ffill 後 fillna(0)
    ndarray   → nan_to_num(0)
    Tensor    → nan_to_num(0)
    """
    if _HAS_PANDAS and isinstance(obj, pd.DataFrame):
        nan_count = int(obj.isnull().sum().sum())
        warnings.warn(
            f"[nan_guard] {func_name}() 參數 `{param_name}` "
            f"含 {nan_count} 個 NaN（DataFrame）→ ffill + fillna(0) 填補",
            stacklevel=4,
        )
        return obj.ffill().fillna(0)

    if _HAS_PANDAS and isinstance(obj, pd.Series):
        nan_count = int(obj.isnull().sum())
        warnings.warn(
            f"[nan_guard] {func_name}() 參數 `{param_name}` "
            f"含 {nan_count} 個 NaN（Series）→ ffill + fillna(0) 填補",
            stacklevel=4,
        )
        return obj.ffill().fillna(0)

    if isinstance(obj, np.ndarray):
        nan_count = int(np.isnan(obj).sum())
        warnings.warn(
            f"[nan_guard] {func_name}() 參數 `{param_name}` "
            f"含 {nan_count} 個 NaN（ndarray）→ nan_to_num(0) 填補",
            stacklevel=4,
        )
        return np.nan_to_num(obj, nan=0.0)

    if _HAS_TORCH and isinstance(obj, torch.Tensor):
        nan_count = int(torch.isnan(obj).sum().item())
        warnings.warn(
            f"[nan_guard] {func_name}() 參數 `{param_name}` "
            f"含 {nan_count} 個 NaN（Tensor）→ nan_to_num(0) 填補",
            stacklevel=4,
        )
        return torch.nan_to_num(obj, nan=0.0)

    return obj


# ── nan_guard 裝飾器 ──────────────────────────────────────────────────────────

def nan_guard() -> Callable:
    """
    掃描所有輸入參數，偵測 NaN 後自動填補，繼續執行不中斷。
    應放在最外層（最先執行）。
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())

            # 處理位置參數
            new_args = list(args)
            for i, val in enumerate(args):
                if i < len(params) and _has_nan(val):
                    new_args[i] = _fix_nan(val, params[i], fn.__qualname__)

            # 處理關鍵字參數
            new_kwargs = dict(kwargs)
            for k, val in kwargs.items():
                if _has_nan(val):
                    new_kwargs[k] = _fix_nan(val, k, fn.__qualname__)

            return fn(*new_args, **new_kwargs)
        return wrapper
    return decorator


# ── validate_input 裝飾器 ─────────────────────────────────────────────────────

def validate_input() -> Callable:
    """
    根據函數 type hints 驗證輸入型別。
    不符合時印出警告，但不 raise，繼續執行。
    應放在 nan_guard 內層。
    """
    def decorator(fn: Callable) -> Callable:
        # 在裝飾時取得 type hints（避免每次呼叫都解析）
        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, value in bound.arguments.items():
                if param_name not in hints:
                    continue
                expected = hints[param_name]

                # 跳過 None 值（Optional 參數）
                if value is None:
                    continue

                # isinstance 對泛型型別（如 list[int]）會失敗，用 try/except 保護
                try:
                    if not isinstance(value, expected):
                        warnings.warn(
                            f"[validate_input] {fn.__qualname__}() 參數 `{param_name}` "
                            f"預期型別 {expected.__name__}，"
                            f"實際收到 {_type_name(value)}",
                            stacklevel=3,
                        )
                except TypeError:
                    # 泛型型別無法用 isinstance 檢查，略過
                    pass

            return fn(*args, **kwargs)
        return wrapper
    return decorator