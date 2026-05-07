"""
通用工具函數：時間格式、路徑處理、JSON 清理
"""
import math
import numpy as np
from datetime import datetime

from diagnostics import register


@register(
    module="Utils",
    inputs={},
    outputs={"return": "str"},
    notes="回傳當前時間字串，格式 YYYY-MM-DD HH:MM:SS",
)
def now_str() -> str:
    """回傳當前時間字串"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@register(
    module="Utils",
    inputs={"dt": "datetime | pd.Timestamp"},
    outputs={"return": "str"},
    notes="pandas Timestamp 或 datetime 轉字串，格式 YYYY-MM-DD",
)
def date_str(dt) -> str:
    """pandas Timestamp 或 datetime 轉字串"""
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m-%d")
    return str(dt)[:10]


@register(
    module="Utils",
    inputs={"obj": "dict | list | np.ndarray | scalar"},
    outputs={"return": "dict | list | scalar"},
    notes="遞迴清理 dict/list，nan/inf → 0.0；numpy 型別轉 Python 原生型別",
)
def sanitize(obj):
    """
    遞迴清理 dict/list，確保 JSON 序列化安全。
    處理 nan/inf 和所有 numpy 型別。
    """
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        v = float(obj)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else v
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    elif isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    elif isinstance(obj, bool):
        return obj
    return obj


@register(
    module="Utils",
    inputs={"v": "Any", "default": "float"},
    outputs={"return": "float"},
    notes="安全轉 float；None / NaN / inf / 無法轉換 → 回傳 default",
)
def safe_float(v, default: float = 0.0) -> float:
    """確保數值 JSON 安全"""
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


@register(
    module="Utils",
    inputs={"lst": "list", "default": "float"},
    outputs={"return": "list[float]"},
    notes="對列表中每個元素套用 safe_float，清理 nan/inf",
)
def safe_list(lst, default: float = 0.0) -> list:
    """清理列表中的 nan/inf"""
    return [safe_float(v, default) for v in lst]