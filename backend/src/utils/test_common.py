"""
tests/src/utils/test_common.py
================================
common.py 的單元測試，對應真實實作：
  - now_str()          → str（格式 YYYY-MM-DD HH:MM:SS）
  - date_str(dt)       → str（格式 YYYY-MM-DD）
  - sanitize(obj)      → JSON 安全的 dict/list/scalar
  - safe_float(v)      → float
  - safe_list(lst)     → list[float]
"""

import math
import numpy as np
import pytest
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# now_str()
# ═══════════════════════════════════════════════════════════════════════════════

class TestNowStr:

    def test_returns_string(self):
        """T1：now_str() 回傳 str。"""
        from src.utils.common import now_str
        assert isinstance(now_str(), str)

    def test_format_matches_pattern(self):
        """T2：格式應為 YYYY-MM-DD HH:MM:SS（可用 datetime.strptime 解析）。"""
        from src.utils.common import now_str
        result = now_str()
        parsed = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        assert isinstance(parsed, datetime)


# ═══════════════════════════════════════════════════════════════════════════════
# date_str()
# ═══════════════════════════════════════════════════════════════════════════════

class TestDateStr:

    def test_datetime_input(self):
        """T3：接受 datetime 物件，回傳 YYYY-MM-DD 字串。"""
        from src.utils.common import date_str
        dt = datetime(2023, 6, 15)
        assert date_str(dt) == "2023-06-15"

    def test_pandas_timestamp_input(self):
        """T4：接受 pandas Timestamp，回傳 YYYY-MM-DD 字串。"""
        import pandas as pd
        from src.utils.common import date_str
        ts = pd.Timestamp("2023-06-15")
        assert date_str(ts) == "2023-06-15"

    def test_string_input_truncated(self):
        """T5：接受日期字串（帶時間），截取前 10 碼。"""
        from src.utils.common import date_str
        assert date_str("2023-06-15 09:30:00") == "2023-06-15"

    def test_output_length_10(self):
        """T6：輸出字串長度應為 10（YYYY-MM-DD）。"""
        from src.utils.common import date_str
        assert len(date_str(datetime(2023, 1, 1))) == 10


# ═══════════════════════════════════════════════════════════════════════════════
# sanitize()
# ═══════════════════════════════════════════════════════════════════════════════

class TestSanitize:

    def test_dict_passthrough(self):
        """T7：正常 dict 應原樣回傳（值不變）。"""
        from src.utils.common import sanitize
        obj = {"a": 1, "b": "hello", "c": True}
        result = sanitize(obj)
        assert result == obj

    def test_nan_float_becomes_zero(self):
        """T8：Python float NaN → 0.0。"""
        from src.utils.common import sanitize
        assert sanitize(float("nan")) == 0.0

    def test_inf_float_becomes_zero(self):
        """T9：Python float inf / -inf → 0.0。"""
        from src.utils.common import sanitize
        assert sanitize(float("inf"))  == 0.0
        assert sanitize(float("-inf")) == 0.0

    def test_numpy_integer_converted(self):
        """T10：np.integer → Python int。"""
        from src.utils.common import sanitize
        result = sanitize(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_floating_nan_becomes_zero(self):
        """T11：np.floating NaN → 0.0。"""
        from src.utils.common import sanitize
        result = sanitize(np.float64(float("nan")))
        assert result == 0.0

    def test_numpy_floating_normal(self):
        """T12：正常 np.floating → Python float，值不變。"""
        from src.utils.common import sanitize
        result = sanitize(np.float64(3.14))
        assert abs(result - 3.14) < 1e-9
        assert isinstance(result, float)

    def test_numpy_bool_converted(self):
        """T13：np.bool_ → Python bool。"""
        from src.utils.common import sanitize
        result = sanitize(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_numpy_array_to_list(self):
        """T14：np.ndarray → 遞迴清理的 list。"""
        from src.utils.common import sanitize
        arr    = np.array([1.0, float("nan"), float("inf")])
        result = sanitize(arr)
        assert isinstance(result, list)
        assert result[0] == 1.0
        assert result[1] == 0.0
        assert result[2] == 0.0

    def test_nested_dict_recursive(self):
        """T15：巢狀 dict 應遞迴清理。"""
        from src.utils.common import sanitize
        obj = {"outer": {"inner": float("nan"), "val": 1}}
        result = sanitize(obj)
        assert result["outer"]["inner"] == 0.0
        assert result["outer"]["val"]   == 1

    def test_list_recursive(self):
        """T16：list 應遞迴清理每個元素。"""
        from src.utils.common import sanitize
        lst    = [1.0, float("nan"), np.int64(3)]
        result = sanitize(lst)
        assert result == [1.0, 0.0, 3]

    def test_bool_not_converted_to_int(self):
        """T17：Python bool 不應被轉為 int（JSON 需區分 true/false vs 0/1）。"""
        from src.utils.common import sanitize
        assert sanitize(True)  is True
        assert sanitize(False) is False


# ═══════════════════════════════════════════════════════════════════════════════
# safe_float()
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafeFloat:

    def test_normal_float_passthrough(self):
        """T18：正常 float 直接回傳，值不變。"""
        from src.utils.common import safe_float
        assert abs(safe_float(3.14) - 3.14) < 1e-9

    def test_int_converted_to_float(self):
        """T19：int 應轉為 float。"""
        from src.utils.common import safe_float
        result = safe_float(5)
        assert isinstance(result, float)
        assert result == 5.0

    def test_none_returns_default(self):
        """T20：None → default。"""
        from src.utils.common import safe_float
        assert safe_float(None, default=99.0) == 99.0

    def test_nan_returns_default(self):
        """T21：float("nan") → default。"""
        from src.utils.common import safe_float
        assert safe_float(float("nan"), default=-1.0) == -1.0

    def test_inf_returns_default(self):
        """T22：float("inf") / float("-inf") → default。"""
        from src.utils.common import safe_float
        assert safe_float(float("inf"),  default=0.0) == 0.0
        assert safe_float(float("-inf"), default=0.0) == 0.0

    def test_string_number_converted(self):
        """T23：可解析的數字字串 → float。"""
        from src.utils.common import safe_float
        assert safe_float("2.71") == pytest.approx(2.71)

    def test_invalid_string_returns_default(self):
        """T24：無法解析的字串 → default。"""
        from src.utils.common import safe_float
        assert safe_float("abc", default=-1.0) == -1.0

    def test_custom_default(self):
        """T25：default 參數應正確使用。"""
        from src.utils.common import safe_float
        assert safe_float(None, default=42.0) == 42.0
        assert safe_float("bad", default=42.0) == 42.0


# ═══════════════════════════════════════════════════════════════════════════════
# safe_list()
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafeList:

    def test_normal_list_passthrough(self):
        """T26：正常數值列表不變。"""
        from src.utils.common import safe_list
        result = safe_list([1.0, 2.0, 3.0])
        assert result == [1.0, 2.0, 3.0]

    def test_nan_in_list_replaced(self):
        """T27：列表中的 NaN → default。"""
        from src.utils.common import safe_list
        result = safe_list([1.0, float("nan"), 3.0], default=0.0)
        assert result == [1.0, 0.0, 3.0]

    def test_inf_in_list_replaced(self):
        """T28：列表中的 inf → default。"""
        from src.utils.common import safe_list
        result = safe_list([float("inf"), 2.0], default=-999.0)
        assert result == [-999.0, 2.0]

    def test_empty_list(self):
        """T29：空列表 → 回傳空列表，不 crash。"""
        from src.utils.common import safe_list
        assert safe_list([]) == []

    def test_returns_list_of_floats(self):
        """T30：回傳值應為 list，每個元素為 float。"""
        from src.utils.common import safe_list
        result = safe_list([1, 2, 3])
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)