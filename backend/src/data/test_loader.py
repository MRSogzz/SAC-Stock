"""
tests/src/data/test_loader.py
==============================
loader.py 的單元測試，對應真實實作：
  - load_stock(stock_id, period) → pd.DataFrame
  - load_all_stocks(period)      → dict[str, pd.DataFrame]

快取機制：
  - 快取檔案路徑：{CACHE_DIR}/{stock_id}_{period}.csv
  - 有效期：CACHE_EXPIRE_HOURS 小時（mtime 判斷）
  - 快取過期或不存在 → 呼叫 FinMind DataLoader

Mock 策略：
  - FinMind API  → patch "src.data.loader.DataLoader"
  - CACHE_DIR    → 重導到 pytest tmp_path（不污染真實快取）
  - 快取時間     → patch "os.path.getmtime"
"""

import os
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ── 假資料工廠 ────────────────────────────────────────────────────────────────

def _make_raw_finmind(stock_id: str = "2330", n: int = 60) -> pd.DataFrame:
    """
    模擬 FinMind taiwan_stock_daily() 回傳的原始格式。
    欄位名稱與真實 API 一致：date / open / max / min / close / Trading_Volume
    """
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    close = 500 + np.random.default_rng(0).standard_normal(n).cumsum()
    return pd.DataFrame({
        "date":           dates.strftime("%Y-%m-%d"),
        "stock_id":       stock_id,
        "open":           close * 0.99,
        "max":            close * 1.01,   # FinMind 用 max
        "min":            close * 0.98,   # FinMind 用 min
        "close":          close,
        "Trading_Volume": np.random.randint(1_000, 50_000, n).astype(float),
    })


def _make_cached_csv(path, n: int = 60):
    """
    模擬已存在的快取 CSV（已轉換為標準 OHLCV 格式）。
    """
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    close = 500 + np.random.default_rng(1).standard_normal(n).cumsum()
    df = pd.DataFrame({
        "Open":   close * 0.99,
        "High":   close * 1.01,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": np.random.randint(1_000, 50_000, n).astype(float),
    }, index=pd.DatetimeIndex(dates, name="Date"))
    df.to_csv(path)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# load_stock() 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadStock:

    def test_returns_ohlcv_dataframe(self, tmp_path):
        """T1：正常下載 → 回傳 DataFrame，欄位為標準 OHLCV，index 為 DatetimeIndex。"""
        from src.data.loader import load_stock

        with patch("src.data.loader.CACHE_DIR", str(tmp_path)), \
             patch("src.data.loader.DataLoader") as MockDL:
            MockDL.return_value.taiwan_stock_daily.return_value = _make_raw_finmind()
            df = load_stock("2330", period="3y")

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"Open", "High", "Low", "Close", "Volume"}
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "Date"
        assert len(df) > 0

    def test_cache_hit_skips_api(self, tmp_path):
        """T2：快取存在且未過期 → 直接讀快取，不呼叫 FinMind API。"""
        from src.data.loader import load_stock

        # 預先建立快取 CSV
        cache_file = tmp_path / "2330_3y.csv"
        original = _make_cached_csv(cache_file)

        # mtime = 剛剛（未過期）
        fresh_mtime = time.time()

        with patch("src.data.loader.CACHE_DIR", str(tmp_path)), \
             patch("src.data.loader.CACHE_EXPIRE_HOURS", 24), \
             patch("os.path.getmtime", return_value=fresh_mtime), \
             patch("src.data.loader.DataLoader") as MockDL:

            df = load_stock("2330", period="3y")

            MockDL.assert_not_called()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(original)

    def test_cache_expired_calls_api(self, tmp_path):
        """T3：快取過期（mtime 超過 CACHE_EXPIRE_HOURS）→ 重新呼叫 API。"""
        from src.data.loader import load_stock

        # 建立快取檔（內容不重要，反正要過期）
        cache_file = tmp_path / "2330_3y.csv"
        _make_cached_csv(cache_file)

        # mtime = 48 小時前
        old_mtime = time.time() - 48 * 3600

        with patch("src.data.loader.CACHE_DIR", str(tmp_path)), \
             patch("src.data.loader.CACHE_EXPIRE_HOURS", 24), \
             patch("os.path.getmtime", return_value=old_mtime), \
             patch("src.data.loader.DataLoader") as MockDL:

            MockDL.return_value.taiwan_stock_daily.return_value = _make_raw_finmind()
            load_stock("2330", period="3y")

            MockDL.return_value.taiwan_stock_daily.assert_called_once()

    def test_no_cache_saves_csv_after_download(self, tmp_path):
        """T4：無快取 → 下載後應將結果存成 CSV。"""
        from src.data.loader import load_stock

        with patch("src.data.loader.CACHE_DIR", str(tmp_path)), \
             patch("src.data.loader.DataLoader") as MockDL:
            MockDL.return_value.taiwan_stock_daily.return_value = _make_raw_finmind()
            load_stock("2330", period="3y")

        cache_file = tmp_path / "2330_3y.csv"
        assert cache_file.exists(), "下載後應自動建立快取 CSV"

    def test_empty_api_response_raises_value_error(self, tmp_path):
        """T5：API 回傳空 DataFrame → raise ValueError（不 crash）。"""
        from src.data.loader import load_stock

        with patch("src.data.loader.CACHE_DIR", str(tmp_path)), \
             patch("src.data.loader.DataLoader") as MockDL:
            MockDL.return_value.taiwan_stock_daily.return_value = pd.DataFrame()

            with pytest.raises(ValueError, match="無法取得"):
                load_stock("INVALID", period="3y")

    def test_suspended_days_filtered(self, tmp_path):
        """T6：停牌日（Close=0 或 Volume=0）→ 過濾後不出現在結果中。"""
        from src.data.loader import load_stock

        raw = _make_raw_finmind(n=10)
        raw.loc[0, "close"]          = 0.0   # Close = 0，停牌
        raw.loc[1, "Trading_Volume"] = 0.0   # Volume = 0，停牌

        with patch("src.data.loader.CACHE_DIR", str(tmp_path)), \
             patch("src.data.loader.DataLoader") as MockDL:
            MockDL.return_value.taiwan_stock_daily.return_value = raw
            df = load_stock("2330", period="3y")

        assert (df["Close"] > 0).all(),  "Close = 0 的停牌日應被過濾"
        assert (df["Volume"] > 0).all(), "Volume = 0 的停牌日應被過濾"
        assert len(df) == 8  # 10 - 2 筆停牌

    def test_output_sorted_ascending_by_date(self, tmp_path):
        """T7：不論 API 回傳順序，輸出 index 應為升序日期。"""
        from src.data.loader import load_stock

        raw = _make_raw_finmind(n=20)
        shuffled = raw.sample(frac=1, random_state=42)  # 打亂順序

        with patch("src.data.loader.CACHE_DIR", str(tmp_path)), \
             patch("src.data.loader.DataLoader") as MockDL:
            MockDL.return_value.taiwan_stock_daily.return_value = shuffled
            df = load_stock("2330", period="3y")

        assert df.index.is_monotonic_increasing, "index 應為升序日期"

    def test_finmind_columns_renamed_correctly(self, tmp_path):
        """T8：FinMind 欄位 max/min 應正確對應到 High/Low（不殘留原始欄位）。"""
        from src.data.loader import load_stock

        with patch("src.data.loader.CACHE_DIR", str(tmp_path)), \
             patch("src.data.loader.DataLoader") as MockDL:
            MockDL.return_value.taiwan_stock_daily.return_value = _make_raw_finmind()
            df = load_stock("2330", period="3y")

        assert "High" in df.columns and "Low" in df.columns
        assert "max"  not in df.columns and "min" not in df.columns


# ═══════════════════════════════════════════════════════════════════════════════
# load_all_stocks() 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadAllStocks:

    # 縮小 STOCK_POOL 為 2 支，加速測試
    _MOCK_POOL = [{"id": "2330"}, {"id": "2317"}]

    def _patch_dl_two_stocks(self, n_2330=60, n_2317=60):
        """回傳一個 side_effect，讓兩支股票各回傳指定長度的假資料。"""
        raw_map = {
            "2330": _make_raw_finmind("2330", n=n_2330),
            "2317": _make_raw_finmind("2317", n=n_2317),
        }
        def side_effect(stock_id, start_date):
            return raw_map[stock_id]
        return side_effect

    def test_returns_dict_of_dataframes(self, tmp_path):
        """T9：正常執行 → 回傳 dict，key 為 stock_id，value 為 DataFrame。"""
        from src.data.loader import load_all_stocks

        with patch("src.data.loader.CACHE_DIR", str(tmp_path)), \
             patch("src.data.loader.STOCK_POOL", self._MOCK_POOL), \
             patch("src.data.loader.DataLoader") as MockDL:
            MockDL.return_value.taiwan_stock_daily.side_effect = \
                self._patch_dl_two_stocks()
            result = load_all_stocks(period="3y")

        assert isinstance(result, dict)
        assert "2330" in result and "2317" in result
        assert all(isinstance(v, pd.DataFrame) for v in result.values())

    def test_common_dates_aligned(self, tmp_path):
        """T10：對齊後所有 DataFrame 的 index 完全相同（取交集）。"""
        from src.data.loader import load_all_stocks

        # 2330 多 5 筆（起始日更早）→ 交集後兩者長度應相同
        with patch("src.data.loader.CACHE_DIR", str(tmp_path)), \
             patch("src.data.loader.STOCK_POOL", self._MOCK_POOL), \
             patch("src.data.loader.DataLoader") as MockDL:
            MockDL.return_value.taiwan_stock_daily.side_effect = \
                self._patch_dl_two_stocks(n_2330=65, n_2317=60)
            result = load_all_stocks(period="3y")

        indices = [df.index for df in result.values()]
        for idx in indices[1:]:
            assert indices[0].equals(idx), "對齊後所有 DataFrame 的 index 應相同"

    def test_failed_stock_skipped_gracefully(self, tmp_path):
        """T11：單支股票下載失敗 → 跳過，其他股票照常回傳，整體不 crash。"""
        from src.data.loader import load_all_stocks

        def side_effect(stock_id, start_date):
            if stock_id == "2317":
                raise ConnectionError("API 連線失敗")
            return _make_raw_finmind("2330", n=60)

        with patch("src.data.loader.CACHE_DIR", str(tmp_path)), \
             patch("src.data.loader.STOCK_POOL", self._MOCK_POOL), \
             patch("src.data.loader.DataLoader") as MockDL:
            MockDL.return_value.taiwan_stock_daily.side_effect = side_effect
            result = load_all_stocks(period="3y")

        assert "2330" in result, "正常股票應照常回傳"
        assert "2317" not in result, "失敗股票應被跳過"

    def test_all_stocks_fail_raises_value_error(self, tmp_path):
        """T12：所有股票都失敗 → raise ValueError（不靜默失敗）。"""
        from src.data.loader import load_all_stocks

        with patch("src.data.loader.CACHE_DIR", str(tmp_path)), \
             patch("src.data.loader.STOCK_POOL", self._MOCK_POOL), \
             patch("src.data.loader.DataLoader") as MockDL:
            MockDL.return_value.taiwan_stock_daily.side_effect = \
                ConnectionError("全部失敗")

            with pytest.raises(ValueError, match="所有股票下載失敗"):
                load_all_stocks(period="3y")