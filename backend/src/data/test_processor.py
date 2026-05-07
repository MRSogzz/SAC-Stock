"""
tests/src/data/test_processor.py
==================================
processor.py 的單元測試，對應真實實作：
  - compute_features(df)              → pd.DataFrame（29 個基礎特徵）
  - align_features(stocks)            → (feat_dfs, prices, volumes, names, dates)
  - scale_features(feat_dict, scaler) → (scaled_dict, scalers)

Mock 策略：
  - TRADEABLE_STOCKS / BENCHMARK_STOCK → patch configs，避免依賴真實設定
  - 跨股票排名需要至少 2 支股票，fixture 準備 3 支
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.preprocessing import StandardScaler

# ── 常數 ──────────────────────────────────────────────────────────────────────

N_FEATURES_BASE  = 29   # compute_features 輸出（含 rank 佔位符，不含真實排名）
N_FEATURES_TOTAL = 31   # align_features 後的欄位數（processor 的 assert）
LOOKBACK         = 252  # pos_252 需要的最小資料長度
MIN_ROWS         = LOOKBACK + 60  # 確保 dropna 後仍有足夠資料

MOCK_TRADEABLE  = ["2330", "2317", "2454"]
MOCK_BENCHMARK  = "0050"


# ── 假資料工廠 ────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = MIN_ROWS, seed: int = 0) -> pd.DataFrame:
    """
    建立符合 loader 輸出格式的 OHLCV DataFrame：
    - DatetimeIndex，index.name = "Date"
    - 欄位：Open / High / Low / Close / Volume
    - 無 NaN、Close > 0、Volume > 0
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 200 + rng.standard_normal(n).cumsum()
    close = np.abs(close) + 10   # 確保全為正值

    df = pd.DataFrame({
        "Open":   close * rng.uniform(0.98, 1.00, n),
        "High":   close * rng.uniform(1.00, 1.02, n),
        "Low":    close * rng.uniform(0.97, 0.99, n),
        "Close":  close,
        "Volume": rng.integers(10_000, 200_000, n).astype(float),
    }, index=pd.DatetimeIndex(dates, name="Date"))
    return df


def _make_stocks_dict(
    tickers: list = None,
    n: int = MIN_ROWS,
) -> dict:
    """建立多支股票的假資料 dict，供 align_features 測試使用。"""
    if tickers is None:
        tickers = MOCK_TRADEABLE + [MOCK_BENCHMARK]
    return {sid: _make_ohlcv(n=n, seed=i) for i, sid in enumerate(tickers)}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def ohlcv_df():
    return _make_ohlcv()


@pytest.fixture
def ohlcv_with_nan():
    df = _make_ohlcv()
    rng = np.random.default_rng(99)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        idx = rng.integers(LOOKBACK, len(df), size=3)
        df.loc[df.index[idx], col] = np.nan
    return df


@pytest.fixture
def stocks_dict():
    return _make_stocks_dict()


@pytest.fixture
def mock_trading_config():
    """patch TRADEABLE_STOCKS / BENCHMARK_STOCK，不依賴真實 configs。"""
    with patch("src.data.processor.TRADEABLE_STOCKS", MOCK_TRADEABLE, create=True), \
         patch("src.data.processor.BENCHMARK_STOCK",  MOCK_BENCHMARK, create=True):
        # align_features 內部 import configs，需同時 patch import 路徑
        with patch("configs.trading_config.TRADEABLE_STOCKS", MOCK_TRADEABLE, create=True), \
             patch("configs.trading_config.BENCHMARK_STOCK",  MOCK_BENCHMARK, create=True):
            yield


# ═══════════════════════════════════════════════════════════════════════════════
# compute_features() 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeFeatures:

    def test_output_is_dataframe(self, ohlcv_df):
        """T1：輸出為 pd.DataFrame，index 為 DatetimeIndex。"""
        from src.data.processor import compute_features
        result = compute_features(ohlcv_df)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_output_has_29_columns(self, ohlcv_df):
        """T2：輸出欄位數為 29（含 rank_* 佔位符，不含真實排名計算）。"""
        from src.data.processor import compute_features
        result = compute_features(ohlcv_df)
        assert len(result.columns) == N_FEATURES_BASE, (
            f"預期 {N_FEATURES_BASE} 個欄位，實際 {len(result.columns)}"
        )

    def test_all_expected_feature_groups_present(self, ohlcv_df):
        """T3：五大特徵群（動能/波動/K線/成交量/位置/反轉）的代表欄位均存在。"""
        from src.data.processor import compute_features
        result = compute_features(ohlcv_df)
        cols = set(result.columns)

        expected = {
            # 動能
            "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
            # 波動
            "vol_5", "vol_10", "vol_20",
            # K線
            "body", "upper_wick", "lower_wick", "hl_range",
            # 成交量
            "vol_ratio_5", "vol_ratio_20", "vol_change",
            # 位置
            "pos_10", "pos_20", "pos_60", "pos_252",
            # 反轉/趨勢
            "rsi_centered", "rsi_slope", "z_score_20", "ratio_20_60",
            "adx_14", "atr_change", "bb_position", "bb_width", "price_vol_corr_10",
            # 排名佔位符
            "rank_ret_5", "rank_ret_20", "rank_vol_5",
        }
        missing = expected - cols
        assert not missing, f"缺少以下欄位：{missing}"

    def test_pos_252_needs_lookback_rows_cut(self, ohlcv_df):
        """T4：pos_252 需要 252 天 lookback，dropna 後行數應明顯少於原始資料。"""
        from src.data.processor import compute_features
        result = compute_features(ohlcv_df)
        assert len(result) < len(ohlcv_df), "dropna 應切除前 252 天的 NaN 行"
        assert len(result) > 0, "dropna 後應仍有資料"

    def test_no_nan_in_output(self, ohlcv_df):
        """T5：輸出 DataFrame 不含 NaN（dropna 與 fillna 應已處理）。"""
        from src.data.processor import compute_features
        result = compute_features(ohlcv_df)
        assert not result.isnull().any().any(), "輸出不應含有 NaN"

    def test_no_inf_in_output(self, ohlcv_df):
        """T6：輸出不含 inf / -inf。"""
        from src.data.processor import compute_features
        result = compute_features(ohlcv_df)
        numeric = result.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any(), "輸出不應含有 inf"

    def test_non_rank_features_clipped_to_10(self, ohlcv_df):
        """T7：非 rank_* 特徵的值應在 [-10, 10] 範圍內。"""
        from src.data.processor import compute_features
        result = compute_features(ohlcv_df)
        non_rank = [c for c in result.columns if not c.startswith("rank_")]
        vals = result[non_rank].values
        assert (vals >= -10).all() and (vals <= 10).all(), (
            "非 rank 特徵應 clip 到 [-10, 10]"
        )

    def test_ret_features_clipped_to_02(self, ohlcv_df):
        """T8：ret_* 特徵的值應在 [-0.2, 0.2] 範圍內（額外 clip）。"""
        from src.data.processor import compute_features
        result = compute_features(ohlcv_df)
        ret_cols = [c for c in result.columns if c.startswith("ret_")]
        vals = result[ret_cols].values
        assert (vals >= -0.2).all() and (vals <= 0.2).all(), (
            "ret_* 特徵應額外 clip 到 [-0.2, 0.2]"
        )

    def test_rank_placeholder_all_zero(self, ohlcv_df):
        """T9：rank_* 佔位符在 compute_features 輸出中應全為 0。"""
        from src.data.processor import compute_features
        result = compute_features(ohlcv_df)
        for col in ["rank_ret_5", "rank_ret_20", "rank_vol_5"]:
            assert (result[col] == 0.0).all(), f"{col} 佔位符應全為 0"

    def test_nan_input_handled_by_nan_guard(self, ohlcv_with_nan):
        """T10：含 NaN 的輸入 → @nan_guard 填補後正常執行，輸出不含 NaN。"""
        from src.data.processor import compute_features
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_features(ohlcv_with_nan)

        assert isinstance(result, pd.DataFrame)
        assert not result.isnull().any().any()
        # @nan_guard 應有發出警告
        nan_warns = [x for x in w if "nan_guard" in str(x.message)]
        assert len(nan_warns) > 0, "@nan_guard 應在偵測到 NaN 時發出警告"


# ═══════════════════════════════════════════════════════════════════════════════
# align_features() 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlignFeatures:

    def test_returns_five_tuple(self, stocks_dict, mock_trading_config):
        """T11：回傳值為 5-tuple（feat_dfs, prices, volumes, names, dates）。"""
        from src.data.processor import align_features
        result = align_features(stocks_dict)
        assert len(result) == 5

    def test_feat_dfs_has_31_columns(self, stocks_dict, mock_trading_config):
        """T12：對齊後每支股票的特徵數為 31（29 基礎 + rank_* 填入真實值）。"""
        from src.data.processor import align_features
        feat_dfs, *_ = align_features(stocks_dict)
        for sid, df in feat_dfs.items():
            assert len(df.columns) == N_FEATURES_TOTAL, (
                f"{sid} 特徵數錯誤：{len(df.columns)}，預期 {N_FEATURES_TOTAL}"
            )

    def test_all_stocks_share_same_index(self, stocks_dict, mock_trading_config):
        """T13：對齊後所有股票的 index 完全一致（共同交易日）。"""
        from src.data.processor import align_features
        feat_dfs, *_ = align_features(stocks_dict)
        indices = [df.index for df in feat_dfs.values()]
        for idx in indices[1:]:
            assert indices[0].equals(idx), "所有股票的 index 應完全一致"

    def test_rank_features_in_minus1_to_1(self, stocks_dict, mock_trading_config):
        """T14：rank_* 特徵值應在 [-1, 1] 範圍內。"""
        from src.data.processor import align_features
        feat_dfs, *_ = align_features(stocks_dict)
        for sid in MOCK_TRADEABLE:
            for col in ["rank_ret_5", "rank_ret_20", "rank_vol_5"]:
                vals = feat_dfs[sid][col].values
                assert (vals >= -1.0 - 1e-6).all() and (vals <= 1.0 + 1e-6).all(), (
                    f"{sid}.{col} 超出 [-1, 1]"
                )

    def test_benchmark_rank_is_zero(self, stocks_dict, mock_trading_config):
        """T15：BENCHMARK_STOCK（0050）的 rank_* 特徵應全為 0。"""
        from src.data.processor import align_features
        feat_dfs, *_ = align_features(stocks_dict)
        for col in ["rank_ret_5", "rank_ret_20", "rank_vol_5"]:
            assert (feat_dfs[MOCK_BENCHMARK][col] == 0.0).all(), (
                f"0050 的 {col} 應全為 0"
            )

    def test_prices_dict_shape_matches_dates(self, stocks_dict, mock_trading_config):
        """T16：prices_dict 每支股票的長度應等於共同交易日數。"""
        from src.data.processor import align_features
        feat_dfs, prices_dict, _, _, dates = align_features(stocks_dict)
        n_dates = len(dates)
        for sid, arr in prices_dict.items():
            assert len(arr) == n_dates, (
                f"{sid} prices 長度 {len(arr)} 不等於 dates 長度 {n_dates}"
            )

    def test_volumes_dict_dtype_float64(self, stocks_dict, mock_trading_config):
        """T17：volumes_dict 應為 float64（供後續零股邏輯使用）。"""
        from src.data.processor import align_features
        _, _, volumes_dict, *_ = align_features(stocks_dict)
        for sid, arr in volumes_dict.items():
            assert arr.dtype == np.float64, (
                f"{sid} volumes dtype 應為 float64，實際為 {arr.dtype}"
            )

    def test_feat_names_and_dates_are_lists(self, stocks_dict, mock_trading_config):
        """T18：feat_names 和 dates 應為 list[str]。"""
        from src.data.processor import align_features
        _, _, _, feat_names, dates = align_features(stocks_dict)
        assert isinstance(feat_names, list) and all(isinstance(x, str) for x in feat_names)
        assert isinstance(dates, list)     and all(isinstance(x, str) for x in dates)


# ═══════════════════════════════════════════════════════════════════════════════
# scale_features() 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestScaleFeatures:

    @pytest.fixture
    def small_feat_dict(self):
        """小型 feat_dict（不需要對齊，直接用 compute_features 輸出）。"""
        tickers = ["2330", "2317"]
        result  = {}
        for i, sid in enumerate(tickers):
            df = _make_ohlcv(seed=i)
            from src.data.processor import compute_features
            result[sid] = compute_features(df)
        return result

    def test_returns_two_tuple(self, small_feat_dict):
        """T19：回傳值為 2-tuple（scaled_dict, scalers）。"""
        from src.data.processor import scale_features
        result = scale_features(small_feat_dict)
        assert len(result) == 2

    def test_scaled_values_clipped_to_5(self, small_feat_dict):
        """T20：標準化後的值應 clip 到 [-5, 5]。"""
        from src.data.processor import scale_features
        scaled_dict, _ = scale_features(small_feat_dict)
        for sid, arr in scaled_dict.items():
            assert (arr >= -5.0 - 1e-8).all() and (arr <= 5.0 + 1e-8).all(), (
                f"{sid} 標準化後超出 [-5, 5]"
            )

    def test_no_nan_after_scaling(self, small_feat_dict):
        """T21：標準化後不含 NaN（inf 已在 scale_features 內替換）。"""
        from src.data.processor import scale_features
        scaled_dict, _ = scale_features(small_feat_dict)
        for sid, arr in scaled_dict.items():
            assert not np.isnan(arr).any(), f"{sid} 標準化後含有 NaN"

    def test_fit_mode_creates_scalers(self, small_feat_dict):
        """T22：scaler_dict=None（fit 模式）→ 回傳包含每支股票 scaler 的 dict。"""
        from src.data.processor import scale_features
        _, scalers = scale_features(small_feat_dict, scaler_dict=None)
        assert set(scalers.keys()) == set(small_feat_dict.keys())
        assert all(isinstance(s, StandardScaler) for s in scalers.values())

    def test_transform_mode_uses_existing_scalers(self, small_feat_dict):
        """T23：傳入已訓練的 scaler_dict → 使用 transform 而非 fit，scaler 不變。"""
        from src.data.processor import scale_features

        # 先 fit
        _, fitted_scalers = scale_features(small_feat_dict)

        # 再 transform（使用相同資料，結果應一致）
        scaled_again, returned_scalers = scale_features(
            small_feat_dict, scaler_dict=fitted_scalers
        )
        # 回傳的 scaler 應與傳入的相同（物件一致）
        assert returned_scalers is fitted_scalers

    def test_output_shape_matches_input(self, small_feat_dict):
        """T24：輸出 ndarray 的 shape 應與輸入 DataFrame shape 一致。"""
        from src.data.processor import scale_features
        scaled_dict, _ = scale_features(small_feat_dict)
        for sid, arr in scaled_dict.items():
            expected_shape = small_feat_dict[sid].shape
            assert arr.shape == expected_shape, (
                f"{sid} shape 不符：{arr.shape} vs {expected_shape}"
            )