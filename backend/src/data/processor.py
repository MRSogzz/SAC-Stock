"""
Feature processing for the portfolio SAC pipeline.

The per-stock Tier 1 features are normalized with a leakage-safe 252 trading day
rolling z-score. Cross-sectional z-score is applied in align_features() only to
Tier 1 features. Market state features are derived from the benchmark (0050) and
broadcast after cross-sectional normalization so they are not collapsed to zero.
"""

import numpy as np
import pandas as pd

from diagnostics import register, nan_guard, validate_input


AVG_VOL_WINDOW = 20
RANK_N = 9
ROLLING_Z_WINDOW = 252
MARKET_FEATURE_COLUMNS = [
    "market_ret_20_z",
    "market_vol_20_z",
    "market_drawdown_60_z",
]


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def _calc_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series]:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()

    up = high.diff()
    down = -low.diff()
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    ndm = np.where((down > up) & (down > 0), down, 0.0)

    pdi = 100 * pd.Series(pdm, index=high.index).rolling(period).mean() / (atr + 1e-8)
    ndi = 100 * pd.Series(ndm, index=high.index).rolling(period).mean() / (atr + 1e-8)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-8)
    return dx.rolling(period).mean(), atr


def _rolling_zscore_series(s: pd.Series, window: int = ROLLING_Z_WINDOW) -> pd.Series:
    hist = s.shift(1)
    mu = hist.rolling(window, min_periods=window).mean()
    sig = hist.rolling(window, min_periods=window).std()
    return (s - mu) / (sig + 1e-8)


def _soft_bound(df: pd.DataFrame, scale: float = 10.0) -> pd.DataFrame:
    return scale * np.tanh(df / scale)


# 爆發型特徵（厚尾、比率或加速度類）：clip(-3,3) 後套用 tanh(0.8x)
# 平穩型特徵（範圍固定）：保護原始差異，只做 10*tanh(x/10) soft boundary
_EXPLOSIVE_FEATURES = frozenset({
    "delta_ret_5", "delta_ret_20", "delta_rsi_14",
    "delta_vol_5", "vol_ratio_accel",
    "upper_wick_ratio", "delta_upper_wick", "volume_impulse",
})


def _postprocess_tier1_features(features: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(index=features.index)

    for col in features.columns:
        s = features[col].replace([np.inf, -np.inf], np.nan).astype(np.float64)
        if col.startswith("ret_"):
            s = s.clip(-0.2, 0.2)
        normalized[col] = _rolling_zscore_series(s)

    normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    result = pd.DataFrame(index=normalized.index)
    for col in normalized.columns:
        x = normalized[col].values
        if col in _EXPLOSIVE_FEATURES:
            # 爆發型：clip(-3,3) → tanh(0.8x)，適度拉開中等與強訊號的距離
            result[col] = np.tanh(0.8 * np.clip(x, -3.0, 3.0))
        else:
            # 平穩型：10*tanh(x/10)，保護原始線性差異，防止極端值
            result[col] = 10.0 * np.tanh(x / 10.0)

    return result.dropna()


def _compute_market_features(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    close = benchmark_df["Close"].astype(np.float64)
    ret = close.pct_change()

    raw = pd.DataFrame(index=benchmark_df.index)
    raw["market_ret_20_z"] = close.pct_change(20).clip(-0.2, 0.2)
    raw["market_vol_20_z"] = ret.rolling(20).std()
    raw["market_drawdown_60_z"] = close / (close.rolling(60).max() + 1e-8) - 1.0

    normalized = pd.DataFrame(index=raw.index)
    for col in raw.columns:
        normalized[col] = _rolling_zscore_series(
            raw[col].replace([np.inf, -np.inf], np.nan)
        )

    normalized = normalized.replace([np.inf, -np.inf], np.nan)
    return _soft_bound(normalized).dropna()


@nan_guard()
@validate_input()
@register(
    module="Proc",
    inputs={"df": "pd.DataFrame"},
    outputs={"return": "pd.DataFrame"},
    notes="Build 35 Tier 1 features and apply 252-day rolling z-score.",
)
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]
    o = df["Open"]

    for w in [3, 5, 10, 20, 60]:
        f[f"ret_{w}"] = c.pct_change(w)

    ret_1 = c.pct_change()
    for w in [5, 10, 20]:
        f[f"vol_{w}"] = ret_1.rolling(w).std()

    f["body"] = (c - o) / (o + 1e-8)
    f["upper_wick"] = (h - c.combine(o, max)) / (h - l + 1e-8)
    f["lower_wick"] = (c.combine(o, min) - l) / (h - l + 1e-8)
    f["hl_range"] = (h - l) / (c + 1e-8)

    for w in [5, 20]:
        f[f"vol_ratio_{w}"] = v / (v.rolling(w).mean() + 1e-8)
    f["vol_change"] = v.pct_change()

    for w in [10, 20, 60]:
        f[f"pos_{w}"] = (c - l.rolling(w).min()) / (
            h.rolling(w).max() - l.rolling(w).min() + 1e-8
        )

    rsi = _calc_rsi(c, 14)
    f["rsi_centered"] = rsi - 50
    f["rsi_slope"] = rsi.diff(3)

    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f["z_score_20"] = (c - ma20) / (std20 + 1e-8)

    ma60 = c.rolling(60).mean()
    f["ratio_20_60"] = (ma20 - ma60) / (ma60 + 1e-8)

    adx, atr = _calc_adx(h, l, c, 14)
    f["adx_14"] = adx
    f["atr_change"] = atr.pct_change(5)

    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    f["bb_position"] = (c - lower) / (upper - lower + 1e-8)
    f["bb_width"] = (upper - lower) / (ma20 + 1e-8)

    vol_series = v.pct_change()
    f["price_vol_corr_10"] = ret_1.rolling(10).corr(vol_series)

    f["delta_ret_5"] = f["ret_5"].diff(3)
    f["delta_ret_20"] = f["ret_20"].diff(3)
    f["delta_rsi_14"] = (rsi - 50).diff(3)

    f["delta_vol_5"] = f["vol_5"].diff(3)
    vr5_lagged = f["vol_ratio_5"].shift(3)
    f["vol_ratio_accel"] = f["vol_ratio_5"] / (vr5_lagged + 1e-8)

    f["upper_wick_ratio"] = f["upper_wick"] / (f["hl_range"] + 1e-8)
    f["delta_upper_wick"] = f["upper_wick_ratio"].diff(3)

    vr5_roll_mean = f["vol_ratio_5"].rolling(3).mean()
    vr5_roll_std = f["vol_ratio_5"].rolling(3).std()
    f["volume_impulse"] = (f["vol_ratio_5"] - vr5_roll_mean) / (
        vr5_roll_std + 1e-8
    )

    return _postprocess_tier1_features(f)


@register(
    module="Proc",
    inputs={"stocks": "dict[str, pd.DataFrame]"},
    outputs={
        "feat_dfs": "dict[str, pd.DataFrame]",
        "prices_dict": "dict[str, np.ndarray]",
        "volumes_dict": "dict[str, np.ndarray]",
        "feat_names": "list[str]",
        "dates": "list[str]",
    },
    notes="Align stocks, apply cross-sectional z-score, append benchmark market state.",
)
def align_features(stocks: dict) -> tuple:
    from configs.trading_config import BENCHMARK_STOCK

    feat_dfs = {sid: compute_features(df) for sid, df in stocks.items()}
    market_df = _compute_market_features(stocks[BENCHMARK_STOCK])

    common_idx = None
    for df in feat_dfs.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
    common_idx = common_idx.intersection(market_df.index)

    feat_dfs = {sid: feat_dfs[sid].loc[common_idx].copy() for sid in feat_dfs}
    market_df = market_df.loc[common_idx]

    tier1_cols = feat_dfs[list(feat_dfs.keys())[0]].columns.tolist()
    all_sids = list(feat_dfs.keys())

    for col in tier1_cols:
        mat = np.array(
            [feat_dfs[sid][col].values for sid in all_sids],
            dtype=np.float64,
        )
        mu = mat.mean(axis=0, keepdims=True)
        sig = mat.std(axis=0, keepdims=True)
        z = (mat - mu) / (sig + 1e-8)
        for j, sid in enumerate(all_sids):
            feat_dfs[sid][col] = 10.0 * np.tanh(z[j] / 3.0)

    for sid in all_sids:
        for col in MARKET_FEATURE_COLUMNS:
            feat_dfs[sid][col] = market_df[col].values

    prices_dict = {
        sid: stocks[sid]["Close"].loc[common_idx].values.flatten()
        for sid in feat_dfs
    }
    volumes_dict = {
        sid: stocks[sid]["Volume"].loc[common_idx].values.flatten().astype(np.float64)
        for sid in feat_dfs
    }

    feat_names = feat_dfs[list(feat_dfs.keys())[0]].columns.tolist()
    dates = feat_dfs[list(feat_dfs.keys())[0]].index.strftime("%Y-%m-%d").tolist()

    print(f"feature count={len(feat_names)}, aligned days={len(common_idx)}")
    assert len(feat_names) == 38, f"feature count mismatch: {len(feat_names)} != 38"

    for sid, vol_arr in volumes_dict.items():
        zero_days = (vol_arr == 0).sum()
        if zero_days > 0:
            print(f"  warning: {sid} has {zero_days} zero-volume aligned days")

    return feat_dfs, prices_dict, volumes_dict, feat_names, dates


@register(
    module="Proc",
    inputs={
        "feat_dict": "dict[str, pd.DataFrame]",
        "scaler_dict": "dict | None",
    },
    outputs={
        "scaled_dict": "dict[str, np.ndarray]",
        "scalers": "dict",
    },
    notes="Deprecated compatibility shim. Features are already rolling-z normalized.",
)
def scale_features(feat_dict: dict, scaler_dict: dict = None) -> tuple:
    scaled_dict = {}
    for sid, df in feat_dict.items():
        feat = df.values.copy().astype(np.float64)
        feat = np.where(np.isposinf(feat), 10.0, feat)
        feat = np.where(np.isneginf(feat), -10.0, feat)
        feat = np.where(np.isnan(feat), 0.0, feat)
        scaled_dict[sid] = np.clip(feat, -10.0, 10.0)
    return scaled_dict, {}