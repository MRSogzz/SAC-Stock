"""
特徵工程：從原始 OHLCV 計算特徵（SAC-Stock-v7 Variant H / Tier 1）

設計原則（黃金基準，不可動搖）：
  - 嚴禁：任何排序特徵（rank_*）
  - 嚴禁：Look-ahead Bias（依賴未來資料）
  - 嚴禁：全域 StandardScaler（改用 252 日 Rolling Z-score）
  - 強制：soft boundary = 10 * tanh(x/10)
  - Cross-sectional Z-score 在 align_features() 跨股票計算

特徵清單（Tier 1，35個）：
  動能：ret_3, ret_5, ret_10, ret_20, ret_60（5）
  波動：vol_5, vol_10, vol_20（3）
  K線：body, upper_wick, lower_wick, hl_range（4）
  成交量：vol_ratio_5, vol_ratio_20, vol_change（3）
  位置：pos_10, pos_20, pos_60（3）
  反轉/趨勢：rsi_centered, rsi_slope, z_score_20,
              ratio_20_60, adx_14, atr_change,
              bb_position, bb_width, price_vol_corr_10（9）
  [Tier 1 新增] 動量加速度：delta_ret_5, delta_ret_20, delta_rsi_14（3）
  [Tier 1 新增] 波動擴張：delta_vol_5, vol_ratio_accel（2）
  [Tier 1 新增] 盤口微結構：upper_wick_ratio, delta_upper_wick, volume_impulse（3）
  合計：5+3+4+3+3+9+3+2+3 = 35
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from diagnostics import register, nan_guard, validate_input

AVG_VOL_WINDOW = 20   # 零股成交率估算用的滾動平均窗口
RANK_N         = 9    # 參與排名的股票數（可交易股票，不含 0050）


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def _calc_adx(high: pd.Series, low: pd.Series, close: pd.Series,
              period: int = 14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    up   = high.diff()
    down = -low.diff()
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)

    pdi = 100 * pd.Series(pdm, index=high.index).rolling(period).mean() / (atr + 1e-8)
    ndi = 100 * pd.Series(ndm, index=high.index).rolling(period).mean() / (atr + 1e-8)
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-8)
    return dx.rolling(period).mean(), atr


def _normalize_rank(ranks: np.ndarray, n: int) -> np.ndarray:
    """
    把 1~N 的排名正規化到 [-1, 1]。
    最強（rank=N）→ 1.0，最弱（rank=1）→ -1.0，中位數 → 0.0。
    公式：(rank - 1) / (N - 1) * 2 - 1
    """
    if n <= 1:
        return np.zeros_like(ranks, dtype=np.float64)
    return (ranks - 1) / (n - 1) * 2 - 1


# ── compute_features ──────────────────────────────────────────────────────────

@nan_guard()
@validate_input()
@register(
    module="Proc",
    inputs={"df": "pd.DataFrame"},
    outputs={"return": "pd.DataFrame"},
    notes="單支股票 29 個基礎特徵（不含 rank_*），dropna 切除前 252 天",
)
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算單支股票的 35 個基礎特徵（Tier 1，不含跨股票 Cross-sectional Z-score）。
    Cross-sectional Z-score 在 align_features() 裡跨股票計算後覆寫特徵值。
    最長 lookback 為 60 天，dropna() 自動切除前端不足的資料。
    """
    f = pd.DataFrame(index=df.index)
    c, h, l, v, o = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]

    # ── 動能 ─────────────────────────────────────────────────────────────
    for w in [3, 5, 10, 20, 60]:
        f[f"ret_{w}"] = c.pct_change(w)

    # ── 波動 ─────────────────────────────────────────────────────────────
    for w in [5, 10, 20]:
        f[f"vol_{w}"] = c.pct_change().rolling(w).std()

    # ── K線結構 ──────────────────────────────────────────────────────────
    f["body"]       = (c - o) / (o + 1e-8)
    f["upper_wick"] = (h - c.combine(o, max)) / (h - l + 1e-8)
    f["lower_wick"] = (c.combine(o, min) - l) / (h - l + 1e-8)
    f["hl_range"]   = (h - l) / (c + 1e-8)

    # ── 成交量動能 ───────────────────────────────────────────────────────
    for w in [5, 20]:
        f[f"vol_ratio_{w}"] = v / (v.rolling(w).mean() + 1e-8)
    f["vol_change"] = v.pct_change()

    # ── 價位位置（短中長週期）────────────────────────────────────────────
    for w in [10, 20, 60]:
        f[f"pos_{w}"] = (c - l.rolling(w).min()) / (
            h.rolling(w).max() - l.rolling(w).min() + 1e-8)

    # pos_252 已永久移除（會造成 look-ahead 污染且與 Shared MLP 相容性差）

    # ── 反轉類特徵 ───────────────────────────────────────────────────────
    rsi = _calc_rsi(c, 14)
    f["rsi_centered"]      = rsi - 50
    f["rsi_slope"]         = rsi.diff(3)

    ma20  = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f["z_score_20"]        = (c - ma20) / (std20 + 1e-8)

    ma60 = c.rolling(60).mean()
    f["ratio_20_60"]       = (ma20 - ma60) / (ma60 + 1e-8)

    adx, atr = _calc_adx(h, l, c, 14)
    f["adx_14"]            = adx
    f["atr_change"]        = atr.pct_change(5)

    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    f["bb_position"]       = (c - lower) / (upper - lower + 1e-8)
    f["bb_width"]          = (upper - lower) / (ma20 + 1e-8)

    ret_series = c.pct_change()
    vol_series = v.pct_change()
    f["price_vol_corr_10"] = ret_series.rolling(10).corr(vol_series)

    # rank_* 特徵已永久移除（ID 依賴，破壞 Shared MLP 的股票無差別性）

    # ── [Tier 1] 動量加速度（捕捉趨勢的變化速度）─────────────────────────
    # 使用 3 日差分平滑單日噪音，區分「加速上漲」與「強弩之末」
    f["delta_ret_5"]   = f["ret_5"].diff(3)    # ret_5(t) - ret_5(t-3)
    f["delta_ret_20"]  = f["ret_20"].diff(3)   # ret_20(t) - ret_20(t-3)
    f["delta_rsi_14"]  = (rsi - 50).diff(3)    # RSI_14(t) - RSI_14(t-3)

    # ── [Tier 1] 波動擴張（捕捉風險結構的轉變）──────────────────────────
    # 行情啟動初期常伴隨波動率的驟升
    f["delta_vol_5"]      = f["vol_5"].diff(3)                           # vol_5(t) - vol_5(t-3)
    vr5_lagged            = f["vol_ratio_5"].shift(3)
    f["vol_ratio_accel"]  = f["vol_ratio_5"] / (vr5_lagged + 1e-8)      # 偵測量能暴衝

    # ── [Tier 1] 盤口微結構（解讀多空博弈意圖）──────────────────────────
    # 讓模型看到 K 線的「潛台詞」
    f["upper_wick_ratio"]  = f["upper_wick"] / (f["hl_range"] + 1e-8)   # 上檔賣壓
    f["delta_upper_wick"]  = f["upper_wick_ratio"].diff(3)               # 賣壓動態

    # volume_impulse：vol_ratio_5 的 3 日 Z-score（量化「量能異於平常」的程度）
    vr5_roll_mean  = f["vol_ratio_5"].rolling(3).mean()
    vr5_roll_std   = f["vol_ratio_5"].rolling(3).std()
    f["volume_impulse"] = (f["vol_ratio_5"] - vr5_roll_mean) / (vr5_roll_std + 1e-8)

    # ── 清理極端值（soft boundary：10 * tanh(x/10)，防硬邊界梯度失真）───
    for col in f.columns:
        f[col] = f[col].replace([np.inf, -np.inf], np.nan)
        if col.startswith("ret_"):
            f[col] = f[col].clip(-0.2, 0.2)
        f[col] = f[col].fillna(0)
        # 廢除 clip(-10,10)，改為 10 * tanh(x/10)
        # 保留 LayerNorm 的內部歸一化能力，且梯度不在邊界截斷
        f[col] = 10.0 * np.tanh(f[col].values / 10.0)

    # pos_252 移除後只需等 60 天（最長 window），dropna 自動處理
    return f.dropna()


# ── align_features ────────────────────────────────────────────────────────────

@register(
    module="Proc",
    inputs={"stocks": "dict[str, pd.DataFrame]"},
    outputs={
        "feat_dfs":     "dict[str, pd.DataFrame]",
        "prices_dict":  "dict[str, np.ndarray]",
        "volumes_dict": "dict[str, np.ndarray]",
        "feat_names":   "list[str]",
        "dates":        "list[str]",
    },
    notes="對齊共同交易日 + 計算跨股票排名特徵（rank_*），0050 rank 固定為 0",
)
def align_features(stocks: dict) -> tuple:
    """
    對齊所有股票的特徵，取共同日期，計算跨股票相對排名特徵。

    流程：
      1. compute_features(每支股票) → 各股 29 個基礎特徵（含排名佔位符）
      2. 取共同日期（dropna 後的交集，自然排除 NaN 前的資料）
      3. 對每個交易日，在 9 支可交易股票之間計算相對排名
      4. 正規化到 [-1, 1] 並寫回各股 DataFrame
      5. 0050 的 rank_* 特徵保持 0（中性，不參與排名競爭）

    注意：
      - Volume 不做任何插值，停牌日已在 loader 過濾
      - rank_* 特徵在對齊後計算，確保每天都有完整的 9 支股票參與排名

    回傳：
      feat_dfs     : {sid: DataFrame}  32 個特徵，已對齊
      prices_dict  : {sid: np.ndarray}
      volumes_dict : {sid: np.ndarray}
      feat_names   : list[str]
      dates        : list[str]
    """
    from configs.trading_config import TRADEABLE_STOCKS, BENCHMARK_STOCK

    feat_dfs = {sid: compute_features(df) for sid, df in stocks.items()}

    # 取共同日期
    common_idx = None
    for df in feat_dfs.values():
        common_idx = (df.index if common_idx is None
                      else common_idx.intersection(df.index))

    feat_dfs = {sid: feat_dfs[sid].loc[common_idx].copy() for sid in feat_dfs}

    # rank_* 特徵已永久移除（ID 依賴，破壞 Shared MLP 的股票無差別性）

    # ── Cross-sectional Z-score（核心）──────────────────────────────────────
    # 對「每天、每個特徵」跨股票標準化：z_{i,f} = (x_{i,f} - μ_f) / (σ_f + ε)
    # 建立相對座標系，讓模型知道「這支股票比今天其他股票強多少」
    feat_cols = feat_dfs[list(feat_dfs.keys())[0]].columns.tolist()
    all_sids  = list(feat_dfs.keys())

    for col in feat_cols:
        mat = np.array([feat_dfs[sid][col].values for sid in all_sids],
                       dtype=np.float64)              # (n_stocks, n_dates)
        mu  = mat.mean(axis=0, keepdims=True)         # (1, n_dates)
        sig = mat.std(axis=0, keepdims=True)          # (1, n_dates)
        z   = (mat - mu) / (sig + 1e-8)              # (n_stocks, n_dates)
        # soft clip：10 * tanh(z/3)，限制在 ±10 內，保留梯度
        for j, sid in enumerate(all_sids):
            feat_dfs[sid][col] = 10.0 * np.tanh(z[j] / 3.0)

    # ── 收盤價與成交量 ────────────────────────────────────────────────────
    prices_dict = {
        sid: stocks[sid]["Close"].loc[common_idx].values.flatten()
        for sid in feat_dfs
    }
    volumes_dict = {
        sid: stocks[sid]["Volume"].loc[common_idx].values.flatten().astype(np.float64)
        for sid in feat_dfs
    }

    feat_names = feat_dfs[list(feat_dfs.keys())[0]].columns.tolist()
    dates      = feat_dfs[list(feat_dfs.keys())[0]].index.strftime("%Y-%m-%d").tolist()

    print(f"特徵數量：{len(feat_names)}，共同交易日：{len(common_idx)}")
    assert len(feat_names) == 35, f"特徵數量錯誤：{len(feat_names)}，預期 35（Tier 1：27基礎+8新增）"

    for sid, vol_arr in volumes_dict.items():
        zero_days = (vol_arr == 0).sum()
        if zero_days > 0:
            print(f"  警告：{sid} 有 {zero_days} 個交易日成交量為 0")

    return feat_dfs, prices_dict, volumes_dict, feat_names, dates


# ── scale_features ────────────────────────────────────────────────────────────

@register(
    module="Proc",
    inputs={
        "feat_dict":   "dict[str, pd.DataFrame]",
        "scaler_dict": "dict | None",
    },
    outputs={
        "scaled_dict": "dict[str, np.ndarray]",
        "scalers":     "dict[str, StandardScaler]",
    },
    notes="StandardScaler 標準化 + clip(-5, 5)；scaler_dict=None 時 fit，否則 transform",
)
def scale_features(feat_dict: dict, scaler_dict: dict = None) -> tuple:
    scaled_dict = {}
    new_scalers = {}

    for sid, df in feat_dict.items():
        feat = df.values.copy().astype(np.float64)
        feat = np.where(np.isposinf(feat),  1e6, feat)
        feat = np.where(np.isneginf(feat), -1e6, feat)
        feat = np.where(np.isnan(feat),      0.0, feat)

        if scaler_dict and sid in scaler_dict:
            scaler = scaler_dict[sid]
            scaled = scaler.transform(feat)
        else:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(feat)
            new_scalers[sid] = scaler

        scaled_dict[sid] = np.clip(scaled, -5.0, 5.0)

    return scaled_dict, (scaler_dict if scaler_dict else new_scalers)