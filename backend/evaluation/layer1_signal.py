"""
evaluation/layer1_signal.py
============================
第一層：預測訊號檢驗（入場券）

對候選新特徵計算 Rank IC 與方向準確率。
這一層只是篩選，不作為最終有效性的證明。

最低門檻（兩項必須同時滿足）：
  - Rank IC > 0.02（5 日前瞻）且 > 0.015（20 日前瞻）
  - 方向準確率 > 52%（5 日）且 > 51%（20 日）

未通過：直接否決，不進入第二層。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ── 最低門檻 ──────────────────────────────────────────────────────────────────
RANK_IC_MIN_5D    = 0.02
RANK_IC_MIN_20D   = 0.015
DIR_ACC_MIN_5D    = 0.52
DIR_ACC_MIN_20D   = 0.51


@dataclass
class Layer1Result:
    passed: bool

    # Rank IC
    rank_ic_5d:  float = 0.0
    rank_ic_20d: float = 0.0
    ic_ir_5d:    float = 0.0   # IC / std(IC)，衡量穩定性
    ic_ir_20d:   float = 0.0

    # 方向準確率
    dir_acc_5d:  float = 0.0
    dir_acc_20d: float = 0.0

    # 每月 IC（用於第三層的穩定性分析）
    monthly_ic_5d:  dict = field(default_factory=dict)

    # 否決原因
    rejection_reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        lines = [
            f"Layer 1 [{status}]",
            f"  Rank IC  (5d): {self.rank_ic_5d:+.4f}  (門檻 > {RANK_IC_MIN_5D})"
            + ("  ✓" if self.rank_ic_5d > RANK_IC_MIN_5D else "  ✗"),
            f"  Rank IC (20d): {self.rank_ic_20d:+.4f}  (門檻 > {RANK_IC_MIN_20D})"
            + ("  ✓" if self.rank_ic_20d > RANK_IC_MIN_20D else "  ✗"),
            f"  IC IR   (5d):  {self.ic_ir_5d:.4f}",
            f"  Dir Acc  (5d): {self.dir_acc_5d:.2%}  (門檻 > {DIR_ACC_MIN_5D:.0%})"
            + ("  ✓" if self.dir_acc_5d > DIR_ACC_MIN_5D else "  ✗"),
            f"  Dir Acc (20d): {self.dir_acc_20d:.2%}  (門檻 > {DIR_ACC_MIN_20D:.0%})"
            + ("  ✓" if self.dir_acc_20d > DIR_ACC_MIN_20D else "  ✗"),
        ]
        if self.rejection_reasons:
            lines.append("  否決原因：" + "；".join(self.rejection_reasons))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "passed":       self.passed,
            "rank_ic_5d":   round(self.rank_ic_5d,  4),
            "rank_ic_20d":  round(self.rank_ic_20d, 4),
            "ic_ir_5d":     round(self.ic_ir_5d,    4),
            "ic_ir_20d":    round(self.ic_ir_20d,   4),
            "dir_acc_5d":   round(self.dir_acc_5d,  4),
            "dir_acc_20d":  round(self.dir_acc_20d, 4),
            "monthly_ic_5d":  {k: round(v, 4) for k, v in self.monthly_ic_5d.items()},
            "rejection_reasons": self.rejection_reasons,
            "thresholds": {
                "rank_ic_5d":  RANK_IC_MIN_5D,
                "rank_ic_20d": RANK_IC_MIN_20D,
                "dir_acc_5d":  DIR_ACC_MIN_5D,
                "dir_acc_20d": DIR_ACC_MIN_20D,
            },
        }


def run_layer1(
    candidate_features: dict[str, pd.DataFrame],
    prices_dict: dict[str, np.ndarray],
    dates: list[str],
    tradeable_ids: list[str],
    val_start: str,
    val_end: str,
    feature_column: str | None = None,
) -> Layer1Result:
    """
    執行第一層：預測訊號檢驗。

    Args:
        candidate_features: {sid: DataFrame}，候選新特徵資料（完整期間）
        prices_dict:        {sid: np.ndarray}，價格序列
        dates:              所有日期列表
        tradeable_ids:      可交易股票代碼列表
        val_start / val_end: 驗證集區間（字串，格式 YYYY-MM-DD）
        feature_column:     要測試的特徵欄位名稱（None 時取第一欄）

    Returns:
        Layer1Result
    """
    dates_arr = np.array(dates)
    mask = (dates_arr >= val_start) & (dates_arr <= val_end)
    val_idx = np.where(mask)[0]

    if len(val_idx) < 30:
        return Layer1Result(
            passed=False,
            rejection_reasons=[f"驗證集不足 30 天（實際 {len(val_idx)} 天）"],
        )

    # 決定要測試的特徵欄位
    first_sid = tradeable_ids[0]
    if feature_column is None:
        feature_column = candidate_features[first_sid].columns[0]

    # 建立特徵矩陣和報酬矩陣 (n_stocks, n_val_days)
    feat_mat = np.full((len(tradeable_ids), len(val_idx)), np.nan)
    ret5_mat = np.full_like(feat_mat, np.nan)
    ret20_mat = np.full_like(feat_mat, np.nan)

    for j, sid in enumerate(tradeable_ids):
        if sid not in candidate_features or sid not in prices_dict:
            continue
        df    = candidate_features[sid]
        price = prices_dict[sid]

        for k, t in enumerate(val_idx):
            if t >= len(df):
                continue
            # 特徵值
            row = df.iloc[t]
            if feature_column in df.columns:
                feat_mat[j, k] = float(row[feature_column])

            # 未來 5 日報酬（look-ahead，僅用於評估，不用於訓練）
            if t + 5 < len(price):
                ret5_mat[j, k] = float(price[t + 5] / (price[t] + 1e-8) - 1)
            # 未來 20 日報酬
            if t + 20 < len(price):
                ret20_mat[j, k] = float(price[t + 20] / (price[t] + 1e-8) - 1)

    # 計算每日 Rank IC
    daily_ic_5d  = []
    daily_ic_20d = []
    daily_dir_5d = []
    daily_dir_20d = []
    monthly_ic_5d: dict[str, list[float]] = {}

    for k in range(len(val_idx)):
        f  = feat_mat[:, k]
        r5 = ret5_mat[:, k]
        r20 = ret20_mat[:, k]

        valid5  = ~(np.isnan(f) | np.isnan(r5))
        valid20 = ~(np.isnan(f) | np.isnan(r20))

        if valid5.sum() >= 5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic5, _ = spearmanr(f[valid5], r5[valid5])
            if not np.isnan(ic5):
                daily_ic_5d.append(ic5)
                dir_acc = float(((f[valid5] > 0) == (r5[valid5] > 0)).mean())
                daily_dir_5d.append(dir_acc)

                # 每月統計
                month_key = dates_arr[val_idx[k]][:7]
                monthly_ic_5d.setdefault(month_key, []).append(ic5)

        if valid20.sum() >= 5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic20, _ = spearmanr(f[valid20], r20[valid20])
            if not np.isnan(ic20):
                daily_ic_20d.append(ic20)
                dir_acc20 = float(((f[valid20] > 0) == (r20[valid20] > 0)).mean())
                daily_dir_20d.append(dir_acc20)

    if not daily_ic_5d:
        return Layer1Result(
            passed=False,
            rejection_reasons=["有效樣本不足，無法計算 Rank IC"],
        )

    rank_ic_5d  = float(np.mean(daily_ic_5d))
    rank_ic_20d = float(np.mean(daily_ic_20d)) if daily_ic_20d else 0.0
    ic_ir_5d    = float(np.mean(daily_ic_5d) / (np.std(daily_ic_5d) + 1e-8))
    ic_ir_20d   = float(np.mean(daily_ic_20d) / (np.std(daily_ic_20d) + 1e-8)) if daily_ic_20d else 0.0
    dir_acc_5d  = float(np.mean(daily_dir_5d))  if daily_dir_5d  else 0.0
    dir_acc_20d = float(np.mean(daily_dir_20d)) if daily_dir_20d else 0.0

    monthly_ic_5d_avg = {k: float(np.mean(v)) for k, v in monthly_ic_5d.items()}

    # 判斷是否通過
    reasons = []
    if rank_ic_5d <= RANK_IC_MIN_5D:
        reasons.append(f"Rank IC(5d)={rank_ic_5d:.4f} ≤ {RANK_IC_MIN_5D}")
    if rank_ic_20d <= RANK_IC_MIN_20D:
        reasons.append(f"Rank IC(20d)={rank_ic_20d:.4f} ≤ {RANK_IC_MIN_20D}")
    if dir_acc_5d <= DIR_ACC_MIN_5D:
        reasons.append(f"Dir Acc(5d)={dir_acc_5d:.2%} ≤ {DIR_ACC_MIN_5D:.0%}")
    if dir_acc_20d <= DIR_ACC_MIN_20D:
        reasons.append(f"Dir Acc(20d)={dir_acc_20d:.2%} ≤ {DIR_ACC_MIN_20D:.0%}")

    passed = len(reasons) == 0

    return Layer1Result(
        passed=passed,
        rank_ic_5d=rank_ic_5d,
        rank_ic_20d=rank_ic_20d,
        ic_ir_5d=ic_ir_5d,
        ic_ir_20d=ic_ir_20d,
        dir_acc_5d=dir_acc_5d,
        dir_acc_20d=dir_acc_20d,
        monthly_ic_5d=monthly_ic_5d_avg,
        rejection_reasons=reasons,
    )