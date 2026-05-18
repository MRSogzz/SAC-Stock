"""
evaluation/extra_diagnostics.py
================================
指定特徵的附加診斷邏輯。

每個診斷函數接受 Layer 2 / Layer 3 的中間產物（weights、curves、IC序列等），
回傳 ExtraDiagnosticResult。

診斷類型：
  low_vol_exposure   — trend_efficiency_20：低波動資產曝險（Layer 2）
  crisis_attribution — vol_regime_shift：危機歸因（Layer 3）
  turnover_defense   — ret5_vol20_ratio / volume_impulse_vol20：換倉防禦（Layer 2）
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ExtraDiagnosticResult:
    triggered: bool          # 是否觸發紅線
    triggered_rules: list[str] = field(default_factory=list)   # 觸發的具體紅線
    metrics: dict            = field(default_factory=dict)      # 量化數值（供前端顯示）
    verdict_override: str | None = None   # "CONFLICT" / "FAIL" / None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 低波動資產曝險診斷（trend_efficiency_20 → Layer 2）
# ═══════════════════════════════════════════════════════════════════════════════

# 三條紅線門檻
LOW_VOL_WEIGHT_SHIFT_THRESHOLD = 0.05   # 低波動組平均權重增加 > 5%
LOW_VOL_FACTOR_BETA_T          = 2.0    # 低波動因子 beta t 統計量 > 2.0
LOW_VOL_BUY_RATIO_THRESHOLD    = 0.60   # 買入中屬於低波動組比例 > 60%
LOW_VOL_TOP_N                  = 3      # 低波動組：vol_60 最低的 N 檔


def diagnose_low_vol_exposure(
    baseline_weights:  list[np.ndarray],   # 每日基準策略 weights，shape (n_val, n_tradeable)
    probe_weights:     list[np.ndarray],   # 每日探針策略 weights，shape (n_val, n_tradeable)
    prices_dict:       dict,               # {sid: np.ndarray}
    tradeable_ids:     list[str],
    val_idx:           np.ndarray,
) -> ExtraDiagnosticResult:
    """
    三條紅線觸發任一即判定 CONFLICT。
    """
    n_days     = len(val_idx)
    n_stocks   = len(tradeable_ids)
    triggered_rules = []

    if n_days < 10 or len(baseline_weights) == 0:
        return ExtraDiagnosticResult(triggered=False, metrics={"insufficient_data": True})

    baseline_w = np.array(baseline_weights)   # (n_days, n_stocks)
    probe_w    = np.array(probe_weights)      # (n_days, n_stocks)

    # ── 計算每日低波動組（vol_60 最低的 TOP_N 檔）────────────────────────────
    low_vol_mask = np.zeros((n_days, n_stocks), dtype=bool)
    for k, t in enumerate(val_idx):
        vols = []
        for j, sid in enumerate(tradeable_ids):
            p = prices_dict[sid]
            start = max(0, int(t) - 60)
            ret   = np.diff(np.log(p[start:int(t) + 1] + 1e-8))
            vols.append(float(np.std(ret)) if len(ret) > 1 else 0.0)
        vols = np.array(vols)
        top_idx = np.argsort(vols)[:LOW_VOL_TOP_N]
        low_vol_mask[k, top_idx] = True

    # ── 紅線 1：低波動組平均總權重差異 ──────────────────────────────────────
    base_low_w  = (baseline_w * low_vol_mask).sum(axis=1).mean()
    probe_low_w = (probe_w   * low_vol_mask).sum(axis=1).mean()
    weight_shift = float(probe_low_w - base_low_w)

    if weight_shift > LOW_VOL_WEIGHT_SHIFT_THRESHOLD:
        triggered_rules.append(
            f"紅線1：低波動組平均權重增加 {weight_shift:.1%}（門檻 > {LOW_VOL_WEIGHT_SHIFT_THRESHOLD:.0%}）"
        )

    # ── 紅線 2：低波動因子 Beta t 統計量 ────────────────────────────────────
    # 超額報酬序列
    excess_ret = np.zeros(n_days)
    for k, t in enumerate(val_idx):
        if int(t) + 1 >= len(prices_dict[tradeable_ids[0]]):
            continue
        p_t1 = np.array([prices_dict[sid][int(t) + 1] for sid in tradeable_ids])
        p_t  = np.array([prices_dict[sid][int(t)]     for sid in tradeable_ids])
        daily_ret  = p_t1 / (p_t + 1e-8) - 1
        base_r     = float((baseline_w[k] * daily_ret).sum())
        probe_r    = float((probe_w[k]    * daily_ret).sum())
        excess_ret[k] = probe_r - base_r

    # 低波動因子日報酬（低波動組等權 - 高波動組等權）
    factor_ret = np.zeros(n_days)
    for k, t in enumerate(val_idx):
        if int(t) + 1 >= len(prices_dict[tradeable_ids[0]]):
            continue
        p_t1 = np.array([prices_dict[sid][int(t) + 1] for sid in tradeable_ids])
        p_t  = np.array([prices_dict[sid][int(t)]     for sid in tradeable_ids])
        daily_ret = p_t1 / (p_t + 1e-8) - 1
        low_mask  = low_vol_mask[k]
        high_mask = ~low_mask
        low_r  = daily_ret[low_mask].mean()  if low_mask.any()  else 0.0
        high_r = daily_ret[high_mask].mean() if high_mask.any() else 0.0
        factor_ret[k] = float(low_r - high_r)

    # OLS beta
    valid = ~(np.isnan(excess_ret) | np.isnan(factor_ret))
    beta_t = 0.0
    beta   = 0.0
    if valid.sum() > 10:
        X = factor_ret[valid]
        Y = excess_ret[valid]
        X_dm = X - X.mean()
        denom = (X_dm ** 2).sum()
        if denom > 1e-10:
            beta    = float((X_dm * (Y - Y.mean())).sum() / denom)
            resid   = Y - (beta * X + (Y.mean() - beta * X.mean()))
            se_beta = float(np.sqrt((resid ** 2).sum() / max(valid.sum() - 2, 1) / denom))
            beta_t  = float(beta / (se_beta + 1e-10))

    if beta > 0 and beta_t > LOW_VOL_FACTOR_BETA_T:
        triggered_rules.append(
            f"紅線2：低波動因子 Beta={beta:.4f}，t={beta_t:.2f}（門檻 Beta>0 且 t>{LOW_VOL_FACTOR_BETA_T}）"
        )

    # ── 紅線 3：買入交易中低波動組比例 ───────────────────────────────────────
    buy_total    = 0
    buy_low_vol  = 0
    for k in range(n_days):
        w_diff = probe_w[k] - baseline_w[k]
        for j in range(n_stocks):
            if w_diff[j] > 0.01:   # 視為買入
                buy_total += 1
                if low_vol_mask[k, j]:
                    buy_low_vol += 1

    buy_low_ratio = buy_low_vol / buy_total if buy_total > 0 else 0.0
    if buy_low_ratio > LOW_VOL_BUY_RATIO_THRESHOLD:
        triggered_rules.append(
            f"紅線3：買入中低波動組比例 {buy_low_ratio:.1%}（門檻 > {LOW_VOL_BUY_RATIO_THRESHOLD:.0%}）"
        )

    triggered = len(triggered_rules) > 0
    return ExtraDiagnosticResult(
        triggered        = triggered,
        triggered_rules  = triggered_rules,
        verdict_override = "CONFLICT" if triggered else None,
        metrics = {
            "weight_shift":    round(weight_shift,   4),
            "base_low_w":      round(base_low_w,     4),
            "probe_low_w":     round(probe_low_w,    4),
            "factor_beta":     round(beta,            4),
            "factor_beta_t":   round(beta_t,          4),
            "buy_low_ratio":   round(buy_low_ratio,   4),
            "buy_total":       buy_total,
            "buy_low_vol":     buy_low_vol,
            "thresholds": {
                "weight_shift":    LOW_VOL_WEIGHT_SHIFT_THRESHOLD,
                "beta_t":          LOW_VOL_FACTOR_BETA_T,
                "buy_low_ratio":   LOW_VOL_BUY_RATIO_THRESHOLD,
            },
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 危機歸因診斷（vol_regime_shift → Layer 3）
# ═══════════════════════════════════════════════════════════════════════════════

CRISIS_CONCENTRATION_THRESHOLD = 0.50   # Top3月份貢獻 > 50%
CRISIS_HIGH_VOL_RATIO_THRESHOLD = 0.40  # 高風險月份貢獻 > 40%
CRISIS_NORMAL_SHARPE_THRESHOLD  = 0.0   # 正常期 ΔSharpe ≤ 0
HIGH_VOL_PCTILE                 = 0.90  # 高風險月份定義：vol_20 最高 10%


def diagnose_crisis_attribution(
    baseline_curve:    np.ndarray,    # 基準資金曲線
    candidate_curve:   np.ndarray,    # 候選資金曲線
    val_idx:           np.ndarray,
    dates:             list[str],
    prices_dict:       dict,
    benchmark_sid:     str,           # 0050
) -> ExtraDiagnosticResult:
    """
    三條紅線觸發任一即判定 FAIL。
    """
    triggered_rules = []
    n_days = len(val_idx)

    if n_days < 20:
        return ExtraDiagnosticResult(triggered=False, metrics={"insufficient_data": True})

    # 每日超額報酬序列
    b_daily = np.diff(baseline_curve)  / (baseline_curve[:-1]  + 1e-8)
    c_daily = np.diff(candidate_curve) / (candidate_curve[:-1] + 1e-8)
    excess  = c_daily - b_daily   # shape: (n_days,)

    # 按月分組
    dates_arr = np.array(dates)
    val_dates = dates_arr[val_idx]
    months    = [d[:7] for d in val_dates]
    unique_months = sorted(set(months))

    month_excess = {}
    for m in unique_months:
        idx = [i for i, d in enumerate(months) if d == m]
        # excess 比 val_idx 少一個（diff），處理邊界
        ex_idx = [i for i in idx if i < len(excess)]
        month_excess[m] = float(np.sum(excess[ex_idx])) if ex_idx else 0.0

    total_excess = sum(month_excess.values())

    # ── 紅線 1：貢獻集中度（Top 3 個月） ─────────────────────────────────────
    sorted_months = sorted(month_excess.items(), key=lambda x: abs(x[1]), reverse=True)
    top3_excess   = sum(abs(v) for _, v in sorted_months[:3])
    total_abs     = sum(abs(v) for v in month_excess.values()) + 1e-10
    concentration = top3_excess / total_abs

    if concentration > CRISIS_CONCENTRATION_THRESHOLD:
        triggered_rules.append(
            f"紅線1：Top3月份貢獻集中度 {concentration:.1%}（門檻 > {CRISIS_CONCENTRATION_THRESHOLD:.0%}）"
            f"  Top3: {[m for m, _ in sorted_months[:3]]}"
        )

    # ── 紅線 2：高風險月份貢獻比例 ───────────────────────────────────────────
    # 計算 benchmark 每月 vol_20
    bm_prices = prices_dict.get(benchmark_sid, list(prices_dict.values())[0])
    month_vol = {}
    for m in unique_months:
        idx = [i for i, d in enumerate(months) if d == m]
        t_vals = val_idx[idx]
        vols   = []
        for t in t_vals:
            start = max(0, int(t) - 20)
            ret   = np.diff(np.log(bm_prices[start:int(t) + 1] + 1e-8))
            if len(ret) > 1:
                vols.append(float(np.std(ret)))
        month_vol[m] = float(np.mean(vols)) if vols else 0.0

    vol_threshold  = np.percentile(list(month_vol.values()), HIGH_VOL_PCTILE * 100)
    high_vol_months = {m for m, v in month_vol.items() if v >= vol_threshold}

    if high_vol_months and total_abs > 1e-10:
        high_vol_excess = sum(abs(month_excess.get(m, 0)) for m in high_vol_months)
        high_vol_ratio  = high_vol_excess / total_abs
        if high_vol_ratio > CRISIS_HIGH_VOL_RATIO_THRESHOLD:
            triggered_rules.append(
                f"紅線2：高風險月份超額報酬佔比 {high_vol_ratio:.1%}（門檻 > {CRISIS_HIGH_VOL_RATIO_THRESHOLD:.0%}）"
                f"  高風險月份: {sorted(high_vol_months)}"
            )
    else:
        high_vol_ratio = 0.0

    # ── 紅線 3：正常期 ΔSharpe ───────────────────────────────────────────────
    normal_idx = [i for i, d in enumerate(months) if d[:7] not in high_vol_months and i < len(excess)]
    if len(normal_idx) >= 10:
        normal_excess = excess[normal_idx]
        normal_sharpe = float(
            np.mean(normal_excess) / (np.std(normal_excess) + 1e-8) * np.sqrt(252)
        )
        if normal_sharpe <= CRISIS_NORMAL_SHARPE_THRESHOLD:
            triggered_rules.append(
                f"紅線3：正常市場期 ΔSharpe={normal_sharpe:.4f} ≤ 0（特徵有效性依賴危機環境）"
            )
    else:
        normal_sharpe = None

    triggered = len(triggered_rules) > 0
    return ExtraDiagnosticResult(
        triggered        = triggered,
        triggered_rules  = triggered_rules,
        verdict_override = "FAIL" if triggered else None,
        metrics = {
            "month_excess":     {m: round(v, 6) for m, v in month_excess.items()},
            "concentration":    round(concentration, 4),
            "top3_months":      [m for m, _ in sorted_months[:3]],
            "high_vol_months":  sorted(high_vol_months),
            "high_vol_ratio":   round(high_vol_ratio, 4) if high_vol_months else 0.0,
            "normal_sharpe":    round(normal_sharpe, 4) if normal_sharpe is not None else None,
            "total_excess":     round(total_excess, 6),
            "thresholds": {
                "concentration":   CRISIS_CONCENTRATION_THRESHOLD,
                "high_vol_ratio":  CRISIS_HIGH_VOL_RATIO_THRESHOLD,
                "normal_sharpe":   CRISIS_NORMAL_SHARPE_THRESHOLD,
            },
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 換倉防禦診斷（ret5_vol20_ratio / volume_impulse_vol20 → Layer 2）
# ═══════════════════════════════════════════════════════════════════════════════

TURNOVER_INCREASE_THRESHOLD   = 0.10    # ΔTurnover > 10% → 觸發
DEFENSE_CORR_THRESHOLD        = -0.10   # 下跌日超額報酬與大盤相關性 > -0.1 代表未能防禦


def diagnose_turnover_defense(
    baseline_weights:  list[np.ndarray],
    probe_weights:     list[np.ndarray],
    baseline_curve:    np.ndarray,
    candidate_curve:   np.ndarray,
    prices_dict:       dict,
    tradeable_ids:     list[str],
    benchmark_sid:     str,
    val_idx:           np.ndarray,
) -> ExtraDiagnosticResult:
    """
    ΔTurnover 顯著上升或下跌時未能防禦即 FAIL。
    """
    triggered_rules = []
    n_days = min(len(baseline_weights), len(probe_weights))

    if n_days < 10:
        return ExtraDiagnosticResult(triggered=False, metrics={"insufficient_data": True})

    bw = np.array(baseline_weights[:n_days])
    pw = np.array(probe_weights[:n_days])

    # ── 換倉率差異 ──────────────────────────────────────────────────────────
    base_to   = np.mean([np.abs(bw[k] - bw[k-1]).sum() / 2 for k in range(1, n_days)])
    probe_to  = np.mean([np.abs(pw[k] - pw[k-1]).sum() / 2 for k in range(1, n_days)])
    delta_to  = float(probe_to - base_to)

    if delta_to > TURNOVER_INCREASE_THRESHOLD:
        triggered_rules.append(
            f"紅線1：ΔTurnover={delta_to:.2%} 顯著上升（門檻 > {TURNOVER_INCREASE_THRESHOLD:.0%}）"
        )

    # ── 下跌日防禦性修正 ─────────────────────────────────────────────────────
    bm_prices = prices_dict.get(benchmark_sid, list(prices_dict.values())[0])
    down_days = []
    for k, t in enumerate(val_idx):
        if int(t) + 1 >= len(bm_prices) or int(t) == 0:
            continue
        bm_ret = float(bm_prices[int(t)] / (bm_prices[int(t) - 1] + 1e-8) - 1)
        if bm_ret < -0.005:   # 大盤下跌超過 0.5%
            down_days.append(k)

    defense_corr = None
    if len(down_days) >= 5:
        b_excess_down = []
        bm_ret_down   = []
        for k in down_days:
            if k >= len(baseline_curve) - 1 or k >= len(candidate_curve) - 1:
                continue
            b_r  = float((baseline_curve[k + 1] - baseline_curve[k]) / (baseline_curve[k] + 1e-8))
            c_r  = float((candidate_curve[k + 1] - candidate_curve[k]) / (candidate_curve[k] + 1e-8))
            t    = val_idx[k]
            bm_r = float(bm_prices[int(t) + 1] / (bm_prices[int(t)] + 1e-8) - 1) if int(t) + 1 < len(bm_prices) else 0.0
            b_excess_down.append(c_r - b_r)
            bm_ret_down.append(bm_r)

        if len(b_excess_down) >= 5:
            x = np.array(bm_ret_down)
            y = np.array(b_excess_down)
            if x.std() > 1e-8 and y.std() > 1e-8:
                defense_corr = float(np.corrcoef(x, y)[0, 1])
                if defense_corr > DEFENSE_CORR_THRESHOLD:
                    triggered_rules.append(
                        f"紅線2：下跌日超額報酬與大盤相關係數={defense_corr:.3f}（> {DEFENSE_CORR_THRESHOLD}，未能提供防禦性修正）"
                    )

    triggered = len(triggered_rules) > 0
    return ExtraDiagnosticResult(
        triggered        = triggered,
        triggered_rules  = triggered_rules,
        verdict_override = "FAIL" if triggered else None,
        metrics = {
            "base_turnover":  round(float(base_to),   4),
            "probe_turnover": round(float(probe_to),  4),
            "delta_turnover": round(delta_to,          4),
            "down_days":      len(down_days),
            "defense_corr":   round(defense_corr, 4) if defense_corr is not None else None,
            "thresholds": {
                "turnover_increase": TURNOVER_INCREASE_THRESHOLD,
                "defense_corr":      DEFENSE_CORR_THRESHOLD,
            },
        },
    )