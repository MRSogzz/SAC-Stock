"""
evaluation/layer3_robustness.py
================================
第三層：反事實穩定性與失敗模式檢查（最終否決閥）

功能：
  1. 市場區間切片：測試增益在各波動率/趨勢環境中是否穩定
  2. 失敗模式檢查：HHI 爆炸、尾部風險、換倉率異常
  3. 外部基準錨定：對比「等權重配置」和「簡單動能策略」

判決：若增益無法顯著超越簡單基準，或引發新風險模式，則新因子無效。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


# ── 失敗模式門檻 ──────────────────────────────────────────────────────────────
HHI_EXPLOSION_THRESHOLD  = 0.75   # 超過此值視為 HHI 爆炸
TAIL_RISK_PCTILE         = 0.05   # 最差 5% 日報酬
MAX_HHI_EXPLOSION_DAYS   = 0.15   # 不超過 15% 的天數觸發 HHI 爆炸
MIN_REGIME_SEGMENTS      = 2      # 至少在 2 個市場環境中有正增益才算穩定

# ── 外部基準：等權重策略 ──────────────────────────────────────────────────────
def _equal_weight_sharpe(
    prices_dict: dict,
    tradeable_ids: list[str],
    val_idx: np.ndarray,
) -> float:
    """等權重買入持有的 Sharpe（非學習基準）。"""
    n = len(tradeable_ids)
    curve = []
    for t in val_idx:
        avg_ret = np.mean([
            prices_dict[sid][t] / (prices_dict[sid][val_idx[0]] + 1e-8) - 1
            for sid in tradeable_ids
            if t < len(prices_dict[sid])
        ])
        curve.append(1.0 + avg_ret)
    if len(curve) < 2:
        return 0.0
    daily = np.diff(np.array(curve))
    return float(np.mean(daily) / (np.std(daily) + 1e-8) * np.sqrt(252))


def _momentum_sharpe(
    prices_dict: dict,
    tradeable_ids: list[str],
    val_idx: np.ndarray,
    momentum_window: int = 20,
) -> float:
    """簡單動能策略的 Sharpe：每月選過去 20 日報酬最高的前 3 支等權重持有。"""
    curve = [1.0]
    current_portfolio = tradeable_ids[:3]   # 初始持倉

    for k, t in enumerate(val_idx):
        if k > 0 and k % momentum_window == 0:
            # 重新排序
            scores = []
            for sid in tradeable_ids:
                start_t = max(0, t - momentum_window)
                if t < len(prices_dict[sid]) and start_t < len(prices_dict[sid]):
                    ret = prices_dict[sid][t] / (prices_dict[sid][start_t] + 1e-8) - 1
                    scores.append((sid, ret))
            scores.sort(key=lambda x: -x[1])
            current_portfolio = [s[0] for s in scores[:3]]

        if t + 1 < min(len(prices_dict[sid]) for sid in tradeable_ids):
            avg_ret = np.mean([
                prices_dict[sid][t + 1] / (prices_dict[sid][t] + 1e-8) - 1
                for sid in current_portfolio
                if t < len(prices_dict[sid]) and t + 1 < len(prices_dict[sid])
            ])
            curve.append(curve[-1] * (1 + avg_ret))

    if len(curve) < 2:
        return 0.0
    daily = np.diff(np.array(curve))
    return float(np.mean(daily) / (np.std(daily) + 1e-8) * np.sqrt(252))


@dataclass
class Layer3Result:
    passed: bool

    # 市場區間切片
    regime_results: dict = field(default_factory=dict)
    stable_regimes: int  = 0   # 有正增益的區間數

    # 失敗模式
    hhi_explosion_rate: float = 0.0
    tail_risk_baseline: float = 0.0
    tail_risk_candidate: float = 0.0
    delta_tail_risk:    float = 0.0

    # 外部基準
    candidate_sharpe:   float = 0.0
    equal_weight_sharpe: float = 0.0
    momentum_sharpe:    float = 0.0
    beats_equal_weight: bool  = False
    beats_momentum:     bool  = False

    rejection_reasons: list[str] = field(default_factory=list)

    # 附加診斷結果（僅特定特徵啟用）
    extra_diagnostic: dict | None = None

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        lines = [
            f"Layer 3 [{status}]",
            f"  市場區間穩定性：{self.stable_regimes}/{len(self.regime_results)} 個區間有正增益",
            f"  HHI 爆炸率：{self.hhi_explosion_rate:.1%}"
            + ("  ✓" if self.hhi_explosion_rate < MAX_HHI_EXPLOSION_DAYS else f"  ✗  (> {MAX_HHI_EXPLOSION_DAYS:.0%})"),
            f"  Δ尾部風險(5th)：{self.delta_tail_risk:+.4f}"
            + ("  ✓" if self.delta_tail_risk >= -0.005 else "  ✗"),
            f"  Sharpe vs 等權重：{self.candidate_sharpe:.4f} vs {self.equal_weight_sharpe:.4f}"
            + ("  ✓" if self.beats_equal_weight else "  ✗"),
            f"  Sharpe vs 動能：  {self.candidate_sharpe:.4f} vs {self.momentum_sharpe:.4f}"
            + ("  ✓" if self.beats_momentum else "  ✗"),
        ]
        if self.regime_results:
            lines.append("  各區間 ΔSharpe：")
            for regime, res in self.regime_results.items():
                sign = "↑" if res["delta_sharpe"] > 0 else "↓"
                lines.append(f"    {regime:12s}: {res['delta_sharpe']:+.4f} {sign}"
                             f"  ({res['n_days']} 天)")
        if self.rejection_reasons:
            lines.append("  否決原因：" + "；".join(self.rejection_reasons))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "passed":              self.passed,
            "regime_results":      self.regime_results,
            "stable_regimes":      self.stable_regimes,
            "hhi_explosion_rate":  round(self.hhi_explosion_rate,  4),
            "tail_risk_baseline":  round(self.tail_risk_baseline,  4),
            "tail_risk_candidate": round(self.tail_risk_candidate, 4),
            "delta_tail_risk":     round(self.delta_tail_risk,     4),
            "candidate_sharpe":    round(self.candidate_sharpe,    4),
            "equal_weight_sharpe": round(self.equal_weight_sharpe, 4),
            "momentum_sharpe":     round(self.momentum_sharpe,     4),
            "beats_equal_weight":  self.beats_equal_weight,
            "beats_momentum":      self.beats_momentum,
            "extra_diagnostic":    self.extra_diagnostic,
            "rejection_reasons":   self.rejection_reasons,
            "thresholds": {
                "hhi_explosion_threshold":  HHI_EXPLOSION_THRESHOLD,
                "max_hhi_explosion_days":   MAX_HHI_EXPLOSION_DAYS,
                "min_stable_regimes":       MIN_REGIME_SEGMENTS,
            },
        }


def _classify_regime(
    prices: np.ndarray,
    t: int,
    vol_window: int = 20,
    trend_window: int = 60,
) -> str:
    """將單日分類為四種市場環境之一。"""
    start_vol   = max(0, t - vol_window)
    start_trend = max(0, t - trend_window)

    if t - start_vol < 5 or t - start_trend < 10:
        return "unknown"

    log_rets = np.diff(np.log(prices[start_vol:t + 1] + 1e-8))
    vol = float(np.std(log_rets)) if len(log_rets) > 1 else 0.0

    ret_60 = float(prices[t] / (prices[start_trend] + 1e-8) - 1) if t > start_trend else 0.0
    vol_60 = float(np.std(np.diff(np.log(prices[start_trend:t + 1] + 1e-8)))) \
             if t - start_trend > 5 else 0.01
    trend_eff = ret_60 / (vol_60 * np.sqrt(trend_window) + 1e-8)

    if trend_eff > 0.5:
        return "bull_trend"
    elif trend_eff < -0.5:
        return "bear_trend"
    elif vol < 0.01:
        return "low_vol_sideways"
    else:
        return "high_vol_sideways"


def run_layer3(
    layer2_result,   # Layer2Result
    prices_dict: dict,
    dates: list[str],
    tradeable_ids: list[str],
    val_start: str,
    val_end: str,
    baseline_actions: list | None = None,
    candidate_actions: list | None = None,
    extra_diagnostic: str | None = None,   # "crisis_attribution" / None
) -> Layer3Result:
    """
    執行第三層：反事實穩定性與失敗模式檢查。

    Args:
        layer2_result:     Layer2Result（包含 baseline/candidate 的 portfolio_curve）
        prices_dict:       {sid: np.ndarray}
        dates:             所有日期
        tradeable_ids:     可交易股票代碼
        val_start/val_end: 驗證集區間
        baseline_actions:  基準策略每日動作（可選，用於 HHI 計算）
        candidate_actions: 候選策略每日動作（可選）

    Returns:
        Layer3Result
    """
    dates_arr = np.array(dates)
    mask      = (dates_arr >= val_start) & (dates_arr <= val_end)
    val_idx   = np.where(mask)[0]

    if len(val_idx) < 20:
        return Layer3Result(
            passed=False,
            rejection_reasons=["驗證集不足 20 天"],
        )

    # ── 市場區間分類 ──────────────────────────────────────────────────────────
    bm_prices = prices_dict[tradeable_ids[0]]   # 用第一支股票的價格做粗略分類
    regime_days: dict[str, list[int]] = {}

    for k, t in enumerate(val_idx):
        regime = _classify_regime(bm_prices, int(t))
        regime_days.setdefault(regime, []).append(k)

    # ── 從 Layer2 的 portfolio_curve 計算各區間 Sharpe ───────────────────────
    baseline_curve  = np.array(layer2_result.baseline_metrics.portfolio_curve,
                               dtype=np.float64)
    candidate_curve = np.array(layer2_result.candidate_metrics.portfolio_curve,
                               dtype=np.float64)

    def _curve_sharpe(curve: np.ndarray, day_indices: list[int]) -> float:
        if len(day_indices) < 5 or len(curve) < 2:
            return 0.0
        idx = [min(d, len(curve) - 2) for d in day_indices]
        daily = np.diff(curve)[idx] / (curve[idx] + 1e-8)
        return float(np.mean(daily) / (np.std(daily) + 1e-8) * np.sqrt(252))

    regime_results: dict[str, dict] = {}
    stable_regimes = 0

    for regime, day_idx in regime_days.items():
        if len(day_idx) < 5:
            continue
        b_sharpe = _curve_sharpe(baseline_curve,  day_idx)
        c_sharpe = _curve_sharpe(candidate_curve, day_idx)
        ds       = c_sharpe - b_sharpe
        if ds > 0:
            stable_regimes += 1
        regime_results[regime] = {
            "n_days":         len(day_idx),
            "baseline_sharpe":  round(b_sharpe, 4),
            "candidate_sharpe": round(c_sharpe, 4),
            "delta_sharpe":     round(ds, 4),
        }

    # ── HHI 爆炸率（若有動作資料）────────────────────────────────────────────
    hhi_explosion_rate = 0.0
    if candidate_actions is not None and len(candidate_actions) > 0:
        hhi_arr = np.array([
            float((np.array(a) ** 2).sum()) for a in candidate_actions
        ])
        hhi_explosion_rate = float((hhi_arr > HHI_EXPLOSION_THRESHOLD).mean())

    # ── 尾部風險（第 5 百分位日報酬）────────────────────────────────────────
    b_daily = np.diff(baseline_curve) / (baseline_curve[:-1] + 1e-8)
    c_daily = np.diff(candidate_curve) / (candidate_curve[:-1] + 1e-8)

    tail_risk_baseline  = float(np.percentile(b_daily, TAIL_RISK_PCTILE * 100)) \
                          if len(b_daily) > 5 else 0.0
    tail_risk_candidate = float(np.percentile(c_daily, TAIL_RISK_PCTILE * 100)) \
                          if len(c_daily) > 5 else 0.0
    delta_tail_risk     = tail_risk_candidate - tail_risk_baseline

    # ── 外部基準 ──────────────────────────────────────────────────────────────
    candidate_sharpe   = layer2_result.candidate_metrics.sharpe
    equal_weight_s     = _equal_weight_sharpe(prices_dict, tradeable_ids, val_idx)
    momentum_s         = _momentum_sharpe(prices_dict, tradeable_ids, val_idx)
    beats_equal_weight = candidate_sharpe > equal_weight_s
    beats_momentum     = candidate_sharpe > momentum_s

    # ── 判決 ─────────────────────────────────────────────────────────────────
    reasons = []
    if stable_regimes < MIN_REGIME_SEGMENTS:
        reasons.append(
            f"僅 {stable_regimes}/{len(regime_results)} 個市場環境有正增益"
            f"（需 ≥ {MIN_REGIME_SEGMENTS}）"
        )
    if hhi_explosion_rate > MAX_HHI_EXPLOSION_DAYS:
        reasons.append(
            f"HHI 爆炸率 {hhi_explosion_rate:.1%} > {MAX_HHI_EXPLOSION_DAYS:.0%}"
        )
    if delta_tail_risk < -0.005:
        reasons.append(f"尾部風險惡化 {delta_tail_risk:+.4f}（5th pctile 日報酬）")
    if not beats_equal_weight:
        reasons.append(
            f"未超越等權重基準（{candidate_sharpe:.4f} vs {equal_weight_s:.4f}）"
        )

    # ── 附加診斷：危機歸因（vol_regime_shift 專屬）─────────────────────────
    extra_diag_result = None
    if extra_diagnostic == "crisis_attribution":
        try:
            from configs.trading_config import BENCHMARK_STOCK
            benchmark_sid = BENCHMARK_STOCK
        except ImportError:
            benchmark_sid = "0050"
        try:
            from .extra_diagnostics import diagnose_crisis_attribution
            print("  [Layer3] 附加診斷：危機歸因...")
            extra_diag_result = diagnose_crisis_attribution(
                baseline_curve  = baseline_curve,
                candidate_curve = candidate_curve,
                val_idx         = val_idx,
                dates           = dates,
                prices_dict     = prices_dict,
                benchmark_sid   = benchmark_sid,
            )
            if extra_diag_result.triggered:
                reasons.extend(extra_diag_result.triggered_rules)
                print(f"  [Layer3] 危機歸因觸發：{extra_diag_result.triggered_rules}")
        except Exception as e:
            print(f"  [Layer3] 危機歸因診斷失敗：{e}")

    extra_diag_dict = None
    if extra_diag_result:
        extra_diag_dict = {k: v for k, v in extra_diag_result.__dict__.items() if k != "verdict_override"}

    return Layer3Result(
        passed               = len(reasons) == 0,
        regime_results       = regime_results,
        stable_regimes       = stable_regimes,
        hhi_explosion_rate   = hhi_explosion_rate,
        tail_risk_baseline   = tail_risk_baseline,
        tail_risk_candidate  = tail_risk_candidate,
        delta_tail_risk      = delta_tail_risk,
        candidate_sharpe     = candidate_sharpe,
        equal_weight_sharpe  = equal_weight_s,
        momentum_sharpe      = momentum_s,
        beats_equal_weight   = beats_equal_weight,
        beats_momentum       = beats_momentum,
        extra_diagnostic     = extra_diag_dict,
        rejection_reasons    = reasons,
    )