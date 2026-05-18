"""
evaluation/layer2_strategy.py  [重構版]
========================================
第二層：特徵微量探針檢驗（核心層）

干預形式永久固定為：
    w_new = normalize( w_base * (1 + ε * zscore(feature)) )
    ε = 0.01

這確保 Layer 2 永遠只是一個微量探針，不可能主導決策。

診斷信心標籤：
  PASS       ΔSharpe 穩定改善且風險不惡化
  WEAK_PASS  改善微弱但方向一致
  NO_EFFECT  無明顯影響
  CONFLICT   Sharpe 改善但換倉率惡化（偽 Alpha 警報）
  FAIL       MaxDD 惡化或區間崩潰

三向否決鐵律：
  1. ΔTurnover 顯著上升但 ΔSharpe 微幅 → CONFLICT
  2. ΔMaxDD 惡化超過門檻               → FAIL
  3. Regime consistency 差              → WEAK_PASS（降級）
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


# ── 探針參數（永久固定，不可調整）────────────────────────────────────────────
PROBE_EPSILON = 0.01   # 微量干預強度，永遠是 0.01

# ── 診斷門檻 ──────────────────────────────────────────────────────────────────
DELTA_SHARPE_PASS      =  0.05   # PASS 的最低 ΔSharpe
DELTA_SHARPE_WEAK      =  0.005  # WEAK_PASS 的最低 ΔSharpe
MAX_MDD_DEGRADATION    = -0.03   # FAIL：MaxDD 惡化超過 3%
MAX_TURNOVER_CONFLICT  =  0.10   # CONFLICT：換倉率上升超過 10%
MIN_REGIME_CONSISTENCY =  0.5    # regime 一致性低於 50% → 降級


# ── 診斷標籤 ──────────────────────────────────────────────────────────────────
VERDICT_PASS      = "PASS"
VERDICT_WEAK_PASS = "WEAK_PASS"
VERDICT_NO_EFFECT = "NO_EFFECT"
VERDICT_CONFLICT  = "CONFLICT"
VERDICT_FAIL      = "FAIL"

# PASS / WEAK_PASS 視為通過 Layer 2
PASSING_VERDICTS = {VERDICT_PASS, VERDICT_WEAK_PASS}


@dataclass
class BacktestMetrics:
    total_return:    float = 0.0
    sharpe:          float = 0.0
    max_drawdown:    float = 0.0
    avg_turnover:    float = 0.0
    n_trades:        int   = 0
    portfolio_curve: list  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_return":   round(self.total_return,  4),
            "sharpe":         round(self.sharpe,         4),
            "max_drawdown":   round(self.max_drawdown,   4),
            "avg_turnover":   round(self.avg_turnover,   4),
            "n_trades":       self.n_trades,
            "portfolio_curve": self.portfolio_curve,
        }


@dataclass
class Layer2Result:
    passed:  bool
    verdict: str = VERDICT_NO_EFFECT   # 診斷信心標籤

    baseline_metrics:  BacktestMetrics = field(default_factory=BacktestMetrics)
    candidate_metrics: BacktestMetrics = field(default_factory=BacktestMetrics)

    delta_sharpe:   float = 0.0
    delta_mdd:      float = 0.0
    delta_turnover: float = 0.0
    delta_return:   float = 0.0

    # 新增：regime 一致性（有正 ΔSharpe 的 regime 比例）
    regime_consistency: float = 0.0

    # 附加診斷結果（僅特定特徵啟用）
    extra_diagnostic: dict | None = None

    rejection_reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verdict_icon = {
            VERDICT_PASS:      "✅",
            VERDICT_WEAK_PASS: "⚠️",
            VERDICT_NO_EFFECT: "➖",
            VERDICT_CONFLICT:  "🚨",
            VERDICT_FAIL:      "❌",
        }.get(self.verdict, "?")

        lines = [
            f"Layer 2 [{verdict_icon} {self.verdict}]",
            f"  ε = {PROBE_EPSILON}（固定微量探針）",
            f"  {'指標':<18} {'基準':>10} {'候選':>10} {'Δ':>10}",
            f"  {'-'*52}",
            f"  {'Total Return':<18} {self.baseline_metrics.total_return:>10.2%} "
            f"{self.candidate_metrics.total_return:>10.2%} "
            f"{self.delta_return:>+10.2%}",
            f"  {'Sharpe':<18} {self.baseline_metrics.sharpe:>10.4f} "
            f"{self.candidate_metrics.sharpe:>10.4f} "
            f"{self.delta_sharpe:>+10.4f}",
            f"  {'Max Drawdown':<18} {self.baseline_metrics.max_drawdown:>10.2%} "
            f"{self.candidate_metrics.max_drawdown:>10.2%} "
            f"{self.delta_mdd:>+10.2%}",
            f"  {'Avg Turnover':<18} {self.baseline_metrics.avg_turnover:>10.2%} "
            f"{self.candidate_metrics.avg_turnover:>10.2%} "
            f"{self.delta_turnover:>+10.2%}",
            f"  Regime 一致性：{self.regime_consistency:.0%}",
        ]
        if self.rejection_reasons:
            lines.append("  診斷原因：" + "；".join(self.rejection_reasons))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "passed":             self.passed,
            "verdict":            self.verdict,
            "probe_epsilon":      PROBE_EPSILON,
            "baseline_metrics":   self.baseline_metrics.to_dict(),
            "candidate_metrics":  self.candidate_metrics.to_dict(),
            "delta_sharpe":       round(self.delta_sharpe,        4),
            "delta_mdd":          round(self.delta_mdd,           4),
            "delta_turnover":     round(self.delta_turnover,      4),
            "delta_return":       round(self.delta_return,        4),
            "regime_consistency": round(self.regime_consistency,  4),
            "extra_diagnostic":   self.extra_diagnostic,
            "rejection_reasons":  self.rejection_reasons,
            "thresholds": {
                "probe_epsilon":          PROBE_EPSILON,
                "delta_sharpe_pass":      DELTA_SHARPE_PASS,
                "delta_sharpe_weak":      DELTA_SHARPE_WEAK,
                "max_mdd_degradation":    MAX_MDD_DEGRADATION,
                "max_turnover_conflict":  MAX_TURNOVER_CONFLICT,
                "min_regime_consistency": MIN_REGIME_CONSISTENCY,
            },
        }


# ── 基準回測（不修改權重）────────────────────────────────────────────────────

def _calc_baseline(
    actor,
    feat_dfs: dict,
    prices_dict: dict,
    tradeable_ids: list,
    observable_ids: list,
    val_idx: np.ndarray,
    initial_capital: float,
    device: Any,
    n_actions: int,
    n_feat_per_stock: int,
) -> tuple[BacktestMetrics, list]:
    """
    純基準回測，同時回傳每日的 base_weights 供後續探針使用。
    """
    from configs.trading_config import LOT_SIZE
    import math

    BROKER_FEE   = 0.001425
    SECURITY_TAX = 0.003
    MIN_FEE_LOT  = 20

    def _fee(a): return float(max(math.ceil(a * BROKER_FEE), MIN_FEE_LOT))
    def _tax(a): return float(math.ceil(a * SECURITY_TAX))

    n_stocks    = len(tradeable_ids)
    capital     = float(initial_capital)
    lots_held   = np.zeros(n_stocks, dtype=np.int64)
    odd_held    = np.zeros(n_stocks, dtype=np.int64)
    logit_state = torch.zeros(1, n_actions, dtype=torch.float32, device=device)
    curve       = [capital]
    turnovers   = []
    prev_w      = np.zeros(n_stocks)
    daily_weights = []   # 每日 base weights

    actor.eval()
    for t in val_idx:
        raw_vecs = [feat_dfs[sid].iloc[t].values for sid in observable_ids]
        raw_vecs = [v[:n_feat_per_stock] for v in raw_vecs]
        feat_vec = np.concatenate(raw_vecs).astype(np.float32)
        feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=10.0, neginf=-10.0)

        prices_T = np.array([prices_dict[sid][t] for sid in tradeable_ids], dtype=np.float64)
        lot_val  = lots_held * LOT_SIZE * prices_T
        odd_val  = odd_held  * prices_T
        total    = capital + lot_val.sum() + odd_val.sum()

        obs = np.concatenate([
            feat_vec,
            lot_val / (total + 1e-8),
            odd_val / (total + 1e-8),
            [capital / (total + 1e-8)],
        ]).astype(np.float32)

        s = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            w, logit_state, _ = actor.sample(s, logit_state=logit_state)
        weights = w.squeeze(0).cpu().numpy()[:n_stocks]
        daily_weights.append(weights.copy())

        turnovers.append(float(np.abs(weights - prev_w).sum() / 2))
        prev_w = weights.copy()

        target_lots = (total * weights / (prices_T + 1e-8) // LOT_SIZE).astype(np.int64)

        for j in range(n_stocks):
            sell = max(0, int(lots_held[j]) - int(target_lots[j]))
            if sell > 0:
                gross = sell * LOT_SIZE * float(prices_T[j])
                capital += gross - _fee(gross) - _tax(gross)
                lots_held[j] -= sell

        for j in range(n_stocks):
            buy = max(0, int(target_lots[j]) - int(lots_held[j]))
            if buy > 0:
                price  = float(prices_T[j])
                afford = int(capital // (LOT_SIZE * price + _fee(LOT_SIZE * price)))
                buy    = min(buy, afford)
                if buy > 0:
                    gross    = buy * LOT_SIZE * price
                    capital -= gross + _fee(gross)
                    lots_held[j] += buy

        for j in range(n_stocks):
            if odd_held[j] >= LOT_SIZE:
                lots_held[j] += odd_held[j] // LOT_SIZE
                odd_held[j]  %= LOT_SIZE

        if t + 1 < len(prices_dict[tradeable_ids[0]]):
            prices_T1 = np.array([prices_dict[sid][t + 1] for sid in tradeable_ids])
            total_T1  = capital + (lots_held * LOT_SIZE * prices_T1).sum()
        else:
            total_T1  = capital + (lots_held * LOT_SIZE * prices_T).sum()
        curve.append(round(total_T1, 0))

    curve_arr    = np.array(curve, dtype=np.float64)
    daily_ret    = np.diff(curve_arr) / (curve_arr[:-1] + 1e-8)
    sharpe       = float(np.mean(daily_ret) / (np.std(daily_ret) + 1e-8) * np.sqrt(252))
    peak         = np.maximum.accumulate(curve_arr)
    mdd          = float(((curve_arr - peak) / (peak + 1e-8)).min())
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    metrics = BacktestMetrics(
        total_return    = float(curve_arr[-1] / curve_arr[0] - 1),
        sharpe          = sharpe,
        max_drawdown    = mdd,
        avg_turnover    = avg_turnover,
        portfolio_curve = curve,
    )
    return metrics, daily_weights


# ── 探針回測（微量干預）──────────────────────────────────────────────────────

def _calc_probe(
    daily_weights: list,
    candidate_feat_dfs: dict,
    prices_dict: dict,
    tradeable_ids: list,
    observable_ids: list,
    val_idx: np.ndarray,
    initial_capital: float,
    feature_col: str,
    epsilon: float = PROBE_EPSILON,
) -> BacktestMetrics:
    """
    探針回測：w_new = normalize( w_base * (1 + ε * zscore(feature)) )

    不需要模型推理，直接用 base_weights 做微量調整。
    """
    from configs.trading_config import LOT_SIZE
    import math

    BROKER_FEE   = 0.001425
    SECURITY_TAX = 0.003
    MIN_FEE_LOT  = 20

    def _fee(a): return float(max(math.ceil(a * BROKER_FEE), MIN_FEE_LOT))
    def _tax(a): return float(math.ceil(a * SECURITY_TAX))

    n_stocks  = len(tradeable_ids)
    capital   = float(initial_capital)
    lots_held = np.zeros(n_stocks, dtype=np.int64)
    odd_held  = np.zeros(n_stocks, dtype=np.int64)
    curve     = [capital]
    turnovers = []
    prev_w    = np.zeros(n_stocks)

    # 預先計算整個驗證期間的特徵 z-score（leakage-safe：用驗證期間自身的 μ/σ）
    # shape: (n_tradeable, n_val_days)
    feat_matrix = np.zeros((n_stocks, len(val_idx)), dtype=np.float64)
    for j, sid in enumerate(tradeable_ids):
        if sid not in candidate_feat_dfs:
            continue
        df = candidate_feat_dfs[sid]
        # 找新特徵欄位（帶 _alpha_ 前綴或原始名稱）
        col = None
        for c in df.columns:
            if c == feature_col or c == f"_alpha_{feature_col}" or c.endswith(feature_col):
                col = c
                break
        if col is None:
            continue
        for k, t in enumerate(val_idx):
            if t < len(df):
                v = df.iloc[t][col]
                feat_matrix[j, k] = float(v) if np.isscalar(v) or (hasattr(v, '__len__') and len(v) == 1) else 0.0

    # 對每日做截面 z-score（跨股票標準化）
    mu  = feat_matrix.mean(axis=0, keepdims=True)
    sig = feat_matrix.std(axis=0, keepdims=True) + 1e-8
    feat_z = np.clip((feat_matrix - mu) / sig, -3.0, 3.0)  # shape: (n_tradeable, n_val_days)

    for k, t in enumerate(val_idx):
        w_base = daily_weights[k]   # 來自基準回測的 base weights

        # 探針干預：w_new = normalize( w_base * (1 + ε * z) )
        z      = feat_z[:, k]
        w_raw  = w_base * (1.0 + epsilon * z)
        w_raw  = np.clip(w_raw, 0.0, None)   # 不允許做空
        w_sum  = w_raw.sum()
        weights = w_raw / (w_sum + 1e-8) if w_sum > 1e-8 else w_base

        turnovers.append(float(np.abs(weights - prev_w).sum() / 2))
        prev_w = weights.copy()

        prices_T    = np.array([prices_dict[sid][t] for sid in tradeable_ids], dtype=np.float64)
        lot_val     = lots_held * LOT_SIZE * prices_T
        odd_val     = odd_held  * prices_T
        total       = capital + lot_val.sum() + odd_val.sum()
        target_lots = (total * weights / (prices_T + 1e-8) // LOT_SIZE).astype(np.int64)

        for j in range(n_stocks):
            sell = max(0, int(lots_held[j]) - int(target_lots[j]))
            if sell > 0:
                gross    = sell * LOT_SIZE * float(prices_T[j])
                capital += gross - _fee(gross) - _tax(gross)
                lots_held[j] -= sell

        for j in range(n_stocks):
            buy = max(0, int(target_lots[j]) - int(lots_held[j]))
            if buy > 0:
                price  = float(prices_T[j])
                afford = int(capital // (LOT_SIZE * price + _fee(LOT_SIZE * price)))
                buy    = min(buy, afford)
                if buy > 0:
                    gross    = buy * LOT_SIZE * price
                    capital -= gross + _fee(gross)
                    lots_held[j] += buy

        for j in range(n_stocks):
            if odd_held[j] >= LOT_SIZE:
                lots_held[j] += odd_held[j] // LOT_SIZE
                odd_held[j]  %= LOT_SIZE

        if t + 1 < len(prices_dict[tradeable_ids[0]]):
            prices_T1 = np.array([prices_dict[sid][t + 1] for sid in tradeable_ids])
            total_T1  = capital + (lots_held * LOT_SIZE * prices_T1).sum()
        else:
            total_T1  = capital + (lots_held * LOT_SIZE * prices_T).sum()
        curve.append(round(total_T1, 0))

    curve_arr    = np.array(curve, dtype=np.float64)
    daily_ret    = np.diff(curve_arr) / (curve_arr[:-1] + 1e-8)
    sharpe       = float(np.mean(daily_ret) / (np.std(daily_ret) + 1e-8) * np.sqrt(252))
    peak         = np.maximum.accumulate(curve_arr)
    mdd          = float(((curve_arr - peak) / (peak + 1e-8)).min())
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    return BacktestMetrics(
        total_return    = float(curve_arr[-1] / curve_arr[0] - 1),
        sharpe          = sharpe,
        max_drawdown    = mdd,
        avg_turnover    = avg_turnover,
        portfolio_curve = curve,
    )


# ── 診斷邏輯（三向否決鐵律）─────────────────────────────────────────────────

def _diagnose(
    delta_sharpe:   float,
    delta_mdd:      float,
    delta_turnover: float,
    regime_consistency: float,
) -> tuple[str, bool, list[str]]:
    """
    根據三向否決鐵律判斷診斷標籤。

    Returns:
        (verdict, passed, reasons)
    """
    reasons = []

    # 鐵律 1：MaxDD 惡化 → FAIL
    if delta_mdd < MAX_MDD_DEGRADATION:
        reasons.append(f"ΔMaxDD={delta_mdd:+.2%} 惡化超過 {-MAX_MDD_DEGRADATION:.0%}")
        return VERDICT_FAIL, False, reasons

    # 鐵律 2：換倉率顯著上升但 ΔSharpe 微幅 → CONFLICT（偽 Alpha）
    if delta_turnover > MAX_TURNOVER_CONFLICT and delta_sharpe < DELTA_SHARPE_PASS:
        reasons.append(
            f"ΔTurnover={delta_turnover:+.2%} 顯著上升，但 ΔSharpe={delta_sharpe:+.4f} 僅微幅改善（偽 Alpha）"
        )
        return VERDICT_CONFLICT, False, reasons

    # 判斷基礎 verdict
    if delta_sharpe >= DELTA_SHARPE_PASS:
        verdict = VERDICT_PASS
    elif delta_sharpe >= DELTA_SHARPE_WEAK:
        verdict = VERDICT_WEAK_PASS
        reasons.append(f"ΔSharpe={delta_sharpe:+.4f} 改善微弱（< {DELTA_SHARPE_PASS}）")
    else:
        verdict  = VERDICT_NO_EFFECT
        reasons.append(f"ΔSharpe={delta_sharpe:+.4f} 無明顯影響（< {DELTA_SHARPE_WEAK}）")
        return verdict, False, reasons

    # 鐵律 3：Regime consistency 差 → 降級
    if regime_consistency < MIN_REGIME_CONSISTENCY:
        reasons.append(
            f"Regime 一致性={regime_consistency:.0%} 偏低（改善僅存在於部分市場環境）"
        )
        if verdict == VERDICT_PASS:
            verdict = VERDICT_WEAK_PASS

    passed = verdict in PASSING_VERDICTS
    return verdict, passed, reasons


# ── 計算 regime 一致性 ────────────────────────────────────────────────────────

def _calc_regime_consistency(
    baseline_curve:  np.ndarray,
    candidate_curve: np.ndarray,
    val_idx:         np.ndarray,
    prices_dict:     dict,
    tradeable_ids:   list,
    dates:           list,
) -> float:
    """
    計算在各市場環境中有正 ΔSharpe 的比例。
    """
    from .layer3_robustness import _classify_regime

    bm_prices  = prices_dict[tradeable_ids[0]]
    regime_days: dict[str, list[int]] = {}
    for k, t in enumerate(val_idx):
        regime = _classify_regime(bm_prices, int(t))
        regime_days.setdefault(regime, []).append(k)

    def _seg_sharpe(curve: np.ndarray, indices: list[int]) -> float:
        if len(indices) < 5 or len(curve) < 2:
            return 0.0
        idx    = [min(d, len(curve) - 2) for d in indices]
        daily  = np.diff(curve)[idx] / (curve[idx] + 1e-8)
        return float(np.mean(daily) / (np.std(daily) + 1e-8) * np.sqrt(252))

    positive = 0
    total    = 0
    for regime, days in regime_days.items():
        if len(days) < 5:
            continue
        bs = _seg_sharpe(baseline_curve,  days)
        cs = _seg_sharpe(candidate_curve, days)
        if cs > bs:
            positive += 1
        total += 1

    return positive / total if total > 0 else 0.0


# ── 主入口 ────────────────────────────────────────────────────────────────────

def run_layer2(
    baseline_model_path: str,
    baseline_feat_dfs:   dict,
    candidate_feat_dfs:  dict,
    prices_dict:         dict,
    volumes_dict:        dict,
    dates:               list[str],
    tradeable_ids:       list[str],
    observable_ids:      list[str],
    val_start:           str,
    val_end:             str,
    initial_capital:     float = 1_000_000.0,
    feature_column:      str | None = None,
    extra_diagnostic:    str | None = None,   # "low_vol_exposure" / "turnover_defense" / None
) -> Layer2Result:
    """
    執行第二層：特徵微量探針檢驗。

    固定干預：w_new = normalize( w_base * (1 + 0.01 * zscore(feature)) )
    extra_diagnostic: 指定附加診斷類型（僅特定特徵啟用）
    """
    try:
        from configs.base_config    import DEVICE
        from configs.trading_config import N_FEATURES
        from src.models.architectures import N_ACTIONS, PortfolioActorLogitDelta
    except ImportError as e:
        return Layer2Result(passed=False, verdict=VERDICT_FAIL,
                            rejection_reasons=[f"Import 失敗：{e}"])

    # 載入模型
    with open(baseline_model_path, "rb") as f:
        ckpt = pickle.load(f)
    actor = PortfolioActorLogitDelta(ckpt["state_dim"], ckpt["n_stocks"])
    actor.load_state_dict(ckpt["actor_state"])
    actor.to(DEVICE)
    actor.eval()

    dates_arr = np.array(dates)
    mask      = (dates_arr >= val_start) & (dates_arr <= val_end)
    val_idx   = np.where(mask)[0]

    if len(val_idx) < 20:
        return Layer2Result(
            passed  = False,
            verdict = VERDICT_FAIL,
            rejection_reasons=[f"驗證集不足 20 天（{len(val_idx)} 天）"],
        )

    # 推斷 feature_column（去掉 _alpha_ 前綴後的原始名稱）
    feat_col = feature_column or ""
    if feat_col.startswith("_alpha_"):
        feat_col = feat_col[len("_alpha_"):]

    print(f"  [Layer2] 基準回測（{len(val_idx)} 天）...")
    baseline_m, daily_weights = _calc_baseline(
        actor, baseline_feat_dfs, prices_dict,
        tradeable_ids, observable_ids, val_idx,
        initial_capital, DEVICE, N_ACTIONS, N_FEATURES,
    )

    print(f"  [Layer2] 探針回測（ε={PROBE_EPSILON}）...")
    candidate_m = _calc_probe(
        daily_weights, candidate_feat_dfs, prices_dict,
        tradeable_ids, observable_ids, val_idx,
        initial_capital, feat_col, PROBE_EPSILON,
    )

    delta_sharpe   = candidate_m.sharpe       - baseline_m.sharpe
    delta_mdd      = candidate_m.max_drawdown - baseline_m.max_drawdown
    delta_turnover = candidate_m.avg_turnover - baseline_m.avg_turnover
    delta_return   = candidate_m.total_return - baseline_m.total_return

    # 計算 regime 一致性
    regime_consistency = _calc_regime_consistency(
        np.array(baseline_m.portfolio_curve,  dtype=np.float64),
        np.array(candidate_m.portfolio_curve, dtype=np.float64),
        val_idx, prices_dict, tradeable_ids, dates,
    )

    verdict, passed, reasons = _diagnose(
        delta_sharpe, delta_mdd, delta_turnover, regime_consistency,
    )

    print(f"  [Layer2] 診斷：{verdict}  ΔSharpe={delta_sharpe:+.4f}  "
          f"ΔMdd={delta_mdd:+.2%}  ΔTurnover={delta_turnover:+.2%}  "
          f"Regime一致性={regime_consistency:.0%}")

    # ── 附加診斷（指定特徵專屬）──────────────────────────────────────────────
    extra_diag_result = None
    try:
        from configs.trading_config import BENCHMARK_STOCK
        benchmark_sid = BENCHMARK_STOCK
    except ImportError:
        benchmark_sid = "0050"

    if extra_diagnostic == "low_vol_exposure":
        from .extra_diagnostics import diagnose_low_vol_exposure
        print("  [Layer2] 附加診斷：低波動資產曝險...")
        extra_diag_result = diagnose_low_vol_exposure(
            baseline_weights = daily_weights,
            probe_weights    = _get_probe_weights(
                daily_weights, candidate_feat_dfs, prices_dict,
                tradeable_ids, observable_ids, val_idx, feat_col,
            ),
            prices_dict    = prices_dict,
            tradeable_ids  = tradeable_ids,
            val_idx        = val_idx,
        )
        if extra_diag_result.triggered:
            verdict = extra_diag_result.verdict_override or verdict
            passed  = verdict in PASSING_VERDICTS
            reasons.extend(extra_diag_result.triggered_rules)
            print(f"  [Layer2] 低波動曝險觸發：{extra_diag_result.triggered_rules}")

    elif extra_diagnostic == "turnover_defense":
        from .extra_diagnostics import diagnose_turnover_defense
        print("  [Layer2] 附加診斷：換倉防禦...")
        extra_diag_result = diagnose_turnover_defense(
            baseline_weights = daily_weights,
            probe_weights    = _get_probe_weights(
                daily_weights, candidate_feat_dfs, prices_dict,
                tradeable_ids, observable_ids, val_idx, feat_col,
            ),
            baseline_curve  = np.array(baseline_m.portfolio_curve, dtype=np.float64),
            candidate_curve = np.array(candidate_m.portfolio_curve, dtype=np.float64),
            prices_dict     = prices_dict,
            tradeable_ids   = tradeable_ids,
            benchmark_sid   = benchmark_sid,
            val_idx         = val_idx,
        )
        if extra_diag_result.triggered:
            verdict = extra_diag_result.verdict_override or verdict
            passed  = verdict in PASSING_VERDICTS
            reasons.extend(extra_diag_result.triggered_rules)
            print(f"  [Layer2] 換倉防禦觸發：{extra_diag_result.triggered_rules}")

    extra_diag_dict = extra_diag_result.__dict__ if extra_diag_result else None
    if extra_diag_dict and "verdict_override" in extra_diag_dict:
        extra_diag_dict = {**extra_diag_dict}
        extra_diag_dict.pop("verdict_override", None)

    return Layer2Result(
        passed             = passed,
        verdict            = verdict,
        baseline_metrics   = baseline_m,
        candidate_metrics  = candidate_m,
        delta_sharpe       = delta_sharpe,
        delta_mdd          = delta_mdd,
        delta_turnover     = delta_turnover,
        delta_return       = delta_return,
        regime_consistency = regime_consistency,
        extra_diagnostic   = extra_diag_dict,
        rejection_reasons  = reasons,
    )


def _get_probe_weights(
    daily_weights:     list,
    candidate_feat_dfs: dict,
    prices_dict:       dict,
    tradeable_ids:     list,
    observable_ids:    list,
    val_idx:           np.ndarray,
    feat_col:          str,
    epsilon:           float = PROBE_EPSILON,
) -> list[np.ndarray]:
    """
    重新計算探針策略的每日 weights（供附加診斷使用）。
    與 _calc_probe 相同邏輯，但只回傳 weights 序列不做資金曲線計算。
    """
    n_stocks = len(tradeable_ids)

    feat_matrix = np.zeros((n_stocks, len(val_idx)), dtype=np.float64)
    for j, sid in enumerate(tradeable_ids):
        if sid not in candidate_feat_dfs:
            continue
        df  = candidate_feat_dfs[sid]
        col = None
        for c in df.columns:
            if c == feat_col or c == f"_alpha_{feat_col}" or c.endswith(feat_col):
                col = c
                break
        if col is None:
            continue
        for k, t in enumerate(val_idx):
            if t < len(df):
                v = df.iloc[t][col]
                feat_matrix[j, k] = float(v) if np.isscalar(v) else 0.0

    mu     = feat_matrix.mean(axis=0, keepdims=True)
    sig    = feat_matrix.std(axis=0, keepdims=True) + 1e-8
    feat_z = np.clip((feat_matrix - mu) / sig, -3.0, 3.0)

    probe_ws = []
    for k in range(len(val_idx)):
        w_base = daily_weights[k]
        z      = feat_z[:, k]
        w_raw  = w_base * (1.0 + epsilon * z)
        w_raw  = np.clip(w_raw, 0.0, None)
        w_sum  = w_raw.sum()
        probe_ws.append(w_raw / (w_sum + 1e-8) if w_sum > 1e-8 else w_base.copy())
    return probe_ws