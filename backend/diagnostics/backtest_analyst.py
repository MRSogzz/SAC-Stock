"""
diagnostics/backtest_analyst.py
================================
回測 / 策略行為分析。

包含：
  BT. diag_backtest_curve()         回測層：資產爆炸偵測
   8. diag_walkforward_summary()    Walk-forward 層：各窗口報酬/勝率
   9. diag_regime_model_selection() Walk-forward 層：Regime 選模型
      detect_regime()               工具函數：偵測當前市場環境
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .logger import DebugLogger


# ═══════════════════════════════════════════════════════════════════════════════
# 工具：Regime 偵測
# ═══════════════════════════════════════════════════════════════════════════════

def detect_regime(benchmark_prices: np.ndarray, lookback: int = 60) -> str:
    """根據 0050 的近期表現判斷市場環境（bull / bear / sideways）。"""
    if len(benchmark_prices) < lookback:
        return "sideways"
    recent = benchmark_prices[-lookback:]
    ret_60 = float(recent[-1] / recent[0] - 1)
    vol_20 = float(np.std(np.diff(np.log(recent[-20:]))))
    if ret_60 > 0.05 and vol_20 < 0.012:
        return "bull"
    elif ret_60 < -0.05:
        return "bear"
    return "sideways"


# ═══════════════════════════════════════════════════════════════════════════════
# BT. 回測層：資產爆炸偵測
# ═══════════════════════════════════════════════════════════════════════════════

def diag_backtest_curve(actor, scalers: dict, feat_dict: dict,
                        prices_dict: dict, volumes_dict: dict,
                        stock_ids: list, initial_capital: float,
                        dates: list, logger: DebugLogger,
                        check_interval: int = 100) -> dict:
    """
    執行回測並追蹤每 check_interval 步的資產狀態，
    找出資產爆炸發生的時間點和原因。

    logit_state 處理：
      - LogitDelta actor：跨步維護 logit_state，確保 Leaky Integrator 正確累積
      - Dirichlet actor：logit_state=None，sample() 內部忽略
    """
    import pandas as pd

    try:
        from configs.trading_config import (
            LOT_SIZE, MIN_FEE_LOT, MIN_FEE_ODD,
            BROKER_FEE, SECURITY_TAX, OBSERVABLE_STOCKS as OBS,
        )
        from configs.base_config import DEVICE
        from src.data.processor import AVG_VOL_WINDOW
        from src.models.architectures import N_ACTIONS
    except ImportError:
        LOT_SIZE = 1000; MIN_FEE_LOT = 20; MIN_FEE_ODD = 1
        BROKER_FEE = 0.001425; SECURITY_TAX = 0.003
        OBS = stock_ids; DEVICE = torch.device("cpu")
        AVG_VOL_WINDOW = 20; N_ACTIONS = len(stock_ids) + 1

    def _lot_fee(a): return float(max(math.ceil(a * BROKER_FEE), MIN_FEE_LOT))
    def _tax(a):     return float(math.ceil(a * SECURITY_TAX))

    n_stocks = len(stock_ids)
    n_steps  = min(len(v) for v in feat_dict.values())

    # ── 標準化特徵 ───────────────────────────────────────────────────────────
    scaled = {}
    for sid in OBS:
        if sid not in feat_dict or sid not in scalers:
            continue
        feat = feat_dict[sid].values[:n_steps].copy().astype(np.float64)
        feat = np.where(np.isnan(feat), 0.0, feat)
        scaled[sid] = np.clip(
            scalers[sid].transform(feat), -5.0, 5.0
        ).astype(np.float32)

    # ── 滾動平均成交量 ───────────────────────────────────────────────────────
    avg_vol = {}
    for sid in stock_ids:
        vol_arr      = volumes_dict[sid][:n_steps].astype(np.float64)
        avg_vol[sid] = pd.Series(vol_arr).rolling(
            AVG_VOL_WINDOW, min_periods=1
        ).mean().values

    capital   = float(initial_capital)
    lots_held = np.zeros(n_stocks, dtype=np.int64)
    odd_held  = np.zeros(n_stocks, dtype=np.int64)

    logger.log(f"\n[診斷 BT] 回測資產追蹤（每 {check_interval} 步）")
    logger.log(f"  {'step':>6}  {'capital':>14}  {'lots_val':>14}  {'total':>14}  {'max_lots':>10}")

    peak_total     = initial_capital
    explosion_step = None

    # logit_state：供 LogitDelta actor 跨步累積，Dirichlet actor 忽略
    logit_state = torch.zeros(1, N_ACTIONS, dtype=torch.float32, device=DEVICE)

    for i in range(n_steps - 1):
        prices_T    = np.array([prices_dict[sid][i]     for sid in stock_ids], dtype=np.float64)
        prices_T1   = np.array([prices_dict[sid][i + 1] for sid in stock_ids], dtype=np.float64)
        lot_val     = lots_held * LOT_SIZE * prices_T
        odd_val     = odd_held  * prices_T
        total_asset = capital + lot_val.sum() + odd_val.sum()
        lot_ratio   = lot_val / (total_asset + 1e-8)
        odd_ratio_v = odd_val / (total_asset + 1e-8)
        cash_ratio  = capital / (total_asset + 1e-8)

        feat_vec = np.concatenate([scaled[sid][i] for sid in OBS if sid in scaled])
        obs      = np.concatenate([feat_vec, lot_ratio, odd_ratio_v, [cash_ratio]])
        obs      = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)

        s = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # w：(1, N_ACTIONS) 當步動作權重
            # logit_state：更新後的 Leaky Integrator 狀態，下步傳入
            # Dirichlet actor 忽略 logit_state 參數
            w, logit_state, _ = actor.sample(s, logit_state=logit_state)

        # 取前 n_stocks 維作為股票倉位，squeeze(0) 確保只壓 batch 維度
        target = np.clip(
            w.squeeze(0).cpu().numpy().astype(np.float64)[:n_stocks], 0.0, 1.0
        )

        target_value  = total_asset * target
        target_shares = target_value / (prices_T + 1e-8)
        target_lots   = (target_shares // LOT_SIZE).astype(np.int64)

        # ── 賣出 ────────────────────────────────────────────────────────────
        for j in range(n_stocks):
            sell_lots = max(0, int(lots_held[j]) - int(target_lots[j]))
            if sell_lots > 0:
                gross = sell_lots * LOT_SIZE * float(prices_T[j])
                capital += gross - _lot_fee(gross) - _tax(gross)
                lots_held[j] -= sell_lots

        # ── 買入 ────────────────────────────────────────────────────────────
        for j in range(n_stocks):
            buy_lots = max(0, int(target_lots[j]) - int(lots_held[j]))
            if buy_lots > 0:
                price        = float(prices_T[j])
                cost_per_lot = LOT_SIZE * price + _lot_fee(LOT_SIZE * price)
                affordable   = int(capital // cost_per_lot)
                buy_lots     = min(buy_lots, affordable)
                if buy_lots > 0:
                    gross = buy_lots * LOT_SIZE * price
                    capital -= gross + _lot_fee(gross)
                    lots_held[j] += buy_lots

        # ── 自動升級 ─────────────────────────────────────────────────────────
        for j in range(n_stocks):
            if odd_held[j] >= LOT_SIZE:
                lots_held[j] += odd_held[j] // LOT_SIZE
                odd_held[j]  %= LOT_SIZE

        lot_val_T1 = lots_held * LOT_SIZE * prices_T1
        total_T1   = capital + lot_val_T1.sum() + odd_held.sum()

        if total_T1 > peak_total * 5 and explosion_step is None:
            explosion_step = i
            logger.log(f"\n  *** 資產爆炸偵測：step={i}  total={total_T1:,.0f} ***")
            logger.log(f"  target={target.round(3)}")
            logger.log(f"  lots_held={lots_held}")
            logger.log(f"  prices_T={prices_T.round(1)}")
            logger.log(f"  capital={capital:,.0f}")
        peak_total = max(peak_total, total_T1)

        if i % check_interval == 0 or i == n_steps - 2:
            logger.log(f"  {i:>6}  {capital:>14,.0f}  "
                       f"{lot_val_T1.sum():>14,.0f}  "
                       f"{total_T1:>14,.0f}  "
                       f"{lots_held.max():>10}")

    final = capital + (
        lots_held * LOT_SIZE *
        np.array([prices_dict[sid][-1] for sid in stock_ids])
    ).sum()
    logger.log(f"\n  最終資產: {final:,.0f}  報酬率: {(final/initial_capital-1)*100:.2f}%")
    logger.log(
        f"  {'*** 爆炸發生在 step=' + str(explosion_step) + ' ***' if explosion_step else '✓ 無異常爆炸'}"
    )

    return {"final": round(final, 0), "explosion_step": explosion_step}


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Walk-forward 各窗口摘要
# ═══════════════════════════════════════════════════════════════════════════════

def diag_walkforward_summary(window_results: list,
                              logger: DebugLogger) -> dict:
    """彙整各窗口的訓練/驗證報酬、Regime、勝率。"""
    logger.log("\n[診斷 8] Walk-forward 各窗口摘要")
    logger.log(f"  {'窗口':>4}  {'訓練期':>22}  {'驗證期':>22}  "
               f"{'訓練報酬':>10}  {'驗證報酬':>10}  {'Regime':>10}  {'勝率':>6}")
    logger.log("  " + "-" * 95)

    for w in window_results:
        train_ret = w.get("train_return", 0)
        val_ret   = w.get("val_return",   0)
        regime    = w.get("regime",       "?")
        win_rate  = w.get("win_rate",     0)
        overfit   = train_ret > val_ret * 3
        status    = "*** 過擬合 ***" if overfit else "✓"
        logger.log(
            f"  {w.get('window','?'):>4}  "
            f"{w.get('train_start','?'):>11}~{w.get('train_end','?'):>11}  "
            f"{w.get('val_start','?'):>11}~{w.get('val_end','?'):>11}  "
            f"{train_ret:>9.1f}%  {val_ret:>9.1f}%  "
            f"{regime:>10}  {win_rate:>5.1f}%  {status}"
        )

    val_returns = [w.get("val_return", 0) for w in window_results]
    avg_val     = round(float(np.mean(val_returns)), 2) if val_returns else 0
    std_val     = round(float(np.std(val_returns)),  2) if val_returns else 0

    logger.log(f"\n  驗證報酬均值: {avg_val}%  標準差: {std_val}%")
    logger.log(
        f"  {'✓ 策略穩定' if std_val < 20 else '*** 警告：各窗口報酬差異過大 ***'}"
    )

    return {
        "avg_val_return": avg_val,
        "std_val_return": std_val,
        "window_results": window_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Walk-forward Regime 選模型
# ═══════════════════════════════════════════════════════════════════════════════

def diag_regime_model_selection(benchmark_prices: np.ndarray,
                                window_regimes: dict,
                                logger: DebugLogger) -> str:
    """根據當前市場環境選擇對應的模型窗口。"""
    current_regime = detect_regime(benchmark_prices)
    matched        = [w for w, r in window_regimes.items() if r == current_regime]
    selected       = matched[-1] if matched else max(window_regimes.keys())

    logger.log(f"\n[診斷 9] Regime 選模型")
    logger.log(f"  當前市場環境: {current_regime}")
    logger.log(f"  各窗口 Regime: {window_regimes}")
    if matched:
        logger.log(f"  ✓ 選擇窗口 {selected}（{current_regime} 環境匹配）")
    else:
        logger.log(f"  ⚠ 無完全匹配，退回最新窗口 {selected}")

    return selected