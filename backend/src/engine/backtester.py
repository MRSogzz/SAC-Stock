"""
回測引擎：與 PortfolioEnv 行為完全一致
  - 整張 + 零股交易
  - 零股成交率基於 20 日滾動平均成交量（ODD_FILL_RATIO = 0.65）
  - 升級（odd >= 1000）自動執行，無成本
  - 手續費無條件進位
  - 時序：obs = features[T]，結算 prices[T+1]/prices[T]
"""
import math
import numpy as np
import pandas as pd
import torch

from configs.base_config import DEVICE
from configs.trading_config import (
    STOCK_POOL, LOT_SIZE,
    MIN_FEE_LOT, MIN_FEE_ODD, BROKER_FEE, SECURITY_TAX,
    ODD_FILL_RATIO, RISK_FREE_DAILY,
    TRADEABLE_STOCKS, OBSERVABLE_STOCKS,
)
from src.data.processor import AVG_VOL_WINDOW
from src.models.architectures import N_ACTIONS
from src.utils.finance import calc_win_rate
from src.utils.common import safe_float, safe_list, sanitize
from diagnostics import register


# ─── 手續費與稅金（與 PortfolioEnv 完全一致）────────────────────────────────

def _lot_fee(amount: float) -> float:
    return float(max(math.ceil(amount * BROKER_FEE), MIN_FEE_LOT))

def _odd_fee(amount: float) -> float:
    return float(max(math.ceil(amount * BROKER_FEE), MIN_FEE_ODD))

def _tax(amount: float) -> float:
    return float(math.ceil(amount * SECURITY_TAX))

def _odd_fill(order_shares: int, avg_vol: float) -> int:
    if order_shares <= 0 or avg_vol <= 0:
        return 0
    available = avg_vol * ODD_FILL_RATIO
    fill_rate = min(1.0, order_shares / (available + 1e-8))
    return max(0, int(order_shares * fill_rate))


@register(
    module="Engine",
    inputs={
        "actor":           "PortfolioActor",
        "feat_dict":       "dict[str, pd.DataFrame]",
        "prices_dict":     "dict[str, np.ndarray]",
        "volumes_dict":    "dict[str, np.ndarray]",
        "stock_ids":       "list[str]",
        "initial_capital": "float",
        "feat_names":      "list[str]",
        "dates":           "list[str]",
    },
    outputs={"return": "dict"},
    notes="Deterministic policy 全程回測；含整張/零股/升級/強制平倉；回傳 portfolio_curve / trade_log / metrics",
)
def run_backtest(
    actor,
    feat_dict:       dict,
    prices_dict:     dict,
    volumes_dict:    dict,
    stock_ids:       list,
    initial_capital: float,
    feat_names:      list,
    dates:           list,
    maybe_dates:     list = None,
    **kwargs,
) -> dict:
    if maybe_dates is not None:
        feat_dict, prices_dict, volumes_dict, stock_ids, initial_capital, feat_names, dates = (
            prices_dict,
            volumes_dict,
            stock_ids,
            initial_capital,
            feat_names,
            dates,
            maybe_dates,
        )

    n_stocks = len(stock_ids)
    n_steps  = min(len(v) for v in feat_dict.values())

    # ── 標準化特徵 ───────────────────────────────────────────────────────
    scaled = {}
    for sid in OBSERVABLE_STOCKS:
        if sid not in feat_dict:
            continue
        feat = feat_dict[sid].values[:n_steps].copy().astype(np.float64)
        feat = np.where(np.isposinf(feat),  10.0, feat)
        feat = np.where(np.isneginf(feat), -10.0, feat)
        feat = np.where(np.isnan(feat),      0.0, feat)
        scaled[sid] = np.clip(feat, -10.0, 10.0).astype(np.float32)

    # ── 預計算 20 日滾動平均成交量（與 PortfolioEnv 一致）────────────────
    avg_vol = {}
    for sid in stock_ids:
        vol_arr = volumes_dict[sid][:n_steps].astype(np.float64)
        avg_vol[sid] = (pd.Series(vol_arr)
                        .rolling(AVG_VOL_WINDOW, min_periods=1)
                        .mean()
                        .values)

    # ── 初始化 ──────────────────────────────────────────────────────────
    capital    = float(initial_capital)
    lots_held  = np.zeros(n_stocks, dtype=np.int64)
    odd_held   = np.zeros(n_stocks, dtype=np.int64)
    lot_cost   = np.zeros(n_stocks, dtype=np.float64)
    odd_cost   = np.zeros(n_stocks, dtype=np.float64)

    portfolio_curve = [capital]
    trade_log       = []
    all_actions     = []
    stock_name      = {s["id"]: s["name"] for s in STOCK_POOL}

    # ── 回測迴圈 ─────────────────────────────────────────────────────────
    # logit_state：供 LogitDelta actor 跨步累積 Leaky Integrator 狀態
    # Dirichlet actor 的 sample() 接受但忽略此參數，兩者介面統一
    logit_state = torch.zeros(1, N_ACTIONS, dtype=torch.float32, device=DEVICE)

    for i in range(n_steps - 1):
        prices_T  = np.array([prices_dict[sid][i]     for sid in stock_ids],
                              dtype=np.float64)
        prices_T1 = np.array([prices_dict[sid][i + 1] for sid in stock_ids],
                              dtype=np.float64)
        avg_vol_T = np.array([avg_vol[sid][i]          for sid in stock_ids],
                              dtype=np.float64)

        feat_vec    = np.concatenate([scaled[sid][i] for sid in OBSERVABLE_STOCKS])
        lot_val     = lots_held * LOT_SIZE * prices_T
        odd_val     = odd_held  * prices_T
        total_asset = capital + lot_val.sum() + odd_val.sum()
        lot_ratio   = lot_val / (total_asset + 1e-8)
        odd_ratio_v = odd_val / (total_asset + 1e-8)
        cash_ratio  = capital / (total_asset + 1e-8)
        obs = np.concatenate([feat_vec, lot_ratio, odd_ratio_v, [cash_ratio]])
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)

        s = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # w：(1, N_ACTIONS) 當步動作權重
            # logit_state：(1, N_ACTIONS) 更新後的 Leaky Integrator 狀態，下步傳入
            # Dirichlet actor 忽略 logit_state 參數，回傳 mean_action 作為第二值
            w, logit_state, _ = actor.sample(s, logit_state=logit_state)

        # 取前 n_stocks 維作為股票倉位，squeeze(0) 確保只壓 batch 維度
        target = np.clip(
            w.squeeze(0).cpu().numpy().astype(np.float64)[:n_stocks], 0.0, 1.0
        )

        all_actions.append(target.copy())
        date = dates[i] if i < len(dates) else ""

        target_value  = total_asset * target
        target_shares = target_value / (prices_T + 1e-8)
        target_lots   = (target_shares // LOT_SIZE).astype(np.int64)
        target_odd    = (target_shares  % LOT_SIZE).astype(np.int64)

        # ── 賣出 ────────────────────────────────────────────────────────
        for j, sid in enumerate(stock_ids):
            price = float(prices_T[j])
            av    = float(avg_vol_T[j])
            name  = stock_name.get(sid, sid)

            sell_lots = max(0, int(lots_held[j]) - int(target_lots[j]))
            if sell_lots > 0:
                gross    = sell_lots * LOT_SIZE * price
                fee      = _lot_fee(gross)
                tax      = _tax(gross)
                proceeds = gross - fee - tax
                profit   = round(proceeds - sell_lots * LOT_SIZE * lot_cost[j], 0)
                lots_held[j] -= sell_lots
                capital      += proceeds
                trade_log.append({
                    "date": date, "stock": sid, "stock_name": name,
                    "action": f"賣出整張×{sell_lots}",
                    "price": round(price, 2), "shares": int(sell_lots * LOT_SIZE),
                    "amount": round(proceeds, 0), "fee": round(fee + tax, 0),
                    "profit": int(profit), "position": round(float(target[j]), 3),
                })

            sell_odd = max(0, int(odd_held[j]) - int(target_odd[j]))
            if sell_odd > 0:
                filled = _odd_fill(sell_odd, av)
                if filled > 0:
                    gross    = filled * price
                    fee      = _odd_fee(gross)
                    tax      = _tax(gross)
                    proceeds = gross - fee - tax
                    profit   = round(proceeds - filled * odd_cost[j], 0)
                    odd_held[j] -= filled
                    capital     += proceeds
                    trade_log.append({
                        "date": date, "stock": sid, "stock_name": name,
                        "action": f"賣出零股×{filled}",
                        "price": round(price, 2), "shares": int(filled),
                        "amount": round(proceeds, 0), "fee": round(fee + tax, 0),
                        "profit": int(profit), "position": round(float(target[j]), 3),
                    })

        # ── 買入 ────────────────────────────────────────────────────────
        for j, sid in enumerate(stock_ids):
            price = float(prices_T[j])
            av    = float(avg_vol_T[j])
            name  = stock_name.get(sid, sid)

            buy_lots = max(0, int(target_lots[j]) - int(lots_held[j]))
            if buy_lots > 0:
                cost_per_lot = LOT_SIZE * price + _lot_fee(LOT_SIZE * price)
                affordable   = int(capital // cost_per_lot)
                buy_lots     = min(buy_lots, affordable)

                if buy_lots > 0:
                    gross      = buy_lots * LOT_SIZE * price
                    fee        = _lot_fee(gross)
                    old_shares = lots_held[j] * LOT_SIZE
                    lot_cost[j] = (lot_cost[j] * old_shares + gross) / (
                        old_shares + buy_lots * LOT_SIZE + 1e-8)
                    lots_held[j] += buy_lots
                    capital      -= gross + fee
                    trade_log.append({
                        "date": date, "stock": sid, "stock_name": name,
                        "action": f"買入整張×{buy_lots}",
                        "price": round(price, 2), "shares": int(buy_lots * LOT_SIZE),
                        "amount": round(gross, 0), "fee": round(fee, 0),
                        "profit": None, "position": round(float(target[j]), 3),
                    })
                else:
                    residual_budget = capital * 0.8
                    extra_odd = min(int(residual_budget / (price + 1e-8)), LOT_SIZE - 1)
                    if extra_odd > 0:
                        filled = _odd_fill(extra_odd, av)
                        if filled > 0:
                            gross      = filled * price
                            fee        = _odd_fee(gross)
                            total_cost = gross + fee
                            if total_cost <= capital:
                                old_odd = odd_held[j]
                                odd_cost[j] = (odd_cost[j] * old_odd + gross) / (
                                    old_odd + filled + 1e-8)
                                odd_held[j] += filled
                                capital     -= total_cost
                                trade_log.append({
                                    "date": date, "stock": sid, "stock_name": name,
                                    "action": f"買入零股(補位)×{filled}",
                                    "price": round(price, 2), "shares": int(filled),
                                    "amount": round(gross, 0), "fee": round(fee, 0),
                                    "profit": None, "position": round(float(target[j]), 3),
                                })

            buy_odd = max(0, int(target_odd[j]) - int(odd_held[j]))
            if buy_odd > 0:
                filled = _odd_fill(buy_odd, av)
                if filled > 0:
                    gross      = filled * price
                    fee        = _odd_fee(gross)
                    total_cost = gross + fee
                    if total_cost <= capital:
                        old_odd = odd_held[j]
                        odd_cost[j] = (odd_cost[j] * old_odd + gross) / (
                            old_odd + filled + 1e-8)
                        odd_held[j] += filled
                        capital     -= total_cost
                        trade_log.append({
                            "date": date, "stock": sid, "stock_name": name,
                            "action": f"買入零股×{filled}",
                            "price": round(price, 2), "shares": int(filled),
                            "amount": round(gross, 0), "fee": round(fee, 0),
                            "profit": None, "position": round(float(target[j]), 3),
                        })

        # ── 自動升級 ──────────────────────────────────────────────────────
        for j, sid in enumerate(stock_ids):
            if odd_held[j] >= LOT_SIZE:
                upgrade      = odd_held[j] // LOT_SIZE
                lots_held[j] += upgrade
                odd_held[j]  %= LOT_SIZE

        # ── 結算 T+1 市值 ────────────────────────────────────────────────
        lot_val_T1 = lots_held * LOT_SIZE * prices_T1
        odd_val_T1 = odd_held  * prices_T1
        total_T1   = capital + lot_val_T1.sum() + odd_val_T1.sum()
        portfolio_curve.append(round(total_T1, 0))

        if capital < 0:
            print(f"  *** capital 變負數 step={i} capital={capital:.0f} ***")

    # ── 強制平倉 ─────────────────────────────────────────────────────────
    prices_last = np.array([prices_dict[sid][-1] for sid in stock_ids],
                            dtype=np.float64)
    for j, sid in enumerate(stock_ids):
        name = stock_name.get(sid, sid)
        if lots_held[j] > 0:
            gross    = lots_held[j] * LOT_SIZE * prices_last[j]
            fee      = _lot_fee(gross)
            tax      = _tax(gross)
            proceeds = gross - fee - tax
            profit   = round(proceeds - lots_held[j] * LOT_SIZE * lot_cost[j], 0)
            capital += proceeds
            trade_log.append({
                "date": dates[-1] if dates else "", "stock": sid, "stock_name": name,
                "action": "賣出整張(結算)",
                "price": round(float(prices_last[j]), 2),
                "shares": int(lots_held[j] * LOT_SIZE),
                "amount": round(proceeds, 0), "fee": round(fee + tax, 0),
                "profit": int(profit), "position": 0.0,
            })
        if odd_held[j] > 0:
            gross    = odd_held[j] * prices_last[j]
            fee      = _odd_fee(gross)
            tax      = _tax(gross)
            proceeds = gross - fee - tax
            profit   = round(proceeds - odd_held[j] * odd_cost[j], 0)
            capital += proceeds
            trade_log.append({
                "date": dates[-1] if dates else "", "stock": sid, "stock_name": name,
                "action": "賣出零股(結算)",
                "price": round(float(prices_last[j]), 2),
                "shares": int(odd_held[j]),
                "amount": round(proceeds, 0), "fee": round(fee + tax, 0),
                "profit": int(profit), "position": 0.0,
            })

    portfolio_curve[-1] = round(capital, 0)

    # ── 計算指標 ─────────────────────────────────────────────────────────
    final_capital    = portfolio_curve[-1]
    total_return_pct = round((final_capital / initial_capital - 1) * 100, 2)
    bh_returns       = [
        (prices_dict[sid][-1] / prices_dict[sid][0] - 1) for sid in stock_ids
    ]
    bh_return_pct    = round(float(np.mean(bh_returns)) * 100, 2)
    risk_free_return = round(RISK_FREE_DAILY * n_steps * 100, 2)
    win_rate         = calc_win_rate(trade_log)
    closed           = [
        t for t in trade_log
        if "賣出" in t.get("action", "") and t.get("profit") is not None
    ]

    if all_actions:
        actions_arr = np.array(all_actions)
        # 若各元素長度不一致，np.array 產生 object array → fallback 為零矩陣
        if actions_arr.ndim != 2 or actions_arr.shape[1] != n_stocks:
            actions_arr = np.zeros((1, n_stocks))
    else:
        actions_arr = np.zeros((1, n_stocks))

    avg_positions = {
        sid: round(float(actions_arr[:, j].mean()) * 100, 1)
        for j, sid in enumerate(stock_ids)
    }

    bh_curve = []
    for k in range(len(portfolio_curve)):
        idx   = min(k, n_steps - 1)
        avg_r = np.mean([
            (prices_dict[sid][min(idx, len(prices_dict[sid]) - 1)]
             / prices_dict[sid][0] - 1)
            for sid in stock_ids
        ])
        bh_curve.append(round(initial_capital * (1 + avg_r), 0))

    return sanitize(dict(
        initial_capital  = initial_capital,
        final_capital    = final_capital,
        total_profit     = round(final_capital - initial_capital, 0),
        total_return     = total_return_pct,
        bh_return        = bh_return_pct,
        risk_free_return = risk_free_return,
        win_rate         = win_rate,
        n_trades         = len(closed),
        portfolio_curve  = safe_list(portfolio_curve[:len(dates)]),
        bh_curve         = safe_list(bh_curve),
        dates            = dates[:len(portfolio_curve)],
        trade_log        = trade_log,
        avg_positions    = avg_positions,
        all_actions      = [[safe_float(x) for x in a] for a in all_actions],
    ))
