"""
金融工具函數：手續費、整張/零股計算、風險指標
"""
import math
import numpy as np
from configs.trading_config import (
    BROKER_FEE, SECURITY_TAX, MIN_FEE_LOT, MIN_FEE_ODD, LOT_SIZE
)
from diagnostics import register


# ─── 手續費計算 ───────────────────────────────────────────────────────────────

@register(
    module="Utils",
    inputs={
        "amount":  "float",
        "is_sell": "bool",
        "is_odd":  "bool",
    },
    outputs={"return": "float"},
    notes="整張買：ceil(×0.1425%)≥20元；零股買：≥1元；賣出額外加 ceil(×0.3%) 證交稅",
)
def calc_fee(amount: float, is_sell: bool = False,
             is_odd: bool = False) -> float:
    """
    計算實際手續費（無條件進位，符合台股實務）
    整張買入：ceil(amount × 0.1425%)，最低 20 元
    零股買入：ceil(amount × 0.1425%)，最低 1 元
    賣出額外加：ceil(amount × 0.3%)（證交稅）
    """
    min_fee   = MIN_FEE_ODD if is_odd else MIN_FEE_LOT
    brokerage = float(max(math.ceil(amount * BROKER_FEE), min_fee))
    tax       = float(math.ceil(amount * SECURITY_TAX)) if is_sell else 0.0
    return brokerage + tax


@register(
    module="Utils",
    inputs={
        "amount": "float",
        "price":  "float",
    },
    outputs={
        "lots":     "int",
        "odd":      "int",
        "lot_amt":  "float",
        "odd_amt":  "float",
    },
    notes="整張優先，不足一張改零股；回傳 (整張數, 零股數, 整張花費, 零股花費)",
)
def calc_shares(amount: float, price: float) -> tuple:
    """
    根據金額和股價決定買幾股。
    整張優先，不足一張改用零股。
    回傳 (整張數, 零股數, 整張花費, 零股花費)
    """
    if price <= 0 or amount <= 0:
        return 0, 0, 0.0, 0.0

    max_lots = int(amount / (price * LOT_SIZE))
    lot_amt  = max_lots * LOT_SIZE * price if max_lots >= 1 else 0.0

    remaining = amount - lot_amt
    odd_count = int(remaining / price)
    odd_amt   = odd_count * price

    return max_lots, odd_count, lot_amt, odd_amt


# ─── 風險指標 ─────────────────────────────────────────────────────────────────

@register(
    module="Utils",
    inputs={"portfolio_curve": "list[float]"},
    outputs={"return": "float"},
    notes="最大回撤（MDD）= min((arr - peak) / peak)，全為 0 時回傳 0.0",
)
def calc_mdd(portfolio_curve: list) -> float:
    """計算最大回撤（Maximum Drawdown）"""
    if not portfolio_curve:
        return 0.0
    arr  = np.array(portfolio_curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / (peak + 1e-8)
    return float(dd.min())


@register(
    module="Utils",
    inputs={
        "returns":         "list[float]",
        "risk_free_daily": "float",
    },
    outputs={"return": "float"},
    notes="年化夏普比率 = (mean(ex) / std(ex)) × √250；少於 2 筆或 std≈0 回傳 0.0",
)
def calc_sharpe(returns: list, risk_free_daily: float = 0.015 / 250) -> float:
    """計算年化夏普比率"""
    if len(returns) < 2:
        return 0.0
    r   = np.array(returns, dtype=float)
    ex  = r - risk_free_daily
    std = ex.std()
    if std < 1e-8:
        return 0.0
    return float(ex.mean() / std * np.sqrt(250))


@register(
    module="Utils",
    inputs={
        "returns":         "list[float]",
        "risk_free_daily": "float",
    },
    outputs={"return": "float"},
    notes="年化 Sortino = (mean(ex) / std(下行)) × √250；無下行波動回傳 inf",
)
def calc_sortino(returns: list, risk_free_daily: float = 0.015 / 250) -> float:
    """計算年化 Sortino Ratio（只考慮下行波動）"""
    if len(returns) < 2:
        return 0.0
    r      = np.array(returns, dtype=float)
    ex     = r - risk_free_daily
    down   = ex[ex < 0]
    if len(down) == 0:
        return float("inf")
    std_dn = down.std()
    if std_dn < 1e-8:
        return 0.0
    return float(ex.mean() / std_dn * np.sqrt(250))


@register(
    module="Utils",
    inputs={"trade_log": "list[dict]"},
    outputs={"return": "float"},
    notes="從交易記錄計算勝率（%）；只計算含 profit 的賣出記錄",
)
def calc_win_rate(trade_log: list) -> float:
    """從交易記錄計算勝率"""
    closed = [t for t in trade_log
              if "賣出" in t.get("action", "") and t.get("profit") is not None]
    if not closed:
        return 0.0
    wins = sum(1 for t in closed if t["profit"] > 0)
    return round(wins / len(closed) * 100, 1)


@register(
    module="Utils",
    inputs={
        "total_return": "float",
        "n_days":       "int",
    },
    outputs={"return": "float"},
    notes="累積報酬率（%）→ 年化報酬率（%），假設一年 250 個交易日",
)
def annualize_return(total_return: float, n_days: int) -> float:
    """把累積報酬率換算成年化報酬率"""
    if n_days <= 0:
        return 0.0
    return round(((1 + total_return / 100) ** (250 / n_days) - 1) * 100, 2)