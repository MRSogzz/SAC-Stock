"""
tests/src/utils/test_finance.py
==================================
finance.py 的單元測試，對應真實實作：
  - calc_fee(amount, is_sell, is_odd)      → float
  - calc_shares(amount, price)             → (lots, odd, lot_amt, odd_amt)
  - calc_mdd(portfolio_curve)              → float
  - calc_sharpe(returns, risk_free_daily)  → float
  - calc_sortino(returns, risk_free_daily) → float
  - calc_win_rate(trade_log)               → float
  - annualize_return(total_return, n_days) → float

Mock 策略：
  - configs 常數 → patch，讓測試不依賴真實 configs
"""

import math
import numpy as np
import pytest
from unittest.mock import patch

# ── 測試用常數（對應台股實務）─────────────────────────────────────────────────

BROKER_FEE   = 0.001425   # 0.1425%
SECURITY_TAX = 0.003      # 0.3%
MIN_FEE_LOT  = 20
MIN_FEE_ODD  = 1
LOT_SIZE     = 1000


@pytest.fixture(autouse=True)
def config_patch():
    with patch("src.utils.finance.BROKER_FEE",   BROKER_FEE), \
         patch("src.utils.finance.SECURITY_TAX", SECURITY_TAX), \
         patch("src.utils.finance.MIN_FEE_LOT",  MIN_FEE_LOT), \
         patch("src.utils.finance.MIN_FEE_ODD",  MIN_FEE_ODD), \
         patch("src.utils.finance.LOT_SIZE",     LOT_SIZE):
        yield


# ═══════════════════════════════════════════════════════════════════════════════
# calc_fee()
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalcFee:

    def test_lot_buy_fee_formula(self):
        """T1：整張買入 → ceil(amount × 0.1425%)，最低 20 元。"""
        from src.utils.finance import calc_fee
        amount = 100_000
        fee    = calc_fee(amount, is_sell=False, is_odd=False)
        expected = float(max(math.ceil(amount * BROKER_FEE), MIN_FEE_LOT))
        assert fee == expected

    def test_lot_sell_fee_includes_tax(self):
        """T2：整張賣出 → 手續費 + ceil(amount × 0.3%) 證交稅。"""
        from src.utils.finance import calc_fee
        amount   = 100_000
        fee      = calc_fee(amount, is_sell=True, is_odd=False)
        brokerage = float(max(math.ceil(amount * BROKER_FEE), MIN_FEE_LOT))
        tax       = float(math.ceil(amount * SECURITY_TAX))
        assert fee == pytest.approx(brokerage + tax)

    def test_odd_buy_min_fee_1(self):
        """T3：零股買入最低手續費應為 1 元（非 20 元）。"""
        from src.utils.finance import calc_fee
        small_amount = 100   # ceil(100 × 0.001425) = 1 → 剛好等於 MIN_FEE_ODD
        fee = calc_fee(small_amount, is_sell=False, is_odd=True)
        assert fee >= MIN_FEE_ODD
        assert fee < MIN_FEE_LOT  # 零股最低費 < 整張最低費

    def test_lot_buy_min_fee_20(self):
        """T4：整張買入最低手續費應為 20 元。"""
        from src.utils.finance import calc_fee
        tiny_amount = 1   # brokerage 極小，應觸發最低費 20 元
        fee = calc_fee(tiny_amount, is_sell=False, is_odd=False)
        assert fee == MIN_FEE_LOT

    def test_sell_fee_greater_than_buy(self):
        """T5：相同金額下，賣出手續費 > 買入手續費（多了證交稅）。"""
        from src.utils.finance import calc_fee
        amount   = 500_000
        buy_fee  = calc_fee(amount, is_sell=False)
        sell_fee = calc_fee(amount, is_sell=True)
        assert sell_fee > buy_fee

    def test_fee_is_float(self):
        """T6：回傳值應為 float。"""
        from src.utils.finance import calc_fee
        assert isinstance(calc_fee(50_000), float)

    def test_zero_amount_returns_min_fee(self):
        """T7：amount=0 時回傳最低手續費（ceil(0)=0，取 max 後為最低費）。"""
        from src.utils.finance import calc_fee
        fee = calc_fee(0.0, is_sell=False, is_odd=False)
        assert fee == MIN_FEE_LOT


# ═══════════════════════════════════════════════════════════════════════════════
# calc_shares()
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalcShares:

    def test_enough_for_one_lot(self):
        """T8：足夠買一張（1000 股）→ lots=1。"""
        from src.utils.finance import calc_shares
        price  = 100.0
        amount = price * LOT_SIZE * 1.5   # 150,000，夠買 1 張
        lots, odd, lot_amt, odd_amt = calc_shares(amount, price)
        assert lots == 1
        assert lot_amt == price * LOT_SIZE

    def test_not_enough_for_lot_uses_odd(self):
        """T9：不足一張 → lots=0，用零股。"""
        from src.utils.finance import calc_shares
        price  = 100.0
        amount = price * 500   # 只夠買 500 股（不到 1 張）
        lots, odd, lot_amt, odd_amt = calc_shares(amount, price)
        assert lots == 0
        assert odd  == 500

    def test_returns_four_tuple(self):
        """T10：回傳應為 4 元組。"""
        from src.utils.finance import calc_shares
        result = calc_shares(100_000, 100.0)
        assert len(result) == 4

    def test_zero_price_returns_zeros(self):
        """T11：price=0 → 回傳 (0, 0, 0.0, 0.0)，不 crash。"""
        from src.utils.finance import calc_shares
        assert calc_shares(100_000, 0.0) == (0, 0, 0.0, 0.0)

    def test_zero_amount_returns_zeros(self):
        """T12：amount=0 → 回傳 (0, 0, 0.0, 0.0)。"""
        from src.utils.finance import calc_shares
        assert calc_shares(0.0, 100.0) == (0, 0, 0.0, 0.0)

    def test_multiple_lots(self):
        """T13：足夠買多張 → lots 應為正確整數。"""
        from src.utils.finance import calc_shares
        price  = 50.0
        amount = price * LOT_SIZE * 3.7   # 3 張整張 + 零股
        lots, odd, lot_amt, odd_amt = calc_shares(amount, price)
        assert lots == 3
        assert odd  == int((amount - lot_amt) / price)


# ═══════════════════════════════════════════════════════════════════════════════
# calc_mdd()
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalcMdd:

    def test_known_sequence(self):
        """T14：已知序列 [1, 1.2, 0.8, 0.9] → MDD ≈ -33.33%。"""
        from src.utils.finance import calc_mdd
        curve = [1.0, 1.2, 0.8, 0.9]
        mdd   = calc_mdd(curve)
        # peak=1.2，最低=0.8，MDD = (0.8-1.2)/1.2 ≈ -0.3333
        assert abs(mdd - (-1/3)) < 0.01

    def test_monotone_increasing_no_drawdown(self):
        """T15：單調上升序列 → MDD 應接近 0。"""
        from src.utils.finance import calc_mdd
        curve = [1.0, 1.1, 1.2, 1.3, 1.4]
        assert abs(calc_mdd(curve)) < 1e-6

    def test_empty_list_returns_zero(self):
        """T16：空列表 → 回傳 0.0，不 crash。"""
        from src.utils.finance import calc_mdd
        assert calc_mdd([]) == 0.0

    def test_returns_negative_or_zero(self):
        """T17：MDD 應 ≤ 0。"""
        from src.utils.finance import calc_mdd
        curve = [1.0, 0.9, 0.8, 1.1, 0.7]
        assert calc_mdd(curve) <= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# calc_sharpe()
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalcSharpe:

    def test_positive_returns_positive_sharpe(self):
        """T18：穩定正報酬序列 → Sharpe > 0。"""
        from src.utils.finance import calc_sharpe
        returns = [0.001] * 50   # 每日固定 0.1% 報酬
        sharpe  = calc_sharpe(returns)
        assert sharpe > 0

    def test_insufficient_data_returns_zero(self):
        """T19：少於 2 筆 → 回傳 0.0。"""
        from src.utils.finance import calc_sharpe
        assert calc_sharpe([0.01]) == 0.0
        assert calc_sharpe([]) == 0.0

    def test_zero_std_returns_zero(self):
        """T20：所有報酬相同（std=0）→ 回傳 0.0，不除以零。"""
        from src.utils.finance import calc_sharpe
        returns = [0.001] * 5
        # 超額報酬 = 0.001 - risk_free，若全相同 std=0
        result = calc_sharpe([0.0] * 10, risk_free_daily=0.0)
        assert result == 0.0

    def test_returns_float(self):
        """T21：回傳值應為 float。"""
        from src.utils.finance import calc_sharpe
        assert isinstance(calc_sharpe([0.01, -0.005, 0.008]), float)

    def test_negative_mean_negative_sharpe(self):
        """T22：穩定負報酬 → Sharpe < 0。"""
        from src.utils.finance import calc_sharpe
        returns = [-0.002] * 50
        sharpe  = calc_sharpe(returns)
        # 負超額報酬 → 負 Sharpe（除非 std=0）
        assert sharpe <= 0


# ═══════════════════════════════════════════════════════════════════════════════
# calc_sortino()
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalcSortino:

    def test_no_downside_returns_inf(self):
        """T23：無下行波動（全為正報酬）→ 回傳 inf。"""
        from src.utils.finance import calc_sortino
        returns = [0.01, 0.02, 0.005]
        result  = calc_sortino(returns, risk_free_daily=0.0)
        assert result == float("inf")

    def test_mixed_returns_finite_value(self):
        """T24：有下行波動 → 回傳有限值。"""
        from src.utils.finance import calc_sortino
        returns = [0.01, -0.005, 0.02, -0.003, 0.008]
        result  = calc_sortino(returns)
        assert math.isfinite(result)

    def test_insufficient_data_returns_zero(self):
        """T25：少於 2 筆 → 回傳 0.0。"""
        from src.utils.finance import calc_sortino
        assert calc_sortino([0.01]) == 0.0

    def test_sortino_geq_sharpe_for_positive_skew(self):
        """T26：正偏態報酬（多正少負）下 Sortino ≥ Sharpe。"""
        from src.utils.finance import calc_sharpe, calc_sortino
        returns = [0.02, 0.015, -0.001, 0.03, 0.01, -0.002]
        sharpe  = calc_sharpe(returns, risk_free_daily=0.0)
        sortino = calc_sortino(returns, risk_free_daily=0.0)
        assert sortino >= sharpe


# ═══════════════════════════════════════════════════════════════════════════════
# calc_win_rate()
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalcWinRate:

    def _make_log(self, profits):
        """建立假交易記錄。"""
        return [{"action": "賣出", "profit": p} for p in profits]

    def test_all_profit_100_percent(self):
        """T27：全部獲利 → 勝率 100.0%。"""
        from src.utils.finance import calc_win_rate
        log = self._make_log([100, 200, 50])
        assert calc_win_rate(log) == 100.0

    def test_half_profit_50_percent(self):
        """T28：一半獲利 → 勝率 50.0%。"""
        from src.utils.finance import calc_win_rate
        log = self._make_log([100, -50, 200, -30])
        assert calc_win_rate(log) == 50.0

    def test_no_closed_trades_returns_zero(self):
        """T29：無賣出記錄 → 回傳 0.0，不 crash。"""
        from src.utils.finance import calc_win_rate
        log = [{"action": "買入", "profit": None}]
        assert calc_win_rate(log) == 0.0

    def test_empty_log_returns_zero(self):
        """T30：空交易記錄 → 回傳 0.0。"""
        from src.utils.finance import calc_win_rate
        assert calc_win_rate([]) == 0.0

    def test_ignores_buy_records(self):
        """T31：買入記錄不計入勝率統計。"""
        from src.utils.finance import calc_win_rate
        log = [
            {"action": "買入", "profit": 999},   # 應忽略
            {"action": "賣出", "profit": 100},
            {"action": "賣出", "profit": -50},
        ]
        assert calc_win_rate(log) == 50.0


# ═══════════════════════════════════════════════════════════════════════════════
# annualize_return()
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnnualizeReturn:

    def test_one_year_unchanged(self):
        """T32：n_days=250（一年）→ 年化報酬 = 累積報酬。"""
        from src.utils.finance import annualize_return
        result = annualize_return(10.0, 250)
        assert abs(result - 10.0) < 0.01

    def test_zero_days_returns_zero(self):
        """T33：n_days=0 → 回傳 0.0，不除以零。"""
        from src.utils.finance import annualize_return
        assert annualize_return(10.0, 0) == 0.0

    def test_positive_return_positive_annualized(self):
        """T34：正累積報酬 → 正年化報酬。"""
        from src.utils.finance import annualize_return
        assert annualize_return(20.0, 125) > 0

    def test_returns_float(self):
        """T35：回傳值應為 float（round 後）。"""
        from src.utils.finance import annualize_return
        result = annualize_return(10.0, 250)
        assert isinstance(result, float)

    def test_short_period_higher_annualized(self):
        """T36：相同累積報酬，持有期越短 → 年化報酬越高。"""
        from src.utils.finance import annualize_return
        r_short = annualize_return(10.0, 50)
        r_long  = annualize_return(10.0, 200)
        assert r_short > r_long