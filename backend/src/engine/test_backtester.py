"""
tests/src/engine/test_backtester.py
=====================================
backtester.py 的單元測試，對應真實實作 run_backtest()。

測試策略：
  - actor → 使用假 actor（固定均分 action），行為可預測
  - configs → 全部 patch，不依賴真實設定
  - 數據 → 小規模假資料（N_STEPS=100，N_STOCKS=2），執行快
"""

import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch, MagicMock

# ── 測試用常數 ────────────────────────────────────────────────────────────────

N_STEPS    = 100
N_STOCKS   = 2          # 只用 2 支可交易股票，加速測試
N_OBS      = 3          # 含 benchmark（0050）
N_FEATURES = 31
LOT_SIZE   = 1000
BROKER_FEE   = 0.001425
SECURITY_TAX = 0.003
MIN_FEE_LOT  = 20
MIN_FEE_ODD  = 1
ODD_FILL_RATIO = 0.65
RISK_FREE_DAILY = 0.015 / 250
INITIAL_CAPITAL = 1_000_000

TRADEABLE   = ["2330", "2317"]
OBSERVABLE  = ["2330", "2317", "0050"]
STOCK_POOL  = [{"id": "2330", "name": "台積電"}, {"id": "2317", "name": "鴻海"},
               {"id": "0050", "name": "元大台灣50"}]


# ── 假資料工廠 ────────────────────────────────────────────────────────────────

def _make_prices(n: int = N_STEPS, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    result = {}
    for i, sid in enumerate(OBSERVABLE):
        base   = rng.uniform(100, 300)
        prices = base + rng.standard_normal(n).cumsum()
        result[sid] = np.abs(prices) + 10.0
    return result


def _make_feat_dict(n: int = N_STEPS) -> dict:
    return {
        sid: pd.DataFrame(
            np.random.default_rng(i).standard_normal((n, N_FEATURES)).astype(np.float32),
            columns=[f"f{j}" for j in range(N_FEATURES)],
        )
        for i, sid in enumerate(OBSERVABLE)
    }


def _make_volumes(n: int = N_STEPS) -> dict:
    rng = np.random.default_rng(99)
    return {
        sid: rng.integers(10_000, 500_000, n).astype(float)
        for sid in OBSERVABLE
    }


def _make_scalers() -> dict:
    """回傳恆等變換的假 scaler（transform 原樣回傳）。"""
    scalers = {}
    for sid in OBSERVABLE:
        scaler = MagicMock()
        scaler.transform = lambda x: x
        scalers[sid] = scaler
    return scalers


def _make_actor(n_stocks: int = N_STOCKS) -> MagicMock:
    """
    假 actor：固定回傳均分 action（每股 1/N_STOCKS）。
    mean_action shape = (1, N_STOCKS)
    """
    actor = MagicMock()
    equal = torch.full((1, n_stocks), 1.0 / n_stocks)

    def fake_sample(obs):
        return equal, torch.zeros(1, 1), equal

    actor.sample = fake_sample
    return actor


def _make_dates(n: int = N_STEPS) -> list:
    return pd.date_range("2023-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def config_patch():
    with patch("src.engine.backtester.OBSERVABLE_STOCKS", OBSERVABLE), \
         patch("src.engine.backtester.TRADEABLE_STOCKS",  TRADEABLE), \
         patch("src.engine.backtester.STOCK_POOL",        STOCK_POOL), \
         patch("src.engine.backtester.LOT_SIZE",          LOT_SIZE), \
         patch("src.engine.backtester.BROKER_FEE",        BROKER_FEE), \
         patch("src.engine.backtester.SECURITY_TAX",      SECURITY_TAX), \
         patch("src.engine.backtester.MIN_FEE_LOT",       MIN_FEE_LOT), \
         patch("src.engine.backtester.MIN_FEE_ODD",       MIN_FEE_ODD), \
         patch("src.engine.backtester.ODD_FILL_RATIO",    ODD_FILL_RATIO), \
         patch("src.engine.backtester.RISK_FREE_DAILY",   RISK_FREE_DAILY), \
         patch("src.engine.backtester.AVG_VOL_WINDOW",    20), \
         patch("src.engine.backtester.DEVICE",            torch.device("cpu")):
        yield


@pytest.fixture
def backtest_result():
    """執行一次標準回測，供多個測試共用。"""
    from src.engine.backtester import run_backtest
    return run_backtest(
        actor           = _make_actor(),
        scalers         = _make_scalers(),
        feat_dict       = _make_feat_dict(),
        prices_dict     = _make_prices(),
        volumes_dict    = _make_volumes(),
        stock_ids       = TRADEABLE,
        initial_capital = INITIAL_CAPITAL,
        feat_names      = [f"f{i}" for i in range(N_FEATURES)],
        dates           = _make_dates(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 回傳結構測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestReturnStructure:

    REQUIRED_KEYS = {
        "initial_capital", "final_capital", "total_profit",
        "total_return", "bh_return", "risk_free_return",
        "win_rate", "n_trades", "portfolio_curve",
        "bh_curve", "dates", "trade_log", "avg_positions", "all_actions",
    }

    def test_returns_dict(self, backtest_result):
        """T1：run_backtest() 回傳 dict。"""
        assert isinstance(backtest_result, dict)

    def test_all_required_keys_present(self, backtest_result):
        """T2：回傳 dict 包含所有必要 key。"""
        missing = self.REQUIRED_KEYS - set(backtest_result.keys())
        assert not missing, f"缺少 key：{missing}"

    def test_portfolio_curve_is_list(self, backtest_result):
        """T3：portfolio_curve 應為 list。"""
        assert isinstance(backtest_result["portfolio_curve"], list)

    def test_trade_log_is_list(self, backtest_result):
        """T4：trade_log 應為 list。"""
        assert isinstance(backtest_result["trade_log"], list)

    def test_all_actions_is_list(self, backtest_result):
        """T5：all_actions 應為 list of list。"""
        aa = backtest_result["all_actions"]
        assert isinstance(aa, list)
        if aa:
            assert isinstance(aa[0], list)


# ═══════════════════════════════════════════════════════════════════════════════
# 財務邏輯測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestFinancialLogic:

    def test_no_nan_in_metrics(self, backtest_result):
        """T6：所有數值指標不含 NaN（sanitize 應已處理）。"""
        for key in ["total_return", "bh_return", "win_rate", "risk_free_return"]:
            val = backtest_result[key]
            assert val is not None
            assert not (isinstance(val, float) and (val != val))  # NaN check

    def test_portfolio_curve_no_nan(self, backtest_result):
        """T7：portfolio_curve 不含 NaN 或 None。"""
        for v in backtest_result["portfolio_curve"]:
            assert v is not None
            assert isinstance(v, (int, float))

    def test_final_capital_positive(self, backtest_result):
        """T8：final_capital 應為正數（即使虧損也不應為負）。"""
        assert backtest_result["final_capital"] > 0

    def test_total_profit_consistent(self, backtest_result):
        """T9：total_profit = final_capital - initial_capital。"""
        expected = backtest_result["final_capital"] - backtest_result["initial_capital"]
        assert abs(backtest_result["total_profit"] - expected) < 1.0

    def test_total_return_consistent(self, backtest_result):
        """T10：total_return(%) 應與 final/initial 一致。"""
        expected = (backtest_result["final_capital"] /
                    backtest_result["initial_capital"] - 1) * 100
        assert abs(backtest_result["total_return"] - expected) < 0.1

    def test_n_trades_matches_trade_log(self, backtest_result):
        """T11：n_trades 應等於 trade_log 中有 profit 的賣出筆數。"""
        closed = [
            t for t in backtest_result["trade_log"]
            if "賣出" in t.get("action", "") and t.get("profit") is not None
        ]
        assert backtest_result["n_trades"] == len(closed)

    def test_win_rate_in_valid_range(self, backtest_result):
        """T12：win_rate 應在 [0, 100] 範圍內。"""
        wr = backtest_result["win_rate"]
        assert 0.0 <= wr <= 100.0

    def test_portfolio_curve_length(self, backtest_result):
        """T13：portfolio_curve 長度應與 dates 長度一致。"""
        pc_len = len(backtest_result["portfolio_curve"])
        dt_len = len(backtest_result["dates"])
        assert pc_len == dt_len, f"portfolio_curve({pc_len}) vs dates({dt_len})"


# ═══════════════════════════════════════════════════════════════════════════════
# 交易邏輯測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestTradingLogic:

    def test_trade_log_records_have_required_fields(self, backtest_result):
        """T14：每筆 trade_log 記錄應包含必要欄位。"""
        required = {"date", "stock", "action", "price", "shares", "amount", "fee"}
        for record in backtest_result["trade_log"]:
            missing = required - set(record.keys())
            assert not missing, f"trade_log 記錄缺少欄位：{missing}"

    def test_all_actions_each_step_sums_to_leq_1(self, backtest_result):
        """T15：每步 action 總和應 ≤ 1.0（softmax 截取）。"""
        for a in backtest_result["all_actions"]:
            assert sum(a) <= 1.0 + 1e-5, f"action 總和超過 1：{sum(a)}"

    def test_sell_records_have_profit(self, backtest_result):
        """T16：賣出記錄（非 None profit）應有整數 profit 值。"""
        sells = [t for t in backtest_result["trade_log"]
                 if "賣出" in t.get("action", "") and t.get("profit") is not None]
        for t in sells:
            assert isinstance(t["profit"], (int, float))

    def test_buy_records_profit_is_none(self, backtest_result):
        """T17：買入記錄的 profit 應為 None（尚未實現損益）。"""
        buys = [t for t in backtest_result["trade_log"] if "買入" in t.get("action", "")]
        for t in buys:
            assert t.get("profit") is None, f"買入記錄不應有 profit：{t}"

    def test_avg_positions_keys_match_stock_ids(self, backtest_result):
        """T18：avg_positions 的 key 應與 stock_ids 一致。"""
        assert set(backtest_result["avg_positions"].keys()) == set(TRADEABLE)

    def test_avg_positions_values_between_0_and_100(self, backtest_result):
        """T19：avg_positions 的值應在 [0, 100]（百分比）。"""
        for sid, pct in backtest_result["avg_positions"].items():
            assert 0.0 <= pct <= 100.0, f"{sid} avg_position={pct} 超出合理範圍"


# ═══════════════════════════════════════════════════════════════════════════════
# 邊界條件與穩健性測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestRobustness:

    def test_hold_cash_actor_low_return(self):
        """T20：全現金 actor（action 全 0）→ 不交易，final_capital 接近 initial。"""
        from src.engine.backtester import run_backtest

        zero_actor = MagicMock()
        zero_actor.sample = lambda obs: (
            torch.zeros(1, N_STOCKS),
            torch.zeros(1, 1),
            torch.zeros(1, N_STOCKS),
        )

        result = run_backtest(
            actor           = zero_actor,
            scalers         = _make_scalers(),
            feat_dict       = _make_feat_dict(),
            prices_dict     = _make_prices(),
            volumes_dict    = _make_volumes(),
            stock_ids       = TRADEABLE,
            initial_capital = INITIAL_CAPITAL,
            feat_names      = [],
            dates           = _make_dates(),
        )
        # 全現金策略：不交易，final_capital ≈ initial_capital
        assert result["n_trades"] == 0
        assert abs(result["final_capital"] - INITIAL_CAPITAL) < 1.0

    def test_bh_curve_same_length_as_portfolio_curve(self, backtest_result):
        """T21：bh_curve 長度應與 portfolio_curve 相同。"""
        assert len(backtest_result["bh_curve"]) == len(backtest_result["portfolio_curve"])

    def test_no_inf_in_portfolio_curve(self, backtest_result):
        """T22：portfolio_curve 不含 inf。"""
        import math
        for v in backtest_result["portfolio_curve"]:
            assert math.isfinite(v), f"portfolio_curve 含有 inf：{v}"

    def test_sanitize_applied_no_raw_numpy(self, backtest_result):
        """T23：sanitize 後不應含有 np.ndarray 或 np.integer 型別。"""
        def _check(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    _check(v)
            elif isinstance(obj, list):
                for item in obj:
                    _check(item)
            else:
                assert not isinstance(obj, (np.ndarray, np.integer, np.floating)), (
                    f"sanitize 未清理：{type(obj)} = {obj}"
                )

        _check(backtest_result)

    def test_different_initial_capital_scales_curve(self):
        """T24：initial_capital 翻倍 → final_capital 應等比例成長（相同策略下）。"""
        from src.engine.backtester import run_backtest

        actor    = _make_actor()
        feat     = _make_feat_dict()
        prices   = _make_prices()
        volumes  = _make_volumes()
        scalers  = _make_scalers()
        dates    = _make_dates()

        r1 = run_backtest(actor, scalers, feat, prices, volumes,
                          TRADEABLE, INITIAL_CAPITAL, [], dates)
        r2 = run_backtest(_make_actor(), _make_scalers(), _make_feat_dict(),
                          _make_prices(), _make_volumes(),
                          TRADEABLE, INITIAL_CAPITAL * 2, [], dates)

        # 報酬率（%）應相近（允許 ±5% 差異，因手續費比例略有不同）
        assert abs(r1["total_return"] - r2["total_return"]) < 5.0