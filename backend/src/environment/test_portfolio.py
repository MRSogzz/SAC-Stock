"""
tests/src/environment/test_portfolio.py
=========================================
portfolio.py 的單元測試，對應真實實作 PortfolioEnv v3：
  - reset()        → np.ndarray (STATE_DIM,)
  - step(action)   → (obs, reward, done)
  - portfolio_value() → float

Mock 策略：
  - OBSERVABLE_STOCKS / TRADEABLE_STOCKS / BENCHMARK_STOCK → patch configs
  - N_FEATURES / STATE_DIM / LOT_SIZE 等常數 → patch configs
  - CompositeReward → 部分測試 mock，避免循環依賴
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# ── 測試用常數（與 configs 解耦）──────────────────────────────────────────────

LOT_SIZE         = 1000
N_FEATURES       = 31          # 每支股票的特徵數
N_TRADEABLE      = 3           # 測試用縮小規模
N_OBSERVABLE     = 4           # 含 benchmark
STATE_DIM        = N_FEATURES * N_OBSERVABLE + N_TRADEABLE * 2 + 1  # 31*4+3*2+1=131
TRADEABLE        = ["2330", "2317", "2454"]
OBSERVABLE       = ["2330", "2317", "2454", "0050"]
BENCHMARK        = "0050"
INITIAL_CAPITAL  = 1_000_000
N_STEPS          = 300


# ── 假資料工廠 ────────────────────────────────────────────────────────────────

def _make_features_dict(n: int = N_STEPS, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        sid: pd.DataFrame(
            rng.standard_normal((n, N_FEATURES)).astype(np.float32),
            columns=[f"f{i}" for i in range(N_FEATURES)],
        )
        for sid in OBSERVABLE
    }


def _make_prices_dict(n: int = N_STEPS, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    result = {}
    for sid in OBSERVABLE:
        base   = rng.uniform(100, 500)
        prices = base + rng.standard_normal(n).cumsum()
        result[sid] = np.abs(prices) + 10.0  # 確保全正
    return result


def _make_volumes_dict(n: int = N_STEPS, seed: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    return {
        sid: rng.integers(10_000, 500_000, n).astype(float)
        for sid in OBSERVABLE
    }


def _make_action(equal_weight: bool = True) -> np.ndarray:
    if equal_weight:
        return np.ones(N_TRADEABLE) / N_TRADEABLE
    return np.array([0.5, 0.3, 0.2])


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config_patch():
    """patch 所有 configs 常數，讓測試不依賴真實設定檔。"""
    patches = {
        "src.environment.portfolio.OBSERVABLE_STOCKS":  OBSERVABLE,
        "src.environment.portfolio.TRADEABLE_STOCKS":   TRADEABLE,
        "src.environment.portfolio.BENCHMARK_STOCK":    BENCHMARK,
        "src.environment.portfolio.N_FEATURES":         N_FEATURES,
        "src.environment.portfolio.STATE_DIM":          STATE_DIM,
        "src.environment.portfolio.LOT_SIZE":           LOT_SIZE,
        "src.environment.portfolio.MIN_FEE_LOT":        20,
        "src.environment.portfolio.MIN_FEE_ODD":        1,
        "src.environment.portfolio.BROKER_FEE":         0.001425,
        "src.environment.portfolio.SECURITY_TAX":       0.003,
        "src.environment.portfolio.ODD_FILL_RATIO":     0.65,
        "src.environment.portfolio.AVG_VOL_WINDOW":     20,
    }
    with patch.multiple("src.environment.portfolio", **{
        k.split(".")[-1]: v for k, v in patches.items()
    }):
        yield


@pytest.fixture
def env(config_patch):
    from src.environment.portfolio import PortfolioEnv
    feat    = _make_features_dict()
    prices  = _make_prices_dict()
    volumes = _make_volumes_dict()
    e = PortfolioEnv(
        features_dict   = feat,
        prices_dict     = prices,
        volumes_dict    = volumes,
        initial_capital = INITIAL_CAPITAL,
    )
    e.reset()
    return e


# ═══════════════════════════════════════════════════════════════════════════════
# __init__ 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioEnvInit:

    def test_length_mismatch_raises(self, config_patch):
        """T1：特徵/價格/成交量長度不一致 → raise ValueError。"""
        from src.environment.portfolio import PortfolioEnv

        feat    = _make_features_dict(n=300)
        prices  = _make_prices_dict(n=300)
        volumes = _make_volumes_dict(n=300)

        # 讓其中一支股票的特徵長度不同
        feat["2330"] = feat["2330"].iloc[:250]

        with pytest.raises(ValueError, match="長度不一致"):
            PortfolioEnv(feat, prices, volumes)

    def test_avg_vol_precomputed_for_tradeable(self, config_patch):
        """T2：初始化後 avg_vol 應包含所有可交易股票的預計算陣列。"""
        from src.environment.portfolio import PortfolioEnv

        feat    = _make_features_dict()
        prices  = _make_prices_dict()
        volumes = _make_volumes_dict()
        env     = PortfolioEnv(feat, prices, volumes)

        for sid in TRADEABLE:
            assert sid in env.avg_vol
            assert len(env.avg_vol[sid]) == N_STEPS


# ═══════════════════════════════════════════════════════════════════════════════
# reset() 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestReset:

    def test_reset_returns_ndarray(self, env):
        """T3：reset() 回傳 np.ndarray。"""
        obs = env.reset()
        assert isinstance(obs, np.ndarray)

    def test_reset_obs_shape(self, env):
        """T4：reset() 回傳的 obs shape 應為 (STATE_DIM,)。"""
        obs = env.reset()
        assert obs.shape == (STATE_DIM,)

    def test_reset_capital_restored(self, env):
        """T5：reset() 後現金應回到 initial_capital。"""
        # 先 step 幾步消耗資金
        action = _make_action()
        for _ in range(5):
            env.step(action)
        env.reset()
        assert env.capital == INITIAL_CAPITAL

    def test_reset_holdings_zeroed(self, env):
        """T6：reset() 後整張與零股持倉應全為 0。"""
        action = _make_action()
        for _ in range(5):
            env.step(action)
        env.reset()
        assert np.all(env.lots_held == 0)
        assert np.all(env.odd_held  == 0)

    def test_reset_step_idx_zeroed(self, env):
        """T7：reset() 後 step_idx 應歸 0。"""
        action = _make_action()
        env.step(action)
        env.reset()
        assert env.step_idx == 0

    def test_obs_no_nan(self, env):
        """T8：reset() 回傳的 obs 不含 NaN。"""
        obs = env.reset()
        assert not np.isnan(obs).any()


# ═══════════════════════════════════════════════════════════════════════════════
# step() 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep:

    def test_step_returns_three_tuple(self, env):
        """T9：step() 回傳 (obs, reward, done) 三元組。"""
        result = env.step(_make_action())
        assert len(result) == 3

    def test_step_obs_shape(self, env):
        """T10：step() 回傳的 obs shape 應為 (STATE_DIM,)。"""
        obs, _, _ = env.step(_make_action())
        assert obs.shape == (STATE_DIM,)

    def test_step_reward_is_float(self, env):
        """T11：reward 應為 Python float。"""
        _, reward, _ = env.step(_make_action())
        assert isinstance(reward, float)

    def test_step_reward_clipped(self, env):
        """T12：reward 應在 [-1, 1] 範圍內（CompositeReward 有 clip）。"""
        for _ in range(20):
            _, reward, _ = env.step(_make_action())
            assert -1.0 <= reward <= 1.0, f"reward={reward} 超出 [-1,1]"

    def test_step_done_false_before_end(self, env):
        """T13：未到最後一步時 done 應為 False。"""
        _, _, done = env.step(_make_action())
        assert done is False

    def test_step_done_true_at_end(self, env):
        """T14：資料耗盡時 done 應為 True。"""
        action = _make_action()
        done   = False
        for _ in range(N_STEPS + 10):
            _, _, done = env.step(action)
            if done:
                break
        assert done is True

    def test_step_idx_increments(self, env):
        """T15：每次 step 後 step_idx 應加 1。"""
        env.step(_make_action())
        assert env.step_idx == 1
        env.step(_make_action())
        assert env.step_idx == 2

    def test_lot_mode_integer_shares(self, env):
        """T16：整張模式 → lots_held 應為整數。"""
        action = _make_action()
        for _ in range(10):
            env.step(action)
            if env.step_idx >= N_STEPS - 1:
                break
        assert env.lots_held.dtype == np.int64
        assert np.all(env.lots_held >= 0)

    def test_insufficient_capital_no_negative_cash(self, env):
        """T17：資金不足時 capital 不應變為負數。"""
        # 全部押注，多次交易
        action = np.array([0.8, 0.1, 0.1])
        for _ in range(50):
            env.step(action)
            assert env.capital >= 0.0, f"capital 為負：{env.capital}"
            if env.step_idx >= N_STEPS - 1:
                break

    def test_upgrade_odd_to_lot(self, env):
        """T18：odd_held >= LOT_SIZE 時應自動升級為整張（無成本）。"""
        # 直接手動設定 odd_held 超過閾值
        env.odd_held[:] = LOT_SIZE + 100
        lots_before = env.lots_held.copy()
        env.step(_make_action())
        # 升級後 lots 應增加
        assert np.all(env.lots_held >= lots_before)

    def test_obs_no_nan_after_step(self, env):
        """T19：step() 回傳的 obs 不含 NaN。"""
        obs, _, _ = env.step(_make_action())
        assert not np.isnan(obs).any()

    def test_all_cash_action_no_holdings(self, env):
        """T20：全現金 action（全 0）→ 不買入任何股票，持倉應保持不變或為 0。"""
        env.reset()
        lots_before = env.lots_held.copy()
        env.step(np.zeros(N_TRADEABLE))
        # 全 0 action 不應增加持倉
        assert np.all(env.lots_held <= lots_before)

    def test_multiple_steps_no_crash(self, env):
        """T21：連續多步 step() 不 crash。"""
        action = _make_action()
        for _ in range(30):
            _, _, done = env.step(action)
            if done:
                break


# ═══════════════════════════════════════════════════════════════════════════════
# portfolio_value() 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioValue:

    def test_portfolio_value_initial_is_one(self, env):
        """T22：reset 後 portfolio_value() 應接近 1.0（未交易前）。"""
        env.reset()
        # 初始時 total_asset = initial_capital，normalized = 1.0
        pv = env.portfolio_value()
        assert abs(pv - 1.0) < 0.01

    def test_portfolio_value_returns_float(self, env):
        """T23：portfolio_value() 回傳 Python float。"""
        assert isinstance(env.portfolio_value(), float)