"""
tests/src/environment/test_reward.py
======================================
reward.py 的單元測試，對應真實實作 CompositeReward v9：
  - __init__()  → 初始化狀態
  - reset()     → 重置歷史狀態
  - compute()   → float，clip 到 [-1, 1]

8 項獎勵組成：
  1. pnl        = log(total_T1_pre / total_T_pre)
  2. dd_penalty = 0.1 * max(0, drawdown - MDD_FLOOR)
  3. alpha      = tanh(excess / ALPHA_SIGMA) * ALPHA_SCALE
  4. smooth     = LAMBDA * ||target - prev_target||²
  5. odd_penalty = ODD_PENALTY_RATE * odd_ratio
  6. hhi_penalty = HHI_SCALE * max(0, HHI - HHI_THRESHOLD)
  合計 × 5，clip(-1, 1)

v9 變更對測試的影響：
  - T3：portfolio_hist 初始值從 1.0 改為 initial_capital（float）
  - T15：MDD 計算改為絕對資產，不再需要手動重置 total_asset 製造假 peak；
         改用「連續上漲 → 單步大回撤」的真實場景驗證
  - T17：portfolio_hist 初始值型態不影響長度計數，測試邏輯不變

Mock 策略：
  - configs 常數 → patch，讓測試不依賴真實設定
"""

import numpy as np
import pytest
from unittest.mock import patch

# ── 測試用常數 ────────────────────────────────────────────────────────────────

N_STOCKS         = 3
INITIAL_CAPITAL  = 1_000_000

# 對應 reward.py 內的常數
MDD_WINDOW           = 60
ACTION_SMOOTH_LAMBDA = 0.05   # v9：從 0.01 更新為 0.05
ALPHA_SIGMA          = 0.01
ALPHA_SCALE          = 0.05
MDD_FLOOR            = 0.05
ODD_PENALTY_RATE     = 0.0005
HHI_THRESHOLD        = 0.50
HHI_SCALE            = 0.02


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def config_patch():
    """patch configs，讓 CompositeReward 使用測試常數。"""
    with patch("src.environment.reward.MDD_WINDOW",           MDD_WINDOW), \
         patch("src.environment.reward.ACTION_SMOOTH_LAMBDA", ACTION_SMOOTH_LAMBDA), \
         patch("src.environment.reward.ALPHA_SIGMA",          ALPHA_SIGMA), \
         patch("src.environment.reward.ALPHA_SCALE",          ALPHA_SCALE), \
         patch("src.environment.reward.MDD_FLOOR",            MDD_FLOOR), \
         patch("src.environment.reward.ODD_PENALTY_RATE",     ODD_PENALTY_RATE), \
         patch("src.environment.reward.HHI_THRESHOLD",        HHI_THRESHOLD), \
         patch("src.environment.reward.HHI_SCALE",            HHI_SCALE):
        yield


@pytest.fixture
def reward_fn():
    from src.environment.reward import CompositeReward
    fn = CompositeReward()
    fn.reset(N_STOCKS, INITIAL_CAPITAL)
    return fn


def _base_kwargs(
    total_T_pre:   float = 1_000_000,
    total_T1_pre:  float = 1_010_000,   # 正報酬 +1%
    odd_ratio:     float = 0.0,
    port_ret:      float = 0.01,
    benchmark_ret: float = 0.005,
    target:        np.ndarray = None,
) -> dict:
    """回傳一組合理的 compute() 參數，方便各測試覆蓋特定項目。"""
    if target is None:
        target = np.array([0.4, 0.3, 0.3])
    return dict(
        total_T_pre   = total_T_pre,
        total_T1_pre  = total_T1_pre,
        odd_ratio     = odd_ratio,
        port_ret      = port_ret,
        benchmark_ret = benchmark_ret,
        target        = target,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# __init__ / reset() 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompositeRewardInit:

    def test_reset_sets_total_asset(self, reward_fn):
        """T1：reset() 後 total_asset 應等於 initial_capital。"""
        assert reward_fn.total_asset == INITIAL_CAPITAL

    def test_reset_prev_target_zeros(self, reward_fn):
        """T2：reset() 後 prev_target 應為全零陣列，shape (N_STOCKS,)。"""
        assert reward_fn.prev_target is not None
        assert reward_fn.prev_target.shape == (N_STOCKS,)
        assert np.all(reward_fn.prev_target == 0.0)

    def test_reset_portfolio_hist_has_one_entry(self, reward_fn):
        """T3：reset() 後 portfolio_hist 應有 1 個初始值，且為 initial_capital。
        
        v9 修正：初始值從 1.0（比值）改為 initial_capital（絕對資產），
        配合 MDD 計算改為絕對資產追蹤。
        """
        assert len(reward_fn.portfolio_hist) == 1
        assert reward_fn.portfolio_hist[0] == INITIAL_CAPITAL


# ═══════════════════════════════════════════════════════════════════════════════
# compute() 基本測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeBasic:

    def test_returns_float(self, reward_fn):
        """T4：compute() 回傳 Python float。"""
        result = reward_fn.compute(**_base_kwargs())
        assert isinstance(result, float)

    def test_reward_clipped_to_minus1_to_1(self, reward_fn):
        """T5：輸出應在 [-1, 1] 範圍內。"""
        for _ in range(20):
            r = reward_fn.compute(**_base_kwargs())
            assert -1.0 <= r <= 1.0, f"reward={r} 超出 [-1,1]"

    def test_positive_pnl_positive_reward(self, reward_fn):
        """T6：資產成長（total_T1_pre > total_T_pre）→ reward 應為正。"""
        r = reward_fn.compute(**_base_kwargs(
            total_T_pre  = 1_000_000,
            total_T1_pre = 1_050_000,  # +5%
            port_ret     = 0.05,
            benchmark_ret= 0.01,
        ))
        assert r > 0, f"正報酬應有正 reward，得到 {r}"

    def test_loss_negative_reward(self, reward_fn):
        """T7：資產虧損（total_T1_pre < total_T_pre）→ reward 應偏負。"""
        r = reward_fn.compute(**_base_kwargs(
            total_T_pre  = 1_000_000,
            total_T1_pre = 900_000,    # -10%
            port_ret     = -0.10,
            benchmark_ret= 0.01,
        ))
        assert r < 0, f"負報酬應有負 reward，得到 {r}"

    def test_no_nan_output(self, reward_fn):
        """T8：正常輸入不應產生 NaN。"""
        r = reward_fn.compute(**_base_kwargs())
        assert not np.isnan(r), "compute() 不應回傳 NaN"

    def test_extreme_inputs_no_crash(self, reward_fn):
        """T9：極端輸入（全虧、全賺）不 crash，輸出有限值。"""
        extremes = [
            _base_kwargs(total_T_pre=1_000_000, total_T1_pre=1.0,
                         port_ret=-0.999, benchmark_ret=0.01),    # 幾乎全虧
            _base_kwargs(total_T_pre=1_000_000, total_T1_pre=5_000_000,
                         port_ret=4.0, benchmark_ret=0.01),       # 暴漲
        ]
        for kw in extremes:
            r = reward_fn.compute(**kw)
            assert np.isfinite(r), f"極端輸入下 reward 應為有限值：{r}"


# ═══════════════════════════════════════════════════════════════════════════════
# 各項獎勵分項驗證
# ═══════════════════════════════════════════════════════════════════════════════

class TestRewardComponents:

    def test_pnl_log_formulation(self, reward_fn):
        """T10：pnl = log(T1/T)，資產不變時 pnl ≈ 0。"""
        # total_T1_pre == total_T_pre → log(1) = 0
        r_flat = reward_fn.compute(**_base_kwargs(
            total_T_pre  = 1_000_000,
            total_T1_pre = 1_000_000,
            port_ret     = 0.0,
            benchmark_ret= 0.0,
        ))
        # reward 接近 0（其他項也很小）
        assert abs(r_flat) < 0.5, f"資產不變時 reward 應接近 0，得到 {r_flat}"

    def test_alpha_reward_beat_benchmark(self, reward_fn):
        """T11：port_ret > benchmark_ret → alpha 為正，提升 reward。"""
        r_beat = reward_fn.compute(**_base_kwargs(
            port_ret=0.05, benchmark_ret=0.01
        ))
        reward_fn.reset(N_STOCKS, INITIAL_CAPITAL)
        r_lag = reward_fn.compute(**_base_kwargs(
            port_ret=0.01, benchmark_ret=0.05
        ))
        assert r_beat > r_lag, "跑贏大盤應有更高 reward"

    def test_smooth_penalty_larger_for_big_change(self, reward_fn):
        """T12：action 變化越大，smooth_penalty 越大，reward 越低。"""
        # 第一步設定 prev_target
        target_init = np.array([1/3, 1/3, 1/3])
        reward_fn.compute(**_base_kwargs(target=target_init))

        # 小變化
        r_small = reward_fn.compute(**_base_kwargs(
            target=np.array([0.35, 0.33, 0.32])
        ))
        reward_fn.reset(N_STOCKS, INITIAL_CAPITAL)
        reward_fn.compute(**_base_kwargs(target=target_init))

        # 大變化
        r_large = reward_fn.compute(**_base_kwargs(
            target=np.array([0.9, 0.05, 0.05])
        ))
        assert r_small > r_large, "action 大幅改變應有更低 reward（smooth penalty）"

    def test_odd_penalty_increases_with_ratio(self, reward_fn):
        """T13：odd_ratio 越高，odd_penalty 越大，reward 越低。"""
        r_low_odd  = reward_fn.compute(**_base_kwargs(odd_ratio=0.0))
        reward_fn.reset(N_STOCKS, INITIAL_CAPITAL)
        r_high_odd = reward_fn.compute(**_base_kwargs(odd_ratio=0.9))
        assert r_low_odd > r_high_odd, "零股比例越高應有更低 reward"

    def test_hhi_penalty_for_concentrated_portfolio(self, reward_fn):
        """T14：高度集中持倉（HHI > threshold）→ hhi_penalty 生效，reward 較低。"""
        # 分散持倉（HHI 低）
        r_diverse = reward_fn.compute(**_base_kwargs(
            target=np.array([0.34, 0.33, 0.33])
        ))
        reward_fn.reset(N_STOCKS, INITIAL_CAPITAL)

        # 集中持倉（HHI 高）
        r_concentrated = reward_fn.compute(**_base_kwargs(
            target=np.array([0.95, 0.03, 0.02])
        ))
        assert r_diverse > r_concentrated, "集中持倉應有更低 reward（HHI penalty）"

    def test_mdd_penalty_after_drawdown(self, reward_fn):
        """T15：真實累積資產場景下，連續上漲後大幅回撤 → MDD 懲罰應生效。

        v9 修正：MDD 改為追蹤絕對資產值，不再需要手動重置 total_asset 製造假 peak。
        直接模擬真實場景：資產先連續成長（推高 window_peak），
        再單步大幅下跌，驗證 dd_penalty 確實被觸發。
        """
        # 連續正報酬，推高窗口內的絕對資產 peak
        asset = INITIAL_CAPITAL
        for _ in range(10):
            next_asset = asset * 1.01   # 每步 +1%
            reward_fn.compute(**_base_kwargs(
                total_T_pre  = asset,
                total_T1_pre = next_asset,
                port_ret     = 0.01,
                benchmark_ret= 0.005,
            ))
            asset = next_asset          # 累積複利，portfolio_hist 存真實資產值

        # 在已建立的高點之後，單步大幅回撤（-20%），超過 MDD_FLOOR=5%
        r_drawdown = reward_fn.compute(**_base_kwargs(
            total_T_pre  = asset,
            total_T1_pre = asset * 0.80,   # -20%
            port_ret     = -0.20,
            benchmark_ret= 0.01,
        ))
        assert r_drawdown < 0, "大幅回撤後 reward 應為負"

    def test_total_asset_updated_after_compute(self, reward_fn):
        """T16：compute() 後 total_asset 應更新為 total_T1_pre。"""
        new_asset = 1_020_000
        reward_fn.compute(**_base_kwargs(total_T1_pre=new_asset))
        assert reward_fn.total_asset == new_asset

    def test_portfolio_hist_grows_after_steps(self, reward_fn):
        """T17：多次 compute() 後 portfolio_hist 長度應增加。"""
        for _ in range(5):
            reward_fn.compute(**_base_kwargs())
        assert len(reward_fn.portfolio_hist) == 6  # 1 (reset) + 5 (compute)

    def test_reset_clears_history(self, reward_fn):
        """T18：多步後 reset() → portfolio_hist 重設為 1 個初始值。"""
        for _ in range(10):
            reward_fn.compute(**_base_kwargs())
        reward_fn.reset(N_STOCKS, INITIAL_CAPITAL)
        assert len(reward_fn.portfolio_hist) == 1
        assert reward_fn.total_asset == INITIAL_CAPITAL