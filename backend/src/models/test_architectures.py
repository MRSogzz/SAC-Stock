"""
tests/src/models/test_architectures.py
=========================================
architectures.py 的單元測試，對應真實實作：
  - SharedStockMLP    內部子模組，間接透過 PortfolioActor 測試
  - PortfolioMLP      內部子模組，間接透過 PortfolioActor 測試
  - PortfolioActor    .forward() / .sample()
  - PortfolioCritic   .forward() / .q_min()

Mock 策略：
  - configs 常數 → patch，測試用縮小規模（避免 OOM / 速度慢）
  - 所有測試使用 CPU，不要求 CUDA

v2 測試對應變更：
  - T8：log_prob 改為 sum 後量級從約 -1 變成約 -(N_STOCKS+1)，
        斷言從「接近 -1」改為「應為有限負數」，不鎖定具體量級
  - T10/T11：修正原測試的錯誤假設。
        action 是從 softmax(N_STOCKS+1 維) 截取前 N_STOCKS 維，
        總和必然 < 1（現金權重被截掉），原本 allclose(1.0) 的斷言在
        數學上是錯的。修正為「總和在 (0, 1] 範圍內」。
  - 新增 T27：log_prob 的 sum vs mean 回歸測試
        直接比較兩種計算方式的輸出，確認實作使用 sum。
  - 新增 T28：PortfolioMLP 輸入維度回歸測試
        確認第一層 Linear 的 in_features 為 N_PORTFOLIO + 1。
  - 新增 T29：Critic Xavier 初始化回歸測試
        確認 Q1/Q2 參數初始化後標準差在合理範圍內（orthogonal gain=0.01
        會使 std 極小，Xavier 不會）。
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

# ── 測試用縮小常數（與 configs 解耦）─────────────────────────────────────────

N_TRADEABLE  = 3
N_OBSERVABLE = 4          # 含 benchmark（0050）
N_FEATURES   = 31
SAC_HIDDEN   = 64
STATE_DIM    = N_OBSERVABLE * N_FEATURES + N_TRADEABLE * 2 + 1  # 124 + 6 + 1 = 131
N_PORTFOLIO  = N_TRADEABLE * 2 + 1   # 7，方便 T28 直接引用

BATCH = 8
DEVICE = torch.device("cpu")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def config_patch():
    """patch configs 常數，讓所有架構使用測試用規模。"""
    with patch("src.models.architectures.N_TRADEABLE",  N_TRADEABLE), \
         patch("src.models.architectures.N_OBSERVABLE", N_OBSERVABLE), \
         patch("src.models.architectures.N_FEATURES",   N_FEATURES), \
         patch("src.models.architectures.SAC_HIDDEN",   SAC_HIDDEN), \
         patch("src.models.architectures.STATE_DIM",    STATE_DIM), \
         patch("src.models.architectures.BENCHMARK_IDX", N_OBSERVABLE - 1), \
         patch("src.models.architectures.N_STOCK_INPUT", N_FEATURES * 2), \
         patch("src.models.architectures.N_PORTFOLIO",   N_PORTFOLIO):
        yield


@pytest.fixture
def actor():
    from src.models.architectures import PortfolioActor
    return PortfolioActor(
        state_dim=STATE_DIM,
        n_stocks=N_TRADEABLE,
        hidden=SAC_HIDDEN,
    ).to(DEVICE)


@pytest.fixture
def critic():
    from src.models.architectures import PortfolioCritic
    return PortfolioCritic(
        state_dim=STATE_DIM,
        n_stocks=N_TRADEABLE,
        hidden=SAC_HIDDEN,
    ).to(DEVICE)


@pytest.fixture
def obs():
    return torch.randn(BATCH, STATE_DIM, device=DEVICE)


@pytest.fixture
def action():
    """合法 action：softmax 輸出，總和為 1。"""
    raw = torch.rand(BATCH, N_TRADEABLE, device=DEVICE)
    return raw / raw.sum(dim=-1, keepdim=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PortfolioActor 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioActor:

    # ── forward() ─────────────────────────────────────────────────────────────

    def test_forward_mean_logits_shape(self, actor, obs):
        """T1：forward() 輸出 mean_logits shape 應為 (B, N_STOCKS+1)。"""
        mean_logits, _ = actor.forward(obs)
        assert mean_logits.shape == (BATCH, N_TRADEABLE + 1), (
            f"mean_logits shape 錯誤：{mean_logits.shape}"
        )

    def test_forward_log_std_shape(self, actor, obs):
        """T2：forward() 輸出 log_std shape 應為 (B, N_STOCKS+1)。"""
        _, log_std = actor.forward(obs)
        assert log_std.shape == (BATCH, N_TRADEABLE + 1)

    def test_forward_mean_logits_bounded(self, actor, obs):
        """T3：mean_logits 應在 [-3, 3] 範圍內（Tanh Bounding 設計）。"""
        mean_logits, _ = actor.forward(obs)
        assert (mean_logits >= -3.0 - 1e-5).all() and (mean_logits <= 3.0 + 1e-5).all(), (
            f"Tanh Bounding 失效，超出 [-3, 3]"
        )

    def test_forward_log_std_clamped(self, actor, obs):
        """T4：log_std 應在 [LOG_STD_MIN, LOG_STD_MAX] = [-2, 1] 範圍內。"""
        _, log_std = actor.forward(obs)
        assert (log_std >= -2.0 - 1e-5).all() and (log_std <= 1.0 + 1e-5).all()

    def test_forward_no_nan(self, actor, obs):
        """T5：forward() 輸出不含 NaN。"""
        mean_logits, log_std = actor.forward(obs)
        assert not torch.isnan(mean_logits).any()
        assert not torch.isnan(log_std).any()

    def test_forward_variable_batch_size(self, actor):
        """T6：不同 batch size 的輸入均能正確輸出。"""
        for bs in [1, 4, 16]:
            o = torch.randn(bs, STATE_DIM, device=DEVICE)
            mean_logits, log_std = actor.forward(o)
            assert mean_logits.shape == (bs, N_TRADEABLE + 1)
            assert log_std.shape     == (bs, N_TRADEABLE + 1)

    # ── sample() ──────────────────────────────────────────────────────────────

    def test_sample_action_shape(self, actor, obs):
        """T7：sample() action shape 應為 (B, N_STOCKS)。"""
        action, _, _ = actor.sample(obs)
        assert action.shape == (BATCH, N_TRADEABLE)

    def test_sample_log_prob_shape(self, actor, obs):
        """T8：sample() log_prob shape 應為 (B, 1)，值應為有限負數。

        v2 修正：log_prob 改為 sum 後量級從約 -1 變成約 -(N_STOCKS+1)。
        不鎖定具體量級，只驗證：
          1. shape 正確
          2. 所有值為有限數（不含 NaN / inf）
          3. 所有值為負數（Normal log_prob 在合理 std 下必然為負）
        """
        _, log_prob, _ = actor.sample(obs)
        assert log_prob.shape == (BATCH, 1)
        assert torch.isfinite(log_prob).all(), "log_prob 應為有限數"
        assert (log_prob < 0).all(), "Normal log_prob 在合理參數下應為負數"

    def test_sample_mean_action_shape(self, actor, obs):
        """T9：sample() mean_action shape 應為 (B, N_STOCKS)。"""
        _, _, mean_action = actor.sample(obs)
        assert mean_action.shape == (BATCH, N_TRADEABLE)

    def test_sample_action_sums_less_than_one(self, actor, obs):
        """T10：action 是從 softmax(N_STOCKS+1 維) 截取前 N_STOCKS 維。

        v2 修正：原測試錯誤地假設 action 總和 == 1。
        正確行為：現金權重被截掉後，股票 action 總和必然 < 1。
        斷言改為：0 < sum <= 1（嚴格正數，且不超過 1）。
        """
        action, _, _ = actor.sample(obs)
        row_sums = action.sum(dim=-1)
        assert (row_sums > 0).all(), f"action 總和應為正數，實際：{row_sums}"
        assert (row_sums <= 1.0 + 1e-5).all(), f"action 總和不應超過 1，實際：{row_sums}"

    def test_sample_mean_action_sums_less_than_one(self, actor, obs):
        """T11：mean_action（deterministic）總和同樣應在 (0, 1] 範圍內。

        v2 修正：與 T10 相同原因，截取後總和必然 < 1。
        """
        _, _, mean_action = actor.sample(obs)
        row_sums = mean_action.sum(dim=-1)
        assert (row_sums > 0).all(), f"mean_action 總和應為正數，實際：{row_sums}"
        assert (row_sums <= 1.0 + 1e-5).all(), f"mean_action 總和不應超過 1，實際：{row_sums}"

    def test_sample_action_non_negative(self, actor, obs):
        """T12：action 所有元素應 >= 0（softmax 輸出保證）。"""
        action, _, _ = actor.sample(obs)
        assert (action >= 0.0).all()

    def test_sample_nan_fallback(self, actor):
        """T13：obs 含 NaN 時，sample() 應 fallback 為均分 action，不 crash。"""
        nan_obs = torch.full((BATCH, STATE_DIM), float("nan"), device=DEVICE)
        action, log_prob, mean_action = actor.sample(nan_obs)
        # fallback 應為均分
        expected = 1.0 / (N_TRADEABLE + 1)
        assert torch.allclose(action, torch.full_like(action, expected), atol=1e-5)

    def test_sample_no_nan_output(self, actor, obs):
        """T14：正常 obs 輸入，sample() 輸出不含 NaN。"""
        action, log_prob, mean_action = actor.sample(obs)
        assert not torch.isnan(action).any()
        assert not torch.isnan(log_prob).any()
        assert not torch.isnan(mean_action).any()

    def test_log_std_is_global_parameter(self, actor):
        """T15：log_std 應為全局 nn.Parameter，不依賴 obs（形狀固定）。"""
        assert isinstance(actor.log_std, nn.Parameter)
        assert actor.log_std.shape == (N_TRADEABLE + 1,)

    # ── 子模組結構 ────────────────────────────────────────────────────────────

    def test_has_shared_stock_mlp(self, actor):
        """T16：Actor 應包含 SharedStockMLP 子模組。"""
        from src.models.architectures import SharedStockMLP
        assert isinstance(actor.stock_mlp, SharedStockMLP)

    def test_has_portfolio_mlp(self, actor):
        """T17：Actor 應包含 PortfolioMLP 子模組。"""
        from src.models.architectures import PortfolioMLP
        assert isinstance(actor.portfolio_mlp, PortfolioMLP)

    def test_has_logit_norm(self, actor):
        """T18：Actor 應包含 logit_norm（LayerNorm）。"""
        assert isinstance(actor.logit_norm, nn.LayerNorm)

    # ── v2 新增回歸測試 ───────────────────────────────────────────────────────

    def test_log_prob_is_sum_not_mean(self, actor, obs):
        """T27：驗證 log_prob 使用 sum 而非 mean（v2 改動一回歸測試）。

        原理：對同一個 z，sum 版本的絕對值應為 mean 版本的 (N_STOCKS+1) 倍。
        直接從 forward() 取得分布參數，分別計算 sum 和 mean，
        確認 sample() 回傳的 log_prob 與 sum 版本一致。
        """
        torch.manual_seed(42)
        mean_logits, log_std = actor.forward(obs)
        std  = log_std.exp().clamp(1e-6, 10.0)
        dist = torch.distributions.Normal(mean_logits, std)

        torch.manual_seed(42)
        action, log_prob, _ = actor.sample(obs)

        # 用相同 seed 重建 z，計算 sum 版本
        torch.manual_seed(42)
        z = dist.rsample()
        expected_log_prob = dist.log_prob(z).sum(-1, keepdim=True)

        assert torch.allclose(log_prob, expected_log_prob, atol=1e-5), (
            f"log_prob 應使用 sum，與預期不符。\n"
            f"  實際:   {log_prob[:3].squeeze().tolist()}\n"
            f"  預期:   {expected_log_prob[:3].squeeze().tolist()}"
        )

    def test_portfolio_mlp_input_dim_includes_market_summary(self, actor):
        """T28：PortfolioMLP 第一層輸入維度應為 N_PORTFOLIO + 1（v2 改動二回歸測試）。

        market_summary（stock_scores 均值）被拼入 PortfolioMLP 輸入，
        使現金 logit 能感知全市場動態。
        """
        first_layer = actor.portfolio_mlp.net[0]
        assert isinstance(first_layer, nn.Linear), "PortfolioMLP 第一層應為 Linear"
        expected_in_features = N_PORTFOLIO + 1
        assert first_layer.in_features == expected_in_features, (
            f"PortfolioMLP 輸入維度應為 {expected_in_features}（N_PORTFOLIO+1），"
            f"實際為 {first_layer.in_features}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PortfolioCritic 測試
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioCritic:

    def test_forward_q1_shape(self, critic, obs, action):
        """T19：forward() Q1 shape 應為 (B, 1)。"""
        q1, _ = critic.forward(obs, action)
        assert q1.shape == (BATCH, 1)

    def test_forward_q2_shape(self, critic, obs, action):
        """T20：forward() Q2 shape 應為 (B, 1)。"""
        _, q2 = critic.forward(obs, action)
        assert q2.shape == (BATCH, 1)

    def test_forward_q1_q2_different(self, critic, obs, action):
        """T21：Q1 與 Q2 應不完全相同（Twin Q 設計保證獨立性）。"""
        q1, q2 = critic.forward(obs, action)
        assert not torch.equal(q1, q2), "Q1 與 Q2 不應完全相同"

    def test_forward_no_nan(self, critic, obs, action):
        """T22：forward() 輸出不含 NaN。"""
        q1, q2 = critic.forward(obs, action)
        assert not torch.isnan(q1).any()
        assert not torch.isnan(q2).any()

    def test_q_min_shape(self, critic, obs, action):
        """T23：q_min() 回傳 shape 應為 (B, 1)。"""
        q_min = critic.q_min(obs, action)
        assert q_min.shape == (BATCH, 1)

    def test_q_min_leq_both_q(self, critic, obs, action):
        """T24：q_min 應 ≤ Q1 且 ≤ Q2（逐元素）。"""
        q1, q2 = critic.forward(obs, action)
        q_min  = critic.q_min(obs, action)
        assert (q_min <= q1 + 1e-6).all()
        assert (q_min <= q2 + 1e-6).all()

    def test_forward_variable_batch_size(self, critic):
        """T25：不同 batch size 均能正確輸出。"""
        for bs in [1, 4, 32]:
            o = torch.randn(bs, STATE_DIM,   device=DEVICE)
            a = torch.randn(bs, N_TRADEABLE, device=DEVICE)
            a = a / a.sum(dim=-1, keepdim=True).abs().clamp(min=1e-6)
            q1, q2 = critic.forward(o, a)
            assert q1.shape == (bs, 1)
            assert q2.shape == (bs, 1)

    def test_has_two_independent_networks(self, critic):
        """T26：Critic 應有獨立的 q1 / q2 兩個子網路。"""
        assert hasattr(critic, "q1") and hasattr(critic, "q2")
        # 參數應獨立（不共享）
        q1_params = list(critic.q1.parameters())
        q2_params = list(critic.q2.parameters())
        for p1, p2 in zip(q1_params, q2_params):
            assert p1.data_ptr() != p2.data_ptr(), "Q1 與 Q2 不應共享參數"

    def test_critic_xavier_initialization(self, critic):
        """T29：Critic 應使用 Xavier 初始化，參數標準差不應極小。

        v2 改動三回歸測試：原 orthogonal(gain=0.01) 會使所有權重 std ≈ 0.01，
        Xavier uniform 對 Linear(in, out) 的 std ≈ sqrt(2/(in+out))，
        對本測試規模（in=134, out=64）約為 0.11，遠大於 0.01。
        斷言：Q1 第一層權重的標準差應 > 0.02（留足夠餘裕避免 flaky）。
        """
        first_layer_q1 = critic.q1[0]
        assert isinstance(first_layer_q1, nn.Linear)
        weight_std = first_layer_q1.weight.std().item()
        assert weight_std > 0.02, (
            f"Critic 第一層權重 std={weight_std:.4f}，"
            f"Xavier 初始化應遠大於 0.02（orthogonal gain=0.01 會小於此值）"
        )