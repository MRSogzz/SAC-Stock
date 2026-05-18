"""
src/environment/reshaping_engine.py
黎明計畫 1.0：價值幾何重建
================================
（放置路徑：backend/src/environment/reshaping_engine.py）
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.trading_config import ALPHA_SCALE
from diagnostics import register

# ── reward 常數：直接定義，避免循環/路徑 import 問題 ────────────────────────
# 與 src/environment/reward.py 中的模組級常數完全一致，若修改 reward.py 需同步
_MAR             = 1.2 / 100 / 252   # 日化無風險利率 ≈ 0.0000476
_DOWNSIDE_LAMBDA = 0.1
_WARMUP_STEPS    = 500
_C_MULTIPLIER    = 3.0
_C_FLOOR         = 0.1
_C_MIN           = 0.005


# ═══════════════════════════════════════════════════════════════════════════════
# 共用常數
# ═══════════════════════════════════════════════════════════════════════════════

N_QUANTILES_TRAIN  = 32
N_QUANTILES_TARGET = 32
TAU_LOW            = 0.05
TAU_HIGH           = 0.95
HUBER_KAPPA        = 1.0
IQN_TARGET_TAU     = 0.990
IQN_EMBED_DIM      = 64
N_REGIMES          = 3
REGIME_EMBED_DIM   = 8
CRITIC_HIDDEN      = 256
ASYMMETRIC_SCALE   = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# P0：IQN Critic
# ═══════════════════════════════════════════════════════════════════════════════

class IQNEmbedding(nn.Module):
    def __init__(self, embed_dim: int = IQN_EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc        = nn.Linear(embed_dim, embed_dim)
        self.act       = nn.ELU()

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """tau: (B, N_tau) → (B, N_tau, embed_dim)"""
        i     = torch.arange(1, self.embed_dim + 1, device=tau.device).float()
        basis = torch.cos(torch.pi * tau.unsqueeze(-1) * i)
        return self.act(self.fc(basis))


class IQNPortfolioCritic(nn.Module):
    def __init__(self, state_dim: int, n_stocks: int,
                 hidden: int = CRITIC_HIDDEN, embed_dim: int = IQN_EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + n_stocks, hidden),
            nn.LayerNorm(hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        self.tau_embed = IQNEmbedding(embed_dim)
        self.fusion    = nn.Sequential(nn.Linear(hidden + embed_dim, hidden), nn.ELU())
        self.out       = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, state, action, tau):
        B, N_tau = tau.shape
        h_sa     = self.encoder(torch.cat([state, action], dim=-1))
        h_tau    = self.tau_embed(tau)
        h_sa_exp = h_sa.unsqueeze(1).expand(-1, N_tau, -1)
        fused    = torch.cat([h_sa_exp, h_tau], dim=-1)
        return self.out(self.fusion(fused)).squeeze(-1)

    @staticmethod
    def sample_tau(batch_size, n_tau, device, low=TAU_LOW, high=TAU_HIGH):
        return torch.zeros(batch_size, n_tau, device=device).uniform_(low, high)

    @staticmethod
    def quantile_huber_loss(pred, target, tau, kappa=HUBER_KAPPA):
        B, N_pred = pred.shape
        N_target  = target.shape[1]
        pe        = pred.unsqueeze(2).expand(-1, -1, N_target)
        te        = target.unsqueeze(1).expand(-1, N_pred, -1)
        delta     = te - pe
        abs_d     = delta.abs()
        huber     = torch.where(abs_d <= kappa, 0.5 * delta**2, kappa * (abs_d - 0.5 * kappa))
        tau_exp   = tau.unsqueeze(2).expand(-1, -1, N_target)
        weight    = (delta < 0).float() - tau_exp
        return (weight.abs() * huber).mean(dim=2).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# P1：Regime Embedding + 條件化 IQN Critic
# ═══════════════════════════════════════════════════════════════════════════════

REGIME_INDEX: dict[str, int] = {"bull": 0, "bear": 1, "sideways": 2}


class RegimeEmbedding(nn.Module):
    def __init__(self, n_regimes=N_REGIMES, embed_dim=REGIME_EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(n_regimes, embed_dim)
        self.norm  = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.embed.weight, std=0.1)

    def forward(self, regime_idx, regime_probs=None):
        if regime_probs is not None:
            all_emb = self.embed.weight.unsqueeze(0)
            emb     = (all_emb * regime_probs.unsqueeze(-1)).sum(dim=1)
        else:
            emb = self.embed(regime_idx)
        return self.norm(emb)


class RegimeConditionedIQNCritic(nn.Module):
    """P0 + P1：Regime 條件化的 IQN Critic。"""

    def __init__(self, state_dim: int, n_stocks: int,
                 hidden: int = CRITIC_HIDDEN,
                 embed_dim: int = IQN_EMBED_DIM,
                 regime_embed_dim: int = REGIME_EMBED_DIM):
        super().__init__()
        encoder_in        = state_dim + n_stocks + regime_embed_dim
        self.regime_embed = RegimeEmbedding(N_REGIMES, regime_embed_dim)
        self.encoder      = nn.Sequential(
            nn.Linear(encoder_in, hidden), nn.LayerNorm(hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        self.tau_embed = IQNEmbedding(embed_dim)
        self.fusion    = nn.Sequential(nn.Linear(hidden + embed_dim, hidden), nn.ELU())
        self.out       = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, state, action, tau, regime_idx, regime_probs=None):
        """Returns: (B, N_tau)"""
        B, N_tau = tau.shape
        r_emb    = self.regime_embed(regime_idx, regime_probs)
        h_sa     = self.encoder(torch.cat([state, action, r_emb], dim=-1))
        h_tau    = self.tau_embed(tau)
        h_sa_exp = h_sa.unsqueeze(1).expand(-1, N_tau, -1)
        fused    = torch.cat([h_sa_exp, h_tau], dim=-1)
        return self.out(self.fusion(fused)).squeeze(-1)

    @staticmethod
    def sample_tau(batch_size, n_tau, device, low=TAU_LOW, high=TAU_HIGH):
        return IQNPortfolioCritic.sample_tau(batch_size, n_tau, device, low, high)

    @staticmethod
    def quantile_huber_loss(pred, target, tau, kappa=HUBER_KAPPA):
        return IQNPortfolioCritic.quantile_huber_loss(pred, target, tau, kappa)

    def mean_q(self, state, action, regime_idx, n_tau=N_QUANTILES_TRAIN, regime_probs=None):
        """E_τ[Z(s,a,τ)]，供 Actor 損失使用。Returns: (B,)"""
        tau = self.sample_tau(state.shape[0], n_tau, state.device)
        return self.forward(state, action, tau, regime_idx, regime_probs).mean(dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# P2：非對稱回報感知
# ═══════════════════════════════════════════════════════════════════════════════

class AsymmetricLinearDownsideReward:
    """
    P2：LinearDownsideReward 的非對稱放大版本。
    不繼承父類（避免 import 鏈問題），直接複製必要狀態與邏輯。
    """

    @register(
        module="Env",
        inputs={},
        outputs={"return": "AsymmetricLinearDownsideReward"},
        notes="P2：LinearDownside + 非對稱 pnl 放大，放大在 raw 層",
    )
    def __init__(self, asymmetric_scale: float = ASYMMETRIC_SCALE):
        self.asymmetric_scale = asymmetric_scale
        self.total_asset      = 1.0
        self.c                = None
        self._warmup_pnls     = []
        self._warmed_up       = False
        self.just_locked      = False

    def reset(self, n_stocks: int, initial_capital: float):
        self.total_asset = initial_capital

    def compute(
        self,
        total_T_pre:   float,
        total_T1_pre:  float,
        odd_ratio:     float,
        port_ret:      float,
        benchmark_ret: float,
        target:        np.ndarray,
        cost_t:        float = 0.0,
    ) -> float:
        pnl              = float(np.log(max(total_T1_pre, 1e-8) / max(total_T_pre, 1e-8)))
        self.total_asset = total_T1_pre

        # P2 非對稱放大：正報酬放大，負報酬不動
        amplifier     = float(np.sqrt(1.0 + self.asymmetric_scale * max(0.0, port_ret)))
        pnl_amplified = pnl * amplifier

        excess      = port_ret - benchmark_ret
        alpha_t     = ALPHA_SCALE * excess
        downside    = _DOWNSIDE_LAMBDA * max(0.0, _MAR - port_ret)
        cost_signal = cost_t / (total_T_pre + 1e-8)
        raw         = pnl_amplified + alpha_t - downside - cost_signal

        if not self._warmed_up:
            self._warmup_pnls.append(raw)
            if len(self._warmup_pnls) >= _WARMUP_STEPS:
                sigma_init = float(np.std(self._warmup_pnls))
                c_locked   = _C_MULTIPLIER * sigma_init
                if c_locked < _C_FLOOR:
                    print(f"  [warmup 熔斷] c_locked={c_locked:.6f} < C_FLOOR={_C_FLOOR}，重置收集")
                    self._warmup_pnls = []
                else:
                    self.c           = c_locked
                    self._warmed_up  = True
                    self.just_locked = True
                    print(f"  [warmup 鎖定] sigma_init={sigma_init:.6f}，c={self.c:.6f}")
            c = _C_MIN
        else:
            c = self.c

        return float(raw / c)

    @property
    def is_warmed_up(self) -> bool:
        return self._warmed_up

    @property
    def scaling_constant(self) -> float:
        return self.c if self.c is not None else _C_MIN

    @property
    def c_locked(self) -> bool:
        return self._warmed_up


# ═══════════════════════════════════════════════════════════════════════════════
# SAC update 輔助函式
# ═══════════════════════════════════════════════════════════════════════════════

def compute_iqn_critic_loss(
    critic, critic_target,
    state, action, reward, next_state, done,
    next_action, log_pi, regime_idx,
    gamma=0.995, alpha=0.2,
    n_tau_pred=N_QUANTILES_TRAIN, n_tau_target=N_QUANTILES_TARGET,
) -> torch.Tensor:
    B      = state.shape[0]
    device = state.device

    with torch.no_grad():
        tau_t         = critic_target.sample_tau(B, n_tau_target, device)
        z_next        = critic_target(next_state, next_action, tau_t, regime_idx)
        entropy_bonus = alpha * log_pi
        z_next        = z_next - entropy_bonus.expand_as(z_next)
        z_target      = reward + gamma * (1.0 - done) * z_next

    tau_p  = critic.sample_tau(B, n_tau_pred, device)
    z_pred = critic(state, action, tau_p, regime_idx)
    return critic.quantile_huber_loss(z_pred, z_target, tau_p)


def soft_update_iqn(critic, target, tau: float = IQN_TARGET_TAU) -> None:
    for p, p_t in zip(critic.parameters(), target.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


# ═══════════════════════════════════════════════════════════════════════════════
# Regime 標籤工具
# ═══════════════════════════════════════════════════════════════════════════════

def regime_label_to_idx(label: str) -> int:
    return REGIME_INDEX.get(label.lower(), 2)


def regime_labels_to_tensor(labels: list[str], device) -> torch.Tensor:
    return torch.tensor([regime_label_to_idx(l) for l in labels],
                        dtype=torch.long, device=device)