"""
神經網路架構：Variant H（SharedFeatureExtractor + LogitDelta + Twin Q）

Run D 使用：PortfolioActorLogitDelta + PortfolioCritic（Variant H）
Run B 保留：PortfolioActorLogitDelta + PortfolioCritic（Variant H）
Run A/C：已拋棄，PortfolioActorDirichlet 保留但不再訓練

Variant H 核心設計：
  - SharedFeatureExtractor：27 維特徵 → 32 維 embedding（所有股票共用同一套權重）
  - 輸入重塑：obs [batch, 270+19] → stock [batch, 10, 27] + account [batch, 19]
  - Actor：Extractor embedding(320) + account(19) → Actor MLP → N_ACTIONS(10)
  - Critic：同樣使用 Extractor，確保 state 表示不含 ID 依賴捷徑
  - LogitDelta：L_{t+1} = 0.995 * L_t + ΔL（有界：ΔL = 0.1 * raw / (1+||raw||)）

移除特徵：rank_ret_5, rank_ret_20, rank_vol_5, pos_252（ID 依賴 / look-ahead）
特徵處理：10 * tanh(x/10) 取代 clip(-10,10)（保留梯度，防硬邊界失真）
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.trading_config import (
    SAC_HIDDEN, N_TRADEABLE, N_FEATURES, STATE_DIM, N_OBSERVABLE
)
from diagnostics import register

BENCHMARK_IDX  = N_OBSERVABLE - 1
N_PORTFOLIO    = N_TRADEABLE * 2 + 1

# Variant H：特徵工程改版後的維度
# N_FEATURES 在 trading_config.py 已經是 27（移除 4 個特徵後的值），不需要再減
N_FEAT_PER_STOCK  = N_FEATURES        # 27 維（已在 trading_config 更新）
N_STOCK_INPUT     = N_FEAT_PER_STOCK  # Shared MLP 每股輸入
EXTRACTOR_DIM     = 32                # Shared Extractor 每股 embedding 維度
EXTRACTOR_FLAT    = N_OBSERVABLE * EXTRACTOR_DIM  # 10 * 32 = 320
ACCOUNT_DIM       = N_PORTFOLIO       # 持倉帳戶特徵維度（19 維）
SHARED_OUT_DIM    = EXTRACTOR_FLAT + ACCOUNT_DIM  # 320 + 19 = 339

# LogitDelta 固定超參數
LEAKY_GAMMA    = 0.995   # 漏水積分器衰減係數（0.999→0.995，加速 L_t 衰減防爆炸）
DELTA_SCALE    = 0.1     # ΔL 縮放係數，全程固定
SOFTMAX_TEMP   = 1.5     # Softmax 溫度，全程固定，不退火
N_ACTIONS      = N_TRADEABLE + 1   # 11 維：10 股票 + 1 現金

# Dirichlet+Beta 固定超參數
ALPHA_MIN = 0.1
BETA_MIN  = 0.1


def _xavier_linear(in_f: int, out_f: int) -> nn.Linear:
    layer = nn.Linear(in_f, out_f)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias, 0.0)
    return layer


# ── 共用子模組 ────────────────────────────────────────────────────────────────

class SharedFeatureExtractor(nn.Module):
    """
    Shared MLP Extractor（Variant H 核心）。

    對每檔股票獨立應用相同的網路，強制模型學習「特徵→行為」映射，
    而非「位置→行為」的捷徑學習。

    輸入：(B * N_stocks, N_FEAT_PER_STOCK)  每檔股票 27 個特徵
    輸出：(B * N_stocks, EXTRACTOR_DIM)     每檔股票 32 維 embedding
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _xavier_linear(N_FEAT_PER_STOCK, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            _xavier_linear(64, EXTRACTOR_DIM),
            nn.LayerNorm(EXTRACTOR_DIM),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedStockMLP(nn.Module):
    """舊版相容性保留，Variant H 改用 SharedFeatureExtractor。"""
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            _xavier_linear(N_FEAT_PER_STOCK, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
            _xavier_linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.LeakyReLU(0.01),
            _xavier_linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PortfolioMLP(nn.Module):
    """投資組合狀態 MLP。輸入 20 維，輸出 N_TRADEABLE 維。"""
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            _xavier_linear(N_PORTFOLIO + 1, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
            _xavier_linear(hidden, N_TRADEABLE),
        )
        with torch.no_grad():
            self.net[-1].bias.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── PortfolioActorLogitDelta（Run B/D）────────────────────────────────────────

class PortfolioActorLogitDelta(nn.Module):
    """
    11 維 Logit Delta Actor（Run B/D 使用）。

    obs 結構（329 維）：
      [0:310] = 10 支股票特徵（各 31 維）
      [310:319] = 整張市值比例（9 維）
      [319:328] = 零股市值比例（9 維）
      [328] = 現金比例（1 維）

    輸出：
      raw_output (B, 11)：Actor 的原始輸出，SACAgent 負責做 Leaky Integrator
      sample() 回傳最終權重 w (B, 11)，其中前 10 維是股票，最後 1 維是現金

    注意：Leaky Integrator 的狀態 L_t 由 SACAgent 管理，
          透過 logit_state 參數傳入 sample()。
    """

    @register(
        module="Model",
        inputs={"state_dim": "int", "n_stocks": "int", "hidden": "int"},
        outputs={"return": "PortfolioActorLogitDelta"},
        notes="Variant H：SharedFeatureExtractor + Actor MLP，10*32+19 → N_ACTIONS",
    )
    def __init__(self, state_dim: int = STATE_DIM,
                 n_stocks: int = N_TRADEABLE,
                 hidden: int = SAC_HIDDEN):
        super().__init__()
        self.n_stocks  = n_stocks
        self.n_outputs = N_ACTIONS

        # Shared Feature Extractor（所有股票共用同一套權重）
        self.extractor = SharedFeatureExtractor()

        # Actor Head：Concat 後禁止 LayerNorm（規格要求），直接接全連接
        # 容量擴大為 256→128，確保能從修復後的 embedding 學出差異
        self.actor_mlp = nn.Sequential(
            _xavier_linear(SHARED_OUT_DIM, 256),
            nn.ReLU(),
            _xavier_linear(256, 128),
            nn.ReLU(),
            _xavier_linear(128, N_ACTIONS),
        )

        # 冷啟動保護：輸出層用極小 std，使初始 raw_output ≈ 0，w ≈ 均勻分布
        with torch.no_grad():
            nn.init.normal_(self.actor_mlp[-1].weight, mean=0.0, std=0.001)
            nn.init.constant_(self.actor_mlp[-1].bias, 0.0)

    def _extract_obs(self, obs: torch.Tensor):
        """
        將 obs 拆解為股票特徵和帳戶特徵。

        obs 結構：
          [0 : N_OBSERVABLE * N_FEAT_PER_STOCK]  10 支股票各 27 維特徵
          [N_OBSERVABLE * N_FEAT_PER_STOCK :]     帳戶特徵（19 維）

        Returns:
            stock_feats : (B, N_OBSERVABLE, N_FEAT_PER_STOCK)
            account     : (B, ACCOUNT_DIM)
        """
        B        = obs.shape[0]
        feat_end = N_OBSERVABLE * N_FEAT_PER_STOCK
        stock_raw   = obs[:, :feat_end]
        account     = obs[:, feat_end:]
        stock_feats = stock_raw.view(B, N_OBSERVABLE, N_FEAT_PER_STOCK)
        return stock_feats, account

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Variant H forward：
          1. 重塑 obs → (B, 10, 27)
          2. Shared Extractor → (B, 10, 32)
          3. Flatten → (B, 320)
          4. Concat 帳戶特徵 → (B, 339)
          5. Actor MLP → (B, N_ACTIONS)
        """
        B = obs.shape[0]
        stock_feats, account = self._extract_obs(obs)
        flat_feats = stock_feats.reshape(B * N_OBSERVABLE, N_FEAT_PER_STOCK)
        embeddings = self.extractor(flat_feats)               # (B*10, 32)
        flat_embed = embeddings.reshape(B, EXTRACTOR_FLAT)    # (B, 320)
        combined   = torch.cat([flat_embed, account], dim=-1) # (B, 339)
        return self.actor_mlp(combined)                       # (B, N_ACTIONS)

    def sample(self, obs: torch.Tensor,
               logit_state: torch.Tensor = None) -> tuple:
        """
        給定當前 obs 和外部管理的 logit_state L_t，
        執行 Leaky Integrator 並回傳最終動作權重。

        Args:
            obs:         (B, STATE_DIM)
            logit_state: (B, N_ACTIONS) 上一步的 L_t，由 SACAgent 管理。
                         若為 None（如診斷/推論場景），自動以零張量初始化。

        Returns:
            w:           (B, N_ACTIONS) 最終動作權重（股票+現金，總和=1）
            new_logit:   (B, N_ACTIONS) 更新後的 L_{t+1}（供 SACAgent 儲存）
            log_prob:    (B, 1)  近似 log_prob（用於 SAC 熵計算）
        """
        if logit_state is None:
            logit_state = torch.zeros(
                obs.shape[0], N_ACTIONS, dtype=obs.dtype, device=obs.device
            )
        raw_output = self.forward(obs)   # (B, N_ACTIONS)

        # 確保 logit_state 維度與 raw_output 一致
        if logit_state.shape[-1] != raw_output.shape[-1]:
            raise ValueError(
                f"logit_state 維度錯誤：{logit_state.shape} vs "
                f"raw_output {raw_output.shape}，N_ACTIONS={N_ACTIONS}"
            )

        # Leaky Integrator（有界動作空間 L2 能量限制）
        # ΔL = 0.1 * a_raw / (1 + ||a_raw||_2)
        # 將無界輸出轉化為有界控制，保留方向梯度與不動點（ΔL=0 when a_raw=0）
        a_norm = raw_output.norm(p=2, dim=-1, keepdim=True)    # (B, 1)
        delta  = DELTA_SCALE * raw_output / (1.0 + a_norm)     # (B, N_ACTIONS)

        # 強制偵錯 print（理論鐵律：||ΔL||_2 per sample ≤ DELTA_SCALE = 0.1）
        if self.training:
            with torch.no_grad():
                delta_l2 = delta.norm(p=2, dim=-1)
                print(f"[DEBUG] delta_l2: mean={delta_l2.mean().item():.6f}, "
                      f"max={delta_l2.max().item():.6f}")
                print(f"[DEBUG] a_raw_norm: mean={a_norm.mean().item():.4f}, "
                      f"max={a_norm.max().item():.4f}")

        new_logit = LEAKY_GAMMA * logit_state + delta           # L_{t+1}

        # 中心化
        l_norm = new_logit - new_logit.mean(dim=-1, keepdim=True)

        # 固定溫度 Softmax
        w = F.softmax(l_norm / SOFTMAX_TEMP, dim=-1)           # (B, N_ACTIONS)

        # log_prob 近似：Gaussian log π，除以 N_ACTIONS 正規化
        log_prob = (
            -0.5 * (delta ** 2).sum(dim=-1, keepdim=True) / N_ACTIONS
        )   # (B, 1)

        return w, new_logit, log_prob

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        with torch.no_grad():
            nn.init.normal_(self.actor_mlp[-1].weight, mean=0.0, std=0.001)
            nn.init.constant_(self.actor_mlp[-1].bias, 0.0)


# ── PortfolioActorDirichlet（Run A/C，舊版維持）──────────────────────────────

class CashHead(nn.Module):
    """獨立的現金評估網路，輸出 Beta 分布的兩個參數。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _xavier_linear(1, 16),
            nn.LeakyReLU(0.01),
            _xavier_linear(16, 2),
        )

    def forward(self, market_summary: torch.Tensor) -> torch.Tensor:
        return self.net(market_summary)


class PortfolioActorDirichlet(nn.Module):
    """
    舊版 Dirichlet+Beta Actor（Run A/C 使用）。
    與之前的 PortfolioActor 完全相同，重命名以區分。
    """

    @register(
        module="Model",
        inputs={"state_dim": "int", "n_stocks": "int", "hidden": "int"},
        outputs={"return": "PortfolioActorDirichlet"},
        notes="舊版 Dirichlet+Beta Actor（Run A/C）",
    )
    def __init__(self, state_dim: int = STATE_DIM,
                 n_stocks: int = N_TRADEABLE,
                 hidden: int = SAC_HIDDEN):
        super().__init__()
        self.n_stocks  = n_stocks
        self.n_outputs = n_stocks + 1

        self.stock_mlp     = SharedStockMLP(hidden=min(hidden, 128))
        self.portfolio_mlp = PortfolioMLP(hidden=64)
        self.cash_head     = CashHead()

    def _extract_inputs(self, obs: torch.Tensor):
        benchmark_feat = obs[:, BENCHMARK_IDX * N_FEATURES:
                               BENCHMARK_IDX * N_FEATURES + N_FEATURES]
        stock_inputs = []
        for i in range(self.n_stocks):
            stock_feat = obs[:, i * N_FEATURES: (i + 1) * N_FEATURES]
            combined   = torch.cat([stock_feat, benchmark_feat], dim=-1)
            stock_inputs.append(combined)
        stock_inputs = torch.stack(stock_inputs, dim=1)
        portfolio = obs[:, N_OBSERVABLE * N_FEATURES:]
        return stock_inputs, portfolio

    def _get_params(self, obs: torch.Tensor):
        stock_inputs, portfolio = self._extract_inputs(obs)
        batch = obs.shape[0]

        flat         = stock_inputs.view(batch * self.n_stocks, N_STOCK_INPUT)
        stock_scores = self.stock_mlp(flat).view(batch, self.n_stocks)
        market_summary  = stock_scores.mean(dim=1, keepdim=True)
        portfolio_input = torch.cat([portfolio, market_summary], dim=-1)
        raw_alpha_adj   = self.portfolio_mlp(portfolio_input)
        raw_alpha = stock_scores + raw_alpha_adj
        alpha     = F.softplus(raw_alpha) + ALPHA_MIN

        raw_beta = self.cash_head(market_summary)
        beta_a   = F.softplus(raw_beta[:, 0:1]) + BETA_MIN
        beta_b   = F.softplus(raw_beta[:, 1:2]) + BETA_MIN
        beta_b   = torch.clamp(beta_b, max=5.0)
        return alpha, beta_a, beta_b

    def forward(self, obs: torch.Tensor):
        alpha, beta_a, beta_b = self._get_params(obs)
        mean_logits = torch.cat([alpha, beta_a], dim=-1)
        log_std     = beta_b.expand(-1, self.n_outputs)
        return mean_logits, log_std

    def sample(self, obs: torch.Tensor, logit_state=None) -> tuple:
        """logit_state 參數保留為相容性，Dirichlet 版本不使用。"""
        alpha, beta_a, beta_b = self._get_params(obs)

        if (torch.isnan(alpha).any() or torch.isnan(beta_a).any()
                or torch.isnan(beta_b).any()):
            uniform_stock = torch.full(
                (obs.shape[0], self.n_stocks),
                1.0 / self.n_outputs, device=obs.device)
            uniform_cash = torch.full(
                (obs.shape[0], 1),
                1.0 / self.n_outputs, device=obs.device)
            zero_lp  = torch.zeros(obs.shape[0], 1, device=obs.device)
            fallback = torch.cat([uniform_stock, uniform_cash], dim=-1)
            return fallback, zero_lp, fallback

        dirichlet_dist = torch.distributions.Dirichlet(alpha)
        x              = dirichlet_dist.rsample()
        log_prob_stock = dirichlet_dist.log_prob(x).unsqueeze(-1)

        beta_dist     = torch.distributions.Beta(beta_a, beta_b)
        cash          = beta_dist.rsample()
        log_prob_cash = beta_dist.log_prob(cash)

        stock_action = x * (1.0 - cash)
        jacobian_correction = -self.n_stocks * torch.log(1.0 - cash + 1e-8)
        log_prob_raw = log_prob_stock + log_prob_cash + jacobian_correction
        log_prob     = log_prob_raw / self.n_outputs

        alpha_sum      = alpha.sum(dim=-1, keepdim=True)
        mode_valid     = (alpha > 1.0).all(dim=-1, keepdim=True)
        dirichlet_mode = (alpha - 1.0) / (alpha_sum - self.n_stocks)
        dirichlet_mean = alpha / alpha_sum
        x_det = torch.where(mode_valid, dirichlet_mode, dirichlet_mean)

        beta_mode_valid = (beta_a > 1.0) & (beta_b > 1.0)
        beta_mode       = (beta_a - 1.0) / (beta_a + beta_b - 2.0)
        beta_mean       = beta_a / (beta_a + beta_b)
        cash_det = torch.where(beta_mode_valid, beta_mode, beta_mean)

        stock_mean  = x_det * (1.0 - cash_det)
        w           = torch.cat([stock_action, cash], dim=-1)
        mean_action = torch.cat([stock_mean,   cash_det], dim=-1)
        return w, log_prob, mean_action

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        with torch.no_grad():
            self.portfolio_mlp.net[-1].bias.fill_(1.0)


# 預設別名：讓舊程式碼繼續工作
PortfolioActor = PortfolioActorDirichlet


# ── PortfolioCritic ───────────────────────────────────────────────────────────

class PortfolioCritic(nn.Module):
    """
    Variant H 雙 Q-network。

    Critic 也使用 SharedFeatureExtractor，確保 Critic 看到的 state 表示
    與 Actor 一致，不存在 ID 依賴的捷徑。

    輸入：state embedding (SHARED_OUT_DIM=339) + action (N_TRADEABLE=9)
    """

    @register(
        module="Model",
        inputs={"state_dim": "int", "n_stocks": "int", "hidden": "int"},
        outputs={"return": "PortfolioCritic"},
        notes="Variant H Twin Q-network（Shared Extractor + Xavier uniform init）",
    )
    def __init__(self, state_dim: int = STATE_DIM,
                 n_stocks: int = N_TRADEABLE,
                 hidden: int = SAC_HIDDEN):
        super().__init__()
        self.n_stocks = n_stocks

        # 共用 Shared Feature Extractor（與 Actor 使用相同架構但獨立權重）
        self.extractor1 = SharedFeatureExtractor()
        self.extractor2 = SharedFeatureExtractor()

        critic_in = SHARED_OUT_DIM + n_stocks   # 339 + 9 = 348

        def _mlp():
            net = nn.Sequential(
                nn.Linear(critic_in, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),    nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
            return net

        self.q1 = _mlp()
        self.q2 = _mlp()

    def _encode(self, state: torch.Tensor, extractor) -> torch.Tensor:
        """用 Shared Extractor 把 state 編碼為固定維度 embedding。"""
        B         = state.shape[0]
        feat_end  = N_OBSERVABLE * N_FEAT_PER_STOCK
        stock_raw = state[:, :feat_end].view(B, N_OBSERVABLE, N_FEAT_PER_STOCK)
        account   = state[:, feat_end:]
        flat      = stock_raw.reshape(B * N_OBSERVABLE, N_FEAT_PER_STOCK)
        emb       = extractor(flat).view(B, EXTRACTOR_FLAT)
        return torch.cat([emb, account], dim=-1)   # (B, 339)

    @register(
        module="Model",
        inputs={"state": "torch.Tensor (B, STATE_DIM)",
                "action": "torch.Tensor (B, N_STOCKS)"},
        outputs={"q1": "torch.Tensor (B, 1)", "q2": "torch.Tensor (B, 1)"},
        notes="Variant H Twin Q forward（Shared Extractor）",
    )
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        enc1 = self._encode(state, self.extractor1)
        enc2 = self._encode(state, self.extractor2)
        sa1  = torch.cat([enc1, action], dim=-1)
        sa2  = torch.cat([enc2, action], dim=-1)
        return self.q1(sa1), self.q2(sa2)

    @register(
        module="Model",
        inputs={"state": "torch.Tensor (B, STATE_DIM)",
                "action": "torch.Tensor (B, N_STOCKS)"},
        outputs={"return": "torch.Tensor (B, 1)"},
        notes="回傳 min(Q1, Q2)",
    )
    def q_min(self, state: torch.Tensor, action: torch.Tensor):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)