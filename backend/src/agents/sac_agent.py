"""
SAC Agent v7（支援兩種 Actor）

Run A/C：SACAgentDirichlet（舊版 Dirichlet+Beta）
Run B/D：SACAgentLogitDelta（新版 11D Logit Delta）

SACAgentLogitDelta 的關鍵改動：
  - 管理 logit_state (B, 11)：上一步的 L_t
  - act() 時將 logit_state 傳入 actor.sample()，取得新 logit 和 w
  - update() 時需要處理 logit_state 的 replay（從 buffer 取出的 s 對應的 L_t）
  - logit_state 存在 ReplayBuffer 的額外欄位（在 buffer 中以 logit 維度擴充）

注意：
  - LogitDelta 的 log_prob 是近似值（-0.5 * ||ΔL||²），不是嚴格機率密度
    但提供 SAC 熵梯度信號，alpha 自動調整仍可運作
  - gradient clipping 放寬至 max_norm=0.3（Actor 與 Critic 一致）
  - target_entropy 暫維持 -12.0，待新基準穩定後再議
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from configs.base_config import DEVICE
from configs.trading_config import (
    SAC_LR, SAC_GAMMA, SAC_TAU,
    SAC_BATCH, SAC_ALPHA_MIN, SAC_TARGET_ENTROPY,
    N_TRADEABLE,
)
from src.agents.base import BaseAgent
from src.agents.memory import ReplayBuffer
from src.models.architectures import (
    PortfolioActorDirichlet,
    PortfolioActorLogitDelta,
    PortfolioCritic,
    N_ACTIONS,
)
from diagnostics import register, nan_guard

# gradient clipping 放寬至 0.3（v7）
GRAD_MAX_NORM = 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# Run A/C：SACAgentDirichlet（舊版，僅調整 grad clip）
# ═══════════════════════════════════════════════════════════════════════════════

class SACAgentDirichlet(BaseAgent):
    """
    舊版 Dirichlet+Beta SAC Agent（Run A/C 使用）。
    與之前版本相同，僅將 grad clip 從 0.1 改為 0.3。
    """

    @register(
        module="Agent",
        inputs={"state_dim": "int", "n_stocks": "int"},
        outputs={"return": "SACAgentDirichlet"},
        notes="舊版 Dirichlet Agent（Run A/C），grad_clip=0.3",
    )
    def __init__(self, state_dim: int, n_stocks: int):
        self.n_stocks  = n_stocks
        self.gamma     = SAC_GAMMA
        self.tau       = SAC_TAU
        self.batch     = SAC_BATCH
        self.alpha_min = SAC_ALPHA_MIN

        self.actor         = PortfolioActorDirichlet(state_dim, n_stocks).to(DEVICE)
        self.critic        = PortfolioCritic(state_dim, n_stocks).to(DEVICE)
        self.critic_target = PortfolioCritic(state_dim, n_stocks).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=SAC_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=SAC_LR)

        self.target_entropy = SAC_TARGET_ENTROPY
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=DEVICE)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=SAC_LR * 0.01)
        self.alpha     = self.log_alpha.exp().item()

        self.buffer = ReplayBuffer()

    @nan_guard()
    @register(
        module="Agent",
        inputs={"obs": "np.ndarray (STATE_DIM,)", "deterministic": "bool"},
        outputs={"return": "np.ndarray (N_STOCKS,)"},
        notes="stochastic/deterministic action，回傳前 N_STOCKS 維",
    )
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            w, _, _ = self.actor.sample(s)
            if deterministic:
                _, _, mean_w = self.actor.sample(s)
                raw = mean_w.squeeze().cpu().numpy()
            else:
                raw = w.squeeze().cpu().numpy()
        return raw[:self.n_stocks].astype(np.float32)

    @register(
        module="Agent",
        inputs={},
        outputs={"critic_loss": "float", "actor_loss": "float", "alpha_loss": "float"},
        notes="SAC 三段更新；grad_clip=0.3",
    )
    def update(self) -> dict | None:
        if len(self.buffer) < max(self.batch, 5000):
            return None

        s, a, r, s_, d = self.buffer.sample(self.batch)

        with torch.no_grad():
            w_, lp_, _ = self.actor.sample(s_)
            a_stock_   = w_[:, :self.n_stocks]
            q_next     = self.critic_target.q_min(s_, a_stock_) - self.alpha * lp_
            q_tgt      = r + self.gamma * (1 - d) * q_next

        q1, q2 = self.critic(s, a)
        c_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)
        self.critic_opt.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRAD_MAX_NORM)
        self.critic_opt.step()

        w_new, lp, _ = self.actor.sample(s)
        a_new_stock  = w_new[:, :self.n_stocks]
        a_loss = (self.alpha * lp - self.critic.q_min(s, a_new_stock)).mean()
        self.actor_opt.zero_grad()
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=GRAD_MAX_NORM)
        self.actor_opt.step()

        al_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        al_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=np.log(self.alpha_min))
        self.alpha = self.log_alpha.exp().item()

        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return {
            "critic_loss": float(c_loss.item()),
            "actor_loss":  float(a_loss.item()),
            "alpha_loss":  float(al_loss.item()),
        }

    def save(self, path: str):
        self.actor.cpu()
        self.critic.cpu()
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "alpha":  float(self.alpha),
        }, path)
        self.actor.to(DEVICE)
        self.critic.to(DEVICE)
        self.critic_target.to(DEVICE)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic"])
        self.actor.to(DEVICE)
        self.critic.to(DEVICE)
        self.critic_target.to(DEVICE)
        with torch.no_grad():
            self.log_alpha.fill_(
                np.log(max(ckpt.get("alpha", 1.0), self.alpha_min))
            )
        self.alpha = self.log_alpha.exp().item()


# ═══════════════════════════════════════════════════════════════════════════════
# Run B/D：SACAgentLogitDelta（新版 11D Logit Delta）
# ═══════════════════════════════════════════════════════════════════════════════

class SACAgentLogitDelta(BaseAgent):
    """
    11D LogitDelta SAC Agent（Run B/D 使用）。

    logit_state 管理：
      - self._logit_state：當前 episode 的 L_t（numpy，(11,)）
      - act() 每步更新 _logit_state
      - episode 開始時（reset_logit_state()）清零
      - buffer 儲存時額外存入 logit_state，供 update() 重建梯度圖

    ReplayBuffer 擴充：
      - 標準 (s, a, r, s_, done) 外，額外儲存 (logit_t, logit_t1)
      - 使用 LogitReplayBuffer（繼承 ReplayBuffer，擴充欄位）
    """

    @register(
        module="Agent",
        inputs={"state_dim": "int", "n_stocks": "int"},
        outputs={"return": "SACAgentLogitDelta"},
        notes="LogitDelta Agent（Run B/D），管理 logit_state，grad_clip=0.3",
    )
    def __init__(self, state_dim: int, n_stocks: int):
        self.n_stocks  = n_stocks
        self.gamma     = SAC_GAMMA
        self.tau       = SAC_TAU
        self.batch     = SAC_BATCH
        self.alpha_min = SAC_ALPHA_MIN

        self.actor         = PortfolioActorLogitDelta(state_dim, n_stocks).to(DEVICE)
        self.critic        = PortfolioCritic(state_dim, n_stocks).to(DEVICE)
        self.critic_target = PortfolioCritic(state_dim, n_stocks).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=SAC_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=SAC_LR)

        self.target_entropy = SAC_TARGET_ENTROPY
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=DEVICE)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=SAC_LR * 0.01)
        self.alpha     = self.log_alpha.exp().item()

        # logit_state：(11,) numpy，episode 開始時清零
        self._logit_state = np.zeros(N_ACTIONS, dtype=np.float32)

        self.buffer = LogitReplayBuffer()

    def reset_logit_state(self):
        """每個 episode 開始時呼叫，清零 logit 狀態。"""
        self._logit_state = np.zeros(N_ACTIONS, dtype=np.float32)

    @nan_guard()
    @register(
        module="Agent",
        inputs={"obs": "np.ndarray (STATE_DIM,)", "deterministic": "bool"},
        outputs={"return": "np.ndarray (N_STOCKS,)"},
        notes="LogitDelta act()：注入 logit_state，更新後儲存新狀態",
    )
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        l = torch.FloatTensor(self._logit_state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            w, new_logit, _ = self.actor.sample(s, logit_state=l)
            raw = w.squeeze().cpu().numpy()
            # 更新 logit_state
            self._logit_state = new_logit.squeeze().cpu().numpy().astype(np.float32)

        # 回傳前 N_STOCKS 維給 env（env 只需要股票權重）
        return raw[:self.n_stocks].astype(np.float32)

    def push_transition(self, obs, action, reward, next_obs, done):
        """
        推入 transition，額外存入 logit_state。
        呼叫時機：env.step() 之後，_logit_state 已更新為 L_{t+1}。
        """
        self.buffer.push(
            obs, action, reward, next_obs, done,
            self._logit_state.copy()   # 存入更新後的 L_{t+1}
        )

    @register(
        module="Agent",
        inputs={},
        outputs={"critic_loss": "float", "actor_loss": "float", "alpha_loss": "float"},
        notes="LogitDelta SAC 三段更新；grad_clip=0.3",
    )
    def update(self) -> dict | None:
        if len(self.buffer) < max(self.batch, 5000):
            return None

        s, a, r, s_, d, logit_t1 = self.buffer.sample(self.batch)

        # ── Critic update ────────────────────────────────────────────────
        with torch.no_grad():
            w_, new_l_, lp_ = self.actor.sample(s_, logit_state=logit_t1)
            a_stock_         = w_[:, :self.n_stocks]
            q_next           = self.critic_target.q_min(s_, a_stock_) - self.alpha * lp_
            q_tgt            = r + self.gamma * (1 - d) * q_next
            # Q-target clamp：臨時保險絲，防止 LinearDownsideReward 初期 reward 尺度
            # 導致 Q 發散。Q 真實範圍 ≈ ±R_max/(1-γ) ≈ ±20，±50 留有充足裕度。
            # 必須在 Critic loss 連續 5000 步維持在 1.0 以下後移除。
            q_tgt = q_tgt.clamp(-50.0, 50.0)

        q1, q2 = self.critic(s, a)
        c_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)
        self.critic_opt.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRAD_MAX_NORM)
        self.critic_opt.step()

        # ── Actor update ─────────────────────────────────────────────────
        w_new, _, lp = self.actor.sample(s, logit_state=logit_t1)
        a_new_stock  = w_new[:, :self.n_stocks]
        a_loss = (self.alpha * lp - self.critic.q_min(s, a_new_stock)).mean()
        self.actor_opt.zero_grad()
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=GRAD_MAX_NORM)
        self.actor_opt.step()

        # ── Alpha update ──────────────────────────────────────────────────
        al_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        al_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=np.log(self.alpha_min))
        self.alpha = self.log_alpha.exp().item()

        # ── Soft update target ───────────────────────────────────────────
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return {
            "critic_loss": float(c_loss.item()),
            "actor_loss":  float(a_loss.item()),
            "alpha_loss":  float(al_loss.item()),
        }

    def save(self, path: str):
        self.actor.cpu()
        self.critic.cpu()
        torch.save({
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "alpha":        float(self.alpha),
            "logit_state":  self._logit_state.tolist(),
        }, path)
        self.actor.to(DEVICE)
        self.critic.to(DEVICE)
        self.critic_target.to(DEVICE)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic"])
        self.actor.to(DEVICE)
        self.critic.to(DEVICE)
        self.critic_target.to(DEVICE)
        with torch.no_grad():
            self.log_alpha.fill_(
                np.log(max(ckpt.get("alpha", 1.0), self.alpha_min))
            )
        self.alpha = self.log_alpha.exp().item()
        if "logit_state" in ckpt:
            self._logit_state = np.array(ckpt["logit_state"], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# LogitReplayBuffer：擴充 logit_state 欄位
# ═══════════════════════════════════════════════════════════════════════════════

class LogitReplayBuffer:
    """
    擴充版 ReplayBuffer，額外儲存 logit_t1（L_{t+1}）。
    供 SACAgentLogitDelta.update() 重建梯度圖使用。
    """

    def __init__(self, capacity: int = None):
        from configs.trading_config import SAC_BUFFER_SIZE
        self.capacity = capacity or SAC_BUFFER_SIZE
        self.buffer   = []
        self.pos      = 0

    def push(self, s, a, r, s_, done, logit_t1):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, s_, done, logit_t1)
        self.pos = (self.pos + 1) % self.capacity

    def clear(self):
        """清空 buffer，重置 pos；capacity 不變。"""
        self.buffer = []
        self.pos    = 0

    def sample(self, batch: int):
        from configs.base_config import DEVICE as _DEVICE
        indices = np.random.choice(len(self.buffer), batch, replace=False)
        batch_data = [self.buffer[i] for i in indices]

        s      = torch.FloatTensor(np.array([d[0] for d in batch_data])).to(_DEVICE)
        a      = torch.FloatTensor(np.array([d[1] for d in batch_data])).to(_DEVICE)
        r      = torch.FloatTensor(np.array([d[2] for d in batch_data])).unsqueeze(1).to(_DEVICE)
        s_     = torch.FloatTensor(np.array([d[3] for d in batch_data])).to(_DEVICE)
        done   = torch.FloatTensor(np.array([d[4] for d in batch_data])).unsqueeze(1).to(_DEVICE)
        logit  = torch.FloatTensor(np.array([d[5] for d in batch_data])).to(_DEVICE)

        return s, a, r, s_, done, logit

    def __len__(self):
        return len(self.buffer)


# 預設別名：讓舊程式碼繼續工作
SACAgent = SACAgentDirichlet