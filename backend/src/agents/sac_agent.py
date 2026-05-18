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
from src.environment.reshaping_engine import (
    RegimeConditionedIQNCritic,
    compute_iqn_critic_loss,
    soft_update_iqn,
    regime_labels_to_tensor,
    N_QUANTILES_TRAIN,
    IQN_TARGET_TAU,
)

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
        self.tau       = 0.015          # Phase 1：提升軟更新速率（原 SAC_TAU）
        self.batch     = SAC_BATCH
        self.alpha_min = SAC_ALPHA_MIN

        self.actor         = PortfolioActorLogitDelta(state_dim, n_stocks).to(DEVICE)
        self.critic        = RegimeConditionedIQNCritic(state_dim, n_stocks).to(DEVICE)
        self.critic_target = RegimeConditionedIQNCritic(state_dim, n_stocks).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ── Phase 1：梯度通道解耦 ─────────────────────────────────────────
        # Backbone：regime_embed + encoder（處理 s/a/regime）
        # Quantile Head：tau_embed + fusion + out（處理 τ 分位數）
        _backbone_params = (
            list(self.critic.regime_embed.parameters())
            + list(self.critic.encoder.parameters())
        )
        _head_params = (
            list(self.critic.tau_embed.parameters())
            + list(self.critic.fusion.parameters())
            + list(self.critic.out.parameters())
        )

        self.actor_opt           = optim.Adam(self.actor.parameters(), lr=SAC_LR)
        self.critic_backbone_opt = optim.Adam(_backbone_params, lr=SAC_LR * 2)
        self.critic_head_opt     = optim.Adam(_head_params,     lr=SAC_LR * 3)
        # 向後相容：critic_opt 指向 backbone_opt（供外部程式碼讀取）
        self.critic_opt = self.critic_backbone_opt

        self.target_entropy = SAC_TARGET_ENTROPY
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=DEVICE)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=SAC_LR * 0.01)
        self.alpha     = self.log_alpha.exp().item()

        # Phase 1 監控狀態（per-episode 累積）
        self._phase1_ep_records: list[dict] = []
        self._phase1_step_buf:   list[dict] = []

        # logit_state：(N_ACTIONS,) numpy，episode 開始時清零
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

    def push_transition(self, obs, action, reward, next_obs, done,
                        regime_label: str = "sideways"):
        """
        推入 transition，額外存入 logit_state 與 regime_label。
        regime_label：當前市場狀態（'bull'/'bear'/'sideways'），
                      供 IQN Critic 條件化使用。預設 'sideways' 向後相容。
        """
        self.buffer.push(
            obs, action, reward, next_obs, done,
            self._logit_state.copy(),
            regime_label,
        )

    @register(
        module="Agent",
        inputs={},
        outputs={"critic_loss": "float", "actor_loss": "float", "alpha_loss": "float"},
        notes="LogitDelta SAC 三段更新；Phase 1 梯度解耦；IQN Critic + Regime 條件化",
    )
    def update(self) -> dict | None:
        if len(self.buffer) < max(self.batch, 5000):
            return None

        s, a, r, s_, d, logit_t1, regime_idx = self.buffer.sample(self.batch)

        # ── Critic update（IQN Quantile Huber Loss）──────────────────────
        with torch.no_grad():
            w_, new_l_, lp_ = self.actor.sample(s_, logit_state=logit_t1)
            a_stock_         = w_[:, :self.n_stocks]

        c_loss = compute_iqn_critic_loss(
            critic        = self.critic,
            critic_target = self.critic_target,
            state         = s,
            action        = a,
            reward        = r,
            next_state    = s_,
            done          = d,
            next_action   = a_stock_,
            log_pi        = lp_,
            regime_idx    = regime_idx,
            gamma         = self.gamma,
            alpha         = self.alpha,
            n_tau_pred    = N_QUANTILES_TRAIN,
            n_tau_target  = N_QUANTILES_TRAIN,
        )

        # Phase 1：分別更新 backbone 和 quantile head（不同 LR）
        self.critic_backbone_opt.zero_grad()
        self.critic_head_opt.zero_grad()
        c_loss.backward()

        # Backbone：沿用全域 grad clip
        backbone_params = (
            list(self.critic.regime_embed.parameters())
            + list(self.critic.encoder.parameters())
        )
        head_params = (
            list(self.critic.tau_embed.parameters())
            + list(self.critic.fusion.parameters())
            + list(self.critic.out.parameters())
        )
        grad_norm_backbone = float(
            torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=GRAD_MAX_NORM).item()
        )
        # Quantile Head：獨立 grad clip = 1.0
        grad_norm_head = float(
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0).item()
        )

        self.critic_backbone_opt.step()
        self.critic_head_opt.step()

        # ── Phase 1 監控指標計算 ──────────────────────────────────────────

        # 1. TD Target Variance
        with torch.no_grad():
            tau_t    = self.critic_target.sample_tau(s.shape[0], N_QUANTILES_TRAIN, s.device)
            z_next   = self.critic_target(s_, a_stock_, tau_t, regime_idx)
            z_target = r + self.gamma * (1.0 - d) * (z_next - self.alpha * lp_.expand_as(z_next))
            td_target_var = float(z_target.var().item())

        # 2. Quantile Crossing Rate
        with torch.no_grad():
            tau_sorted = torch.linspace(
                0.05, 0.95, N_QUANTILES_TRAIN, device=s.device
            ).unsqueeze(0).expand(s.shape[0], -1)
            z_sorted = self.critic(s, a, tau_sorted, regime_idx)   # (B, N_tau)
            # 檢查 z[i] <= z[i+1]（分位數應遞增）
            crossings = (z_sorted[:, :-1] > z_sorted[:, 1:]).float()
            quantile_crossing_rate = float(crossings.mean().item())

        # ── Actor update ──────────────────────────────────────────────────
        w_new, _, lp = self.actor.sample(s, logit_state=logit_t1)
        a_new_stock  = w_new[:, :self.n_stocks]
        q_val        = self.critic.mean_q(s, a_new_stock, regime_idx, N_QUANTILES_TRAIN)
        a_loss       = (self.alpha * lp - q_val.unsqueeze(1)).mean()
        self.actor_opt.zero_grad()
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=GRAD_MAX_NORM)
        self.actor_opt.step()

        # 3. EAR（Entropy-to-Advantage Ratio）
        with torch.no_grad():
            advantage = (q_val - q_val.mean()).unsqueeze(1)
            ear = float(
                (self.alpha * lp.abs() / (advantage.abs() + 1e-8)).mean().item()
            )

        # ── Alpha update ───────────────────────────────────────────────────
        al_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        al_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=np.log(self.alpha_min))
        self.alpha = self.log_alpha.exp().item()

        # ── Soft update target（Phase 1：tau=0.015）────────────────────────
        soft_update_iqn(self.critic, self.critic_target, tau=self.tau)

        step_record = {
            "EAR":                       round(ear, 6),
            "td_target_var":             round(td_target_var, 6),
            "quantile_crossing_rate":    round(quantile_crossing_rate, 6),
            "critic_grad_norm_backbone": round(grad_norm_backbone, 6),
            "critic_grad_norm_head":     round(grad_norm_head, 6),
        }
        self._phase1_step_buf.append(step_record)

        return {
            "critic_loss": float(c_loss.item()),
            "actor_loss":  float(a_loss.item()),
            "alpha_loss":  float(al_loss.item()),
            **step_record,
        }

    def flush_phase1_episode(self, ep: int) -> dict | None:
        """
        每 episode 結束後呼叫，將本 episode 的步級監控彙總為 episode 級記錄。
        自動清空 step buffer。
        """
        if not self._phase1_step_buf:
            return None

        buf = self._phase1_step_buf
        record = {
            "ep":                        ep,
            "EAR":                       round(float(np.mean([x["EAR"] for x in buf])), 6),
            "td_target_var":             round(float(np.mean([x["td_target_var"] for x in buf])), 6),
            "quantile_crossing_rate":    round(float(np.mean([x["quantile_crossing_rate"] for x in buf])), 6),
            "critic_grad_norm_backbone": round(float(np.mean([x["critic_grad_norm_backbone"] for x in buf])), 6),
            "critic_grad_norm_head":     round(float(np.mean([x["critic_grad_norm_head"] for x in buf])), 6),
        }
        self._phase1_ep_records.append(record)
        self._phase1_step_buf = []
        return record

    def save_phase1_report(self, output_path: str = None) -> dict:
        """
        輸出 phase1_repair_monitoring.json。
        verdict 依穩定性判據自動生成。
        """
        import json, os

        records = self._phase1_ep_records
        if not records:
            return {}

        mean_ear      = float(np.mean([r["EAR"] for r in records]))
        mean_td_var   = float(np.mean([r["td_target_var"] for r in records]))
        mean_qcr      = float(np.mean([r["quantile_crossing_rate"] for r in records]))

        # 判定 td_target_var 是否呈上升趨勢（後半段 > 前半段）
        half = len(records) // 2
        td_vars = [r["td_target_var"] for r in records]
        td_rising = (
            np.mean(td_vars[half:]) > np.mean(td_vars[:half]) * 1.1
            if half > 0 else False
        )

        if mean_qcr > 0.1:
            verdict = "CRITICAL"
        elif mean_qcr > 0.05 and td_rising:
            verdict = "UNSTABLE"
        else:
            verdict = "STABLE"

        report = {
            "config": {
                "actor_lr":                    SAC_LR,
                "critic_backbone_lr":          SAC_LR * 2,
                "quantile_head_lr":            SAC_LR * 3,
                "tau_target":                  self.tau,
                "quantile_head_max_grad_norm": 1.0,
            },
            "episodes": records,
            "summary": {
                "mean_EAR":                    round(mean_ear, 6),
                "mean_td_target_var":          round(mean_td_var, 6),
                "mean_quantile_crossing_rate": round(mean_qcr, 6),
                "td_var_trend":                "rising" if td_rising else "stable",
                "verdict":                     verdict,
            },
        }

        if output_path is None:
            output_path = os.path.join(
                BACKEND_DIR if "BACKEND_DIR" in dir() else ".",
                "diagnostics", "output", "phase1_repair_monitoring.json"
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[Phase 1] 監控報告已輸出：{output_path}")
        print(f"  verdict={verdict}  mean_EAR={mean_ear:.4f}"
              f"  mean_QCR={mean_qcr:.4f}  td_var_rising={td_rising}")

        return report

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

    def push(self, s, a, r, s_, done, logit_t1, regime_label: str = "sideways"):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, s_, done, logit_t1, regime_label)
        self.pos = (self.pos + 1) % self.capacity

    def clear(self):
        """清空 buffer，重置 pos；capacity 不變。"""
        self.buffer = []
        self.pos    = 0

    def sample(self, batch: int):
        from configs.base_config import DEVICE as _DEVICE
        indices    = np.random.choice(len(self.buffer), batch, replace=False)
        batch_data = [self.buffer[i] for i in indices]

        s      = torch.FloatTensor(np.array([d[0] for d in batch_data])).to(_DEVICE)
        a      = torch.FloatTensor(np.array([d[1] for d in batch_data])).to(_DEVICE)
        r      = torch.FloatTensor(np.array([d[2] for d in batch_data])).unsqueeze(1).to(_DEVICE)
        s_     = torch.FloatTensor(np.array([d[3] for d in batch_data])).to(_DEVICE)
        done   = torch.FloatTensor(np.array([d[4] for d in batch_data])).unsqueeze(1).to(_DEVICE)
        logit  = torch.FloatTensor(np.array([d[5] for d in batch_data])).to(_DEVICE)
        regime = regime_labels_to_tensor([d[6] for d in batch_data], _DEVICE)

        return s, a, r, s_, done, logit, regime

    def __len__(self):
        return len(self.buffer)


# 預設別名：讓舊程式碼繼續工作
SACAgent = SACAgentDirichlet