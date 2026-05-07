"""
Walk-forward 訓練模塊 v7
實驗矩陣（四路對照，同種子、同起點）：

  Run A：Dirichlet（舊）+ CompositeRewardV10（舊）→ 基準
  Run B：LogitDelta（新）+ CompositeRewardV10（舊）→ 測動作慣性
  Run C：Dirichlet（舊）+ LinearDownside（新）    → 測獎勵穩定
  Run D：LogitDelta（新）+ LinearDownside（新）    → 最終方案

監控儀表板（每 50 episode 記錄）：
  - ||ΔL||_2 分佈（Run B/D）
  - HHI
  - std(L)、range(L)（Run B/D）
  - 換倉率累積曲線
  - Critic Q loss（均值 + 標準差）
"""
import os
import pickle
import time
import numpy as np
import torch

from configs.base_config import MODEL_DIR, DEVICE
from configs.trading_config import (
    STOCK_POOL, N_FEATURES, DEFAULT_INITIAL_CAP,
    TRADEABLE_STOCKS, OBSERVABLE_STOCKS, BENCHMARK_STOCK,
    STATE_DIM, N_TRADEABLE,
)
from src.data.loader import load_all_stocks
from src.data.processor import align_features, scale_features
from src.environment.portfolio import PortfolioEnv
from src.agents.sac_agent import (
    SACAgentDirichlet, SACAgentLogitDelta, SACAgent,
)
from src.models.architectures import (
    PortfolioActorDirichlet, PortfolioActorLogitDelta, N_ACTIONS,
)
from src.engine.backtester import run_backtest
from src.utils.common import sanitize, now_str
from diagnostics import register, detect_regime

EPISODES_PER_WINDOW = 200   # 150,000 步 ÷ 750 步/episode（Run D 從頭訓練規格）
VAL_DAYS            = 250
TRAIN_DAYS          = 750

# 實驗矩陣定義
RUN_CONFIGS = {
    # Run A（Composite Reward）和 Run C（Linear Downside + Dirichlet）已拋棄
    # A：無法防止過擬合，Composite Reward 缺乏對風險的適當懲罰
    # C：Q 值發散，LinearDownsideReward 尺度問題無法根治
    "B": {"actor": "logit",     "reward": "composite", "desc": "Action only"},
    "D": {"actor": "logit",     "reward": "linear",    "desc": "Full new"},
}


# ─── 路徑 ─────────────────────────────────────────────────────────────────────

def window_model_path(window: int, run_id: str = "D") -> str:
    return os.path.join(MODEL_DIR, f"portfolio_w{window}_run{run_id}.pkl")

def wf_meta_path(run_id: str = "D") -> str:
    return os.path.join(MODEL_DIR, f"walkforward_meta_run{run_id}.pkl")

def monitor_log_path(run_id: str, window: int) -> str:
    log_dir = os.path.join(MODEL_DIR, "monitor_logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"monitor_run{run_id}_w{window}.pkl")


# ─── Agent 工廠 ───────────────────────────────────────────────────────────────

def _make_agent(run_id: str, state_dim: int, n_stocks: int):
    cfg = RUN_CONFIGS[run_id]
    if cfg["actor"] == "logit":
        return SACAgentLogitDelta(state_dim, n_stocks)
    else:
        return SACAgentDirichlet(state_dim, n_stocks)


def _make_env(train_feat, train_prices, train_volumes,
              run_id: str, initial_capital: float,
              scalers: dict = None):
    cfg = RUN_CONFIGS[run_id]
    return PortfolioEnv(
        train_feat, train_prices, train_volumes,
        initial_capital=initial_capital,
        reward_mode=cfg["reward"],
        scalers=scalers,   # None 時 PortfolioEnv 內部 fit；傳入時直接 transform
    )


# ─── 監控：記錄 Logit 幾何屬性（Run B/D）────────────────────────────────────

def _monitor_logit_delta(agent, obs_batch: torch.Tensor) -> dict:
    """計算 Run B/D 的 Logit 幾何監控指標。"""
    if not isinstance(agent, SACAgentLogitDelta):
        return {}

    agent.actor.eval()
    with torch.no_grad():
        l_t = torch.FloatTensor(agent._logit_state).unsqueeze(0).to(DEVICE)
        raw = agent.actor.forward(obs_batch[:1])

        from src.models.architectures import DELTA_SCALE, LEAKY_GAMMA
        # 修正：與 sample() 一致，使用有界公式 ΔL = 0.1 * raw / (1 + ||raw||_2)
        a_norm    = raw.norm(p=2, dim=-1, keepdim=True)
        delta     = DELTA_SCALE * raw / (1.0 + a_norm)
        new_logit = LEAKY_GAMMA * l_t + delta
        l_norm    = new_logit - new_logit.mean(dim=-1, keepdim=True)

        # per-sample L2 norm（理論上 ≤ DELTA_SCALE = 0.1）
        delta_l2  = float(delta.norm(p=2, dim=-1).mean().item())
        l_std     = float(l_norm.std(dim=-1).mean().item())
        l_range   = float((l_norm.max(dim=-1).values
                           - l_norm.min(dim=-1).values).mean().item())
        l_mean    = float(l_norm.mean().item())

    agent.actor.train()
    return {
        "delta_l2": round(delta_l2, 4),
        "l_std":    round(l_std, 4),
        "l_range":  round(l_range, 4),
        "l_mean":   round(l_mean, 4),
    }


# ─── 單一窗口訓練 ─────────────────────────────────────────────────────────────

@register(
    module="Engine",
    inputs={
        "run_id":          "str",
        "window":          "int",
        "feat_dfs":        "dict",
        "prices_dict":     "dict",
        "volumes_dict":    "dict",
        "dates":           "list",
        "train_start_idx": "int",
        "train_end_idx":   "int",
        "val_start_idx":   "int",
        "val_end_idx":     "int",
        "initial_capital": "float",
        "episodes":        "int",
        "on_episode":      "Callable | None",
    },
    outputs={"return": "dict"},
    notes="v7：四路實驗矩陣的單窗口訓練，含 Logit 幾何監控",
)
def train_window(
    run_id:          str,
    window:          int,
    feat_dfs:        dict,
    prices_dict:     dict,
    volumes_dict:    dict,
    dates:           list,
    train_start_idx: int,
    train_end_idx:   int,
    val_start_idx:   int,
    val_end_idx:     int,
    initial_capital: float = DEFAULT_INITIAL_CAP,
    episodes:        int   = EPISODES_PER_WINDOW,
    on_episode             = None,
) -> dict:

    cfg = RUN_CONFIGS[run_id]
    print(f"\n{'='*60}")
    print(f"Run {run_id}（{cfg['desc']}）窗口 {window} 訓練開始")
    print(f"  Actor: {cfg['actor']}  Reward: {cfg['reward']}")
    print(f"  訓練: {dates[train_start_idx]} ~ {dates[train_end_idx-1]}"
          f"（{train_end_idx - train_start_idx} 筆）")
    print(f"  驗證: {dates[val_start_idx]} ~ {dates[val_end_idx-1]}"
          f"（{val_end_idx - val_start_idx} 筆）")
    print(f"{'='*60}")

    # ── 切訓練期與驗證期特徵 ──────────────────────────────────────────────
    train_feat_raw = {sid: feat_dfs[sid].iloc[train_start_idx:train_end_idx]
                      for sid in feat_dfs}
    train_prices   = {sid: prices_dict[sid][train_start_idx:train_end_idx]
                      for sid in prices_dict}
    train_volumes  = {sid: volumes_dict[sid][train_start_idx:train_end_idx]
                      for sid in volumes_dict}
    train_dates    = dates[train_start_idx:train_end_idx]

    val_feat_raw = {sid: feat_dfs[sid].iloc[val_start_idx:val_end_idx]
                    for sid in feat_dfs}
    val_prices   = {sid: prices_dict[sid][val_start_idx:val_end_idx]
                    for sid in prices_dict}
    val_volumes  = {sid: volumes_dict[sid][val_start_idx:val_end_idx]
                    for sid in volumes_dict}
    val_dates    = dates[val_start_idx:val_end_idx]

    # ── Scaler：只用訓練期 fit，驗證期用訓練期 scaler transform（防止 look-ahead）
    _, train_scalers = scale_features(train_feat_raw, scaler_dict=None)  # fit on train only
    train_feat = {sid: feat_dfs[sid].iloc[train_start_idx:train_end_idx]
                  for sid in feat_dfs}
    val_feat   = {sid: feat_dfs[sid].iloc[val_start_idx:val_end_idx]
                  for sid in feat_dfs}

    # ── 建立訓練環境（傳入訓練期 scaler，避免環境內部重新 fit）─────────────
    env   = _make_env(train_feat, train_prices, train_volumes, run_id,
                      initial_capital, scalers=train_scalers)
    agent = _make_agent(run_id, env.state_dim, env.n_tradeable)

    from src.agents.memory import ReplayBuffer
    from diagnostics import (
        new_logger, diag_random_policy, diag_actor_logits,
        diag_stochastic_vs_deterministic, diag_final_holdings,
        diag_training_curve,
    )

    logger = new_logger(tag=f"run{run_id}_window{window}")
    logger.log(f"\nRun {run_id}（{cfg['desc']}）窗口 {window}")
    logger.log(f"  訓練: {dates[train_start_idx]} ~ {dates[train_end_idx-1]}")
    logger.log(f"  驗證: {dates[val_start_idx]} ~ {dates[val_end_idx-1]}")

    # 嘗試載入舊模型
    existing      = _load_window_model(window, run_id)
    episodes_done = 0

    if (existing is not None
            and existing.get("state_dim") == env.state_dim
            and existing.get("n_stocks")  == env.n_tradeable
            and existing.get("run_id")    == run_id):
        episodes_done = existing.get("episodes_done", 0)
        logger.log(f"  載入 Run {run_id} 窗口 {window} 舊模型"
                   f"（已累積 {episodes_done} 回合），本次新增 {episodes} 回合")
        agent.actor.load_state_dict(existing["actor_state"])
        agent.critic.load_state_dict(existing["critic_state"])
        agent.critic_target.load_state_dict(existing["critic_state"])
        agent.actor.to(DEVICE)
        agent.critic.to(DEVICE)
        agent.critic_target.to(DEVICE)
        with torch.no_grad():
            agent.log_alpha.fill_(
                np.log(max(existing.get("alpha", 1.0), agent.alpha_min))
            )
        agent.alpha = agent.log_alpha.exp().item()
        if isinstance(agent, SACAgentLogitDelta) and "logit_state" in existing:
            agent._logit_state = np.array(existing["logit_state"], dtype=np.float32)
    else:
        logger.log(f"  Run {run_id} 窗口 {window} 從頭訓練")

    # Random policy 診斷
    diag_random_policy(train_feat, train_prices, train_volumes,
                       env.scalers, initial_capital, logger)

    # 監控記錄
    monitor_records = []
    episode_returns = []
    episode_losses  = []
    alphas          = []
    ep_times        = []
    window_start    = time.time()
    nan_count       = 0

    for ep in range(episodes):
        ep_start   = time.time()
        obs        = env.reset()

        # LogitDelta：每個 episode 開始重置 logit 狀態
        if isinstance(agent, SACAgentLogitDelta):
            agent.reset_logit_state()

        done        = False
        step_count  = 0
        ep_losses_d = {"critic_loss": [], "actor_loss": [], "alpha_loss": []}
        trade_count = 0

        while not done:
            action = agent.act(obs)

            if np.isnan(action).any() or np.isinf(action).any():
                action = np.full(env.n_tradeable, 1.0 / (env.n_tradeable + 1))
                nan_count += 1
                if nan_count >= 10:
                    if cfg["actor"] == "logit":
                        agent.actor = PortfolioActorLogitDelta(
                            env.state_dim, env.n_tradeable).to(DEVICE)
                    else:
                        agent.actor = PortfolioActorDirichlet(
                            env.state_dim, env.n_tradeable).to(DEVICE)
                    agent.actor._init_weights()
                    agent.actor_opt = torch.optim.Adam(
                        agent.actor.parameters(), lr=3e-4)
                    nan_count = 0

            prev_obs = obs
            next_obs, reward, done = env.step(action)
            if np.isnan(reward) or np.isinf(reward):
                reward = 0.0

            # LogitDelta 用 push_transition；Dirichlet 用標準 buffer.push
            if isinstance(agent, SACAgentLogitDelta):
                agent.push_transition(obs, action, reward, next_obs, float(done))
            else:
                agent.buffer.push(obs, action, reward, next_obs, float(done))

            step_count  += 1
            trade_count += int(env._traded_this_step)

            # 模塊三：LinearDownsideReward warmup 鎖定 c 後，清空 buffer
            # 確保 buffer 裡不留用 C_MIN 縮放的舊 reward 污染訓練
            if (hasattr(env, "_reward_fn")
                    and hasattr(env._reward_fn, "just_locked")
                    and env._reward_fn.just_locked):
                # 優先用 clear()；若 buffer 未實作則重建一個新的
                if hasattr(agent.buffer, "clear"):
                    agent.buffer.clear()
                else:
                    from src.agents.memory import ReplayBuffer
                    agent.buffer = ReplayBuffer()
                env._reward_fn.just_locked = False
                print(f"  [warmup 鎖定] c={env._reward_fn.c:.6f}，"
                      f"buffer 已清空，從頭收集乾淨樣本")

            if step_count % 2 == 0:
                loss_info = agent.update()
                if loss_info:
                    for k in ep_losses_d:
                        ep_losses_d[k].append(loss_info[k])

            obs = next_obs

        for _ in range(4):
            loss_info = agent.update()
            if loss_info:
                for k in ep_losses_d:
                    ep_losses_d[k].append(loss_info[k])

        ep_sec = time.time() - ep_start
        ep_times.append(ep_sec)

        ret = float(env.portfolio_value())
        ret = ret if (not np.isnan(ret) and not np.isinf(ret)) else 1.0
        episode_returns.append(ret)
        alphas.append(agent.alpha)

        avg_losses = {k: float(np.mean(v)) if v else 0.0
                      for k, v in ep_losses_d.items()}
        episode_losses.append(avg_losses)

        if on_episode:
            on_episode(episodes_done + ep + 1, episodes, ret,
                       agent.alpha, avg_losses, trade_count)

        global_ep = episodes_done + ep + 1
        if (ep + 1) % 10 == 0:
            recent_avg    = float(np.mean(ep_times[-10:]))
            total_elapsed = time.time() - window_start
            eta_sec       = recent_avg * (episodes - ep - 1)
            logger.log(
                f"  [Run {run_id} 窗口{window}] ep {global_ep}  "
                f"return={ret:.4f}  α={agent.alpha:.3f}  "
                f"c_loss={avg_losses['critic_loss']:.4f}  "
                f"trades={trade_count}  "
                f"ep_time={ep_sec:.1f}s  "
                f"elapsed={total_elapsed/60:.1f}m  "
                f"ETA={eta_sec/60:.1f}m"
            )

        # ── Episode 100 特殊診斷報告 ────────────────────────────────────────────
        # 在訓練早期確認 std(L)、HHI、validation return 是否健康
        if (ep + 1) == 100:
            logger.log(f"\n{'='*60}")
            logger.log(f"  [Run {run_id} 窗口{window}] Episode 100 診斷報告")
            logger.log(f"{'='*60}")

            if len(agent.buffer) >= 4:
                if isinstance(agent, SACAgentLogitDelta):
                    _buf_ep100 = agent.buffer.sample(4)[0]
                else:
                    _buf_ep100, *_ = agent.buffer.sample(4)

                _logit_ep100 = _monitor_logit_delta(agent, _buf_ep100)
                with torch.no_grad():
                    if isinstance(agent, SACAgentLogitDelta):
                        _l_ep100 = torch.FloatTensor(
                            agent._logit_state).unsqueeze(0).to(DEVICE)
                        _w_ep100, _, _ = agent.actor.sample(
                            _buf_ep100[:1], logit_state=_l_ep100)
                    else:
                        _w_ep100, _, _ = agent.actor.sample(_buf_ep100[:1])
                    _hhi_ep100 = float(
                        (_w_ep100[0, :env.n_tradeable].cpu().numpy() ** 2).sum())

                # 快速跑驗證集一個 episode
                _val_env = _make_env(
                    val_feat, val_prices, val_volumes, run_id,
                    initial_capital, scalers=train_scalers)
                _val_obs  = _val_env.reset()
                _val_done = False
                if isinstance(agent, SACAgentLogitDelta):
                    agent.reset_logit_state()
                while not _val_done:
                    _va = agent.act(_val_obs, deterministic=True)
                    _val_obs, _, _val_done = _val_env.step(_va)
                _val_ret_ep100 = float(_val_env.portfolio_value())
                if isinstance(agent, SACAgentLogitDelta):
                    agent.reset_logit_state()

                logger.log(f"  std(L)          = {_logit_ep100.get('l_std', 'N/A')}")
                logger.log(f"  HHI             = {_hhi_ep100:.4f}"
                           + ("  *** collapse ***" if _hhi_ep100 > 0.7 else "  ✓"))
                logger.log(f"  val_return      = {_val_ret_ep100:.4f}"
                           + ("  *** 低於 1.0 ***" if _val_ret_ep100 < 1.0 else "  ✓"))
                logger.log(f"  ΔL_L2           = {_logit_ep100.get('delta_l2', 'N/A')}"
                           + ("  *** > 0.1 公式失效 ***"
                              if float(_logit_ep100.get('delta_l2', 0)) > 0.1 else "  ✓"))
                logger.log(f"{'='*60}\n")

        # ── Episode 150 最終判決書診斷報告（七項數據）─────────────────────────
        if (ep + 1) == 150:
            logger.log(f"\n{'='*70}")
            logger.log(f"  [Run {run_id} 窗口{window}] Episode 150 最終判決書")
            logger.log(f"{'='*70}")

            _diag_ok = len(agent.buffer) >= 4
            if _diag_ok:
                if isinstance(agent, SACAgentLogitDelta):
                    _buf150 = agent.buffer.sample(min(32, len(agent.buffer)))[0]
                else:
                    _buf150, *_ = agent.buffer.sample(min(32, len(agent.buffer)))

                # 1. logit 幾何指標
                _logit150 = _monitor_logit_delta(agent, _buf150)

                # 2. HHI + Diversity（stochastic 多次採樣）
                with torch.no_grad():
                    if isinstance(agent, SACAgentLogitDelta):
                        _l150 = torch.FloatTensor(
                            agent._logit_state).unsqueeze(0).to(DEVICE)
                        _w150_det, _, _ = agent.actor.sample(
                            _buf150[:1], logit_state=_l150)
                    else:
                        _w150_det, _, _ = agent.actor.sample(_buf150[:1])
                    _hhi150 = float(
                        (_w150_det[0, :env.n_tradeable].cpu().numpy() ** 2).sum())

                    # 採樣 20 次計算 diversity
                    _actions_stoc = []
                    for _ in range(20):
                        if isinstance(agent, SACAgentLogitDelta):
                            _w_s, _, _ = agent.actor.sample(
                                _buf150[:1], logit_state=_l150)
                        else:
                            _w_s, _, _ = agent.actor.sample(_buf150[:1])
                        _actions_stoc.append(
                            _w_s[0, :env.n_tradeable].cpu().numpy())
                    import numpy as _np
                    _stoc_arr = _np.array(_actions_stoc)
                    _diversity150 = float(_stoc_arr.std(axis=0).mean())

                # 3. log_pi 統計（從 buffer 批量採樣）
                with torch.no_grad():
                    if isinstance(agent, SACAgentLogitDelta):
                        _l_batch = torch.zeros(
                            _buf150.shape[0], N_ACTIONS, device=DEVICE)
                        _, _, _lp_batch = agent.actor.sample(
                            _buf150, logit_state=_l_batch)
                    else:
                        _, _lp_batch, _ = agent.actor.sample(_buf150)
                    _lp_mean = float(_lp_batch.mean().item())
                    _lp_std  = float(_lp_batch.std().item())

                # 4. validation return
                _val_env150 = _make_env(
                    val_feat, val_prices, val_volumes, run_id,
                    initial_capital, scalers=train_scalers)
                _vo150  = _val_env150.reset()
                _vd150  = False
                if isinstance(agent, SACAgentLogitDelta):
                    agent.reset_logit_state()
                while not _vd150:
                    _va150  = agent.act(_vo150, deterministic=True)
                    _vo150, _, _vd150 = _val_env150.step(_va150)
                _val_ret150 = float(_val_env150.portfolio_value())
                if isinstance(agent, SACAgentLogitDelta):
                    agent.reset_logit_state()

                # ── 輸出七項數據 ──────────────────────────────────────────────
                _hhi_ok  = 0.3 <= _hhi150 <= 0.6
                _div_ok  = _diversity150 > 0.02
                _val_ok  = _val_ret150 > 1.0

                logger.log(f"  {'項目':<18}  {'數值':>12}  {'判決'}")
                logger.log(f"  {'-'*55}")
                logger.log(f"  {'alpha':<18}  {agent.alpha:>12.4f}  "
                           + ("⚠ 單調下降" if agent.alpha < 0.5 else "✓"))
                logger.log(f"  {'log_pi_mean':<18}  {_lp_mean:>12.4f}  "
                           + ("⚠ Q值壓制" if _lp_mean < -5.0 else "✓"))
                logger.log(f"  {'log_pi_std':<18}  {_lp_std:>12.4f}")
                logger.log(f"  {'diversity':<18}  {_diversity150:>12.4f}  "
                           + ("✓" if _div_ok else "⚠ 探索不足 < 0.02"))
                logger.log(f"  {'HHI':<18}  {_hhi150:>12.4f}  "
                           + ("✓ 0.3~0.6" if _hhi_ok else
                              "⚠ collapse" if _hhi150 > 0.6 else "⚠ 過度分散"))
                logger.log(f"  {'std(L)':<18}  {_logit150.get('l_std','N/A'):>12}  "
                           + ("⚠ 爆炸" if float(_logit150.get('l_std',0)) > 5.0
                              else "⚠ 崩潰" if float(_logit150.get('l_std',1)) < 0.1
                              else "✓"))
                logger.log(f"  {'validation_return':<18}  {_val_ret150:>12.4f}  "
                           + ("✓ 正報酬" if _val_ok else "⚠ 未獲利"))
                logger.log(f"  {'-'*55}")

                # 判決
                _pass = sum([_hhi_ok, _div_ok, _val_ok])
                if _pass == 3:
                    logger.log("  ✅ 三項通過：RL 架構成功，SAC-Stock-v7 正式定版")
                elif _pass == 2 and _val_ok is False:
                    logger.log("  ⚠️  僅結構通過，無獲利：RL 任務結束，轉向特徵工程")
                elif not _hhi_ok:
                    logger.log("  ❌ 第一項未過：仍有強主導訊號，凍結 RL，啟動特徵清查")
                else:
                    logger.log(f"  ⚠️  通過 {_pass}/3 項，繼續觀察")
                logger.log(f"{'='*70}\n")

        # 每 50 episode 記錄監控指標
        if (ep + 1) % 50 == 0:
            if len(agent.buffer) >= 4:
                if isinstance(agent, SACAgentLogitDelta):
                    buf_data = agent.buffer.sample(4)
                    _obs_batch = buf_data[0]
                else:
                    buf_states, _, _, _, _ = agent.buffer.sample(4)
                    _obs_batch = buf_states

                # Logit 幾何監控（Run B/D）
                logit_metrics = _monitor_logit_delta(agent, _obs_batch)

                # HHI（所有 run）
                with torch.no_grad():
                    if isinstance(agent, SACAgentLogitDelta):
                        l_t = torch.FloatTensor(agent._logit_state).unsqueeze(0).to(DEVICE)
                        w, _, _ = agent.actor.sample(_obs_batch[:1], logit_state=l_t)
                    else:
                        w, _, _ = agent.actor.sample(_obs_batch[:1])
                    w_stock = w[0, :env.n_tradeable].cpu().numpy()
                    hhi = float((w_stock ** 2).sum())

                # Critic Q loss std（最近 50 episode）
                recent_losses = ep_losses_d["critic_loss"][
                    max(0, len(ep_losses_d["critic_loss"]) - 50):]
                q_loss_std = float(np.std(recent_losses)) if recent_losses else 0.0

                record = {
                    "episode":     global_ep,
                    "return":      round(ret, 4),
                    "hhi":         round(hhi, 4),
                    "alpha":       round(agent.alpha, 4),
                    "c_loss_mean": round(avg_losses["critic_loss"], 4),
                    "c_loss_std":  round(q_loss_std, 4),
                    "trade_count": trade_count,
                    **logit_metrics,
                }
                monitor_records.append(record)

                # 印出監控摘要
                logger.log(f"\n  [監控 ep={global_ep}]")
                logger.log(f"    HHI={hhi:.4f}  alpha={agent.alpha:.3f}  "
                           f"c_loss_std={q_loss_std:.4f}")
                if logit_metrics:
                    logger.log(f"    ΔL_L2={logit_metrics.get('delta_l2','N/A')}  "
                               f"std(L)={logit_metrics.get('l_std','N/A')}  "
                               f"range(L)={logit_metrics.get('l_range','N/A')}  "
                               f"mean(L)={logit_metrics.get('l_mean','N/A')}")

                # 健康度警告
                if hhi > 0.7:
                    logger.log(f"    *** 警告：HHI={hhi:.4f} > 0.7，疑似 one-hot collapse ***")
                if logit_metrics.get("l_std", 1.0) < 1.0:
                    logger.log(f"    *** 警告：std(L)={logit_metrics['l_std']:.4f} < 1.0，"
                               f"Logit 可能正在崩潰 ***")
                if logit_metrics.get("l_std", 1.0) > 5.0:
                    logger.log(f"    *** 警告：std(L)={logit_metrics['l_std']:.4f} > 5.0，"
                               f"Logit 可能正在爆炸 ***")

    # 儲存監控記錄
    with open(monitor_log_path(run_id, window), "wb") as f:
        pickle.dump(monitor_records, f)

    # 訓練後診斷
    logger.log(f"\n[Run {run_id} 窗口{window}] 訓練後 Actor 診斷...")

    _holdings_snapshot = {
        "capital":   env.capital,
        "lots_held": env.lots_held.copy(),
        "odd_held":  env.odd_held.copy(),
        "step_idx":  env.step_idx,
    }
    _diag_obs = env.reset()
    diag_stochastic_vs_deterministic(agent, _diag_obs, logger)
    diag_final_holdings(env, logger, snapshot=_holdings_snapshot)
    diag_training_curve(episode_returns, episode_losses, alphas, logger)

    # ── 回測（均使用訓練期 scaler，確保驗證期無 look-ahead）────────────────
    from diagnostics import diag_backtest_curve
    logger.log(f"\n[Run {run_id} 窗口{window}] 訓練集回測診斷...")
    diag_backtest_curve(
        agent.actor, train_scalers,
        train_feat, train_prices, train_volumes,
        TRADEABLE_STOCKS, initial_capital, train_dates,
        logger, check_interval=100
    )

    bt_train = run_backtest(
        agent.actor, train_scalers,
        train_feat, train_prices, train_volumes,
        TRADEABLE_STOCKS, initial_capital, [], train_dates
    )
    bt_val = run_backtest(
        agent.actor, train_scalers,   # 驗證期也用訓練期 scaler transform
        val_feat, val_prices, val_volumes,
        TRADEABLE_STOCKS, initial_capital, [], val_dates
    )

    benchmark_prices_val = val_prices[BENCHMARK_STOCK]
    regime = detect_regime(benchmark_prices_val)

    train_return  = bt_train["total_return"]
    val_return    = bt_val["total_return"]
    overfit_ratio = (train_return / val_return) if abs(val_return) > 1e-6 else float("inf")

    # 換倉率統計
    total_trades    = sum(r.get("trade_count", 0) for r in monitor_records)
    avg_trade_rate  = total_trades / max(len(monitor_records), 1)

    summary = {
        "run_id":          run_id,
        "window":          window,
        "train_start":     dates[train_start_idx],
        "train_end":       dates[train_end_idx - 1],
        "val_start":       dates[val_start_idx],
        "val_end":         dates[val_end_idx - 1],
        "train_return":    train_return,
        "val_return":      val_return,
        "val_win_rate":    bt_val["win_rate"],
        "regime":          regime,
        "episodes":        episodes,
        "episodes_done":   episodes_done + episodes,
        "initial_capital": initial_capital,
        "overfit_ratio":   round(overfit_ratio, 1),
        "avg_trade_rate":  round(avg_trade_rate, 1),
        "monitor_records": monitor_records,
    }

    logger.log(f"\n[Run {run_id} 窗口{window}] 訓練結果：")
    logger.log(f"  訓練集報酬: {train_return:.2f}%")
    logger.log(f"  驗證集報酬: {val_return:.2f}%")
    logger.log(f"  過擬合比率: {overfit_ratio:.1f}x")
    logger.log(f"  Regime: {regime}")
    logger.log(f"  平均換倉率: {avg_trade_rate:.1f} trades/50ep")
    logger.close()

    _save_window_model(window, run_id, agent, env, summary)
    return summary


# ─── 模型存取 ─────────────────────────────────────────────────────────────────

def _save_window_model(window: int, run_id: str,
                       agent, env, summary: dict):
    os.makedirs(MODEL_DIR, exist_ok=True)
    agent.actor.cpu()
    agent.critic.cpu()
    payload = {
        "actor_state":   agent.actor.state_dict(),
        "critic_state":  agent.critic.state_dict(),
        "alpha":         float(agent.alpha),
        "state_dim":     env.state_dim,
        "n_stocks":      env.n_tradeable,
        "stock_ids":     env.tradeable_ids,
        "scalers":       env.scalers,
        "summary":       summary,
        "saved_at":      now_str(),
        "window":        window,
        "run_id":        run_id,
        "episodes_done": summary.get("episodes_done", 0),
    }
    if hasattr(agent, "_logit_state"):
        payload["logit_state"] = agent._logit_state.tolist()

    with open(window_model_path(window, run_id), "wb") as f:
        pickle.dump(payload, f)
    agent.actor.to(DEVICE)
    agent.critic.to(DEVICE)
    agent.critic_target.to(DEVICE)
    print(f"[Run {run_id}] 窗口 {window} 模型已儲存"
          f"（累積 {summary.get('episodes_done', 0)} 回合）")


def _load_window_model(window: int, run_id: str) -> dict | None:
    path = window_model_path(window, run_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── 實驗矩陣主流程 ───────────────────────────────────────────────────────────

@register(
    module="Engine",
    inputs={
        "period":          "str",
        "initial_capital": "float",
        "episodes":        "int",
        "runs":            "list[str]",
        "on_episode":      "Callable | None",
    },
    outputs={"return": "dict"},
    notes="v7：四路實驗矩陣（A/B/C/D），順序跑，各自儲存獨立模型",
)
def train_experiment_matrix(
    period:          str   = "6y",
    initial_capital: float = DEFAULT_INITIAL_CAP,
    episodes:        int   = EPISODES_PER_WINDOW,
    runs:            list  = None,
    on_episode             = None,
    seed:            int   = 42,
) -> dict:
    """
    執行實驗矩陣。

    Args:
        runs: 要跑的 run 列表，例如 ["B", "D"]
              None 表示跑 ["D"]（Run A/C 已拋棄）
        seed: 全局隨機種子，確保四路 run 的模型初始化與環境起點完全相同
    """
    import random
    runs = runs or ["D"]   # 預設只跑 Run D；如需 B 請明確傳入 runs=["B","D"]

    # 過濾掉已拋棄的 Run A 和 C（防止外部傳入舊的 runs 列表）
    valid_runs = [r for r in runs if r in RUN_CONFIGS]
    if len(valid_runs) < len(runs):
        dropped = [r for r in runs if r not in RUN_CONFIGS]
        print(f"  [警告] 以下 Run 已拋棄，自動跳過：{dropped}")
    runs = valid_runs
    if not runs:
        raise ValueError(f"沒有有效的 Run 可執行，RUN_CONFIGS 只包含：{list(RUN_CONFIGS.keys())}")

    print("\n" + "=" * 60)
    print(f"Walk-forward 實驗矩陣（Run {'/'.join(runs)}）")
    for r in runs:
        cfg = RUN_CONFIGS[r]
        print(f"  Run {r}：{cfg['desc']}"
              f"（Actor={cfg['actor']}，Reward={cfg['reward']}）")
    print("=" * 60)

    stocks = load_all_stocks(period)
    feat_dfs, prices_dict, volumes_dict, feat_names, dates = align_features(stocks)
    n_total = len(dates)
    print(f"資料範圍: {dates[0]} ~ {dates[-1]}（{n_total} 筆）")

    # 計算窗口索引（所有 run 共用相同窗口）
    windows = []
    step    = VAL_DAYS
    for w in range(3):
        train_start = w * step
        train_end   = train_start + TRAIN_DAYS
        val_start   = train_end
        val_end     = min(val_start + VAL_DAYS, n_total)
        if val_end > n_total or train_end > n_total:
            print(f"  窗口 {w+1} 資料不足，跳過")
            break
        windows.append({
            "window":          w + 1,
            "train_start_idx": train_start,
            "train_end_idx":   train_end,
            "val_start_idx":   val_start,
            "val_end_idx":     val_end,
        })
        print(f"  窗口 {w+1}: {dates[train_start]}~{dates[train_end-1]}"
              f"（訓）{dates[val_start]}~{dates[val_end-1]}（驗）")

    if len(windows) < 2:
        raise ValueError(f"資料不足，需要至少 {TRAIN_DAYS + VAL_DAYS * 2} 筆")

    # 依序執行各 Run
    all_results = {}
    for run_id in runs:
        # 模塊四：每個 Run 開始前重置到相同種子，確保模型初始化完全一致
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        print(f"\n{'='*60}")
        print(f"開始 Run {run_id}（{RUN_CONFIGS[run_id]['desc']}）[seed={seed}]")
        print(f"{'='*60}")

        run_results = []
        for w_config in windows:
            result = train_window(
                run_id          = run_id,
                window          = w_config["window"],
                feat_dfs        = feat_dfs,
                prices_dict     = prices_dict,
                volumes_dict    = volumes_dict,
                dates           = dates,
                train_start_idx = w_config["train_start_idx"],
                train_end_idx   = w_config["train_end_idx"],
                val_start_idx   = w_config["val_start_idx"],
                val_end_idx     = w_config["val_end_idx"],
                initial_capital = initial_capital,
                episodes        = episodes,
                on_episode      = on_episode,
            )
            run_results.append(result)

        all_results[run_id] = run_results

        # 儲存 Run 的 meta
        meta = {
            "run_id":          run_id,
            "window_results":  run_results,
            "saved_at":        now_str(),
            "period":          period,
        }
        with open(wf_meta_path(run_id), "wb") as f:
            pickle.dump(meta, f)
        print(f"\n[Run {run_id}] 元資料已儲存：{wf_meta_path(run_id)}")

    # 列印比較摘要
    _print_comparison_summary(all_results, runs)

    return sanitize({"runs": all_results})


def _print_comparison_summary(all_results: dict, runs: list):
    """列印四路對照的最終比較摘要。"""
    print("\n" + "=" * 70)
    print("實驗矩陣結果比較")
    print("=" * 70)
    print(f"  {'Run':<6} {'Actor':<12} {'Reward':<12} "
          f"{'訓練報酬':>10} {'驗證報酬':>10} {'換倉率':>8} {'過擬合':>8}")
    print("  " + "-" * 66)

    for run_id in runs:
        if run_id not in all_results:
            continue
        cfg = RUN_CONFIGS[run_id]
        results = all_results[run_id]
        avg_train = np.mean([r["train_return"] for r in results])
        avg_val   = np.mean([r["val_return"]   for r in results])
        avg_trade = np.mean([r.get("avg_trade_rate", 0) for r in results])
        avg_of    = np.mean([abs(r.get("overfit_ratio", 0)) for r in results])
        print(f"  {run_id:<6} {cfg['actor']:<12} {cfg['reward']:<12} "
              f"{avg_train:>9.1f}% {avg_val:>9.1f}% "
              f"{avg_trade:>7.0f} {avg_of:>7.1f}x")

    print("=" * 70)
    print("\n升格 v8 判准：")
    print("  Run D turnover 比 Run A 降 ≥35%")
    print("  且 Q loss 走勢平穩（std 不爆）")
    print("  且 Sharpe 不低於 Run A")


# ─── 單一 Run 的 Walk-forward（向後相容）────────────────────────────────────

@register(
    module="Engine",
    inputs={
        "period":          "str",
        "initial_capital": "float",
        "episodes":        "int",
        "run_id":          "str",
        "on_episode":      "Callable | None",
    },
    outputs={"return": "dict"},
    notes="單一 Run 的 Walk-forward（向後相容），預設 Run A",
)
def train_walkforward(
    period:          str   = "6y",
    initial_capital: float = DEFAULT_INITIAL_CAP,
    episodes:        int   = EPISODES_PER_WINDOW,
    run_id:          str   = "D",
    on_episode             = None,
) -> dict:
    """向後相容的單一 Run walk-forward 入口。"""
    result = train_experiment_matrix(
        period          = period,
        initial_capital = initial_capital,
        episodes        = episodes,
        runs            = [run_id],
        on_episode      = on_episode,
    )
    return result["runs"].get(run_id, {})
@register(
    module="Engine",
    inputs={"period": "str", "run_id": "str"},
    outputs={"return": "dict"},
    notes="detect_regime → 選對應窗口模型 → 最新特徵推論 → 輸出各股建議倉位",
)
def predict_walkforward(period: str = "6y") -> dict:
    """
    從 Run A/B/C/D 中，依當前 Regime 自動選出驗證報酬最高的 Run + 窗口，
    推論明日建議倉位。

    選模型邏輯：
      1. 偵測當前市場 Regime（bull / bear / sideways）
      2. 對每個 Run，篩選 regime 相符的窗口；若無相符則取所有窗口
      3. 比較所有候選（Run × 窗口）的 val_return，選最高者
      4. 載入該 Run + 窗口的 actor 推論
    """
    stocks           = load_all_stocks(period)
    benchmark_prices = stocks[BENCHMARK_STOCK]["Close"].values
    current_regime   = detect_regime(benchmark_prices)

    # ── 跨四個 Run 選出最佳候選 ───────────────────────────────────────────
    best_run    = None
    best_window = None
    best_val    = -float("inf")

    for rid in ["B", "D"]:   # Run A 和 C 已拋棄
        m_path = wf_meta_path(rid)
        if not os.path.exists(m_path):
            continue
        with open(m_path, "rb") as f:
            m = pickle.load(f)
        window_results = m.get("window_results", [])
        if not window_results:
            continue

        # 優先選 regime 相符的窗口；若無相符則用全部窗口
        matched = [r for r in window_results if r.get("regime") == current_regime]
        candidates = matched if matched else window_results

        for r in candidates:
            val = r.get("val_return", -float("inf"))
            # 確認對應的 model 檔案確實存在
            if val > best_val and _load_window_model(r["window"], rid) is not None:
                best_val    = val
                best_run    = rid
                best_window = r["window"]

    if best_run is None:
        raise ValueError(
            "找不到任何已完成的 Run 元資料，請先執行 train_experiment_matrix()"
        )

    print(f"\n[預測] 選擇 Run {best_run} 窗口 {best_window}"
          f"（val_return={best_val:.2f}%，regime={current_regime}）")

    payload = _load_window_model(best_window, best_run)
    cfg     = RUN_CONFIGS[best_run]

    if cfg["actor"] == "logit":
        actor = PortfolioActorLogitDelta(payload["state_dim"], payload["n_stocks"])
    else:
        actor = PortfolioActorDirichlet(payload["state_dim"], payload["n_stocks"])

    actor.load_state_dict(payload["actor_state"])
    actor.to(DEVICE)
    actor.eval()

    scalers   = payload["scalers"]
    stock_ids = payload["stock_ids"]

    from src.data.processor import compute_features
    feat_dfs   = {sid: compute_features(stocks[sid]) for sid in OBSERVABLE_STOCKS}
    common_idx = None
    for df in feat_dfs.values():
        common_idx = (df.index if common_idx is None
                      else common_idx.intersection(df.index))
    feat_dfs = {sid: feat_dfs[sid].loc[common_idx] for sid in OBSERVABLE_STOCKS}

    latest_feats = []
    for sid in OBSERVABLE_STOCKS:
        feat = feat_dfs[sid].iloc[[-1]].values.astype(np.float64)
        feat = np.where(np.isposinf(feat),  1e6, feat)
        feat = np.where(np.isneginf(feat), -1e6, feat)
        feat = np.where(np.isnan(feat),      0.0, feat)
        scaled = np.clip(scalers[sid].transform(feat), -5.0, 5.0)
        latest_feats.append(scaled[0])

    latest_feats = np.concatenate(latest_feats)
    obs = np.concatenate([
        latest_feats,
        np.zeros(N_TRADEABLE),
        np.zeros(N_TRADEABLE),
        [1.0],
    ]).astype(np.float32)

    s = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if cfg["actor"] == "logit":
            l_t = torch.zeros(1, N_ACTIONS, device=DEVICE)
            w, _, _ = actor.sample(s, logit_state=l_t)
        else:
            w, _, _ = actor.sample(s)
    target = w.squeeze(0).cpu().numpy()[:len(TRADEABLE_STOCKS)]

    recommendations = []
    for j, sid in enumerate(TRADEABLE_STOCKS):
        pos    = float(target[j])
        name   = next(s_["name"] for s_ in STOCK_POOL if s_["id"] == sid)
        price  = float(stocks[sid]["Close"].loc[common_idx].iloc[-1])
        action = "買入" if pos > 0.25 else ("持有" if pos > 0.05 else "觀望")
        recommendations.append({
            "stock_id":     sid,
            "stock_name":   name,
            "action":       action,
            "target_pct":   round(pos * 100, 1),
            "latest_price": round(price, 2),
        })

    recommendations.sort(key=lambda x: -x["target_pct"])

    return sanitize({
        "as_of_date":      feat_dfs[stock_ids[0]].index[-1].strftime("%Y-%m-%d"),
        "recommendations": recommendations,
        "cash_pct":        round(max(1.0 - float(target.sum()), 0.0) * 100, 1),
        "current_regime":  current_regime,
        "selected_window": int(best_window),
        "selected_run":    best_run,
        "run_id":          best_run,
        "best_val_return": round(best_val, 2),
        "model_saved_at":  payload.get("saved_at"),
    })