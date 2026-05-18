"""
engine/trainer_walk_forward.py
滾動窗口 Walk-Forward 訓練引擎（原 walk_forward.py 核心）。

實驗矩陣（四路對照，現役 Run B/D）：
  Run B：LogitDelta + CompositeRewardV10  → 測動作慣性
  Run D：LogitDelta + LinearDownside      → 最終方案

依賴共用模組：
  engine.persistence  → save_window_model / load_window_model
                        wf_meta_path / monitor_log_path
  engine.rules        → extract_portfolio_rules
  engine.factory      → RUN_CONFIGS / make_env / make_agent / rebuild_actor

公開 API：
  train_window(run_id, window, feat_dfs, ...) -> dict
  train_experiment_matrix(period, ...)        -> dict
  train_walkforward(period, ...)              -> dict   （向後相容）
  predict_walkforward(period)                 -> dict
"""

import copy
import os
import pickle
import random
import time

import numpy as np
import torch

from configs.base_config import MODEL_DIR
from configs.base_config import DEVICE
from configs.trading_config import (
    STOCK_POOL, N_FEATURES, DEFAULT_INITIAL_CAP,
    TRADEABLE_STOCKS, OBSERVABLE_STOCKS, BENCHMARK_STOCK,
    N_TRADEABLE,
)
from src.data.loader import load_all_stocks
from src.data.processor import align_features
from src.agents.sac_agent import SACAgentLogitDelta
from src.models.architectures import N_ACTIONS
from src.engine.backtester import run_backtest
from src.utils.common import sanitize, now_str
from diagnostics import register, detect_regime

from src.engine.persistence import (
    save_window_model, load_window_model,
    wf_meta_path, monitor_log_path,
)
from src.engine.factory import RUN_CONFIGS, make_env, make_agent, rebuild_actor


# ─── 超參數 ───────────────────────────────────────────────────────────────────

EPISODES_PER_WINDOW = 200
VAL_DAYS            = 250
TRAIN_DAYS          = 750


# ─── Logit 幾何監控（Run B/D 專用）──────────────────────────────────────────

def _monitor_logit_delta(agent, obs_batch: torch.Tensor) -> dict:
    """計算 Run B/D 的 Logit 幾何監控指標。"""
    if not isinstance(agent, SACAgentLogitDelta):
        return {}

    from src.models.architectures import DELTA_SCALE, LEAKY_GAMMA

    agent.actor.eval()
    with torch.no_grad():
        l_t = torch.FloatTensor(agent._logit_state).unsqueeze(0).to(DEVICE)
        raw = agent.actor.forward(obs_batch[:1])

        a_norm    = raw.norm(p=2, dim=-1, keepdim=True)
        delta     = DELTA_SCALE * raw / (1.0 + a_norm)
        new_logit = LEAKY_GAMMA * l_t + delta
        l_norm    = new_logit - new_logit.mean(dim=-1, keepdim=True)

        delta_l2 = float(delta.norm(p=2, dim=-1).mean().item())
        l_std    = float(l_norm.std(dim=-1).mean().item())
        l_range  = float(
            (l_norm.max(dim=-1).values - l_norm.min(dim=-1).values).mean().item()
        )
        l_mean   = float(l_norm.mean().item())

    agent.actor.train()
    return {
        "delta_l2": round(delta_l2, 4),
        "l_std":    round(l_std, 4),
        "l_range":  round(l_range, 4),
        "l_mean":   round(l_mean, 4),
    }


# ─── 早停輔助 ─────────────────────────────────────────────────────────────────

_EARLY_STOP_PATIENCE   = 20
_EARLY_STOP_LAMBDA1    = 0.1
_EARLY_STOP_LAMBDA2    = 0.5
_EARLY_STOP_HHI_TARGET = 0.35


def _composite_health_score(val_ret_pct: float, hhi: float, l_std: float) -> float:
    return (
        val_ret_pct
        - _EARLY_STOP_LAMBDA1 * abs(hhi - _EARLY_STOP_HHI_TARGET)
        - _EARLY_STOP_LAMBDA2 * max(0.0, 1.0 - l_std)
    )


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
    print(
        f"  訓練: {dates[train_start_idx]} ~ {dates[train_end_idx-1]}"
        f"（{train_end_idx - train_start_idx} 筆）"
    )
    print(
        f"  驗證: {dates[val_start_idx]} ~ {dates[val_end_idx-1]}"
        f"（{val_end_idx - val_start_idx} 筆）"
    )
    print(f"{'='*60}")

    # ── 切分資料 ──────────────────────────────────────────────────────────────
    train_feat = {sid: feat_dfs[sid].iloc[train_start_idx:train_end_idx] for sid in feat_dfs}
    train_prices  = {sid: prices_dict[sid][train_start_idx:train_end_idx]  for sid in prices_dict}
    train_volumes = {sid: volumes_dict[sid][train_start_idx:train_end_idx] for sid in volumes_dict}
    train_dates   = dates[train_start_idx:train_end_idx]

    val_feat    = {sid: feat_dfs[sid].iloc[val_start_idx:val_end_idx] for sid in feat_dfs}
    val_prices  = {sid: prices_dict[sid][val_start_idx:val_end_idx]   for sid in prices_dict}
    val_volumes = {sid: volumes_dict[sid][val_start_idx:val_end_idx]  for sid in volumes_dict}
    val_dates   = dates[val_start_idx:val_end_idx]

    # Scaler 只用訓練期 fit（防 look-ahead）
    train_scalers: dict = {}

    # ── 建環境與 Agent ─────────────────────────────────────────────────────────
    env   = make_env(train_feat, train_prices, train_volumes,
                     initial_capital, run_id=run_id, scalers=train_scalers)
    agent = make_agent(env.state_dim, env.n_tradeable, run_id=run_id)

    from src.agents.memory import ReplayBuffer
    from diagnostics import (
        new_logger, diag_random_policy, diag_stochastic_vs_deterministic,
        diag_final_holdings, diag_training_curve,
    )

    logger = new_logger(tag=f"run{run_id}_window{window}")
    logger.log(f"\nRun {run_id}（{cfg['desc']}）窗口 {window}")
    logger.log(f"  訓練: {dates[train_start_idx]} ~ {dates[train_end_idx-1]}")
    logger.log(f"  驗證: {dates[val_start_idx]} ~ {dates[val_end_idx-1]}")

    # ── 嘗試載入舊模型（接續訓練）──────────────────────────────────────────────
    existing      = load_window_model(window, run_id)
    episodes_done = 0

    if (
        existing is not None
        and existing.get("state_dim") == env.state_dim
        and existing.get("n_stocks")  == env.n_tradeable
        and existing.get("run_id")    == run_id
    ):
        episodes_done = existing.get("episodes_done", 0)

        # ── Actor 載入（架構未變，直接載入）────────────────────────────────
        agent.actor.load_state_dict(existing["actor_state"])
        agent.actor.to(DEVICE)

        # ── Critic 載入（需偵測架構是否相容）───────────────────────────────
        critic_keys     = set(existing.get("critic_state", {}).keys())
        iqn_keys        = {"regime_embed.embed.weight", "tau_embed.fc.weight", "encoder.0.weight"}
        old_critic_keys = {"extractor1.net.0.weight", "q1.0.weight"}

        critic_compatible = bool(critic_keys & iqn_keys) and not bool(critic_keys & old_critic_keys)

        if critic_compatible:
            agent.critic.load_state_dict(existing["critic_state"])
            agent.critic_target.load_state_dict(existing["critic_state"])
            agent.critic.to(DEVICE)
            agent.critic_target.to(DEVICE)
            logger.log(
                f"  載入 Run {run_id} 窗口 {window} 舊模型（Actor + IQN Critic）"
                f"  已累積 {episodes_done} 回合，本次新增 {episodes} 回合"
            )
        else:
            # 舊版 PortfolioCritic 架構（extractor1/q1/q2），無法載入 IQN Critic
            # Critic 從頭訓練；Actor 保留繼續使用
            agent.critic.to(DEVICE)
            agent.critic_target.to(DEVICE)
            logger.log(
                f"  載入 Run {run_id} 窗口 {window} 舊模型（Actor 接續，Critic 架構不相容→重置）"
                f"  已累積 {episodes_done} 回合，本次新增 {episodes} 回合"
            )

        with torch.no_grad():
            agent.log_alpha.fill_(
                np.log(max(existing.get("alpha", 1.0), agent.alpha_min))
            )
        agent.alpha = agent.log_alpha.exp().item()
        if isinstance(agent, SACAgentLogitDelta) and "logit_state" in existing:
            agent._logit_state = np.array(existing["logit_state"], dtype=np.float32)
    else:
        logger.log(f"  Run {run_id} 窗口 {window} 從頭訓練")

    diag_random_policy(train_feat, train_prices, train_volumes,
                       env.scalers, initial_capital, logger)

    # ── 訓練迴圈 ───────────────────────────────────────────────────────────────
    monitor_records = []
    episode_returns = []
    episode_losses  = []
    alphas          = []
    ep_times        = []
    window_start    = time.time()
    nan_count       = 0

    best_score       = -float("inf")
    best_score_ep    = 0
    patience_count   = 0
    best_agent_state = None

    for ep in range(episodes):
        ep_start = time.time()
        obs      = env.reset()

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
                    agent.actor = rebuild_actor(run_id, env.state_dim, env.n_tradeable)
                    agent.actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=3e-4)
                    nan_count = 0

            prev_obs = obs
            next_obs, reward, done = env.step(action)
            if np.isnan(reward) or np.isinf(reward):
                reward = 0.0

            if isinstance(agent, SACAgentLogitDelta):
                agent.push_transition(obs, action, reward, next_obs, float(done))
            else:
                agent.buffer.push(obs, action, reward, next_obs, float(done))

            step_count  += 1
            trade_count += int(env._traded_this_step)

            # LinearDownsideReward warmup 鎖定後清 buffer
            if (
                hasattr(env, "_reward_fn")
                and hasattr(env._reward_fn, "just_locked")
                and env._reward_fn.just_locked
            ):
                if hasattr(agent.buffer, "clear"):
                    agent.buffer.clear()
                else:
                    agent.buffer = ReplayBuffer()
                env._reward_fn.just_locked = False
                print(f"  [warmup 鎖定] c={env._reward_fn.c:.6f}，buffer 已清空")

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

        avg_losses = {k: float(np.mean(v)) if v else 0.0 for k, v in ep_losses_d.items()}
        episode_losses.append(avg_losses)

        global_ep = episodes_done + ep + 1

        # Phase 1：彙總本 episode 的監控指標
        if hasattr(agent, "flush_phase1_episode"):
            agent.flush_phase1_episode(global_ep)

        if on_episode:
            on_episode(global_ep, episodes, ret, agent.alpha, avg_losses, trade_count)

        # ── Composite Health Score 早停（每 5 episode）─────────────────────────
        if (ep + 1) % 5 == 0 and len(agent.buffer) >= agent.batch:
            with torch.no_grad():
                if isinstance(agent, SACAgentLogitDelta):
                    _buf_es = agent.buffer.sample(4)
                    _obs_es = _buf_es[0]
                    _l_es   = torch.FloatTensor(agent._logit_state).unsqueeze(0).to(DEVICE)
                    _w_es, _, _ = agent.actor.sample(_obs_es[:1], logit_state=_l_es)
                else:
                    _obs_es, *_ = agent.buffer.sample(4)
                    _w_es, _, _ = agent.actor.sample(_obs_es[:1])
                _hhi_es = float((_w_es[0, :env.n_tradeable].cpu().numpy() ** 2).sum())

            _logit_es = _monitor_logit_delta(agent, _obs_es)
            _l_std_es = float(_logit_es.get("l_std", 1.0))

            _val_env_es = make_env(
                val_feat, val_prices, val_volumes,
                initial_capital, run_id=run_id, scalers=train_scalers
            )
            _vo_es = _val_env_es.reset()
            _vd_es = False
            if isinstance(agent, SACAgentLogitDelta):
                agent.reset_logit_state()
            while not _vd_es:
                _va_es = agent.act(_vo_es, deterministic=True)
                _vo_es, _, _vd_es = _val_env_es.step(_va_es)
            _val_ret_es = (float(_val_env_es.portfolio_value()) - 1.0) * 100
            if isinstance(agent, SACAgentLogitDelta):
                agent.reset_logit_state()

            _score_es = _composite_health_score(_val_ret_es, _hhi_es, _l_std_es)

            if _score_es > best_score:
                best_score    = _score_es
                best_score_ep = global_ep
                patience_count = 0
                best_agent_state = {
                    "actor":     copy.deepcopy(agent.actor.state_dict()),
                    "critic":    copy.deepcopy(agent.critic.state_dict()),
                    "log_alpha": agent.log_alpha.item(),
                }
                if isinstance(agent, SACAgentLogitDelta):
                    best_agent_state["logit_state"] = agent._logit_state.copy()
                logger.log(
                    f"  [早停] ep={global_ep} ✅ Score={_score_es:.4f}"
                    f"  val={_val_ret_es:.2f}% HHI={_hhi_es:.3f} std(L)={_l_std_es:.3f}"
                )
            else:
                patience_count += 1
                if patience_count % 4 == 0:
                    logger.log(
                        f"  [早停] ep={global_ep} patience={patience_count*5}/{_EARLY_STOP_PATIENCE*5}"
                        f"  Score={_score_es:.4f} best={best_score:.4f}@ep{best_score_ep}"
                    )

            if patience_count >= _EARLY_STOP_PATIENCE:
                logger.log(
                    f"\n  [早停觸發] ep={global_ep} 連續 {_EARLY_STOP_PATIENCE*5} ep 無改善，"
                    f"回滾至最佳 ep={best_score_ep}（Score={best_score:.4f}）"
                )
                episodes_done += ep + 1
                break

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

        # ── Episode 100 診斷 ───────────────────────────────────────────────────
        if (ep + 1) == 100:
            _run_ep100_diag(agent, env, val_feat, val_prices, val_volumes,
                            run_id, window, global_ep, initial_capital,
                            train_scalers, logger)

        # ── Episode 150 最終判決書 ─────────────────────────────────────────────
        if (ep + 1) == 150:
            _run_ep150_verdict(agent, env, val_feat, val_prices, val_volumes,
                               run_id, window, global_ep, initial_capital,
                               train_scalers, logger)

        # ── 每 50 episode 監控記錄 ─────────────────────────────────────────────
        if (ep + 1) % 50 == 0 and len(agent.buffer) >= 4:
            record = _collect_monitor_record(
                agent, env, global_ep, ret, avg_losses, trade_count, ep_losses_d
            )
            monitor_records.append(record)
            _log_monitor(record, logger)

    # 儲存監控記錄
    with open(monitor_log_path(run_id, window), "wb") as f:
        pickle.dump(monitor_records, f)

    # ── 回滾至最佳模型 ─────────────────────────────────────────────────────────
    if best_agent_state is not None:
        agent.actor.load_state_dict(best_agent_state["actor"])
        agent.critic.load_state_dict(best_agent_state["critic"])
        with torch.no_grad():
            agent.log_alpha.fill_(best_agent_state["log_alpha"])
        agent.alpha = agent.log_alpha.exp().item()
        if isinstance(agent, SACAgentLogitDelta) and "logit_state" in best_agent_state:
            agent._logit_state = best_agent_state["logit_state"]
        logger.log(f"  ✅ 已回滾至最佳模型 ep={best_score_ep}，Score={best_score:.4f}")
    else:
        logger.log("  ⚠ 無最佳模型記錄，使用最終模型")

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

    # 回測
    from diagnostics import diag_backtest_curve
    diag_backtest_curve(
        actor=agent.actor, scalers=train_scalers,
        feat_dict=train_feat, prices_dict=train_prices, volumes_dict=train_volumes,
        stock_ids=TRADEABLE_STOCKS, initial_capital=initial_capital,
        dates=train_dates, logger=logger, check_interval=100,
    )

    bt_train = run_backtest(
        agent.actor,
        train_feat, train_prices, train_volumes,
        TRADEABLE_STOCKS, initial_capital, [], train_dates,
    )
    bt_val = run_backtest(
        agent.actor, train_scalers,
        val_feat, val_prices, val_volumes,
        TRADEABLE_STOCKS, initial_capital, [], val_dates,
    )

    benchmark_prices_val = val_prices[BENCHMARK_STOCK]
    regime        = detect_regime(benchmark_prices_val)
    train_return  = bt_train["total_return"]
    val_return    = bt_val["total_return"]
    overfit_ratio = (train_return / val_return) if abs(val_return) > 1e-6 else float("inf")

    total_trades   = sum(r.get("trade_count", 0) for r in monitor_records)
    avg_trade_rate = total_trades / max(len(monitor_records), 1)

    summary = {
        "run_id":          run_id,
        "window":          window,
        "best_score":      round(best_score, 4),
        "best_score_ep":   best_score_ep,
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

    # Phase 1：輸出監控報告
    if hasattr(agent, "save_phase1_report") and agent._phase1_ep_records:
        import os as _os
        _report_dir  = _os.path.join(MODEL_DIR, "..", "diagnostics", "output")
        _report_path = _os.path.join(
            _report_dir, f"phase1_repair_monitoring_run{run_id}_w{window}.json"
        )
        agent.save_phase1_report(_report_path)

    save_window_model(window, run_id, agent, env, summary)
    return summary


# ─── 主流程：實驗矩陣 ─────────────────────────────────────────────────────────

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
    notes="v7：四路實驗矩陣（B/D），順序跑，各自儲存獨立模型",
)
def train_experiment_matrix(
    period:          str   = "6y",
    initial_capital: float = DEFAULT_INITIAL_CAP,
    episodes:        int   = EPISODES_PER_WINDOW,
    runs:            list  = None,
    on_episode             = None,
    seed:            int   = 42,
) -> dict:
    runs = runs or ["D"]

    valid_runs = [r for r in runs if r in RUN_CONFIGS]
    if len(valid_runs) < len(runs):
        dropped = [r for r in runs if r not in RUN_CONFIGS]
        print(f"  [警告] 以下 Run 已拋棄，自動跳過：{dropped}")
    runs = valid_runs
    if not runs:
        raise ValueError(
            f"沒有有效的 Run 可執行，RUN_CONFIGS 只包含：{list(RUN_CONFIGS.keys())}"
        )

    print("\n" + "=" * 60)
    print(f"Walk-forward 實驗矩陣（Run {'/'.join(runs)}）")
    for r in runs:
        cfg = RUN_CONFIGS[r]
        print(f"  Run {r}：{cfg['desc']}（Actor={cfg['actor']}，Reward={cfg['reward']}）")
    print("=" * 60)

    stocks = load_all_stocks(period)
    feat_dfs, prices_dict, volumes_dict, feat_names, dates = align_features(stocks)
    n_total = len(dates)
    print(f"資料範圍: {dates[0]} ~ {dates[-1]}（{n_total} 筆）")

    # 計算窗口索引
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
        print(
            f"  窗口 {w+1}: {dates[train_start]}~{dates[train_end-1]}"
            f"（訓）{dates[val_start]}~{dates[val_end-1]}（驗）"
        )

    if len(windows) < 2:
        raise ValueError(f"資料不足，需要至少 {TRAIN_DAYS + VAL_DAYS * 2} 筆")

    all_results = {}
    for run_id in runs:
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

        meta = {
            "run_id":         run_id,
            "window_results": run_results,
            "saved_at":       now_str(),
            "period":         period,
        }
        with open(wf_meta_path(run_id), "wb") as f:
            pickle.dump(meta, f)
        print(f"\n[Run {run_id}] 元資料已儲存：{wf_meta_path(run_id)}")

    _print_comparison_summary(all_results, runs)
    return sanitize({"runs": all_results})


# ─── 向後相容：單一 Run walk-forward ──────────────────────────────────────────

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
    notes="單一 Run 的 Walk-forward（向後相容），預設 Run D",
)
def train_walkforward(
    period:          str   = "6y",
    initial_capital: float = DEFAULT_INITIAL_CAP,
    episodes:        int   = EPISODES_PER_WINDOW,
    run_id:          str   = "D",
    on_episode             = None,
) -> dict:
    result = train_experiment_matrix(
        period=period, initial_capital=initial_capital,
        episodes=episodes, runs=[run_id], on_episode=on_episode,
    )
    return result["runs"].get(run_id, {})


# ─── 預測 ─────────────────────────────────────────────────────────────────────

@register(
    module="Engine",
    inputs={"period": "str", "run_id": "str"},
    outputs={"return": "dict"},
    notes="detect_regime → 選對應窗口模型 → 最新特徵推論 → 輸出各股建議倉位",
)
def predict_walkforward(period: str = "6y") -> dict:
    stocks           = load_all_stocks(period)
    benchmark_prices = stocks[BENCHMARK_STOCK]["Close"].values
    current_regime   = detect_regime(benchmark_prices)

    best_run    = None
    best_window = None
    best_val    = -float("inf")

    for rid in ["B", "D"]:
        m_path = wf_meta_path(rid)
        if not os.path.exists(m_path):
            continue
        with open(m_path, "rb") as f:
            m = pickle.load(f)
        window_results = m.get("window_results", [])
        if not window_results:
            continue

        matched    = [r for r in window_results if r.get("regime") == current_regime]
        candidates = matched if matched else window_results

        for r in candidates:
            val = r.get("val_return", -float("inf"))
            if val > best_val and load_window_model(r["window"], rid) is not None:
                best_val    = val
                best_run    = rid
                best_window = r["window"]

    if best_run is None:
        raise ValueError("找不到任何已完成的 Run 元資料，請先執行 train_experiment_matrix()")

    print(
        f"\n[預測] 選擇 Run {best_run} 窗口 {best_window}"
        f"（val_return={best_val:.2f}%，regime={current_regime}）"
    )

    payload = load_window_model(best_window, best_run)
    actor   = rebuild_actor(best_run, payload["state_dim"], payload["n_stocks"])
    actor.load_state_dict(payload["actor_state"])
    actor.to(DEVICE)
    actor.eval()

    stock_ids  = payload["stock_ids"]
    feat_dfs, _, _, _, _ = align_features(stocks)
    common_idx = feat_dfs[OBSERVABLE_STOCKS[0]].index

    latest_feats = []
    for sid in OBSERVABLE_STOCKS:
        feat = feat_dfs[sid].iloc[[-1]].values.astype(float)
        feat = np.where(np.isposinf(feat),  10.0, feat)
        feat = np.where(np.isneginf(feat), -10.0, feat)
        feat = np.where(np.isnan(feat),      0.0, feat)
        latest_feats.append(np.clip(feat, -10.0, 10.0)[0])

    latest_feats = np.concatenate(latest_feats)
    obs = np.concatenate([
        latest_feats,
        np.zeros(N_TRADEABLE),
        np.zeros(N_TRADEABLE),
        [1.0],
    ]).astype(np.float32)

    s = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
    cfg = RUN_CONFIGS[best_run]
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


# ─── 內部診斷工具 ─────────────────────────────────────────────────────────────

def _collect_monitor_record(agent, env, global_ep, ret, avg_losses, trade_count, ep_losses_d):
    if isinstance(agent, SACAgentLogitDelta):
        buf_data   = agent.buffer.sample(4)
        _obs_batch = buf_data[0]
    else:
        buf_states, _, _, _, _ = agent.buffer.sample(4)
        _obs_batch = buf_states

    logit_metrics = _monitor_logit_delta(agent, _obs_batch)

    with torch.no_grad():
        if isinstance(agent, SACAgentLogitDelta):
            l_t = torch.FloatTensor(agent._logit_state).unsqueeze(0).to(DEVICE)
            w, _, _ = agent.actor.sample(_obs_batch[:1], logit_state=l_t)
        else:
            w, _, _ = agent.actor.sample(_obs_batch[:1])
        w_stock = w[0, :env.n_tradeable].cpu().numpy()
        hhi     = float((w_stock ** 2).sum())

    recent_losses = ep_losses_d["critic_loss"][max(0, len(ep_losses_d["critic_loss"]) - 50):]
    q_loss_std    = float(np.std(recent_losses)) if recent_losses else 0.0

    return {
        "episode":     global_ep,
        "return":      round(ret, 4),
        "hhi":         round(hhi, 4),
        "alpha":       round(agent.alpha, 4),
        "c_loss_mean": round(avg_losses["critic_loss"], 4),
        "c_loss_std":  round(q_loss_std, 4),
        "trade_count": trade_count,
        **logit_metrics,
    }


def _log_monitor(record: dict, logger) -> None:
    logger.log(f"\n  [監控 ep={record['episode']}]")
    logger.log(
        f"    HHI={record['hhi']:.4f}  alpha={record['alpha']:.3f}  "
        f"c_loss_std={record['c_loss_std']:.4f}"
    )
    if record.get("delta_l2") is not None:
        logger.log(
            f"    ΔL_L2={record.get('delta_l2','N/A')}  "
            f"std(L)={record.get('l_std','N/A')}  "
            f"range(L)={record.get('l_range','N/A')}  "
            f"mean(L)={record.get('l_mean','N/A')}"
        )
    if record["hhi"] > 0.7:
        logger.log(f"    *** 警告：HHI={record['hhi']:.4f} > 0.7，疑似 one-hot collapse ***")
    l_std = float(record.get("l_std", 1.0))
    if l_std < 1.0:
        logger.log(f"    *** 警告：std(L)={l_std:.4f} < 1.0，Logit 可能正在崩潰 ***")
    if l_std > 5.0:
        logger.log(f"    *** 警告：std(L)={l_std:.4f} > 5.0，Logit 可能正在爆炸 ***")


def _quick_val_return(agent, val_feat, val_prices, val_volumes,
                      run_id, initial_capital, train_scalers) -> float:
    _val_env = make_env(val_feat, val_prices, val_volumes,
                        initial_capital, run_id=run_id, scalers=train_scalers)
    _vo = _val_env.reset()
    _vd = False
    if isinstance(agent, SACAgentLogitDelta):
        agent.reset_logit_state()
    while not _vd:
        _va = agent.act(_vo, deterministic=True)
        _vo, _, _vd = _val_env.step(_va)
    if isinstance(agent, SACAgentLogitDelta):
        agent.reset_logit_state()
    return float(_val_env.portfolio_value())


def _run_ep100_diag(agent, env, val_feat, val_prices, val_volumes,
                    run_id, window, global_ep, initial_capital, train_scalers, logger):
    if len(agent.buffer) < 4:
        return
    logger.log(f"\n{'='*60}")
    logger.log(f"  [Run {run_id} 窗口{window}] Episode 100 診斷報告")
    logger.log(f"{'='*60}")

    if isinstance(agent, SACAgentLogitDelta):
        _buf = agent.buffer.sample(4)[0]
    else:
        _buf, *_ = agent.buffer.sample(4)

    _logit = _monitor_logit_delta(agent, _buf)
    with torch.no_grad():
        if isinstance(agent, SACAgentLogitDelta):
            _l = torch.FloatTensor(agent._logit_state).unsqueeze(0).to(DEVICE)
            _w, _, _ = agent.actor.sample(_buf[:1], logit_state=_l)
        else:
            _w, _, _ = agent.actor.sample(_buf[:1])
        _hhi = float((_w[0, :env.n_tradeable].cpu().numpy() ** 2).sum())

    _val_ret = _quick_val_return(agent, val_feat, val_prices, val_volumes,
                                 run_id, initial_capital, train_scalers)
    logger.log(f"  std(L)          = {_logit.get('l_std', 'N/A')}")
    logger.log(f"  HHI             = {_hhi:.4f}"
               + ("  *** collapse ***" if _hhi > 0.7 else "  ✓"))
    logger.log(f"  val_return      = {_val_ret:.4f}"
               + ("  *** 低於 1.0 ***" if _val_ret < 1.0 else "  ✓"))
    logger.log(f"  ΔL_L2           = {_logit.get('delta_l2', 'N/A')}"
               + ("  *** > 0.1 公式失效 ***"
                  if float(_logit.get('delta_l2', 0)) > 0.1 else "  ✓"))
    logger.log(f"{'='*60}\n")


def _run_ep150_verdict(agent, env, val_feat, val_prices, val_volumes,
                       run_id, window, global_ep, initial_capital, train_scalers, logger):
    if len(agent.buffer) < 4:
        return
    logger.log(f"\n{'='*70}")
    logger.log(f"  [Run {run_id} 窗口{window}] Episode 150 最終判決書")
    logger.log(f"{'='*70}")

    if isinstance(agent, SACAgentLogitDelta):
        _buf = agent.buffer.sample(min(32, len(agent.buffer)))[0]
    else:
        _buf, *_ = agent.buffer.sample(min(32, len(agent.buffer)))

    _logit = _monitor_logit_delta(agent, _buf)

    with torch.no_grad():
        if isinstance(agent, SACAgentLogitDelta):
            _l = torch.FloatTensor(agent._logit_state).unsqueeze(0).to(DEVICE)
            _w_det, _, _ = agent.actor.sample(_buf[:1], logit_state=_l)
        else:
            _w_det, _, _ = agent.actor.sample(_buf[:1])
        _hhi = float((_w_det[0, :env.n_tradeable].cpu().numpy() ** 2).sum())

        _actions_stoc = []
        for _ in range(20):
            if isinstance(agent, SACAgentLogitDelta):
                _ws, _, _ = agent.actor.sample(_buf[:1], logit_state=_l)
            else:
                _ws, _, _ = agent.actor.sample(_buf[:1])
            _actions_stoc.append(_ws[0, :env.n_tradeable].cpu().numpy())
        _diversity = float(np.array(_actions_stoc).std(axis=0).mean())

        if isinstance(agent, SACAgentLogitDelta):
            _l_batch = torch.zeros(_buf.shape[0], N_ACTIONS, device=DEVICE)
            _, _, _lp = agent.actor.sample(_buf, logit_state=_l_batch)
        else:
            _, _lp, _ = agent.actor.sample(_buf)
        _lp_mean = float(_lp.mean().item())
        _lp_std  = float(_lp.std().item())

    _val_ret = _quick_val_return(agent, val_feat, val_prices, val_volumes,
                                 run_id, initial_capital, train_scalers)

    _hhi_ok = 0.3 <= _hhi <= 0.6
    _div_ok = _diversity > 0.02
    _val_ok = _val_ret > 1.0

    logger.log(f"  {'項目':<18}  {'數值':>12}  {'判決'}")
    logger.log(f"  {'-'*55}")
    logger.log(f"  {'alpha':<18}  {agent.alpha:>12.4f}  "
               + ("⚠ 單調下降" if agent.alpha < 0.5 else "✓"))
    logger.log(f"  {'log_pi_mean':<18}  {_lp_mean:>12.4f}  "
               + ("⚠ Q值壓制" if _lp_mean < -5.0 else "✓"))
    logger.log(f"  {'log_pi_std':<18}  {_lp_std:>12.4f}")
    logger.log(f"  {'diversity':<18}  {_diversity:>12.4f}  "
               + ("✓" if _div_ok else "⚠ 探索不足 < 0.02"))
    logger.log(f"  {'HHI':<18}  {_hhi:>12.4f}  "
               + ("✓ 0.3~0.6" if _hhi_ok else
                  "⚠ collapse" if _hhi > 0.6 else "⚠ 過度分散"))
    logger.log(f"  {'std(L)':<18}  {_logit.get('l_std','N/A'):>12}  "
               + ("⚠ 爆炸" if float(_logit.get('l_std', 0)) > 5.0
                  else "⚠ 崩潰" if float(_logit.get('l_std', 1)) < 0.1 else "✓"))
    logger.log(f"  {'validation_return':<18}  {_val_ret:>12.4f}  "
               + ("✓ 正報酬" if _val_ok else "⚠ 未獲利"))
    logger.log(f"  {'-'*55}")

    _pass = sum([_hhi_ok, _div_ok, _val_ok])
    if _pass == 3:
        logger.log("  ✅ 三項通過：RL 架構成功，SAC-Stock-v7 正式定版")
    elif _pass == 2 and not _val_ok:
        logger.log("  ⚠️  僅結構通過，無獲利：RL 任務結束，轉向特徵工程")
    elif not _hhi_ok:
        logger.log("  ❌ 第一項未過：仍有強主導訊號，凍結 RL，啟動特徵清查")
    else:
        logger.log(f"  ⚠️  通過 {_pass}/3 項，繼續觀察")
    logger.log(f"{'='*70}\n")


def _print_comparison_summary(all_results: dict, runs: list) -> None:
    print("\n" + "=" * 70)
    print("實驗矩陣結果比較")
    print("=" * 70)
    print(
        f"  {'Run':<6} {'Actor':<12} {'Reward':<12} "
        f"{'訓練報酬':>10} {'驗證報酬':>10} {'換倉率':>8} {'過擬合':>8}"
    )
    print("  " + "-" * 66)
    for run_id in runs:
        if run_id not in all_results:
            continue
        cfg       = RUN_CONFIGS[run_id]
        results   = all_results[run_id]
        avg_train = np.mean([r["train_return"] for r in results])
        avg_val   = np.mean([r["val_return"]   for r in results])
        avg_trade = np.mean([r.get("avg_trade_rate", 0) for r in results])
        avg_of    = np.mean([abs(r.get("overfit_ratio", 0)) for r in results])
        print(
            f"  {run_id:<6} {cfg['actor']:<12} {cfg['reward']:<12} "
            f"{avg_train:>9.1f}% {avg_val:>9.1f}% "
            f"{avg_trade:>7.0f} {avg_of:>7.1f}x"
        )
    print("=" * 70)
    print("\n升格 v8 判准：")
    print("  Run D turnover 比 Run A 降 ≥35%")
    print("  且 Q loss 走勢平穩（std 不爆）")
    print("  且 Sharpe 不低於 Run A")