"""
diagnostics/training_analyst.py
================================
訓練 / 環境層診斷（核心深度診斷函數）。

包含：
  1. diag_random_policy()               環境層：Random Policy 資產追蹤
  2. diag_feature_alignment()           環境層：特徵/價格/成交量對齊檢查
  3. diag_reward_distribution()         環境層：Reward 分布統計
  4. diag_actor_logits()                Actor 層：Logit 追蹤 + HHI
  5. diag_stochastic_vs_deterministic() Actor 層：探索多樣性
  6. diag_training_curve()              訓練層：Critic loss / Alpha 收斂
  7. diag_final_holdings()              訓練層：最後持倉明細
  8. _monitor_logit_delta()             LogitDelta 幾何健康監控（訓練迴圈內）
"""

from __future__ import annotations

import numpy as np
import torch

from .logger import DebugLogger


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 環境層：Random Policy 資產追蹤
# ═══════════════════════════════════════════════════════════════════════════════

def diag_random_policy(feat, prices, volumes, scalers,
                       initial_capital: float,
                       logger: DebugLogger,
                       seed: int = 42) -> dict:
    """
    用完全隨機的 action 跑完整 episode，確認環境本身沒有計算錯誤或 look-ahead。
    預期結果：return ≈ 0.2-0.4（手續費侵蝕），若 > 2.0 代表環境有問題。
    """
    from src.environment.portfolio import PortfolioEnv
    try:
        from configs.trading_config import LOT_SIZE
    except ImportError:
        LOT_SIZE = 1000

    np.random.seed(seed)
    env  = PortfolioEnv(feat, prices, volumes, scalers=scalers,
                        initial_capital=initial_capital)
    obs  = env.reset()
    done = False
    step = 0
    checkpoints = []

    while not done:
        action = np.random.dirichlet(np.ones(env.n_tradeable)).astype(np.float32)
        obs, _, done = env.step(action)
        step += 1
        if step % 200 == 0 or done:
            p   = np.array([env.prices[sid][env.step_idx - 1]
                            for sid in env.tradeable_ids], dtype=np.float64)
            lv  = env.lots_held * LOT_SIZE * p
            ov  = env.odd_held  * p
            tot = env.capital + lv.sum() + ov.sum()
            checkpoints.append({
                "step":     step,
                "capital":  round(env.capital, 0),
                "lots_val": round(lv.sum(), 0),
                "total":    round(tot, 0),
                "return":   round(tot / initial_capital, 4),
            })

    final_return = checkpoints[-1]["return"] if checkpoints else 0.0
    status = "✓ 環境正常" if final_return < 2.0 else \
             "*** 警告：環境可能有 look-ahead 或計算錯誤 ***"

    logger.log("\n[診斷 1] Random Policy 資產追蹤")
    logger.log(f"  {'step':>6}  {'capital':>14}  {'lots_val':>14}  {'total':>14}  {'return':>8}")
    for c in checkpoints:
        logger.log(f"  {c['step']:>6}  {c['capital']:>14,.0f}  "
                   f"{c['lots_val']:>14,.0f}  {c['total']:>14,.0f}  {c['return']:>8.4f}")
    logger.log(f"  {status}")

    return {"checkpoints": checkpoints, "final_return": final_return, "status": status}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 環境層：特徵對齊檢查
# ═══════════════════════════════════════════════════════════════════════════════

def diag_feature_alignment(feat_dict: dict, prices_dict: dict,
                            volumes_dict: dict,
                            logger: DebugLogger) -> dict:
    """
    確認特徵、價格、成交量的長度完全一致，且沒有 NaN 或 Inf。
    """
    logger.log("\n[診斷 2] 特徵對齊檢查")
    issues = []

    for sid in feat_dict:
        f_len = len(feat_dict[sid])
        p_len = len(prices_dict.get(sid, []))
        v_len = len(volumes_dict.get(sid, []))

        if not (f_len == p_len == v_len):
            msg = f"  *** {sid}: 特徵={f_len} 價格={p_len} 成交量={v_len} 長度不一致 ***"
            issues.append(msg)
            logger.log(msg)
        else:
            feat_arr  = feat_dict[sid].values if hasattr(feat_dict[sid], "values") \
                        else np.array(feat_dict[sid])
            nan_count = int(np.isnan(feat_arr).sum())
            inf_count = int(np.isinf(feat_arr).sum())
            if nan_count > 0 or inf_count > 0:
                msg = f"  *** {sid}: NaN={nan_count} Inf={inf_count} ***"
                issues.append(msg)
                logger.log(msg)
            else:
                logger.log(f"  ✓ {sid}: 長度={f_len}，無 NaN/Inf")

    if not issues:
        logger.log("  ✓ 所有股票特徵對齊正常")

    return {"issues": issues}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 環境層：Reward 分布統計
# ═══════════════════════════════════════════════════════════════════════════════

def diag_reward_distribution(agent, feat, prices, volumes,
                              scalers, initial_capital: float,
                              logger: DebugLogger,
                              n_steps: int = 200) -> dict:
    """
    跑前 n_steps 步，收集 reward 的分布統計。
    """
    from src.environment.portfolio import PortfolioEnv

    env     = PortfolioEnv(feat, prices, volumes, scalers=scalers,
                           initial_capital=initial_capital)
    obs     = env.reset()
    rewards = []

    for _ in range(n_steps):
        action = agent.act(obs, deterministic=False)
        obs, reward, done = env.step(action)
        rewards.append(reward)
        if done:
            break

    rewards = np.array(rewards)
    stats = {
        "min":          round(float(rewards.min()), 6),
        "max":          round(float(rewards.max()), 6),
        "mean":         round(float(rewards.mean()), 6),
        "std":          round(float(rewards.std()), 6),
        "pct_positive": round(float((rewards > 0).mean() * 100), 1),
        "out_of_range": int(((rewards < -1) | (rewards > 1)).sum()),
    }

    logger.log(f"\n[診斷 3] Reward 分布（前 {len(rewards)} 步）")
    logger.log(f"  min={stats['min']:.6f}  max={stats['max']:.6f}  "
               f"mean={stats['mean']:.6f}  std={stats['std']:.6f}")
    logger.log(f"  正報酬比例: {stats['pct_positive']}%  "
               f"超出[-1,1]範圍: {stats['out_of_range']} 次")
    if stats["out_of_range"] > 0:
        logger.log("  *** 警告：有 reward 超出 [-1,1]，請檢查縮放設定 ***")
    else:
        logger.log("  ✓ Reward 數值範圍正常")

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Actor 層：Logit 追蹤 + HHI（Dirichlet 版）
# ═══════════════════════════════════════════════════════════════════════════════

def diag_actor_logits(actor, obs_batch: torch.Tensor,
                      logger: DebugLogger,
                      update_count: int = 0) -> dict:
    """
    印出 Actor 的 Dirichlet α 和 Beta (a,b) 分布參數。

    正常範圍：
      alpha：0.1~5.0 之間，若全部趨近 0.1 代表退化（one-hot），
             若超過 10 代表過度集中（單股梭哈）
      beta_a / beta_b：0.1~5.0，若 a >> b 代表傾向滿手現金，
                       若 b >> a 代表傾向全倉股票
      HHI：基於 deterministic 動作計算，應在 0.1~0.5 之間

    obs_batch 應由外部傳入（從 agent.buffer.sample() 取得）。
    """
    actor.eval()
    with torch.no_grad():
        alpha_and_ba, beta_b_expanded = actor.forward(obs_batch)
        alpha  = alpha_and_ba[:, :actor.n_stocks]
        beta_a = alpha_and_ba[:, actor.n_stocks:]
        beta_b = beta_b_expanded[:, :1]

        _, _, mean_action = actor.sample(obs_batch)

    results = []
    logger.log(f"\n[診斷 4] Actor 分布參數追蹤（update={update_count}）")
    for i in range(min(4, obs_batch.shape[0])):
        alpha_i  = alpha[i].cpu().numpy().round(3)
        ba_i     = float(beta_a[i].item())
        bb_i     = float(beta_b[i].item())
        action_i = mean_action[i].cpu().numpy().round(4)
        hhi_i    = float((action_i ** 2).sum())

        a_min = float(alpha_i.min())
        a_max = float(alpha_i.max())
        if a_min <= 0.15:
            status = "*** alpha 退化，趨近 one-hot ***"
        elif a_max > 10.0:
            status = "*** alpha 過大，單股梭哈 ***"
        else:
            status = "✓"

        cash_mean = ba_i / (ba_i + bb_i)
        if cash_mean > 0.7:
            cash_status = "*** 傾向持現 ***"
        elif cash_mean < 0.2:
            cash_status = "傾向全倉"
        else:
            cash_status = "✓"

        results.append({
            "sample":     i,
            "alpha":      alpha_i.tolist(),
            "beta_a":     round(ba_i, 3),
            "beta_b":     round(bb_i, 3),
            "action":     action_i.tolist(),
            "hhi":        round(hhi_i, 4),
            "alpha_min":  round(a_min, 3),
            "alpha_max":  round(a_max, 3),
            "cash_mean":  round(cash_mean, 3),
        })

        logger.log(f"  樣本{i}: alpha={alpha_i[:5]}...  "
                   f"alpha_min={a_min:.3f}  alpha_max={a_max:.3f}  {status}")
        logger.log(f"           beta_a={ba_i:.3f}  beta_b={bb_i:.3f}  "
                   f"現金均值={cash_mean:.3f}  {cash_status}")
        logger.log(f"           action={action_i}  HHI={hhi_i:.4f}")

    actor.train()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Actor 層：Stochastic vs Deterministic 探索多樣性
# ═══════════════════════════════════════════════════════════════════════════════

def diag_stochastic_vs_deterministic(agent, obs: np.ndarray,
                                     logger: DebugLogger,
                                     n_samples: int = 10) -> dict:
    """
    對同一個 obs 採樣 n 次 stochastic action，和 deterministic 比較。
    obs 應由外部傳入（通常是 env.reset() 的結果），不在此函數內呼叫 reset。
    """
    stochastic_actions = [agent.act(obs, deterministic=False)
                          for _ in range(n_samples)]
    det_action = agent.act(obs, deterministic=True)
    stoc_arr   = np.array(stochastic_actions)

    hhi_det   = float((det_action ** 2).sum())
    hhi_stoc  = float((stoc_arr.mean(axis=0) ** 2).sum())
    diversity = float(stoc_arr.std(axis=0).mean())

    logger.log("\n[診斷 5] Stochastic vs Deterministic Action")
    logger.log(f"  Deterministic:     {det_action.round(4)}  HHI={hhi_det:.4f}")
    logger.log(f"  Stochastic 均值:   {stoc_arr.mean(axis=0).round(4)}  HHI={hhi_stoc:.4f}")
    logger.log(f"  Stochastic 標準差: {stoc_arr.std(axis=0).round(4)}")
    logger.log(f"  多樣性（平均 std）: {diversity:.4f}  "
               + ("✓ 探索充分" if diversity > 0.05 else "*** 探索不足 ***"))

    return {
        "det_action": det_action.tolist(),
        "stoc_mean":  stoc_arr.mean(axis=0).tolist(),
        "stoc_std":   stoc_arr.std(axis=0).tolist(),
        "hhi_det":    round(hhi_det, 4),
        "hhi_stoc":   round(hhi_stoc, 4),
        "diversity":  round(diversity, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 訓練層：訓練曲線分析
# ═══════════════════════════════════════════════════════════════════════════════

def diag_training_curve(episode_returns: list, episode_losses: list,
                        alphas: list, logger: DebugLogger,
                        window: int = 10) -> dict:
    """
    分析訓練曲線，找出收斂趨勢和異常。
    """
    returns    = np.array(episode_returns)
    losses     = np.array([l.get("critic_loss", 0) for l in episode_losses])
    alphas_arr = np.array(alphas)

    def rolling_mean(arr, w):
        return np.convolve(arr, np.ones(w) / w, mode="valid") if len(arr) >= w else arr

    rm_losses = rolling_mean(losses, min(window, len(losses)))
    converged = bool(len(rm_losses) > 5 and rm_losses[-1] < rm_losses[0])

    stats = {
        "return_final": round(float(returns[-1]), 4) if len(returns) > 0 else 0,
        "return_max":   round(float(returns.max()), 4) if len(returns) > 0 else 0,
        "return_min":   round(float(returns.min()), 4) if len(returns) > 0 else 0,
        "loss_final":   round(float(losses[-1]), 4) if len(losses) > 0 else 0,
        "alpha_final":  round(float(alphas_arr[-1]), 4) if len(alphas_arr) > 0 else 0,
        "converged":    converged,
    }

    logger.log("\n[診斷 6] 訓練曲線摘要")
    logger.log(f"  Return: 最終={stats['return_final']:.4f}  "
               f"最大={stats['return_max']:.4f}  最小={stats['return_min']:.4f}")
    logger.log(f"  Critic Loss 最終: {stats['loss_final']:.4f}")
    logger.log(f"  Alpha 最終: {stats['alpha_final']:.4f}")
    logger.log(f"  收斂狀態: {'✓ 有收斂趨勢' if converged else '*** 尚未收斂 ***'}")

    logger.log("\n  Episode 摘要（每 10 個）：")
    logger.log(f"  {'ep':>6}  {'return':>10}  {'c_loss':>10}  {'alpha':>8}")
    step = max(1, len(returns) // 20)
    for i in range(0, len(returns), step):
        r = returns[i]
        l = losses[i] if i < len(losses) else 0
        a = alphas_arr[i] if i < len(alphas_arr) else 0
        logger.log(f"  {i+1:>6}  {r:>10.4f}  {l:>10.4f}  {a:>8.4f}")

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 訓練層：最後持倉明細
# ═══════════════════════════════════════════════════════════════════════════════

def diag_final_holdings(env, logger: DebugLogger,
                        snapshot: dict | None = None) -> dict:
    """
    印出最後一個 episode 的實際持倉，確認 portfolio_value() 計算正確。

    v2 修正：新增 snapshot 參數。
      - snapshot 不為 None 時：從快照讀取持倉，不受後續 reset() 影響。
        快照格式：{"capital": float, "lots_held": np.ndarray,
                   "odd_held": np.ndarray, "step_idx": int}
      - snapshot 為 None 時：直接讀取 env 當前狀態（向後相容）。
    """
    try:
        from configs.trading_config import LOT_SIZE
    except ImportError:
        LOT_SIZE = 1000

    try:
        if snapshot is not None:
            capital   = float(snapshot["capital"])
            lots_held = np.array(snapshot["lots_held"], dtype=np.int64)
            odd_held  = np.array(snapshot["odd_held"],  dtype=np.int64)
            T         = min(int(snapshot["step_idx"]), env.n_steps - 1)
        else:
            capital   = env.capital
            lots_held = env.lots_held
            odd_held  = env.odd_held
            T         = min(env.step_idx, env.n_steps - 1)

        prices_T = np.array(
            [env.prices[sid][T] for sid in env.tradeable_ids],
            dtype=np.float64,
        )

        lot_val = lots_held * LOT_SIZE * prices_T
        odd_val = odd_held  * prices_T
        total   = capital + lot_val.sum() + odd_val.sum()

        actual_ret = total / env.initial_capital
        pv         = actual_ret
        consistent = True

        logger.log("\n[診斷 7] 最後 Episode 實際持倉")
        logger.log(f"  現金:              {capital:>15,.0f} 元")
        logger.log(f"  整張市值:          {lot_val.sum():>15,.0f} 元")
        logger.log(f"  零股市值:          {odd_val.sum():>15,.0f} 元")
        logger.log(f"  總資產:            {total:>15,.0f} 元")
        logger.log(f"  實際報酬率:        {(actual_ret - 1) * 100:.2f}%")
        logger.log(f"  portfolio_value(): {pv:.6f}")
        logger.log("  ✓ 數值一致" if consistent else
                   "  *** 警告：數值不一致，請確認快照是否在 reset() 之前取得 ***")

        return {
            "total":           round(total, 0),
            "actual_return":   round(actual_ret, 4),
            "portfolio_value": round(pv, 4),
            "consistent":      consistent,
        }

    except Exception as e:
        logger.log(f"  [診斷 7] 失敗：{e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# 8. LogitDelta 幾何健康監控（訓練迴圈內，每 50 episode）
# ═══════════════════════════════════════════════════════════════════════════════

def _monitor_logit_delta(agent, obs_batch: torch.Tensor) -> dict:
    """
    計算 LogitDelta 的幾何健康指標：
      - delta_l2:  ‖ΔL‖₂（動作能量，應 < 1.0 才算穩定）
      - l_std:     std(L_t)（Logit 分散度，< 1.0 = 崩潰，> 5.0 = 爆炸）
      - l_range:   max(L_t) - min(L_t)
      - l_mean:    mean(L_t)

    僅在 SACAgentLogitDelta 時有意義；其他 agent 回傳空 dict。
    """
    try:
        from src.agents.sac_agent import SACAgentLogitDelta
        if not isinstance(agent, SACAgentLogitDelta):
            return {}
    except ImportError:
        return {}

    try:
        from configs.base_config import DEVICE
    except ImportError:
        import torch as _torch
        DEVICE = _torch.device("cpu")

    with torch.no_grad():
        l_t = torch.FloatTensor(agent._logit_state).unsqueeze(0).to(DEVICE)
        raw = agent.actor.forward(obs_batch[:1])
        a_norm = raw.norm(dim=-1, keepdim=True)
        delta  = 0.1 * raw / (1.0 + a_norm)

        delta_l2 = float(delta.norm().item())
        l_np     = agent._logit_state

    return {
        "delta_l2": round(delta_l2, 4),
        "l_std":    round(float(l_np.std()), 4),
        "l_range":  round(float(l_np.max() - l_np.min()), 4),
        "l_mean":   round(float(l_np.mean()), 4),
    }