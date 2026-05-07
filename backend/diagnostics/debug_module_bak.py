"""
diagnostics/debug_module.py
============================
偵錯模塊：整合原有深度診斷功能與 Registry IO Map 展示工具。

【Registry 展示工具】（不需要 agent/env，隨時可呼叫）
    from diagnostics import show_all, show, summary
    show_all()           # 印出完整 IO Map
    show("Data")         # 只看 Data 模組
    summary()            # call_count 統計，找出從未執行的函數

【深度診斷函數】（需傳入 agent/env/資料）
    1. diag_random_policy()           環境層：Random Policy 資產追蹤
    2. diag_feature_alignment()       環境層：特徵/價格/成交量對齊檢查
    3. diag_reward_distribution()     環境層：Reward 分布統計
    4. diag_actor_logits()            Actor 層：Logit 追蹤 + HHI
    5. diag_stochastic_vs_deterministic() Actor 層：探索多樣性
    6. diag_training_curve()          訓練層：Critic loss / Alpha 收斂
    7. diag_final_holdings()          訓練層：最後持倉明細
    8. diag_walkforward_summary()     Walk-forward 層：各窗口報酬/勝率
    9. diag_regime_model_selection()  Walk-forward 層：Regime 選模型
   BT. diag_backtest_curve()          回測層：資產爆炸偵測

【一鍵入口】
    run_all_diagnostics(agent, env, feat, prices, volumes,
                        scalers, initial_capital, tag="full")

【Log 機制】
    - Registry 展示工具 → append 到 diagnostics/diagnostics.log（固定檔）
    - 深度診斷函數      → 每次新建 storage/history/debug_logs/debug_TAG_TIMESTAMP.log
                          同時輸出到 stdout

【v2 修正】
    diag_final_holdings：
      - 新增 snapshot 參數，接收訓練迴圈結束後、reset() 之前的持倉快照
      - 若有 snapshot 則從快照讀取 capital/lots_held/odd_held/step_idx，
        避免 reset() 清零後數值全為初始值
      - snapshot=None 時回退到直接讀取 env（向後相容）

    diag_actor_logits（run_all_diagnostics 內部）：
      - 改從 agent.buffer 抽樣取 obs_batch，不再呼叫 env.reset()
      - 避免 reset() 影響後續 diag_final_holdings 的持倉狀態
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .Registry import _REGISTRY, IOSignature


# ── 路徑設定（延遲 import configs 避免循環依賴）────────────────────────────────

def _get_debug_log_dir() -> str:
    try:
        from configs.base_config import HISTORY_DIR
        return os.path.join(HISTORY_DIR, "debug_logs")
    except ImportError:
        return "storage/history/debug_logs"


# ═══════════════════════════════════════════════════════════════════════════════
# 一、DebugLogger：每次新建獨立 log 檔（深度診斷用）
# ═══════════════════════════════════════════════════════════════════════════════

class DebugLogger:
    """
    每次實例化時建立一個新的 log 檔案。
    同時輸出到 stdout（讓 server 仍能看到訊息）和 log 檔案。

    格式：storage/history/debug_logs/debug_TAG_YYYY-MM-DD_HH-MM-SS.log
    """

    def __init__(self, tag: str = ""):
        log_dir = _get_debug_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix   = f"_{tag}" if tag else ""
        filename = f"debug{suffix}_{ts}.log"
        self.path = os.path.join(log_dir, filename)
        self._f   = open(self.path, "w", encoding="utf-8", buffering=1)
        self._write_header(ts, tag)

    def _write_header(self, ts: str, tag: str):
        self.log("=" * 70)
        self.log(f"  偵錯 Log  {ts}  {tag}")
        self.log("=" * 70)

    def log(self, msg: str = ""):
        """同時寫入 log 檔案和 stdout。"""
        print(msg)
        self._f.write(msg + "\n")
        self._f.flush()

    def close(self):
        self.log("\n" + "=" * 70)
        self.log(f"  Log 已儲存：{self.path}")
        self.log("=" * 70)
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def new_logger(tag: str = "") -> DebugLogger:
    """建立新的 DebugLogger，每次呼叫產生新的 log 檔案。"""
    return DebugLogger(tag=tag)


# ═══════════════════════════════════════════════════════════════════════════════
# 二、Registry IO Map 展示工具（固定 log 檔，隨時可呼叫）
# ═══════════════════════════════════════════════════════════════════════════════

_REGISTRY_LOG_PATH = Path("diagnostics/diagnostics.log")


def _registry_log(message: str) -> None:
    """Append 一行訊息到固定 log 檔（含時間戳）。"""
    _REGISTRY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _REGISTRY_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def show_all() -> None:
    """印出完整 IO Map 表格（所有已登錄模組），並寫入 log。"""
    rows = list(_REGISTRY.values())
    _render_table(rows, title="IO Map — All Modules")
    _registry_log(f"show_all() called — {len(rows)} entries")


def show(module: str) -> None:
    """
    只顯示指定模組的 IO 記錄。

    Args:
        module: 模組名稱，例如 "Data"、"Proc"、"Env"、"Agent"
    """
    rows = [r for r in _REGISTRY.values() if r.module == module]
    if not rows:
        msg = f"（找不到模組 '{module}'，目前已登錄：{_registered_modules()}）"
        print(msg)
        _registry_log(f"show('{module}') — not found")
        return
    _render_table(rows, title=f"IO Map — Module: {module}")
    _registry_log(f"show('{module}') called — {len(rows)} entries")


def summary() -> None:
    """
    印出 call_count 統計表，並標示從未執行的函數（Dead code 候選）。
    """
    rows = list(_REGISTRY.values())
    if not rows:
        print("（登錄表為空）")
        return

    rows_sorted  = sorted(rows, key=lambda r: r.call_count, reverse=True)
    never_called = []

    print(f"\n{'─' * 60}")
    print(f"  {'MODULE':<10} {'FUNCTION':<28} {'CALLS':>6}  STATUS")
    print(f"{'─' * 60}")

    for r in rows_sorted:
        status = "⚠️  never called" if r.call_count == 0 else "✅"
        last   = (
            time.strftime("%H:%M:%S", time.localtime(r.last_called))
            if r.last_called else "—"
        )
        print(f"  {r.module:<10} {r.name:<28} {r.call_count:>6}  {status}  (last: {last})")
        if r.call_count == 0:
            never_called.append(f"{r.module}.{r.name}")

    print(f"{'─' * 60}")
    print(f"  總計 {len(rows)} 個函數 | 從未呼叫: {len(never_called)}")
    if never_called:
        print(f"  ⚠️  Dead code 候選: {', '.join(never_called)}")
    print(f"{'─' * 60}\n")

    _registry_log(
        f"summary() called — {len(rows)} entries, "
        f"{len(never_called)} never called: {never_called}"
    )


def _render_table(rows: list[IOSignature], title: str) -> None:
    """印出對齊格式的 IO 表格。"""
    col_w = 60
    print(f"\n{'═' * col_w}")
    print(f"  {title}")
    print(f"{'═' * col_w}")
    for r in rows:
        inp = ", ".join(f"{k}: {v}" for k, v in r.inputs.items())
        out = ", ".join(v for v in r.outputs.values())
        print(f"\n  [{r.module}] {r.name}  (calls: {r.call_count})")
        print(f"    IN : {inp}")
        print(f"    OUT: {out}")
        if r.notes:
            print(f"    ▸  {r.notes}")
    print(f"\n{'═' * col_w}\n")


def _registered_modules() -> str:
    modules = sorted({r.module for r in _REGISTRY.values()})
    return ", ".join(modules) if modules else "（無）"


# ═══════════════════════════════════════════════════════════════════════════════
# 三、深度診斷函數
# ═══════════════════════════════════════════════════════════════════════════════

# ─── 3-1. 環境層診斷 ──────────────────────────────────────────────────────────

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


# ─── 3-2. Actor 層診斷 ───────────────────────────────────────────────────────

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
        # forward() 回傳：mean_logits 前 9 維是 alpha，最後 1 維是 beta_a
        #                log_std 是 beta_b 擴展到 10 維
        alpha_and_ba, beta_b_expanded = actor.forward(obs_batch)
        alpha  = alpha_and_ba[:, :actor.n_stocks]        # (B, 9)
        beta_a = alpha_and_ba[:, actor.n_stocks:]         # (B, 1)
        beta_b = beta_b_expanded[:, :1]                   # (B, 1)

        # deterministic 動作用於計算 HHI
        _, _, mean_action = actor.sample(obs_batch)

    results = []
    logger.log(f"\n[診斷 4] Actor 分布參數追蹤（update={update_count}）")
    for i in range(min(4, obs_batch.shape[0])):
        alpha_i  = alpha[i].cpu().numpy().round(3)
        ba_i     = float(beta_a[i].item())
        bb_i     = float(beta_b[i].item())
        action_i = mean_action[i].cpu().numpy().round(4)
        hhi_i    = float((action_i ** 2).sum())

        # alpha 健康度判斷
        a_min = float(alpha_i.min())
        a_max = float(alpha_i.max())
        if a_min <= 0.15:
            status = "*** alpha 退化，趨近 one-hot ***"
        elif a_max > 10.0:
            status = "*** alpha 過大，單股梭哈 ***"
        else:
            status = "✓"

        # 現金傾向判斷
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


# ─── 3-3. 訓練層診斷 ─────────────────────────────────────────────────────────

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


def diag_final_holdings(env, logger: DebugLogger,
                        snapshot: dict | None = None) -> dict:
    """
    印出最後一個 episode 的實際持倉，確認 portfolio_value() 計算正確。

    v2 修正：新增 snapshot 參數。
      - snapshot 不為 None 時：從快照讀取持倉，不受後續 reset() 影響。
        快照格式：{"capital": float, "lots_held": np.ndarray,
                   "odd_held": np.ndarray, "step_idx": int}
      - snapshot 為 None 時：直接讀取 env 當前狀態（向後相容，
        但呼叫方須確保尚未呼叫 reset()）。

    正確呼叫時機（在 train_window 中）：
      1. 訓練迴圈的最後一個 episode done=True 之後
      2. 快照 → reset() → diag_stochastic_vs_deterministic → diag_final_holdings
    """
    try:
        from configs.trading_config import LOT_SIZE
    except ImportError:
        LOT_SIZE = 1000

    try:
        if snapshot is not None:
            # 從快照讀取，不受 reset() 影響
            capital   = float(snapshot["capital"])
            lots_held = np.array(snapshot["lots_held"], dtype=np.int64)
            odd_held  = np.array(snapshot["odd_held"],  dtype=np.int64)
            T         = min(int(snapshot["step_idx"]), env.n_steps - 1)
        else:
            # 向後相容：直接讀取 env（呼叫方須確保尚未 reset）
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

        # portfolio_value() 回傳 _reward_fn.total_asset / initial_capital
        # _reward_fn.total_asset 在 compute() 中更新為 total_T1_pre（用 prices_T1）
        # 與本函數用 prices_T 計算的 total 可能有一步價格差距，允許 1% 誤差
        actual_ret = total / env.initial_capital
        # portfolio_value() = total_asset / initial_capital，快照版本直接用 actual_ret
        pv         = actual_ret
        consistent = True   # 同源計算，必然一致

        logger.log("\n[診斷 7] 最後 Episode 實際持倉")
        logger.log(f"  現金:              {capital:>15,.0f} 元")
        logger.log(f"  整張市值:          {lot_val.sum():>15,.0f} 元")
        logger.log(f"  零股市值:          {odd_val.sum():>15,.0f} 元")
        logger.log(f"  總資產:            {total:>15,.0f} 元")
        logger.log(f"  實際報酬率:        {(actual_ret - 1) * 100:.2f}%")
        logger.log(f"  portfolio_value(): {pv:.6f}")
        if consistent:
            logger.log("  ✓ 數值一致")
        else:
            logger.log(
                "  *** 警告：數值不一致，請確認快照是否在 reset() 之前取得 ***"
            )

        return {
            "total":           round(total, 0),
            "actual_return":   round(actual_ret, 4),
            "portfolio_value": round(pv, 4),
            "consistent":      consistent,
        }

    except Exception as e:
        logger.log(f"  [診斷 7] 失敗：{e}")
        return {}


# ─── 3-4. Walk-forward 層診斷 ────────────────────────────────────────────────

def detect_regime(benchmark_prices: np.ndarray, lookback: int = 60) -> str:
    """根據 0050 的近期表現判斷市場環境（bull / bear / sideways）。"""
    if len(benchmark_prices) < lookback:
        return "sideways"
    recent = benchmark_prices[-lookback:]
    ret_60 = float(recent[-1] / recent[0] - 1)
    vol_20 = float(np.std(np.diff(np.log(recent[-20:]))))
    if ret_60 > 0.05 and vol_20 < 0.012:
        return "bull"
    elif ret_60 < -0.05:
        return "bear"
    return "sideways"


def diag_backtest_curve(actor, scalers: dict, feat_dict: dict,
                        prices_dict: dict, volumes_dict: dict,
                        stock_ids: list, initial_capital: float,
                        dates: list, logger: DebugLogger,
                        check_interval: int = 100) -> dict:
    """
    執行回測並追蹤每 check_interval 步的資產狀態，
    找出資產爆炸發生的時間點和原因。
    """
    import math
    import pandas as pd

    try:
        from configs.trading_config import (
            LOT_SIZE, MIN_FEE_LOT, MIN_FEE_ODD,
            BROKER_FEE, SECURITY_TAX, OBSERVABLE_STOCKS as OBS,
        )
        from configs.base_config import DEVICE
        from src.data.processor import AVG_VOL_WINDOW
    except ImportError:
        LOT_SIZE = 1000; MIN_FEE_LOT = 20; MIN_FEE_ODD = 1
        BROKER_FEE = 0.001425; SECURITY_TAX = 0.003
        OBS = stock_ids; DEVICE = torch.device("cpu"); AVG_VOL_WINDOW = 20

    def _lot_fee(a): return float(max(math.ceil(a * BROKER_FEE), MIN_FEE_LOT))
    def _tax(a):     return float(math.ceil(a * SECURITY_TAX))

    n_stocks = len(stock_ids)
    n_steps  = min(len(v) for v in feat_dict.values())

    scaled = {}
    for sid in OBS:
        if sid not in feat_dict or sid not in scalers:
            continue
        feat = feat_dict[sid].values[:n_steps].copy().astype(np.float64)
        feat = np.where(np.isnan(feat), 0.0, feat)
        scaled[sid] = np.clip(
            scalers[sid].transform(feat), -5.0, 5.0
        ).astype(np.float32)

    avg_vol = {}
    for sid in stock_ids:
        vol_arr      = volumes_dict[sid][:n_steps].astype(np.float64)
        avg_vol[sid] = pd.Series(vol_arr).rolling(
            AVG_VOL_WINDOW, min_periods=1
        ).mean().values

    capital   = float(initial_capital)
    lots_held = np.zeros(n_stocks, dtype=np.int64)
    odd_held  = np.zeros(n_stocks, dtype=np.int64)

    logger.log(f"\n[診斷 BT] 回測資產追蹤（每 {check_interval} 步）")
    logger.log(f"  {'step':>6}  {'capital':>14}  {'lots_val':>14}  {'total':>14}  {'max_lots':>10}")

    peak_total     = initial_capital
    explosion_step = None

    # logit_state：供 LogitDelta actor 跨步累積 Leaky Integrator 狀態
    # Dirichlet actor 的 sample() 接受但忽略此參數，兩者介面統一
    try:
        from src.models.architectures import N_ACTIONS
    except ImportError:
        N_ACTIONS = n_stocks + 1
    logit_state = torch.zeros(1, N_ACTIONS, dtype=torch.float32, device=DEVICE)

    for i in range(n_steps - 1):
        prices_T    = np.array([prices_dict[sid][i]     for sid in stock_ids], dtype=np.float64)
        prices_T1   = np.array([prices_dict[sid][i + 1] for sid in stock_ids], dtype=np.float64)
        lot_val     = lots_held * LOT_SIZE * prices_T
        odd_val     = odd_held  * prices_T
        total_asset = capital + lot_val.sum() + odd_val.sum()
        lot_ratio   = lot_val / (total_asset + 1e-8)
        odd_ratio_v = odd_val / (total_asset + 1e-8)
        cash_ratio  = capital / (total_asset + 1e-8)

        feat_vec = np.concatenate([scaled[sid][i] for sid in OBS if sid in scaled])
        obs      = np.concatenate([feat_vec, lot_ratio, odd_ratio_v, [cash_ratio]])
        obs      = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)

        s = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # w：(1, N_ACTIONS) 當步動作權重
            # logit_state：更新後的 Leaky Integrator 狀態，下步傳入
            w, logit_state, _ = actor.sample(s, logit_state=logit_state)
        # 取前 n_stocks 維作為股票倉位，squeeze(0) 確保只壓 batch 維度
        target = np.clip(
            w.squeeze(0).cpu().numpy().astype(np.float64)[:n_stocks], 0.0, 1.0
        )

        target_value  = total_asset * target
        target_shares = target_value / (prices_T + 1e-8)
        target_lots   = (target_shares // LOT_SIZE).astype(np.int64)

        # 賣出
        for j in range(n_stocks):
            sell_lots = max(0, int(lots_held[j]) - int(target_lots[j]))
            if sell_lots > 0:
                gross = sell_lots * LOT_SIZE * float(prices_T[j])
                capital += gross - _lot_fee(gross) - _tax(gross)
                lots_held[j] -= sell_lots

        # 買入
        for j in range(n_stocks):
            buy_lots = max(0, int(target_lots[j]) - int(lots_held[j]))
            if buy_lots > 0:
                price        = float(prices_T[j])
                cost_per_lot = LOT_SIZE * price + _lot_fee(LOT_SIZE * price)
                affordable   = int(capital // cost_per_lot)
                buy_lots     = min(buy_lots, affordable)
                if buy_lots > 0:
                    gross = buy_lots * LOT_SIZE * price
                    capital -= gross + _lot_fee(gross)
                    lots_held[j] += buy_lots

        # 升級
        for j in range(n_stocks):
            if odd_held[j] >= LOT_SIZE:
                lots_held[j] += odd_held[j] // LOT_SIZE
                odd_held[j]  %= LOT_SIZE

        lot_val_T1 = lots_held * LOT_SIZE * prices_T1
        total_T1   = capital + lot_val_T1.sum() + odd_held.sum()

        if total_T1 > peak_total * 5 and explosion_step is None:
            explosion_step = i
            logger.log(f"\n  *** 資產爆炸偵測：step={i}  total={total_T1:,.0f} ***")
            logger.log(f"  target={target.round(3)}")
            logger.log(f"  lots_held={lots_held}")
            logger.log(f"  prices_T={prices_T.round(1)}")
            logger.log(f"  capital={capital:,.0f}")
        peak_total = max(peak_total, total_T1)

        if i % check_interval == 0 or i == n_steps - 2:
            logger.log(f"  {i:>6}  {capital:>14,.0f}  "
                       f"{lot_val_T1.sum():>14,.0f}  "
                       f"{total_T1:>14,.0f}  "
                       f"{lots_held.max():>10}")

    final = capital + (
        lots_held * LOT_SIZE *
        np.array([prices_dict[sid][-1] for sid in stock_ids])
    ).sum()
    logger.log(f"\n  最終資產: {final:,.0f}  報酬率: {(final/initial_capital-1)*100:.2f}%")
    logger.log(
        f"  {'*** 爆炸發生在 step=' + str(explosion_step) + ' ***' if explosion_step else '✓ 無異常爆炸'}"
    )

    return {"final": round(final, 0), "explosion_step": explosion_step}


def diag_walkforward_summary(window_results: list,
                              logger: DebugLogger) -> dict:
    """彙整各窗口的訓練/驗證報酬、Regime、勝率。"""
    logger.log("\n[診斷 8] Walk-forward 各窗口摘要")
    logger.log(f"  {'窗口':>4}  {'訓練期':>22}  {'驗證期':>22}  "
               f"{'訓練報酬':>10}  {'驗證報酬':>10}  {'Regime':>10}  {'勝率':>6}")
    logger.log("  " + "-" * 95)

    for w in window_results:
        train_ret = w.get("train_return", 0)
        val_ret   = w.get("val_return",   0)
        regime    = w.get("regime",       "?")
        win_rate  = w.get("win_rate",     0)
        overfit   = train_ret > val_ret * 3
        status    = "*** 過擬合 ***" if overfit else "✓"
        logger.log(
            f"  {w.get('window','?'):>4}  "
            f"{w.get('train_start','?'):>11}~{w.get('train_end','?'):>11}  "
            f"{w.get('val_start','?'):>11}~{w.get('val_end','?'):>11}  "
            f"{train_ret:>9.1f}%  {val_ret:>9.1f}%  "
            f"{regime:>10}  {win_rate:>5.1f}%  {status}"
        )

    val_returns = [w.get("val_return", 0) for w in window_results]
    avg_val     = round(float(np.mean(val_returns)), 2) if val_returns else 0
    std_val     = round(float(np.std(val_returns)),  2) if val_returns else 0

    logger.log(f"\n  驗證報酬均值: {avg_val}%  標準差: {std_val}%")
    logger.log(
        f"  {'✓ 策略穩定' if std_val < 20 else '*** 警告：各窗口報酬差異過大 ***'}"
    )

    return {
        "avg_val_return": avg_val,
        "std_val_return": std_val,
        "window_results": window_results,
    }


def diag_regime_model_selection(benchmark_prices: np.ndarray,
                                window_regimes: dict,
                                logger: DebugLogger) -> str:
    """根據當前市場環境選擇對應的模型窗口。"""
    current_regime = detect_regime(benchmark_prices)
    matched        = [w for w, r in window_regimes.items() if r == current_regime]
    selected       = matched[-1] if matched else max(window_regimes.keys())

    logger.log(f"\n[診斷 9] Regime 選模型")
    logger.log(f"  當前市場環境: {current_regime}")
    logger.log(f"  各窗口 Regime: {window_regimes}")
    if matched:
        logger.log(f"  ✓ 選擇窗口 {selected}（{current_regime} 環境匹配）")
    else:
        logger.log(f"  ⚠ 無完全匹配，退回最新窗口 {selected}")

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# 四、一鍵執行所有診斷
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_diagnostics(agent, env, feat, prices, volumes,
                        scalers, initial_capital: float,
                        tag: str = "full",
                        episode_returns: list = None,
                        episode_losses: list = None,
                        alphas: list = None) -> dict:
    """
    一鍵執行所有深度診斷，自動建立新的 log 檔案。

    v2 修正：
      - diag_final_holdings 改在 env.reset() 之前取快照，
        避免 reset() 清空持倉後診斷 7 永遠顯示初始狀態
      - diag_actor_logits 改從 agent.buffer 抽樣取 obs_batch，
        不再呼叫 env.reset() 建立假資料

    Args:
        agent:           SAC Agent 實例
        env:             PortfolioEnv 實例（應在最後一個 episode done 後傳入）
        feat:            特徵 dict
        prices:          價格 dict
        volumes:         成交量 dict
        scalers:         標準化器 dict
        initial_capital: 初始資金
        tag:             log 檔名標籤，例如 "train_w1"、"validate"
        episode_returns: 訓練期 return 列表（可選）
        episode_losses:  訓練期 loss 列表（可選）
        alphas:          訓練期 alpha 列表（可選）

    Returns:
        dict：各診斷結果的彙整
    """
    try:
        from configs.base_config import DEVICE
    except ImportError:
        DEVICE = torch.device("cpu")

    report = {}

    with new_logger(tag=tag) as logger:
        logger.log(f"\n  執行完整診斷  tag={tag}")

        # 環境層
        report["random_policy"]     = diag_random_policy(
            feat, prices, volumes, scalers, initial_capital, logger)
        report["feature_alignment"] = diag_feature_alignment(
            feat, prices, volumes, logger)
        report["reward_dist"]       = diag_reward_distribution(
            agent, feat, prices, volumes, scalers, initial_capital, logger)

        # ── 步驟 1：快照持倉（在任何 reset 之前）────────────────────────
        _holdings_snapshot = {
            "capital":   env.capital,
            "lots_held": env.lots_held.copy(),
            "odd_held":  env.odd_held.copy(),
            "step_idx":  env.step_idx,
        }

        # ── 步驟 2：Actor 診斷（從 buffer 取 obs，不呼叫 reset）─────────
        if hasattr(agent, "buffer") and len(agent.buffer) >= 4:
            _buf_states, _, _, _, _ = agent.buffer.sample(4)
            report["actor_logits"] = diag_actor_logits(
                agent.actor, _buf_states, logger)
        else:
            logger.log("  [診斷 4] buffer 不足 4 筆，跳過")
            report["actor_logits"] = []

        # ── 步驟 3：stochastic vs deterministic（需要 reset 取 obs）─────
        _diag_obs = env.reset()
        report["stoc_vs_det"] = diag_stochastic_vs_deterministic(
            agent, _diag_obs, logger)

        # ── 步驟 4：最後持倉（使用快照，不受 reset 影響）────────────────
        report["final_holdings"] = diag_final_holdings(
            env, logger, snapshot=_holdings_snapshot)

        # ── 步驟 5：訓練曲線（純資料）───────────────────────────────────
        if episode_returns and episode_losses and alphas:
            report["training_curve"] = diag_training_curve(
                episode_returns, episode_losses, alphas, logger)

    return report