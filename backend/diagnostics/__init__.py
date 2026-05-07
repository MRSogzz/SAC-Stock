# diagnostics/__init__.py
from .Registry import register, export_md, show_registry, _REGISTRY
from .validators import validate_input, nan_guard
from .logger import DebugLogger, new_logger, registry_log
from .inspector import show_all, show, summary
from .training_analyst import (
    diag_random_policy,
    diag_feature_alignment,
    diag_reward_distribution,
    diag_actor_logits,
    diag_stochastic_vs_deterministic,
    diag_training_curve,
    diag_final_holdings,
    _monitor_logit_delta,
)
from .backtest_analyst import (
    detect_regime,
    diag_backtest_curve,
    diag_walkforward_summary,
    diag_regime_model_selection,
)


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
        tag:             log 檔名標籤，例如 "runA_w1"、"validate"
        episode_returns: 訓練期 return 列表（可選）
        episode_losses:  訓練期 loss 列表（可選）
        alphas:          訓練期 alpha 列表（可選）

    Returns:
        dict：各診斷結果的彙整
    """
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

        # ── 步驟 1：快照持倉（在任何 reset 之前）────────────────────────────
        _holdings_snapshot = {
            "capital":   env.capital,
            "lots_held": env.lots_held.copy(),
            "odd_held":  env.odd_held.copy(),
            "step_idx":  env.step_idx,
        }

        # ── 步驟 2：Actor 診斷（從 buffer 取 obs，不呼叫 reset）─────────────
        if hasattr(agent, "buffer") and len(agent.buffer) >= 4:
            # LogitDelta buffer 回傳 6 個值，Dirichlet 回傳 5 個值，統一取第一個
            buf_data    = agent.buffer.sample(4)
            _buf_states = buf_data[0]
            report["actor_logits"] = diag_actor_logits(
                agent.actor, _buf_states, logger)
        else:
            logger.log("  [診斷 4] buffer 不足 4 筆，跳過")
            report["actor_logits"] = []

        # ── 步驟 3：stochastic vs deterministic（需要 reset 取 obs）──────────
        _diag_obs = env.reset()
        report["stoc_vs_det"] = diag_stochastic_vs_deterministic(
            agent, _diag_obs, logger)

        # ── 步驟 4：最後持倉（使用快照，不受 reset 影響）─────────────────────
        report["final_holdings"] = diag_final_holdings(
            env, logger, snapshot=_holdings_snapshot)

        # ── 步驟 5：訓練曲線（純資料）────────────────────────────────────────
        if episode_returns and episode_losses and alphas:
            report["training_curve"] = diag_training_curve(
                episode_returns, episode_losses, alphas, logger)

    return report


__all__ = [
    # Registry
    "register", "export_md", "show_registry", "_REGISTRY",
    # Validators
    "validate_input", "nan_guard",
    # Log
    "DebugLogger", "new_logger", "registry_log",
    # Inspector
    "show_all", "show", "summary",
    # Training analyst
    "diag_random_policy", "diag_feature_alignment", "diag_reward_distribution",
    "diag_actor_logits", "diag_stochastic_vs_deterministic",
    "diag_training_curve", "diag_final_holdings", "_monitor_logit_delta",
    # Backtest analyst
    "detect_regime", "diag_backtest_curve",
    "diag_walkforward_summary", "diag_regime_model_selection",
    # Entry point
    "run_all_diagnostics",
]