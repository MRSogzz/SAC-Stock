"""
engine/factory.py
統一的 Agent 與 Env 工廠。

trainer_standard 使用預設 SACAgent + PortfolioEnv（無 reward_mode 參數）。
trainer_walk_forward 使用實驗矩陣（RUN_CONFIGS），支援 logit/dirichlet actor
與 composite/linear reward。

公開 API：
  make_env(feat, prices, volumes, initial_capital,
           run_id=None, scalers=None)   -> PortfolioEnv
  make_agent(state_dim, n_stocks,
             run_id=None)               -> SACAgent 子類別
  rebuild_actor(run_id, state_dim, n_stocks) -> actor module

RUN_CONFIGS 也從此處集中匯出，供外部模組（predictor、trainer_walk_forward）查閱。
"""

from configs.base_config import DEVICE
from src.environment.portfolio import PortfolioEnv
from src.agents.sac_agent import SACAgent, SACAgentDirichlet, SACAgentLogitDelta
from src.models.architectures import PortfolioActorDirichlet, PortfolioActorLogitDelta


# ─── 實驗矩陣定義（集中維護）────────────────────────────────────────────────
# Run A/C 已拋棄；此處只保留現役 Run B/D。
# 外部程式若需擴充，直接修改此 dict，工廠方法自動支援。
RUN_CONFIGS: dict[str, dict] = {
    "B": {"actor": "logit", "reward": "composite", "desc": "Action only"},
    "D": {"actor": "logit", "reward": "linear",    "desc": "Full new"},
}


# ─── Env 工廠 ────────────────────────────────────────────────────────────────

def make_env(
    feat: dict,
    prices: dict,
    volumes: dict,
    initial_capital: float,
    run_id: str | None = None,
    scalers: dict | None = None,
) -> PortfolioEnv:
    """建立 PortfolioEnv。

    Args:
        feat / prices / volumes: {stock_id: array/DataFrame}
        initial_capital:         初始資金。
        run_id:  若為 None，使用 trainer_standard 的預設模式（無 reward_mode）；
                 若指定（"B"/"D"），依 RUN_CONFIGS 帶入 reward_mode。
        scalers: 傳入訓練期 scaler 可避免環境內部重新 fit（防止 look-ahead）。
                 傳入 None 時，PortfolioEnv 內部自行 fit。
    """
    if run_id is None:
        # trainer_standard 模式：不傳 reward_mode，使用環境預設
        return PortfolioEnv(feat, prices, volumes, initial_capital=initial_capital)

    cfg = _get_cfg(run_id)
    return PortfolioEnv(
        feat, prices, volumes,
        initial_capital=initial_capital,
        reward_mode=cfg["reward"],
        scalers=scalers,
    )


# ─── Agent 工廠 ──────────────────────────────────────────────────────────────

def make_agent(
    state_dim: int,
    n_stocks: int,
    run_id: str | None = None,
):
    """建立 SAC agent。

    run_id=None → 標準 SACAgent（trainer_standard）
    run_id="B"/"D" → 依 RUN_CONFIGS 選擇 SACAgentLogitDelta 或 SACAgentDirichlet
    """
    if run_id is None:
        return SACAgent(state_dim=state_dim, n_stocks=n_stocks)

    cfg = _get_cfg(run_id)
    if cfg["actor"] == "logit":
        return SACAgentLogitDelta(state_dim, n_stocks)
    return SACAgentDirichlet(state_dim, n_stocks)


# ─── Actor 重建（NaN 恢復 / predictor 載入用）────────────────────────────────

def rebuild_actor(
    run_id: str | None,
    state_dim: int,
    n_stocks: int,
):
    """建立一個新的（未載入權重的）actor，搬到 DEVICE。

    用途：
      1. 訓練迴圈中 NaN 時重置 actor。
      2. predictor / validate 中從 payload 重建 actor 後再 load_state_dict。

    run_id=None → PortfolioActor（standard）
    """
    if run_id is None:
        from src.models.architectures import PortfolioActor
        actor = PortfolioActor(state_dim, n_stocks)
    else:
        cfg = _get_cfg(run_id)
        if cfg["actor"] == "logit":
            actor = PortfolioActorLogitDelta(state_dim, n_stocks)
        else:
            actor = PortfolioActorDirichlet(state_dim, n_stocks)

    actor._init_weights()
    return actor.to(DEVICE)


# ─── 內部工具 ─────────────────────────────────────────────────────────────────

def _get_cfg(run_id: str) -> dict:
    if run_id not in RUN_CONFIGS:
        raise ValueError(
            f"未知的 run_id={run_id!r}，"
            f"有效值：{list(RUN_CONFIGS.keys())}"
        )
    return RUN_CONFIGS[run_id]