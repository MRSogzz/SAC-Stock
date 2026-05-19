"""
engine/persistence.py
統一的模型儲存與載入。

支援兩種命名格式：
  - Period 模型：portfolio_{period}.pkl          （來自 trainer_standard）
  - Window 模型：portfolio_w{window}_run{run_id}.pkl （來自 trainer_walk_forward）

公開 API：
  save_period_model(period, agent, env, rules, summary, training_curve)
  load_period_model(period) -> dict | None
  list_period_models()      -> list[dict]

  save_window_model(window, run_id, agent, env, summary)
  load_window_model(window, run_id) -> dict | None

  period_model_path(period)         -> str
  window_model_path(window, run_id) -> str
  wf_meta_path(run_id)              -> str
  monitor_log_path(run_id, window)  -> str
"""

import os
import pickle

from configs.base_config import MODEL_DIR, DEVICE
from src.utils.common import now_str


# ─── 路徑工具 ────────────────────────────────────────────────────────────────

def period_model_path(period: str) -> str:
    return os.path.join(MODEL_DIR, f"portfolio_{period}.pkl")


def window_model_path(window: int, run_id: str = "D") -> str:
    return os.path.join(MODEL_DIR, f"portfolio_w{window}_run{run_id}.pkl")


def wf_meta_path(run_id: str = "D") -> str:
    return os.path.join(MODEL_DIR, f"walkforward_meta_run{run_id}.pkl")


def monitor_log_path(run_id: str, window: int) -> str:
    log_dir = os.path.join(MODEL_DIR, "monitor_logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"monitor_run{run_id}_w{window}.pkl")


# ─── Period 模型（trainer_standard 使用）────────────────────────────────────

def save_period_model(
    period: str,
    agent,
    env,
    rules: dict,
    summary: dict,
    training_curve: list,
) -> None:
    """序列化 actor/critic state_dict + scalers 到 .pkl。
    儲存前搬回 CPU，完成後搬回 DEVICE。
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    agent.actor.cpu()
    agent.critic.cpu()
    payload = {
        "actor_state":    agent.actor.state_dict(),
        "critic_state":   agent.critic.state_dict(),
        "alpha":          float(agent.alpha),
        "state_dim":      env.state_dim,
        "n_stocks":       env.n_tradeable,
        "stock_ids":      env.tradeable_ids,
        "scalers":        {},
        "rules":          rules,
        "summary":        summary,
        "training_curve": training_curve,
        "saved_at":       now_str(),
        "period":         period,
    }
    path = period_model_path(period)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    agent.actor.to(DEVICE)
    agent.critic.to(DEVICE)
    agent.critic_target.to(DEVICE)
    print(f"模型已儲存：{path}")


def load_period_model(period: str) -> dict | None:
    """從 .pkl 反序列化模型；檔案不存在回傳 None。"""
    path = period_model_path(period)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    print(f"載入模型：{path}（{payload.get('saved_at', '?')}）")
    return payload


def list_period_models() -> list[dict]:
    """掃描 MODEL_DIR 列出所有 portfolio_*.pkl，按 saved_at 降序排列。"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    models = []
    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith(".pkl") or not fname.startswith("portfolio"):
            continue
        # 排除 window 模型（portfolio_w{n}_run{r}.pkl）
        if "_run" in fname:
            continue
        try:
            with open(os.path.join(MODEL_DIR, fname), "rb") as f:
                p = pickle.load(f)
            summary = p.get("summary", {})
            total_return = summary.get("total_return") or p.get("total_return")
            period = (
                p.get("period")
                or summary.get("period")
                or fname.replace("portfolio_", "").replace(".pkl", "")
            )
            episodes_done = summary.get("episodes_done") or summary.get("episodes")
            models.append({
                "file":          fname,
                "ticker":        "Portfolio",
                "period":        period,
                "saved_at":      p.get("saved_at"),
                "total_return":  total_return,
                "episodes":      summary.get("episodes"),
                "episodes_done": episodes_done,
                "stock_ids":     p.get("stock_ids", []),
            })
        except Exception as e:
            print(f"list_period_models 讀取 {fname} 失敗：{e}")
    return sorted(models, key=lambda x: x.get("saved_at", ""), reverse=True)


# ─── Window 模型（trainer_walk_forward 使用）────────────────────────────────

def save_window_model(
    window: int,
    run_id: str,
    agent,
    env,
    summary: dict,
) -> None:
    """序列化 window 模型到 .pkl；儲存前搬回 CPU，完成後搬回 DEVICE。"""
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
        "scalers":       {},
        "summary":       summary,
        "saved_at":      now_str(),
        "window":        window,
        "run_id":        run_id,
        "episodes_done": summary.get("episodes_done", 0),
    }
    if hasattr(agent, "_logit_state"):
        payload["logit_state"] = agent._logit_state.tolist()

    path = window_model_path(window, run_id)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    agent.actor.to(DEVICE)
    agent.critic.to(DEVICE)
    agent.critic_target.to(DEVICE)
    print(
        f"[Run {run_id}] 窗口 {window} 模型已儲存"
        f"（累積 {summary.get('episodes_done', 0)} 回合）"
    )


def load_window_model(window: int, run_id: str) -> dict | None:
    """載入 window 模型；檔案不存在回傳 None。"""
    path = window_model_path(window, run_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def list_all_models() -> list[dict]:
    """掃描 MODEL_DIR 列出所有模型（標準 + Walk-Forward），按 saved_at 降序排列。"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    models = []
    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith(".pkl") or not fname.startswith("portfolio"):
            continue
        try:
            with open(os.path.join(MODEL_DIR, fname), "rb") as f:
                p = pickle.load(f)

            is_wf = "_run" in fname
            summary = p.get("summary", {})
            total_return = summary.get("total_return") or p.get("total_return")

            if is_wf:
                # Walk-Forward 模型：portfolio_w{window}_run{run_id}.pkl
                window = p.get("window")
                run_id = p.get("run_id")
                period = f"窗口{window} Run{run_id}"
                episodes_done = p.get("episodes_done") or summary.get("episodes_done")
                # Walk-Forward summary 的報酬欄位可能是 val_return 或 total_return
                wf_return = (
                    summary.get("val_return")
                    or summary.get("total_return")
                    or p.get("total_return")
                )
                models.append({
                    "file":          fname,
                    "model_type":    "walkforward",
                    "period":        period,
                    "window":        window,
                    "run_id":        run_id,
                    "saved_at":      p.get("saved_at"),
                    "total_return":  wf_return,
                    "episodes":      summary.get("episodes"),
                    "episodes_done": episodes_done,
                    "stock_ids":     p.get("stock_ids", []),
                })
            else:
                # 標準模型：portfolio_{period}.pkl
                period = (
                    p.get("period")
                    or summary.get("period")
                    or fname.replace("portfolio_", "").replace(".pkl", "")
                )
                episodes_done = summary.get("episodes_done") or summary.get("episodes")
                models.append({
                    "file":          fname,
                    "model_type":    "standard",
                    "period":        period,
                    "saved_at":      p.get("saved_at"),
                    "total_return":  total_return,
                    "episodes":      summary.get("episodes"),
                    "episodes_done": episodes_done,
                    "stock_ids":     p.get("stock_ids", []),
                })
        except Exception as e:
            print(f"list_all_models 讀取 {fname} 失敗：{e}")
    return sorted(models, key=lambda x: x.get("saved_at", ""), reverse=True)