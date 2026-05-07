"""
訓練流程：train、validate、predict_next、模型存取
"""
import os
import pickle
import numpy as np
import torch
from datetime import datetime

from configs.base_config import MODEL_DIR, DEVICE, PERIOD_START, HISTORY_DIR
import csv as _csv
from configs.trading_config import (
    STOCK_POOL, N_FEATURES, DEFAULT_EPISODES,
    DEFAULT_VAL_DAYS, DEFAULT_INITIAL_CAP, DEFAULT_PERIOD,
    TRADEABLE_STOCKS, OBSERVABLE_STOCKS, BENCHMARK_STOCK, STATE_DIM,
    N_OBSERVABLE, N_TRADEABLE,
)
from src.data.loader import load_all_stocks
from src.data.processor import align_features
from src.environment.portfolio import PortfolioEnv
from src.agents.sac_agent import SACAgent
from src.models.architectures import PortfolioActor
from src.engine.backtester import run_backtest
from src.utils.common import sanitize, now_str
from diagnostics import register


# ─── Model Save / Load ───────────────────────────────────────────────────────

def model_path(period: str) -> str:
    return os.path.join(MODEL_DIR, f"portfolio_{period}.pkl")


@register(
    module="Engine",
    inputs={
        "period":         "str",
        "agent":          "SACAgent",
        "env":            "PortfolioEnv",
        "rules":          "dict",
        "summary":        "dict",
        "training_curve": "list[float]",
    },
    outputs={},
    notes="序列化 actor/critic state_dict + scalers 到 .pkl；儲存前搬回 CPU，完成後搬回 DEVICE",
)
def save_model(period: str, agent: SACAgent, env: PortfolioEnv,
               rules: dict, summary: dict, training_curve: list):
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
        "scalers":        env.scalers,
        "rules":          rules,
        "summary":        summary,
        "training_curve": training_curve,
        "saved_at":       now_str(),
        "period":         period,
    }
    with open(model_path(period), "wb") as f:
        pickle.dump(payload, f)
    agent.actor.to(DEVICE)
    agent.critic.to(DEVICE)
    agent.critic_target.to(DEVICE)
    print(f"模型已儲存：{model_path(period)}")


@register(
    module="Engine",
    inputs={"period": "str"},
    outputs={"return": "dict | None"},
    notes="從 .pkl 反序列化模型；檔案不存在回傳 None",
)
def load_model(period: str) -> dict | None:
    path = model_path(period)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    print(f"載入模型：{path}（{payload.get('saved_at', '?')}）")
    return payload


@register(
    module="Engine",
    inputs={},
    outputs={"return": "list[dict]"},
    notes="掃描 MODEL_DIR 列出所有 portfolio_*.pkl，按 saved_at 降序排列",
)
def list_models() -> list:
    os.makedirs(MODEL_DIR, exist_ok=True)
    models = []
    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith(".pkl") or not fname.startswith("portfolio"):
            continue
        try:
            with open(os.path.join(MODEL_DIR, fname), "rb") as f:
                p = pickle.load(f)
            summary = p.get("summary", {})

            total_return = (summary.get("total_return")
                            or p.get("total_return"))
            period = (p.get("period")
                      or summary.get("period")
                      or fname.replace("portfolio_", "").replace(".pkl", ""))
            episodes_done = (summary.get("episodes_done")
                             or summary.get("episodes"))

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
            print(f"list_models 讀取 {fname} 失敗：{e}")
            continue
    return sorted(models, key=lambda x: x.get("saved_at", ""), reverse=True)


# ─── Rule Extraction ─────────────────────────────────────────────────────────

@register(
    module="Engine",
    inputs={
        "all_actions":  "list[list[float]]",
        "stock_ids":    "list[str]",
        "feat_df_dict": "dict[str, pd.DataFrame]",
        "feat_names":   "list[str]",
    },
    outputs={"return": "dict[str, dict]"},
    notes="DecisionTree(depth=3) 擬合 buy/sell 標籤，輸出 top_5 特徵重要性 + tree_text",
)
def extract_portfolio_rules(all_actions: list, stock_ids: list,
                            feat_df_dict: dict, feat_names: list) -> dict:
    from sklearn.tree import DecisionTreeClassifier, export_text
    results = {}
    actions_arr = np.array(all_actions)

    for i, sid in enumerate(stock_ids):
        stock_actions = actions_arr[:, i]
        feat_df       = feat_df_dict[sid]
        n             = min(len(feat_df), len(stock_actions))

        discrete = np.array([
            1 if a > 0.2 else (2 if a < 0.05 else 0)
            for a in stock_actions[:n]
        ])
        mask = discrete != 0
        if mask.sum() < 10:
            results[sid] = {"top_features": [], "tree_text": "", "n_buy": 0, "n_sell": 0}
            continue

        X   = feat_df.values[:n][mask]
        y   = discrete[mask]
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
        clf.fit(X, y)
        importances = clf.feature_importances_
        top         = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:5]
        results[sid] = {
            "top_features": [{"name": n, "importance": round(float(v), 4)}
                             for n, v in top if v > 0],
            "tree_text": export_text(clf, feature_names=feat_names),
            "n_buy":  int((y == 1).sum()),
            "n_sell": int((y == 2).sum()),
        }
    return results


# ─── Train ───────────────────────────────────────────────────────────────────

@register(
    module="Engine",
    inputs={
        "period":          "str",
        "episodes":        "int",
        "initial_capital": "float",
        "val_days":        "int",
        "on_episode":      "Callable | None",
    },
    outputs={"return": "dict"},
    notes="SAC 訓練主流程：載入資料 → 切分訓練/驗證 → 接續訓練 → 回測 → 儲存模型",
)
def train(
    period:          str   = DEFAULT_PERIOD,
    episodes:        int   = DEFAULT_EPISODES,
    initial_capital: float = DEFAULT_INITIAL_CAP,
    val_days:        int   = DEFAULT_VAL_DAYS,
    on_episode=None,
) -> dict:

    stocks    = load_all_stocks(period)
    stock_ids = list(stocks.keys())
    feat_dfs, prices_dict, volumes_dict, feat_names, dates = align_features(stocks)

    total = len(feat_dfs[stock_ids[0]])
    if total <= val_days + 60:
        raise ValueError(f"數據不足，需要 {val_days+60} 筆，目前 {total} 筆")

    train_feat    = {sid: feat_dfs[sid].iloc[:-val_days]   for sid in feat_dfs}
    train_prices  = {sid: prices_dict[sid][:-val_days]      for sid in prices_dict}
    train_volumes = {sid: volumes_dict[sid][:-val_days]     for sid in volumes_dict}
    train_dates   = dates[:-val_days]

    print(f"訓練集: {len(train_dates)} 筆  保留驗證: {val_days} 筆")

    payload       = load_model(period)
    episodes_done = 0
    env           = PortfolioEnv(
        train_feat, train_prices, train_volumes,
        initial_capital=initial_capital
    )
    agent         = SACAgent(state_dim=env.state_dim, n_stocks=env.n_tradeable)

    from src.agents.memory import ReplayBuffer
    agent.buffer = ReplayBuffer()

    episode_returns = []

    if payload and payload.get("state_dim") == env.state_dim:
        print(f"繼續訓練（已完成 {payload['summary'].get('episodes', 0)} 回合）")
        agent.actor.load_state_dict(payload["actor_state"])
        agent.actor.to(DEVICE)
        if "critic_state" in payload:
            agent.critic.load_state_dict(payload["critic_state"])
            agent.critic_target.load_state_dict(payload["critic_state"])
            agent.critic.to(DEVICE)
            agent.critic_target.to(DEVICE)
        if "alpha" in payload:
            with torch.no_grad():
                agent.log_alpha.fill_(np.log(max(payload["alpha"], agent.alpha_min)))
            agent.alpha = agent.log_alpha.exp().item()
        episodes_done   = payload["summary"].get("episodes", 0)
        episode_returns = payload.get("training_curve", [])

    total_episodes = episodes_done + episodes
    print(f"SAC 訓練 {episodes} 回合（第 {episodes_done+1}~{total_episodes} 回合），"
          f"state_dim={env.state_dim}...")

    _diag_obs  = env.reset()
    _diag_det  = agent.act(_diag_obs, deterministic=True)
    _diag_stoc = agent.act(_diag_obs, deterministic=False)
    print("=" * 60)
    print("[診斷] 訓練前 deterministic action:")
    print("  各股倉位:", _diag_det.round(4))
    print(f"  總股票倉位: {_diag_det.sum():.4f}")
    print("[診斷] 訓練前 stochastic action:")
    print("  各股倉位:", _diag_stoc.round(4))
    print(f"  總股票倉位: {_diag_stoc.sum():.4f}")
    print("=" * 60)

    log_dir  = os.path.join(HISTORY_DIR, "training_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"train_{period}_{now_str().replace(' ','_').replace(':','-')}.csv"
    )
    log_fields = ["global_ep", "step", "critic_loss", "actor_loss", "alpha_loss", "alpha"]
    log_rows   = []

    pos_start = env.n_observable * N_FEATURES
    nan_count = 0

    for ep in range(episodes):
        obs         = env.reset()
        done        = False
        step_count  = 0
        ep_losses   = {"critic_loss": [], "actor_loss": [], "alpha_loss": []}
        trade_count = 0

        while not done:
            action = agent.act(obs)
            if np.isnan(action).any() or np.isinf(action).any():
                action = np.full(env.n_tradeable, 1.0 / (env.n_tradeable + 1))
                nan_count += 1
                if nan_count >= 10:
                    print(f"  WARNING: 連續 nan，重置 actor...")
                    agent.actor = PortfolioActor(env.state_dim, env.n_tradeable).to(DEVICE)
                    agent.actor._init_weights()
                    agent.actor_opt = torch.optim.Adam(
                        agent.actor.parameters(), lr=3e-4)
                    nan_count = 0

            prev_obs = obs
            next_obs, reward, done = env.step(action)

            if ep == 0 and step_count < 5:
                print(f"[診斷] ep=0 step={step_count}  "
                      f"action={action.round(4)}  reward={reward:.6f}")

            if np.isnan(reward) or np.isinf(reward):
                reward = 0.0
            agent.buffer.push(obs, action, reward, next_obs, float(done))
            step_count += 1

            if step_count > 1:
                prev_pos    = prev_obs[pos_start: pos_start + env.n_tradeable]
                trade_count += int((np.abs(action - prev_pos) > 0.05).any())

            if step_count % 2 == 0:
                loss_info = agent.update()
                if loss_info:
                    global_step = (episodes_done + ep) * env.n_steps + step_count
                    for k in ep_losses:
                        ep_losses[k].append(loss_info[k])
                    log_rows.append({
                        "global_ep":   episodes_done + ep + 1,
                        "step":        global_step,
                        "critic_loss": round(loss_info["critic_loss"], 6),
                        "actor_loss":  round(loss_info["actor_loss"],  6),
                        "alpha_loss":  round(loss_info["alpha_loss"],  6),
                        "alpha":       round(agent.alpha, 6),
                    })
            obs = next_obs

        for _ in range(4):
            loss_info = agent.update()
            if loss_info:
                for k in ep_losses:
                    ep_losses[k].append(loss_info[k])

        ret = float(env.portfolio_value())
        ret = ret if (not np.isnan(ret) and not np.isinf(ret)) else 1.0
        episode_returns.append(ret)

        avg_losses = {k: float(np.mean(v)) if v else 0.0
                      for k, v in ep_losses.items()}

        global_ep = episodes_done + ep + 1
        if on_episode:
            on_episode(global_ep, total_episodes, ret, agent.alpha,
                       avg_losses, trade_count)
        if (ep + 1) % 10 == 0:
            print(f"  ep {global_ep}/{total_episodes}  "
                  f"return={ret:.4f}  α={agent.alpha:.3f}  "
                  f"c_loss={avg_losses['critic_loss']:.4f}  "
                  f"trades={trade_count}")

    _diag_obs2 = env.reset()
    _diag_det2 = agent.act(_diag_obs2, deterministic=True)
    print("=" * 60)
    print("[診斷] 訓練後 deterministic action:")
    print("  各股倉位:", _diag_det2.round(4))

    try:
        _last_idx    = env.n_steps - 1
        _prices_last = np.array([
            env.prices[sid][_last_idx] for sid in env.tradeable_ids
        ], dtype=np.float64)
        _lot_val = env.lots_held * 1000 * _prices_last
        _odd_val = env.odd_held  * _prices_last
        _total   = env.capital + _lot_val.sum() + _odd_val.sum()
        _ret     = _total / env.initial_capital
        print(f"  實際報酬率: {(_ret - 1) * 100:.2f}%  "
              f"portfolio_value(): {env.portfolio_value():.6f}  "
              f"差距: {abs(_ret - env.portfolio_value()):.6f}")
        if abs(_ret - env.portfolio_value()) > 0.01:
            print("  *** 警告：portfolio_value() 與實際資產不一致 ***")
        else:
            print("  ✓ 數值一致")
    except Exception as e:
        print(f"  [診斷] 持倉確認失敗：{e}")
    print("=" * 60)

    if log_rows:
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"訓練 log 已儲存：{log_file}")

    bt    = run_backtest(
        agent.actor, env.scalers, train_feat, train_prices, train_volumes,
        TRADEABLE_STOCKS, initial_capital, feat_names, train_dates
    )
    rules = extract_portfolio_rules(
        bt["all_actions"], TRADEABLE_STOCKS,
        {sid: train_feat[sid] for sid in TRADEABLE_STOCKS}, feat_names)

    summary = {
        "total_return":     bt["total_return"],
        "bh_return":        bt["bh_return"],
        "risk_free_return": bt["risk_free_return"],
        "win_rate":         bt["win_rate"],
        "n_trades":         bt["n_trades"],
        "episodes":         total_episodes,
        "initial_capital":  initial_capital,
        "final_capital":    bt["final_capital"],
        "val_days":         val_days,
        "stock_ids":        TRADEABLE_STOCKS,
    }
    save_model(period, agent, env, rules, summary, episode_returns)

    return sanitize(dict(
        ticker             = "Portfolio",
        episodes           = total_episodes,
        episodes_this_run  = episodes,
        episodes_before    = episodes_done,
        training_curve     = episode_returns,
        stock_ids          = stock_ids,
        model_saved        = True,
        saved_at           = now_str(),
        train_days         = len(train_dates),
        val_days           = val_days,
        rules              = rules,
        **{k: v for k, v in bt.items() if k != "all_actions"},
    ))


# ─── Validate ────────────────────────────────────────────────────────────────

@register(
    module="Engine",
    inputs={
        "period":          "str",
        "val_days":        "int",
        "initial_capital": "float",
        "on_episode":      "Callable | None",
    },
    outputs={"return": "dict"},
    notes="載入模型 → 取最後 val_days 筆資料 → run_backtest → 回傳驗證集指標",
)
def validate(
    period:          str   = DEFAULT_PERIOD,
    val_days:        int   = DEFAULT_VAL_DAYS,
    initial_capital: float = DEFAULT_INITIAL_CAP,
    on_episode=None,
) -> dict:

    payload = load_model(period)
    if payload is None:
        raise ValueError(f"找不到 Portfolio {period} 的模型，請先訓練")

    actor = PortfolioActor(payload["state_dim"], payload["n_stocks"])
    actor.load_state_dict(payload["actor_state"])
    actor.to(DEVICE)
    actor.eval()

    stock_ids = payload["stock_ids"]
    scalers   = payload["scalers"]

    stocks = load_all_stocks(period)
    feat_dfs, prices_dict, volumes_dict, feat_names, dates = align_features(stocks)
    total = len(feat_dfs[stock_ids[0]])

    if total <= val_days + 60:
        raise ValueError(f"數據不足，需要 {val_days+60} 筆，目前 {total} 筆")

    val_feat    = {sid: feat_dfs[sid].iloc[-val_days:]   for sid in feat_dfs}
    val_prices  = {sid: prices_dict[sid][-val_days:]      for sid in prices_dict}
    val_volumes = {sid: volumes_dict[sid][-val_days:]     for sid in volumes_dict}
    val_dates   = dates[-val_days:]

    print(f"驗證集: {len(val_dates)} 筆（{val_dates[0]} ~ {val_dates[-1]}）")

    bt    = run_backtest(
        actor, scalers, val_feat, val_prices, val_volumes,
        TRADEABLE_STOCKS, initial_capital, feat_names, val_dates
    )
    rules = extract_portfolio_rules(
        bt["all_actions"], TRADEABLE_STOCKS,
        {sid: val_feat[sid] for sid in TRADEABLE_STOCKS}, feat_names)

    return sanitize(dict(
        mode             = "validation",
        val_days         = len(val_dates),
        val_start        = val_dates[0],
        val_end          = val_dates[-1],
        model_trained_at = payload.get("saved_at"),
        stock_ids        = stock_ids,
        rules            = rules,
        **{k: v for k, v in bt.items() if k != "all_actions"},
    ))


# ─── Predict Next Day ────────────────────────────────────────────────────────

@register(
    module="Engine",
    inputs={"period": "str"},
    outputs={"return": "dict"},
    notes="載入模型 → 取最新一天特徵 → deterministic 推論 → 回傳建議倉位與動作（買入/持有/觀望）",
)
def predict_next(period: str = DEFAULT_PERIOD) -> dict:
    payload = load_model(period)
    if payload is None:
        raise ValueError(f"找不到 Portfolio {period} 的模型，請先訓練")

    actor = PortfolioActor(payload["state_dim"], payload["n_stocks"])
    actor.load_state_dict(payload["actor_state"])
    actor.to(DEVICE)
    actor.eval()

    stock_ids = payload["stock_ids"]
    scalers   = payload["scalers"]
    rules     = payload.get("rules", {})
    summary   = payload.get("summary", {})

    stocks = load_all_stocks(period)
    from src.data.processor import align_features

    feat_dfs, _, _, _, _ = align_features(stocks)

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

    latest_feats      = np.concatenate(latest_feats)
    expected_feat_dim = N_OBSERVABLE * N_FEATURES
    assert len(latest_feats) == expected_feat_dim, \
        f"latest_feats 維度錯誤：{len(latest_feats)}，預期 {expected_feat_dim}"

    obs = np.concatenate([
        latest_feats,
        np.zeros(len(TRADEABLE_STOCKS)),
        np.zeros(len(TRADEABLE_STOCKS)),
        [1.0],
    ]).astype(np.float32)
    assert len(obs) == STATE_DIM, f"obs 維度錯誤：{len(obs)}，預期 {STATE_DIM}"

    with torch.no_grad():
        _, _, mean_act = actor.sample(
            torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        )
    target = mean_act.squeeze().cpu().numpy()[:len(TRADEABLE_STOCKS)]

    recommendations = []
    for j, sid in enumerate(TRADEABLE_STOCKS):
        pos    = float(target[j])
        name   = next(s["name"] for s in STOCK_POOL if s["id"] == sid)
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
        "as_of_date":       feat_dfs[stock_ids[0]].index[-1].strftime("%Y-%m-%d"),
        "recommendations":  recommendations,
        "cash_pct":         round(max(1.0 - float(target.sum()), 0.0) * 100, 1),
        "model_trained_at": payload.get("saved_at"),
        "rules":            rules,
        "model_summary":    summary,
    })