"""
Information Coefficient（IC）計算
衡量模型對個股強弱排名的預測能力

執行方式：
  cd backend
  python calc_ic.py --period 5y --val-days 250
"""
import sys, os, argparse
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

from src.engine.trainer import load_model
from src.data.loader import load_all_stocks
from src.data.processor import align_features
from src.models.architectures import PortfolioActor
from configs.base_config import DEVICE
from configs.trading_config import OBSERVABLE_STOCKS, TRADEABLE_STOCKS

import torch


def calc_ic(period: str = "5y", val_days: int = 250):
    # ── 載入模型 ──────────────────────────────────────────────────────────
    payload = load_model(period)
    if payload is None:
        print(f"找不到模型 portfolio_{period}.pkl，請先訓練")
        return

    actor = PortfolioActor(payload["state_dim"], payload["n_stocks"])
    actor.load_state_dict(payload["actor_state"])
    actor.to(DEVICE)
    actor.eval()

    stock_ids = TRADEABLE_STOCKS   # 9 支可交易

    print(f"模型訓練時間：{payload.get('saved_at')}")
    print(f"使用驗證集：最後 {val_days} 個交易日")
    print(f"股票池：{stock_ids}")
    print()

    # ── 載入數據 ──────────────────────────────────────────────────────────
    stocks = load_all_stocks(period)
    feat_dfs, prices_dict, _, feat_names, dates = align_features(stocks)

    total = len(feat_dfs[stock_ids[0]])
    if total <= val_days + 60:
        print(f"數據不足：{total} 筆，需要 {val_days+60} 筆")
        return

    # 切出驗證集（全部 10 支，含 0050 作為觀測）
    val_feat   = {sid: feat_dfs[sid].iloc[-val_days:]   for sid in feat_dfs}
    val_prices = {sid: prices_dict[sid][-val_days:]      for sid in prices_dict}
    val_dates  = dates[-val_days:]
    n_steps    = val_days

    # 標準化特徵
    scaled = {}
    for sid in OBSERVABLE_STOCKS:
        if sid not in val_feat:
            continue
        feat = val_feat[sid].values.copy().astype(np.float64)
        feat = np.where(np.isposinf(feat),  10.0, feat)
        feat = np.where(np.isneginf(feat), -10.0, feat)
        feat = np.where(np.isnan(feat),      0.0, feat)
        scaled[sid] = np.clip(feat, -10.0, 10.0)

    # ── 逐日計算模型預測倉位和實際報酬 ───────────────────────────────────
    ic_daily    = []
    rank_ic_daily = []
    dates_used  = []

    positions = np.zeros(len(stock_ids))

    for i in range(1, n_steps - 1):
        # 建構觀測
        feat_vec  = np.concatenate([scaled[sid][i] for sid in OBSERVABLE_STOCKS])
        total_sh  = np.zeros(len(stock_ids), dtype=int)
        odd_ratio = np.zeros(len(stock_ids))
        cash      = max(1.0 - positions.sum(), 0.0)
        obs       = np.concatenate([feat_vec, positions, odd_ratio, [cash]]).astype(np.float32)
        obs       = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

        # 模型預測
        with torch.no_grad():
            _, _, mean = actor.sample(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
        pred_weights = mean.squeeze().cpu().numpy()
        pred_weights = np.clip(pred_weights, 0.0, 0.4)

        # 隔日實際報酬（防止除以零）
        actual_rets = np.array([
            val_prices[sid][i] / val_prices[sid][i-1] - 1.0
            if val_prices[sid][i-1] > 0 else 0.0
            for sid in stock_ids
        ])

        # 過濾 nan/inf 或異常值
        if np.isnan(actual_rets).any() or np.isnan(pred_weights).any() or np.isinf(actual_rets).any():
            continue

        # 過濾空倉天數（pred_weights 全為 0 或標準差極小，IC 無意義）
        if pred_weights.std() < 0.001:
            continue

        # Pearson IC（預測倉位 vs 實際報酬）
        pearson_r, pearson_p = stats.pearsonr(pred_weights, actual_rets)

        # Spearman Rank IC（排名相關）
        spearman_r, spearman_p = stats.spearmanr(pred_weights, actual_rets)

        ic_daily.append(pearson_r)
        rank_ic_daily.append(spearman_r)
        dates_used.append(val_dates[i])

        # 更新 positions（用 deterministic 動作）
        positions = pred_weights.copy()

    # ── 計算統計指標 ──────────────────────────────────────────────────────
    ic_arr      = np.array(ic_daily)
    rank_ic_arr = np.array(rank_ic_daily)

    ic_mean   = float(np.mean(ic_arr))
    ic_std    = float(np.std(ic_arr))
    ic_ir     = ic_mean / (ic_std + 1e-8)   # Information Ratio
    ic_pos    = float((ic_arr > 0).mean())   # IC > 0 的比例

    rank_ic_mean = float(np.mean(rank_ic_arr))
    rank_ic_std  = float(np.std(rank_ic_arr))
    rank_ic_ir   = rank_ic_mean / (rank_ic_std + 1e-8)
    rank_ic_pos  = float((rank_ic_arr > 0).mean())

    # t 檢定（IC 是否顯著異於 0）
    t_stat, p_value = stats.ttest_1samp(ic_arr, 0)
    t_stat_r, p_value_r = stats.ttest_1samp(rank_ic_arr, 0)

    print("=" * 55)
    print("Pearson IC（預測強度 vs 實際報酬）")
    print("=" * 55)
    print(f"  IC Mean:      {ic_mean:+.4f}  {'★ 顯著正相關' if ic_mean > 0.05 else '△ 弱正相關' if ic_mean > 0 else '✗ 負相關'}")
    print(f"  IC Std:       {ic_std:.4f}")
    print(f"  IC IR:        {ic_ir:+.4f}  {'★ 優秀' if ic_ir > 0.5 else '△ 尚可' if ic_ir > 0.3 else ''}")
    print(f"  IC > 0 比例:  {ic_pos:.1%}")
    print(f"  t統計量:      {t_stat:.3f}  p值: {p_value:.4f}  {'(顯著)' if p_value < 0.05 else '(不顯著)'}")
    print()
    print("=" * 55)
    print("Spearman Rank IC（排名預測能力）")
    print("=" * 55)
    print(f"  Rank IC Mean: {rank_ic_mean:+.4f}  {'★ 顯著正相關' if rank_ic_mean > 0.05 else '△ 弱正相關' if rank_ic_mean > 0 else '✗ 負相關'}")
    print(f"  Rank IC Std:  {rank_ic_std:.4f}")
    print(f"  Rank IC IR:   {rank_ic_ir:+.4f}")
    print(f"  Rank IC > 0:  {rank_ic_pos:.1%}")
    print(f"  有效計算天數: {len(rank_ic_daily)} 天（過濾空倉天數後）")
    print(f"  t統計量:      {t_stat_r:.3f}  p值: {p_value_r:.4f}  {'(顯著)' if p_value_r < 0.05 else '(不顯著)'}")
    print()

    # ── 解讀 ──────────────────────────────────────────────────────────────
    print("=" * 55)
    print("解讀")
    print("=" * 55)
    if rank_ic_mean > 0.1:
        print("  ★ 優秀：模型有顯著的選股排名能力")
        print("    業界頂尖量化基金的 Rank IC 約在 0.05~0.15")
    elif rank_ic_mean > 0.05:
        print("  △ 良好：模型有一定的選股能力")
        print("    持倉比例太低導致報酬不佳，但選股邏輯是對的")
    elif rank_ic_mean > 0:
        print("  ○ 弱正相關：模型略優於隨機，但效果不顯著")
    else:
        print("  ✗ 負相關：模型的排名判斷與實際相反")

    # ── 自相關分析（均值回歸 vs 趨勢延續）──────────────────────────────────
    print("=" * 55)
    print("各股 Lag-1 自相關（市場特性診斷）")
    print("=" * 55)
    autocorrs = []
    for sid in stock_ids:
        prices = np.array([val_prices[sid][j] for j in range(n_steps)])
        prices = prices[prices > 0]   # 過濾停牌日
        if len(prices) < 10:
            continue
        rets_raw = np.diff(prices) / prices[:-1]
        rets_raw = rets_raw[np.isfinite(rets_raw)]
        if len(rets_raw) < 5:
            continue
        ac = float(np.corrcoef(rets_raw[:-1], rets_raw[1:])[0, 1])
        autocorrs.append(ac)
        name = next((s["name"] for s in __import__("configs.trading_config",
                     fromlist=["STOCK_POOL"]).STOCK_POOL if s["id"] == sid), sid)
        tag = "均值回歸" if ac < -0.05 else ("趨勢延續" if ac > 0.05 else "隨機")
        print(f"  {name}({sid}): {ac:+.4f}  [{tag}]")
    if autocorrs:
        mean_ac = float(np.nanmean(autocorrs))
        print(f"  平均: {mean_ac:+.4f}  {'→ 整體均值回歸，AI 需學習反向操作' if mean_ac < -0.03 else '→ 整體趨勢延續'}")
    print()

    # ── 儲存每日 IC 到 CSV ────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "storage", "history")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ic_result_{period}.csv")

    df = pd.DataFrame({
        "date":       dates_used,
        "pearson_ic": ic_daily,
        "rank_ic":    rank_ic_daily,
    })
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n每日 IC 已儲存：{out_path}")
    print(f"（可用 Excel 開啟，繪製 IC 隨時間的變化趨勢）")

    return {
        "ic_mean":        ic_mean,
        "ic_ir":          ic_ir,
        "rank_ic_mean":   rank_ic_mean,
        "rank_ic_ir":     rank_ic_ir,
        "n_days":         len(ic_daily),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period",   default="5y")
    parser.add_argument("--val-days", type=int, default=250, dest="val_days")
    args = parser.parse_args()
    calc_ic(period=args.period, val_days=args.val_days)
