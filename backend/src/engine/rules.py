"""
engine/rules.py
統一的 XAI 決策樹規則提取。

trainer_standard 與 trainer_walk_forward 皆呼叫同一個 extract_portfolio_rules()，
不再各自持有重複實作。
"""

import numpy as np


def extract_portfolio_rules(
    all_actions: list,
    stock_ids: list,
    feat_df_dict: dict,
    feat_names: list,
) -> dict:
    """以 DecisionTree(depth=3) 擬合 buy/sell 標籤，輸出 top_5 特徵重要性 + tree_text。

    Args:
        all_actions:  shape (T, n_stocks) 的動作紀錄列表。
        stock_ids:    可交易股票 ID 列表，與 all_actions 第二維對應。
        feat_df_dict: {stock_id: pd.DataFrame}，index 為日期，columns 為特徵名。
        feat_names:   特徵名列表，與 feat_df_dict 的 columns 對應。

    Returns:
        {stock_id: {
            "top_features": [{"name": str, "importance": float}, ...],
            "tree_text":    str,
            "n_buy":        int,
            "n_sell":       int,
        }}
    """
    from sklearn.tree import DecisionTreeClassifier, export_text

    results = {}
    actions_arr = np.array(all_actions)

    for i, sid in enumerate(stock_ids):
        stock_actions = actions_arr[:, i]
        feat_df = feat_df_dict.get(sid)

        if feat_df is None:
            results[sid] = {"top_features": [], "tree_text": "", "n_buy": 0, "n_sell": 0}
            continue

        n = min(len(feat_df), len(stock_actions))

        discrete = np.array([
            1 if a > 0.2 else (2 if a < 0.05 else 0)
            for a in stock_actions[:n]
        ])
        mask = discrete != 0

        if mask.sum() < 10:
            results[sid] = {"top_features": [], "tree_text": "", "n_buy": 0, "n_sell": 0}
            continue

        X = feat_df.values[:n][mask]
        y = discrete[mask]

        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
        clf.fit(X, y)
        importances = clf.feature_importances_
        top = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:5]

        results[sid] = {
            "top_features": [
                {"name": name, "importance": round(float(val), 4)}
                for name, val in top
                if val > 0
            ],
            "tree_text": export_text(clf, feature_names=feat_names),
            "n_buy":  int((y == 1).sum()),
            "n_sell": int((y == 2).sum()),
        }

    return results