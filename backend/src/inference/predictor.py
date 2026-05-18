"""
inference/predictor.py
生產環境推論服務。

兼容兩種模型格式：
  - 常規模型（trainer_standard）：portfolio_{period}.pkl
  - Walk-Forward 最佳模型（trainer_walk_forward）：自動 regime 選擇

公開 API：
  predict(period, mode)           -> dict
    mode="standard"   → 直接呼叫 trainer_standard.predict_next()
    mode="walkforward" → 直接呼叫 trainer_walk_forward.predict_walkforward()
    mode="auto"（預設）→ 若 Walk-Forward meta 存在則優先使用，否則 fallback 至 standard

  health_check(period)            -> dict   回傳兩種模型的可用性摘要
"""

import os

from configs.base_config import MODEL_DIR

from src.engine.persistence import (
    load_period_model, load_window_model, wf_meta_path,
)
from src.engine.trainer_standard import predict_next as _predict_standard
from src.engine.trainer_walk_forward import predict_walkforward as _predict_walkforward


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def predict(period: str = "6y", mode: str = "auto") -> dict:
    """統一推論入口。

    Args:
        period: 資料期間，例如 "6y"、"3y"。
        mode:
          "standard"    → 常規模型（period 命名）
          "walkforward" → Walk-Forward 最佳模型（regime 自動選擇）
          "auto"        → 優先 Walk-Forward；無 meta 時 fallback 至 standard

    Returns:
        推論結果 dict，含 recommendations、cash_pct、as_of_date 等欄位，
        以及 "__source__" 欄位說明使用了哪種模型。
    """
    if mode == "standard":
        result = _predict_standard(period)
        result["__source__"] = "standard"
        return result

    if mode == "walkforward":
        result = _predict_walkforward(period)
        result["__source__"] = "walkforward"
        return result

    # auto：有 Walk-Forward meta 則優先使用
    if _has_walkforward_meta():
        try:
            result = _predict_walkforward(period)
            result["__source__"] = "walkforward"
            return result
        except Exception as e:
            print(f"[predictor] Walk-Forward 推論失敗（{e}），fallback 至 standard")

    if load_period_model(period) is None:
        raise ValueError(
            f"找不到任何可用模型（period={period}）。"
            "請先執行 train() 或 train_experiment_matrix()。"
        )

    result = _predict_standard(period)
    result["__source__"] = "standard (fallback)"
    return result


# ─── 健康檢查 ─────────────────────────────────────────────────────────────────

def health_check(period: str = "6y") -> dict:
    """回傳兩種模型的可用性摘要，方便監控或 API 端點使用。"""
    standard_payload = load_period_model(period)
    wf_available     = _has_walkforward_meta()

    # 找出最新的 Walk-Forward window 資訊
    wf_info: dict | None = None
    if wf_available:
        import pickle
        for rid in ["B", "D"]:
            m_path = wf_meta_path(rid)
            if not os.path.exists(m_path):
                continue
            with open(m_path, "rb") as f:
                m = pickle.load(f)
            window_results = m.get("window_results", [])
            if window_results:
                best = max(window_results, key=lambda r: r.get("val_return", -float("inf")))
                wf_info = {
                    "run_id":     rid,
                    "best_window":best.get("window"),
                    "val_return": best.get("val_return"),
                    "saved_at":   m.get("saved_at"),
                }
                break

    return {
        "standard": {
            "available":  standard_payload is not None,
            "period":     period,
            "saved_at":   standard_payload.get("saved_at") if standard_payload else None,
            "episodes":   (standard_payload.get("summary", {}).get("episodes")
                           if standard_payload else None),
        },
        "walkforward": {
            "available": wf_available,
            **(wf_info or {}),
        },
        "recommended_mode": "walkforward" if wf_available else "standard",
    }


# ─── 內部工具 ─────────────────────────────────────────────────────────────────

def _has_walkforward_meta() -> bool:
    """判斷是否有任何 Walk-Forward meta 檔案存在。"""
    for rid in ["B", "D"]:
        if os.path.exists(wf_meta_path(rid)):
            return True
    return False