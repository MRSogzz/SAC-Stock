"""
evaluation/
===========
SAC-Stock-v7 Alpha 科學驗證協議。

這個模組是「不可學習的外部裁判」，完全獨立於訓練流程。
任何新特徵或策略改動必須先通過此協議，才能進入正式訓練。

使用方式：
    from evaluation.alpha_validator import AlphaValidator

    validator = AlphaValidator(
        baseline_model_path="storage/models/best_model_w2_ep125.pkl",
        validation_period=("2025-10-02", "2026-05-07"),
        data_period="6y",
    )
    report = validator.run(candidate_feature_config={
        "name": "momentum_accel_v2",
        "description": "改進版動量加速度",
        "compute_fn": my_feature_fn,  # callable(stocks) -> dict[str, pd.DataFrame]
    })
    report.save("reports/momentum_accel_v2.json")

三層測試：
  Layer 1：預測訊號檢驗（Rank IC > 0.02，方向準確率 > 52%）
  Layer 2：策略增益檢驗（ΔSharpe、Δ最大回撤、Δ換倉率）
  Layer 3：反事實穩定性與失敗模式檢查

禁止在此模組中：
  ❌ 任何可訓練參數
  ❌ 修改 SAC 參數
  ❌ 修改 processor.py 核心邏輯
  ❌ 直接導入新特徵（必須通過驗證協議）
"""

from .alpha_validator import AlphaValidator, ValidationReport

__all__ = ["AlphaValidator", "ValidationReport"]