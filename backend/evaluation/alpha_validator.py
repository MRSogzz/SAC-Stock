"""
evaluation/alpha_validator.py
==============================
Alpha 科學驗證協議的核心執行引擎。

使用方式：
    validator = AlphaValidator(
        baseline_model_path="storage/models/portfolio_w2_runD.pkl",
        validation_period=("2025-10-02", "2026-05-07"),
        data_period="6y",
    )
    report = validator.run(
        candidate_feature_config={
            "name":        "my_feature",
            "description": "說明",
            "compute_fn":  my_fn,
        },
        feature_column="my_feature",
    )
    print(report.summary())
    report.save("reports/my_feature.json")
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .layer1_signal     import Layer1Result, run_layer1
from .layer2_strategy   import Layer2Result, run_layer2
from .layer3_robustness import Layer3Result, run_layer3


# ═══════════════════════════════════════════════════════════════════════════════
# ValidationReport
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationReport:
    feature_name:   str
    description:    str
    generated_at:   str

    layer1: Layer1Result   | None = None
    layer2: Layer2Result   | None = None
    layer3: Layer3Result   | None = None

    final_verdict:  str        = "PENDING"
    stop_at_layer:  int | None = None
    verdict_reason: str        = ""

    baseline_model_path: str = ""
    val_start:           str = ""
    val_end:             str = ""
    data_period:         str = ""

    def summary(self) -> str:
        sep = "=" * 60
        lines = [
            sep,
            f"  Alpha 驗證報告：{self.feature_name}",
            f"  {self.description}",
            f"  驗證期間：{self.val_start} ~ {self.val_end}",
            f"  基準模型：{Path(self.baseline_model_path).name}",
            f"  生成時間：{self.generated_at}",
            sep,
        ]
        if self.layer1:
            lines.append(self.layer1.summary())
        else:
            lines.append("Layer 1：未執行")
        lines.append("")
        if self.layer2:
            lines.append(self.layer2.summary())
        elif self.stop_at_layer == 1:
            lines.append("Layer 2：已跳過（Layer 1 未通過）")
        else:
            lines.append("Layer 2：未執行")
        lines.append("")
        if self.layer3:
            lines.append(self.layer3.summary())
        elif self.stop_at_layer in (1, 2):
            lines.append(f"Layer 3：已跳過（Layer {self.stop_at_layer} 未通過）")
        else:
            lines.append("Layer 3：未執行")
        lines.append("")
        lines.append(sep)
        icons = {"PASS": "✅", "FAIL": "❌", "PARTIAL": "⚠️", "PENDING": "⏳"}
        lines.append(f"  最終判決：{icons.get(self.final_verdict, '?')} {self.final_verdict}")
        if self.verdict_reason:
            lines.append(f"  判決原因：{self.verdict_reason}")
        lines.append(sep)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "metadata": {
                "feature_name":        self.feature_name,
                "description":         self.description,
                "generated_at":        self.generated_at,
                "baseline_model_path": self.baseline_model_path,
                "val_start":           self.val_start,
                "val_end":             self.val_end,
                "data_period":         self.data_period,
            },
            "verdict": {
                "final_verdict":  self.final_verdict,
                "stop_at_layer":  self.stop_at_layer,
                "verdict_reason": self.verdict_reason,
            },
            "layer1": self.layer1.to_dict() if self.layer1 else None,
            "layer2": self.layer2.to_dict() if self.layer2 else None,
            "layer3": self.layer3.to_dict() if self.layer3 else None,
        }

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[AlphaValidator] 報告已儲存：{out.resolve()}")

    @classmethod
    def load(cls, path: str) -> "ValidationReport":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = data["metadata"]
        v    = data["verdict"]
        return cls(
            feature_name        = meta["feature_name"],
            description         = meta["description"],
            generated_at        = meta["generated_at"],
            baseline_model_path = meta["baseline_model_path"],
            val_start           = meta["val_start"],
            val_end             = meta["val_end"],
            data_period         = meta["data_period"],
            final_verdict       = v["final_verdict"],
            stop_at_layer       = v["stop_at_layer"],
            verdict_reason      = v["verdict_reason"],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AlphaValidator
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaValidator:
    """
    Alpha 科學驗證協議主引擎。

    Args:
        baseline_model_path: 黃金基準模型 checkpoint 路徑（.pkl）
        validation_period:   驗證集區間 ("YYYY-MM-DD", "YYYY-MM-DD")
        data_period:         資料期間（如 "6y"）
    """

    def __init__(
        self,
        baseline_model_path: str,
        validation_period: tuple[str, str],
        data_period: str = "6y",
    ):
        self.baseline_model_path = baseline_model_path
        self.val_start, self.val_end = validation_period
        self.data_period = data_period
        print("[AlphaValidator] 載入市場資料...")
        self._load_market_data()

    def _load_market_data(self) -> None:
        from src.data.loader    import load_all_stocks
        from src.data.processor import align_features
        from configs.trading_config import TRADEABLE_STOCKS, OBSERVABLE_STOCKS

        stocks = load_all_stocks(self.data_period)
        (self._baseline_feat_dfs,
         self._prices_dict,
         self._volumes_dict,
         self._feat_names,
         self._dates) = align_features(stocks)

        self._tradeable_ids  = TRADEABLE_STOCKS
        self._observable_ids = OBSERVABLE_STOCKS
        self._stocks         = stocks
        print(f"  特徵數：{len(self._feat_names)}，"
              f"日期：{self._dates[0]} ~ {self._dates[-1]}")

    def _build_candidate_feat_dfs(self, compute_fn: Callable) -> dict:
        import pandas as pd

        print("[AlphaValidator] 計算候選特徵...")
        new_feats = compute_fn(self._stocks)

        # 取共同 index
        common_idx = None
        for sid, df in self._baseline_feat_dfs.items():
            common_idx = df.index if common_idx is None \
                         else common_idx.intersection(df.index)
        for sid, df in new_feats.items():
            if sid in self._tradeable_ids:
                common_idx = common_idx.intersection(df.index)

        candidate_feat_dfs = {}
        for sid in self._baseline_feat_dfs:
            base = self._baseline_feat_dfs[sid].loc[common_idx].copy()
            if sid in new_feats:
                extra = new_feats[sid].reindex(common_idx)
                candidate_feat_dfs[sid] = pd.concat([base, extra], axis=1)
            else:
                candidate_feat_dfs[sid] = base

        first_sid = list(candidate_feat_dfs.keys())[0]
        n_new = (len(candidate_feat_dfs[first_sid].columns)
                 - len(self._baseline_feat_dfs[first_sid].columns))
        print(f"  基準 {len(self._baseline_feat_dfs[first_sid].columns)} 個"
              f" + 新增 {n_new} 個 = "
              f"{len(candidate_feat_dfs[first_sid].columns)} 個特徵")
        return candidate_feat_dfs

    # ── 主要入口 ──────────────────────────────────────────────────────────────

    def run(
        self,
        candidate_feature_config: dict,
        feature_column:   str | None = None,
        skip_layer1:      bool = False,
        extra_diagnostic: str | None = None,
    ) -> ValidationReport:
        """
        依序執行三層驗證。

        Args:
            candidate_feature_config: {
                "name":        str,
                "description": str,
                "compute_fn":  Callable,
            }
            feature_column:   Layer1 要測試的特定欄位（None 取第一欄）
            skip_layer1:      True 時跳過 Layer1
            extra_diagnostic: 附加診斷類型，可選：
                              "low_vol_exposure" / "crisis_attribution" /
                              "turnover_defense" / None
        """
        name    = candidate_feature_config.get("name", "unnamed")
        desc    = candidate_feature_config.get("description", "")
        compute = candidate_feature_config.get("compute_fn")

        if compute is None:
            raise ValueError("candidate_feature_config 必須包含 'compute_fn'")

        report = ValidationReport(
            feature_name        = name,
            description         = desc,
            generated_at        = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            baseline_model_path = self.baseline_model_path,
            val_start           = self.val_start,
            val_end             = self.val_end,
            data_period         = self.data_period,
        )

        print(f"\n{'='*60}")
        print(f"  Alpha 驗證：{name}")
        print(f"  驗證期間：{self.val_start} ~ {self.val_end}")
        print(f"{'='*60}")

        # 計算候選特徵
        try:
            candidate_feat_dfs = self._build_candidate_feat_dfs(compute)
        except Exception as e:
            report.final_verdict  = "FAIL"
            report.verdict_reason = f"候選特徵計算失敗：{e}"
            print(f"[AlphaValidator] ❌ {report.verdict_reason}")
            return report

        # ── Layer 1 ───────────────────────────────────────────────────────────
        if not skip_layer1:
            print("\n[Layer 1] 預測訊號檢驗...")
            try:
                layer1 = run_layer1(
                    candidate_features = candidate_feat_dfs,
                    prices_dict        = self._prices_dict,
                    dates              = self._dates,
                    tradeable_ids      = self._tradeable_ids,
                    val_start          = self.val_start,
                    val_end            = self.val_end,
                    feature_column     = feature_column,
                )
            except Exception as e:
                layer1 = Layer1Result(passed=False, rejection_reasons=[f"執行錯誤：{e}"])
            report.layer1 = layer1
            print(layer1.summary())

            if not layer1.passed:
                report.final_verdict  = "FAIL"
                report.stop_at_layer  = 1
                report.verdict_reason = "Layer 1 未通過：" + "；".join(layer1.rejection_reasons)
                print(f"\n[AlphaValidator] ❌ Layer 1 否決，終止驗證")
                return report
        else:
            print("\n[Layer 1] 已跳過（skip_layer1=True）")

        # ── Layer 2 ───────────────────────────────────────────────────────────
        print("\n[Layer 2] 特徵微量探針檢驗...")
        try:
            layer2 = run_layer2(
                baseline_model_path = self.baseline_model_path,
                baseline_feat_dfs   = self._baseline_feat_dfs,
                candidate_feat_dfs  = candidate_feat_dfs,
                prices_dict         = self._prices_dict,
                volumes_dict        = self._volumes_dict,
                dates               = self._dates,
                tradeable_ids       = self._tradeable_ids,
                observable_ids      = self._observable_ids,
                val_start           = self.val_start,
                val_end             = self.val_end,
                feature_column      = feature_column,
                extra_diagnostic    = extra_diagnostic if extra_diagnostic in (
                    "low_vol_exposure", "turnover_defense"
                ) else None,
            )
        except Exception as e:
            layer2 = Layer2Result(passed=False, rejection_reasons=[f"執行錯誤：{e}"])
        report.layer2 = layer2
        print(layer2.summary())

        if not layer2.passed:
            report.final_verdict  = "FAIL"
            report.stop_at_layer  = 2
            report.verdict_reason = (
                f"Layer 2 [{layer2.verdict}]：" + "；".join(layer2.rejection_reasons)
                if layer2.rejection_reasons else f"Layer 2 [{layer2.verdict}]"
            )
            print(f"\n[AlphaValidator] ❌ Layer 2 否決，終止驗證")
            return report

        # ── Layer 3 ───────────────────────────────────────────────────────────
        print("\n[Layer 3] 反事實穩定性與失敗模式檢查...")
        try:
            layer3 = run_layer3(
                layer2_result    = layer2,
                prices_dict      = self._prices_dict,
                dates            = self._dates,
                tradeable_ids    = self._tradeable_ids,
                val_start        = self.val_start,
                val_end          = self.val_end,
                extra_diagnostic = extra_diagnostic if extra_diagnostic == "crisis_attribution" else None,
            )
        except Exception as e:
            layer3 = Layer3Result(passed=False, rejection_reasons=[f"執行錯誤：{e}"])
        report.layer3 = layer3
        print(layer3.summary())

        if not layer3.passed:
            report.final_verdict  = "FAIL"
            report.stop_at_layer  = 3
            report.verdict_reason = "Layer 3 未通過：" + "；".join(layer3.rejection_reasons)
            print(f"\n[AlphaValidator] ❌ Layer 3 否決")
            return report

        # ── 全部通過 ──────────────────────────────────────────────────────────
        report.final_verdict  = "PASS"
        report.verdict_reason = (
            f"三層全數通過：ΔSharpe={layer2.delta_sharpe:+.4f}，"
            f"穩定區間={layer3.stable_regimes}/{len(layer3.regime_results)}"
        )
        print(f"\n[AlphaValidator] ✅ {report.verdict_reason}")
        return report

    # ── 批次驗證 ──────────────────────────────────────────────────────────────

    def run_batch(
        self,
        configs:          list[dict],
        output_dir:       str = "reports/alpha_validation",
        skip_layer1:      bool = False,
        extra_diagnostic: str | None = None,
    ) -> dict[str, ValidationReport]:
        results: dict[str, ValidationReport] = {}
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for i, cfg in enumerate(configs):
            name = cfg.get("name", f"feature_{i}")
            print(f"\n{'#'*60}")
            print(f"# [{i+1}/{len(configs)}] 驗證：{name}")
            print(f"{'#'*60}")
            report = self.run(cfg, skip_layer1=skip_layer1,
                              extra_diagnostic=extra_diagnostic)
            results[name] = report
            report.save(str(out / f"{name}.json"))

        print(f"\n{'='*60}")
        print(f"  批次驗證摘要（{len(configs)} 個候選特徵）")
        print(f"{'='*60}")
        passed = [n for n, r in results.items() if r.final_verdict == "PASS"]
        failed = [n for n, r in results.items() if r.final_verdict == "FAIL"]
        print(f"  ✅ PASS ({len(passed)}): {', '.join(passed) or '無'}")
        print(f"  ❌ FAIL ({len(failed)}): {', '.join(failed) or '無'}")

        index = {
            "generated_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "val_start":      self.val_start,
            "val_end":        self.val_end,
            "baseline_model": self.baseline_model_path,
            "results": {
                name: {
                    "verdict":      r.final_verdict,
                    "stop_at":      r.stop_at_layer,
                    "delta_sharpe": r.layer2.delta_sharpe if r.layer2 else None,
                }
                for name, r in results.items()
            },
        }
        (out / "_index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return results