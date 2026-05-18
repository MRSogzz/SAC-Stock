"""
backend/routers/alpha_validation.py
=====================================
Alpha 因子驗證的 FastAPI 路由。

掛載方式（在 server.py 已完成）：
    app.include_router(alpha_router, prefix="/alpha_validation", tags=["Alpha Validation"])
"""
from __future__ import annotations

import json
import os
import textwrap
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

REPORTS_DIR = Path("reports/alpha_validation")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── JSON 序列化工具（處理 numpy 型別）────────────────────────────────────────

def _sanitize(obj):
    """
    遞迴將 numpy 型別轉為 Python 原生型別，確保 json.dumps 不報錯。
    涵蓋 float32/float64/int32/int64/bool_/ndarray 以及巢狀 dict/list。
    """
    import numpy as np
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    return obj


# ── Request / Response models ─────────────────────────────────────────────────

# 附加診斷類型對應表（特徵名稱 → 診斷類型）
EXTRA_DIAGNOSTIC_MAP = {
    "trend_efficiency_20":  "low_vol_exposure",
    "vol_regime_shift":     "crisis_attribution",
    "ret5_vol20_ratio":     "turnover_defense",
    "volume_impulse_vol20": "turnover_defense",
}


class RunValidationRequest(BaseModel):
    name:           str
    description:    str = ""
    feature_code:   str
    feature_column: Optional[str] = None
    skip_layer1:    bool = False
    model_path:     Optional[str] = None
    extra_diagnostic: Optional[str] = None   # 由前端傳入，或根據 name 自動判斷


class ReportListItem(BaseModel):
    feature_name:      str
    description:       str = ""
    final_verdict:     str
    validation_period: Optional[list] = None
    saved_at:          str = ""


# ── 關鍵：將 to_dict() 的巢狀結構攤平為前端期待的格式 ────────────────────────

def _flatten_report(raw: dict) -> dict:
    """
    ValidationReport.to_dict() 輸出：
      { "metadata": {...}, "verdict": {...}, "layer1": {...}, ... }

    前端期待：
      { "feature_name": ..., "final_verdict": ..., "layer1": {...}, ... }
    """
    meta    = raw.get("metadata", {})
    verdict = raw.get("verdict",  {})

    return {
        # 頂層 metadata 攤平
        "feature_name":        meta.get("feature_name", ""),
        "description":         meta.get("description", ""),
        "generated_at":        meta.get("generated_at", ""),
        "baseline_model_path": meta.get("baseline_model_path", ""),
        "validation_period":   [meta.get("val_start", ""), meta.get("val_end", "")],
        "data_period":         meta.get("data_period", ""),

        # verdict 攤平
        "final_verdict":  verdict.get("final_verdict", "PENDING"),
        "stop_at_layer":  verdict.get("stop_at_layer"),
        "verdict_reason": verdict.get("verdict_reason", ""),

        # 三層結果（保持原結構，前端 Panel 直接讀）
        "layer1": raw.get("layer1"),
        "layer2": raw.get("layer2"),
        "layer3": raw.get("layer3"),

        # 額外補存資訊（給 ReportsListPanel 用）
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _flatten_saved_report(raw: dict) -> dict:
    """載入已儲存的 JSON 時也做同樣攤平（JSON 格式與 to_dict() 相同）。"""
    # 如果已經是攤平格式（有 feature_name 直接在頂層），直接回傳
    if "feature_name" in raw and "metadata" not in raw:
        return raw
    return _flatten_report(raw)


# ── Helper：安全執行使用者程式碼 ──────────────────────────────────────────────

def _exec_feature_code(code: str):
    dedented = textwrap.dedent(code)
    namespace: dict = {}
    try:
        exec(dedented, namespace)   # noqa: S102
    except SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"程式碼語法錯誤：{e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"程式碼執行失敗：{e}")

    if "compute_fn" not in namespace:
        raise HTTPException(
            status_code=400,
            detail="找不到 compute_fn 函數，請確認函數名稱為 compute_fn",
        )
    return namespace["compute_fn"]


def _resolve_model_path(model_path: Optional[str]) -> str:
    """
    前端有傳 model_path 就用，否則在 storage/models 找最新的主模型 pkl。
    排除 meta / monitor 輔助檔，優先選窗口 2（walk-forward 黃金基準）。
    """
    storage = Path("storage/models")

    if model_path:
        # 前端傳來的可能是 file 欄位（如 portfolio_w2_runD.pkl）或完整路徑
        p = Path(model_path)
        if p.exists():
            return str(p)
        # 嘗試在 storage/models 下尋找同名檔案
        alt = storage / p.name
        if alt.exists():
            return str(alt)
        raise HTTPException(status_code=400, detail=f"模型檔案不存在：{model_path}")

    if not storage.exists():
        raise HTTPException(status_code=500, detail="找不到 storage/models 目錄")

    EXCLUDE = {"meta", "monitor"}
    candidates = [
        p for p in storage.glob("*.pkl")
        if not any(ex in p.stem.lower() for ex in EXCLUDE)
    ]

    if not candidates:
        candidates = list(storage.glob("*.pkl"))

    if not candidates:
        raise HTTPException(
            status_code=500,
            detail="storage/models 目錄內找不到任何 .pkl，請先完成訓練"
        )

    w2 = [p for p in candidates if "w2" in p.stem]
    chosen = max(w2 or candidates, key=lambda p: p.stat().st_mtime)
    print(f"[alpha_validation] 自動選取模型：{chosen}")
    return str(chosen)


def _infer_feature_column(feature_column: Optional[str], feature_name: str) -> Optional[str]:
    """
    feature_column 的優先順序：
      1. 前端明確傳入的值
      2. feature_name（通常與欄位名相同）
      3. None（讓 Layer1 自動取第一欄，但那是基準特徵的欄位，會算錯！）
    所以這裡至少 fallback 到 feature_name。
    """
    if feature_column and feature_column.strip():
        return feature_column.strip()
    if feature_name and feature_name.strip():
        return feature_name.strip()
    return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/run")
async def run_validation(req: RunValidationRequest):
    """執行完整的 3 層 Alpha 驗證管線，回傳攤平格式的報告 JSON。"""
    try:
        from evaluation.alpha_validator import AlphaValidator
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"無法匯入 AlphaValidator：{e}")

    raw_compute_fn = _exec_feature_code(req.feature_code)
    baseline_path  = _resolve_model_path(req.model_path)

    # 把新特徵欄位加上 _alpha_ 前綴，避免與 baseline 38 個欄位重名。
    # baseline 已有 ret_3/vol_5/rsi_centered 等，直接同名 concat
    # 後 .iloc[t][col] 會回傳 Series 而非 scalar，導致 float() 失敗。
    PREFIX = "_alpha_"

    def compute_fn(stocks):
        raw = raw_compute_fn(stocks)
        result = {}
        for sid, df in raw.items():
            result[sid] = df.rename(columns={c: f"{PREFIX}{c}" for c in df.columns})
        return result

    raw_col     = _infer_feature_column(req.feature_column, req.name)
    feature_col = f"{PREFIX}{raw_col}" if raw_col else None

    print(f"\n[alpha_validation] 開始驗證：{req.name}")
    print(f"  model_path     = {baseline_path}")
    print(f"  feature_column = {feature_col}  (raw: {raw_col})")
    print(f"  skip_layer1    = {req.skip_layer1}")

    try:
        from configs.base_config import VAL_START, VAL_END, DATA_PERIOD
    except ImportError:
        VAL_START   = "2025-10-02"
        VAL_END     = "2026-05-07"
        DATA_PERIOD = "6y"

    try:
        validator = AlphaValidator(
            baseline_model_path=baseline_path,
            validation_period=(VAL_START, VAL_END),
            data_period=DATA_PERIOD,
        )

        # 自動判斷附加診斷類型（前端傳入優先，否則根據特徵名稱自動映射）
        extra_diag = req.extra_diagnostic or EXTRA_DIAGNOSTIC_MAP.get(req.name)
        print(f"  extra_diagnostic = {extra_diag}")

        report = validator.run(
            candidate_feature_config={
                "name":        req.name,
                "description": req.description,
                "compute_fn":  compute_fn,
            },
            feature_column=feature_col,   # 帶 _alpha_ 前綴的完整欄位名
            skip_layer1=req.skip_layer1,
            extra_diagnostic=extra_diag,
        )

        # 攤平成前端格式，並消毒 numpy 型別
        flat = _sanitize(_flatten_report(report.to_dict()))

        # 儲存攤平後的 JSON（這樣 load 回來也是同格式）
        report_path = REPORTS_DIR / f"{req.name}.json"
        report_path.write_text(
            json.dumps(flat, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[alpha_validation] 報告已儲存：{report_path}")

        return flat

    except HTTPException:
        raise
    except Exception:
        tb = traceback.format_exc()
        print(f"[alpha_validation] 驗證失敗：\n{tb}")
        raise HTTPException(status_code=500, detail=f"驗證執行失敗：\n{tb}")


@router.get("/reports")
async def list_reports():
    """列出所有已儲存的驗證報告（摘要）。"""
    items = []
    for path in sorted(REPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            raw  = json.loads(path.read_text(encoding="utf-8"))
            flat = _flatten_saved_report(raw)
            items.append({
                "feature_name":      flat.get("feature_name", path.stem),
                "description":       flat.get("description", ""),
                "final_verdict":     flat.get("final_verdict", "UNKNOWN"),
                "validation_period": flat.get("validation_period"),
                "saved_at":          flat.get("saved_at", ""),
            })
        except Exception:
            continue
    return {"reports": items}


@router.get("/reports/{name}")
async def get_report(name: str):
    """載入特定名稱的完整驗證報告（攤平格式）。"""
    path = REPORTS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"找不到報告：{name}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _flatten_saved_report(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"讀取報告失敗：{e}")


@router.delete("/reports/{name}")
async def delete_report(name: str):
    """刪除特定報告。"""
    path = REPORTS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"找不到報告：{name}")
    path.unlink()
    return {"deleted": name}