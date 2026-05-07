"""
FastAPI server — 使用新模組結構
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import uuid, time, traceback, csv
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.engine.trainer import train, validate, predict_next, list_models
from src.engine.walk_forward import (
    train_walkforward, train_experiment_matrix, predict_walkforward,
)
from src.utils.common import sanitize
from configs.base_config import HISTORY_DIR
from configs.trading_config import STOCK_POOL

from diagnostics import export_md, show_registry

HISTORY_FILE = os.path.join(HISTORY_DIR, "predictions.csv")
os.makedirs(HISTORY_DIR, exist_ok=True)

app = FastAPI(title="Portfolio AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

jobs: dict = {}


# ─── 每日自動預測 ─────────────────────────────────────────────────────────────

def run_daily_prediction():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] 每日自動預測...")
    models = list_models()
    if not models:
        return
    rows = []
    for m in models:
        period = m.get("period")
        if not period:
            continue
        try:
            pred = predict_next(period)
            date = pred.get("as_of_date", datetime.now().strftime("%Y-%m-%d"))
            for rec in pred.get("recommendations", []):
                rows.append({
                    "date":          date,
                    "run_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "period":        period,
                    "stock_id":      rec["stock_id"],
                    "stock_name":    rec["stock_name"],
                    "action":        rec["action"],
                    "target_pct":    rec["target_pct"],
                    "latest_price":  rec["latest_price"],
                    "actual_return": "",
                })
        except Exception as e:
            print(f"  {period} 預測失敗：{e}")
    if rows:
        file_exists = os.path.exists(HISTORY_FILE)
        with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
        print(f"  寫入 {len(rows)} 筆")


# ─── APScheduler ─────────────────────────────────────────────────────────────

scheduler = BackgroundScheduler(timezone="Asia/Taipei")
scheduler.add_job(
    run_daily_prediction,
    trigger=CronTrigger(day_of_week="mon-fri", hour=15, minute=30,
                        timezone="Asia/Taipei"),
    id="daily_prediction", replace_existing=True,
)

@app.on_event("startup")
def startup():
    scheduler.start()
    job = scheduler.get_job("daily_prediction")
    print(f"APScheduler 已啟動，下次執行：{job.next_run_time}")

    try:
        print("\n🔍 [Diagnostics] 偵測到服務啟動，正在更新 IO Map...")
        # 這裡執行時，所有的 import (train, validate, etc.) 已經完成
        # 因此 @register 已經將資料寫入 _REGISTRY
        export_md() 
        show_registry()
    except Exception as e:
        print(f"[Diagnostics] 生成 IO Map 失敗: {e}")

@app.on_event("shutdown")
def shutdown():
    scheduler.shutdown()


# ─── 排程 API ────────────────────────────────────────────────────────────────

@app.get("/scheduler/status")
def scheduler_status():
    job = scheduler.get_job("daily_prediction")
    count = 0
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, encoding="utf-8") as f:
            count = max(0, sum(1 for _ in csv.reader(f)) - 1)
    return {
        "running":       scheduler.running,
        "next_run":      str(job.next_run_time) if job else None,
        "history_count": count,
    }

@app.post("/scheduler/run-now")
def run_now():
    try:
        run_daily_prediction()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/scheduler/history")
def get_history(limit: int = 100):
    if not os.path.exists(HISTORY_FILE):
        return {"records": [], "total": 0}
    with open(HISTORY_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.reverse()
    return {"records": rows[:limit], "total": len(rows)}


# ─── Job 輔助 ─────────────────────────────────────────────────────────────────

def _run_job(job_id: str, fn, req):
    jobs[job_id]["status"]   = "running"
    jobs[job_id]["start"]    = time.time()
    jobs[job_id]["progress"] = []

    def on_episode(ep, total, ret, alpha, avg_losses=None, trade_count=0):
        elapsed = round(time.time() - jobs[job_id]["start"], 1)
        jobs[job_id]["progress"].append({
            "ep":          ep,
            "total":       total,
            "ret":         round(ret, 4),
            "alpha":       round(alpha, 4),
            "critic_loss": round(avg_losses.get("critic_loss", 0), 4) if avg_losses else 0,
            "actor_loss":  round(avg_losses.get("actor_loss",  0), 4) if avg_losses else 0,
            "alpha_loss":  round(avg_losses.get("alpha_loss",  0), 4) if avg_losses else 0,
            "trade_count": trade_count,
            "elapsed":     elapsed,
        })
        jobs[job_id]["current_ep"] = ep

    try:
        kwargs = req.dict()
        kwargs["on_episode"] = on_episode
        result = fn(**kwargs)
        jobs[job_id]["status"]  = "done"
        jobs[job_id]["result"]  = sanitize(result)
        jobs[job_id]["elapsed"] = round(time.time() - jobs[job_id]["start"], 1)
    except Exception:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = traceback.format_exc()


# ─── Train（原有）────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    period:          str   = "6y"
    episodes:        int   = 80
    initial_capital: float = 1_000_000
    val_days:        int   = 250

@app.post("/train")
def start_train(req: TrainRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "config": req.dict(),
                    "progress": [], "current_ep": 0}
    bg.add_task(_run_job, job_id, train, req)
    return {"job_id": job_id}


# ─── Walk-forward Train ───────────────────────────────────────────────────────

class WalkForwardRequest(BaseModel):
    period:          str        = "6y"
    episodes:        int        = 200          # 每個窗口 200 ep（= 150,000步）
    initial_capital: float      = 1_000_000
    runs:            list[str]  = ["D"]        # 預設只跑 Run D；Run A/C 已拋棄

@app.post("/train_walkforward")
def start_train_walkforward(req: WalkForwardRequest, bg: BackgroundTasks):
    """
    啟動 walk-forward 訓練（3 窗口 × 3年訓練 + 1年驗證）。
    runs 預設 ["D"]；如需 B 請傳入 ["B","D"]。Run A/C 已拋棄。
    每個窗口訓練完後自動執行偵錯診斷。
    """
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "config": req.dict(),
                    "progress": [], "current_ep": 0,
                    "mode": "walkforward"}
    bg.add_task(_run_job, job_id, train_experiment_matrix, req)
    return {"job_id": job_id}


# ─── Walk-forward Predict ─────────────────────────────────────────────────────

@app.get("/predict_walkforward/{period}")
def predict_wf(period: str):
    """
    根據當前市場 Regime 選擇對應窗口的模型進行預測。
    回傳結果包含 current_regime 和 selected_window。
    """
    try:
        return sanitize(predict_walkforward(period))
    except Exception:
        raise HTTPException(400, detail=traceback.format_exc())


# ─── Walk-forward Status ──────────────────────────────────────────────────────

@app.get("/walkforward_status")
def walkforward_status():
    """
    查詢各 Run（A/B/C/D）各窗口模型的狀態和績效摘要。
    """
    import pickle
    from src.engine.walk_forward import wf_meta_path, window_model_path

    result = {"runs": {}}

    for run_id in ["B", "D"]:   # Run A/C 已拋棄
        run_info = {"windows": [], "meta": None}

        # 讀取各 Run 的元資料
        meta_path = wf_meta_path(run_id)
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            run_info["meta"] = {
                "saved_at":       meta.get("saved_at"),
                "period":         meta.get("period"),
                "window_regimes": meta.get("window_regimes"),
                "avg_val_return": round(float(
                    sum(w.get("val_return", 0)
                        for w in meta.get("window_results", [])) /
                    max(len(meta.get("window_results", [])), 1)
                ), 2),
            }

        # 讀取各窗口模型
        for w in range(1, 4):
            path = window_model_path(w, run_id)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                summary = payload.get("summary", {})
                run_info["windows"].append({
                    "window":       w,
                    "saved_at":     payload.get("saved_at"),
                    "train_start":  summary.get("train_start"),
                    "train_end":    summary.get("train_end"),
                    "val_start":    summary.get("val_start"),
                    "val_end":      summary.get("val_end"),
                    "train_return": summary.get("train_return"),
                    "val_return":   summary.get("val_return"),
                    "regime":       summary.get("regime"),
                    "win_rate":     summary.get("val_win_rate"),
                })

        result["runs"][run_id] = sanitize(run_info)

    return result


# ─── Validate（原有）────────────────────────────────────────────────────────

class ValidateRequest(BaseModel):
    period:          str   = "6y"
    val_days:        int   = 250
    initial_capital: float = 1_000_000

@app.post("/validate")
def start_validate(req: ValidateRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "config": req.dict(),
                    "progress": [], "current_ep": 0}
    bg.add_task(_run_job, job_id, validate, req)
    return {"job_id": job_id}


# ─── Status / Result ─────────────────────────────────────────────────────────

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return {
        "job_id":     job_id,
        "status":     j["status"],
        "config":     j.get("config"),
        "error":      j.get("error"),
        "elapsed":    j.get("elapsed"),
        "current_ep": j.get("current_ep", 0),
        "progress":   j.get("progress", []),
        "mode":       j.get("mode", "train"),
    }

@app.get("/result/{job_id}")
def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "done":
        raise HTTPException(400, f"Job status: {j['status']}")
    return j["result"]


# ─── Predict / Models ────────────────────────────────────────────────────────

@app.get("/predict/{period}")
def predict(period: str):
    try:
        return sanitize(predict_next(period))
    except Exception:
        raise HTTPException(400, detail=traceback.format_exc())

@app.get("/models")
def get_models():
    return {"models": list_models()}

@app.get("/stock-pool")
def get_stock_pool():
    return {"stocks": STOCK_POOL}

@app.get("/health")
def health():
    return {"ok": True}