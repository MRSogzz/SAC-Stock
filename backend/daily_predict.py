"""
每日自動預測腳本
由 Windows 工作排程器在每天 15:30 執行，不需要後端常駐。
直接呼叫新模組結構執行預測並存檔，同時回填前一日實際報酬。
"""

import sys, os, csv
import pandas as pd
from datetime import datetime

# 確保能找到 src/ 和 configs/
sys.path.insert(0, os.path.dirname(__file__))

from configs.base_config import HISTORY_DIR
from src.engine.trainer import list_models, predict_next
from src.data.loader import load_all_stocks

HISTORY_FILE = os.path.join(HISTORY_DIR, "predictions.csv")
LOG_FILE     = os.path.join(HISTORY_DIR, "run_log.txt")
os.makedirs(HISTORY_DIR, exist_ok=True)


def log(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run():
    log("=== 每日自動預測開始 ===")

    models = list_models()
    if not models:
        log("沒有已訓練的模型，請先訓練後再執行")
        sys.exit(0)

    # ── 預測 ──────────────────────────────────────────────────────────────────
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
            log(f"period={period} 預測完成，{len(pred.get('recommendations', []))} 支股票")
        except Exception as e:
            log(f"period={period} 預測失敗：{e}")

    if not rows:
        log("沒有產生任何預測結果")
        sys.exit(0)

    # ── 寫入 CSV ──────────────────────────────────────────────────────────────
    file_exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    log(f"寫入 {len(rows)} 筆預測紀錄 → {HISTORY_FILE}")

    # ── 回填前一日實際報酬 ────────────────────────────────────────────────────
    try:
        stocks   = load_all_stocks("1y")
        all_rows = []
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))

        filled = 0
        for row in all_rows:
            if row["actual_return"] != "":
                continue
            sid = row["stock_id"]
            if sid not in stocks:
                continue
            df      = stocks[sid]
            pred_dt = pd.Timestamp(row["date"])
            future  = df[df.index > pred_dt]
            if future.empty:
                continue
            next_price  = float(future["Close"].iloc[0])
            entry_price = float(row["latest_price"])
            ret         = round((next_price / entry_price - 1) * 100, 3)
            row["actual_return"] = f"{ret:+.3f}%"
            filled += 1

        if filled > 0:
            with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_rows)
            log(f"回填了 {filled} 筆實際報酬")

    except Exception as e:
        log(f"回填失敗（不影響預測紀錄）：{e}")

    log("=== 每日自動預測完成 ===\n")


if __name__ == "__main__":
    run()