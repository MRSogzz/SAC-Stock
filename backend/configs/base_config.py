"""
基礎設定：路徑、裝置、快取參數
"""
import os
import torch

# ─── 根目錄 ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
CACHE_DIR   = os.path.join(STORAGE_DIR, "cache")
MODEL_DIR   = os.path.join(STORAGE_DIR, "models")
HISTORY_DIR = os.path.join(STORAGE_DIR, "history")

for _dir in [CACHE_DIR, MODEL_DIR, HISTORY_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─── 裝置 ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"使用 GPU：{torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("使用 Apple Silicon GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    print("使用 CPU（未偵測到 GPU）")

# ─── 快取 ────────────────────────────────────────────────────────────────────
CACHE_EXPIRE_HOURS = 24   # 快取有效期（小時）

# ─── 期間對應起始日期 ─────────────────────────────────────────────────────────
from datetime import datetime, timedelta

PERIOD_START = {
    "1y": (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
    "2y": (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d"),
    "3y": (datetime.today() - timedelta(days=1095)).strftime("%Y-%m-%d"),
    "5y": (datetime.today() - timedelta(days=1825)).strftime("%Y-%m-%d"),
    "6y": (datetime.today() - timedelta(days=2190)).strftime("%Y-%m-%d"),
}