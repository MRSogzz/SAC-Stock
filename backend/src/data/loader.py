"""
數據載入：FinMind API 下載 + 本地快取管理
"""
import os
import pandas as pd
from datetime import datetime

from configs.base_config import CACHE_DIR, CACHE_EXPIRE_HOURS, PERIOD_START
from configs.trading_config import STOCK_POOL


def load_stock(stock_id: str, period: str = "3y") -> pd.DataFrame:
    """
    下載單支股票歷史數據，優先讀取快取。
    快取有效期：CACHE_EXPIRE_HOURS 小時。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    start_date = PERIOD_START.get(period, PERIOD_START["3y"])
    cache_path = os.path.join(CACHE_DIR, f"{stock_id}_{period}.csv")

    # 檢查快取
    if os.path.exists(cache_path):
        age_hr = (datetime.now() - datetime.fromtimestamp(
            os.path.getmtime(cache_path))).total_seconds() / 3600
        if age_hr < CACHE_EXPIRE_HOURS:
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            return df
        print(f"快取過期（{age_hr:.1f}h），重新下載 {stock_id}...")

    # 從 FinMind 下載
    from FinMind.data import DataLoader
    print(f"下載 {stock_id}（{start_date} 起）...")
    dl  = DataLoader()
    raw = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)

    if raw is None or raw.empty:
        raise ValueError(f"FinMind 無法取得 {stock_id} 的數據")

    df = raw.rename(columns={
        "date": "Date", "open": "Open", "max": "High",
        "min": "Low", "close": "Close", "Trading_Volume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # 過濾停牌日：收盤價為 0 或成交量為 0 代表當日停牌
    before = len(df)
    df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
    if len(df) < before:
        print(f"  已過濾 {before - len(df)} 筆停牌日數據")

    df.to_csv(cache_path)
    print(f"已快取：{cache_path}（{len(df)} 筆）")
    return df


def load_all_stocks(period: str = "3y") -> dict:
    """
    下載股票池所有股票，對齊共同交易日。
    回傳 {stock_id: DataFrame}
    """
    stocks = {}
    for s in STOCK_POOL:
        try:
            stocks[s["id"]] = load_stock(s["id"], period)
        except Exception as e:
            print(f"警告：{s['id']} 下載失敗，跳過。原因：{e}")

    if not stocks:
        raise ValueError("所有股票下載失敗，請檢查網路連線")

    # 取共同交易日
    common_idx = None
    for df in stocks.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    for sid in stocks:
        stocks[sid] = stocks[sid].loc[common_idx]

    print(f"共同交易日：{len(common_idx)} 筆"
          f"（{common_idx[0].date()} ~ {common_idx[-1].date()}）")
    return stocks