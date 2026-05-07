"""
CLI 入口點：不啟動後端直接跑訓練或回測
用法：
  python main.py train --period 5y --episodes 100
  python main.py validate --period 5y
  python main.py predict --period 5y
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json

from src.engine.trainer import train, validate, predict_next
from configs.trading_config import (
    DEFAULT_PERIOD, DEFAULT_EPISODES,
    DEFAULT_VAL_DAYS, DEFAULT_INITIAL_CAP
)


def cmd_train(args):
    print(f"開始訓練：period={args.period}, episodes={args.episodes}")
    result = train(
        period          = args.period,
        episodes        = args.episodes,
        initial_capital = args.capital,
        val_days        = args.val_days,
    )
    print(f"\n訓練完成")
    print(f"  回測報酬：{result['total_return']}%")
    print(f"  買入持有：{result['bh_return']}%")
    print(f"  無風險基準：{result['risk_free_return']}%")
    print(f"  交易勝率：{result['win_rate']}%")


def cmd_validate(args):
    print(f"開始驗證：period={args.period}, val_days={args.val_days}")
    result = validate(
        period          = args.period,
        val_days        = args.val_days,
        initial_capital = args.capital,
    )
    print(f"\n驗證完成（{result['val_start']} ~ {result['val_end']}）")
    print(f"  AI 報酬：{result['total_return']}%")
    print(f"  買入持有：{result['bh_return']}%")
    print(f"  無風險基準：{result['risk_free_return']}%")
    print(f"  交易勝率：{result['win_rate']}%")
    print(f"  交易筆數：{result['n_trades']}")


def cmd_predict(args):
    print(f"預測：period={args.period}")
    result = predict_next(period=args.period)
    print(f"\n截至 {result['as_of_date']} 的持倉建議：")
    for rec in result["recommendations"]:
        print(f"  {rec['stock_name']:8s}  {rec['action']:6s}  目標 {rec['target_pct']}%"
              f"  現價 {rec['latest_price']}")
    print(f"\n  建議現金：{result['cash_pct']}%")


def main():
    parser = argparse.ArgumentParser(description="Portfolio AI CLI")
    sub    = parser.add_subparsers(dest="cmd")

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--period",   default=DEFAULT_PERIOD)
    p_train.add_argument("--episodes", type=int,   default=DEFAULT_EPISODES)
    p_train.add_argument("--capital",  type=float, default=DEFAULT_INITIAL_CAP)
    p_train.add_argument("--val-days", type=int,   default=DEFAULT_VAL_DAYS, dest="val_days")

    # validate
    p_val = sub.add_parser("validate")
    p_val.add_argument("--period",   default=DEFAULT_PERIOD)
    p_val.add_argument("--val-days", type=int,   default=DEFAULT_VAL_DAYS, dest="val_days")
    p_val.add_argument("--capital",  type=float, default=DEFAULT_INITIAL_CAP)

    # predict
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--period", default=DEFAULT_PERIOD)

    args = parser.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()