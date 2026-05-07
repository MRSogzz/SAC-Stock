import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

# 修正路徑以調用 src 與 configs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import load_all_stocks 
from src.utils.finance import calc_fee, calc_mdd, calc_sharpe

# 中文顯示設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 

class PortfolioOracle:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital

    def solve(self, stocks_dict):
        # 1. 數據對齊
        df_prices = pd.DataFrame({sid: df['Close'] for sid, df in stocks_dict.items()}).dropna()
        prices = df_prices.values
        stock_ids = df_prices.columns.tolist()
        dates = df_prices.index
        n_days, n_stocks = prices.shape

        # dp[day][state]: state 0 為現金, 1~n 為各支股票
        dp = np.zeros((n_days, n_stocks + 1))
        path = np.zeros((n_days, n_stocks + 1), dtype=int)

        # 初始狀態 (Day 0)
        dp[0, 0] = self.initial_capital # 持有現金
        for j in range(n_stocks):
            p = prices[0, j]
            shares = self.initial_capital // p
            dp[0, j+1] = (shares * p) - calc_fee(shares * p, is_sell=False)

        # 動態規劃主循環 (考慮手續費與證交稅)
        for i in range(1, n_days):
            for curr_s in range(n_stocks + 1):
                # 測試從昨天所有可能的狀態轉移過來
                candidates = []
                for prev_s in range(n_stocks + 1):
                    prev_val = dp[i-1, prev_s]
                    
                    # 情況 A: 狀態不變 (續抱或持續觀望)
                    if curr_s == prev_s:
                        if curr_s == 0: # 續持現金
                            val = prev_val
                        else: # 續抱股票 (按漲跌幅計算價值)
                            val = prev_val * (prices[i, curr_s-1] / prices[i-1, curr_s-1])
                    
                    # 情況 B: 賣出股票換現金
                    elif curr_s == 0 and prev_s > 0:
                        val_at_sell = prev_val * (prices[i, prev_s-1] / prices[i-1, prev_s-1])
                        val = val_at_sell - calc_fee(val_at_sell, is_sell=True)
                    
                    # 情況 C: 持現買入股票
                    elif curr_s > 0 and prev_s == 0:
                        p = prices[i, curr_s-1]
                        shares = prev_val // p
                        val = (shares * p) - calc_fee(shares * p, is_sell=False)
                    
                    # 情況 D: 直接換股 (賣 A 買 B)
                    else:
                        # 先賣再買
                        val_after_sell = prev_val * (prices[i, prev_s-1] / prices[i-1, prev_s-1])
                        val_after_sell -= calc_fee(val_after_sell, is_sell=True)
                        p_new = prices[i, curr_s-1]
                        shares = val_after_sell // p_new
                        val = (shares * p_new) - calc_fee(shares * p_new, is_sell=False)
                    
                    candidates.append(val)
                
                dp[i, curr_s] = max(candidates)
                path[i, curr_s] = np.argmax(candidates)

        # 回溯最優路徑
        best_path = []
        curr_state = np.argmax(dp[-1, :])
        for i in range(n_days-1, -1, -1):
            best_path.append(curr_state)
            curr_state = path[i, curr_state]
        best_path.reverse()

        # 生成對帳單與資產曲線
        results = []
        last_s = -1
        for i, s in enumerate(best_path):
            asset_name = "現金" if s == 0 else stock_ids[s-1]
            val = dp[i, s]
            results.append({"Date": dates[i], "Asset": asset_name, "Value": val})
            
        return pd.DataFrame(results), stock_ids

def visualize_path(df_res, stock_ids):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # 1. 資產曲線
    ax1.plot(df_res['Date'], df_res['Value'], color='#D4AF37', linewidth=2, label='上帝組合淨值')
    ax1.set_title("9 支股票組合：歷史最優路徑(5Y) 與 資金分佈圖", fontsize=16)
    ax1.set_ylabel("帳戶總資產 (TWD)")
    ax1.grid(True, alpha=0.2)

    # 2. 資金流向分佈圖 (甘特圖樣式)
    colors = plt.cm.get_cmap('tab20', len(stock_ids) + 1)
    asset_to_color = {name: colors(i) for i, name in enumerate(["現金"] + stock_ids)}
    
    for i in range(len(df_res)-1):
        asset = df_res.iloc[i]['Asset']
        ax2.axvspan(df_res.iloc[i]['Date'], df_res.iloc[i+1]['Date'], 
                   color=asset_to_color[asset], alpha=0.8)

    # 建立圖例
    patches = [mpatches.Patch(color=asset_to_color[name], label=name) for name in ["現金"] + stock_ids]
    ax2.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), title="持有資產")
    ax2.set_yticks([])
    ax2.set_ylabel("資金配置")

    plt.tight_layout()
    plt.savefig("portfolio_oracle_path.png")
    print(f"✅ 可視化路徑圖已儲存: portfolio_oracle_path.png")

if __name__ == "__main__":
    print("🧠 正在運算上帝路徑對帳單...")
    stocks = load_all_stocks(period="5y")
    oracle = PortfolioOracle()
    df_res, s_ids = oracle.solve(stocks)
    
    # 輸出對帳單 (僅列出換股日期)
    df_res['Prev_Asset'] = df_res['Asset'].shift(1)
    trades = df_res[df_res['Asset'] != df_res['Prev_Asset']].copy()
    trades.to_csv("optimal_trades_log.csv", index=False, encoding='utf-8-sig')
    
    print(f"📝 交易對帳單已生成 (共 {len(trades)} 次切換)，請查看 optimal_trades_log.csv")
    visualize_path(df_res, s_ids)