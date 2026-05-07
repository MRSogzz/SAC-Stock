import pandas as pd
import numpy as np
import os
import sys

# 修正路徑以調用你的 src 模組與 configs
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from src.data.loader import load_all_stocks 
    from configs.trading_config import STOCK_POOL
except ImportError:
    print("❌ 錯誤：找不到相關模組。請確保在 stock-ai 專案根目錄或 backend 目錄下執行。")
    sys.exit(1)

class PortfolioOracle:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.fee_rate = 0.001425  # 台灣券商手續費率 (0.1425%)
        self.tax_rate = 0.003     # 證交稅率 (0.3%)
        self.min_fee = 20         # 手續費低消 20 元

    def calc_costs(self, amount, is_sell=False):
        """ 精確計算台灣交易成本，防止單筆金額異常導致的崩潰 """
        if not np.isfinite(amount) or amount <= 0:
            return 0
        fee = max(self.min_fee, int(amount * self.fee_rate))
        tax = int(amount * self.tax_rate) if is_sell else 0
        return fee + tax

    def solve(self, stocks_dict):
        # 1. 建立初步 DataFrame 並對齊日期
        temp_df = pd.DataFrame({sid: df['Close'] for sid, df in stocks_dict.items()})
        
        # 【診斷區】找出肉眼看不見的 0 或 NaN
        bad_mask = (temp_df <= 0).any(axis=1) | temp_df.isna().any(axis=1)
        bad_data = temp_df[bad_mask]
        if not bad_data.empty:
            print(f"\n🔍 偵測到 {len(bad_data)} 筆異常數據（含 0 或 NaN），已從回測中剔除。")
            print("部分異常日期範例：")
            print(bad_data.head(5))
            print("-" * 30)

        # 2. 嚴格清洗數據：確保所有參與計算的價格 > 0
        df_prices = temp_df[(temp_df > 0).all(axis=1)].dropna()
        
        if df_prices.empty:
            print("❌ 錯誤：數據清洗後為空，請檢查原始資料來源。")
            return pd.DataFrame(), self.initial_capital

        prices = df_prices.values
        stock_ids = df_prices.columns.tolist()
        dates = df_prices.index
        n_days, n_stocks = prices.shape

        # 3. 動態規劃 (DP) 核心矩陣
        dp = np.zeros((n_days, n_stocks + 1))
        path = np.zeros((n_days, n_stocks + 1), dtype=int)

        # 初始狀態 (Day 0)
        dp[0, 0] = self.initial_capital
        for j in range(n_stocks):
            p = prices[0, j]
            # 買入公式：可用資金 / (價格 * (1 + 手續費率))
            shares = self.initial_capital // (p * (1 + self.fee_rate))
            amt = shares * p
            dp[0, j+1] = amt - self.calc_costs(amt, is_sell=False)

        # 4. 尋找最大化報酬路徑
        for i in range(1, n_days):
            for curr_s in range(n_stocks + 1):
                candidates = []
                for prev_s in range(n_stocks + 1):
                    prev_val = dp[i-1, prev_s]
                    
                    if curr_s == prev_s: # 狀態不變 (持現或續抱)
                        val = prev_val if curr_s == 0 else prev_val * (prices[i, curr_s-1] / prices[i-1, curr_s-1])
                    elif curr_s == 0: # 賣出標的換成現金
                        v_at_sell = prev_val * (prices[i, prev_s-1] / prices[i-1, prev_s-1])
                        val = v_at_sell - self.calc_costs(v_at_sell, is_sell=True)
                    elif curr_s > 0 and prev_s == 0: # 持有現金買入標的
                        p_buy = prices[i, curr_s-1]
                        denom = p_buy * (1 + self.fee_rate)
                        sh = prev_val // denom if denom > 0 else 0
                        amt = sh * p_buy
                        val = amt - self.calc_costs(amt, is_sell=False)
                    else: # 換股 (先賣 A 再買 B)
                        v_at_sell = prev_val * (prices[i, prev_s-1] / prices[i-1, prev_s-1])
                        v_cash = v_at_sell - self.calc_costs(v_at_sell, is_sell=True)
                        p_buy = prices[i, curr_s-1]
                        denom = p_buy * (1 + self.fee_rate)
                        sh = v_cash // denom if denom > 0 else 0
                        amt = sh * p_buy
                        val = amt - self.calc_costs(amt, is_sell=False)
                    
                    candidates.append(val if np.isfinite(val) else -1)
                
                dp[i, curr_s] = max(candidates)
                path[i, curr_s] = np.argmax(candidates)

        # 5. 回溯最優決策序列
        best_states = []
        curr = np.argmax(dp[-1, :])
        for i in range(n_days-1, -1, -1):
            best_states.append(curr)
            curr = path[i, curr]
        best_states.reverse()

        # 6. 依照最優路徑生成「對帳單明細」
        trade_log = []
        current_shares = 0
        last_buy_total_cost = 0

        for i in range(n_days):
            prev_s = best_states[i-1] if i > 0 else 0
            curr_s = best_states[i]
            
            if curr_s != prev_s:
                date_str = dates[i].strftime('%Y-%m-%d')
                
                # 處理賣出紀錄
                if prev_s > 0:
                    sid = stock_ids[prev_s-1]
                    p_sell = prices[i, prev_s-1]
                    amt = current_shares * p_sell
                    cost = self.calc_costs(amt, is_sell=True)
                    net_proceeds = amt - cost
                    profit = net_proceeds - last_buy_total_cost
                    
                    trade_log.append({
                        "日期": date_str, "股票": sid, "操作": "賣出",
                        "價格": f"${p_sell:,.1f}", "股數": f"{current_shares:,}",
                        "金額": f"${amt:,.0f}", "手續費/稅": f"${cost:,.0f}",
                        "損益": f"{profit:+,.0f}", "倉位": "0%"
                    })

                # 處理買入紀錄
                if curr_s > 0:
                    sid = stock_ids[curr_s-1]
                    p_buy = prices[i, curr_s-1]
                    # 獲取前一天的現金餘額
                    cash_available = dp[i-1, prev_s] if i > 0 else self.initial_capital
                    denom = p_buy * (1 + self.fee_rate)
                    current_shares = int(cash_available // denom) if denom > 0 else 0
                    amt = current_shares * p_buy
                    cost = self.calc_costs(amt, is_sell=False)
                    last_buy_total_cost = amt + cost
                    
                    trade_log.append({
                        "日期": date_str, "股票": sid, "操作": "買入",
                        "價格": f"${p_buy:,.1f}", "股數": f"{current_shares:,}",
                        "金額": f"${amt:,.0f}", "手續費/稅": f"${cost:,.0f}",
                        "損益": "—", "倉位": "100%"
                    })

        return pd.DataFrame(trade_log), dp[-1].max()

def export_to_html(df, final_val):
    """ 生成符合用戶要求的互動式對帳單 HTML """
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: "Microsoft JhengHei", sans-serif; background: #121212; color: #e0e0e0; padding: 30px; }}
            h2 {{ color: #ffcc00; border-bottom: 2px solid #ffcc00; padding-bottom: 10px; }}
            .summary {{ background: #1e1e1e; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #ffcc00; font-size: 1.1em; }}
            table {{ width: 100%; border-collapse: collapse; background: #222; border: 1px solid #444; }}
            th {{ background: #333; color: #ffcc00; padding: 12px; border: 1px solid #444; position: sticky; top: 0; }}
            td {{ padding: 10px; border: 1px solid #444; text-align: center; }}
            tr:nth-child(even) {{ background: #2a2a2a; }}
            tr:hover {{ background: #383838; }}
            .profit-pos {{ color: #ff4d4d; font-weight: bold; }} /* 台灣習慣：紅盈 */
            .profit-neg {{ color: #33ff33; font-weight: bold; }} /* 台灣習慣：綠虧 */
            .buy {{ color: #ffcc00; }}
            .sell {{ color: #00d4ff; }}
        </style>
    </head>
    <body>
        <h2>📈 上帝視角：回測交易明細報告 (5Y)</h2>
        <div class="summary">
            <b>初始投資:</b> $100,000 <br>
            <b>理論最終資產:</b> ${final_val:,.0f} <br>
            <b>交易成本設定:</b> 手續費 0.1425% (低消 $20) / 證交稅 0.3%
        </div>
        {df.to_html(index=False, escape=False)}
    </body>
    <script>
        document.querySelectorAll('td').forEach(td => {{
            const t = td.innerText;
            if (t.includes('+')) td.className = 'profit-pos';
            if (t.includes('-')) td.className = 'profit-neg';
            if (t === '買入') td.className = 'buy';
            if (t === '賣出') td.className = 'sell';
        }});
    </script>
    </html>
    """
    output_path = os.path.join(current_dir, "trade_details_full.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path

if __name__ == "__main__":
    print("🚀 啟動上帝視角策略回測...")
    
    # 1. 載入資料 (自動調用你的 loader)
    try:
        stocks_data = load_all_stocks(period="5y")
        
        # 2. 執行運算
        oracle = PortfolioOracle(initial_capital=100000)
        df_trades, final_value = oracle.solve(stocks_data)
        
        # 3. 輸出報告
        if not df_trades.empty:
            report_file = export_to_html(df_trades, final_value)
            print(f"\n✅ 回測完成！")
            print(f"💰 最終資產極限: ${final_value:,.0f}")
            print(f"📄 報告已生成，請用瀏覽器開啟: {report_file}")
        else:
            print("⚠️ 未能產生交易紀錄，請檢查 STOCK_POOL 是否有誤。")
            
    except Exception as e:
        print(f"❌ 執行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()