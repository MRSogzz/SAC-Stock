"""
交易環境：PortfolioEnv v7
觀測空間：市場特徵(310) + 整張市值比例(9) + 零股市值比例(9) + 現金比例(1) = 329
動作空間：9支可交易股票的目標倉位比例（含現金為第10維，由外部決定）

核心設計：
  - 真實整張 + 零股交易，訓練與回測行為一致
  - 零股成交率基於 20 日滾動平均成交量估算
  - 升級（odd >= 1000）自動執行，無成本
  - 資金不足時：整張改用零股

v7 改動：
  - 移除交易死區（Dead Zone），讓 env 如實執行所有交易指令
  - 加入 cost_t 輸出：每步實際發生的手續費+稅金，傳給 reward 函數
  - 加入現金 MAR（無風險利率）計入 port_ret：
      r_total = Σ(w_i * r_i) + w_cash * MAR
  - 支援兩種 reward 函數（CompositeRewardV10 / LinearDownsideReward）
    透過建構子的 reward_fn 參數切換
  - _traded_this_step 旗標保留，供 walkforward.py 統計實際交易次數
"""
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from configs.trading_config import (
    N_FEATURES, LOT_SIZE,
    MIN_FEE_LOT, MIN_FEE_ODD, BROKER_FEE, SECURITY_TAX,
    ODD_FILL_RATIO, STATE_DIM,
    OBSERVABLE_STOCKS, TRADEABLE_STOCKS, BENCHMARK_STOCK,
    RISK_FREE_DAILY,
)
from src.data.processor import AVG_VOL_WINDOW
from src.environment.reward import CompositeRewardV10, LinearDownsideReward
from diagnostics import register, nan_guard

# 日化無風險利率（現金 MAR）
MAR_DAILY = 1.2 / 100 / 252   # ≈ 0.0000476


# ─── 手續費與稅金 ─────────────────────────────────────────────────────────────

def _lot_fee(amount: float) -> float:
    return float(max(math.ceil(amount * BROKER_FEE), MIN_FEE_LOT))

def _odd_fee(amount: float) -> float:
    return float(max(math.ceil(amount * BROKER_FEE), MIN_FEE_ODD))

def _tax(amount: float) -> float:
    return float(math.ceil(amount * SECURITY_TAX))

def _odd_fill(order_shares: int, avg_vol: float) -> int:
    if order_shares <= 0 or avg_vol <= 0:
        return 0
    available = avg_vol * ODD_FILL_RATIO
    fill_rate = min(1.0, order_shares / (available + 1e-8))
    return max(0, int(order_shares * fill_rate))


# ─── PortfolioEnv ─────────────────────────────────────────────────────────────

class PortfolioEnv:

    @register(
        module="Env",
        inputs={
            "features_dict":   "dict[str, pd.DataFrame]",
            "prices_dict":     "dict[str, np.ndarray]",
            "volumes_dict":    "dict[str, np.ndarray]",
            "scalers":         "dict | None",
            "initial_capital": "float",
            "reward_mode":     "str",
        },
        outputs={"return": "PortfolioEnv"},
        notes="v7：支援兩種 reward（composite/linear），輸出 cost_t，現金計入 MAR",
    )
    def __init__(
        self,
        features_dict:   dict,
        prices_dict:     dict,
        volumes_dict:    dict,
        scalers:         dict  = None,
        initial_capital: float = 1_000_000,
        reward_mode:     str   = "composite",   # "composite" | "linear"
    ):
        self.observable_ids  = OBSERVABLE_STOCKS
        self.tradeable_ids   = TRADEABLE_STOCKS
        self.benchmark_id    = BENCHMARK_STOCK
        self.n_observable    = len(self.observable_ids)
        self.n_tradeable     = len(self.tradeable_ids)
        self.initial_capital = initial_capital
        self.reward_mode     = reward_mode

        self.n_steps = min(len(v) for v in features_dict.values())

        for sid in self.observable_ids:
            f_len = len(features_dict[sid])
            p_len = len(prices_dict[sid])
            v_len = len(volumes_dict[sid])
            if not (f_len == p_len == v_len):
                raise ValueError(
                    f"[{sid}] 特徵({f_len})、價格({p_len})、"
                    f"成交量({v_len}) 長度不一致"
                )

        print(f"[PortfolioEnv v7] n_steps={self.n_steps}  "
              f"state_dim={STATE_DIM}  reward_mode={reward_mode}")

        self.features = {}
        self.scalers  = {}
        self.prices   = {}
        self.avg_vol  = {}

        for sid in self.observable_ids:
            feat = features_dict[sid].values[:self.n_steps].copy().astype(np.float64)
            feat = np.where(np.isposinf(feat),  1e6, feat)
            feat = np.where(np.isneginf(feat), -1e6, feat)
            feat = np.where(np.isnan(feat),      0.0, feat)

            if scalers and sid in scalers:
                scaler = scalers[sid]
                scaled = scaler.transform(feat)
            else:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(feat)

            self.features[sid] = np.clip(scaled, -5.0, 5.0).astype(np.float32)
            self.scalers[sid]  = scaler
            self.prices[sid]   = prices_dict[sid][:self.n_steps]

        for sid in self.tradeable_ids:
            vol_arr = volumes_dict[sid][:self.n_steps].astype(np.float64)
            avg = (pd.Series(vol_arr)
                   .rolling(AVG_VOL_WINDOW, min_periods=1)
                   .mean()
                   .values)
            self.avg_vol[sid] = avg

        self.state_dim = STATE_DIM

        # 依 reward_mode 初始化 reward 函數
        if reward_mode == "linear":
            self._reward_fn = LinearDownsideReward()
        else:
            self._reward_fn = CompositeRewardV10()

        self._traded_this_step = False

    # ── 重置 ──────────────────────────────────────────────────────────────────

    @register(
        module="Env",
        inputs={},
        outputs={"return": "np.ndarray (STATE_DIM,)"},
        notes="重置環境狀態，回傳初始觀測",
    )
    def reset(self) -> np.ndarray:
        self.step_idx          = 0
        self.capital           = float(self.initial_capital)
        self.lots_held         = np.zeros(self.n_tradeable, dtype=np.int64)
        self.odd_held          = np.zeros(self.n_tradeable, dtype=np.int64)
        self._traded_this_step = False
        self._reward_fn.reset(self.n_tradeable, self.initial_capital)
        return self._obs()

    # ── 觀測 ──────────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        T = self.step_idx
        feat_vec = np.concatenate([
            self.features[sid][T] for sid in self.observable_ids
        ])
        prices_T    = np.array([self.prices[sid][T] for sid in self.tradeable_ids],
                               dtype=np.float64)
        lot_val     = self.lots_held * LOT_SIZE * prices_T
        odd_val     = self.odd_held  * prices_T
        total_asset = self.capital + lot_val.sum() + odd_val.sum()
        lot_ratio   = lot_val / (total_asset + 1e-8)
        odd_ratio   = odd_val / (total_asset + 1e-8)
        cash_ratio  = self.capital / (total_asset + 1e-8)
        obs = np.concatenate([feat_vec, lot_ratio, odd_ratio, [cash_ratio]])
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)

    # ── 執行一步 ──────────────────────────────────────────────────────────────

    @nan_guard()
    @register(
        module="Env",
        inputs={"action": "np.ndarray (N_TRADEABLE,)"},
        outputs={
            "obs":    "np.ndarray (STATE_DIM,)",
            "reward": "float",
            "done":   "bool",
        },
        notes="v7：無死區，輸出 cost_t，現金計入 MAR",
    )
    def step(self, action: np.ndarray):
        T  = self.step_idx
        T1 = T + 1

        self._traded_this_step = False

        prices_T  = np.array([self.prices[sid][T]  for sid in self.tradeable_ids],
                              dtype=np.float64)
        prices_T1 = np.array([self.prices[sid][T1] for sid in self.tradeable_ids],
                              dtype=np.float64)
        avg_vol_T = np.array([self.avg_vol[sid][T] for sid in self.tradeable_ids],
                              dtype=np.float64)

        benchmark_ret = float(
            self.prices[self.benchmark_id][T1] /
            (self.prices[self.benchmark_id][T] + 1e-8) - 1.0
        )

        lot_val_T   = self.lots_held * LOT_SIZE * prices_T
        odd_val_T   = self.odd_held  * prices_T
        total_asset = self.capital + lot_val_T.sum() + odd_val_T.sum()
        total_T_pre = float(total_asset)

        # 現金比例（用於計算 MAR 收益）
        cash_ratio_T = self.capital / (total_asset + 1e-8)

        target        = np.clip(action.astype(np.float64), 0.0, 1.0)
        target_value  = total_asset * target
        target_shares = target_value / (prices_T + 1e-8)
        target_lots   = (target_shares // LOT_SIZE).astype(np.int64)
        target_odd    = (target_shares  % LOT_SIZE).astype(np.int64)

        tc_cost = 0.0   # v7：每步實際手續費+稅金，傳給 reward

        # ── 賣出 ──────────────────────────────────────────────────────────
        for j in range(self.n_tradeable):
            price   = float(prices_T[j])
            avg_vol = float(avg_vol_T[j])

            sell_lots = max(0, int(self.lots_held[j]) - int(target_lots[j]))
            if sell_lots > 0:
                gross    = sell_lots * LOT_SIZE * price
                fee      = _lot_fee(gross)
                tax      = _tax(gross)
                proceeds = gross - fee - tax
                self.lots_held[j]      -= sell_lots
                self.capital           += proceeds
                tc_cost                += fee + tax
                self._traded_this_step  = True

            sell_odd = max(0, int(self.odd_held[j]) - int(target_odd[j]))
            if sell_odd > 0:
                filled = _odd_fill(sell_odd, avg_vol)
                if filled > 0:
                    gross    = filled * price
                    fee      = _odd_fee(gross)
                    tax      = _tax(gross)
                    proceeds = gross - fee - tax
                    self.odd_held[j]       -= filled
                    self.capital           += proceeds
                    tc_cost                += fee + tax
                    self._traded_this_step  = True

        # ── 買入 ──────────────────────────────────────────────────────────
        for j in range(self.n_tradeable):
            price   = float(prices_T[j])
            avg_vol = float(avg_vol_T[j])

            buy_lots = max(0, int(target_lots[j]) - int(self.lots_held[j]))
            if buy_lots > 0:
                cost_per_lot = LOT_SIZE * price + _lot_fee(LOT_SIZE * price)
                affordable   = int(self.capital // cost_per_lot)
                buy_lots     = min(buy_lots, affordable)

                if buy_lots > 0:
                    gross      = buy_lots * LOT_SIZE * price
                    fee        = _lot_fee(gross)
                    self.lots_held[j]      += buy_lots
                    self.capital           -= (gross + fee)
                    tc_cost                += fee
                    self._traded_this_step  = True
                else:
                    residual_budget = self.capital * 0.8
                    extra_odd = min(
                        int(residual_budget / (price + 1e-8)),
                        LOT_SIZE - 1
                    )
                    if extra_odd > 0:
                        filled = _odd_fill(extra_odd, avg_vol)
                        if filled > 0:
                            gross      = filled * price
                            fee        = _odd_fee(gross)
                            total_cost = gross + fee
                            if total_cost <= self.capital:
                                self.odd_held[j]       += filled
                                self.capital           -= total_cost
                                tc_cost                += fee
                                self._traded_this_step  = True

            buy_odd = max(0, int(target_odd[j]) - int(self.odd_held[j]))
            if buy_odd > 0:
                filled = _odd_fill(buy_odd, avg_vol)
                if filled > 0:
                    gross      = filled * price
                    fee        = _odd_fee(gross)
                    total_cost = gross + fee
                    if total_cost <= self.capital:
                        self.odd_held[j]       += filled
                        self.capital           -= total_cost
                        tc_cost                += fee
                        self._traded_this_step  = True

        # ── 自動升級 ──────────────────────────────────────────────────────
        for j in range(self.n_tradeable):
            if self.odd_held[j] >= LOT_SIZE:
                upgrade           = self.odd_held[j] // LOT_SIZE
                self.lots_held[j] += upgrade
                self.odd_held[j]  %= LOT_SIZE

        # ── 計算 reward ───────────────────────────────────────────────────
        lot_val_T1   = self.lots_held * LOT_SIZE * prices_T1
        odd_val_T1   = self.odd_held  * prices_T1
        total_T1_pre = float(self.capital + lot_val_T1.sum() + odd_val_T1.sum())

        # 交易後實際持倉權重（用 T 價格）
        lot_val_exec = self.lots_held * LOT_SIZE * prices_T
        odd_val_exec = self.odd_held  * prices_T
        total_exec   = self.capital + lot_val_exec.sum() + odd_val_exec.sum()
        exec_w       = (lot_val_exec + odd_val_exec) / (total_exec + 1e-8)

        # v7：port_ret 加入現金 MAR
        # r_total = Σ(w_i * r_i) + w_cash * MAR
        stock_ret    = float((exec_w * (prices_T1 / (prices_T + 1e-8) - 1)).sum())
        cash_w_exec  = self.capital / (total_exec + 1e-8)
        port_ret     = stock_ret + cash_w_exec * MAR_DAILY

        odd_ratio    = float(odd_val_exec.sum() / (total_exec + 1e-8))

        reward = self._reward_fn.compute(
            total_T_pre   = total_T_pre,
            total_T1_pre  = total_T1_pre,
            odd_ratio     = odd_ratio,
            port_ret      = port_ret,
            benchmark_ret = benchmark_ret,
            target        = target,
            cost_t        = tc_cost,    # v7：傳入實際交易成本
        )

        self.step_idx = T1
        done = self.step_idx >= self.n_steps - 1

        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        return self._obs(), float(reward), done

    # ── 工具 ──────────────────────────────────────────────────────────────────

    @register(
        module="Env",
        inputs={},
        outputs={"return": "float"},
        notes="回傳當前總資產 / 初始資金（從實際持倉即時計算）",
    )
    def portfolio_value(self) -> float:
        T        = min(self.step_idx, self.n_steps - 1)
        prices_T = np.array(
            [self.prices[sid][T] for sid in self.tradeable_ids],
            dtype=np.float64,
        )
        lot_val = self.lots_held * LOT_SIZE * prices_T
        odd_val = self.odd_held  * prices_T
        total   = self.capital + lot_val.sum() + odd_val.sum()
        return float(total / self.initial_capital)