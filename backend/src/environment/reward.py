"""
獎勵函數：兩版並存

CompositeRewardV10（Run A/B 使用）：
  - 舊版 Composite Reward，含 pnl + alpha - MDD - smooth - odd - HHI
  - raw * 5 + clip[-1,1]
  - ACTION_SMOOTH_LAMBDA=0.005，L1 懲罰
  - prev_target 每步更新

LinearDownsideReward（Run C/D 使用）：
  - 裸線性，無 tanh、無 clip、無動態 RMS
  - R_t = (PnL_t - Cost_t + Alpha_t - Downside) / c
  - c 在前 500 步 warmup 後鎖定為 max(3 * σ_init, 0.1)；若 c 等於 fallback（C_FLOOR）則拒絕啟動
  - Downside = λ * max(0, MAR - r_portfolio)，線性，λ=0.1
  - MAR = 1.2% / 252（日化台幣無風險利率）
  - 移除：odd_penalty、smooth L1、MDD、clip、tanh
  - Alpha 項保留（tanh 壓縮移除，改為線性超額報酬）
"""
import numpy as np
from collections import deque

from configs.trading_config import (
    MDD_WINDOW, ACTION_SMOOTH_LAMBDA, ALPHA_SIGMA, ALPHA_SCALE,
    RISK_FREE_DAILY,
)
from diagnostics import register

# ── CompositeRewardV10 常數 ───────────────────────────────────────────────────
MDD_FLOOR        = 0.05
ODD_PENALTY_RATE = 0.0005
HHI_THRESHOLD    = 0.50
HHI_SCALE        = 0.02

# ── LinearDownsideReward 常數 ─────────────────────────────────────────────────
MAR              = 1.2 / 100 / 252   # 日化無風險利率 ≈ 0.0000476
DOWNSIDE_LAMBDA  = 0.1               # 線性下行懲罰係數（λ=0.1，配合 c=3σ 保持相對比例）
WARMUP_STEPS     = 500               # warmup 步數，收集完整分子 std
C_MULTIPLIER     = 3.0               # c = max(C_MULTIPLIER * σ_init, C_FLOOR)
C_FLOOR          = 0.1               # c 的最低值（熔斷門檻：低於此值拒絕啟動）
C_MIN            = 0.005             # warmup 期間暫用預設值（非鎖定值）


# ═══════════════════════════════════════════════════════════════════════════════
# Run A/B：CompositeRewardV10（舊版，維持不變）
# ═══════════════════════════════════════════════════════════════════════════════

class CompositeRewardV10:
    """
    舊版 Composite Reward（Run A/B 使用）。
    與之前的 CompositeReward v10 完全相同，重命名以區分。
    """

    @register(
        module="Env",
        inputs={},
        outputs={"return": "CompositeRewardV10"},
        notes="舊版 Composite Reward（Run A/B）",
    )
    def __init__(self):
        self.total_asset    = 1.0
        self.portfolio_hist = deque(maxlen=MDD_WINDOW)
        self.prev_target    = None

    def reset(self, n_stocks: int, initial_capital: float):
        self.total_asset    = initial_capital
        self.portfolio_hist = deque(maxlen=MDD_WINDOW)
        self.portfolio_hist.append(initial_capital)
        self.prev_target    = np.zeros(n_stocks)

    def compute(
        self,
        total_T_pre:   float,
        total_T1_pre:  float,
        odd_ratio:     float,
        port_ret:      float,
        benchmark_ret: float,
        target:        np.ndarray,
        cost_t:        float = 0.0,   # 相容性參數，舊版不使用
    ) -> float:

        # 1. 對數單步報酬
        pnl = float(np.log(max(total_T1_pre, 1e-8) / max(total_T_pre, 1e-8)))

        # 2. MDD 窗口更新
        self.total_asset = total_T1_pre
        self.portfolio_hist.append(total_T1_pre)

        # 3. 滑動 MDD 懲罰
        window_peak = max(self.portfolio_hist)
        drawdown    = max(0.0, (window_peak - total_T1_pre) / (window_peak + 1e-8))
        dd_penalty  = 0.1 * max(0.0, drawdown - MDD_FLOOR)

        # 4. Alpha 獎勵
        excess       = port_ret - benchmark_ret
        alpha_reward = float(np.tanh(excess / ALPHA_SIGMA)) * ALPHA_SCALE

        # 5. Action Smoothing（L1，prev_target 每步更新）
        if self.prev_target is not None:
            diff           = target - self.prev_target
            smooth_penalty = ACTION_SMOOTH_LAMBDA * float(np.abs(diff).sum())
        else:
            smooth_penalty = 0.0
        self.prev_target = target.copy()

        # 6. 零股流動性懲罰
        odd_penalty = ODD_PENALTY_RATE * odd_ratio

        # 7. HHI 集中度懲罰
        active = target[target > 0.01]
        if len(active) > 1:
            hhi = float((active ** 2).sum())
        elif len(active) == 1:
            hhi = 1.0
        else:
            hhi = 0.0
        hhi_penalty = HHI_SCALE * max(0.0, hhi - HHI_THRESHOLD)

        # 8. 合計並縮放
        raw = (pnl - dd_penalty + alpha_reward
               - smooth_penalty - odd_penalty - hhi_penalty)
        return float(np.clip(raw * 5, -1.0, 1.0))


# 預設別名，讓舊程式碼繼續工作
CompositeReward = CompositeRewardV10


# ═══════════════════════════════════════════════════════════════════════════════
# Run C/D：LinearDownsideReward（新版線性獎勵）
# ═══════════════════════════════════════════════════════════════════════════════

class LinearDownsideReward:
    """
    線性常數縮放獎勵（Run C/D 使用）。

    公式：R_t = (PnL_t - Cost_t + Alpha_t - Downside) / c

    設計原則：
      - 裸線性：無 tanh、無 clip、無動態 RMS
      - c 在 warmup 500 步後鎖定，全程不變
      - Downside = λ * max(0, MAR - r_portfolio)，線性，梯度恆定
      - 移除：odd_penalty、smooth L1、MDD、HHI 懲罰
      - Alpha 項保留但改為線性（不做 tanh 壓縮）

    禁用清單（硬性）：
      ❌ clip / tanh / softsign
      ❌ 動態 RMS / running normalization
      ❌ MDD 直接 per-step 懲罰
      ❌ Downside 二次方（必須線性）
      ❌ weight center / HHI 懲罰
    """

    @register(
        module="Env",
        inputs={},
        outputs={"return": "LinearDownsideReward"},
        notes="線性獎勵（Run C/D），warmup 500 步後鎖定縮放常數 c",
    )
    def __init__(self):
        self.total_asset  = 1.0
        self.c            = None      # warmup 後鎖定
        self._warmup_pnls = []        # warmup 期間收集完整分子
        self._warmed_up   = False
        self.just_locked  = False     # c 剛鎖定時為 True，train_window 讀取後重置

    def reset(self, n_stocks: int, initial_capital: float):
        self.total_asset = initial_capital
        # warmup 狀態跨 episode 保留，直到 c 鎖定
        # 若 c 已鎖定則不重置 _warmup_pnls

    def compute(
        self,
        total_T_pre:   float,
        total_T1_pre:  float,
        odd_ratio:     float,
        port_ret:      float,
        benchmark_ret: float,
        target:        np.ndarray,
        cost_t:        float = 0.0,
    ) -> float:

        # 1. 對數單步 PnL
        pnl = float(np.log(max(total_T1_pre, 1e-8) / max(total_T_pre, 1e-8)))
        self.total_asset = total_T1_pre

        # 2. Alpha：線性超額報酬（不做 tanh 壓縮）
        excess      = port_ret - benchmark_ret
        alpha_t     = ALPHA_SCALE * excess

        # 3. Downside：線性下行懲罰（λ=0.1，配合 c=3σ 保持獎懲相對比例）
        downside = DOWNSIDE_LAMBDA * max(0.0, MAR - port_ret)

        # 4. 成本信號：成本佔比，量級和 pnl 一致
        cost_signal = cost_t / (total_T_pre + 1e-8)

        # 5. 完整分子（warmup 收集此值的 std，確保 c 對實際獎勵尺度準確）
        raw = pnl + alpha_t - downside - cost_signal

        # 6. Warmup：收集前 500 步的完整分子，鎖定 c
        if not self._warmed_up:
            self._warmup_pnls.append(raw)
            if len(self._warmup_pnls) >= WARMUP_STEPS:
                sigma_init = float(np.std(self._warmup_pnls))
                c_locked   = C_MULTIPLIER * sigma_init   # c = 3 * σ_init

                # 熔斷機制：若 c_locked < C_FLOOR 代表 warmup 期間幾乎不動作，
                # reward 尺度無法估計，拒絕鎖定並重置收集
                if c_locked < C_FLOOR:
                    print(f"  [warmup 熔斷] c_locked={c_locked:.6f} < C_FLOOR={C_FLOOR}，"
                          f"sigma_init={sigma_init:.6f}，重置收集，延遲鎖定")
                    self._warmup_pnls = []   # 重置，繼續收集
                else:
                    self.c           = c_locked
                    self._warmed_up  = True
                    self.just_locked = True   # 通知 train_window 清空 buffer
                    print(f"  [warmup 鎖定] sigma_init={sigma_init:.6f}，"
                          f"c={self.c:.6f}（={C_MULTIPLIER}×σ）")
            # warmup 期間（含熔斷後重新收集）使用預設 c，避免除零
            c = C_MIN
        else:
            c = self.c

        return float(raw / c)

    @property
    def is_warmed_up(self) -> bool:
        return self._warmed_up

    @property
    def scaling_constant(self) -> float:
        return self.c if self.c is not None else C_MIN

    @property
    def c_locked(self) -> bool:
        """c 是否已鎖定（warmup 完成且通過熔斷檢查）。"""
        return self._warmed_up