"""
交易設定：股票池、手續費、回測參數、模型超參數
"""

# ─── 股票池 ──────────────────────────────────────────────────────────────────
STOCK_POOL = [
    {"id": "2330", "name": "台積電",    "sector": "半導體"},
    {"id": "2317", "name": "鴻海",      "sector": "電子製造"},
    {"id": "2454", "name": "聯發科",    "sector": "IC設計"},
    {"id": "2412", "name": "中華電",    "sector": "電信"},
    {"id": "2308", "name": "台達電",    "sector": "電源"},
    {"id": "2882", "name": "國泰金",    "sector": "金融"},
    {"id": "1301", "name": "台塑",      "sector": "石化"},
    {"id": "2002", "name": "中鋼",      "sector": "鋼鐵"},
    {"id": "2886", "name": "兆豐金",    "sector": "金融"},
    {"id": "0050", "name": "元大台灣50", "sector": "ETF"},
]

N_STOCKS   = len(STOCK_POOL)
N_FEATURES = 35    # 27 → 35

# ─── 大盤基準與交易限制 ───────────────────────────────────────────────────────
BENCHMARK_STOCK   = "0050"
OBSERVABLE_STOCKS = [s["id"] for s in STOCK_POOL]
TRADEABLE_STOCKS  = [s["id"] for s in STOCK_POOL if s["id"] != BENCHMARK_STOCK]
N_OBSERVABLE      = len(OBSERVABLE_STOCKS)   # 10
N_TRADEABLE       = len(TRADEABLE_STOCKS)    # 9

STATE_DIM  = N_OBSERVABLE * N_FEATURES + N_TRADEABLE * 2 + 1  # = 369

# ─── 倉位限制 ─────────────────────────────────────────────────────────────────
MAX_POSITION = 0.4

# ─── 機會成本懲罰 ─────────────────────────────────────────────────────────────
MDD_WINDOW = 20    # 滑動回撤窗口（交易日）

# ─── 台股交易成本 ─────────────────────────────────────────────────────────────
BROKER_FEE       = 0.001425
SECURITY_TAX     = 0.003
MIN_FEE_LOT      = 20       # 整張最低手續費（元），無條件進位
MIN_FEE_ODD      = 1        # 零股最低手續費（元），無條件進位
LOT_SIZE         = 1000     # 一張股數

# ─── 零股交易參數 ─────────────────────────────────────────────────────────────
ODD_FILL_RATIO   = 0.65     # 零股成交率基準（買入與賣出均適用）
                             # 生產環境應改為基於歷史成交率的時變參數

# ─── 無風險利率 ───────────────────────────────────────────────────────────────
RISK_FREE_ANNUAL = 0.015
RISK_FREE_DAILY  = RISK_FREE_ANNUAL / 250

# ─── SAC 超參數 ───────────────────────────────────────────────────────────────
SAC_LR           = 3e-4
SAC_GAMMA        = 0.97          # 從 0.99 降低：日線交易中遠期報酬影響有限，
                                 # 0.99 會放大 Q 值誤差累積導致發散
SAC_TAU          = 0.005
SAC_BATCH        = 1024
SAC_BUFFER_SIZE  = 500_000
SAC_HIDDEN       = 384
SAC_ALPHA_MIN    = 0.1

# Dirichlet + Beta 的 entropy 目標：
# 當 alpha ≈ 1.0 時 Dirichlet entropy ≈ 2.2（接近均勻分布），
# SAC 若目標設太高（如 -2.0）會持續鼓勵高 entropy，
# 導致 Actor 永遠維持均分、不敢分化股票選擇。
# 設為 -5.0 讓 SAC 允許 Actor 學習集中倉位。
SAC_TARGET_ENTROPY = -2.1

TC_MULTIPLIER        = 1.0    # 從 2.0 降低：訓練期手續費感知回歸真實
ACTION_SMOOTH_LAMBDA = 0.005  # 換倉懲罰係數
ALPHA_SIGMA          = 0.02   # tanh 標準化參數
ALPHA_SCALE          = 0.2   # alpha reward 縮放（從 0.01 提升，與 pnl 量級相近）

# ─── 訓練預設值 ───────────────────────────────────────────────────────────────
DEFAULT_PERIOD      = "6y"
DEFAULT_EPISODES    = 80
DEFAULT_VAL_DAYS    = 250
DEFAULT_INITIAL_CAP = 1_000_000