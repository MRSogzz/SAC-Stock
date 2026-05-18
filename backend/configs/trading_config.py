"""Trading and model configuration."""

STOCK_POOL = [
    {"id": "2330", "name": "TSMC", "sector": "Semiconductor"},
    {"id": "2317", "name": "Hon Hai", "sector": "Electronics"},
    {"id": "2454", "name": "MediaTek", "sector": "IC Design"},
    {"id": "2412", "name": "Chunghwa Telecom", "sector": "Telecom"},
    {"id": "2308", "name": "Delta", "sector": "Electronics"},
    {"id": "2882", "name": "Cathay Financial", "sector": "Financial"},
    {"id": "1301", "name": "Formosa Plastics", "sector": "Plastics"},
    {"id": "2002", "name": "China Steel", "sector": "Steel"},
    {"id": "2886", "name": "Mega Financial", "sector": "Financial"},
    {"id": "0050", "name": "Yuanta Taiwan 50", "sector": "ETF"},
]

N_STOCKS = len(STOCK_POOL)
N_FEATURES = 38

BENCHMARK_STOCK = "0050"
OBSERVABLE_STOCKS = [s["id"] for s in STOCK_POOL]
TRADEABLE_STOCKS = [s["id"] for s in STOCK_POOL if s["id"] != BENCHMARK_STOCK]
N_OBSERVABLE = len(OBSERVABLE_STOCKS)
N_TRADEABLE = len(TRADEABLE_STOCKS)

STATE_DIM = N_OBSERVABLE * N_FEATURES + N_TRADEABLE * 2 + 1

MAX_POSITION = 0.4
MDD_WINDOW = 20

BROKER_FEE = 0.001425
SECURITY_TAX = 0.003
MIN_FEE_LOT = 20
MIN_FEE_ODD = 1
LOT_SIZE = 1000
ODD_FILL_RATIO = 0.65

RISK_FREE_ANNUAL = 0.015
RISK_FREE_DAILY = RISK_FREE_ANNUAL / 250

SAC_LR = 3e-4
SAC_GAMMA = 0.97
SAC_TAU = 0.005
SAC_BATCH = 1024
SAC_BUFFER_SIZE = 500_000
SAC_HIDDEN = 384
SAC_ALPHA_MIN = 0.1
SAC_TARGET_ENTROPY = -2.1

TC_MULTIPLIER = 1.0
ACTION_SMOOTH_LAMBDA = 0.005
ALPHA_SIGMA = 0.02
ALPHA_SCALE = 0.2

DEFAULT_PERIOD = "6y"
DEFAULT_EPISODES = 80
DEFAULT_VAL_DAYS = 250
DEFAULT_INITIAL_CAP = 1_000_000
