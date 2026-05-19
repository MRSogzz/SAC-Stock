export const API = "http://localhost:8000";

export const PERIOD_OPTIONS = [
  { value: "1y", label: "1 年" },
  { value: "2y", label: "2 年" },
  { value: "3y", label: "3 年" },
  { value: "5y", label: "5 年" },
  { value: "6y", label: "6 年（建議）" },
];

export const REGIME_LABEL = {
  bull: "牛市",
  bear: "熊市",
  sideways: "盤整",
};

export const REGIME_COLOR = {
  bull: "var(--color-text-success)",
  bear: "var(--color-text-danger)",
  sideways: "var(--color-text-secondary)",
};

export const RUN_DESC = {
  D: "LogitDelta + Linear",
};

export const STOCK_POOL = [
  { id: "2330", name: "台積電",     sector: "半導體"   },
  { id: "2317", name: "鴻海",       sector: "電子製造" },
  { id: "2454", name: "聯發科",     sector: "IC設計"   },
  { id: "2412", name: "中華電",     sector: "電信"     },
  { id: "2308", name: "台達電",     sector: "電源"     },
  { id: "2882", name: "國泰金",     sector: "金融"     },
  { id: "1301", name: "台塑",       sector: "石化"     },
  { id: "2002", name: "中鋼",       sector: "鋼鐵"     },
  { id: "2886", name: "兆豐金",     sector: "金融"     },
  { id: "0050", name: "元大台灣50", sector: "ETF"      },
];

export const CAPITAL_OPTIONS = [
  10000, 30000, 50000, 100000, 500000, 1000000, 3000000,
];

export const WALK_FORWARD_WINDOWS = [
  { w: 1, train: "2021-04 ~ 2024-04", val: "2024-04 ~ 2025-04", regime: "sideways" },
  { w: 2, train: "2022-04 ~ 2025-04", val: "2025-04 ~ 2026-04", regime: "bear"     },
];