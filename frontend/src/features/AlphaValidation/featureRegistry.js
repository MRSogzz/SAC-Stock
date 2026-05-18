/**
 * 所有 Tier 1 特徵的定義，對應 processor.py 的 compute_features()
 * 用於前端特徵選取 UI 與自動產生 compute_fn 程式碼
 */

export const FEATURE_GROUPS = [
  {
    id: "return",
    label: "報酬率",
    color: "#1D9E75",
    features: [
      { id: "ret_3",  label: "ret_3",  desc: "3日收盤報酬率",  code: "c.pct_change(3)" },
      { id: "ret_5",  label: "ret_5",  desc: "5日收盤報酬率",  code: "c.pct_change(5)" },
      { id: "ret_10", label: "ret_10", desc: "10日收盤報酬率", code: "c.pct_change(10)" },
      { id: "ret_20", label: "ret_20", desc: "20日收盤報酬率", code: "c.pct_change(20)" },
      { id: "ret_60", label: "ret_60", desc: "60日收盤報酬率", code: "c.pct_change(60)" },
    ],
  },
  {
    id: "volatility",
    label: "波動率",
    color: "#378ADD",
    features: [
      { id: "vol_5",  label: "vol_5",  desc: "5日日報酬標準差",  code: "c.pct_change().rolling(5).std()" },
      { id: "vol_10", label: "vol_10", desc: "10日日報酬標準差", code: "c.pct_change().rolling(10).std()" },
      { id: "vol_20", label: "vol_20", desc: "20日日報酬標準差", code: "c.pct_change().rolling(20).std()" },
    ],
  },
  {
    id: "candle",
    label: "K線結構",
    color: "#9B59B6",
    features: [
      { id: "body",       label: "body",       desc: "實體比例 (Close-Open)/Open",          code: "(c - o) / (o + 1e-8)" },
      { id: "upper_wick", label: "upper_wick", desc: "上影線比例 (High-max(C,O))/(H-L)",    code: "(h - c.combine(o, max)) / (h - l + 1e-8)" },
      { id: "lower_wick", label: "lower_wick", desc: "下影線比例 (min(C,O)-Low)/(H-L)",     code: "(c.combine(o, min) - l) / (h - l + 1e-8)" },
      { id: "hl_range",   label: "hl_range",   desc: "高低振幅 (High-Low)/Close",           code: "(h - l) / (c + 1e-8)" },
    ],
  },
  {
    id: "volume",
    label: "成交量",
    color: "#E67E22",
    features: [
      { id: "vol_ratio_5",   label: "vol_ratio_5",   desc: "5日量比 Volume/MA5(Volume)",    code: "v / (v.rolling(5).mean() + 1e-8)" },
      { id: "vol_ratio_20",  label: "vol_ratio_20",  desc: "20日量比 Volume/MA20(Volume)",  code: "v / (v.rolling(20).mean() + 1e-8)" },
      { id: "vol_change",    label: "vol_change",    desc: "成交量日變化率",                 code: "v.pct_change()" },
      { id: "volume_impulse",label: "volume_impulse",desc: "量能衝擊（量比偏離3日均值/標準差）",code: null },
    ],
  },
  {
    id: "position",
    label: "區間位置",
    color: "#16A085",
    features: [
      { id: "pos_10",  label: "pos_10",  desc: "10日內高低區間位置 [0,1]", code: "(c - l.rolling(10).min()) / (h.rolling(10).max() - l.rolling(10).min() + 1e-8)" },
      { id: "pos_20",  label: "pos_20",  desc: "20日內高低區間位置 [0,1]", code: "(c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-8)" },
      { id: "pos_60",  label: "pos_60",  desc: "60日內高低區間位置 [0,1]", code: "(c - l.rolling(60).min()) / (h.rolling(60).max() - l.rolling(60).min() + 1e-8)" },
    ],
  },
  {
    id: "momentum",
    label: "動能指標",
    color: "#C0392B",
    features: [
      { id: "rsi_centered", label: "rsi_centered", desc: "RSI(14) − 50，以 0 為中心",      code: "(_calc_rsi(c, 14) - 50)" },
      { id: "rsi_slope",    label: "rsi_slope",    desc: "RSI(14) 3日斜率",                code: "_calc_rsi(c, 14).diff(3)" },
      { id: "adx_14",       label: "adx_14",       desc: "ADX(14) 趨勢強度",               code: null },
      { id: "atr_change",   label: "atr_change",   desc: "ATR(14) 5日變化率",              code: null },
    ],
  },
  {
    id: "mean_reversion",
    label: "均值回歸",
    color: "#2980B9",
    features: [
      { id: "z_score_20",   label: "z_score_20",   desc: "(Close − MA20) / Std20",         code: "(c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-8)" },
      { id: "ratio_20_60",  label: "ratio_20_60",  desc: "(MA20 − MA60) / MA60 乖離率",    code: "(c.rolling(20).mean() - c.rolling(60).mean()) / (c.rolling(60).mean() + 1e-8)" },
      { id: "bb_position",  label: "bb_position",  desc: "布林通道位置 (C−下軌)/(上軌−下軌)", code: null },
      { id: "bb_width",     label: "bb_width",     desc: "布林通道寬度 (上軌−下軌)/MA20",   code: null },
    ],
  },
  {
    id: "acceleration",
    label: "加速度（爆發型）",
    color: "#D35400",
    features: [
      { id: "delta_ret_5",      label: "delta_ret_5",      desc: "ret_5 的 3日加速度",         code: "c.pct_change(5).diff(3)" },
      { id: "delta_ret_20",     label: "delta_ret_20",     desc: "ret_20 的 3日加速度",        code: "c.pct_change(20).diff(3)" },
      { id: "delta_rsi_14",     label: "delta_rsi_14",     desc: "(RSI−50) 的 3日加速度",      code: "(_calc_rsi(c, 14) - 50).diff(3)" },
      { id: "delta_vol_5",      label: "delta_vol_5",      desc: "vol_5 的 3日加速度",         code: "c.pct_change().rolling(5).std().diff(3)" },
      { id: "vol_ratio_accel",  label: "vol_ratio_accel",  desc: "vol_ratio_5 / 3日前 vol_ratio_5",code: null },
      { id: "upper_wick_ratio", label: "upper_wick_ratio", desc: "upper_wick / hl_range 比例",  code: null },
      { id: "delta_upper_wick", label: "delta_upper_wick", desc: "upper_wick_ratio 的 3日加速度", code: null },
    ],
  },
  {
    id: "correlation",
    label: "價量相關",
    color: "#7F8C8D",
    features: [
      { id: "price_vol_corr_10", label: "price_vol_corr_10", desc: "10日價格報酬與成交量變化的滾動相關係數", code: "c.pct_change().rolling(10).corr(v.pct_change())" },
    ],
  },
  {
    id: "composite",
    label: "複合特徵（附加診斷）",
    color: "#E91E8C",
    features: [
      {
        id: "trend_efficiency_20",
        label: "trend_efficiency_20",
        desc: "趨勢效率：ret_20 / vol_20，區分穩定上漲與高波動上漲",
        code: null,
        extraDiagnostic: "low_vol_exposure",
        extraDiagnosticDesc: "Layer 2 附加：低波動資產曝險診斷（三條紅線觸發任一→CONFLICT）",
      },
      {
        id: "vol_regime_shift",
        label: "vol_regime_shift",
        desc: "波動率狀態轉換：vol_20 / vol_60，連接慢變量分析與動態擇時",
        code: "c.pct_change().rolling(20).std() / (c.pct_change().rolling(60).std() + 1e-8)",
        extraDiagnostic: "crisis_attribution",
        extraDiagnosticDesc: "Layer 3 附加：危機歸因診斷（三條紅線觸發任一→FAIL）",
      },
      {
        id: "ret5_vol20_ratio",
        label: "ret5_vol20_ratio",
        desc: "試點快變量：delta_ret_5 / vol_20，動量加速度除以波動率正規化",
        code: null,
        extraDiagnostic: "turnover_defense",
        extraDiagnosticDesc: "Layer 2 附加：ΔTurnover 與防禦性修正診斷",
      },
      {
        id: "volume_impulse_vol20",
        label: "volume_impulse_vol20",
        desc: "試點快變量：volume_impulse / vol_20，量能衝擊除以波動率正規化",
        code: null,
        extraDiagnostic: "turnover_defense",
        extraDiagnosticDesc: "Layer 2 附加：換倉摩擦與市場衝擊脆弱性診斷",
      },
    ],
  },
];

// 附加診斷類型對應表（特徵 id → 診斷類型）
export const EXTRA_DIAGNOSTIC_MAP = {
  trend_efficiency_20:  "low_vol_exposure",   // Layer 2 低波動曝險
  vol_regime_shift:     "crisis_attribution", // Layer 3 危機歸因
  ret5_vol20_ratio:     "turnover_defense",   // Layer 2 換倉防禦
  volume_impulse_vol20: "turnover_defense",   // Layer 2 換倉防禦
};

// 診斷類型的顯示設定
export const DIAGNOSTIC_CONFIG = {
  low_vol_exposure: {
    label: "低波動資產曝險診斷",
    layer: 2,
    failLabel: "CONFLICT",
    color: "#E67E22",
    desc: "三條紅線觸發任一即判定 CONFLICT：權重遷移 > 5%、低波動因子 Beta t > 2.0、買入低波動比例 > 60%",
  },
  crisis_attribution: {
    label: "危機歸因診斷",
    layer: 3,
    failLabel: "FAIL",
    color: "#C0392B",
    desc: "三條紅線觸發任一即判定 FAIL：貢獻集中度 > 50%、高風險月份佔比 > 40%、正常期 ΔSharpe ≤ 0",
  },
  turnover_defense: {
    label: "換倉摩擦與防禦診斷",
    layer: 2,
    failLabel: "FAIL",
    color: "#8E44AD",
    desc: "ΔTurnover 顯著上升或下跌時未能提供防禦性修正即判定 FAIL",
  },
};

// 展平成 {id: feature} 的 lookup map
export const FEATURE_MAP = Object.fromEntries(
  FEATURE_GROUPS.flatMap((g) => g.features.map((f) => [f.id, { ...f, group: g.id, groupLabel: g.label, color: g.color }]))
);

// 所有特徵 ID 清單（共 35 個）
export const ALL_FEATURE_IDS = FEATURE_GROUPS.flatMap((g) => g.features.map((f) => f.id));

/**
 * 根據選取的特徵 IDs，產生可執行的 compute_fn 程式碼字串
 * 對於需要複雜計算（code: null）的特徵，引用 processor.py 的完整實作
 */
export function generateComputeFn(selectedIds, featureName) {
  // 需要輔助函數的特徵
  const needsRsi = selectedIds.some((id) =>
    ["rsi_centered", "rsi_slope", "delta_rsi_14"].includes(id)
  );
  const needsAdx = selectedIds.some((id) =>
    ["adx_14", "atr_change"].includes(id)
  );
  const needsComplex = selectedIds.some((id) =>
    ["bb_position", "bb_width", "vol_ratio_accel",
     "upper_wick_ratio", "delta_upper_wick", "volume_impulse"].includes(id)
  );

  const helpers = [];

  if (needsRsi) {
    helpers.push(`    def _calc_rsi(close, period=14):
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))`);
  }

  if (needsAdx) {
    helpers.push(`    def _calc_adx(high, low, close, period=14):
        import numpy as np
        tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        up = high.diff(); down = -low.diff()
        pdm = np.where((up > down) & (up > 0), up, 0.0)
        ndm = np.where((down > up) & (down > 0), down, 0.0)
        pdi = 100 * pd.Series(pdm, index=high.index).rolling(period).mean() / (atr + 1e-8)
        ndi = 100 * pd.Series(ndm, index=high.index).rolling(period).mean() / (atr + 1e-8)
        dx = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-8)
        return dx.rolling(period).mean(), atr`);
  }

  // 單選時欄位名稱用 featureName，多選時各自用自己的 id
  const useFeatureName = selectedIds.length === 1;

  // 產生各特徵的計算行
  const featureLines = selectedIds.map((id) => {
    const f = FEATURE_MAP[id];
    if (!f) return "";
    const colName = useFeatureName ? (featureName || id) : id;

    // 有簡單 code 直接用
    if (f.code) return `        feat["${colName}"] = ${f.code}`;

    // 複雜特徵展開
    switch (id) {
      case "bb_position":
        return `        _ma20 = c.rolling(20).mean(); _std20 = c.rolling(20).std()\n        feat["${colName}"] = (c - (_ma20 - 2*_std20)) / (4*_std20 + 1e-8)`;
      case "bb_width":
        return `        _ma20b = c.rolling(20).mean(); _std20b = c.rolling(20).std()\n        feat["${colName}"] = 4 * _std20b / (_ma20b + 1e-8)`;
      case "vol_ratio_accel":
        return `        _vr5 = v / (v.rolling(5).mean() + 1e-8)\n        feat["${colName}"] = _vr5 / (_vr5.shift(3) + 1e-8)`;
      case "upper_wick_ratio":
        return `        _hl = (h - l + 1e-8); _uw = (h - c.combine(o, max)) / _hl\n        feat["${colName}"] = _uw / _hl`;
      case "delta_upper_wick":
        return `        _hl2 = (h - l + 1e-8); _uw2 = (h - c.combine(o, max)) / _hl2; _uwr = _uw2 / _hl2\n        feat["${colName}"] = _uwr.diff(3)`;
      case "volume_impulse":
        return `        _vr = v / (v.rolling(5).mean() + 1e-8)\n        feat["${colName}"] = (_vr - _vr.rolling(3).mean()) / (_vr.rolling(3).std() + 1e-8)`;
      case "adx_14":
        return `        feat["${colName}"], _ = _calc_adx(h, l, c, 14)`;
      case "atr_change":
        return `        _, _atr = _calc_adx(h, l, c, 14)\n        feat["${colName}"] = _atr.pct_change(5)`;
      case "trend_efficiency_20":
        return `        _ret20 = c.pct_change(20); _vol20 = c.pct_change().rolling(20).std()\n        feat["${colName}"] = _ret20 / (_vol20 + 1e-8)`;
      case "ret5_vol20_ratio":
        return `        _dr5 = c.pct_change(5).diff(3); _v20 = c.pct_change().rolling(20).std()\n        feat["${colName}"] = _dr5 / (_v20 + 1e-8)`;
      case "volume_impulse_vol20":
        return `        _vr = v / (v.rolling(5).mean() + 1e-8)\n        _vimp = (_vr - _vr.rolling(3).mean()) / (_vr.rolling(3).std() + 1e-8)\n        _v20b = c.pct_change().rolling(20).std()\n        feat["${colName}"] = _vimp / (_v20b + 1e-8)`;
      default:
        return `        # ${id}: 請手動實作`;
    }
  }).filter(Boolean);

  const fnName = featureName || "my_new_feature";

  return [
    `def compute_fn(stocks):`,
    `    import pandas as pd`,
    `    result = {}`,
    `    for sid, df in stocks.items():`,
    `        c = df["Close"]; h = df["High"]; l = df["Low"]`,
    `        v = df["Volume"]; o = df["Open"]`,
    `        feat = pd.DataFrame(index=df.index)`,
    ...(helpers.length ? helpers : []),
    ...featureLines,
    `        result[sid] = feat`,
    `    return result`,
  ].join("\n");
}