import { REGIME_LABEL, REGIME_COLOR } from "../constants/config";

const ACTION_COLOR = {
  買入: "var(--color-text-success)",
  持有: "var(--color-text-info)",
  觀望: "var(--color-text-tertiary)",
};
const ACTION_BG = {
  買入: "rgba(29,158,117,0.08)",
  持有: "rgba(55,138,221,0.08)",
  觀望: "transparent",
};

export default function PredictionPanel({ pred }) {
  if (!pred) return null;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        <div style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>截至 {pred.as_of_date}</div>
        {pred.current_regime && (
          <span style={{
            fontSize: 12, padding: "2px 8px", borderRadius: 4,
            background: "var(--color-background-secondary)",
            color: REGIME_COLOR[pred.current_regime] || "var(--color-text-secondary)",
          }}>
            {REGIME_LABEL[pred.current_regime] || pred.current_regime}
            {pred.selected_window && ` · 窗口 ${pred.selected_window}`}
          </span>
        )}
        <div style={{ marginLeft: "auto", fontSize: 12, color: "var(--color-text-secondary)" }}>
          建議現金：<span style={{ fontWeight: 500, color: "var(--color-text-primary)" }}>{pred.cash_pct}%</span>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 8 }}>
        {pred.recommendations?.map((r) => (
          <div key={r.stock_id} style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            padding: "10px 14px", borderRadius: "var(--border-radius-md)",
            background: ACTION_BG[r.action],
            border: `0.5px solid ${r.action === "觀望" ? "var(--color-border-tertiary)" : "transparent"}`,
          }}>
            <div>
              <span style={{ fontWeight: 500, fontSize: 13 }}>{r.stock_name}</span>
              <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginLeft: 6 }}>{r.stock_id}</span>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 13, fontWeight: 500, color: ACTION_COLOR[r.action] }}>{r.action}</div>
              <div style={{ fontSize: 12, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                目標 {r.target_pct}%
              </div>
            </div>
          </div>
        ))}
      </div>
      <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginTop: 12 }}>
        ⚠ 本預測僅供參考，不構成投資建議
      </div>
    </div>
  );
}