import { REGIME_LABEL, REGIME_COLOR } from "../../constants/config";

export default function WalkForwardWindowCard({ w }) {
  const ret = w.val_return;
  return (
    <div style={{
      background: "var(--color-background-secondary)",
      borderRadius: "var(--border-radius-md)",
      padding: "14px 16px",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
        <span style={{ fontSize: 13, fontWeight: 500 }}>窗口 {w.window}</span>
        <span style={{
          fontSize: 12, padding: "2px 8px", borderRadius: 4,
          background: "var(--color-background-primary)",
          color: REGIME_COLOR[w.regime] || "var(--color-text-secondary)",
        }}>
          {REGIME_LABEL[w.regime] || w.regime || "—"}
        </span>
      </div>
      <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 8 }}>
        訓練 {w.train_start?.slice(0, 7)}~{w.train_end?.slice(0, 7)}<br />
        驗證 {w.val_start?.slice(0, 7)}~{w.val_end?.slice(0, 7)}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <div>
          <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>訓練報酬</div>
          <div style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-secondary)" }}>
            {w.train_return != null ? (w.train_return >= 0 ? "+" : "") + w.train_return + "%" : "—"}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>驗證報酬</div>
          <div style={{ fontSize: 14, fontWeight: 500,
            color: ret == null ? "var(--color-text-secondary)" : ret >= 0 ? "var(--color-text-success)" : "var(--color-text-danger)" }}>
            {ret != null ? (ret >= 0 ? "+" : "") + ret + "%" : "—"}
          </div>
        </div>
      </div>
      {w.win_rate != null && (
        <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 6 }}>
          勝率 {w.win_rate}%
        </div>
      )}
    </div>
  );
}