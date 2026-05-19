const HEADERS = ["期間", "類型", "目標回合", "累積訓練", "AI 報酬", "儲存時間"];

export default function SavedModelsTable({ models, type }) {
  const filtered = models?.filter((m) => {
    if (!type) return true;
    if (type === "standard")    return m.model_type === "standard"    || !m.model_type;
    if (type === "walkforward") return m.model_type === "walkforward";
    return true;
  });

  if (!filtered?.length) return null;

  const typeLabel = (m) => {
    if (m.model_type === "walkforward") return { label: "Walk-Forward", color: "var(--color-text-success)" };
    return { label: "單模型", color: "var(--color-text-secondary)" };
  };

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 16 }}>已儲存的模型</div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
            {HEADERS.map((h) => (
              <th key={h} style={{ padding: "8px 12px", textAlign: "left", fontWeight: 500, fontSize: 12, color: "var(--color-text-secondary)" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {filtered.map((m, i) => {
            const tl = typeLabel(m);
            return (
              <tr key={i} style={{ borderBottom: "0.5px solid var(--color-border-tertiary)", background: i % 2 === 0 ? "transparent" : "var(--color-background-secondary)" }}>
                <td style={{ padding: "9px 12px", fontWeight: 500 }}>{m.period || "—"}</td>
                <td style={{ padding: "9px 12px", fontSize: 11, color: tl.color }}>{tl.label}</td>
                <td style={{ padding: "9px 12px", fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                  {m.episodes != null ? m.episodes : "—"}
                </td>
                <td style={{ padding: "9px 12px", fontFamily: "var(--font-mono)",
                  color: m.episodes_done > m.episodes ? "var(--color-text-success)" :
                         m.episodes_done === m.episodes ? "var(--color-text-primary)" : "var(--color-text-warning)" }}>
                  {m.episodes_done != null ? m.episodes_done : "—"}
                  {m.episodes_done != null && m.episodes_done >= m.episodes &&
                    <span style={{ fontSize: 11, marginLeft: 4, color: "var(--color-text-success)" }}>✓</span>}
                </td>
                <td style={{ padding: "9px 12px", fontFamily: "var(--font-mono)", fontWeight: 500,
                  color: m.total_return == null ? "var(--color-text-tertiary)" :
                         m.total_return >= 0 ? "var(--color-text-success)" : "var(--color-text-danger)" }}>
                  {m.total_return != null ? `${m.total_return >= 0 ? "+" : ""}${m.total_return}%` : "—"}
                </td>
                <td style={{ padding: "9px 12px", fontSize: 12, color: "var(--color-text-tertiary)", fontFamily: "var(--font-mono)" }}>
                  {m.saved_at || "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const cardStyle = {
  background: "var(--color-background-primary)",
  border: "0.5px solid var(--color-border-tertiary)",
  borderRadius: "var(--border-radius-lg)",
  padding: "20px 24px",
  marginBottom: 16,
};