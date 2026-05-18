export default function PositionBar({ name, pct, sector }) {
  const color =
    pct > 25 ? "#1D9E75" :
    pct > 10 ? "#378ADD" :
    "var(--color-text-tertiary)";

  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
        <span>
          {name}{" "}
          <span style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>{sector}</span>
        </span>
        <span style={{ fontWeight: 500, color, fontFamily: "var(--font-mono)" }}>{pct}%</span>
      </div>
      <div style={{ height: 6, background: "var(--color-background-secondary)", borderRadius: 3 }}>
        <div style={{
          height: "100%",
          width: `${Math.min((pct / 40) * 100, 100)}%`,
          background: color,
          borderRadius: 3,
          transition: "width 0.8s ease",
        }} />
      </div>
    </div>
  );
}