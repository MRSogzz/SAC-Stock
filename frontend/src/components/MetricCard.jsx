export default function MetricCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: "var(--color-background-secondary)",
      borderRadius: "var(--border-radius-md)",
      padding: "14px 16px",
    }}>
      <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 20, fontWeight: 500, color: color || "var(--color-text-primary)" }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 2 }}>
          {sub}
        </div>
      )}
    </div>
  );
}