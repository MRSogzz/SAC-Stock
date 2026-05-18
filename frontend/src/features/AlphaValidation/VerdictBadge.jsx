const CONFIG = {
  PASS:    { icon: "✅", color: "var(--color-text-success)", bg: "rgba(29,158,117,0.1)",  border: "rgba(29,158,117,0.3)"  },
  FAIL:    { icon: "❌", color: "var(--color-text-danger)",  bg: "rgba(226,75,74,0.1)",   border: "rgba(226,75,74,0.3)"   },
  PARTIAL: { icon: "⚠️", color: "var(--color-text-warning)", bg: "rgba(186,117,23,0.1)",  border: "rgba(186,117,23,0.3)"  },
  PENDING: { icon: "⏳", color: "var(--color-text-secondary)",bg: "var(--color-background-secondary)", border: "var(--color-border-tertiary)" },
};

export default function VerdictBadge({ verdict, size = "md" }) {
  const cfg = CONFIG[verdict] || CONFIG.PENDING;
  const fontSize = size === "lg" ? 16 : size === "sm" ? 11 : 13;
  const padding  = size === "lg" ? "8px 18px" : size === "sm" ? "2px 8px" : "4px 12px";

  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      padding, borderRadius: 6, fontSize, fontWeight: 600,
      color: cfg.color,
      background: cfg.bg,
      border: `0.5px solid ${cfg.border}`,
    }}>
      <span>{cfg.icon}</span>
      {verdict}
    </span>
  );
}