import { useRef, useEffect } from "react";

const THRESHOLDS = {
  rank_ic_5d:  0.02,
  rank_ic_20d: 0.015,
  dir_acc_5d:  0.52,
  dir_acc_20d: 0.51,
};

function MetricRow({ label, value, threshold, format = "decimal", digits = 4 }) {
  const passed = value > threshold;
  const display = format === "pct" ? `${(value * 100).toFixed(2)}%` : value.toFixed(digits);
  const thresholdDisplay = format === "pct"
    ? `> ${(threshold * 100).toFixed(0)}%`
    : `> ${threshold}`;

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "10px 14px", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
      <div>
        <span style={{ fontSize: 13, fontWeight: 500 }}>{label}</span>
        <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginLeft: 8 }}>
          門檻 {thresholdDisplay}
        </span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{
          fontFamily: "var(--font-mono)", fontSize: 14, fontWeight: 600,
          color: passed ? "var(--color-text-success)" : "var(--color-text-danger)",
        }}>{display}</span>
        <span style={{ fontSize: 14 }}>{passed ? "✓" : "✗"}</span>
      </div>
    </div>
  );
}

// 月 IC 條形圖（Canvas）
function MonthlyICChart({ data }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current || !data || Object.keys(data).length === 0) return;
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    const W = canvas.offsetWidth, H = 100;
    canvas.width = W * 2; canvas.height = H * 2; ctx.scale(2, 2);
    ctx.clearRect(0, 0, W, H);

    const entries = Object.entries(data).sort(([a], [b]) => a.localeCompare(b));
    const vals = entries.map(([, v]) => v);
    const maxAbs = Math.max(...vals.map(Math.abs), 0.04);
    const pad = { l: 8, r: 8, t: 12, b: 20 };
    const bw = (W - pad.l - pad.r) / entries.length;
    const midY = pad.t + (H - pad.t - pad.b) / 2;
    const scale = (H - pad.t - pad.b) / 2 / maxAbs;

    // Zero line
    ctx.strokeStyle = "rgba(128,128,128,0.3)"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, midY); ctx.lineTo(W - pad.r, midY); ctx.stroke();

    entries.forEach(([month, val], i) => {
      const x = pad.l + i * bw + bw * 0.1;
      const bh = val * scale;
      const color = val >= 0 ? "#1D9E75" : "#E24B4A";
      ctx.fillStyle = color + "CC";
      ctx.fillRect(x, val >= 0 ? midY - bh : midY, bw * 0.8, Math.abs(bh));

      // Month label
      ctx.fillStyle = "rgba(128,128,128,0.6)";
      ctx.font = "8px sans-serif";
      ctx.fillText(month.slice(5), x, H - 4);
    });
  }, [data]);

  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 6 }}>月 IC 分佈（5日）</div>
      <canvas ref={ref} style={{ width: "100%", height: 100, display: "block" }} />
    </div>
  );
}

export default function Layer1Panel({ layer1, stopped }) {
  if (!layer1 && !stopped) return (
    <div style={skippedStyle}>Layer 1：未執行</div>
  );
  if (!layer1) return (
    <div style={skippedStyle}>Layer 1：已跳過（Layer 1 未通過）</div>
  );

  const d = layer1;
  return (
    <div style={panelStyle}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <span style={{ fontSize: 13, fontWeight: 600 }}>Layer 1</span>
        <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>預測訊號檢驗</span>
        <span style={{ marginLeft: "auto", fontSize: 13 }}>{d.passed ? "✅ PASS" : "❌ FAIL"}</span>
      </div>

      <div style={{ borderRadius: "var(--border-radius-md)", overflow: "hidden",
        border: "0.5px solid var(--color-border-tertiary)", marginBottom: 16 }}>
        <MetricRow label="Rank IC (5d)"  value={d.rank_ic_5d}  threshold={THRESHOLDS.rank_ic_5d}  />
        <MetricRow label="Rank IC (20d)" value={d.rank_ic_20d} threshold={THRESHOLDS.rank_ic_20d} />
        <MetricRow label="IC IR (5d)"    value={d.ic_ir_5d}    threshold={0} />
        <MetricRow label="方向準確率 (5d)"  value={d.dir_acc_5d}  threshold={THRESHOLDS.dir_acc_5d}  format="pct" digits={2} />
        <MetricRow label="方向準確率 (20d)" value={d.dir_acc_20d} threshold={THRESHOLDS.dir_acc_20d} format="pct" digits={2} />
      </div>

      {d.monthly_ic_5d && Object.keys(d.monthly_ic_5d).length > 0 && (
        <MonthlyICChart data={d.monthly_ic_5d} />
      )}

      {d.rejection_reasons?.length > 0 && (
        <div style={reasonStyle}>
          {d.rejection_reasons.map((r, i) => <div key={i}>• {r}</div>)}
        </div>
      )}
    </div>
  );
}

const panelStyle = {
  background: "var(--color-background-primary)",
  border: "0.5px solid var(--color-border-tertiary)",
  borderRadius: "var(--border-radius-lg)",
  padding: "20px 24px",
};
const skippedStyle = {
  padding: "14px 20px",
  fontSize: 13,
  color: "var(--color-text-tertiary)",
  background: "var(--color-background-secondary)",
  borderRadius: "var(--border-radius-lg)",
  border: "0.5px solid var(--color-border-tertiary)",
};
const reasonStyle = {
  marginTop: 12,
  padding: "10px 14px",
  borderRadius: "var(--border-radius-md)",
  background: "rgba(226,75,74,0.06)",
  border: "0.5px solid rgba(226,75,74,0.2)",
  fontSize: 12,
  color: "var(--color-text-danger)",
  lineHeight: 1.8,
};