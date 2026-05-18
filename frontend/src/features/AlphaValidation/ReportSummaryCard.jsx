import VerdictBadge from "./VerdictBadge";

const PIPELINE = ["Layer 1\n訊號檢驗", "Layer 2\n策略增益", "Layer 3\n穩定性"];

function PipelineStep({ label, passed, active, skipped, index }) {
  const color = skipped
    ? "var(--color-text-tertiary)"
    : passed === true
    ? "var(--color-text-success)"
    : passed === false
    ? "var(--color-text-danger)"
    : "var(--color-text-secondary)";

  const bg = skipped
    ? "var(--color-background-secondary)"
    : passed === true
    ? "rgba(29,158,117,0.08)"
    : passed === false
    ? "rgba(226,75,74,0.08)"
    : "var(--color-background-secondary)";

  const icon = skipped ? "—" : passed === true ? "✅" : passed === false ? "❌" : "⏳";

  return (
    <div style={{ flex: 1, textAlign: "center" }}>
      <div style={{
        padding: "14px 8px",
        borderRadius: "var(--border-radius-md)",
        background: bg,
        border: `0.5px solid ${skipped || passed == null
          ? "var(--color-border-tertiary)"
          : passed ? "rgba(29,158,117,0.3)" : "rgba(226,75,74,0.3)"}`,
        opacity: skipped ? 0.5 : 1,
      }}>
        <div style={{ fontSize: 18, marginBottom: 4 }}>{icon}</div>
        <div style={{ fontSize: 12, fontWeight: 500, color, whiteSpace: "pre-line", lineHeight: 1.4 }}>
          {label}
        </div>
      </div>
      {index < PIPELINE.length - 1 && (
        <div style={{ position: "absolute", right: "-12px", top: "50%", transform: "translateY(-50%)",
          fontSize: 16, color: "var(--color-text-tertiary)" }}>→</div>
      )}
    </div>
  );
}

export default function ReportSummaryCard({ report }) {
  if (!report) return null;

  const { feature_name, description, validation_period, final_verdict, layer1, layer2, layer3 } = report;

  const l1 = layer1 || null;
  const l2 = layer2 || null;
  const l3 = layer3 || null;

  const verdictKey = final_verdict === "PASS" ? "PASS"
    : final_verdict === "FAIL" ? "FAIL"
    : "PARTIAL";

  return (
    <div style={{
      background: "var(--color-background-primary)",
      border: `1px solid ${final_verdict === "PASS" ? "rgba(29,158,117,0.3)"
        : final_verdict === "FAIL" ? "rgba(226,75,74,0.3)"
        : "var(--color-border-tertiary)"}`,
      borderRadius: "var(--border-radius-lg)",
      padding: "20px 24px",
      marginBottom: 16,
    }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 16 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <span style={{ fontSize: 18, fontWeight: 600 }}>{feature_name}</span>
            <VerdictBadge verdict={verdictKey} size="lg" />
          </div>
          {description && (
            <div style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>{description}</div>
          )}
          {validation_period && (
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 4, fontFamily: "var(--font-mono)" }}>
              驗證區間：{validation_period[0]} ～ {validation_period[1]}
            </div>
          )}
        </div>
      </div>

      {/* Pipeline 流程圖 */}
      <div style={{ display: "flex", gap: 8, position: "relative" }}>
        <PipelineStep index={0} label={PIPELINE[0]}
          passed={l1?.passed}
          skipped={!l1} />
        <div style={{ display: "flex", alignItems: "center", color: "var(--color-text-tertiary)", fontSize: 14, paddingBottom: 2 }}>→</div>
        <PipelineStep index={1} label={PIPELINE[1]}
          passed={l2?.passed}
          skipped={!l2} />
        <div style={{ display: "flex", alignItems: "center", color: "var(--color-text-tertiary)", fontSize: 14, paddingBottom: 2 }}>→</div>
        <PipelineStep index={2} label={PIPELINE[2]}
          passed={l3?.passed}
          skipped={!l3} />
      </div>

      {/* 快速摘要指標 */}
      {(l1 || l2 || l3) && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14 }}>
          {l1 && <>
            <Chip label="IC(5d)" value={l1.rank_ic_5d?.toFixed(4)} ok={l1.rank_ic_5d > 0.02} />
            <Chip label="Dir Acc(5d)" value={(l1.dir_acc_5d * 100).toFixed(1) + "%"} ok={l1.dir_acc_5d > 0.52} />
          </>}
          {l2 && <>
            <Chip label="ΔSharpe" value={(l2.delta_sharpe >= 0 ? "+" : "") + l2.delta_sharpe?.toFixed(4)} ok={l2.delta_sharpe > 0.05} />
            <Chip label="ΔMdd" value={(l2.delta_mdd >= 0 ? "+" : "") + (l2.delta_mdd * 100).toFixed(2) + "%"} ok={l2.delta_mdd >= -0.03} />
          </>}
          {l3 && <>
            <Chip label="穩定區間" value={`${l3.stable_regimes}/${Object.keys(l3.regime_results || {}).length}`} ok={l3.stable_regimes >= 2} />
            <Chip label="vs等權重" value={l3.beats_equal_weight ? "勝" : "負"} ok={l3.beats_equal_weight} />
          </>}
        </div>
      )}
    </div>
  );
}

function Chip({ label, value, ok }) {
  return (
    <div style={{
      display: "inline-flex", flexDirection: "column", alignItems: "center",
      padding: "6px 12px", borderRadius: 6, fontSize: 11,
      background: ok ? "rgba(29,158,117,0.07)" : "rgba(226,75,74,0.07)",
      border: `0.5px solid ${ok ? "rgba(29,158,117,0.2)" : "rgba(226,75,74,0.2)"}`,
    }}>
      <span style={{ color: "var(--color-text-tertiary)", marginBottom: 2 }}>{label}</span>
      <span style={{ fontWeight: 600, fontFamily: "var(--font-mono)",
        color: ok ? "var(--color-text-success)" : "var(--color-text-danger)" }}>
        {value}
      </span>
    </div>
  );
}