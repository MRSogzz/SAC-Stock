import { useRef, useEffect } from "react";

const VERDICT_CONFIG = {
  PASS:       { icon: "✅", color: "var(--color-text-success)",  bg: "rgba(29,158,117,0.08)",  border: "rgba(29,158,117,0.3)",  label: "策略增益顯著" },
  WEAK_PASS:  { icon: "⚠️", color: "#854f0b",                    bg: "rgba(186,117,23,0.08)",  border: "rgba(186,117,23,0.3)",  label: "改善微弱但方向一致" },
  NO_EFFECT:  { icon: "➖", color: "var(--color-text-secondary)", bg: "var(--color-background-secondary)", border: "var(--color-border-tertiary)", label: "無明顯影響" },
  CONFLICT:   { icon: "🚨", color: "#993c1d",                    bg: "rgba(226,75,74,0.08)",   border: "rgba(226,75,74,0.3)",   label: "偽 Alpha 警報" },
  FAIL:       { icon: "❌", color: "var(--color-text-danger)",   bg: "rgba(226,75,74,0.08)",   border: "rgba(226,75,74,0.3)",   label: "風險惡化" },
};

// 雙曲線 Canvas
function CurveChart({ baseline, candidate }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current || !baseline?.length) return;
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    const W = canvas.offsetWidth, H = 160;
    canvas.width = W * 2; canvas.height = H * 2; ctx.scale(2, 2);
    ctx.clearRect(0, 0, W, H);

    const all = [...baseline, ...(candidate || [])];
    const min = Math.min(...all), max = Math.max(...all), range = max - min || 1;
    const pad = { l: 52, r: 12, t: 12, b: 20 };
    const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
    const n = baseline.length;
    const x = (i) => pad.l + (i / (n - 1)) * iw;
    const y = (v) => pad.t + ih - ((v - min) / range) * ih;

    [0, 0.5, 1].forEach((t) => {
      const yy = pad.t + ih - t * ih;
      ctx.strokeStyle = "rgba(128,128,128,0.1)"; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(pad.l + iw, yy); ctx.stroke();
      const val = min + t * range;
      ctx.fillStyle = "rgba(128,128,128,0.5)"; ctx.font = "9px sans-serif";
      ctx.fillText("$" + (val / 10000).toFixed(0) + "W", 2, yy + 3);
    });

    ctx.beginPath(); ctx.moveTo(x(0), y(baseline[0]));
    baseline.forEach((v, i) => ctx.lineTo(x(i), y(v)));
    ctx.strokeStyle = "rgba(128,128,128,0.5)"; ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);

    if (candidate?.length) {
      ctx.beginPath(); ctx.moveTo(x(0), y(candidate[0]));
      candidate.forEach((v, i) => ctx.lineTo(x(i), y(v)));
      const c = candidate[candidate.length - 1] >= candidate[0] ? "#1D9E75" : "#E24B4A";
      ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.stroke();
    }
  }, [baseline, candidate]);

  return (
    <div>
      <canvas ref={ref} style={{ width: "100%", height: 160, display: "block" }} />
      <div style={{ display: "flex", gap: 16, marginTop: 6, fontSize: 11, color: "var(--color-text-secondary)" }}>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 16, height: 2, borderTop: "2px dashed rgba(128,128,128,0.5)", display: "inline-block" }} />基準
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 16, height: 2, background: "#1D9E75", display: "inline-block" }} />探針策略
        </span>
        <span style={{ marginLeft: "auto", fontSize: 10, color: "var(--color-text-tertiary)" }}>
          ε = 0.01（固定微量探針）
        </span>
      </div>
    </div>
  );
}

function DeltaRow({ label, baseline, candidate, delta, ok, format = "decimal" }) {
  const fmt = (v) => format === "pct" ? `${(v * 100).toFixed(2)}%` : v.toFixed(4);
  return (
    <tr style={{ borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
      <td style={{ padding: "10px 12px", fontSize: 13, color: "var(--color-text-secondary)", fontWeight: 500 }}>{label}</td>
      <td style={{ padding: "10px 12px", fontFamily: "var(--font-mono)", fontSize: 13, textAlign: "right" }}>{fmt(baseline)}</td>
      <td style={{ padding: "10px 12px", fontFamily: "var(--font-mono)", fontSize: 13, textAlign: "right" }}>{fmt(candidate)}</td>
      <td style={{ padding: "10px 12px", fontFamily: "var(--font-mono)", fontSize: 13, textAlign: "right", fontWeight: 600,
        color: ok ? "var(--color-text-success)" : delta === 0 ? "var(--color-text-secondary)" : "var(--color-text-danger)" }}>
        {delta >= 0 ? "+" : ""}{fmt(delta)}
      </td>
      <td style={{ padding: "10px 12px", textAlign: "center", fontSize: 14 }}>
        {ok ? "✓" : delta === 0 ? "—" : "✗"}
      </td>
    </tr>
  );
}

// Regime 一致性指示條
function RegimeConsistency({ value }) {
  const pct  = Math.round((value ?? 0) * 100);
  const color = pct >= 75 ? "#1D9E75" : pct >= 50 ? "#E6A817" : "#E24B4A";
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12,
        color: "var(--color-text-secondary)", marginBottom: 4 }}>
        <span>Regime 一致性</span>
        <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600, color }}>{pct}%</span>
      </div>
      <div style={{ height: 6, background: "var(--color-background-secondary)", borderRadius: 3 }}>
        <div style={{ height: "100%", borderRadius: 3, background: color,
          width: `${pct}%`, transition: "width 0.6s ease" }} />
      </div>
      <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 3 }}>
        在各市場環境中有正 ΔSharpe 的比例（≥ 50% 才算穩定）
      </div>
    </div>
  );
}

export default function Layer2Panel({ layer2, stopAt }) {
  if (!layer2 && stopAt === 1) return <SkippedBox msg="Layer 2：已跳過（Layer 1 未通過）" />;
  if (!layer2) return <SkippedBox msg="Layer 2：未執行" />;

  const d   = layer2;
  const bm  = d.baseline_metrics;
  const cm  = d.candidate_metrics;
  const vcfg = VERDICT_CONFIG[d.verdict] || VERDICT_CONFIG.NO_EFFECT;

  return (
    <div style={panelStyle}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <span style={{ fontSize: 13, fontWeight: 600 }}>Layer 2</span>
        <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>特徵微量探針檢驗</span>
        <span style={{ marginLeft: "auto" }}>
          <span style={{
            display: "inline-flex", alignItems: "center", gap: 6,
            padding: "4px 12px", borderRadius: 6, fontSize: 13, fontWeight: 600,
            color: vcfg.color, background: vcfg.bg, border: `0.5px solid ${vcfg.border}`,
          }}>
            {vcfg.icon} {d.verdict}
            <span style={{ fontSize: 11, fontWeight: 400, color: "var(--color-text-secondary)" }}>
              — {vcfg.label}
            </span>
          </span>
        </span>
      </div>

      {/* 資金曲線 */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 8 }}>資金曲線對比</div>
        <CurveChart baseline={bm.portfolio_curve} candidate={cm.portfolio_curve} />
      </div>

      {/* 指標表格 */}
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, marginBottom: 4 }}>
        <thead>
          <tr style={{ borderBottom: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-secondary)" }}>
            {["指標", "基準", "探針策略", "Δ", ""].map((h) => (
              <th key={h} style={{ padding: "8px 12px", textAlign: h === "指標" ? "left" : "right",
                fontWeight: 500, color: "var(--color-text-secondary)", fontSize: 11 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <DeltaRow label="Total Return" baseline={bm.total_return} candidate={cm.total_return}
            delta={d.delta_return} ok={d.delta_return > 0} format="pct" />
          <DeltaRow label="Sharpe" baseline={bm.sharpe} candidate={cm.sharpe}
            delta={d.delta_sharpe} ok={d.delta_sharpe >= 0.05} />
          <DeltaRow label="Max Drawdown" baseline={bm.max_drawdown} candidate={cm.max_drawdown}
            delta={d.delta_mdd} ok={d.delta_mdd >= -0.03} format="pct" />
          <DeltaRow label="Avg Turnover" baseline={bm.avg_turnover} candidate={cm.avg_turnover}
            delta={d.delta_turnover} ok={d.delta_turnover <= 0.10} format="pct" />
        </tbody>
      </table>

      {/* Regime 一致性 */}
      <RegimeConsistency value={d.regime_consistency} />

      {/* 附加診斷 */}
      {d.extra_diagnostic && (
        <ExtraDiagnosticPanel
          data={d.extra_diagnostic}
          type={
            d.extra_diagnostic.metrics?.weight_shift !== undefined ? "low_vol_exposure" :
            d.extra_diagnostic.metrics?.delta_turnover !== undefined ? "turnover_defense" : null
          }
        />
      )}

      {/* 診斷原因 */}
      {d.rejection_reasons?.length > 0 && (
        <div style={{ marginTop: 12, padding: "10px 14px", borderRadius: "var(--border-radius-md)",
          background: vcfg.bg, border: `0.5px solid ${vcfg.border}`,
          fontSize: 12, color: vcfg.color, lineHeight: 1.8 }}>
          {d.rejection_reasons.map((r, i) => <div key={i}>• {r}</div>)}
        </div>
      )}
    </div>
  );
}

function SkippedBox({ msg }) {
  return (
    <div style={{ padding: "14px 20px", fontSize: 13, color: "var(--color-text-tertiary)",
      background: "var(--color-background-secondary)",
      borderRadius: "var(--border-radius-lg)",
      border: "0.5px solid var(--color-border-tertiary)" }}>
      {msg}
    </div>
  );
}

const panelStyle = {
  background: "var(--color-background-primary)",
  border: "0.5px solid var(--color-border-tertiary)",
  borderRadius: "var(--border-radius-lg)",
  padding: "20px 24px",
};

// ── 附加診斷面板 ──────────────────────────────────────────────────────────────
export function ExtraDiagnosticPanel({ data, type }) {
  if (!data) return null;

  const CONFIG = {
    low_vol_exposure: { label: "低波動資產曝險診斷", color: "#E67E22", failLabel: "CONFLICT" },
    turnover_defense: { label: "換倉防禦診斷",       color: "#8E44AD", failLabel: "FAIL"     },
  };
  const cfg = CONFIG[type] || { label: "附加診斷", color: "#666", failLabel: "FAIL" };
  const triggered = data.triggered;

  return (
    <div style={{
      marginTop: 12,
      padding: "12px 16px",
      borderRadius: "var(--border-radius-md)",
      background: triggered ? "rgba(226,75,74,0.05)" : "rgba(29,158,117,0.05)",
      border: `0.5px solid ${triggered ? "rgba(226,75,74,0.3)" : "rgba(29,158,117,0.2)"}`,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: cfg.color }}>{cfg.label}</span>
        <span style={{
          fontSize: 11, padding: "1px 8px", borderRadius: 4,
          background: triggered ? "rgba(226,75,74,0.12)" : "rgba(29,158,117,0.12)",
          color: triggered ? "var(--color-text-danger)" : "var(--color-text-success)",
          fontWeight: 600,
        }}>
          {triggered ? `🚨 ${cfg.failLabel}` : "✓ 通過"}
        </span>
      </div>

      {/* 量化指標 */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: triggered ? 8 : 0 }}>
        {type === "low_vol_exposure" && <>
          <Metric label="低波動組權重差" value={`${((data.metrics?.weight_shift || 0) * 100).toFixed(1)}%`}
            threshold={`> 5%`} bad={Math.abs(data.metrics?.weight_shift || 0) > 0.05} />
          <Metric label="因子 Beta" value={(data.metrics?.factor_beta || 0).toFixed(4)}
            threshold="" bad={false} />
          <Metric label="因子 t 統計量" value={(data.metrics?.factor_beta_t || 0).toFixed(2)}
            threshold={`> 2.0`} bad={(data.metrics?.factor_beta_t || 0) > 2.0} />
          <Metric label="買入低波動比" value={`${((data.metrics?.buy_low_ratio || 0) * 100).toFixed(1)}%`}
            threshold={`> 60%`} bad={(data.metrics?.buy_low_ratio || 0) > 0.6} />
        </>}
        {type === "turnover_defense" && <>
          <Metric label="基準換倉率" value={`${((data.metrics?.base_turnover || 0) * 100).toFixed(2)}%`} threshold="" bad={false} />
          <Metric label="探針換倉率" value={`${((data.metrics?.probe_turnover || 0) * 100).toFixed(2)}%`} threshold="" bad={false} />
          <Metric label="ΔTurnover" value={`${((data.metrics?.delta_turnover || 0) * 100).toFixed(2)}%`}
            threshold={`> 10%`} bad={(data.metrics?.delta_turnover || 0) > 0.1} />
          {data.metrics?.defense_corr != null && (
            <Metric label="防禦相關係數" value={(data.metrics.defense_corr).toFixed(3)}
              threshold={`> -0.1`} bad={data.metrics.defense_corr > -0.1} />
          )}
        </>}
      </div>

      {/* 觸發的紅線 */}
      {triggered && data.triggered_rules?.length > 0 && (
        <div style={{ fontSize: 11, color: "var(--color-text-danger)", lineHeight: 1.7 }}>
          {data.triggered_rules.map((r, i) => <div key={i}>• {r}</div>)}
        </div>
      )}
    </div>
  );
}

function Metric({ label, value, threshold, bad }) {
  return (
    <div style={{
      display: "inline-flex", flexDirection: "column", alignItems: "center",
      padding: "5px 10px", borderRadius: 6, fontSize: 11,
      background: bad ? "rgba(226,75,74,0.08)" : "var(--color-background-secondary)",
      border: `0.5px solid ${bad ? "rgba(226,75,74,0.3)" : "var(--color-border-tertiary)"}`,
    }}>
      <span style={{ color: "var(--color-text-tertiary)", marginBottom: 2 }}>{label}</span>
      <span style={{ fontWeight: 600, fontFamily: "var(--font-mono)",
        color: bad ? "var(--color-text-danger)" : "var(--color-text-primary)" }}>
        {value}
      </span>
      {threshold && <span style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>{threshold}</span>}
    </div>
  );
}