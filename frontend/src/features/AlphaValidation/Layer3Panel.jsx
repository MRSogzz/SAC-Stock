import { useRef, useEffect } from "react";

const REGIME_LABEL = {
  bull_trend:       "牛市趨勢",
  bear_trend:       "熊市趨勢",
  low_vol_sideways: "低波動盤整",
  high_vol_sideways:"高波動盤整",
  unknown:          "未分類",
};

// Regime 橫向條形圖
function RegimeChart({ regimeResults }) {
  const entries = Object.entries(regimeResults || {});
  if (!entries.length) return null;
  const maxAbs = Math.max(...entries.map(([, r]) => Math.abs(r.delta_sharpe)), 0.05);

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 8 }}>各市場環境 ΔSharpe</div>
      {entries.map(([regime, r]) => {
        const pct = Math.abs(r.delta_sharpe) / maxAbs * 100;
        const pos = r.delta_sharpe >= 0;
        return (
          <div key={regime} style={{ marginBottom: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 3 }}>
              <span>{REGIME_LABEL[regime] || regime}
                <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginLeft: 6 }}>{r.n_days}天</span>
              </span>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600,
                color: pos ? "var(--color-text-success)" : "var(--color-text-danger)" }}>
                {r.delta_sharpe >= 0 ? "+" : ""}{r.delta_sharpe.toFixed(4)}
              </span>
            </div>
            <div style={{ height: 6, background: "var(--color-background-secondary)", borderRadius: 3 }}>
              <div style={{
                height: "100%", borderRadius: 3,
                width: `${pct}%`,
                background: pos ? "#1D9E75" : "#E24B4A",
                transition: "width 0.6s ease",
              }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Sharpe vs 外部基準對比
function BenchmarkRow({ label, candidateSharpe, benchmarkSharpe, beats }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "10px 14px", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
      <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>{label}</span>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--color-text-tertiary)" }}>
          基準 {benchmarkSharpe.toFixed(4)}
        </span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, fontWeight: 600,
          color: beats ? "var(--color-text-success)" : "var(--color-text-danger)" }}>
          候選 {candidateSharpe.toFixed(4)}
        </span>
        <span>{beats ? "✓" : "✗"}</span>
      </div>
    </div>
  );
}

export default function Layer3Panel({ layer3, stopAt }) {
  if (!layer3 && stopAt === 1) return (
    <div style={skippedStyle}>Layer 3：已跳過（Layer 1 未通過）</div>
  );
  if (!layer3 && stopAt === 2) return (
    <div style={skippedStyle}>Layer 3：已跳過（Layer 2 未通過）</div>
  );
  if (!layer3) return (
    <div style={skippedStyle}>Layer 3：未執行</div>
  );

  const d = layer3;
  const hhi_ok = d.hhi_explosion_rate < 0.15;
  const tail_ok = d.delta_tail_risk >= -0.005;

  return (
    <div style={panelStyle}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <span style={{ fontSize: 13, fontWeight: 600 }}>Layer 3</span>
        <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>穩定性與失敗模式</span>
        <span style={{ marginLeft: "auto", fontSize: 13 }}>{d.passed ? "✅ PASS" : "❌ FAIL"}</span>
      </div>

      {/* 市場環境穩定性 */}
      <RegimeChart regimeResults={d.regime_results} />

      <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 16 }}>
        穩定環境數：
        <span style={{ fontWeight: 600, color: d.stable_regimes >= 2 ? "var(--color-text-success)" : "var(--color-text-danger)",
          marginLeft: 4 }}>
          {d.stable_regimes} / {Object.keys(d.regime_results || {}).length}
        </span>
        <span style={{ color: "var(--color-text-tertiary)", marginLeft: 4 }}>（需 ≥ 2）</span>
      </div>

      {/* 失敗模式 */}
      <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 8 }}>失敗模式檢查</div>
      <div style={{ borderRadius: "var(--border-radius-md)", overflow: "hidden",
        border: "0.5px solid var(--color-border-tertiary)", marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "10px 14px", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
          <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
            HHI 爆炸率
            <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginLeft: 6 }}>門檻 &lt; 15%</span>
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, fontWeight: 600,
              color: hhi_ok ? "var(--color-text-success)" : "var(--color-text-danger)" }}>
              {(d.hhi_explosion_rate * 100).toFixed(1)}%
            </span>
            <span>{hhi_ok ? "✓" : "✗"}</span>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "10px 14px" }}>
          <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
            Δ尾部風險 (5th pctile)
            <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginLeft: 6 }}>門檻 ≥ −0.005</span>
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--color-text-tertiary)" }}>
              基準 {d.tail_risk_baseline.toFixed(4)} → 候選 {d.tail_risk_candidate.toFixed(4)}
            </span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, fontWeight: 600,
              color: tail_ok ? "var(--color-text-success)" : "var(--color-text-danger)" }}>
              {d.delta_tail_risk >= 0 ? "+" : ""}{d.delta_tail_risk.toFixed(4)}
            </span>
            <span>{tail_ok ? "✓" : "✗"}</span>
          </div>
        </div>
      </div>

      {/* 外部基準 */}
      <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 8 }}>外部基準比較（Sharpe）</div>
      <div style={{ borderRadius: "var(--border-radius-md)", overflow: "hidden",
        border: "0.5px solid var(--color-border-tertiary)", marginBottom: 12 }}>
        <BenchmarkRow label="vs 等權重買持" candidateSharpe={d.candidate_sharpe}
          benchmarkSharpe={d.equal_weight_sharpe} beats={d.beats_equal_weight} />
        <BenchmarkRow label="vs 動能策略" candidateSharpe={d.candidate_sharpe}
          benchmarkSharpe={d.momentum_sharpe} beats={d.beats_momentum} />
      </div>

      {/* 危機歸因附加診斷 */}
      {d.extra_diagnostic && d.extra_diagnostic.metrics?.concentration !== undefined && (
        <CrisisAttributionPanel data={d.extra_diagnostic} />
      )}

      {d.rejection_reasons?.length > 0 && (
        <div style={reasonStyle}>
          {d.rejection_reasons.map((r, i) => <div key={i}>• {r}</div>)}
        </div>
      )}
    </div>
  );
}

function CrisisAttributionPanel({ data }) {
  const triggered = data.triggered;
  const m = data.metrics || {};
  return (
    <div style={{
      marginTop: 12, padding: "12px 16px",
      borderRadius: "var(--border-radius-md)",
      background: triggered ? "rgba(192,57,43,0.06)" : "rgba(29,158,117,0.05)",
      border: `0.5px solid ${triggered ? "rgba(192,57,43,0.3)" : "rgba(29,158,117,0.2)"}`,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: "#C0392B" }}>危機歸因診斷</span>
        <span style={{
          fontSize: 11, padding: "1px 8px", borderRadius: 4, fontWeight: 600,
          background: triggered ? "rgba(192,57,43,0.12)" : "rgba(29,158,117,0.12)",
          color: triggered ? "var(--color-text-danger)" : "var(--color-text-success)",
        }}>
          {triggered ? "🚨 FAIL（恐慌記憶）" : "✓ 通過"}
        </span>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 8 }}>
        {[
          { label: "Top3月貢獻集中度", value: `${((m.concentration || 0) * 100).toFixed(1)}%`, bad: (m.concentration || 0) > 0.5, threshold: "> 50%" },
          { label: "高風險月份佔比",   value: `${((m.high_vol_ratio || 0) * 100).toFixed(1)}%`, bad: (m.high_vol_ratio || 0) > 0.4, threshold: "> 40%" },
          { label: "正常期 ΔSharpe",   value: m.normal_sharpe != null ? m.normal_sharpe.toFixed(4) : "—", bad: m.normal_sharpe != null && m.normal_sharpe <= 0, threshold: "≤ 0 = FAIL" },
        ].map(({ label, value, bad, threshold }) => (
          <div key={label} style={{
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
            <span style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>{threshold}</span>
          </div>
        ))}
      </div>

      {m.top3_months?.length > 0 && (
        <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 4 }}>
          Top3 高貢獻月份：{m.top3_months.join("、")}
          {m.high_vol_months?.length > 0 && `　高風險月份：${m.high_vol_months.join("、")}`}
        </div>
      )}

      {triggered && data.triggered_rules?.length > 0 && (
        <div style={{ fontSize: 11, color: "var(--color-text-danger)", lineHeight: 1.7 }}>
          {data.triggered_rules.map((r, i) => <div key={i}>• {r}</div>)}
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
  padding: "14px 20px", fontSize: 13, color: "var(--color-text-tertiary)",
  background: "var(--color-background-secondary)",
  borderRadius: "var(--border-radius-lg)",
  border: "0.5px solid var(--color-border-tertiary)",
};
const reasonStyle = {
  padding: "10px 14px", borderRadius: "var(--border-radius-md)",
  background: "rgba(226,75,74,0.06)", border: "0.5px solid rgba(226,75,74,0.2)",
  fontSize: 12, color: "var(--color-text-danger)", lineHeight: 1.8,
};