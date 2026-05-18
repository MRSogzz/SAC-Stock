// Scheduler status + history (shared between both tabs)

export default function SchedulerPanel({ scheduler, history, showHistory, onRunNow, onToggleHistory }) {
  const HISTORY_HEADERS = ["預測日期", "股票", "建議", "目標倉位", "當時價格", "實際報酬"];

  return (
    <div style={cardStyle}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 4 }}>每日自動預測排程</div>
          <div style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>每週一至週五 15:30 自動執行</div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={onRunNow} style={{ fontSize: 12, padding: "6px 14px" }}>立即執行</button>
          <button onClick={onToggleHistory} style={{ fontSize: 12, padding: "6px 14px" }}>
            {showHistory ? "隱藏歷史" : "查看歷史"}
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
        <StatusCard label="排程狀態"
          value={scheduler?.running ? "執行中" : "已停止"}
          color={scheduler?.running ? "var(--color-text-success)" : "var(--color-text-danger)"} />
        <StatusCard label="下次執行" value={scheduler?.next_run?.slice(0, 16) || "—"} mono />
        <StatusCard label="歷史紀錄" value={`${scheduler?.history_count || 0} 筆`} />
      </div>

      {showHistory && history.length > 0 && (
        <div style={{ marginTop: 16, overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
                {HISTORY_HEADERS.map((h) => (
                  <th key={h} style={{ padding: "6px 10px", textAlign: "left", fontWeight: 500, color: "var(--color-text-secondary)" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.map((r, i) => {
                const isPos = r.actual_return?.startsWith("+");
                const isNeg = r.actual_return?.startsWith("-");
                return (
                  <tr key={i} style={{ borderBottom: "0.5px solid var(--color-border-tertiary)", background: i % 2 === 0 ? "transparent" : "var(--color-background-secondary)" }}>
                    <td style={{ padding: "6px 10px", fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>{r.date}</td>
                    <td style={{ padding: "6px 10px", fontWeight: 500 }}>
                      {r.stock_name}
                      <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginLeft: 4 }}>{r.stock_id}</span>
                    </td>
                    <td style={{ padding: "6px 10px" }}>
                      <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 11,
                        background: r.action === "買入" ? "rgba(29,158,117,0.1)" : r.action === "持有" ? "rgba(55,138,221,0.1)" : "var(--color-background-secondary)",
                        color: r.action === "買入" ? "#0f6e56" : r.action === "持有" ? "#185FA5" : "var(--color-text-tertiary)" }}>
                        {r.action}
                      </span>
                    </td>
                    <td style={{ padding: "6px 10px", fontFamily: "var(--font-mono)" }}>{r.target_pct}%</td>
                    <td style={{ padding: "6px 10px", fontFamily: "var(--font-mono)" }}>${parseFloat(r.latest_price).toLocaleString()}</td>
                    <td style={{ padding: "6px 10px", fontFamily: "var(--font-mono)", fontWeight: 500,
                      color: isPos ? "var(--color-text-success)" : isNeg ? "var(--color-text-danger)" : "var(--color-text-tertiary)" }}>
                      {r.actual_return || "待回填"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function StatusCard({ label, value, color, mono }) {
  return (
    <div style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "12px 16px" }}>
      <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: mono ? 13 : 14, fontWeight: 500, fontFamily: mono ? "var(--font-mono)" : undefined, color: color || "var(--color-text-primary)" }}>
        {value}
      </div>
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