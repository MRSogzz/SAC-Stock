import VerdictBadge from "./VerdictBadge";

export default function ReportsListPanel({ reports, loading, onLoad, onRefresh }) {
  return (
    <div style={cardStyle}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
        <div style={{ fontSize: 14, fontWeight: 500 }}>已儲存的驗證報告</div>
        <button onClick={onRefresh} disabled={loading} style={{ fontSize: 12, padding: "5px 14px" }}>
          {loading ? "載入中…" : "重新整理"}
        </button>
      </div>

      {!reports.length && !loading && (
        <div style={{ fontSize: 13, color: "var(--color-text-tertiary)", textAlign: "center", padding: "20px 0" }}>
          尚無報告，執行驗證後會自動儲存至 reports/ 目錄
        </div>
      )}

      {reports.map((r, i) => {
        const verdictKey = r.final_verdict === "PASS" ? "PASS"
          : r.final_verdict === "FAIL" ? "FAIL"
          : "PARTIAL";
        return (
          <div key={i} style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            padding: "10px 14px", borderRadius: "var(--border-radius-md)",
            background: "var(--color-background-secondary)",
            marginBottom: 8, gap: 12,
          }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 500, fontSize: 13, marginBottom: 2 }}>{r.feature_name}</div>
              {r.description && (
                <div style={{ fontSize: 11, color: "var(--color-text-tertiary)",
                  overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {r.description}
                </div>
              )}
              {r.validation_period && (
                <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", fontFamily: "var(--font-mono)", marginTop: 2 }}>
                  {r.validation_period[0]} ～ {r.validation_period[1]}
                </div>
              )}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
              <VerdictBadge verdict={verdictKey} size="sm" />
              <button onClick={() => onLoad(r.feature_name)} style={{ fontSize: 12, padding: "4px 12px" }}>
                載入
              </button>
            </div>
          </div>
        );
      })}
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