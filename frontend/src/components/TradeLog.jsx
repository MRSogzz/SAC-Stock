const HEADERS = ["日期", "股票", "操作", "價格", "股數", "金額", "手續費", "損益", "倉位"];

export default function TradeLog({ trades }) {
  if (!trades?.length) {
    return (
      <div style={{ fontSize: 13, color: "var(--color-text-tertiary)", padding: "20px 0", textAlign: "center" }}>
        尚無交易紀錄
      </div>
    );
  }

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
            {HEADERS.map((h) => (
              <th key={h} style={{
                padding: "8px 10px", textAlign: "left",
                fontWeight: 500, fontSize: 12, color: "var(--color-text-secondary)",
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {trades.map((t, i) => {
            const isBuy = t.action?.includes("買入");
            return (
              <tr key={i} style={{
                borderBottom: "0.5px solid var(--color-border-tertiary)",
                background: i % 2 === 0 ? "transparent" : "var(--color-background-secondary)",
              }}>
                <td style={{ padding: "8px 10px", color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)", fontSize: 12 }}>{t.date}</td>
                <td style={{ padding: "8px 10px", fontWeight: 500 }}>
                  {t.stock_name}
                  <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginLeft: 4 }}>{t.stock}</span>
                </td>
                <td style={{ padding: "8px 10px" }}>
                  <span style={{
                    display: "inline-block", padding: "2px 8px", borderRadius: 4, fontSize: 12, fontWeight: 500,
                    background: isBuy ? "rgba(29,158,117,0.1)" : "rgba(226,75,74,0.1)",
                    color: isBuy ? "#0f6e56" : "#993c1d",
                  }}>{t.action}</span>
                </td>
                <td style={{ padding: "8px 10px", fontFamily: "var(--font-mono)" }}>${t.price?.toLocaleString()}</td>
                <td style={{ padding: "8px 10px", fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                  {t.shares?.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </td>
                <td style={{ padding: "8px 10px", fontFamily: "var(--font-mono)" }}>${t.amount?.toLocaleString()}</td>
                <td style={{ padding: "8px 10px", fontFamily: "var(--font-mono)", color: "var(--color-text-tertiary)" }}>
                  {t.fee != null ? "$" + t.fee?.toLocaleString() : "—"}
                </td>
                <td style={{
                  padding: "8px 10px", fontFamily: "var(--font-mono)", fontWeight: 500,
                  color: t.profit == null ? "var(--color-text-tertiary)" : t.profit >= 0 ? "var(--color-text-success)" : "var(--color-text-danger)",
                }}>
                  {t.profit == null ? "—" : (t.profit >= 0 ? "+" : "") + "$" + t.profit?.toLocaleString()}
                </td>
                <td style={{ padding: "8px 10px", fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                  {t.position != null ? (t.position * 100).toFixed(0) + "%" : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}