import { RUN_DESC } from "../../constants/config";

const RUN_IDS = ["A", "B", "C", "D"];
const WINDOWS  = [1, 2, 3];

const retColor = (v) =>
  v == null ? "var(--color-text-tertiary)" :
  v >= 0    ? "var(--color-text-success)"  :
              "var(--color-text-danger)";

const fmt = (v) => v != null ? `${v >= 0 ? "+" : ""}${v}%` : "—";

export default function RunComparisonTable({ runs }) {
  if (!runs) return null;

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
            <th style={{ padding: "8px 12px", textAlign: "left", fontWeight: 500, color: "var(--color-text-secondary)", minWidth: 80 }}>窗口</th>
            {RUN_IDS.map((id) => (
              <th key={id} colSpan={2} style={{ padding: "8px 12px", textAlign: "center", fontWeight: 500, borderLeft: "0.5px solid var(--color-border-tertiary)" }}>
                <span style={{ color: "var(--color-text-primary)" }}>Run {id}</span>
                <div style={{ fontSize: 10, fontWeight: 400, color: "var(--color-text-tertiary)", marginTop: 2 }}>{RUN_DESC[id]}</div>
              </th>
            ))}
          </tr>
          <tr style={{ borderBottom: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-secondary)" }}>
            <th style={{ padding: "4px 12px", textAlign: "left", color: "var(--color-text-tertiary)", fontWeight: 400 }}></th>
            {RUN_IDS.flatMap((id) => [
              <th key={id + "t"} style={{ padding: "4px 8px", textAlign: "center", color: "var(--color-text-tertiary)", fontWeight: 400, borderLeft: "0.5px solid var(--color-border-tertiary)" }}>訓練</th>,
              <th key={id + "v"} style={{ padding: "4px 8px", textAlign: "center", color: "var(--color-text-tertiary)", fontWeight: 400 }}>驗證</th>,
            ])}
          </tr>
        </thead>
        <tbody>
          {WINDOWS.map((w, wi) => (
            <tr key={w} style={{ borderBottom: "0.5px solid var(--color-border-tertiary)", background: wi % 2 === 0 ? "transparent" : "var(--color-background-secondary)" }}>
              <td style={{ padding: "10px 12px", fontWeight: 500 }}>
                窗口 {w}
                {RUN_IDS.map((id) => runs[id]?.windows?.find((x) => x.window === w)).find(Boolean) && (() => {
                  const d = RUN_IDS.map((id) => runs[id]?.windows?.find((x) => x.window === w)).find(Boolean);
                  return (
                    <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 2, fontWeight: 400 }}>
                      {d?.val_start?.slice(0, 7) || ""}~{d?.val_end?.slice(0, 7) || ""}
                    </div>
                  );
                })()}
              </td>
              {RUN_IDS.flatMap((id) => {
                const wd = runs[id]?.windows?.find((x) => x.window === w);
                return [
                  <td key={id + "t"} style={{ padding: "10px 8px", textAlign: "center", fontFamily: "var(--font-mono)", borderLeft: "0.5px solid var(--color-border-tertiary)", color: "var(--color-text-secondary)" }}>
                    {fmt(wd?.train_return)}
                  </td>,
                  <td key={id + "v"} style={{ padding: "10px 8px", textAlign: "center", fontFamily: "var(--font-mono)", fontWeight: 500, color: retColor(wd?.val_return) }}>
                    {fmt(wd?.val_return)}
                  </td>,
                ];
              })}
            </tr>
          ))}
          <tr style={{ borderTop: "1px solid var(--color-border-tertiary)", background: "var(--color-background-secondary)" }}>
            <td style={{ padding: "10px 12px", fontWeight: 500, fontSize: 11, color: "var(--color-text-secondary)" }}>平均驗證</td>
            {RUN_IDS.flatMap((id) => {
              const avg = runs[id]?.meta?.avg_val_return;
              return [
                <td key={id + "t"} style={{ borderLeft: "0.5px solid var(--color-border-tertiary)" }}></td>,
                <td key={id + "v"} style={{ padding: "10px 8px", textAlign: "center", fontFamily: "var(--font-mono)", fontWeight: 600, color: retColor(avg) }}>
                  {fmt(avg)}
                </td>,
              ];
            })}
          </tr>
        </tbody>
      </table>
    </div>
  );
}