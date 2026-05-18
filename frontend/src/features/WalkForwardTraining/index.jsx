import { useState } from "react";
import PredictionPanel       from "../../components/PredictionPanel";
import RunComparisonTable    from "./RunComparisonTable";
import { useWalkForward }   from "../../hooks/useWalkForward";
import {
  PERIOD_OPTIONS, CAPITAL_OPTIONS, REGIME_LABEL, REGIME_COLOR,
  RUN_DESC, WALK_FORWARD_WINDOWS,
} from "../../constants/config";

export default function WalkForwardTraining() {
  const [period,    setPeriod]    = useState("6y");
  const [wfEpisodes, setWfEpisodes] = useState(600);
  const [capital,   setCapital]   = useState(1000000);

  const {
    status, error, elapsed, serverStatus,
    prediction, predLoading,
    startTraining, fetchServerStatus, fetchPrediction,
  } = useWalkForward();

  const isRunning = status === "running" || status === "starting";

  return (
    <>
      {/* 說明橫幅 */}
      <div style={infoBanner}>
        <div style={{ fontWeight: 500, marginBottom: 6 }}>Walk-forward 訓練策略</div>
        <div style={{ color: "var(--color-text-secondary)", lineHeight: 1.7 }}>
          將資料切成 3 個滾動窗口（各 3 年訓練 + 1 年驗證），每個窗口獨立訓練一個模型。
          預測時根據當前市場環境（牛市/熊市/盤整）自動選擇最適合的模型，提升策略泛化能力。
        </div>
      </div>

      {/* 設定 */}
      <div style={cardStyle}>
        <div style={sectionTitle}>Walk-forward 設定</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16, marginBottom: 20 }}>
          <label style={labelWrap}>
            <span style={labelText}>資料期間（建議 6y）</span>
            <select value={period} onChange={(e) => setPeriod(e.target.value)} style={{ width: "100%" }}>
              {PERIOD_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </label>
          <label style={labelWrap}>
            <span style={labelText}>每窗口回合：{wfEpisodes}</span>
            <input type="range" min={10} max={700} step={10} value={wfEpisodes}
              onChange={(e) => setWfEpisodes(+e.target.value)} style={{ width: "100%" }} />
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 4 }}>
              總計約 {wfEpisodes * 3} 回合（3 個窗口）
            </div>
          </label>
          <label style={labelWrap}>
            <span style={labelText}>初始資金</span>
            <select value={capital} onChange={(e) => setCapital(+e.target.value)} style={{ width: "100%" }}>
              {CAPITAL_OPTIONS.map((v) => (
                <option key={v} value={v}>{v >= 10000 ? (v / 10000) + "萬" : v}</option>
              ))}
            </select>
          </label>
        </div>

        {/* 窗口預覽 */}
        <div style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "12px 16px", marginBottom: 16, fontSize: 12 }}>
          <div style={{ fontWeight: 500, marginBottom: 8, color: "var(--color-text-secondary)" }}>窗口規劃（3年訓練 + 1年驗證）</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 8 }}>
            {WALK_FORWARD_WINDOWS.map(({ w, train, val, regime }) => (
              <div key={w} style={{ background: "var(--color-background-primary)", borderRadius: "var(--border-radius-md)", padding: "10px 12px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontWeight: 500 }}>窗口 {w}</span>
                  <span style={{ color: REGIME_COLOR[regime] }}>{REGIME_LABEL[regime]}</span>
                </div>
                <div style={{ color: "var(--color-text-tertiary)", fontSize: 11 }}>
                  訓練 {train}<br />驗證 {val}
                </div>
              </div>
            ))}
          </div>
        </div>

        <button onClick={() => startTraining({ period, episodes: wfEpisodes, capital })}
          disabled={isRunning} style={{ fontSize: 14, padding: "8px 24px" }}>
          {status === "running" ? `Walk-forward 訓練中… ${elapsed}s` :
           status === "starting" ? "啟動中…" : "開始 Walk-forward 訓練 →"}
        </button>

        {status === "running" && (
          <div style={{ marginTop: 12, padding: "10px 14px", borderRadius: "var(--border-radius-md)",
            background: "var(--color-background-secondary)", fontSize: 12, color: "var(--color-text-secondary)" }}>
            正在訓練 3 個窗口，每個窗口 {wfEpisodes} 回合。訓練時間約 {Math.round(wfEpisodes * 3 * 2 / 60)} 分鐘…
          </div>
        )}
        {error && (
          <div style={{ marginTop: 12, padding: "10px 14px", borderRadius: "var(--border-radius-md)",
            background: "var(--color-background-danger)", color: "var(--color-text-danger)", fontSize: 13 }}>
            {error}
          </div>
        )}
      </div>

      {/* Run A~D 對比績效表 */}
      {serverStatus?.runs && (
        <div style={cardStyle}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
            <div>
              <div style={sectionTitle}>Run A ~ D 策略對比</div>
              <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 2 }}>
                3 個窗口 × 4 種策略的訓練集 / 驗證集報酬
              </div>
            </div>
            <button onClick={fetchServerStatus} style={{ fontSize: 12, padding: "4px 12px" }}>重新整理</button>
          </div>
          <RunComparisonTable runs={serverStatus.runs} />
        </div>
      )}

      {/* Walk-forward 預測 */}
      <div style={cardStyle}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
          <div>
            <div style={sectionTitle}>明日持倉建議（Regime 選模型）</div>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 2 }}>
              根據當前市場環境自動選擇最適合的窗口模型
            </div>
          </div>
          <button onClick={() => fetchPrediction(period)} disabled={predLoading} style={{ fontSize: 12, padding: "5px 14px" }}>
            {predLoading ? "預測中…" : "取得預測"}
          </button>
        </div>

        {prediction && (
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
            {prediction.selected_run && (
              <div style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 12px", borderRadius: 6, fontSize: 12, fontWeight: 500,
                background: "rgba(29,158,117,0.1)", border: "0.5px solid rgba(29,158,117,0.3)", color: "#0f6e56" }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#1D9E75", display: "inline-block" }} />
                Run {prediction.selected_run}
                <span style={{ fontWeight: 400, color: "var(--color-text-secondary)" }}>
                  — {RUN_DESC[prediction.selected_run] || ""}
                </span>
              </div>
            )}
            {prediction.selected_window && (
              <div style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 12px", borderRadius: 6, fontSize: 12,
                background: "var(--color-background-secondary)", color: "var(--color-text-secondary)" }}>
                窗口 {prediction.selected_window}
              </div>
            )}
            {prediction.current_regime && (
              <div style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 12px", borderRadius: 6, fontSize: 12,
                background: "var(--color-background-secondary)",
                color: REGIME_COLOR[prediction.current_regime] || "var(--color-text-secondary)" }}>
                {REGIME_LABEL[prediction.current_regime] || prediction.current_regime}
              </div>
            )}
          </div>
        )}
        <PredictionPanel pred={prediction} />
      </div>
    </>
  );
}

// ── local style helpers ──────────────────────────────────────────────────────
const cardStyle = {
  background: "var(--color-background-primary)",
  border: "0.5px solid var(--color-border-tertiary)",
  borderRadius: "var(--border-radius-lg)",
  padding: "20px 24px",
  marginBottom: 16,
};
const sectionTitle = { fontSize: 14, fontWeight: 500, marginBottom: 12 };
const labelWrap    = { display: "block" };
const labelText    = { fontSize: 12, color: "var(--color-text-secondary)", display: "block", marginBottom: 6 };
const infoBanner   = {
  background: "rgba(29,158,117,0.06)", border: "0.5px solid rgba(29,158,117,0.2)",
  borderRadius: "var(--border-radius-lg)", padding: "16px 20px",
  marginBottom: 16, fontSize: 13,
};