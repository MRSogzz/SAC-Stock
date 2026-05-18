import MetricCard      from "../../components/MetricCard";
import PortfolioChart  from "../../components/PortfolioChart";
import PositionBar     from "../../components/PositionBar";
import TradeLog        from "../../components/TradeLog";
import PredictionPanel from "../../components/PredictionPanel";
import { useTraining } from "../../hooks/useTraining";
import { PERIOD_OPTIONS, STOCK_POOL, CAPITAL_OPTIONS } from "../../constants/config";
import { useState } from "react";

export default function StandardTraining() {
  const [period,     setPeriod]     = useState("6y");
  const [episodes,   setEpisodes]   = useState(80);
  const [capital,    setCapital]    = useState(1000000);
  const [valDays,    setValDays]    = useState(250);
  const [valCapital, setValCapital] = useState(100000);

  const {
    status, result, error, elapsed, progress, currentEp,
    prediction, predLoading,
    valResult, valLoading,
    startTraining, startValidation, fetchPrediction,
  } = useTraining();

  const aiReturn   = result?.total_return;
  const rfReturn   = result?.risk_free_return;
  const outperform = aiReturn != null && rfReturn != null
    ? (aiReturn - rfReturn).toFixed(2)
    : null;

  const isRunning = status === "running" || status === "starting";

  return (
    <>
      {/* 訓練設定 */}
      <div style={cardStyle}>
        <div style={sectionTitle}>訓練設定</div>
        {period !== "6y" && (
          <div style={warnBanner}>
            建議選擇「6 年」以配合 pos_252 年度位置特徵
          </div>
        )}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginBottom: 20 }}>
          <label style={labelWrap}>
            <span style={labelText}>歷史資料期間</span>
            <select value={period} onChange={(e) => setPeriod(e.target.value)} style={{ width: "100%" }}>
              {PERIOD_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </label>
          <label style={labelWrap}>
            <span style={labelText}>訓練回合：{episodes}</span>
            <input type="range" min={20} max={200} step={10} value={episodes}
              onChange={(e) => setEpisodes(+e.target.value)} style={{ width: "100%" }} />
          </label>
          <label style={labelWrap}>
            <span style={labelText}>初始資金</span>
            <select value={capital} onChange={(e) => setCapital(+e.target.value)} style={{ width: "100%" }}>
              {CAPITAL_OPTIONS.map((v) => (
                <option key={v} value={v}>{v >= 10000 ? (v / 10000) + "萬" : v}</option>
              ))}
            </select>
          </label>
          <label style={labelWrap}>
            <span style={labelText}>驗證段：{valDays} 天</span>
            <input type="range" min={60} max={500} step={10} value={valDays}
              onChange={(e) => setValDays(+e.target.value)} style={{ width: "100%" }} />
          </label>
        </div>

        <button onClick={() => startTraining({ period, episodes, capital, valDays })}
          disabled={isRunning} style={{ fontSize: 14, padding: "8px 24px" }}>
          {status === "running" ? `訓練中… ${elapsed}s` : status === "starting" ? "啟動中…" : "開始訓練 AI →"}
        </button>

        {status === "running" && (
          <div style={{ marginTop: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 8 }}>
              <span>Episode {currentEp} / {episodes}</span>
              <span>{elapsed}s</span>
            </div>
            <div style={{ height: 6, background: "var(--color-background-secondary)", borderRadius: 3, overflow: "hidden", marginBottom: 12 }}>
              <div style={{ height: "100%", borderRadius: 3, background: "#1D9E75",
                width: `${episodes > 0 ? Math.round(currentEp / episodes * 100) : 0}%`,
                transition: "width 0.4s ease" }} />
            </div>
            {progress.length > 0 && (
              <div style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)",
                padding: "10px 14px", maxHeight: 140, overflowY: "auto",
                fontFamily: "var(--font-mono)", fontSize: 12 }}>
                {[...progress].reverse().slice(0, 6).map((p, i) => (
                  <div key={i} style={{ display: "flex", gap: 16, padding: "3px 0",
                    borderBottom: "0.5px solid var(--color-border-tertiary)",
                    color: i === 0 ? "var(--color-text-primary)" : "var(--color-text-tertiary)" }}>
                    <span style={{ minWidth: 70 }}>ep {String(p.ep).padStart(3, "0")}/{p.total}</span>
                    <span style={{ minWidth: 80, color: p.ret >= 1 ? "var(--color-text-success)" : "var(--color-text-danger)" }}>
                      {p.ret >= 1 ? "+" : ""}{((p.ret - 1) * 100).toFixed(2)}%
                    </span>
                    <span>α={p.alpha}</span>
                    <span style={{ color: "var(--color-text-tertiary)" }}>{p.elapsed}s</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {error && (
          <div style={{ marginTop: 12, padding: "10px 14px", borderRadius: "var(--border-radius-md)",
            background: "var(--color-background-danger)", color: "var(--color-text-danger)", fontSize: 13 }}>
            {error}
          </div>
        )}
      </div>

      {/* 訓練結果 */}
      {result && (<>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 16 }}>
          <MetricCard label="初始資金" value={`$${result.initial_capital?.toLocaleString()}`} sub="投入本金" />
          <MetricCard label="最終資金" value={`$${result.final_capital?.toLocaleString()}`}
            sub={`報酬率 ${aiReturn >= 0 ? "+" : ""}${aiReturn}%`}
            color={result.final_capital >= result.initial_capital ? "var(--color-text-success)" : "var(--color-text-danger)"} />
          <MetricCard label="超過無風險利率" value={`${outperform >= 0 ? "+" : ""}${outperform}%`}
            sub={`定存基準 +${rfReturn}%`}
            color={outperform >= 0 ? "var(--color-text-success)" : "var(--color-text-danger)"} />
          <MetricCard label="交易勝率" value={`${result.win_rate}%`}
            sub={`共 ${result.n_trades} 筆交易`}
            color={result.win_rate >= 50 ? "var(--color-text-success)" : "var(--color-text-danger)"} />
        </div>

        {/* 明日持倉建議 */}
        <div style={cardStyle}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
            <div style={sectionTitle}>明日持倉建議</div>
            <button onClick={() => fetchPrediction(period)} disabled={predLoading} style={{ fontSize: 12, padding: "5px 14px" }}>
              {predLoading ? "預測中…" : "重新預測"}
            </button>
          </div>
          <PredictionPanel pred={prediction} />
        </div>

        {/* AI 平均持倉比例 */}
        {result.avg_positions && (
          <div style={cardStyle}>
            <div style={sectionTitle}>AI 平均持倉比例</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 24px" }}>
              {STOCK_POOL.filter((s) => s.id !== "0050").map((s) => (
                <PositionBar key={s.id} name={s.name} pct={result.avg_positions[s.id] || 0} sector={s.sector} />
              ))}
            </div>
          </div>
        )}

        {/* 報酬曲線 */}
        {result.portfolio_curve && (
          <div style={cardStyle}>
            <div style={sectionTitle}>報酬曲線</div>
            <PortfolioChart
              portfolio={result.portfolio_curve}
              bh={result.bh_curve}
              dates={result.dates}
            />
          </div>
        )}

        {/* 交易記錄 */}
        {result.trade_log && (
          <div style={cardStyle}>
            <div style={sectionTitle}>交易記錄</div>
            <TradeLog trades={result.trade_log} />
          </div>
        )}

        {/* 驗證區段 */}
        <div style={cardStyle}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
            <div>
              <div style={sectionTitle}>樣本外驗證</div>
              <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 2 }}>
                使用已訓練模型對指定期間進行回測
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <label style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
                驗證資金
                <select value={valCapital} onChange={(e) => setValCapital(+e.target.value)}
                  style={{ marginLeft: 8 }}>
                  {CAPITAL_OPTIONS.map((v) => (
                    <option key={v} value={v}>{v >= 10000 ? (v / 10000) + "萬" : v}</option>
                  ))}
                </select>
              </label>
              <button onClick={() => startValidation({ period, valDays, valCapital })}
                disabled={valLoading} style={{ fontSize: 12, padding: "6px 16px" }}>
                {valLoading ? "驗證中…" : "執行驗證"}
              </button>
            </div>
          </div>
          {valResult && (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 16 }}>
                <MetricCard label="驗證報酬" value={`${valResult.total_return >= 0 ? "+" : ""}${valResult.total_return}%`}
                  color={valResult.total_return >= 0 ? "var(--color-text-success)" : "var(--color-text-danger)"} />
                <MetricCard label="最大回撤" value={`${valResult.max_drawdown}%`} color="var(--color-text-danger)" />
                <MetricCard label="勝率" value={`${valResult.win_rate}%`}
                  color={valResult.win_rate >= 50 ? "var(--color-text-success)" : "var(--color-text-danger)"} />
                <MetricCard label="交易筆數" value={valResult.n_trades} />
              </div>
              <TradeLog trades={valResult.trade_log} />
            </>
          )}
        </div>
      </>)}
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
const warnBanner   = {
  background: "rgba(186,117,23,0.08)", border: "0.5px solid rgba(186,117,23,0.3)",
  borderRadius: "var(--border-radius-md)", padding: "10px 14px",
  marginBottom: 16, fontSize: 12, color: "#854f0b",
};