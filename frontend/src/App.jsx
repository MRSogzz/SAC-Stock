import { useState, useEffect } from "react";

import StandardTraining    from "./features/StandardTraining";
import WalkForwardTraining from "./features/WalkForwardTraining";
import AlphaValidation     from "./features/AlphaValidation";
import SchedulerPanel      from "./features/SystemStatus/SchedulerPanel";
import SavedModelsTable    from "./features/SystemStatus/SavedModelsTable";
import { useScheduler }   from "./hooks/useScheduler";
import { STOCK_POOL }     from "./constants/config";

const TABS = [
  { id: "standard",    label: "單模型訓練"         },
  { id: "walkforward", label: "Walk-forward 訓練"  },
  { id: "alpha",       label: "Alpha 因子驗證"     },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("standard");

  const {
    scheduler, history, showHistory, savedModels,
    fetchScheduler, fetchHistory, fetchModels,
    runNow, toggleHistory,
  } = useScheduler();

  useEffect(() => {
    fetchModels();
    fetchScheduler();
    fetchHistory();
  }, []);

  return (
    <div style={{
      minHeight: "100vh",
      background: "var(--color-background-tertiary)",
      fontFamily: "var(--font-sans)",
      color: "var(--color-text-primary)",
    }}>
      <div style={{
        background: "var(--color-background-primary)",
        borderBottom: "0.5px solid var(--color-border-tertiary)",
        padding: "16px 24px",
        position: "sticky", top: 0, zIndex: 10,
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 500 }}>AI 投資組合管理系統</div>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 2 }}>
              Portfolio SAC · 10 支台股 · 31 特徵 · Walk-forward 驗證
            </div>
          </div>
          <div style={{
            fontSize: 11, color: "var(--color-text-tertiary)",
            background: "var(--color-background-secondary)",
            padding: "4px 10px", borderRadius: "var(--border-radius-md)",
          }}>台股 10 支</div>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          {TABS.map((t) => (
            <button key={t.id} style={tabStyle(activeTab === t.id)} onClick={() => setActiveTab(t.id)}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 960, margin: "0 auto", padding: "24px 16px" }}>
        {activeTab !== "alpha" && (
          <div style={cardStyle}>
            <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 12 }}>股票池（10 支）</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 8 }}>
              {STOCK_POOL.map((s) => (
                <div key={s.id} style={{
                  background: "var(--color-background-secondary)",
                  borderRadius: "var(--border-radius-md)",
                  padding: "8px 12px",
                }}>
                  <div style={{ fontSize: 13, fontWeight: 500 }}>{s.name}</div>
                  <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>{s.id} · {s.sector}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === "standard"    && <StandardTraining />}
        {activeTab === "walkforward" && <WalkForwardTraining />}
        {activeTab === "alpha"       && <AlphaValidation />}

        {activeTab !== "alpha" && (
          <>
            <SchedulerPanel
              scheduler={scheduler}
              history={history}
              showHistory={showHistory}
              onRunNow={runNow}
              onToggleHistory={toggleHistory}
            />
            <SavedModelsTable models={savedModels} />
          </>
        )}
      </div>
    </div>
  );
}

const tabStyle = (active) => ({
  padding: "8px 20px", fontSize: 14,
  fontWeight: active ? 500 : 400,
  color: active ? "var(--color-text-primary)" : "var(--color-text-secondary)",
  cursor: "pointer", background: "transparent", border: "none",
  borderBottom: active ? "2px solid #1D9E75" : "2px solid transparent",
});

const cardStyle = {
  background: "var(--color-background-primary)",
  border: "0.5px solid var(--color-border-tertiary)",
  borderRadius: "var(--border-radius-lg)",
  padding: "20px 24px", marginBottom: 16,
};