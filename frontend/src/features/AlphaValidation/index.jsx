import { useState, useEffect } from "react";
import { useAlphaValidation }    from "../../hooks/useAlphaValidation";
import ReportSummaryCard         from "./ReportSummaryCard";
import Layer1Panel               from "./Layer1Panel";
import Layer2Panel               from "./Layer2Panel";
import Layer3Panel               from "./Layer3Panel";
import ReportsListPanel          from "./ReportsListPanel";
import ModelSelector             from "./ModelSelector";
import FeatureSelector           from "./FeatureSelector";
import { generateComputeFn, FEATURE_MAP } from "./featureRegistry";

const INPUT_MODES = [
  { id: "selector", label: "選取特徵" },
  { id: "manual",   label: "手動輸入程式碼" },
];

const DEFAULT_MANUAL_CODE = `def compute_fn(stocks):
    import pandas as pd
    result = {}
    for sid, df in stocks.items():
        c = df["Close"]
        feat = pd.DataFrame(index=df.index)
        feat["my_new_feature"] = c.pct_change(5).diff(3)
        result[sid] = feat
    return result
`;

export default function AlphaValidation() {
  // ── 表單狀態 ──────────────────────────────────────────────────────────────
  const [name,             setName]             = useState("");
  const [description,      setDescription]      = useState("");
  const [skipLayer1,       setSkipLayer1]        = useState(false);
  const [inputMode,        setInputMode]         = useState("selector");
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [manualCode,       setManualCode]        = useState(DEFAULT_MANUAL_CODE);
  const [showReports,      setShowReports]       = useState(false);

  // ── 批次執行狀態 ──────────────────────────────────────────────────────────
  const [batchQueue,    setBatchQueue]    = useState([]);   // 待執行的 feature id 陣列
  const [batchCurrent,  setBatchCurrent]  = useState(null); // 目前正在跑的 id
  const [batchResults,  setBatchResults]  = useState([]);   // [{id, report}]
  const [batchDone,     setBatchDone]     = useState(false);
  const [viewingResult, setViewingResult] = useState(null); // 目前顯示的報告

  const {
    status, report, error,
    reports, reportsLoading,
    models, modelsLoading, selectedModel, setSelectedModel,
    runValidation, fetchReports, loadReport, fetchModels, reset,
  } = useAlphaValidation();

  useEffect(() => { fetchModels(); }, []);

  // 單選時自動填入 name/desc
  const isSingleSelector = inputMode === "selector" && selectedFeatures.length === 1;
  const isMultiSelector  = inputMode === "selector" && selectedFeatures.length > 1;
  const isManual         = inputMode === "manual";

  useEffect(() => {
    if (isSingleSelector) {
      const f = FEATURE_MAP[selectedFeatures[0]];
      if (f) {
        setName(selectedFeatures[0]);
        setDescription(f.desc);
      }
    } else if (isMultiSelector) {
      // 多選時清空名稱欄，讓使用者知道不需要填
      setName("");
      setDescription("");
    }
  }, [selectedFeatures, inputMode]);

  // 批次執行：每當 queue 有東西且目前沒在跑就繼續
  useEffect(() => {
    if (batchQueue.length === 0 || status === "running") return;

    // 如果上一個跑完了，把結果存起來
    if (batchCurrent && status === "done" && report) {
      setBatchResults((prev) => {
        const exists = prev.find((r) => r.id === batchCurrent);
        if (exists) return prev;
        return [...prev, { id: batchCurrent, report }];
      });
      // 預設顯示第一個結果
      setViewingResult((v) => v ?? { id: batchCurrent, report });
    }

    const [next, ...rest] = batchQueue;
    setBatchQueue(rest);
    setBatchCurrent(next);

    if (rest.length === 0) setBatchDone(true);

    // 為這個特徵產生 compute_fn
    const f = FEATURE_MAP[next];
    const code = generateComputeFn([next], next);
    runValidation({
      name:          next,
      description:   f?.desc || "",
      featureCode:   code,
      featureColumn: next,
      skipLayer1,
      modelPath:     selectedModel?.file || null,
    });
  }, [batchQueue, status]);

  // 最後一個特徵完成後補存結果
  useEffect(() => {
    if (batchDone && batchCurrent && status === "done" && report) {
      setBatchResults((prev) => {
        const exists = prev.find((r) => r.id === batchCurrent);
        if (exists) return prev;
        const next = [...prev, { id: batchCurrent, report }];
        setViewingResult((v) => v ?? { id: batchCurrent, report: next[next.length - 1].report });
        return next;
      });
    }
  }, [status, batchDone]);

  // ── 執行邏輯 ──────────────────────────────────────────────────────────────
  const handleRun = () => {
    if (isManual && !manualCode.trim()) { alert("請輸入 compute_fn 程式碼"); return; }

    if (isMultiSelector) {
      // 多選：批次執行，不需要名稱
      if (selectedFeatures.length === 0) { alert("請至少選取一個特徵"); return; }
      setBatchResults([]);
      setViewingResult(null);
      setBatchDone(false);
      setBatchCurrent(null);
      reset();
      setBatchQueue([...selectedFeatures]);
      return;
    }

    // 單選 or 手動：單次執行
    const finalName = name.trim() || (isSingleSelector ? selectedFeatures[0] : "");
    if (!finalName) { alert("請輸入特徵名稱"); return; }

    setBatchResults([]);
    setViewingResult(null);
    setBatchDone(false);
    setBatchCurrent(null);

    if (isManual) {
      runValidation({
        name:          finalName,
        description,
        featureCode:   manualCode,
        featureColumn: finalName,
        skipLayer1,
        modelPath:     selectedModel?.file || null,
      });
    } else {
      // 單選 selector
      const code = generateComputeFn(selectedFeatures, finalName);
      runValidation({
        name:          finalName,
        description,
        featureCode:   code,
        featureColumn: finalName,
        skipLayer1,
        modelPath:     selectedModel?.file || null,
      });
    }
  };

  const handleReset = () => {
    reset();
    setBatchResults([]);
    setViewingResult(null);
    setBatchDone(false);
    setBatchCurrent(null);
    setBatchQueue([]);
    setName("");
    setDescription("");
  };

  const isRunning   = status === "running" || batchQueue.length > 0;
  const isBatchMode = isMultiSelector;
  const showResults = (status === "done" || batchResults.length > 0);

  // 目前顯示的報告
  const displayReport = viewingResult?.report ?? report;
  const displayStopAt = displayReport
    ? (!displayReport.layer1?.passed && !skipLayer1) ? 1
      : (!displayReport.layer2?.passed) ? 2
      : null
    : null;

  // 產生程式碼預覽（selector 模式）
  const codePreview = inputMode === "selector" && selectedFeatures.length > 0
    ? isSingleSelector
      ? generateComputeFn(selectedFeatures, name || selectedFeatures[0])
      : `# 多選模式：將為以下 ${selectedFeatures.length} 個特徵各別產生 compute_fn：\n` +
        selectedFeatures.map((id) => `#   ${id}`).join("\n")
    : null;

  return (
    <>
      <div style={infoBanner}>
        <div style={{ fontWeight: 500, marginBottom: 4 }}>Alpha 因子驗證管線（3層）</div>
        <div style={{ color: "var(--color-text-secondary)", fontSize: 12, lineHeight: 1.8 }}>
          Layer 1：Rank IC & 方向準確率（入場券）→
          Layer 2：黃金模型 ΔSharpe 策略增益（核心）→
          Layer 3：多環境穩定性 & 失敗模式（最終否決閥）
        </div>
      </div>

      {/* Step 1：基準模型 */}
      <div style={cardStyle}>
        <div style={stepHeader}>
          <StepBadge n={1} />
          <div>
            <div style={{ fontSize: 14, fontWeight: 500 }}>選擇基準模型</div>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 2 }}>
              Layer 2 會用此模型做「加入特徵前後」的策略回測對比
            </div>
          </div>
        </div>
        <ModelSelector
          models={models} loading={modelsLoading}
          selected={selectedModel} onSelect={setSelectedModel} onRefresh={fetchModels}
        />
      </div>

      {/* Step 2：定義特徵 */}
      <div style={cardStyle}>
        <div style={stepHeader}>
          <StepBadge n={2} />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 14, fontWeight: 500 }}>定義候選特徵</div>
            {isBatchMode && (
              <div style={{ fontSize: 12, color: "#378ADD", marginTop: 2 }}>
                批次模式：已選 {selectedFeatures.length} 個特徵，將依序各別驗證
              </div>
            )}
          </div>
          <div style={{ display: "flex", borderRadius: 6, overflow: "hidden",
            border: "0.5px solid var(--color-border-tertiary)" }}>
            {INPUT_MODES.map((m) => (
              <button key={m.id} onClick={() => setInputMode(m.id)} style={{
                padding: "6px 14px", fontSize: 12, cursor: "pointer", border: "none",
                background: inputMode === m.id ? "var(--color-background-secondary)" : "transparent",
                fontWeight: inputMode === m.id ? 500 : 400,
                color: inputMode === m.id ? "var(--color-text-primary)" : "var(--color-text-secondary)",
              }}>{m.label}</button>
            ))}
          </div>
        </div>

        {/* 名稱欄：多選時隱藏，單選或手動時顯示 */}
        {!isBatchMode && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 16 }}>
            <label style={labelWrap}>
              <span style={labelText}>特徵名稱（英文）</span>
              <input value={name} onChange={(e) => setName(e.target.value)}
                placeholder={isSingleSelector ? selectedFeatures[0] : "例：my_alpha"}
                style={{ width: "100%", boxSizing: "border-box" }} />
            </label>
            <label style={labelWrap}>
              <span style={labelText}>說明</span>
              <input value={description} onChange={(e) => setDescription(e.target.value)}
                placeholder="例：5日報酬的3日加速度"
                style={{ width: "100%", boxSizing: "border-box" }} />
            </label>
          </div>
        )}

        {inputMode === "selector" && (
          <>
            <FeatureSelector selected={selectedFeatures} onChange={setSelectedFeatures} />
            {codePreview && (
              <div style={{ marginTop: 16 }}>
                <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 6 }}>
                  {isBatchMode ? `批次計劃（${selectedFeatures.length} 個特徵各別驗證）` : "自動產生的 compute_fn 預覽"}
                </div>
                <pre style={{
                  background: "var(--color-background-secondary)",
                  border: "0.5px solid var(--color-border-tertiary)",
                  borderRadius: "var(--border-radius-md)",
                  padding: "12px 14px", margin: 0,
                  fontFamily: "var(--font-mono)", fontSize: 12,
                  color: "var(--color-text-secondary)",
                  overflowX: "auto", maxHeight: 200, overflowY: "auto",
                  lineHeight: 1.6, whiteSpace: "pre",
                }}>{codePreview}</pre>
              </div>
            )}
          </>
        )}

        {isManual && (
          <label style={{ display: "block" }}>
            <span style={labelText}>compute_fn 程式碼</span>
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 6 }}>
              函數名稱固定為 <code style={inlineCode}>compute_fn</code>，接受{" "}
              <code style={inlineCode}>{`{sid: pd.DataFrame}`}</code>，回傳同格式含新特徵欄位的 DataFrame。
            </div>
            <textarea value={manualCode} onChange={(e) => setManualCode(e.target.value)}
              spellCheck={false} style={{
                width: "100%", boxSizing: "border-box", minHeight: 220, resize: "vertical",
                fontFamily: "var(--font-mono)", fontSize: 13,
                background: "var(--color-background-secondary)",
                border: "0.5px solid var(--color-border-tertiary)",
                borderRadius: "var(--border-radius-md)",
                padding: "12px 14px", color: "var(--color-text-primary)", lineHeight: 1.6,
              }} />
          </label>
        )}
      </div>

      {/* Step 3：驗證設定 & 執行 */}
      <div style={cardStyle}>
        <div style={stepHeader}>
          <StepBadge n={3} />
          <div style={{ fontSize: 14, fontWeight: 500 }}>驗證設定</div>
        </div>

        <label style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
          <input type="checkbox" checked={skipLayer1} onChange={(e) => setSkipLayer1(e.target.checked)} />
          <span style={{ fontSize: 13 }}>
            跳過 Layer 1
            <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginLeft: 4 }}>
              （直接測試策略增益）
            </span>
          </span>
        </label>

        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <button onClick={handleRun} disabled={isRunning} style={{ fontSize: 14, padding: "9px 28px" }}>
            {isRunning
              ? batchQueue.length > 0
                ? `批次驗證中 ${batchResults.length + 1}/${selectedFeatures.length}…`
                : "驗證中…"
              : isBatchMode
                ? `批次驗證 ${selectedFeatures.length} 個特徵 →`
                : "開始驗證 →"}
          </button>
          {showResults && (
            <button onClick={handleReset} style={{ fontSize: 14, padding: "9px 20px", cursor: "pointer",
              background: "transparent", border: "0.5px solid var(--color-border-tertiary)",
              borderRadius: "var(--border-radius-md)" }}>
              重置
            </button>
          )}
          <button onClick={() => { setShowReports((v) => !v); fetchReports(); }}
            style={{ fontSize: 14, padding: "9px 20px", marginLeft: "auto", cursor: "pointer",
              background: "transparent", border: "0.5px solid var(--color-border-tertiary)",
              borderRadius: "var(--border-radius-md)" }}>
            {showReports ? "隱藏報告庫" : "查看歷史報告"}
          </button>
        </div>

        {/* 批次進度條 */}
        {isBatchMode && (isRunning || batchResults.length > 0) && (
          <div style={{ marginTop: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12,
              color: "var(--color-text-secondary)", marginBottom: 6 }}>
              <span>
                {batchDone && !isRunning
                  ? `批次完成：${batchResults.length} 個特徵`
                  : `正在驗證 ${batchCurrent}（${batchResults.length + 1} / ${selectedFeatures.length}）`}
              </span>
              <span>{Math.round((batchResults.length / selectedFeatures.length) * 100)}%</span>
            </div>
            <div style={{ height: 6, background: "var(--color-background-secondary)", borderRadius: 3 }}>
              <div style={{
                height: "100%", borderRadius: 3, background: "#1D9E75",
                width: `${(batchResults.length / selectedFeatures.length) * 100}%`,
                transition: "width 0.4s ease",
              }} />
            </div>

            {/* 各特徵結果小卡 */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
              {selectedFeatures.map((id) => {
                const result = batchResults.find((r) => r.id === id);
                const isCurrent = batchCurrent === id && status === "running";
                const verdict = result?.report?.final_verdict;
                const isViewing = viewingResult?.id === id;

                return (
                  <button key={id} onClick={() => result && setViewingResult(result)}
                    disabled={!result} style={{
                      padding: "5px 12px", borderRadius: 6, fontSize: 12,
                      cursor: result ? "pointer" : "default",
                      fontFamily: "var(--font-mono)",
                      background: isViewing
                        ? "rgba(29,158,117,0.12)"
                        : isCurrent
                          ? "rgba(55,138,221,0.08)"
                          : result
                            ? "var(--color-background-secondary)"
                            : "transparent",
                      border: `0.5px solid ${
                        isViewing    ? "rgba(29,158,117,0.5)"  :
                        isCurrent    ? "rgba(55,138,221,0.4)"  :
                        verdict === "PASS" ? "rgba(29,158,117,0.3)" :
                        verdict === "FAIL" ? "rgba(226,75,74,0.3)"  :
                        "var(--color-border-tertiary)"
                      }`,
                      color: isCurrent ? "#378ADD" :
                        verdict === "PASS" ? "#0f6e56" :
                        verdict === "FAIL" ? "#993c1d" :
                        "var(--color-text-tertiary)",
                      fontWeight: isViewing ? 600 : 400,
                    }}>
                    {isCurrent && <span style={{ marginRight: 4 }}>⟳</span>}
                    {verdict === "PASS" && <span style={{ marginRight: 4 }}>✅</span>}
                    {verdict === "FAIL" && <span style={{ marginRight: 4 }}>❌</span>}
                    {id}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {!isBatchMode && isRunning && (
          <div style={{ marginTop: 14, fontSize: 13, color: "var(--color-text-secondary)",
            display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%",
              background: "#1D9E75", animation: "pulse 1.2s ease-in-out infinite" }} />
            正在執行三層驗證，視資料量約需 30–120 秒…
          </div>
        )}

        {error && !isRunning && (
          <div style={{ marginTop: 12, padding: "10px 14px", borderRadius: "var(--border-radius-md)",
            background: "rgba(226,75,74,0.06)", border: "0.5px solid rgba(226,75,74,0.2)",
            fontSize: 13, color: "var(--color-text-danger)" }}>
            {error}
          </div>
        )}
      </div>

      {/* 歷史報告庫 */}
      {showReports && (
        <ReportsListPanel reports={reports} loading={reportsLoading}
          onLoad={(n) => { loadReport(n); setShowReports(false); }}
          onRefresh={fetchReports} />
      )}

      {/* 結果展示 */}
      {displayReport && (
        <>
          <ReportSummaryCard report={displayReport} />
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <Layer1Panel layer1={displayReport.layer1} stopped={displayStopAt === 1} />
            <Layer2Panel layer2={displayReport.layer2} stopAt={displayStopAt} />
            <Layer3Panel layer3={displayReport.layer3} stopAt={displayStopAt} />
          </div>
        </>
      )}

      <style>{`@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.3)} }`}</style>
    </>
  );
}

function StepBadge({ n }) {
  return (
    <div style={{
      width: 26, height: 26, borderRadius: "50%", flexShrink: 0,
      background: "var(--color-background-secondary)",
      border: "0.5px solid var(--color-border-tertiary)",
      display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: 12, fontWeight: 600, color: "var(--color-text-secondary)",
    }}>{n}</div>
  );
}

const cardStyle  = { background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", padding: "20px 24px", marginBottom: 16 };
const stepHeader = { display: "flex", alignItems: "center", gap: 10, marginBottom: 16 };
const infoBanner = { background: "rgba(55,138,221,0.06)", border: "0.5px solid rgba(55,138,221,0.2)", borderRadius: "var(--border-radius-lg)", padding: "14px 20px", marginBottom: 16, fontSize: 13 };
const labelWrap  = { display: "block" };
const labelText  = { fontSize: 12, color: "var(--color-text-secondary)", display: "block", marginBottom: 6 };
const inlineCode = { fontFamily: "var(--font-mono)", fontSize: 11, background: "var(--color-background-secondary)", padding: "1px 5px", borderRadius: 3 };