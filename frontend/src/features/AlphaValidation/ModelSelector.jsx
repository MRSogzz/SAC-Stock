import { useEffect } from "react";

export default function ModelSelector({ models, loading, selected, onSelect, onRefresh }) {
  // 預設選第一個
  useEffect(() => {
    if (models.length > 0 && !selected) onSelect(models[0]);
  }, [models]);

  if (loading) return (
    <div style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>載入模型清單中…</div>
  );

  if (!models.length) return (
    <div style={{ fontSize: 13, color: "var(--color-text-danger)" }}>
      找不到已儲存的模型，請先完成訓練。
    </div>
  );

  // 用 file 欄位做唯一識別
  const selectedFile = selected?.file ?? null;

  const handleClick = (m) => {
    if (m.file === selectedFile) {
      onSelect(null);   // 再次點擊取消選取
    } else {
      onSelect(m);
    }
  };

  return (
    <div>
      {selected && (
        <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 8 }}>
          已選：
          <span style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
            {selected.file}
          </span>
          <span style={{ marginLeft: 8 }}>（再次點擊可取消）</span>
        </div>
      )}

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {models.map((m, i) => {
          const isSelected = m.file === selectedFile;
          return (
            <button
              key={m.file || i}
              onClick={() => handleClick(m)}
              style={{
                padding: "10px 16px",
                borderRadius: "var(--border-radius-md)",
                cursor: "pointer",
                textAlign: "left",
                background: isSelected ? "rgba(29,158,117,0.08)" : "var(--color-background-secondary)",
                border: `0.5px solid ${isSelected ? "rgba(29,158,117,0.5)" : "var(--color-border-tertiary)"}`,
                outline: isSelected ? "1.5px solid rgba(29,158,117,0.3)" : "none",
                transition: "all 0.15s",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <span style={{
                  width: 8, height: 8, borderRadius: "50%", flexShrink: 0,
                  background: isSelected ? "#1D9E75" : "var(--color-border-tertiary)",
                  border: `1.5px solid ${isSelected ? "#1D9E75" : "var(--color-text-tertiary)"}`,
                  transition: "all 0.15s",
                }} />
                <span style={{ fontSize: 13, fontWeight: isSelected ? 600 : 400, fontFamily: "var(--font-mono)" }}>
                  {m.file}
                </span>
              </div>
              <div style={{ display: "flex", gap: 12, fontSize: 11, color: "var(--color-text-tertiary)", paddingLeft: 16 }}>
                {m.period    && <span>期間 {m.period}</span>}
                {m.episodes_done != null && <span>{m.episodes_done} ep</span>}
                {m.total_return != null && (
                  <span style={{
                    color: m.total_return >= 0 ? "var(--color-text-success)" : "var(--color-text-danger)",
                    fontFamily: "var(--font-mono)",
                  }}>
                    {m.total_return >= 0 ? "+" : ""}{m.total_return}%
                  </span>
                )}
                {m.saved_at  && <span>{String(m.saved_at).slice(0, 10)}</span>}
              </div>
            </button>
          );
        })}
      </div>

      <button onClick={onRefresh} style={{
        fontSize: 11, padding: "3px 10px", marginTop: 8,
        background: "transparent", border: "0.5px solid var(--color-border-tertiary)",
        borderRadius: 4, color: "var(--color-text-secondary)", cursor: "pointer",
      }}>
        重新整理模型清單
      </button>
    </div>
  );
}