import { useState } from "react";
import { FEATURE_GROUPS, ALL_FEATURE_IDS } from "./featureRegistry";

function FeatureChip({ feature, selected, onToggle, groupColor }) {
  return (
    <button
      onClick={() => onToggle(feature.id)}
      title={feature.desc}
      style={{
        display: "inline-flex", flexDirection: "column", alignItems: "flex-start",
        padding: "7px 11px", borderRadius: 6, cursor: "pointer",
        fontSize: 12, textAlign: "left", transition: "all 0.15s",
        background: selected ? groupColor + "18" : "var(--color-background-secondary)",
        border: `0.5px solid ${selected ? groupColor + "80" : "var(--color-border-tertiary)"}`,
        color: selected ? "var(--color-text-primary)" : "var(--color-text-secondary)",
        fontWeight: selected ? 500 : 400,
        outline: "none",
      }}
    >
      <span style={{
        fontFamily: "var(--font-mono)", fontSize: 11,
        color: selected ? groupColor : "inherit",
      }}>
        {feature.label}
      </span>
      <span style={{
        fontSize: 10, color: "var(--color-text-tertiary)",
        marginTop: 2, lineHeight: 1.3, maxWidth: 140,
        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
      }}>
        {feature.desc}
      </span>
      {feature.extraDiagnostic && (
        <span style={{
          marginTop: 3, fontSize: 9, padding: "1px 5px", borderRadius: 3,
          background: groupColor + "22", color: groupColor, fontWeight: 600,
          whiteSpace: "nowrap",
        }}>
          {feature.extraDiagnostic === "low_vol_exposure"   ? "＋低波動診斷" :
           feature.extraDiagnostic === "crisis_attribution" ? "＋危機歸因診斷" :
           feature.extraDiagnostic === "turnover_defense"   ? "＋換倉防禦診斷" : "＋附加診斷"}
        </span>
      )}
    </button>
  );
}

export default function FeatureSelector({ selected, onChange }) {
  const [expandedGroups, setExpandedGroups] = useState(
    Object.fromEntries(FEATURE_GROUPS.map((g) => [g.id, true]))
  );

  const toggleGroup = (gid) =>
    setExpandedGroups((s) => ({ ...s, [gid]: !s[gid] }));

  const toggleFeature = (fid) => {
    if (selected.includes(fid)) {
      onChange(selected.filter((id) => id !== fid));
    } else {
      onChange([...selected, fid]);
    }
  };

  const toggleGroupAll = (group) => {
    const ids = group.features.map((f) => f.id);
    const allSelected = ids.every((id) => selected.includes(id));
    if (allSelected) {
      onChange(selected.filter((id) => !ids.includes(id)));
    } else {
      const toAdd = ids.filter((id) => !selected.includes(id));
      onChange([...selected, ...toAdd]);
    }
  };

  const selectAll = () => onChange([...ALL_FEATURE_IDS]);
  const clearAll  = () => onChange([]);

  return (
    <div>
      {/* 頂部工具列 */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
          已選 <span style={{ fontWeight: 600, color: "var(--color-text-primary)" }}>{selected.length}</span> / {ALL_FEATURE_IDS.length} 個特徵
        </span>
        <button onClick={selectAll} style={smallBtn}>全選</button>
        <button onClick={clearAll}  style={smallBtn}>清除</button>
        <div style={{ marginLeft: "auto", fontSize: 11, color: "var(--color-text-tertiary)" }}>
          點擊特徵名稱選取；hover 可看說明
        </div>
      </div>

      {/* 分組特徵列表 */}
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {FEATURE_GROUPS.map((group) => {
          const groupIds = group.features.map((f) => f.id);
          const selectedCount = groupIds.filter((id) => selected.includes(id)).length;
          const allGroupSelected = selectedCount === groupIds.length;
          const expanded = expandedGroups[group.id];

          return (
            <div key={group.id} style={{
              border: `0.5px solid ${selectedCount > 0 ? group.color + "40" : "var(--color-border-tertiary)"}`,
              borderRadius: "var(--border-radius-md)",
              overflow: "hidden",
            }}>
              {/* 群組標題列 */}
              <div
                style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "8px 14px", cursor: "pointer",
                  background: selectedCount > 0 ? group.color + "08" : "var(--color-background-secondary)",
                  userSelect: "none",
                }}
                onClick={() => toggleGroup(group.id)}
              >
                <span style={{
                  display: "inline-block", width: 8, height: 8, borderRadius: "50%",
                  background: group.color, flexShrink: 0,
                }} />
                <span style={{ fontSize: 13, fontWeight: 500 }}>{group.label}</span>
                <span style={{
                  fontSize: 11, fontFamily: "var(--font-mono)",
                  color: selectedCount > 0 ? group.color : "var(--color-text-tertiary)",
                  marginLeft: 2,
                }}>
                  {selectedCount}/{groupIds.length}
                </span>
                <button
                  onClick={(e) => { e.stopPropagation(); toggleGroupAll(group); }}
                  style={{
                    ...smallBtn,
                    marginLeft: "auto",
                    color: allGroupSelected ? group.color : "var(--color-text-secondary)",
                  }}
                >
                  {allGroupSelected ? "全取消" : "全選此組"}
                </button>
                <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginLeft: 4 }}>
                  {expanded ? "▲" : "▼"}
                </span>
              </div>

              {/* 特徵 chips */}
              {expanded && (
                <div style={{
                  display: "flex", flexWrap: "wrap", gap: 6,
                  padding: "10px 14px",
                  borderTop: "0.5px solid var(--color-border-tertiary)",
                }}>
                  {group.features.map((f) => (
                    <FeatureChip
                      key={f.id}
                      feature={f}
                      selected={selected.includes(f.id)}
                      onToggle={toggleFeature}
                      groupColor={group.color}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const smallBtn = {
  fontSize: 11, padding: "3px 10px", cursor: "pointer",
  background: "transparent",
  border: "0.5px solid var(--color-border-tertiary)",
  borderRadius: 4, color: "var(--color-text-secondary)",
};