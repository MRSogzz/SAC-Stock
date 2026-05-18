import { useState } from "react";
import { API } from "../constants/config";

export function useScheduler() {
  const [scheduler, setScheduler]   = useState(null);
  const [history, setHistory]       = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [savedModels, setSavedModels] = useState([]);

  const fetchScheduler = async () => {
    try {
      setScheduler(await fetch(`${API}/scheduler/status`).then((r) => r.json()));
    } catch (e) {}
  };

  const fetchHistory = async () => {
    try {
      const d = await fetch(`${API}/scheduler/history?limit=50`).then((r) => r.json());
      setHistory(d.records || []);
    } catch (e) {}
  };

  const fetchModels = async () => {
    try {
      const d = await fetch(`${API}/models`).then((r) => r.json());
      setSavedModels(d.models || []);
    } catch (e) {}
  };

  const runNow = async () => {
    try {
      await fetch(`${API}/scheduler/run-now`, { method: "POST" });
      setTimeout(() => { fetchHistory(); fetchScheduler(); }, 2000);
    } catch (e) {
      alert("執行失敗：" + e.message);
    }
  };

  const toggleHistory = () => {
    setShowHistory((h) => !h);
    fetchHistory();
  };

  return {
    // state
    scheduler, history, showHistory, savedModels,
    // actions
    fetchScheduler, fetchHistory, fetchModels,
    runNow, toggleHistory,
  };
}