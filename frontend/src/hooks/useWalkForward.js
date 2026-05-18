import { useState, useRef, useEffect } from "react";
import { API } from "../constants/config";

export function useWalkForward() {
  const [status, setStatus]           = useState("idle"); // idle | starting | running | done | error
  const [error, setError]             = useState(null);
  const [elapsed, setElapsed]         = useState(0);
  const [result, setResult]           = useState(null);
  const [serverStatus, setServerStatus] = useState(null);
  const [prediction, setPrediction]   = useState(null);
  const [predLoading, setPredLoading] = useState(false);

  const pollRef  = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => () => {
    clearInterval(pollRef.current);
    clearInterval(timerRef.current);
  }, []);

  const fetchServerStatus = async () => {
    try {
      setServerStatus(await fetch(`${API}/walkforward_status`).then((r) => r.json()));
    } catch (e) {}
  };

  const fetchPrediction = async (period) => {
    setPredLoading(true);
    setPrediction(null);
    try {
      const res = await fetch(`${API}/predict_walkforward/${period}`);
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
      setPrediction(await res.json());
    } catch (e) {
      alert("Walk-forward 預測失敗：" + e.message);
    } finally {
      setPredLoading(false);
    }
  };

  const startTraining = async ({ period, episodes, capital }, onDone) => {
    setStatus("starting"); setError(null); setResult(null); setElapsed(0);
    try {
      const res = await fetch(`${API}/train_walkforward`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ period, episodes, initial_capital: capital }),
      });
      const { job_id } = await res.json();
      setStatus("running");
      timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
      pollRef.current = setInterval(async () => {
        const s = await fetch(`${API}/status/${job_id}`).then((r) => r.json());
        if (s.status === "done") {
          clearInterval(pollRef.current); clearInterval(timerRef.current);
          const r = await fetch(`${API}/result/${job_id}`).then((r) => r.json());
          setResult(r); setStatus("done");
          fetchServerStatus();
          fetchPrediction(period);
          onDone?.();
        } else if (s.status === "error") {
          clearInterval(pollRef.current); clearInterval(timerRef.current);
          setError(s.error); setStatus("error");
        }
      }, 3000);
    } catch (e) {
      setError("無法連接後端"); setStatus("error");
    }
  };

  return {
    // state
    status, error, elapsed, result, serverStatus,
    prediction, predLoading,
    // actions
    startTraining,
    fetchServerStatus,
    fetchPrediction,
  };
}