import { useState, useRef, useEffect } from "react";
import { API } from "../constants/config";

export function useTraining() {
  const [status, setStatus]       = useState("idle"); // idle | starting | running | done | error
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const [elapsed, setElapsed]     = useState(0);
  const [progress, setProgress]   = useState([]);
  const [currentEp, setCurrentEp] = useState(0);
  const [prediction, setPrediction]   = useState(null);
  const [predLoading, setPredLoading] = useState(false);
  const [valResult, setValResult]     = useState(null);
  const [valLoading, setValLoading]   = useState(false);

  const pollRef  = useRef(null);
  const timerRef = useRef(null);

  // Cleanup on unmount
  useEffect(() => () => {
    clearInterval(pollRef.current);
    clearInterval(timerRef.current);
  }, []);

  const fetchPrediction = async (period) => {
    setPredLoading(true);
    setPrediction(null);
    try {
      const res = await fetch(`${API}/predict/${period}`);
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
      setPrediction(await res.json());
    } catch (e) {
      alert("預測失敗：" + e.message);
    } finally {
      setPredLoading(false);
    }
  };

  const startTraining = async ({ period, episodes, capital, valDays }) => {
    setStatus("starting"); setError(null); setResult(null);
    setElapsed(0); setProgress([]); setCurrentEp(0);
    try {
      const res = await fetch(`${API}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ period, episodes, initial_capital: capital, val_days: valDays }),
      });
      const { job_id } = await res.json();
      setStatus("running");
      timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
      pollRef.current = setInterval(async () => {
        const s = await fetch(`${API}/status/${job_id}`).then((r) => r.json());
        if (s.progress) setProgress(s.progress);
        if (s.current_ep) setCurrentEp(s.current_ep);
        if (s.status === "done") {
          clearInterval(pollRef.current); clearInterval(timerRef.current);
          const r = await fetch(`${API}/result/${job_id}`).then((r) => r.json());
          setResult(r); setStatus("done");
          fetchPrediction(period);
        } else if (s.status === "error") {
          clearInterval(pollRef.current); clearInterval(timerRef.current);
          setError(s.error); setStatus("error");
        }
      }, 2000);
    } catch (e) {
      setError("無法連接後端"); setStatus("error");
    }
  };

  const startValidation = async ({ period, valDays, valCapital }, onDone) => {
    setValLoading(true); setValResult(null);
    try {
      const res = await fetch(`${API}/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ period, val_days: valDays, initial_capital: valCapital }),
      });
      const { job_id } = await res.json();
      const poll = setInterval(async () => {
        const s = await fetch(`${API}/status/${job_id}`).then((r) => r.json());
        if (s.status === "done") {
          clearInterval(poll);
          setValResult(await fetch(`${API}/result/${job_id}`).then((r) => r.json()));
          setValLoading(false);
          onDone?.();
        } else if (s.status === "error") {
          clearInterval(poll);
          alert("驗證失敗：" + s.error);
          setValLoading(false);
        }
      }, 2000);
    } catch (e) {
      alert("驗證失敗：" + e.message);
      setValLoading(false);
    }
  };

  return {
    // state
    status, result, error, elapsed, progress, currentEp,
    prediction, predLoading,
    valResult, valLoading,
    // actions
    startTraining,
    startValidation,
    fetchPrediction,
  };
}