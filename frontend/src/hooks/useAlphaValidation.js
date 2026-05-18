import { useState } from "react";
import { API } from "../constants/config";

export function useAlphaValidation() {
  const [status, setStatus]           = useState("idle");
  const [report, setReport]           = useState(null);
  const [error, setError]             = useState(null);
  const [reports, setReports]         = useState([]);
  const [reportsLoading, setReportsLoading] = useState(false);
  const [models, setModels]           = useState([]);
  const [modelsLoading, setModelsLoading]   = useState(false);
  const [selectedModel, setSelectedModel]   = useState(null);

  const runValidation = async ({ name, description, featureCode, featureColumn, skipLayer1, modelPath }) => {
    setStatus("running");
    setError(null);
    setReport(null);
    try {
      const res = await fetch(`${API}/alpha_validation/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          description,
          feature_code:   featureCode,
          feature_column: featureColumn || null,
          skip_layer1:    skipLayer1 || false,
          model_path:     modelPath   || null,
        }),
      });
      if (!res.ok) {
        const e = await res.json();
        throw new Error(e.detail || "驗證失敗");
      }
      setReport(await res.json());
      setStatus("done");
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  };

  const fetchReports = async () => {
    setReportsLoading(true);
    try {
      const data = await fetch(`${API}/alpha_validation/reports`).then((r) => r.json());
      setReports(data.reports || []);
    } catch (e) {}
    finally { setReportsLoading(false); }
  };

  const loadReport = async (name) => {
    try {
      const data = await fetch(`${API}/alpha_validation/reports/${name}`).then((r) => r.json());
      setReport(data);
      setStatus("done");
    } catch (e) {
      setError("載入報告失敗：" + e.message);
    }
  };

  const fetchModels = async () => {
    setModelsLoading(true);
    try {
      const data = await fetch(`${API}/models`).then((r) => r.json());
      const list = data.models || [];
      setModels(list);
      if (list.length > 0 && !selectedModel) setSelectedModel(list[0]);
    } catch (e) {}
    finally { setModelsLoading(false); }
  };

  const reset = () => {
    setStatus("idle");
    setReport(null);
    setError(null);
  };

  return {
    status, report, error,
    reports, reportsLoading,
    models, modelsLoading, selectedModel, setSelectedModel,
    runValidation, fetchReports, loadReport, fetchModels, reset,
  };
}