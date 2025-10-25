"use client"
import React, { useState } from "react";
import { analyzeScreenshots } from "../../lib/api-client";

export default function ScreenshotAnalysis() {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onChange = (e) => setFiles(Array.from(e.target.files || []));

  const submit = async (e) => {
    e.preventDefault();
    if (!files.length) return setError("Please select files");
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await analyzeScreenshots(files);
      if (!res.success) throw new Error(res.error || "Analyze failed");
      setResult(res.data);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{padding:20}}>
      <h1>Screenshot Analysis</h1>
      <form onSubmit={submit}>
        <input type="file" multiple accept="image/*" onChange={onChange} />
        <div><button type="submit" disabled={loading}>{loading ? "Analyzing..." : "Run OCR & Verify (calls backend)"}</button></div>
      </form>
      <pre>{JSON.stringify(result, null, 2)}</pre>
      {error && <div style={{color:"red"}}>{error}</div>}
    </div>
)
}
