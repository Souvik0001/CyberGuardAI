"use client"
import React, { useState } from "react";
import { analyzeChat } from "../../lib/api-client";

export default function StalkerDetectionPage() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await analyzeChat({ text });
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
      <h1>Stalker Detection</h1>
      <form onSubmit={submit}>
        <textarea value={text} onChange={(e)=>setText(e.target.value)} rows={8} cols={80} />
        <div>
          <button type="submit" disabled={loading}>{loading ? "Analyzing..." : "Analyze Behavior (calls backend)"}</button>
        </div>
      </form>
      <pre>{JSON.stringify(result, null, 2)}</pre>
      {error && <div style={{color:"red"}}>{error}</div>}
    </div>
)
}
