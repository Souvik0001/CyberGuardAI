"use client";

import React, { useState } from "react";
import { analyzeScreenshots } from "@/lib/api-client"; // you can also keep "../../lib/api-client" if you prefer relative

export default function ScreenshotAnalysisPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!files.length) {
      setError("Please select files");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await analyzeScreenshots(files);

      if (!res.success) {
        throw new Error(res.error || "Analyze failed");
      }

      setResult(res.data);
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Screenshot Analysis</h1>

      <form onSubmit={submit}>
        <input
          type="file"
          multiple
          accept="image/*"
          onChange={onChange}
        />

        <div style={{ marginTop: "1rem" }}>
          <button type="submit" disabled={loading}>
            {loading
              ? "Analyzing..."
              : "Run OCR & Verify (calls backend)"}
          </button>
        </div>
      </form>

      {result && (
        <pre style={{ marginTop: "1rem" }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      )}

      {error && (
        <div style={{ color: "red", marginTop: "1rem" }}>
          {error}
        </div>
      )}
    </div>
  );
}
