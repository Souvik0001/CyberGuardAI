"use client";

import React, { useState } from "react";
import { analyzeChat } from "../../lib/api-client";
import dynamic from "next/dynamic";
import EvidenceUploader from "../../components/EvidenceUploader";

// Load the client-only uploader component (ensure the path matches where you created it)
// const EvidenceUploader = dynamic(() => import("../../components/EvidenceUploader"), { ssr: false });

type EvidenceResult = {
  filename: string;
  verdict?: string;
  confidence?: number;
  ocr_text?: string;
  error?: string;
};

export default function StalkerDetectionPage() {
  const API_BASE = (process.env.NEXT_PUBLIC_API_URL as string) || "http://localhost:8000";

  // text analysis state
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // evidence state (uploader will call onComplete to populate this)
  const [evidenceResults, setEvidenceResults] = useState<EvidenceResult[] | null>(null);
  const [evidenceStatus, setEvidenceStatus] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await analyzeChat({ text });
      if (!res || !res.success) {
        throw new Error(res?.error || "Analyze failed");
      }
      setResult(res.data ?? res);
    } catch (err: any) {
      console.error("Analyze error:", err);
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20, maxWidth: 980, margin: "0 auto" }}>
      <h1>Stalker Detection</h1>

      <form onSubmit={submit}>
        <label style={{ display: "block", marginBottom: 8, fontWeight: 600 }}>Account / Description / Messages</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste messages, descriptions, or notes here..."
          rows={10}
          style={{ width: "100%", padding: 10, borderRadius: 6, background: "#111", color: "#fff", border: "1px solid #333" }}
        />

        <div style={{ marginTop: 12 }}>
          <button type="submit" disabled={loading} style={{ padding: "10px 16px", fontSize: 16 }}>
            {loading ? "Analyzing..." : "Analyze Now"}
          </button>
        </div>
      </form>

      <div style={{ marginTop: 20 }}>
        <h3>Upload Evidence (Optional)</h3>
        {/* EvidenceUploader handles the file input and upload. It will call onComplete with the backend JSON. */}
        <EvidenceUploader
          apiBase={API_BASE}
          onComplete={(res) => {
            // The backend returns { results: [...] } from analyze_screenshot
            if (!res) {
              setEvidenceResults(null);
              setEvidenceStatus("No response");
              return;
            }
            if (res.results && Array.isArray(res.results)) {
              setEvidenceResults(res.results);
              setEvidenceStatus("Evidence uploaded and analyzed");
            } else {
              // fallback: if uploader returns other shape, store it anyway
              setEvidenceResults(Array.isArray(res) ? res : [res]);
              setEvidenceStatus("Evidence uploaded (unexpected shape)");
            }
          }}
        />
      </div>

      <div style={{ marginTop: 20 }}>
        <h3>Chat Analysis Result</h3>
        <div style={{ background: "#0b0b0b", padding: 12, borderRadius: 6, color: "#ddd" }}>
          <pre style={{ whiteSpace: "pre-wrap", maxHeight: 380, overflow: "auto" }}>
            {result ? JSON.stringify(result, null, 2) : "No chat result yet"}
          </pre>
        </div>
      </div>

      <div style={{ marginTop: 20 }}>
        <h3>Evidence Analysis Result</h3>
        <div style={{ background: "#0b0b0b", padding: 12, borderRadius: 6, color: "#ddd" }}>
          <div style={{ marginBottom: 8 }}>
            <strong>Status:</strong> {evidenceStatus ?? "No evidence uploaded"}
          </div>
          {evidenceResults ? (
            <div>
              {evidenceResults.map((r: any, idx: number) => (
                <div key={idx} style={{ marginBottom: 10 }}>
                  <div style={{ fontWeight: 600 }}>{r.filename}</div>
                  {r.error ? (
                    <div style={{ color: "salmon" }}>{r.error}</div>
                  ) : (
                    <div>
                      <div>Verdict: {r.verdict}</div>
                      <div>Confidence: {r.confidence}</div>
                      <div style={{ marginTop: 6, fontSize: 13, whiteSpace: "pre-wrap" }}>
                        OCR snippet: {r.ocr_text ? r.ocr_text.slice(0, 300) : "(none)"}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div>No evidence results</div>
          )}
        </div>
      </div>

      {error && (
        <div style={{ marginTop: 18, color: "salmon" }}>
          <strong>Error:</strong> {error}
        </div>
      )}
    </div>
  );
}
