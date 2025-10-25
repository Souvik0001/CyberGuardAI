"use client";

import React, { useState } from "react";
import { analyzeChat } from "../../lib/api-client"; // keep your existing API client

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

  // evidence upload state
  const [files, setFiles] = useState<File[]>([]);
  const [evidenceResults, setEvidenceResults] = useState<EvidenceResult[] | null>(null);
  const [evidenceStatus, setEvidenceStatus] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    setFiles(Array.from(e.target.files));
    setEvidenceResults(null);
    setEvidenceStatus(null);
  };

  const uploadEvidence = async (): Promise<EvidenceResult[] | null> => {
    if (!files || files.length === 0) return null;
    setEvidenceStatus("Uploading evidence...");
    try {
      const form = new FormData();
      // backend expects key 'files' (list)
      files.forEach((f) => form.append("files", f));

      const res = await fetch(`${API_BASE}/analyze_screenshot`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const txt = await res.text();
        setEvidenceStatus(`Evidence upload failed: ${res.status} ${res.statusText} - ${txt}`);
        return null;
      }

      const json = await res.json();
      // /analyze_screenshot returns { results: [...] } in your backend
      const arr = Array.isArray(json.results) ? json.results : [];
      setEvidenceResults(arr);
      setEvidenceStatus("Evidence uploaded and analyzed");
      return arr;
    } catch (err: any) {
      console.error("uploadEvidence error:", err);
      setEvidenceStatus("Evidence upload error: " + String(err));
      return null;
    }
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setEvidenceResults(null);
    setEvidenceStatus(null);

    try {
      // Run text analysis (your existing analyzeChat helper)
      const chatPromise = analyzeChat({ text });

      // Run evidence upload in parallel (if any files selected)
      const evidencePromise = files.length > 0 ? uploadEvidence() : Promise.resolve(null);

      const [chatRes, evidenceRes] = await Promise.all([chatPromise, evidencePromise]);

      // Handle chat response shape from your analyzeChat client
      if (!chatRes || !chatRes.success) {
        throw new Error(chatRes?.error || "Chat analysis failed");
      }

      setResult(chatRes.data ?? chatRes); // adapt depending on your client

      if (evidenceRes) {
        setEvidenceResults(evidenceRes as EvidenceResult[]);
      }
    } catch (err: any) {
      console.error("Analyze error:", err);
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const clearFiles = () => {
    setFiles([]);
    setEvidenceResults(null);
    setEvidenceStatus(null);
    // also clear the file input DOM value if needed
    const fileInput = document.querySelector<HTMLInputElement>("input[type=file][data-evidence-input]");
    if (fileInput) fileInput.value = "";
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
          <label style={{ display: "block", marginBottom: 6 }}>Upload Evidence (Optional) — images or PDFs</label>
          <input
            data-evidence-input
            type="file"
            accept="image/*,application/pdf"
            multiple
            onChange={handleFileChange}
            style={{ display: "block", marginBottom: 8 }}
          />
          <div style={{ marginBottom: 8 }}>
            {files.length === 0 ? (
              <div style={{ color: "#aaa" }}>No files selected</div>
            ) : (
              files.map((f, i) => (
                <div key={i} style={{ fontSize: 13 }}>
                  {f.name} — {Math.round(f.size / 1024)} KB
                </div>
              ))
            )}
          </div>
          {files.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <button type="button" onClick={clearFiles} style={{ marginRight: 8 }}>
                Clear Files
              </button>
            </div>
          )}
        </div>

        <div style={{ marginTop: 12 }}>
          <button type="submit" disabled={loading} style={{ padding: "10px 16px", fontSize: 16 }}>
            {loading ? "Analyzing..." : "Analyze Now"}
          </button>
        </div>
      </form>

      <div style={{ marginTop: 20 }}>
        <h3>Chat Analysis Result</h3>
        <div style={{ background: "#0b0b0b", padding: 12, borderRadius: 6, color: "#ddd" }}>
          <pre style={{ whiteSpace: "pre-wrap", maxHeight: 380, overflow: "auto" }}>{result ? JSON.stringify(result, null, 2) : "No chat result yet"}</pre>
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
