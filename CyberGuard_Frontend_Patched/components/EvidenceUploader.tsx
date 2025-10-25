"use client";

import React, { useState, useRef, useEffect } from "react";

type Props = {
  apiBase?: string;
  onComplete?: (result: any) => void;
};

export default function EvidenceUploader({ apiBase, onComplete }: Props) {
  const API_BASE = apiBase || (process.env.NEXT_PUBLIC_API_URL as string) || "http://localhost:8000";
  const [files, setFiles] = useState<File[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [responseJson, setResponseJson] = useState<any>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    // debug: print when component mounts
    console.log("EvidenceUploader mounted", { API_BASE });
  }, [API_BASE]);

  function onFilesChange(e: React.ChangeEvent<HTMLInputElement>) {
    console.log("onFilesChange fired", e);
    if (!e.target.files) return;
    setFiles(Array.from(e.target.files));
    setStatus(null);
    setResponseJson(null);
  }

  async function uploadFiles() {
    if (files.length === 0) {
      setStatus("No files selected");
      return;
    }
    setStatus("Uploading...");
    try {
      const form = new FormData();
      for (const f of files) form.append("files", f);

      console.log("Uploading files:", files.map(f => f.name));
      const res = await fetch(`${API_BASE}/analyze_screenshot`, {
        method: "POST",
        body: form,
      });

      const text = await res.text();
      try {
        const json = JSON.parse(text);
        setResponseJson(json);
        if (onComplete) onComplete(json);
      } catch {
        // sometime backend returns text
        setResponseJson(text);
        if (onComplete) onComplete(text);
      }

      if (!res.ok) {
        setStatus(`Upload returned ${res.status} ${res.statusText}`);
      } else {
        setStatus("Upload complete");
      }
    } catch (err: any) {
      console.error("Upload error", err);
      setStatus("Upload error: " + String(err));
    }
  }

  // Make the input fill the dashed box so clicking anywhere opens the file picker.
  const inputStyle: React.CSSProperties = {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    opacity: 0,
    cursor: "pointer",
    zIndex: 20,
  };

  const boxStyle: React.CSSProperties = {
    position: "relative",
    cursor: "pointer",
    border: "2px dashed rgba(255,255,255,0.08)",
    padding: 22,
    borderRadius: 12,
    textAlign: "center",
    color: "rgba(255,255,255,0.8)",
    background: "transparent",
    minHeight: 90,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  };

  return (
    <div>
      <div ref={wrapperRef} style={boxStyle} aria-label="Click to upload evidence">
        <div style={{ pointerEvents: "none" }}>
          <div style={{ fontSize: 14, marginBottom: 6 }}>Click to upload or drag and drop</div>
          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.5)" }}>PNG, JPG, PDF up to 10MB</div>
        </div>

        {/* Invisible input overlay that catches clicks even if parent styles block pointer-events */}
        <input
          ref={inputRef}
          type="file"
          accept="image/*,application/pdf"
          multiple
          onChange={onFilesChange}
          style={inputStyle}
          data-testid="evidence-input"
        />
      </div>

      <div style={{ marginTop: 10 }}>
        {files.length > 0 ? (
          <>
            <div style={{ marginBottom: 8 }}>
              {files.map((f, i) => (
                <div key={i} style={{ fontSize: 13 }}>
                  {f.name} — {Math.round(f.size / 1024)} KB
                </div>
              ))}
            </div>
            <div>
              <button type="button" onClick={uploadFiles} style={{ padding: "8px 12px" }}>
                Upload Evidence
              </button>
              <button
                type="button"
                onClick={() => {
                  setFiles([]);
                  setResponseJson(null);
                  setStatus(null);
                  if (inputRef.current) inputRef.current.value = "";
                }}
                style={{ marginLeft: 8 }}
              >
                Clear
              </button>
            </div>
          </>
        ) : (
          <div style={{ marginTop: 6, color: "#aaa", fontSize: 13 }}>No files selected</div>
        )}
      </div>

      <div style={{ marginTop: 10, fontSize: 13 }}>
        <strong>Status:</strong> {status ?? "idle"}
      </div>

      {responseJson && (
        <pre style={{ whiteSpace: "pre-wrap", marginTop: 8, maxHeight: 240, overflow: "auto" }}>
          {JSON.stringify(responseJson, null, 2)}
        </pre>
      )}
    </div>
  );
}
