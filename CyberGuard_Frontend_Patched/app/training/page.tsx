"use client";

import React, { useState } from "react";
import { AppLayout } from "@/components/app-layout";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/ui/card";

import {
  saveHarassmentSample,
  saveTamperSample,
  retrainModels,
  uploadHarassmentDataset,
  uploadTamperDataset,
} from "@/lib/api-client";

import { Switch } from "@/components/ui/switch"; // if you don't have this, replace with checkbox
import { AlertCircle } from "lucide-react";

export default function TrainingPage() {
  // --- single harassment sample form ---
  const [chatText, setChatText] = useState(
    `Example:\n"Send me pics or I'll leak everything."\n"Stop ignoring me. I'm outside your house."`
  );
  const [isAbusive, setIsAbusive] = useState(true);
  const [harassmentStatus, setHarassmentStatus] = useState<null | string>(null);

  // --- single tamper sample form ---
  const [elaScore, setElaScore] = useState("62.5");
  const [widthPx, setWidthPx] = useState("1080");
  const [heightPx, setHeightPx] = useState("1920");
  const [isTampered, setIsTampered] = useState(false);
  const [tamperStatus, setTamperStatus] = useState<null | string>(null);

  // --- retrain status ---
  const [retrainStatus, setRetrainStatus] = useState<null | string>(null);

  // --- bulk upload forms ---
  const [harassmentFile, setHarassmentFile] = useState<File | null>(null);
  const [tamperFile, setTamperFile] = useState<File | null>(null);
  const [bulkStatus, setBulkStatus] = useState<null | string>(null);

  // Save single harassment sample
  async function handleSaveHarassment() {
    setHarassmentStatus("Saving...");
    const res = await saveHarassmentSample(chatText, isAbusive);
    if (!res.success) {
      setHarassmentStatus("Error: " + res.error);
    } else {
      setHarassmentStatus(
        `Saved. (${res.data?.added ?? 1} sample appended)`
      );
    }
  }

  // Save single tamper sample
  async function handleSaveTamper() {
    setTamperStatus("Saving...");
    const res = await saveTamperSample(
      elaScore,
      widthPx,
      heightPx,
      isTampered
    );
    if (!res.success) {
      setTamperStatus("Error: " + res.error);
    } else {
      setTamperStatus(
        `Saved. (${res.data?.added ?? 1} sample appended)`
      );
    }
  }

  // Retrain both models
  async function handleRetrain() {
    setRetrainStatus("Training...");
    const res = await retrainModels();
    if (!res.success) {
      setRetrainStatus("Error: " + res.error);
    } else {
      const d = res.data;
      setRetrainStatus(
        `Done. ${d.harassment_model_trained ? "Harass OK" : "Harass FAIL"}, ${d.tamper_model_trained ? "Tamper OK" : "Tamper FAIL"}.`
      );
    }
  }

  // Upload BIG harassment dataset (.jsonl or .csv)
  async function handleBulkHarassmentUpload() {
    if (!harassmentFile) {
        setBulkStatus("Select a harassment dataset first.");
        return;
    }
    setBulkStatus("Uploading harassment dataset...");
    const res = await uploadHarassmentDataset(harassmentFile);
    if (!res.success) {
      setBulkStatus("Error: " + res.error);
    } else {
      setBulkStatus(
        `Harassment dataset added: ${res.data?.added} rows. Now click Retrain Models.`
      );
    }
  }

  // Upload BIG tamper dataset (.jsonl or .csv)
  async function handleBulkTamperUpload() {
    if (!tamperFile) {
        setBulkStatus("Select a tamper dataset first.");
        return;
    }
    setBulkStatus("Uploading tamper dataset...");
    const res = await uploadTamperDataset(tamperFile);
    if (!res.success) {
      setBulkStatus("Error: " + res.error);
    } else {
      setBulkStatus(
        `Tamper dataset added: ${res.data?.added} rows. Now click Retrain Models.`
      );
    }
  }

  return (
    <AppLayout>
      <div className="p-6 space-y-8 text-foreground">
        <h1 className="text-3xl font-bold">
          Training &amp; Model Update
        </h1>
        <p className="text-muted-foreground max-w-3xl text-sm">
          Add labeled examples of harassment / stalking messages or tampered
          screenshots. Then click <b>Retrain Models</b> so CyberGuard learns
          from your data. You can also bulk upload thousands of samples.
        </p>

        {/* Row 1: add harassment sample + add tamper sample */}
        <div className="grid gap-6 lg:grid-cols-2">
          {/* --- Harassment / Coercion / Stalking text samples --- */}
          <Card className="border-border bg-card/50">
            <CardHeader>
              <CardTitle className="text-lg font-semibold">
                Add Conversation Example (Harassment / Stalking)
              </CardTitle>
              <p className="text-xs text-muted-foreground">
                Paste one or more abusive / coercive / threatening lines.
                This improves harassment + coercion detection.
              </p>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <label className="block text-xs font-medium text-muted-foreground">
                Chat text
              </label>
              <textarea
                className="w-full h-40 rounded-md bg-background border border-border p-3 text-xs text-foreground resize-none"
                value={chatText}
                onChange={(e) => setChatText(e.target.value)}
              />

              <div className="flex items-start gap-3">
                <div className="flex items-center gap-2">
                  {/* If you don't have Switch, replace with checkbox */}
                  <Switch
                    checked={isAbusive}
                    onCheckedChange={(val: boolean) => setIsAbusive(val)}
                    className="data-[state=checked]:bg-red-600"
                  />
                  <span className="text-sm font-medium text-foreground">
                    Mark as abusive / threatening
                  </span>
                </div>
              </div>
              <p className="text-[10px] text-muted-foreground leading-snug">
                Turn OFF if this is actually a normal / safe message.
                (We also need safe messages as negative training data.)
              </p>

              <button
                onClick={handleSaveHarassment}
                className="w-full bg-red-600 hover:bg-red-700 text-white text-sm font-medium py-2 rounded-md transition-colors"
              >
                Save Harassment Sample
              </button>

              {harassmentStatus && (
                <div className="text-[11px] text-muted-foreground whitespace-pre-wrap">
                  {harassmentStatus}
                </div>
              )}
            </CardContent>
          </Card>

          {/* --- Tamper forensics samples --- */}
          <Card className="border-border bg-card/50">
            <CardHeader>
              <CardTitle className="text-lg font-semibold">
                Add Image Forensics Example (Tampering)
              </CardTitle>
              <p className="text-xs text-muted-foreground">
                Add metadata from a screenshot you already inspected
                (ELA score, resolution, and whether it was edited).
                This feeds the tamper/forgery detector.
              </p>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="space-y-2">
                <label className="block text-xs font-medium text-muted-foreground">
                  ELA score
                </label>
                <input
                  className="w-full rounded-md bg-background border border-border p-2 text-xs text-foreground"
                  placeholder="e.g. 62.5"
                  value={elaScore}
                  onChange={(e) => setElaScore(e.target.value)}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="block text-xs font-medium text-muted-foreground">
                    Width (px)
                  </label>
                  <input
                    className="w-full rounded-md bg-background border border-border p-2 text-xs text-foreground"
                    value={widthPx}
                    onChange={(e) => setWidthPx(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <label className="block text-xs font-medium text-muted-foreground">
                    Height (px)
                  </label>
                  <input
                    className="w-full rounded-md bg-background border border-border p-2 text-xs text-foreground"
                    value={heightPx}
                    onChange={(e) => setHeightPx(e.target.value)}
                  />
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="flex items-center gap-2">
                  <Switch
                    checked={isTampered}
                    onCheckedChange={(val: boolean) => setIsTampered(val)}
                    className="data-[state=checked]:bg-red-600"
                  />
                  <span className="text-sm font-medium text-foreground">
                    Mark as tampered / edited
                  </span>
                </div>
              </div>
              <p className="text-[10px] text-muted-foreground leading-snug">
                Flip ON if this screenshot was altered/spliced.
                OFF if it's clean.
              </p>

              <button
                onClick={handleSaveTamper}
                className="w-full bg-red-600 hover:bg-red-700 text-white text-sm font-medium py-2 rounded-md transition-colors"
              >
                Save Tamper Sample
              </button>

              {tamperStatus && (
                <div className="text-[11px] text-muted-foreground whitespace-pre-wrap">
                  {tamperStatus}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Row 2: Retrain Models */}
        <Card className="border-border bg-card/50">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">
              Retrain Models From Saved Samples
            </CardTitle>
            <p className="text-xs text-muted-foreground">
              This will:
            </p>
            <ul className="text-xs text-muted-foreground list-disc ml-5 space-y-1">
              <li>
                Re-train the harassment / coercion / stalking classifier on{" "}
                <code className="text-[10px] bg-black/30 px-1 py-0.5 rounded">
                  data/harassment_samples.jsonl
                </code>
              </li>
              <li>
                Re-train the tamper/forgery classifier on{" "}
                <code className="text-[10px] bg-black/30 px-1 py-0.5 rounded">
                  data/tamper_samples.jsonl
                </code>
              </li>
            </ul>
            <p className="text-[10px] text-muted-foreground">
              After that, the backend immediately starts using the new
              models (no restart needed).
            </p>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            <button
              onClick={handleRetrain}
              className="w-full bg-red-600 hover:bg-red-700 text-white text-sm font-medium py-2 rounded-md transition-colors"
            >
              Retrain Models
            </button>

            {retrainStatus && (
              <div className="text-[11px] text-muted-foreground whitespace-pre-wrap">
                {retrainStatus}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Row 3: Bulk Upload */}
        <Card className="border-border bg-card/50">
          <CardHeader>
            <CardTitle className="text-lg font-semibold flex items-center gap-2">
              <span>Bulk Upload Datasets (Advanced)</span>
              <AlertCircle className="h-4 w-4 text-yellow-400" />
            </CardTitle>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Upload thousands of labeled samples at once.
              After upload, click <b>Retrain Models</b> above.
            </p>
          </CardHeader>

          <CardContent className="grid gap-6 md:grid-cols-2 text-sm">
            {/* Bulk harassment uploader */}
            <div className="space-y-3">
              <div className="text-sm font-medium text-foreground">
                Harassment / Stalking Dataset
              </div>
              <p className="text-[11px] text-muted-foreground leading-snug">
                Accepts CSV or JSONL. Each row/line must have:
                <code className="text-[10px] bg-black/30 px-1 py-0.5 rounded ml-1">
                  text
                </code>
                ,
                <code className="text-[10px] bg-black/30 px-1 py-0.5 rounded ml-1">
                  label
                </code>
                . Label 1 = abusive / stalking / coercive.
                Label 0 = normal / safe.
              </p>

              <input
                type="file"
                className="block w-full text-xs text-muted-foreground file:mr-3 file:rounded-md file:border-0 file:bg-red-600 file:px-3 file:py-1.5 file:text-white hover:file:bg-red-700 file:text-xs file:font-medium"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  setHarassmentFile(f || null);
                }}
              />

              <button
                onClick={handleBulkHarassmentUpload}
                className="w-full bg-red-600 hover:bg-red-700 text-white text-sm font-medium py-2 rounded-md transition-colors"
              >
                Upload Harassment Dataset
              </button>
            </div>

            {/* Bulk tamper uploader */}
            <div className="space-y-3">
              <div className="text-sm font-medium text-foreground">
                Tamper / Forgery Dataset
              </div>
              <p className="text-[11px] text-muted-foreground leading-snug">
                Accepts CSV or JSONL with columns/fields:
                <code className="text-[10px] bg-black/30 px-1 py-0.5 rounded ml-1">
                  ela
                </code>
                ,
                <code className="text-[10px] bg-black/30 px-1 py-0.5 rounded ml-1">
                  res_w
                </code>
                ,
                <code className="text-[10px] bg-black/30 px-1 py-0.5 rounded ml-1">
                  res_h
                </code>
                ,
                <code className="text-[10px] bg-black/30 px-1 py-0.5 rounded ml-1">
                  label
                </code>
                . Label 1 = edited / spliced. Label 0 = clean.
              </p>

              <input
                type="file"
                className="block w-full text-xs text-muted-foreground file:mr-3 file:rounded-md file:border-0 file:bg-red-600 file:px-3 file:py-1.5 file:text-white hover:file:bg-red-700 file:text-xs file:font-medium"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  setTamperFile(f || null);
                }}
              />

              <button
                onClick={handleBulkTamperUpload}
                className="w-full bg-red-600 hover:bg-red-700 text-white text-sm font-medium py-2 rounded-md transition-colors"
              >
                Upload Tamper Dataset
              </button>
            </div>

            {/* Status / instructions */}
            <div className="md:col-span-2 text-[11px] text-muted-foreground whitespace-pre-wrap leading-relaxed">
              {bulkStatus
                ? bulkStatus
                : "Tip: After uploads, press Retrain Models so the new data is learned immediately."}
            </div>
          </CardContent>
        </Card>
      </div>
    </AppLayout>
  );
}
