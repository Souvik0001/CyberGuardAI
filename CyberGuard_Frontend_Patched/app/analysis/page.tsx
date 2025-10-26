"use client";

import { useState } from "react";
import { AppLayout } from "@/components/app-layout";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

import { ScreenshotUpload } from "@/components/screenshot-upload";
import { AuthenticityReport } from "@/components/authenticity-report";
import { ManipulationIndicators } from "@/components/manipulation-indicators";

import {
  CheckCircle2,
  AlertCircle,
  ShieldAlert,
  Shield,
} from "lucide-react";

import { analyzeScreenshots } from "@/lib/api-client";

// This is the normalized shape we use in the UI.
// We derive it from backend `results[0]`.
interface AnalysisResult {
  // overall authenticity
  authenticityScore: number; // % bar for "authenticity confidence"
  isAuthentic: boolean; // true if not tampered
  confidence: number; // % version of backend.confidence (0..1 -> 0..100)

  // tamper / manipulation
  tampered: boolean;
  tamperScore: number; // 0..100 scale-ish UI bar
  tamperReason: string;
  tamperProbability: number; // 0..1

  // harassment / coercion
  harassment: boolean;
  harassmentPhrases: string[];
  harassmentScore: number;
  harassmentProbability: number; // 0..1

  // scam / social engineering
  scam: boolean;
  scamReason: string;
  scamPhrases: string[];
  scamScore: number;
  scamProbability: number; // 0..1

  // visual forensic manipulations (for ManipulationIndicators component)
  manipulations: Array<{
    id: string;
    type: string;
    location: string;
    severity: "low" | "medium" | "high";
    description: string;
  }>;

  // metadata for AuthenticityReport
  metadata: {
    resolution: string;
    format: string;
    fileSize: string;
    captureTime: string;
    device: string;
  };

  // descriptive text for AuthenticityReport
  details: {
    pixelAnalysis: string;
    compressionArtifacts: string;
    fontConsistency: string;
    lightingAnalysis: string;
    perspectiveAnalysis: string;
  };
}

export default function AnalysisPage() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  // raw backend for debug panel
  const [rawBackend, setRawBackend] = useState<any | null>(null);

  // local preview img
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  // error display
  const [error, setError] = useState<string | null>(null);

  function formatBytes(bytes: number) {
    const mb = bytes / (1024 * 1024);
    return mb.toFixed(2) + " MB";
  }

  // ====== Upload handler -> calls backend -> maps to AnalysisResult ======
  const handleUpload = async (file: File) => {
    // show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);

    // reset state
    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    setRawBackend(null);

    // call backend /analyze_screenshot
    const apiRes = await analyzeScreenshots([file]);

    if (!apiRes.success || !apiRes.data) {
      setIsAnalyzing(false);
      setError(apiRes.error || "Failed to analyze screenshot.");
      return;
    }

    const first = apiRes.data.results?.[0];
    if (!first) {
      setIsAnalyzing(false);
      setError("No analysis returned from server.");
      return;
    }

    // first is expected to look like:
    // {
    //   verdict,
    //   confidence,               // float 0..1
    //   ocr_text,
    //   tampered, tamper_score, tamper_reason, tamper_probability,
    //   harassment, harassment_phrases, harassment_score, harassment_probability,
    //   scam, scam_phrases, scam_score, scam_probability, scam_reason,
    //   ...
    // }

    const confidencePct = Math.round(Number(first.confidence) * 100);

    // If backend says tampered, we push a manipulation indicator.
    // Otherwise we leave it empty.
    const manipulationsList =
      first.tampered === true
        ? [
            {
              id: "tampering",
              type: "Possible Editing / Splicing",
              location: "Inconsistent regions detected",
              severity: "high" as const,
              description:
                "Localized compression anomalies or manual redaction suggest portions of this screenshot may have been altered.",
            },
          ]
        : [];

    // Build normalized result for UI
    const mapped: AnalysisResult = {
      authenticityScore: confidencePct,
      isAuthentic: !first.tampered,
      confidence: confidencePct,

      tampered: !!first.tampered,
      tamperScore: Number(first.tamper_score ?? 0),
      tamperReason: first.tamper_reason || "",
      tamperProbability: Number(first.tamper_probability ?? 0), // 0..1

      harassment: !!first.harassment,
      harassmentPhrases: first.harassment_phrases || [],
      harassmentScore: Number(first.harassment_score ?? 0),
      harassmentProbability: Number(first.harassment_probability ?? 0),

      scam: !!first.scam,
      scamReason:
        first.scam_reason ||
        "Language consistent with recruitment / payout / crypto-style scam offer.",
      scamPhrases: first.scam_phrases || [],
      scamScore: Number(first.scam_score ?? 0),
      scamProbability: Number(first.scam_probability ?? 0),

      manipulations: manipulationsList,

      metadata: {
        resolution: "unknown", // could be derived if you measure the <img> dims
        format: file.type || "unknown",
        fileSize: formatBytes(file.size),
        captureTime: new Date().toISOString(),
        device: "Unknown",
      },

      details: {
        pixelAnalysis:
          "Visual/text consistency and chat bubble layout were examined.",
        compressionArtifacts:
          "Compression differences were analyzed to detect pasted or edited regions.",
        fontConsistency:
          "OCR text vs typical chat phrasing was compared for abnormalities.",
        lightingAnalysis:
          "Relative brightness and bubble/background contrast were observed.",
        perspectiveAnalysis:
          "Bubble alignment and structure were checked for irregularities.",
      },
    };

    setResult(mapped);
    setRawBackend(first);
    setIsAnalyzing(false);
  };

  // ====== Summary Card Logic ======
  // Priority:
  // 1. tampered
  // 2. harassment
  // 3. scam
  // 4. clean / no obvious edits
  function getVerdictVisualState(r: AnalysisResult | null) {
    if (!r) {
      return {
        tone: "idle" as const,
        icon: <AlertCircle className="h-6 w-6 text-muted-foreground" />,
        title: "No analysis yet",
        description:
          "Upload a screenshot to begin analysis. We'll check for edits, scams, and coercive language.",
        confidenceLabel: "",
        extraDetails: [] as React.ReactNode[],
        headingColorClasses: "text-foreground",
        borderToneClass: "border-border",
      };
    }

    // base defaults
    let icon = (
      <CheckCircle2 className="h-6 w-6 text-green-500" />
    );
    let title = "No obvious edits found";
    let headingColorClasses = "text-green-500";
    let description =
      "No strong manipulation indicators were detected. Content may still be harmful.";
    let borderToneClass = "border-green-500/30";

    const extraDetails: React.ReactNode[] = [];

    // CASE 1: tampered
    if (r.tampered) {
      icon = <AlertCircle className="h-6 w-6 text-destructive" />;
      title = "Possible Forgery / Edited Screenshot";
      headingColorClasses = "text-destructive";
      borderToneClass = "border-destructive/30";
      description =
        "This screenshot may have been altered or spliced. Treat with caution.";

      extraDetails.push(
        <div key="tamperDetail" className="text-xs text-muted-foreground">
          <div className="font-semibold text-foreground mb-1">
            Why flagged:
          </div>
          <div>{r.tamperReason}</div>
          <div className="mt-2">
            Tamper Score: {Math.round(r.tamperScore)} / 100
          </div>
          <div>
            ML tamper probability:{" "}
            {Math.round((r.tamperProbability ?? 0) * 100)}%
          </div>
        </div>
      );
    }

    // CASE 2: harassment (and not tampered)
    else if (r.harassment) {
      icon = <ShieldAlert className="h-6 w-6 text-yellow-400" />;
      title = "Abusive / Coercive Language Detected";
      headingColorClasses = "text-yellow-400";
      borderToneClass = "border-yellow-400/30";
      description =
        "The screenshot looks visually consistent, but the language appears threatening / coercive. Consider preserving evidence and escalating.";

      extraDetails.push(
        <div key="harassDetail" className="text-xs text-muted-foreground">
          <div className="font-semibold text-foreground mb-1">
            Matched phrases:
          </div>
          <div className="break-words">
            {r.harassmentPhrases.length > 0
              ? r.harassmentPhrases.join(", ")
              : "(flagged phrases not listed)"}
          </div>
          <div className="mt-2">
            Severity score: {r.harassmentScore} | Probability:{" "}
            {Math.round((r.harassmentProbability ?? 0) * 100)}%
          </div>
        </div>
      );

      // also mention authenticity context for harassment case
      extraDetails.push(
        <div
          key="harassTamperContext"
          className="text-[10px] text-muted-foreground mt-3 leading-relaxed"
        >
          <div className="font-semibold text-foreground">
            Screenshot authenticity context
          </div>
          <div>{r.tamperReason}</div>
          <div className="mt-1">
            Tamper Score: {Math.round(r.tamperScore)} / 100
          </div>
          <div>
            ML tamper probability:{" "}
            {Math.round((r.tamperProbability ?? 0) * 100)}%
          </div>
          <div className="italic">
            High compression differences are common in chat screenshots (UI
            elements, stickers, emojis).
          </div>
        </div>
      );
    }

    // CASE 3: scam / social engineering (and not tampered, not harassment)
    else if (r.scam) {
      icon = <Shield className="h-6 w-6 text-yellow-400" />;
      title = "Possible Social Engineering / Payment Scam";
      headingColorClasses = "text-yellow-400";
      borderToneClass = "border-yellow-400/30";
      description =
        r.scamReason ||
        "Language consistent with recruitment / payout / crypto-style scam offer. Be careful with money, personal data, or account actions requested here.";

      extraDetails.push(
        <div key="scamDetail" className="text-xs text-muted-foreground">
          <div className="font-semibold text-foreground mb-1">
            Why flagged:
          </div>
          <div className="break-words">{r.scamReason}</div>
          {r.scamPhrases?.length > 0 && (
            <div className="mt-2 text-[11px] leading-relaxed">
              Phrases: {r.scamPhrases.join(", ")}
            </div>
          )}
          <div className="mt-2">
            Scam score: {r.scamScore} | Probability:{" "}
            {Math.round((r.scamProbability ?? 0) * 100)}%
          </div>
        </div>
      );

      // also surface authenticity context for scam cases,
      // because sometimes scam messages are real screenshots
      extraDetails.push(
        <div
          key="scamTamperContext"
          className="text-[10px] text-muted-foreground mt-3 leading-relaxed"
        >
          <div className="font-semibold text-foreground">
            Screenshot authenticity context
          </div>
          <div>{r.tamperReason}</div>
          <div className="mt-1">
            Tamper Score: {Math.round(r.tamperScore)} / 100
          </div>
          <div>
            ML tamper probability:{" "}
            {Math.round((r.tamperProbability ?? 0) * 100)}%
          </div>
          <div className="italic">
            High compression differences can happen normally in chat
            screenshots (UI elements, stickers, emojis). No obvious manual
            redaction detected.
          </div>
        </div>
      );
    }

    // CASE 4: "clean"/not tampered, not harassment, not scam
    else {
      icon = <CheckCircle2 className="h-6 w-6 text-green-500" />;
      // we adapt copy if backend verdict was "Suspicious"
      const suspicious =
        rawBackend?.verdict === "Suspicious" ? true : false;

      title = suspicious
        ? "Likely Authentic (Chat Screenshot Artifacts)"
        : "No obvious edits found";
      headingColorClasses = "text-green-500";
      borderToneClass = "border-green-500/30";

      description = suspicious
        ? "High compression differences found. These can occur in normal chat screenshots (UI elements, stickers, emojis). No obvious manual redaction detected."
        : "No strong manipulation indicators were detected. Content may still be harmful.";

      extraDetails.push(
        <div key="cleanContext" className="text-xs text-muted-foreground">
          <div className="font-semibold text-foreground mb-1">
            Screenshot authenticity context
          </div>
          <div>{r.tamperReason}</div>
          <div className="mt-2">
            Tamper Score (higher = more suspicious):{" "}
            {Math.round(r.tamperScore)} / 100
          </div>
          <div>
            ML tamper probability:{" "}
            {Math.round((r.tamperProbability ?? 0) * 100)}%
          </div>
        </div>
      );
    }

    // label for confidence in header row
    const confidenceLabel = `Confidence ${result?.confidence ?? 0}%`;

    return {
      tone: "ok" as const,
      icon,
      title,
      headingColorClasses,
      borderToneClass,
      description,
      confidenceLabel,
      extraDetails,
    };
  }

  const verdictState = getVerdictVisualState(result);

  // ============================
  // RENDER
  // ============================
  return (
    <AppLayout>
      <div className="space-y-6 p-6">
        {/* Page header */}
        <div>
          <h1 className="text-3xl font-bold text-foreground">
            Screenshot Analysis
          </h1>
          <p className="text-muted-foreground mt-2">
            Check for tampering, fake screenshots, scams, and abusive or
            coercive language
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* LEFT COLUMN: upload + status */}
          <div className="lg:col-span-1">
            <ScreenshotUpload
              onUpload={handleUpload}
              isLoading={isAnalyzing}
            />

            {error && (
              <div className="mt-4 text-sm text-red-500 border border-red-500/30 bg-red-500/10 rounded-md p-2">
                {error}
              </div>
            )}
          </div>

          {/* RIGHT COLUMNS (preview, verdict, rest) */}
          <div className="lg:col-span-2 space-y-6">
            {uploadedImage ? (
              <>
                {/* Screenshot Preview */}
                <Card className="border-border">
                  <CardHeader>
                    <CardTitle>Screenshot Preview</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="relative w-full bg-muted rounded-lg overflow-hidden max-h-96">
                      <img
                        src={uploadedImage || "/placeholder.svg"}
                        alt="Uploaded screenshot"
                        className="w-full h-auto object-contain"
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* After analysis: main verdict + bars, etc.
                   If result is null we're still waiting */}
                {result ? (
                  <>
                    {/* MAIN VERDICT / RISK SUMMARY */}
                    <Card
                      className={`border-border ${verdictState.borderToneClass}`}
                    >
                      <CardHeader>
                        <CardTitle className="flex flex-col gap-2 md:flex-row md:items-center">
                          <span
                            className={`flex items-center gap-2 ${verdictState.headingColorClasses}`}
                          >
                            {verdictState.icon}
                            {verdictState.title}
                          </span>

                          <span className="text-sm font-normal text-muted-foreground md:ml-2">
                            {verdictState.confidenceLabel}
                          </span>
                        </CardTitle>
                      </CardHeader>

                      <CardContent className="space-y-4">
                        <p className="text-sm text-muted-foreground">
                          {verdictState.description}
                        </p>

                        {/* Confidence + Tamper Score bars */}
                        <div className="grid gap-4 md:grid-cols-2">
                          {/* Authenticity Confidence */}
                          <div className="space-y-2">
                            <p className="text-sm text-muted-foreground">
                              Authenticity Confidence
                            </p>
                            <div className="flex items-center gap-2">
                              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                                <div
                                  className={`h-full transition-all ${
                                    result.authenticityScore >= 70
                                      ? "bg-green-500"
                                      : result.authenticityScore >= 40
                                      ? "bg-yellow-500"
                                      : "bg-destructive"
                                  }`}
                                  style={{
                                    width: `${result.authenticityScore}%`,
                                  }}
                                />
                              </div>
                              <span className="font-semibold text-foreground">
                                {result.authenticityScore}%
                              </span>
                            </div>
                          </div>

                          {/* Tamper Score */}
                          <div className="space-y-2">
                            <p className="text-sm text-muted-foreground">
                              Tamper Score (higher = more suspicious)
                            </p>
                            <div className="flex items-center gap-2">
                              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                                <div
                                  className={`h-full transition-all ${
                                    result.tampered
                                      ? "bg-destructive"
                                      : "bg-primary"
                                  }`}
                                  style={{
                                    width: `${
                                      Math.min(
                                        100,
                                        Math.max(
                                          0,
                                          Math.round(result.tamperScore)
                                        )
                                      )
                                    }%`,
                                  }}
                                />
                              </div>
                              <span className="font-semibold text-foreground">
                                {Math.round(result.tamperScore)}
                              </span>
                            </div>

                            <p className="text-xs text-muted-foreground leading-relaxed">
                              {result.tamperReason}
                            </p>
                            <p className="text-[10px] text-muted-foreground">
                              ML tamper probability:{" "}
                              {Math.round(
                                (result.tamperProbability ?? 0) * 100
                              )}
                              %
                            </p>
                          </div>
                        </div>

                        {/* Extra detail about harassment / scam / authenticity context */}
                        {verdictState.extraDetails.length > 0 && (
                          <div className="space-y-4 text-sm">
                            {verdictState.extraDetails.map((block, idx) => (
                              <div key={idx}>{block}</div>
                            ))}
                          </div>
                        )}
                      </CardContent>
                    </Card>

                    {/* Harassment alert box (kept for clarity, even though we surface it above) */}
                    {result.harassment && (
                      <Card className="border-yellow-400/30 border-border">
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2 text-yellow-400">
                            <ShieldAlert className="h-5 w-5 text-yellow-400" />
                            Abusive / Coercive Language Detected
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-2 text-sm text-muted-foreground">
                          <p>
                            This screenshot contains language that may indicate
                            harassment, coercion, threats, or sexual pressure.
                            Consider preserving this evidence and reporting.
                          </p>
                          <p className="text-xs">
                            Matched phrases:{" "}
                            {result.harassmentPhrases.length > 0
                              ? result.harassmentPhrases.join(", ")
                              : "(flagged phrases not listed)"}
                          </p>
                          <p className="text-xs">
                            Severity score: {result.harassmentScore} | Est.
                            probability:{" "}
                            {Math.round(
                              (result.harassmentProbability ?? 0) * 100
                            )}
                            %
                          </p>
                        </CardContent>
                      </Card>
                    )}

                    {/* Manipulation indicators (visual forgery hints) */}
                    <ManipulationIndicators
                      manipulations={result.manipulations}
                    />

                    {/* Metadata + analysis narrative */}
                    <AuthenticityReport
                      metadata={result.metadata}
                      details={result.details}
                    />

                    {/* Debug / raw backend panel */}
                    {rawBackend && (
                      <Card className="border-border">
                        <CardHeader>
                          <CardTitle>Backend Raw Result</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4 text-sm text-muted-foreground leading-relaxed">
                          <div>
                            <span className="font-semibold text-foreground">
                              Verdict from API:
                            </span>{" "}
                            {rawBackend.verdict}
                          </div>

                          <div>
                            <span className="font-semibold text-foreground">
                              Model Confidence (0–1):
                            </span>{" "}
                            {String(rawBackend.confidence)}
                          </div>

                          <div>
                            <span className="font-semibold text-foreground">
                              Tampered?{" "}
                            </span>
                            {rawBackend.tampered ? "YES" : "no"} | Tamper Score:{" "}
                            {String(rawBackend.tamper_score)} | Tamper Prob:{" "}
                            {Math.round(
                              (rawBackend.tamper_probability ?? 0) * 100
                            )}
                            %
                            <div className="text-[10px] text-muted-foreground">
                              {rawBackend.tamper_reason}
                            </div>
                          </div>

                          <div>
                            <span className="font-semibold text-foreground">
                              Harassment?{" "}
                            </span>
                            {rawBackend.harassment ? "YES" : "no"} | Harass Prob:{" "}
                            {Math.round(
                              (rawBackend.harassment_probability ?? 0) * 100
                            )}
                            %
                            <div className="text-[10px] text-muted-foreground">
                              Phrases:{" "}
                              {rawBackend.harassment_phrases &&
                              rawBackend.harassment_phrases.length > 0
                                ? rawBackend.harassment_phrases.join(", ")
                                : "(none listed)"}
                            </div>
                          </div>

                          <div>
                            <span className="font-semibold text-foreground">
                              Scam / Social Engineering?{" "}
                            </span>
                            {rawBackend.scam ? "YES" : "no"} | Scam Prob:{" "}
                            {Math.round(
                              (rawBackend.scam_probability ?? 0) * 100
                            )}
                            %
                            <div className="text-[10px] text-muted-foreground">
                              {rawBackend.scam_reason}
                            </div>
                            {rawBackend.scam_phrases &&
                              rawBackend.scam_phrases.length > 0 && (
                                <div className="text-[10px] text-muted-foreground mt-1">
                                  Phrases:{" "}
                                  {rawBackend.scam_phrases.join(", ")}
                                </div>
                              )}
                          </div>

                          <div className="space-y-1">
                            <span className="font-semibold text-foreground">
                              OCR Text Extracted:
                            </span>
                            <pre className="text-xs whitespace-pre-wrap break-words text-muted-foreground bg-black/30 border border-border rounded-md p-2 max-h-[400px] overflow-auto">
                              {rawBackend.ocr_text || "(no OCR text returned)"}
                            </pre>
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </>
                ) : (
                  // Waiting for result from backend after upload
                  <Card className="border-border border-dashed">
                    <CardContent className="flex flex-col items-center justify-center py-12">
                      {isAnalyzing ? (
                        <>
                          <div className="h-12 w-12 animate-spin rounded-full border-4 border-muted border-t-primary mb-4" />
                          <p className="text-muted-foreground text-center">
                            Analyzing screenshot...
                          </p>
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
                          <p className="text-muted-foreground text-center">
                            Upload complete. Waiting for analysis result...
                          </p>
                        </>
                      )}
                    </CardContent>
                  </Card>
                )}
              </>
            ) : (
              // No screenshot uploaded yet
              <Card className="border-border border-dashed">
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-muted-foreground text-center">
                    Upload a screenshot to begin analysis
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
