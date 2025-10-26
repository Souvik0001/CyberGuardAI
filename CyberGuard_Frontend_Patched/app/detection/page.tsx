"use client"

import { useState } from "react"
import { AppLayout } from "@/components/app-layout"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { DetectionForm } from "@/components/detection-form"
import { ThreatAnalysis } from "@/components/threat-analysis"
import { RiskGauge } from "@/components/risk-gauge"
import { AlertCircle, CheckCircle2, AlertTriangle } from "lucide-react"
import { analyzeChat } from "@/lib/api-client"

interface DetectionResult {
  riskLevel: "low" | "medium" | "high" | "critical"
  riskScore: number // 0-100
  threats: Array<{
    id: string
    type: string
    severity: "low" | "medium" | "high"
    description: string
    timestamp: string
  }>
  recommendations: string[]
  analysisTime: number // seconds
}

export default function DetectionPage() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<DetectionResult | null>(null)

  // Kick off backend analysis when the user presses "Analyze Now"
  const handleAnalyze = async (data: any) => {
    setIsAnalyzing(true)
    setResult(null)

    try {
      // Build a single text blob for the backend.
      // You can extend this with uploaded evidence later.
      const pieces: string[] = []
      if (data.accountName) pieces.push(`Account: ${data.accountName}`)
      if (data.platform) pieces.push(`Platform: ${data.platform}`)
      if (data.dataType) pieces.push(`Data type: ${data.dataType}`)
      if (data.description) pieces.push(`Description: ${data.description}`)
      const textPayload = pieces.join("\n")

      // Call backend /analyze_chat through the shared API client.
      const res = await analyzeChat({ text: textPayload })
      if (!res.success) {
        throw new Error(res.error || "Backend analyze failed")
      }

      const backend = res.data
      // backend should look like:
      // {
      //   risk: 0.82,
      //   risk_level: "high",
      //   anomalies: [
      //     { type, severity, message, timestamp, ... },
      //   ],
      //   recommendations: [ "Block them", ... ],
      //   analysis_time_ms: 142
      // }

      // Map backend → UI shape DetectionResult
      const mapped: DetectionResult = {
        riskScore: Math.round((backend.risk ?? 0) * 100),
        riskLevel: (backend.risk_level ??
          ((backend.risk ?? 0) >= 0.85
            ? "critical"
            : (backend.risk ?? 0) >= 0.66
            ? "high"
            : (backend.risk ?? 0) >= 0.33
            ? "medium"
            : "low")) as DetectionResult["riskLevel"],
        threats: (backend.anomalies || []).map(
          (t: any, i: number): DetectionResult["threats"][number] => ({
            id: String(t.id ?? i),
            type: t.type ?? t.name ?? "Suspicious Activity",
            severity:
              (t.severity as "low" | "medium" | "high") ??
              (t.level as "low" | "medium" | "high") ??
              "medium",
            description:
              t.description ??
              t.message ??
              JSON.stringify(t).slice(0, 200),
            timestamp: t.timestamp ?? t.time ?? "Just now",
          })
        ),
        recommendations: backend.recommendations ?? [],
        analysisTime: Math.round(
          ((backend.analysis_time_ms ?? 0) / 1000) * 10
        ) / 10,
      }

      setResult(mapped)
    } catch (err: any) {
      console.error("Analysis failed:", err)
      setResult(null)
    } finally {
      setIsAnalyzing(false)
    }
  }

  // Color helpers for the “Risk Assessment” card
  const getRiskColor = (level: string) => {
    switch (level) {
      case "critical":
        return "text-destructive"
      case "high":
        return "text-orange-500"
      case "medium":
        return "text-yellow-500"
      case "low":
      default:
        return "text-green-500"
    }
  }

  const getRiskBgColor = (level: string) => {
    switch (level) {
      case "critical":
        return "bg-destructive/10 border border-destructive/30"
      case "high":
        return "bg-orange-500/10 border border-orange-500/30"
      case "medium":
        return "bg-yellow-500/10 border border-yellow-500/30"
      case "low":
      default:
        return "bg-green-500/10 border border-green-500/30"
    }
  }

  return (
    <AppLayout>
      <div className="space-y-6 p-6">
        {/* Page header */}
        <div>
          <h1 className="text-3xl font-bold text-foreground">
            Stalker Detection
          </h1>
          <p className="text-muted-foreground mt-2">
            Analyze suspicious behavior and detect potential cyberstalking
            threats
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* LEFT COLUMN: the form */}
          <div className="lg:col-span-1">
            <DetectionForm
              onAnalyze={handleAnalyze}
              isLoading={isAnalyzing}
            />

            {/* You can surface an explicit status if you want */}
            <div className="mt-4 text-xs text-muted-foreground">
              Status:{" "}
              {isAnalyzing
                ? "analyzing…"
                : result
                ? "analysis complete"
                : "idle"}
            </div>
          </div>

          {/* RIGHT COLUMN: analysis result */}
          <div className="lg:col-span-2 space-y-6">
            {result ? (
              <>
                {/* Risk Overview */}
                <Card className="border-border">
                  <CardHeader>
                    <CardTitle>Risk Assessment</CardTitle>
                    <CardDescription>
                      Current threat level and analysis summary
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-6 md:grid-cols-2">
                      <RiskGauge
                        riskScore={result.riskScore}
                        riskLevel={result.riskLevel}
                      />

                      <div className="space-y-4">
                        <div
                          className={`p-4 rounded-lg ${getRiskBgColor(
                            result.riskLevel
                          )}`}
                        >
                          <div className="flex items-center gap-2 mb-2">
                            {result.riskLevel === "critical" && (
                              <AlertCircle className="h-5 w-5 text-destructive" />
                            )}
                            {result.riskLevel === "high" && (
                              <AlertTriangle className="h-5 w-5 text-orange-500" />
                            )}
                            {result.riskLevel === "medium" && (
                              <AlertTriangle className="h-5 w-5 text-yellow-500" />
                            )}
                            {result.riskLevel === "low" && (
                              <CheckCircle2 className="h-5 w-5 text-green-500" />
                            )}
                            <span
                              className={`font-semibold capitalize ${getRiskColor(
                                result.riskLevel
                              )}`}
                            >
                              {result.riskLevel} Risk
                            </span>
                          </div>
                          <p className="text-sm text-foreground">
                            {result.riskLevel === "critical" &&
                              "Immediate action required. Significant threat detected."}
                            {result.riskLevel === "high" &&
                              "High threat level. Take protective measures immediately."}
                            {result.riskLevel === "medium" &&
                              "Moderate threat detected. Review recommendations and take precautions."}
                            {result.riskLevel === "low" &&
                              "Low threat level. Continue monitoring for changes."}
                          </p>

                          <p className="text-[11px] text-muted-foreground mt-3">
                            Analysis time: {result.analysisTime}s
                          </p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Detected Threats */}
                <ThreatAnalysis threats={result.threats} />

                {/* Recommended Actions */}
                <Card className="border-border">
                  <CardHeader>
                    <CardTitle>Recommended Actions</CardTitle>
                    <CardDescription>
                      Steps to protect yourself
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-3">
                      {result.recommendations.map((rec, i) => (
                        <li key={i} className="flex gap-3 items-start">
                          <CheckCircle2 className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                          <span className="text-foreground">{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card className="border-border border-dashed">
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-muted-foreground text-center">
                    Submit data to analyze for stalking threats
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </AppLayout>
  )
}
