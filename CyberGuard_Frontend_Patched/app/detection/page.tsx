"use client"

import { useState } from "react"
import { AppLayout } from "@/components/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { DetectionForm } from "@/components/detection-form"
import { ThreatAnalysis } from "@/components/threat-analysis"
import { RiskGauge } from "@/components/risk-gauge"
import { AlertCircle, CheckCircle2, AlertTriangle } from "lucide-react"
import { analyzeChat } from "@/lib/api-client"

interface DetectionResult {
  riskLevel: "low" | "medium" | "high" | "critical"
  riskScore: number
  threats: Array<{
    id: string
    type: string
    severity: "low" | "medium" | "high"
    description: string
    timestamp: string
  }>
  recommendations: string[]
  analysisTime: number
}

export default function DetectionPage() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<DetectionResult | null>(null)

  // const handleAnalyze = async (data: any) => {
  //   setIsAnalyzing(true)
  //   // Simulate API call
  //   await new Promise((resolve) => setTimeout(resolve, 2000))

  //   const mockResult: DetectionResult = {
  //     riskLevel: "medium",
  //     riskScore: 65,
  //     threats: [
  //       {
  //         id: "1",
  //         type: "Unusual Access Pattern",
  //         severity: "high",
  //         description: "Multiple login attempts from different locations within 1 hour",
  //         timestamp: new Date().toISOString(),
  //       },
  //       {
  //         id: "2",
  //         type: "Profile Monitoring",
  //         severity: "medium",
  //         description: "Account viewed 47 times in the last 24 hours",
  //         timestamp: new Date(Date.now() - 3600000).toISOString(),
  //       },
  //       {
  //         id: "3",
  //         type: "Message Frequency",
  //         severity: "low",
  //         description: "Increased message frequency from unknown account",
  //         timestamp: new Date(Date.now() - 7200000).toISOString(),
  //       },
  //     ],
  //     recommendations: [
  //       "Enable two-factor authentication immediately",
  //       "Review recent login activity and revoke suspicious sessions",
  //       "Block or report the suspicious account",
  //       "Consider making your profile private temporarily",
  //       "Document all interactions for potential legal action",
  //     ],
  //     analysisTime: 2.3,
  //   }

  //   setResult(mockResult)
  //   setIsAnalyzing(false)
  // }
  const handleAnalyze = async (data: any) => {
  setIsAnalyzing(true)
  setResult(null)

  try {
    // Build a single text blob from the form fields so backend can analyze it.
    // You can change this to upload CSVs or other data formats as needed.
    const pieces: string[] = []
    if (data.accountName) pieces.push(`Account: ${data.accountName}`)
    if (data.platform) pieces.push(`Platform: ${data.platform}`)
    if (data.dataType) pieces.push(`Data type: ${data.dataType}`)
    if (data.description) pieces.push(`Description: ${data.description}`)
    // include any other fields (evidence filenames etc.)
    const textPayload = pieces.join("\n")

    // Call the centralized frontend API client
    const res = await analyzeChat({ text: textPayload })

    if (!res.success) {
      throw new Error(res.error || "Backend analyze failed")
    }

    const backend = res.data

    // Map backend fields to DetectionResult shape expected by UI.
    // Adjust names if your backend returns slightly different keys.
    const mapped: DetectionResult = {
      // riskScore: try to read overall_risk or compute from fields
      riskScore: Math.round((backend.overall_risk ?? backend.risk ?? 0) * 100),
      // riskLevel thresholds — tune these to your app's semantics
      riskLevel:
        (backend.overall_risk ?? 0) >= 0.85 ? "critical" :
        (backend.overall_risk ?? 0) >= 0.66 ? "high" :
        (backend.overall_risk ?? 0) >= 0.33 ? "medium" : "low",
      threats: (backend.anomalies || backend.threats || []).map((t: any, i: number) => ({
        id: String(t.id ?? i),
        type: t.type ?? t.name ?? (t.message ? "Message anomaly" : "Unknown"),
        severity: t.severity ?? (t.is_anomaly_if ? "high" : "low"),
        description: t.description ?? t.message ?? JSON.stringify(t).slice(0, 200),
        timestamp: t.timestamp ?? t.time ?? "Just now",
      })),
      recommendations: backend.recommendations ?? (backend.suggested_actions || []),
      analysisTime: Math.round((backend.analysis_time ?? backend.analysisTime ?? 0) * 10) / 10,
    }

    setResult(mapped)
  } catch (err: any) {
    console.error("Analysis failed:", err)
    // Optionally show a notification to the user
    setResult(null)
    // If you want to surface errors separately, add state for error message
  } finally {
    setIsAnalyzing(false)
  }
}


  const getRiskColor = (level: string) => {
    switch (level) {
      case "critical":
        return "text-destructive"
      case "high":
        return "text-orange-500"
      case "medium":
        return "text-yellow-500"
      case "low":
        return "text-green-500"
      default:
        return "text-muted-foreground"
    }
  }

  const getRiskBgColor = (level: string) => {
    switch (level) {
      case "critical":
        return "bg-destructive/10"
      case "high":
        return "bg-orange-500/10"
      case "medium":
        return "bg-yellow-500/10"
      case "low":
        return "bg-green-500/10"
      default:
        return "bg-muted"
    }
  }

  return (
    <AppLayout>
      <div className="space-y-6 p-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-foreground">Stalker Detection</h1>
          <p className="text-muted-foreground mt-2">
            Analyze suspicious behavior and detect potential cyberstalking threats
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Detection Form */}
          <div className="lg:col-span-1">
            <DetectionForm onAnalyze={handleAnalyze} isLoading={isAnalyzing} />
          </div>

          {/* Results */}
          <div className="lg:col-span-2 space-y-6">
            {result ? (
              <>
                {/* Risk Overview */}
                <Card className="border-border">
                  <CardHeader>
                    <CardTitle>Risk Assessment</CardTitle>
                    <CardDescription>Current threat level and analysis summary</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-6 md:grid-cols-2">
                      <RiskGauge riskScore={result.riskScore} riskLevel={result.riskLevel} />

                      <div className="space-y-4">
                        <div className={`p-4 rounded-lg ${getRiskBgColor(result.riskLevel)}`}>
                          <div className="flex items-center gap-2 mb-2">
                            {result.riskLevel === "critical" && <AlertCircle className="h-5 w-5 text-destructive" />}
                            {result.riskLevel === "high" && <AlertTriangle className="h-5 w-5 text-orange-500" />}
                            {result.riskLevel === "medium" && <AlertTriangle className="h-5 w-5 text-yellow-500" />}
                            {result.riskLevel === "low" && <CheckCircle2 className="h-5 w-5 text-green-500" />}
                            <span className={`font-semibold capitalize ${getRiskColor(result.riskLevel)}`}>
                              {result.riskLevel} Risk
                            </span>
                          </div>
                          <p className="text-sm text-foreground">
                            {result.riskLevel === "critical" &&
                              "Immediate action required. Significant threat detected."}
                            {result.riskLevel === "high" && "High threat level. Take protective measures immediately."}
                            {result.riskLevel === "medium" &&
                              "Moderate threat detected. Review recommendations and take precautions."}
                            {result.riskLevel === "low" && "Low threat level. Continue monitoring for changes."}
                          </p>
                        </div>

                        <div className="text-sm text-muted-foreground">
                          <p>Analysis completed in {result.analysisTime}s</p>
                          <p>Threats detected: {result.threats.length}</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Threats */}
                <ThreatAnalysis threats={result.threats} />

                {/* Recommendations */}
                <Card className="border-border">
                  <CardHeader>
                    <CardTitle>Recommended Actions</CardTitle>
                    <CardDescription>Steps to protect yourself</CardDescription>
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
                  <p className="text-muted-foreground text-center">Submit data to analyze for stalking threats</p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </AppLayout>
  )
}
