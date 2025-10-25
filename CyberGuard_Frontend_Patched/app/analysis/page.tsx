"use client"

import { useState } from "react"
import { AppLayout } from "@/components/app-layout"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScreenshotUpload } from "@/components/screenshot-upload"
import { AuthenticityReport } from "@/components/authenticity-report"
import { ManipulationIndicators } from "@/components/manipulation-indicators"
import { CheckCircle2, AlertCircle } from "lucide-react"

interface AnalysisResult {
  authenticityScore: number
  isAuthentic: boolean
  confidence: number
  manipulations: Array<{
    id: string
    type: string
    location: string
    severity: "low" | "medium" | "high"
    description: string
  }>
  metadata: {
    resolution: string
    format: string
    fileSize: string
    captureTime: string
    device: string
  }
  details: {
    pixelAnalysis: string
    compressionArtifacts: string
    fontConsistency: string
    lightingAnalysis: string
    perspectiveAnalysis: string
  }
}

export default function AnalysisPage() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)

  const handleUpload = async (file: File) => {
    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string)
    }
    reader.readAsDataURL(file)

    setIsAnalyzing(true)
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 3000))

    const mockResult: AnalysisResult = {
      authenticityScore: 78,
      isAuthentic: true,
      confidence: 92,
      manipulations: [
        {
          id: "1",
          type: "Minor Compression Artifacts",
          location: "Bottom right corner",
          severity: "low",
          description: "Standard JPEG compression artifacts detected, consistent with normal screenshot compression",
        },
        {
          id: "2",
          type: "Text Consistency",
          location: "Message area",
          severity: "low",
          description: "Font rendering is consistent with platform standards",
        },
      ],
      metadata: {
        resolution: "1080 x 1920",
        format: "PNG",
        fileSize: "2.4 MB",
        captureTime: "2024-10-25 14:32:15",
        device: "iPhone 14 Pro",
      },
      details: {
        pixelAnalysis: "Pixel distribution analysis shows natural patterns consistent with authentic screenshots",
        compressionArtifacts: "Minimal compression artifacts detected, typical for PNG format",
        fontConsistency: "All text uses consistent font rendering with platform-standard anti-aliasing",
        lightingAnalysis: "Lighting and shadows are consistent throughout the image",
        perspectiveAnalysis: "No perspective distortion or warping detected",
      },
    }

    setResult(mockResult)
    setIsAnalyzing(false)
  }

  return (
    <AppLayout>
      <div className="space-y-6 p-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-foreground">Screenshot Analysis</h1>
          <p className="text-muted-foreground mt-2">
            Verify screenshot authenticity and detect manipulations or forgeries
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Upload Section */}
          <div className="lg:col-span-1">
            <ScreenshotUpload onUpload={handleUpload} isLoading={isAnalyzing} />
          </div>

          {/* Results */}
          <div className="lg:col-span-2 space-y-6">
            {uploadedImage && (
              <>
                {/* Image Preview */}
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

                {result ? (
                  <>
                    {/* Authenticity Summary */}
                    <Card
                      className={`border-border ${result.isAuthentic ? "border-green-500/30" : "border-destructive/30"}`}
                    >
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          {result.isAuthentic ? (
                            <>
                              <CheckCircle2 className="h-6 w-6 text-green-500" />
                              Likely Authentic
                            </>
                          ) : (
                            <>
                              <AlertCircle className="h-6 w-6 text-destructive" />
                              Potential Forgery Detected
                            </>
                          )}
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid gap-4 md:grid-cols-2">
                          <div className="space-y-2">
                            <p className="text-sm text-muted-foreground">Authenticity Score</p>
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
                                  style={{ width: `${result.authenticityScore}%` }}
                                />
                              </div>
                              <span className="font-semibold text-foreground">{result.authenticityScore}%</span>
                            </div>
                          </div>
                          <div className="space-y-2">
                            <p className="text-sm text-muted-foreground">Confidence Level</p>
                            <div className="flex items-center gap-2">
                              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-primary transition-all"
                                  style={{ width: `${result.confidence}%` }}
                                />
                              </div>
                              <span className="font-semibold text-foreground">{result.confidence}%</span>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Manipulation Indicators */}
                    <ManipulationIndicators manipulations={result.manipulations} />

                    {/* Detailed Analysis */}
                    <AuthenticityReport metadata={result.metadata} details={result.details} />
                  </>
                ) : (
                  <Card className="border-border border-dashed">
                    <CardContent className="flex flex-col items-center justify-center py-12">
                      <div className="h-12 w-12 animate-spin rounded-full border-4 border-muted border-t-primary mb-4" />
                      <p className="text-muted-foreground text-center">Analyzing screenshot...</p>
                    </CardContent>
                  </Card>
                )}
              </>
            )}

            {!uploadedImage && (
              <Card className="border-border border-dashed">
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
                  <p className="text-muted-foreground text-center">Upload a screenshot to begin analysis</p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </AppLayout>
  )
}
