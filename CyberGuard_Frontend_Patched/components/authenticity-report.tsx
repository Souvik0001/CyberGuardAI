import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { CheckCircle2 } from "lucide-react"

interface AuthenticityReportProps {
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

export function AuthenticityReport({ metadata, details }: AuthenticityReportProps) {
  return (
    <>
      {/* Metadata */}
      <Card className="border-border">
        <CardHeader>
          <CardTitle>Image Metadata</CardTitle>
          <CardDescription>Technical information about the screenshot</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            {Object.entries(metadata).map(([key, value]) => (
              <div key={key} className="space-y-1">
                <p className="text-xs font-semibold text-muted-foreground uppercase">
                  {key.replace(/([A-Z])/g, " $1").trim()}
                </p>
                <p className="text-sm text-foreground">{value}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Detailed Analysis */}
      <Card className="border-border">
        <CardHeader>
          <CardTitle>Detailed Analysis</CardTitle>
          <CardDescription>In-depth examination results</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {Object.entries(details).map(([key, value]) => (
            <div key={key} className="space-y-2">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-green-500 flex-shrink-0" />
                <h4 className="font-semibold text-foreground">{key.replace(/([A-Z])/g, " $1").trim()}</h4>
              </div>
              <p className="text-sm text-foreground/80 ml-6">{value}</p>
            </div>
          ))}
        </CardContent>
      </Card>
    </>
  )
}
