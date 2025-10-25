import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertCircle, AlertTriangle, Info } from "lucide-react"

interface Threat {
  id: string
  type: string
  severity: "low" | "medium" | "high"
  description: string
  timestamp: string
}

interface ThreatAnalysisProps {
  threats: Threat[]
}

export function ThreatAnalysis({ threats }: ThreatAnalysisProps) {
  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case "high":
        return <AlertCircle className="h-5 w-5 text-destructive" />
      case "medium":
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />
      case "low":
        return <Info className="h-5 w-5 text-blue-500" />
      default:
        return null
    }
  }

  const getSeverityBg = (severity: string) => {
    switch (severity) {
      case "high":
        return "bg-destructive/10 border-destructive/20"
      case "medium":
        return "bg-yellow-500/10 border-yellow-500/20"
      case "low":
        return "bg-blue-500/10 border-blue-500/20"
      default:
        return "bg-muted"
    }
  }

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const hours = Math.floor(diff / 3600000)
    const minutes = Math.floor((diff % 3600000) / 60000)

    if (hours > 0) return `${hours}h ago`
    if (minutes > 0) return `${minutes}m ago`
    return "Just now"
  }

  return (
    <Card className="border-border">
      <CardHeader>
        <CardTitle>Detected Threats</CardTitle>
        <CardDescription>Suspicious activities and patterns identified</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {threats.map((threat) => (
            <div key={threat.id} className={`p-4 rounded-lg border ${getSeverityBg(threat.severity)}`}>
              <div className="flex gap-3 items-start">
                {getSeverityIcon(threat.severity)}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <h4 className="font-semibold text-foreground">{threat.type}</h4>
                    <span className="text-xs text-muted-foreground whitespace-nowrap">
                      {formatTime(threat.timestamp)}
                    </span>
                  </div>
                  <p className="text-sm text-foreground/80">{threat.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
