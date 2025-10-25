import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertCircle, AlertTriangle, Info, CheckCircle2 } from "lucide-react"

export function ThreatTimeline() {
  const threats = [
    {
      id: "1",
      type: "High Severity Threat",
      platform: "Instagram",
      account: "@suspicious_user_123",
      severity: "high",
      timestamp: "2024-10-25 14:32",
      status: "resolved",
      description: "Multiple login attempts from different locations",
    },
    {
      id: "2",
      type: "Medium Severity Threat",
      platform: "Twitter/X",
      account: "@unknown_account",
      severity: "medium",
      timestamp: "2024-10-25 12:15",
      status: "in-progress",
      description: "Unusual message frequency detected",
    },
    {
      id: "3",
      type: "Low Severity Threat",
      platform: "Discord",
      account: "user#1234",
      severity: "low",
      timestamp: "2024-10-25 10:45",
      status: "resolved",
      description: "Profile viewed multiple times",
    },
    {
      id: "4",
      type: "High Severity Threat",
      platform: "Facebook",
      account: "Suspicious Profile",
      severity: "high",
      timestamp: "2024-10-24 18:20",
      status: "resolved",
      description: "Attempted to add as friend with fake profile",
    },
    {
      id: "5",
      type: "Medium Severity Threat",
      platform: "Instagram",
      account: "@another_suspicious",
      severity: "medium",
      timestamp: "2024-10-24 15:10",
      status: "resolved",
      description: "Persistent messaging attempts",
    },
  ]

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

  return (
    <Card className="border-border">
      <CardHeader>
        <CardTitle>Threat Timeline</CardTitle>
        <CardDescription>Recent threats and their status</CardDescription>
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
                    <div className="flex items-center gap-2">
                      {threat.status === "resolved" && <CheckCircle2 className="h-4 w-4 text-green-500" />}
                      <span className="text-xs text-muted-foreground whitespace-nowrap">{threat.timestamp}</span>
                    </div>
                  </div>
                  <p className="text-sm text-foreground/80 mb-2">{threat.description}</p>
                  <div className="flex flex-wrap gap-2 text-xs">
                    <span className="px-2 py-1 rounded bg-background text-foreground">{threat.platform}</span>
                    <span className="px-2 py-1 rounded bg-background text-foreground">{threat.account}</span>
                    <span
                      className={`px-2 py-1 rounded capitalize ${
                        threat.status === "resolved"
                          ? "bg-green-500/20 text-green-700"
                          : "bg-yellow-500/20 text-yellow-700"
                      }`}
                    >
                      {threat.status}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
