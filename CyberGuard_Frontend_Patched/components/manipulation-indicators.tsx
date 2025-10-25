import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertCircle, AlertTriangle, Info, CheckCircle2 } from "lucide-react"

interface Manipulation {
  id: string
  type: string
  location: string
  severity: "low" | "medium" | "high"
  description: string
}

interface ManipulationIndicatorsProps {
  manipulations: Manipulation[]
}

export function ManipulationIndicators({ manipulations }: ManipulationIndicatorsProps) {
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

  const highSeverityCount = manipulations.filter((m) => m.severity === "high").length
  const mediumSeverityCount = manipulations.filter((m) => m.severity === "medium").length

  return (
    <Card className="border-border">
      <CardHeader>
        <CardTitle>Manipulation Detection</CardTitle>
        <CardDescription>
          {manipulations.length === 0
            ? "No manipulations detected"
            : `${manipulations.length} indicator${manipulations.length !== 1 ? "s" : ""} found`}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {manipulations.length === 0 ? (
          <div className="text-center py-8">
            <CheckCircle2 className="h-12 w-12 text-green-500 mx-auto mb-3" />
            <p className="text-foreground font-medium">No manipulations detected</p>
            <p className="text-sm text-muted-foreground">Screenshot appears authentic</p>
          </div>
        ) : (
          <div className="space-y-3">
            {highSeverityCount > 0 && (
              <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20">
                <p className="text-sm font-semibold text-destructive">
                  {highSeverityCount} high-severity indicator{highSeverityCount !== 1 ? "s" : ""}
                </p>
              </div>
            )}
            {mediumSeverityCount > 0 && (
              <div className="p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                <p className="text-sm font-semibold text-yellow-600">
                  {mediumSeverityCount} medium-severity indicator{mediumSeverityCount !== 1 ? "s" : ""}
                </p>
              </div>
            )}

            <div className="space-y-3 mt-4">
              {manipulations.map((manipulation) => (
                <div key={manipulation.id} className={`p-4 rounded-lg border ${getSeverityBg(manipulation.severity)}`}>
                  <div className="flex gap-3 items-start">
                    {getSeverityIcon(manipulation.severity)}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <h4 className="font-semibold text-foreground">{manipulation.type}</h4>
                        <span className="text-xs text-muted-foreground whitespace-nowrap">{manipulation.location}</span>
                      </div>
                      <p className="text-sm text-foreground/80">{manipulation.description}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
