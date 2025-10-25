import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { CheckCircle2 } from "lucide-react"

interface ModelPerformanceProps {
  metrics: {
    precision: number
    recall: number
    f1Score: number
    auc: number
  }
  accuracy: number
}

export function ModelPerformance({ metrics, accuracy }: ModelPerformanceProps) {
  const performanceMetrics = [
    { label: "Precision", value: metrics.precision, description: "True positives / All positives" },
    { label: "Recall", value: metrics.recall, description: "True positives / Actual positives" },
    { label: "F1 Score", value: metrics.f1Score, description: "Harmonic mean of precision and recall" },
    { label: "AUC-ROC", value: metrics.auc, description: "Area under the ROC curve" },
  ]

  return (
    <Card className="border-border">
      <CardHeader>
        <CardTitle>Model Performance</CardTitle>
        <CardDescription>Current model evaluation metrics</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {performanceMetrics.map((metric) => (
            <div key={metric.label} className="space-y-3 p-4 rounded-lg bg-muted border border-border">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-foreground">{metric.label}</p>
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              </div>
              <div className="space-y-2">
                <p className="text-2xl font-bold text-foreground">{(metric.value * 100).toFixed(1)}%</p>
                <div className="w-full h-2 bg-background rounded-full overflow-hidden">
                  <div className="h-full bg-primary transition-all" style={{ width: `${metric.value * 100}%` }} />
                </div>
              </div>
              <p className="text-xs text-muted-foreground">{metric.description}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
