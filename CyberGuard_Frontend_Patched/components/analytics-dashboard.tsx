import { Card, CardContent } from "@/components/ui/card"
import { TrendingUp, AlertCircle, CheckCircle2, Clock } from "lucide-react"

export function AnalyticsDashboard() {
  const metrics = [
    {
      label: "Total Threats",
      value: "247",
      change: "+12%",
      trend: "up",
      icon: AlertCircle,
      color: "text-destructive",
    },
    {
      label: "Resolved",
      value: "189",
      change: "+8%",
      trend: "up",
      icon: CheckCircle2,
      color: "text-green-500",
    },
    {
      label: "In Progress",
      value: "34",
      change: "-3%",
      trend: "down",
      icon: Clock,
      color: "text-yellow-500",
    },
    {
      label: "Detection Rate",
      value: "94.2%",
      change: "+2.1%",
      trend: "up",
      icon: TrendingUp,
      color: "text-primary",
    },
  ]

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {metrics.map((metric) => {
        const Icon = metric.icon
        return (
          <Card key={metric.label} className="border-border">
            <CardContent className="pt-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">{metric.label}</p>
                  <Icon className={`h-5 w-5 ${metric.color}`} />
                </div>
                <div>
                  <p className="text-3xl font-bold text-foreground">{metric.value}</p>
                  <p
                    className={`text-xs font-medium mt-1 ${metric.trend === "up" ? "text-green-500" : "text-red-500"}`}
                  >
                    {metric.change} from last month
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
