import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"

export function AnalyticsCharts() {
  // Threats over time
  const threatTrendData = [
    { date: "Oct 1", threats: 12, resolved: 10 },
    { date: "Oct 5", threats: 18, resolved: 15 },
    { date: "Oct 10", threats: 24, resolved: 20 },
    { date: "Oct 15", threats: 19, resolved: 18 },
    { date: "Oct 20", threats: 28, resolved: 25 },
    { date: "Oct 25", threats: 31, resolved: 28 },
  ]

  // Platform distribution
  const platformData = [
    { name: "Instagram", value: 45, color: "#f97316" },
    { name: "Twitter/X", value: 28, color: "#8b5cf6" },
    { name: "Facebook", value: 15, color: "#3b82f6" },
    { name: "Discord", value: 12, color: "#ec4899" },
  ]

  // Threat severity
  const severityData = [
    { severity: "Critical", count: 8 },
    { severity: "High", count: 34 },
    { severity: "Medium", count: 89 },
    { severity: "Low", count: 116 },
  ]

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Threat Trend */}
      <Card className="border-border">
        <CardHeader>
          <CardTitle>Threat Trend</CardTitle>
          <CardDescription>Threats detected and resolved over time</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={threatTrendData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis dataKey="date" stroke="var(--color-muted-foreground)" />
              <YAxis stroke="var(--color-muted-foreground)" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--color-card)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "0.5rem",
                }}
                labelStyle={{ color: "var(--color-foreground)" }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="threats"
                stroke="var(--color-destructive)"
                strokeWidth={2}
                dot={false}
                name="Detected"
              />
              <Line
                type="monotone"
                dataKey="resolved"
                stroke="var(--color-primary)"
                strokeWidth={2}
                dot={false}
                name="Resolved"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Platform Distribution */}
      <Card className="border-border">
        <CardHeader>
          <CardTitle>Platform Distribution</CardTitle>
          <CardDescription>Threats by social platform</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={platformData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {platformData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--color-card)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "0.5rem",
                }}
                labelStyle={{ color: "var(--color-foreground)" }}
              />
            </PieChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Threat Severity Distribution */}
      <Card className="border-border lg:col-span-2">
        <CardHeader>
          <CardTitle>Threat Severity Distribution</CardTitle>
          <CardDescription>Breakdown of threats by severity level</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={severityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis dataKey="severity" stroke="var(--color-muted-foreground)" />
              <YAxis stroke="var(--color-muted-foreground)" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--color-card)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "0.5rem",
                }}
                labelStyle={{ color: "var(--color-foreground)" }}
              />
              <Bar dataKey="count" fill="var(--color-primary)" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}
