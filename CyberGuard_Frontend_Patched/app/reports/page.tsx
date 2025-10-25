"use client"

import { useState } from "react"
import { AppLayout } from "@/components/app-layout"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { AnalyticsDashboard } from "@/components/analytics-dashboard"
import { ThreatTimeline } from "@/components/threat-timeline"
import { AnalyticsCharts } from "@/components/analytics-charts"
import { ReportGenerator } from "@/components/report-generator"
import { Calendar, Download } from "lucide-react"

export default function ReportsPage() {
  const [dateRange, setDateRange] = useState({
    startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split("T")[0],
    endDate: new Date().toISOString().split("T")[0],
  })

  const [showReportGenerator, setShowReportGenerator] = useState(false)

  return (
    <AppLayout>
      <div className="space-y-6 p-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Reports & Analytics</h1>
            <p className="text-muted-foreground mt-2">Track threats, analyze patterns, and generate reports</p>
          </div>
          <Button onClick={() => setShowReportGenerator(true)} className="gap-2">
            <Download className="h-4 w-4" />
            Generate Report
          </Button>
        </div>

        {/* Date Range Filter */}
        <Card className="border-border">
          <CardContent className="pt-6">
            <div className="flex flex-col sm:flex-row gap-4 items-end">
              <div className="flex-1 space-y-2">
                <label className="text-sm font-medium text-foreground flex items-center gap-2">
                  <Calendar className="h-4 w-4" />
                  Date Range
                </label>
                <div className="flex gap-2">
                  <input
                    type="date"
                    value={dateRange.startDate}
                    onChange={(e) => setDateRange({ ...dateRange, startDate: e.target.value })}
                    className="flex-1 px-3 py-2 rounded-lg border border-border bg-input text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                  <input
                    type="date"
                    value={dateRange.endDate}
                    onChange={(e) => setDateRange({ ...dateRange, endDate: e.target.value })}
                    className="flex-1 px-3 py-2 rounded-lg border border-border bg-input text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                </div>
              </div>
              <Button variant="outline">Apply Filter</Button>
            </div>
          </CardContent>
        </Card>

        {/* Dashboard Overview */}
        <AnalyticsDashboard />

        {/* Charts */}
        <AnalyticsCharts />

        {/* Threat Timeline */}
        <ThreatTimeline />

        {/* Report Generator Modal */}
        {showReportGenerator && <ReportGenerator onClose={() => setShowReportGenerator(false)} />}
      </div>
    </AppLayout>
  )
}
