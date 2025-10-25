"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { X, Download } from "lucide-react"

interface ReportGeneratorProps {
  onClose: () => void
}

export function ReportGenerator({ onClose }: ReportGeneratorProps) {
  const [reportConfig, setReportConfig] = useState({
    reportType: "comprehensive",
    dateRange: "30days",
    includeCharts: true,
    includeThreatTimeline: true,
    includeRecommendations: true,
    format: "pdf",
  })

  const [isGenerating, setIsGenerating] = useState(false)

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsGenerating(true)

    // Simulate report generation
    await new Promise((resolve) => setTimeout(resolve, 2000))

    // Trigger download
    const element = document.createElement("a")
    element.setAttribute("href", "data:text/plain;charset=utf-8,Report generated successfully")
    element.setAttribute(
      "download",
      `cyberguard-report-${new Date().toISOString().split("T")[0]}.${reportConfig.format}`,
    )
    element.style.display = "none"
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)

    setIsGenerating(false)
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <Card className="border-border w-full max-w-md">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
          <div>
            <CardTitle>Generate Report</CardTitle>
            <CardDescription>Create a custom security report</CardDescription>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-muted rounded-lg transition-colors">
            <X className="h-5 w-5" />
          </button>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleGenerate} className="space-y-4">
            {/* Report Type */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Report Type</label>
              <select
                value={reportConfig.reportType}
                onChange={(e) => setReportConfig({ ...reportConfig, reportType: e.target.value })}
                className="w-full px-3 py-2 rounded-lg border border-border bg-input text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="comprehensive">Comprehensive</option>
                <option value="summary">Summary</option>
                <option value="threats">Threats Only</option>
                <option value="analytics">Analytics Only</option>
              </select>
            </div>

            {/* Date Range */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Date Range</label>
              <select
                value={reportConfig.dateRange}
                onChange={(e) => setReportConfig({ ...reportConfig, dateRange: e.target.value })}
                className="w-full px-3 py-2 rounded-lg border border-border bg-input text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="7days">Last 7 Days</option>
                <option value="30days">Last 30 Days</option>
                <option value="90days">Last 90 Days</option>
                <option value="custom">Custom Range</option>
              </select>
            </div>

            {/* Include Options */}
            <div className="space-y-3">
              <label className="text-sm font-medium text-foreground">Include in Report</label>
              <div className="space-y-2">
                {[
                  { key: "includeCharts", label: "Analytics Charts" },
                  { key: "includeThreatTimeline", label: "Threat Timeline" },
                  { key: "includeRecommendations", label: "Recommendations" },
                ].map((option) => (
                  <label key={option.key} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={reportConfig[option.key as keyof typeof reportConfig] as boolean}
                      onChange={(e) =>
                        setReportConfig({
                          ...reportConfig,
                          [option.key]: e.target.checked,
                        })
                      }
                      className="w-4 h-4 rounded border-border"
                    />
                    <span className="text-sm text-foreground">{option.label}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Format */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Format</label>
              <select
                value={reportConfig.format}
                onChange={(e) => setReportConfig({ ...reportConfig, format: e.target.value })}
                className="w-full px-3 py-2 rounded-lg border border-border bg-input text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="pdf">PDF</option>
                <option value="csv">CSV</option>
                <option value="json">JSON</option>
              </select>
            </div>

            {/* Buttons */}
            <div className="flex gap-2 pt-4">
              <Button type="button" variant="outline" onClick={onClose} className="flex-1 bg-transparent">
                Cancel
              </Button>
              <Button type="submit" disabled={isGenerating} className="flex-1 gap-2">
                {isGenerating ? (
                  <>
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4" />
                    Generate
                  </>
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
