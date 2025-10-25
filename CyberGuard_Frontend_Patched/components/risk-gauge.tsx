"use client"

export interface RiskGaugeProps {
  riskScore: number
  riskLevel: "low" | "medium" | "high" | "critical"
}

export function RiskGauge({ riskScore, riskLevel }: RiskGaugeProps) {
  const getRiskColor = (level: string) => {
    switch (level) {
      case "critical":
        return "#ef4444"
      case "high":
        return "#f97316"
      case "medium":
        return "#eab308"
      case "low":
        return "#22c55e"
      default:
        return "#6b7280"
    }
  }

  const circumference = 2 * Math.PI * 45
  const strokeDashoffset = circumference - (riskScore / 100) * circumference

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative w-40 h-40">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 120 120">
          {/* Background circle */}
          <circle cx="60" cy="60" r="45" fill="none" stroke="currentColor" strokeWidth="8" className="text-muted" />
          {/* Progress circle */}
          <circle
            cx="60"
            cy="60"
            r="45"
            fill="none"
            stroke={getRiskColor(riskLevel)}
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-500"
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold text-foreground">{riskScore}</span>
          <span className="text-xs text-muted-foreground">Risk Score</span>
        </div>
      </div>
    </div>
  )
}
