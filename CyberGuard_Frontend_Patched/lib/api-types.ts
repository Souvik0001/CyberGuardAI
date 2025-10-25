// Type definitions for CyberGuard API responses

export interface User {
  id: string
  email: string
  name: string
  createdAt: string
  updatedAt: string
}

export interface AuthResponse {
  token: string
  user: User
}

export interface Threat {
  id: string
  type: string
  severity: "low" | "medium" | "high" | "critical"
  platform: string
  account: string
  description: string
  timestamp: string
  status: "open" | "in-progress" | "resolved"
  userId: string
}

export interface DetectionResult {
  id: string
  riskLevel: "low" | "medium" | "high" | "critical"
  riskScore: number
  threats: Threat[]
  recommendations: string[]
  analysisTime: number
  createdAt: string
}

export interface ScreenshotAnalysis {
  id: string
  authenticityScore: number
  isAuthentic: boolean
  confidence: number
  manipulations: Array<{
    type: string
    location: string
    severity: "low" | "medium" | "high"
    description: string
  }>
  metadata: {
    resolution: string
    format: string
    fileSize: string
    captureTime: string
    device: string
  }
  createdAt: string
}

export interface TrainingSession {
  id: string
  name: string
  status: "idle" | "training" | "completed" | "failed"
  progress: number
  accuracy: number
  loss: number
  epoch: number
  totalEpochs: number
  metrics: {
    precision: number
    recall: number
    f1Score: number
    auc: number
  }
  createdAt: string
  updatedAt: string
}

export interface AnalyticsData {
  totalThreats: number
  resolvedThreats: number
  inProgressThreats: number
  detectionRate: number
  threatTrend: Array<{
    date: string
    threats: number
    resolved: number
  }>
  platformDistribution: Array<{
    platform: string
    count: number
  }>
  severityDistribution: Array<{
    severity: string
    count: number
  }>
}

export interface Report {
  id: string
  type: "comprehensive" | "summary" | "threats" | "analytics"
  format: "pdf" | "csv" | "json"
  dateRange: {
    startDate: string
    endDate: string
  }
  createdAt: string
  url: string
}
