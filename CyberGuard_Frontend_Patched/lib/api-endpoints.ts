// API endpoint constants for CyberGuard

export const API_ENDPOINTS = {
  // Authentication
  AUTH: {
    LOGIN: "/auth/login",
    SIGNUP: "/auth/signup",
    LOGOUT: "/auth/logout",
    ME: "/auth/me",
    REFRESH: "/auth/refresh",
  },

  // Detection
  DETECTION: {
    ANALYZE: "/detection/analyze",
    HISTORY: "/detection/history",
    GET: (id: string) => `/detection/${id}`,
  },

  // Screenshot Analysis
  ANALYSIS: {
    UPLOAD: "/analysis/upload",
    HISTORY: "/analysis/history",
    GET: (id: string) => `/analysis/${id}`,
  },

  // Model Training
  TRAINING: {
    START: "/training/start",
    STATUS: (id: string) => `/training/${id}`,
    HISTORY: "/training/history",
    PAUSE: (id: string) => `/training/${id}/pause`,
    STOP: (id: string) => `/training/${id}/stop`,
  },

  // Reports
  REPORTS: {
    GENERATE: "/reports/generate",
    HISTORY: "/reports/history",
    GET: (id: string) => `/reports/${id}`,
    DOWNLOAD: (id: string) => `/reports/${id}/download`,
  },

  // Analytics
  ANALYTICS: {
    DASHBOARD: "/analytics/dashboard",
    THREATS: "/analytics/threats",
    PLATFORMS: "/analytics/platforms",
    SEVERITY: "/analytics/severity",
  },

  // Datasets
  DATASETS: {
    UPLOAD: "/datasets/upload",
    LIST: "/datasets",
    DELETE: (id: string) => `/datasets/${id}`,
  },
}
