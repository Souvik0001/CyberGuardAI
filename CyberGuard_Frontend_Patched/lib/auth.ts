// Authentication utilities for CyberGuard

export function isTokenValid(token: string): boolean {
  try {
    const parts = token.split(".")
    if (parts.length !== 3) return false

    const payload = JSON.parse(atob(parts[1]))
    const expirationTime = payload.exp * 1000

    return Date.now() < expirationTime
  } catch {
    return false
  }
}

export function getTokenExpiration(token: string): Date | null {
  try {
    const parts = token.split(".")
    if (parts.length !== 3) return null

    const payload = JSON.parse(atob(parts[1]))
    return new Date(payload.exp * 1000)
  } catch {
    return null
  }
}

export function decodeToken(token: string): Record<string, unknown> | null {
  try {
    const parts = token.split(".")
    if (parts.length !== 3) return null

    return JSON.parse(atob(parts[1]))
  } catch {
    return null
  }
}
