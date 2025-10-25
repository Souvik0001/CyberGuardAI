"use client"

import { useState, useCallback, useEffect } from "react"
import { apiClient } from "@/lib/api-client"
import type { User, AuthResponse } from "@/lib/api-types"

export function useAuth() {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Check if user is already logged in
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem("auth_token")
      if (token) {
        apiClient.setToken(token)
        try {
          const response = await apiClient.get<User>("/auth/me")
          if (response.success && response.data) {
            setUser(response.data)
          } else {
            localStorage.removeItem("auth_token")
            apiClient.clearToken()
          }
        } catch (err) {
          localStorage.removeItem("auth_token")
          apiClient.clearToken()
        }
      }
      setIsLoading(false)
    }

    checkAuth()
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await apiClient.post<AuthResponse>("/auth/login", { email, password })

      if (response.success && response.data) {
        apiClient.setToken(response.data.token)
        setUser(response.data.user)
        return { success: true, user: response.data.user }
      } else {
        const errorMessage = response.error || "Login failed"
        setError(errorMessage)
        return { success: false, error: errorMessage }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Login failed"
      setError(errorMessage)
      return { success: false, error: errorMessage }
    } finally {
      setIsLoading(false)
    }
  }, [])

  const signup = useCallback(async (email: string, password: string, name: string) => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await apiClient.post<AuthResponse>("/auth/signup", { email, password, name })

      if (response.success && response.data) {
        apiClient.setToken(response.data.token)
        setUser(response.data.user)
        return { success: true, user: response.data.user }
      } else {
        const errorMessage = response.error || "Signup failed"
        setError(errorMessage)
        return { success: false, error: errorMessage }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Signup failed"
      setError(errorMessage)
      return { success: false, error: errorMessage }
    } finally {
      setIsLoading(false)
    }
  }, [])

  const logout = useCallback(() => {
    apiClient.clearToken()
    setUser(null)
    localStorage.removeItem("auth_token")
  }, [])

  return {
    user,
    isLoading,
    error,
    login,
    signup,
    logout,
    isAuthenticated: !!user,
  }
}
