"use client"

import { useState, useCallback } from "react"
import { apiClient, type ApiResponse } from "@/lib/api-client"

interface UseApiOptions {
  onSuccess?: (data: unknown) => void
  onError?: (error: string) => void
}

export function useApi<T>(options: UseApiOptions = {}) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const execute = useCallback(
    async (method: "get" | "post" | "put" | "delete", endpoint: string, payload?: unknown): Promise<ApiResponse<T>> => {
      setIsLoading(true)
      setError(null)

      try {
        let response: ApiResponse<T>

        switch (method) {
          case "get":
            response = await apiClient.get<T>(endpoint)
            break
          case "post":
            response = await apiClient.post<T>(endpoint, payload)
            break
          case "put":
            response = await apiClient.put<T>(endpoint, payload)
            break
          case "delete":
            response = await apiClient.delete<T>(endpoint)
            break
          default:
            throw new Error("Invalid method")
        }

        if (response.success && response.data) {
          setData(response.data)
          options.onSuccess?.(response.data)
        } else {
          const errorMessage = response.error || "An error occurred"
          setError(errorMessage)
          options.onError?.(errorMessage)
        }

        return response
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "An unknown error occurred"
        setError(errorMessage)
        options.onError?.(errorMessage)
        return { success: false, error: errorMessage }
      } finally {
        setIsLoading(false)
      }
    },
    [options],
  )

  const get = useCallback((endpoint: string) => execute("get", endpoint), [execute])
  const post = useCallback((endpoint: string, payload?: unknown) => execute("post", endpoint, payload), [execute])
  const put = useCallback((endpoint: string, payload?: unknown) => execute("put", endpoint, payload), [execute])
  const del = useCallback((endpoint: string) => execute("delete", endpoint), [execute])

  return {
    data,
    error,
    isLoading,
    get,
    post,
    put,
    delete: del,
    execute,
  }
}
