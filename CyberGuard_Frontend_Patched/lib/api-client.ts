/**
 * lib/api-client.ts
 * Centralized client for CyberGuard backend.
 * Uses NEXT_PUBLIC_API_URL or falls back to http://localhost:8000
 */

export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
}

const API_BASE = (process.env.NEXT_PUBLIC_API_URL as string) || "http://localhost:8000"

async function handleJsonResponse(res: Response) {
  try {
    return await res.json()
  } catch (e) {
    return {}
  }
}

/**
 * Train models.
 * options.file (optional) -> File to upload (CSV)
 * options.epochs (optional) -> number of epochs to request from backend
 */
export async function trainModels(options?: { file?: File; epochs?: number }): Promise<ApiResponse> {
  try {
    const fd = new FormData()
    if (options?.file) fd.append("file", options.file)
    if (typeof options?.epochs === "number") fd.append("epochs", String(options.epochs))

    const res = await fetch(`${API_BASE.replace(/\/$/, "")}/train_models`, {
      method: "POST",
      body: fd,
    })

    const data = await handleJsonResponse(res)

    if (!res.ok) {
      const errMsg = data?.error ?? data?.message ?? `Status ${res.status}`
      return { success: false, error: errMsg }
    }

    return { success: true, data }
  } catch (err: any) {
    return { success: false, error: err?.message ?? String(err) }
  }
}

/** Analyze chat (text or csv_file form field) */
export async function analyzeChat(opts: { text?: string; csvFile?: File }): Promise<ApiResponse> {
  try {
    const fd = new FormData()
    if (opts.text) fd.append("text", opts.text)
    if (opts.csvFile) fd.append("csv_file", opts.csvFile)

    const res = await fetch(`${API_BASE.replace(/\/$/, "")}/analyze_chat`, {
      method: "POST",
      body: fd,
    })
    const data = await handleJsonResponse(res)
    if (!res.ok) return { success: false, error: data?.error ?? data?.message ?? `Status ${res.status}` }
    return { success: true, data }
  } catch (err: any) {
    return { success: false, error: err?.message ?? String(err) }
  }
}

/** Analyze screenshots (files form field 'files') */
export async function analyzeScreenshots(files: File[]): Promise<ApiResponse> {
  try {
    const fd = new FormData()
    files.forEach((f) => fd.append("files", f))
    const res = await fetch(`${API_BASE.replace(/\/$/, "")}/analyze_screenshot`, {
      method: "POST",
      body: fd,
    })
    const data = await handleJsonResponse(res)
    if (!res.ok) return { success: false, error: data?.error ?? data?.message ?? `Status ${res.status}` }
    return { success: true, data }
  } catch (err: any) {
    return { success: false, error: err?.message ?? String(err) }
  }
}

/** Upload dataset helper (field 'file') - placeholder if you add upload_dataset later */
export async function uploadDataset(file: File): Promise<ApiResponse> {
  try {
    const fd = new FormData()
    fd.append("file", file)
    const res = await fetch(`${API_BASE.replace(/\/$/, "")}/upload_dataset`, {
      method: "POST",
      body: fd,
    })
    const data = await handleJsonResponse(res)
    if (!res.ok) return { success: false, error: data?.error ?? data?.message ?? `Status ${res.status}` }
    return { success: true, data }
  } catch (err: any) {
    return { success: false, error: err?.message ?? String(err) }
  }
}
