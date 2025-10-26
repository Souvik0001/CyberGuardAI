// /**
//  * lib/api-client.ts
//  * Centralized client for CyberGuard backend.
//  * Uses NEXT_PUBLIC_API_URL or falls back to http://localhost:8000
//  */

// export interface ApiResponse<T = any> {
//   success: boolean
//   data?: T
//   error?: string
// }

// const API_BASE = (process.env.NEXT_PUBLIC_API_URL as string) || "http://localhost:8000"

// async function handleJsonResponse(res: Response) {
//   try {
//     return await res.json()
//   } catch (e) {
//     return {}
//   }
// }

// /**
//  * Train models.
//  * options.file (optional) -> File to upload (CSV)
//  * options.epochs (optional) -> number of epochs to request from backend
//  */
// export async function trainModels(options?: { file?: File; epochs?: number }): Promise<ApiResponse> {
//   try {
//     const fd = new FormData()
//     if (options?.file) fd.append("file", options.file)
//     if (typeof options?.epochs === "number") fd.append("epochs", String(options.epochs))

//     const res = await fetch(`${API_BASE.replace(/\/$/, "")}/train_models`, {
//       method: "POST",
//       body: fd,
//     })

//     const data = await handleJsonResponse(res)

//     if (!res.ok) {
//       const errMsg = data?.error ?? data?.message ?? `Status ${res.status}`
//       return { success: false, error: errMsg }
//     }

//     return { success: true, data }
//   } catch (err: any) {
//     return { success: false, error: err?.message ?? String(err) }
//   }
// }

// /** Analyze chat (text or csv_file form field) */
// export async function analyzeChat(opts: { text?: string; csvFile?: File }): Promise<ApiResponse> {
//   try {
//     const fd = new FormData()
//     if (opts.text) fd.append("text", opts.text)
//     if (opts.csvFile) fd.append("csv_file", opts.csvFile)

//     const res = await fetch(`${API_BASE.replace(/\/$/, "")}/analyze_chat`, {
//       method: "POST",
//       body: fd,
//     })
//     const data = await handleJsonResponse(res)
//     if (!res.ok) return { success: false, error: data?.error ?? data?.message ?? `Status ${res.status}` }
//     return { success: true, data }
//   } catch (err: any) {
//     return { success: false, error: err?.message ?? String(err) }
//   }
// }

// /** Analyze screenshots (files form field 'files') */
// export async function analyzeScreenshots(files: File[]): Promise<ApiResponse> {
//   try {
//     const fd = new FormData()
//     files.forEach((f) => fd.append("files", f))
//     const res = await fetch(`${API_BASE.replace(/\/$/, "")}/analyze_screenshot`, {
//       method: "POST",
//       body: fd,
//     })
//     const data = await handleJsonResponse(res)
//     if (!res.ok) return { success: false, error: data?.error ?? data?.message ?? `Status ${res.status}` }
//     return { success: true, data }
//   } catch (err: any) {
//     return { success: false, error: err?.message ?? String(err) }
//   }
// }

// /** Upload dataset helper (field 'file') - placeholder if you add upload_dataset later */
// export async function uploadDataset(file: File): Promise<ApiResponse> {
//   try {
//     const fd = new FormData()
//     fd.append("file", file)
//     const res = await fetch(`${API_BASE.replace(/\/$/, "")}/upload_dataset`, {
//       method: "POST",
//       body: fd,
//     })
//     const data = await handleJsonResponse(res)
//     if (!res.ok) return { success: false, error: data?.error ?? data?.message ?? `Status ${res.status}` }
//     return { success: true, data }
//   } catch (err: any) {
//     return { success: false, error: err?.message ?? String(err) }
//   }
// }

/**
 * lib/api-client.ts
 * Centralized client for CyberGuard backend.
 * Uses NEXT_PUBLIC_API_URL or falls back to http://localhost:8000
 *
 * Reworked to: - log requests/responses for easier debugging
 *               - handle non-JSON responses gracefully
 *               - provide clearer error messages
 */

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
}

const API_BASE = (process.env.NEXT_PUBLIC_API_URL as string) || "http://localhost:8000";

async function handleJsonResponse(res: Response) {
  // Try parse JSON; if fails return the plain text so caller can report error.
  const text = await res.text();
  try {
    return text ? JSON.parse(text) : {};
  } catch (e) {
    // Not JSON - return text under a predictable key so error messages are visible
    return { _rawText: text };
  }
}

function buildUrl(path: string) {
  // ensure no double-slash
  return `${API_BASE.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
}

async function safeFetch(url: string, opts: RequestInit) {
  // wrapper to log and run fetch
  console.debug("[api-client] fetch ->", url, opts);
  const res = await fetch(url, opts);
  const parsed = await handleJsonResponse(res);
  console.debug("[api-client] response ->", url, res.status, parsed);
  return { res, parsed };
}

/** Train models. options.file (optional) -> File to upload (CSV). options.epochs (optional) -> number */
export async function trainModels(options?: { file?: File; epochs?: number }): Promise<ApiResponse> {
  try {
    const fd = new FormData();
    if (options?.file) fd.append("file", options.file);
    if (typeof options?.epochs === "number") fd.append("epochs", String(options.epochs));

    const url = buildUrl("/train_models");
    const { res, parsed } = await safeFetch(url, {
      method: "POST",
      body: fd,
      // browser sets multipart boundary automatically
    });

    if (!res.ok) {
      const errMsg = (parsed && (parsed.error || parsed.message || parsed._rawText)) || `Status ${res.status}`;
      return { success: false, error: String(errMsg) };
    }

    return { success: true, data: parsed };
  } catch (err: any) {
    console.error("[api-client.trainModels] error", err);
    return { success: false, error: err?.message ?? String(err) };
  }
}

/** Analyze chat (text or csv_file form field) */
export async function analyzeChat(opts: { text?: string; csvFile?: File }): Promise<ApiResponse> {
  try {
    const fd = new FormData();
    if (opts.text) fd.append("text", opts.text);
    if (opts.csvFile) fd.append("csv_file", opts.csvFile);

    const url = buildUrl("/analyze_chat");
    const { res, parsed } = await safeFetch(url, {
      method: "POST",
      body: fd,
    });

    if (!res.ok) {
      const errMsg = (parsed && (parsed.error || parsed.message || parsed._rawText)) || `Status ${res.status}`;
      return { success: false, error: String(errMsg) };
    }
    return { success: true, data: parsed };
  } catch (err: any) {
    console.error("[api-client.analyzeChat] error", err);
    return { success: false, error: err?.message ?? String(err) };
  }
}

/** Analyze screenshots (files form field 'files') */
export async function analyzeScreenshots(files: File[]): Promise<ApiResponse> {
  try {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));

    const url = buildUrl("/analyze_screenshot");
    const { res, parsed } = await safeFetch(url, {
      method: "POST",
      body: fd,
    });

    if (!res.ok) {
      const errMsg = (parsed && (parsed.error || parsed.message || parsed._rawText)) || `Status ${res.status}`;
      return { success: false, error: String(errMsg) };
    }

    return { success: true, data: parsed };
  } catch (err: any) {
    console.error("[api-client.analyzeScreenshots] error", err);
    return { success: false, error: err?.message ?? String(err) };
  }
}

/** Upload dataset helper (field 'file') - placeholder if you add upload_dataset later */
export async function uploadDataset(file: File): Promise<ApiResponse> {
  try {
    const fd = new FormData();
    fd.append("file", file);
    const url = buildUrl("/upload_dataset");
    const { res, parsed } = await safeFetch(url, {
      method: "POST",
      body: fd,
    });

    if (!res.ok) {
      const errMsg = (parsed && (parsed.error || parsed.message || parsed._rawText)) || `Status ${res.status}`;
      return { success: false, error: String(errMsg) };
    }
    return { success: true, data: parsed };
  } catch (err: any) {
    console.error("[api-client.uploadDataset] error", err);
    return { success: false, error: err?.message ?? String(err) };
  }
}
