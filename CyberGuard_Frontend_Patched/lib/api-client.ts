// frontend/lib/api-client.ts

const API_BASE = "http://localhost:8000"; // FastAPI server

// -------------------------------
// Screenshot Analysis (already in your app)
// -------------------------------
export async function analyzeScreenshots(files: File[]) {
  const formData = new FormData();
  for (const f of files) {
    formData.append("files", f);
  }

  try {
    const res = await fetch(`${API_BASE}/analyze_screenshot`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      return { success: false, error: `HTTP ${res.status}` };
    }

    const data = await res.json();
    return { success: true, data };
  } catch (err: any) {
    return { success: false, error: String(err) };
  }
}

// -------------------------------
// Stalker Detection (for /detection page)
// -------------------------------
export async function analyzeChat(
  descriptionText: string,
  csvFile: File | null
) {
  const formData = new FormData();
  if (descriptionText && descriptionText.trim() !== "") {
    formData.append("text", descriptionText);
  }
  if (csvFile) {
    formData.append("csv_file", csvFile);
  }

  try {
    const res = await fetch(`${API_BASE}/analyze_chat`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      return { success: false, error: `HTTP ${res.status}` };
    }
    const data = await res.json();
    return { success: true, data };
  } catch (err: any) {
    return { success: false, error: String(err) };
  }
}

// -------------------------------
// Training helpers already in UI
// -------------------------------

// Save one harassment sample (single message or multi-line) to harassment_samples.jsonl
export async function saveHarassmentSample(text: string, isAbusive: boolean) {
  // We'll POST JSON to a tiny helper route we simulate client-side,
  // but we actually don't have a dedicated backend route for single-sample
  // append in the code we shipped -- in your current codebase you probably
  // already had something like this. If not, easiest path:
  // client writes directly to harassment_samples.jsonl is not possible from browser.
  //
  // For now we'll *simulate* by using bulk_upload_harassment with a tiny JSONL blob.
  const fileBlob = new Blob(
    [
      JSON.stringify({
        text,
        label: isAbusive ? 1 : 0,
      }) + "\n",
    ],
    { type: "application/jsonl" }
  );

  const formData = new FormData();
  formData.append("dataset", fileBlob, "sample.jsonl");

  try {
    const res = await fetch(`${API_BASE}/bulk_upload_harassment`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      return { success: false, error: `HTTP ${res.status}` };
    }
    const data = await res.json();
    return { success: true, data };
  } catch (err: any) {
    return { success: false, error: String(err) };
  }
}

// Save one tamper sample (ela/res etc.) to tamper_samples.jsonl
export async function saveTamperSample(
  elaScore: string,
  widthPx: string,
  heightPx: string,
  isTampered: boolean
) {
  const fileBlob = new Blob(
    [
      JSON.stringify({
        ela: parseFloat(elaScore || "0"),
        res_w: parseInt(widthPx || "0", 10),
        res_h: parseInt(heightPx || "0", 10),
        label: isTampered ? 1 : 0,
      }) + "\n",
    ],
    { type: "application/jsonl" }
  );

  const formData = new FormData();
  formData.append("dataset", fileBlob, "tamper.jsonl");

  try {
    const res = await fetch(`${API_BASE}/bulk_upload_tamper`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      return { success: false, error: `HTTP ${res.status}` };
    }
    const data = await res.json();
    return { success: true, data };
  } catch (err: any) {
    return { success: false, error: String(err) };
  }
}

// Retrain models using whatever is in the .jsonl files
export async function retrainModels() {
  try {
    const res = await fetch(`${API_BASE}/train_models`, {
      method: "POST",
    });
    if (!res.ok) {
      return { success: false, error: `HTTP ${res.status}` };
    }
    const data = await res.json();
    return { success: true, data };
  } catch (err: any) {
    return { success: false, error: String(err) };
  }
}

// -------------------------------
// NEW: bulk dataset upload from UI
// -------------------------------
export async function uploadHarassmentDataset(file: File) {
  const formData = new FormData();
  formData.append("dataset", file);

  try {
    const res = await fetch(`${API_BASE}/bulk_upload_harassment`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      return { success: false, error: `HTTP ${res.status}` };
    }
    const data = await res.json();
    return { success: true, data };
  } catch (err: any) {
    return { success: false, error: String(err) };
  }
}

export async function uploadTamperDataset(file: File) {
  const formData = new FormData();
  formData.append("dataset", file);

  try {
    const res = await fetch(`${API_BASE}/bulk_upload_tamper`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      return { success: false, error: `HTTP ${res.status}` };
    }
    const data = await res.json();
    return { success: true, data };
  } catch (err: any) {
    return { success: false, error: String(err) };
  }
}
