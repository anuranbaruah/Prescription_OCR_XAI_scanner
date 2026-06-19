// Thin API client for the FastAPI backend (proxied via Vite to :8000).

export async function getHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error("Backend not reachable");
  return res.json();
}

export async function getModelComparison() {
  const res = await fetch("/api/model-comparison");
  if (!res.ok) throw new Error("Could not load benchmarks");
  return res.json();
}

export async function analyzePrescription(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/analyze", { method: "POST", body: form });
  if (!res.ok) {
    let detail = "Analysis failed";
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}
