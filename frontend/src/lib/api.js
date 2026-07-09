export const API_URL = (import.meta.env.VITE_API_URL || "").trim().replace(/\/$/, "");

function apiUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return API_URL ? `${API_URL}${normalizedPath}` : normalizedPath;
}

export const endpoints = {
  health: apiUrl("/health"),
  status: apiUrl("/api/status"),
  cameras: apiUrl("/api/cameras"),
  stream: apiUrl("/api/stream"),
  cameraStatus: (cameraId) => apiUrl(`/api/cameras/${encodeURIComponent(cameraId)}/status`),
  cameraStream: (cameraId) => apiUrl(`/api/cameras/${encodeURIComponent(cameraId)}/stream`),
  metrics: apiUrl("/api/metrics"),
  metricTrends: apiUrl("/api/metrics/trends"),
  zoneStatus: apiUrl("/api/zones/status"),
  tactical: apiUrl("/api/tactical"),
  tacticalLatest: (cameraId, runId) =>
    withQuery(apiUrl("/api/tactical/latest"), { camera_id: cameraId, run_id: runId }),
  alerts: apiUrl("/api/alerts"),
  observations: apiUrl("/api/observations"),
  observationsSummary: apiUrl("/api/observations/summary"),
  shiftReportCsv: (runId) => withQuery(apiUrl("/api/reports/shift.csv"), { run_id: runId }),
  shiftReportXlsx: (runId) => withQuery(apiUrl("/api/reports/shift.xlsx"), { run_id: runId }),
};

export function resolveApiUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return apiUrl(path);
}

export function withQuery(url, params = {}) {
  const isAbsoluteUrl = /^https?:\/\//i.test(url);
  const nextUrl = new URL(url, isAbsoluteUrl ? undefined : window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      nextUrl.searchParams.set(key, value);
    }
  });
  return isAbsoluteUrl || API_URL ? nextUrl.toString() : `${nextUrl.pathname}${nextUrl.search}`;
}

export async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      Accept: "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`Request failed with ${response.status}`);
  }

  return response.json();
}
