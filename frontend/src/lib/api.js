export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

export const endpoints = {
  health: `${API_URL}/health`,
  status: `${API_URL}/api/status`,
  cameras: `${API_URL}/api/cameras`,
  stream: `${API_URL}/api/stream`,
  cameraStatus: (cameraId) => `${API_URL}/api/cameras/${encodeURIComponent(cameraId)}/status`,
  cameraStream: (cameraId) => `${API_URL}/api/cameras/${encodeURIComponent(cameraId)}/stream`,
  metrics: `${API_URL}/api/metrics`,
  alerts: `${API_URL}/api/alerts`,
  observations: `${API_URL}/api/observations`,
  observationsSummary: `${API_URL}/api/observations/summary`,
};

export function resolveApiUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export function withQuery(url, params = {}) {
  const nextUrl = new URL(url);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      nextUrl.searchParams.set(key, value);
    }
  });
  return nextUrl.toString();
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
