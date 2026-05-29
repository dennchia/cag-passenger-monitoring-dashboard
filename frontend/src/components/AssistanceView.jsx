import { Camera, Clock, RefreshCw, Search, ShieldCheck, Trash2, UserRound } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { endpoints, fetchJson, resolveApiUrl, withQuery } from "../lib/api.js";

const POLL_MS = 3000;

function formatTime(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    month: "short",
    day: "numeric",
  });
}

function formatAge(value) {
  const age = Number(value);
  if (!Number.isFinite(age)) return "Unknown";
  return Number.isInteger(age) ? String(age) : age.toFixed(1);
}

function formatConfidence(value) {
  const confidence = Number(value);
  if (!Number.isFinite(confidence)) return "Not provided";
  return `${Math.round(confidence * 100)}%`;
}

const initialFilters = {
  gender: "",
  min_age: "",
  max_age: "",
  camera_id: "",
  run_id: "",
};

export default function AssistanceView({ cameras = [] }) {
  const [filters, setFilters] = useState(initialFilters);
  const [observations, setObservations] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  const queryUrl = useMemo(() => withQuery(endpoints.observations, filters), [filters]);

  async function loadObservations() {
    try {
      const data = await fetchJson(queryUrl);
      setObservations(Array.isArray(data) ? data : []);
      setError("");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not load observations.");
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    let isMounted = true;

    async function poll() {
      try {
        const data = await fetchJson(queryUrl);
        if (!isMounted) return;
        setObservations(Array.isArray(data) ? data : []);
        setError("");
      } catch (nextError) {
        if (!isMounted) return;
        setError(nextError instanceof Error ? nextError.message : "Could not load observations.");
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }

    poll();
    const interval = window.setInterval(poll, POLL_MS);
    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, [queryUrl]);

  function updateFilter(key, value) {
    setFilters((current) => ({ ...current, [key]: value }));
  }

  async function clearDemoLogs() {
    const confirmed = window.confirm("Clear all saved passenger assistance demo observations and crop images?");
    if (!confirmed) return;

    try {
      await fetchJson(endpoints.observations, { method: "DELETE" });
      setObservations([]);
      setError("");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Could not clear observations.");
    }
  }

  return (
    <section className="grid gap-5">
      <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="mb-2 inline-flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-cyan-200">
              <ShieldCheck className="h-4 w-4" />
              Assistance Filter
            </div>
            <h2 className="text-2xl font-black text-white">Passenger Assistance</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              Filter model-estimated age and gender observations from person crops. This view helps staff narrow a
              manual check; it does not identify people or perform face recognition.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={loadObservations}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-bold text-slate-200 transition hover:border-slate-500 hover:text-white"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
            <button
              type="button"
              onClick={clearDemoLogs}
              className="inline-flex items-center gap-2 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm font-bold text-red-100 transition hover:bg-red-500/20"
            >
              <Trash2 className="h-4 w-4" />
              Clear Logs
            </button>
          </div>
        </div>

        <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          <label className="grid gap-1 text-sm">
            <span className="font-bold text-slate-300">Gender</span>
            <select
              value={filters.gender}
              onChange={(event) => updateFilter("gender", event.target.value)}
              className="h-10 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-white outline-none focus:border-cyan-300"
            >
              <option value="">Any gender</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="unknown">Unknown / Other</option>
            </select>
          </label>

          <label className="grid gap-1 text-sm">
            <span className="font-bold text-slate-300">Min Age</span>
            <input
              type="number"
              min="0"
              max="120"
              value={filters.min_age}
              onChange={(event) => updateFilter("min_age", event.target.value)}
              placeholder="0"
              className="h-10 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-white outline-none placeholder:text-slate-600 focus:border-cyan-300"
            />
          </label>

          <label className="grid gap-1 text-sm">
            <span className="font-bold text-slate-300">Max Age</span>
            <input
              type="number"
              min="0"
              max="120"
              value={filters.max_age}
              onChange={(event) => updateFilter("max_age", event.target.value)}
              placeholder="120"
              className="h-10 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-white outline-none placeholder:text-slate-600 focus:border-cyan-300"
            />
          </label>

          <label className="grid gap-1 text-sm">
            <span className="font-bold text-slate-300">Camera</span>
            <select
              value={filters.camera_id}
              onChange={(event) => updateFilter("camera_id", event.target.value)}
              className="h-10 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-white outline-none focus:border-cyan-300"
            >
              <option value="">Any camera</option>
              {cameras.map((camera) => (
                <option key={camera.camera_id} value={camera.camera_id}>
                  {camera.camera_id}
                </option>
              ))}
            </select>
          </label>

          <label className="grid gap-1 text-sm">
            <span className="font-bold text-slate-300">Run ID</span>
            <input
              type="text"
              value={filters.run_id}
              onChange={(event) => updateFilter("run_id", event.target.value)}
              placeholder="default"
              className="h-10 rounded-md border border-slate-700 bg-slate-950 px-3 text-sm text-white outline-none placeholder:text-slate-600 focus:border-cyan-300"
            />
          </label>
        </div>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm font-bold text-red-100">
          {error}
        </div>
      ) : null}

      <div className="flex items-center justify-between gap-3">
        <div className="inline-flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-slate-300">
          <Search className="h-4 w-4 text-cyan-300" />
          {isLoading ? "Loading observations" : `${observations.length} matching observations`}
        </div>
        <div className="text-xs text-slate-500">Updated every 3 seconds</div>
      </div>

      {observations.length ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {observations.map((observation) => (
            <article
              key={observation.id}
              className="overflow-hidden rounded-lg border border-slate-800 bg-slate-900/70"
            >
              <div className="aspect-[4/5] bg-slate-950">
                <img
                  src={resolveApiUrl(observation.image_url)}
                  alt={`Passenger crop from ${observation.camera_id}`}
                  className="h-full w-full object-cover"
                />
              </div>
              <div className="grid gap-3 p-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-2xl font-black text-white">{formatAge(observation.age)}</div>
                    <div className="text-xs font-bold uppercase tracking-wide text-slate-500">estimated age</div>
                  </div>
                  <div className="rounded-full border border-cyan-300/30 bg-cyan-300/10 px-3 py-1 text-sm font-bold capitalize text-cyan-100">
                    {observation.gender || "unknown"}
                  </div>
                </div>

                <div className="grid gap-2 text-sm text-slate-300">
                  <div className="flex items-center gap-2">
                    <Camera className="h-4 w-4 text-slate-500" />
                    <span>{observation.camera_id}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4 text-slate-500" />
                    <span>{formatTime(observation.timestamp)}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <UserRound className="h-4 w-4 text-slate-500" />
                    <span>Track {observation.track_id || "not provided"}</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 border-t border-slate-800 pt-3 text-xs">
                  <div>
                    <div className="font-bold text-slate-300">{formatConfidence(observation.age_confidence)}</div>
                    <div className="text-slate-500">age conf.</div>
                  </div>
                  <div>
                    <div className="font-bold text-slate-300">{formatConfidence(observation.gender_confidence)}</div>
                    <div className="text-slate-500">gender conf.</div>
                  </div>
                </div>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <div className="rounded-lg border border-dashed border-slate-700 bg-slate-900/40 px-4 py-12 text-center">
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-slate-800">
            <Search className="h-5 w-5 text-slate-400" />
          </div>
          <div className="text-lg font-bold text-white">No matching observations</div>
          <p className="mt-1 text-sm text-slate-400">
            Observations will appear here after the processing pipeline posts age, gender, and a person crop.
          </p>
        </div>
      )}
    </section>
  );
}
