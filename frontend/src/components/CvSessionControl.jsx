import { AlertTriangle, LoaderCircle, Play, Square } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { endpoints, fetchJson } from "../lib/api.js";

const POLL_MS = 1000;

const labels = {
  offline: "Offline",
  loading: "Preparing computer vision",
  ready: "Ready",
  starting: "Starting session",
  running: "Session running",
  stopping: "Stopping session",
  failed: "Computer vision unavailable",
};

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export default function CvSessionControl() {
  const [status, setStatus] = useState(null);
  const [operatorToken, setOperatorToken] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [requestError, setRequestError] = useState("");

  const requestHeaders = useMemo(
    () => (operatorToken ? { "X-Operator-Token": operatorToken } : {}),
    [operatorToken],
  );

  useEffect(() => {
    let mounted = true;

    async function poll() {
      try {
        const nextStatus = await fetchJson(endpoints.cvStatus, { headers: requestHeaders });
        if (mounted) setStatus(nextStatus);
      } catch (error) {
        if (mounted) {
          setRequestError(error instanceof Error ? error.message : "Unable to read CV status.");
        }
      }
    }

    poll();
    const interval = window.setInterval(poll, POLL_MS);
    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, [requestHeaders]);

  async function sendControl(url) {
    setSubmitting(true);
    setRequestError("");
    try {
      const nextStatus = await fetchJson(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...requestHeaders },
        body: JSON.stringify({}),
      });
      setStatus(nextStatus);
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : "The request failed.");
    } finally {
      setSubmitting(false);
    }
  }

  const state = status?.state || "offline";
  const loading = state === "loading";
  const active = ["starting", "running", "stopping"].includes(state);
  const canStart = Boolean(status?.ready && status?.control_allowed && !submitting);
  const canStop = Boolean(
    ["starting", "running"].includes(state) && status?.control_allowed && !submitting,
  );
  const showAccessCode = status?.control_mode === "token" && !status?.control_allowed;

  return (
    <section className="mb-5 rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-2xl">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {loading || state === "starting" || state === "stopping" ? (
              <LoaderCircle className="h-5 w-5 animate-spin text-cyan-300" />
            ) : state === "failed" ? (
              <AlertTriangle className="h-5 w-5 text-red-300" />
            ) : (
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  state === "running"
                    ? "bg-emerald-400"
                    : state === "ready"
                      ? "bg-cyan-300"
                      : "bg-slate-500"
                }`}
              />
            )}
            <h2 className="text-base font-black text-white">{labels[state] || state}</h2>
          </div>
          <p className="mt-1 text-sm text-slate-400">
            {loading
              ? status?.loading_stage || "Loading required models"
              : state === "ready"
                ? "The system is prepared and waiting for an operator."
                : state === "running"
                  ? `Run ${status?.run_id || "active"} · Started ${formatTime(status?.started_at)}`
                  : state === "failed"
                    ? status?.error || "Restart the server or ask a technician for assistance."
                    : active
                      ? "Please wait for this operation to complete."
                      : "Waiting for the computer-vision worker."}
          </p>
          <div className="mt-2 flex flex-wrap gap-2 text-xs font-bold">
            <span
              className={`rounded-full px-2.5 py-1 ${
                status?.mqtt_broker_reachable
                  ? "bg-emerald-500/15 text-emerald-100"
                  : "bg-amber-500/15 text-amber-100"
              }`}
            >
              MQTT {status?.mqtt_broker_reachable ? "Connected" : "Unavailable"}
            </span>
            {status?.run_id ? (
              <span className="rounded-full bg-slate-800 px-2.5 py-1 text-slate-300">
                Run ID: {status.run_id}
              </span>
            ) : null}
          </div>
        </div>

        <div className="flex min-w-[210px] flex-col gap-2">
          {showAccessCode ? (
            <input
              type="password"
              value={operatorToken}
              onChange={(event) => setOperatorToken(event.target.value)}
              placeholder="Operator access code"
              autoComplete="off"
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-cyan-300"
            />
          ) : null}
          {state === "ready" ? (
            <button
              type="button"
              disabled={!canStart}
              onClick={() => sendControl(endpoints.cvStart)}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-300 px-5 py-3 text-sm font-black text-slate-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Play className="h-4 w-4" />
              Start Session
            </button>
          ) : active ? (
            <button
              type="button"
              disabled={!canStop}
              onClick={() => sendControl(endpoints.cvStop)}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-red-500 px-5 py-3 text-sm font-black text-white transition hover:bg-red-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Square className="h-4 w-4" />
              {state === "stopping" ? "Stopping…" : "Stop Session"}
            </button>
          ) : (
            <button
              type="button"
              disabled
              className="rounded-lg bg-slate-800 px-5 py-3 text-sm font-black text-slate-400 opacity-70"
            >
              Start Session
            </button>
          )}
          {!status?.control_allowed && status ? (
            <p className="text-center text-xs text-amber-200">
              {status.control_mode === "local_only"
                ? "Session control is available only on the server computer."
                : "Enter the operator access code to control this session."}
            </p>
          ) : null}
          {requestError ? <p className="text-center text-xs text-red-300">{requestError}</p> : null}
        </div>
      </div>
    </section>
  );
}
