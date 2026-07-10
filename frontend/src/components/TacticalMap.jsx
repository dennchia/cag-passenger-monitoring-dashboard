import { MapPinned, RadioTower, UsersRound } from "lucide-react";

const VIEW_SIZE = 100;
const TENT_INSET = 8;
const TENT_SIZE = VIEW_SIZE - TENT_INSET * 2;
const TENT_END = TENT_INSET + TENT_SIZE;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatAge(seconds) {
  if (seconds === null || seconds === undefined) return "waiting";
  if (seconds < 1) return "live";
  return `${seconds.toFixed(1)}s ago`;
}

function formatMeters(cm) {
  const meters = cm / 100;
  return Number.isInteger(meters) ? `${meters}m` : `${meters.toFixed(1)}m`;
}

function classifyPoint(point, mapSize, outsideContext) {
  if (point?.area === "inside" || point?.area === "outside_visible") {
    return point.area;
  }

  const x = Number(point?.x);
  const y = Number(point?.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  if (x >= 0 && x <= mapSize && y >= 0 && y <= mapSize) return "inside";
  if (
    x >= -outsideContext &&
    x <= mapSize + outsideContext &&
    y >= -outsideContext &&
    y <= mapSize + outsideContext
  ) {
    return "outside_visible";
  }
  return null;
}

function mapAxis(value, mapSize, outsideContext) {
  if (value < 0) {
    if (outsideContext <= 0) return TENT_INSET;
    return clamp(((value + outsideContext) / outsideContext) * TENT_INSET, 0, TENT_INSET);
  }

  if (value > mapSize) {
    if (outsideContext <= 0) return TENT_END;
    return clamp(TENT_END + ((value - mapSize) / outsideContext) * TENT_INSET, TENT_END, VIEW_SIZE);
  }

  return TENT_INSET + (value / mapSize) * TENT_SIZE;
}

function mapPoint(point, mapSize, outsideContext) {
  const x = Number(point?.x);
  const y = Number(point?.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;

  const area = classifyPoint(point, mapSize, outsideContext);
  if (!area) return null;

  return {
    area,
    x: mapAxis(x, mapSize, outsideContext),
    y: mapAxis(y, mapSize, outsideContext),
  };
}

function countByArea(points, area) {
  return points.filter((point) => point.area === area).length;
}

function safeCount(value, fallback) {
  const count = Number(value);
  return Number.isFinite(count) ? count : fallback;
}

export default function TacticalMap({ state, cameraId, apiOnline }) {
  const hasData = Boolean(state?.has_data);
  const stale = !apiOnline || Boolean(state?.stale);
  const mapSize = Number(state?.map_size_cm) > 0 ? Number(state.map_size_cm) : 300;
  const outsideContext = Number(state?.outside_context_cm) >= 0 ? Number(state.outside_context_cm) : 700;
  const positions = Array.isArray(state?.positions_cm) ? state.positions_cm : [];
  const plottedPositions = positions
    .map((point) => mapPoint(point, mapSize, outsideContext))
    .filter(Boolean);
  const insideCount = safeCount(state?.inside_count, countByArea(plottedPositions, "inside"));
  const outsideVisibleCount = safeCount(
    state?.outside_visible_count,
    countByArea(plottedPositions, "outside_visible")
  );
  const sourceId = state?.camera_id || cameraId || "fused";
  const sourceLabel = sourceId === "fused" ? "fused map" : sourceId;
  const statusLabel = !apiOnline ? "Backend offline" : !hasData ? "Waiting for tactical data" : stale ? "Stale" : "Live";
  const gridMarks = [0.25, 0.5, 0.75].map((ratio) => TENT_INSET + TENT_SIZE * ratio);
  const tentLabel = `${formatMeters(mapSize)} x ${formatMeters(mapSize)} monitored tent`;

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-300">
            <MapPinned className="h-4 w-4 text-cyan-300" />
            Tactical Floor Map
          </div>
          <p className="mt-1 text-xs text-slate-500">Live fused X/Y foot-position dots from CV homography</p>
          <div className="mt-2 flex flex-wrap gap-3 text-xs font-semibold text-slate-400">
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.55)]" />
              Inside occupancy
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-cyan-300 shadow-[0_0_8px_rgba(103,232,249,0.5)]" />
              Outside visible
            </span>
          </div>
        </div>
        <div
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-bold ${
            !hasData || stale
              ? "border-amber-500/30 bg-amber-500/10 text-amber-100"
              : "border-emerald-500/30 bg-emerald-500/10 text-emerald-100"
          }`}
        >
          <RadioTower className="h-3.5 w-3.5" />
          {statusLabel}
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_150px]">
        <div className="relative overflow-hidden rounded-lg border border-slate-800 bg-slate-950 p-3">
          <svg
            viewBox={`0 0 ${VIEW_SIZE} ${VIEW_SIZE}`}
            role="img"
            aria-label={`Global tactical map from ${sourceLabel}`}
            className="aspect-square w-full"
          >
            <rect x="0" y="0" width={VIEW_SIZE} height={VIEW_SIZE} rx="4" fill="#082f49" />
            <rect
              x="1.5"
              y="1.5"
              width={VIEW_SIZE - 3}
              height={VIEW_SIZE - 3}
              rx="4"
              fill="none"
              stroke="#0e7490"
              strokeWidth="0.8"
              strokeDasharray="2 2"
            />
            <text x="50" y="5" textAnchor="middle" fill="#67e8f9" fontSize="3" fontWeight="700">
              Outside visible area
            </text>

            <rect
              x={TENT_INSET}
              y={TENT_INSET}
              width={TENT_SIZE}
              height={TENT_SIZE}
              rx="3.5"
              fill="#eef2ff"
              stroke="#334155"
              strokeWidth="1.2"
            />
            {gridMarks.map((mark) => (
              <g key={mark}>
                <line x1={mark} y1={TENT_INSET} x2={mark} y2={TENT_END} stroke="#cbd5e1" strokeWidth="0.45" />
                <line x1={TENT_INSET} y1={mark} x2={TENT_END} y2={mark} stroke="#cbd5e1" strokeWidth="0.45" />
              </g>
            ))}
            <text x="11" y="15" fill="#475569" fontSize="3.4" fontWeight="800">
              {tentLabel}
            </text>

            {plottedPositions
              .filter((point) => point.area === "outside_visible")
              .map((point, index) => (
                <g key={`outside-${point.x}-${point.y}-${index}`}>
                  <circle cx={point.x} cy={point.y} r="2.7" fill="#22d3ee" opacity="0.18" />
                  <circle cx={point.x} cy={point.y} r="1.55" fill="#67e8f9" stroke="#155e75" strokeWidth="0.5" />
                </g>
              ))}

            {plottedPositions
              .filter((point) => point.area === "inside")
              .map((point, index) => (
                <g key={`inside-${point.x}-${point.y}-${index}`}>
                  <circle cx={point.x} cy={point.y} r="3.4" fill="#dc2626" opacity="0.16" />
                  <circle cx={point.x} cy={point.y} r="1.85" fill="#ef4444" stroke="#7f1d1d" strokeWidth="0.55" />
                </g>
              ))}
          </svg>

          {!hasData ? (
            <div className="absolute inset-3 flex items-center justify-center rounded-lg bg-slate-950/75 px-4 text-center backdrop-blur-sm">
              <p className="max-w-xs text-sm font-semibold text-slate-300">
                Waiting for `/api/tactical` updates from the CV pipeline.
              </p>
            </div>
          ) : null}
        </div>

        <div className="grid content-start gap-3">
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
            <div className="text-xs font-bold uppercase tracking-wide text-slate-500">Source</div>
            <div className="mt-1 truncate text-lg font-black text-white">{sourceLabel}</div>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
            <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-500">
              <UsersRound className="h-3.5 w-3.5" />
              Inside
            </div>
            <div className="mt-1 text-3xl font-black text-red-400">{insideCount}</div>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
            <div className="text-xs font-bold uppercase tracking-wide text-slate-500">Outside visible</div>
            <div className="mt-1 text-3xl font-black text-cyan-300">{outsideVisibleCount}</div>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
            <div className="text-xs font-bold uppercase tracking-wide text-slate-500">Last update</div>
            <div className="mt-1 text-sm font-bold text-white">{formatAge(state?.age_seconds)}</div>
          </div>
        </div>
      </div>
    </section>
  );
}
