import { MapPinned, RadioTower, UsersRound } from "lucide-react";

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatAge(seconds) {
  if (seconds === null || seconds === undefined) return "waiting";
  if (seconds < 1) return "live";
  return `${seconds.toFixed(1)}s ago`;
}

function normalizePoint(point, mapSize) {
  const x = Number(point?.x);
  const y = Number(point?.y);
  return {
    x: Number.isFinite(x) ? clamp(x, 0, mapSize) : 0,
    y: Number.isFinite(y) ? clamp(y, 0, mapSize) : 0,
  };
}

export default function TacticalMap({ state, cameraId, apiOnline }) {
  const hasData = Boolean(state?.has_data);
  const stale = !apiOnline || Boolean(state?.stale);
  const mapSize = Number(state?.map_size_cm) > 0 ? Number(state.map_size_cm) : 300;
  const positions = Array.isArray(state?.positions_cm) ? state.positions_cm : [];
  const peopleCount = Number.isFinite(Number(state?.people_count)) ? Number(state.people_count) : positions.length;
  const activeCamera = state?.camera_id || cameraId || "camera";
  const statusLabel = !apiOnline ? "Backend offline" : !hasData ? "Waiting for tactical data" : stale ? "Stale" : "Live";
  const gridMarks = [0.25, 0.5, 0.75].map((ratio) => Math.round(mapSize * ratio));

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-3">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-300">
            <MapPinned className="h-4 w-4 text-cyan-300" />
            Tactical Floor Map
          </div>
          <p className="mt-1 text-xs text-slate-500">Live X/Y foot-position dots from CV homography</p>
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
            viewBox={`0 0 ${mapSize} ${mapSize}`}
            role="img"
            aria-label={`Tactical map for ${activeCamera}`}
            className="aspect-square w-full"
          >
            <rect x="0" y="0" width={mapSize} height={mapSize} rx="12" fill="#f8fafc" />
            <rect
              x="8"
              y="8"
              width={mapSize - 16}
              height={mapSize - 16}
              rx="10"
              fill="#eef2ff"
              stroke="#334155"
              strokeWidth="2"
            />
            {gridMarks.map((mark) => (
              <g key={mark}>
                <line x1={mark} y1="8" x2={mark} y2={mapSize - 8} stroke="#cbd5e1" strokeWidth="1" />
                <line x1="8" y1={mark} x2={mapSize - 8} y2={mark} stroke="#cbd5e1" strokeWidth="1" />
              </g>
            ))}
            <path
              d={`M ${mapSize * 0.15} ${mapSize * 0.08} H ${mapSize * 0.85} L ${mapSize * 0.93} ${
                mapSize * 0.5
              } L ${mapSize * 0.85} ${mapSize * 0.92} H ${mapSize * 0.15} L ${mapSize * 0.07} ${
                mapSize * 0.5
              } Z`}
              fill="none"
              stroke="#0891b2"
              strokeDasharray="6 5"
              strokeWidth="2"
            />
            <text x="16" y="25" fill="#475569" fontSize="12" fontWeight="700">
              {mapSize}cm x {mapSize}cm monitored area
            </text>
            {positions.map((point, index) => {
              const normalized = normalizePoint(point, mapSize);
              return (
                <g key={`${normalized.x}-${normalized.y}-${index}`}>
                  <circle cx={normalized.x} cy={normalized.y} r="10" fill="#dc2626" opacity="0.16" />
                  <circle cx={normalized.x} cy={normalized.y} r="5.5" fill="#ef4444" stroke="#7f1d1d" strokeWidth="1.5" />
                  <text
                    x={normalized.x + 9}
                    y={normalized.y - 8}
                    fill="#7f1d1d"
                    fontSize="10"
                    fontWeight="800"
                  >
                    {index + 1}
                  </text>
                </g>
              );
            })}
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
            <div className="text-xs font-bold uppercase tracking-wide text-slate-500">Camera</div>
            <div className="mt-1 truncate text-lg font-black text-white">{activeCamera}</div>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
            <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-500">
              <UsersRound className="h-3.5 w-3.5" />
              Dots
            </div>
            <div className="mt-1 text-3xl font-black text-red-400">{peopleCount}</div>
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
