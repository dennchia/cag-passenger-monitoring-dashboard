export default function StatusCard({ label, value, tone = "neutral", detail }) {
  const tones = {
    good: "border-emerald-500/30 bg-emerald-500/10 text-emerald-100",
    warning: "border-amber-500/30 bg-amber-500/10 text-amber-100",
    critical: "border-red-500/30 bg-red-500/10 text-red-100",
    neutral: "border-slate-700 bg-slate-800/70 text-slate-100",
  };

  return (
    <div className={`rounded-lg border p-3 ${tones[tone] || tones.neutral}`}>
      <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 text-lg font-bold leading-tight">{value}</div>
      {detail ? <div className="mt-1 truncate text-xs text-slate-400">{detail}</div> : null}
    </div>
  );
}
