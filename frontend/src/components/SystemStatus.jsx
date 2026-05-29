import { Activity, Server, Video } from "lucide-react";
import StatusCard from "./StatusCard.jsx";

export default function SystemStatus({ apiOnline, cameras, status }) {
  const connected = Boolean(status?.camera_connected);
  const onlineCameraCount = cameras.filter((camera) => camera.camera_connected).length;
  const cameraCount = cameras.length;
  const resolution = status?.resolution;
  const resolutionText =
    resolution?.width && resolution?.height ? `${resolution.width} x ${resolution.height}` : "Unknown";

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/70 p-4">
      <div className="mb-3 flex items-center gap-2">
        <Activity className="h-4 w-4 text-cyan-300" />
        <h2 className="text-sm font-bold uppercase tracking-wide text-slate-200">System Status</h2>
      </div>
      <div className="grid gap-3">
        <StatusCard
          label="Backend"
          value={apiOnline ? "Online" : "Offline"}
          tone={apiOnline ? "good" : "critical"}
          detail="FastAPI service"
        />
        <StatusCard
          label="Selected Camera"
          value={connected ? "Connected" : "Disconnected"}
          tone={connected ? "good" : "warning"}
          detail={status?.last_error || status?.camera_id || "Live stream active"}
        />
        <StatusCard
          label="All Cameras"
          value={cameraCount ? `${onlineCameraCount}/${cameraCount} online` : "None"}
          tone={onlineCameraCount > 0 ? "good" : "warning"}
          detail={cameraCount ? cameras.map((camera) => camera.camera_id).join(", ") : "No cameras configured"}
        />
        <StatusCard label="Resolution" value={resolutionText} detail="Latest frame" />
      </div>
      <div className="mt-3 flex items-center gap-2 text-xs text-slate-500">
        <Server className="h-3.5 w-3.5" />
        <span>Polling every 3 seconds</span>
        <Video className="ml-auto h-3.5 w-3.5" />
      </div>
    </section>
  );
}
