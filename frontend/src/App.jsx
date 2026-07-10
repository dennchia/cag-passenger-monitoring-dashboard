import { useEffect, useState } from "react";
import AssistanceView from "./components/AssistanceView.jsx";
import DashboardLayout from "./components/DashboardLayout.jsx";
import MetricTrendSparkline from "./components/MetricTrendSparkline.jsx";
import OperationsSidebarTabs from "./components/OperationsSidebarTabs.jsx";
import OperationsStatusPills from "./components/OperationsStatusPills.jsx";
import TacticalMap from "./components/TacticalMap.jsx";
import VideoPlayer from "./components/VideoPlayer.jsx";
import ZoneCapacityBars from "./components/ZoneCapacityBars.jsx";
import ExportShiftReportButton from "./features/reports/ExportShiftReportButton.jsx";
import { endpoints, fetchJson } from "./lib/api.js";

const POLL_MS = 3000;
const TACTICAL_POLL_MS = 1000;

export default function App() {
  const [activeTab, setActiveTab] = useState("assistance");
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [metricTrend, setMetricTrend] = useState([]);
  const [tacticalState, setTacticalState] = useState(null);
  const [zoneStatus, setZoneStatus] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [apiOnline, setApiOnline] = useState(false);

  useEffect(() => {
    let isMounted = true;

    async function poll() {
      try {
        const [nextCameras, nextMetrics, nextMetricTrend, nextZoneStatus, nextAlerts] = await Promise.all([
          fetchJson(endpoints.cameras),
          fetchJson(endpoints.metrics),
          fetchJson(endpoints.metricTrends),
          fetchJson(endpoints.zoneStatus),
          fetchJson(endpoints.alerts),
        ]);

        if (!isMounted) return;
        const normalizedCameras = Array.isArray(nextCameras) ? nextCameras : [];
        setCameras(normalizedCameras);
        setSelectedCameraId((current) => {
          if (current && normalizedCameras.some((camera) => camera.camera_id === current)) {
            return current;
          }
          return normalizedCameras[0]?.camera_id || current;
        });
        setMetrics(Array.isArray(nextMetrics) ? nextMetrics : []);
        setMetricTrend(Array.isArray(nextMetricTrend) ? nextMetricTrend : []);
        setZoneStatus(Array.isArray(nextZoneStatus) ? nextZoneStatus : []);
        setAlerts(Array.isArray(nextAlerts) ? nextAlerts : []);
        setApiOnline(true);
      } catch (error) {
        if (!isMounted) return;
        setApiOnline(false);
        setCameras((current) =>
          current.map((camera) => ({
            ...camera,
            camera_connected: false,
            last_error: error instanceof Error ? error.message : "Backend unavailable",
          })),
        );
        setMetricTrend([]);
        setZoneStatus([]);
      }
    }

    poll();
    const interval = window.setInterval(poll, POLL_MS);
    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, []);

  const selectedCamera =
    cameras.find((camera) => camera.camera_id === selectedCameraId) || cameras[0] || null;
  const currentRunId = metrics[0]?.run_id || "";

  useEffect(() => {
    if (activeTab !== "operations") {
      return undefined;
    }

    let isMounted = true;
    setTacticalState(null);

    async function pollTactical() {
      try {
        const nextTacticalState = await fetchJson(endpoints.tacticalLatestGlobal(currentRunId));
        if (!isMounted) return;
        setTacticalState(nextTacticalState);
      } catch (error) {
        if (!isMounted) return;
        setTacticalState((current) => ({
          ...(current || {}),
          camera_id: current?.camera_id || "fused",
          has_data: Boolean(current?.has_data),
          stale: true,
        }));
      }
    }

    pollTactical();
    const interval = window.setInterval(pollTactical, TACTICAL_POLL_MS);
    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, [activeTab, currentRunId]);

  return (
    <DashboardLayout
      activeTab={activeTab}
      onTabChange={setActiveTab}
      cameras={cameras}
      status={selectedCamera}
      sidebar={
        activeTab === "operations" ? (
          <OperationsSidebarTabs metrics={metrics} alerts={alerts} />
        ) : null
      }
    >
      {activeTab === "operations" ? (
        <section className="grid gap-4">
          <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-stretch">
            <OperationsStatusPills apiOnline={apiOnline} cameras={cameras} status={selectedCamera} />
            <ExportShiftReportButton runId={currentRunId} />
          </div>
          <div className="grid gap-4 2xl:grid-cols-[minmax(420px,0.95fr)_minmax(0,1.05fr)]">
            <TacticalMap state={tacticalState} cameraId="fused" apiOnline={apiOnline} />
            <div className="grid content-start gap-4">
              <ZoneCapacityBars zones={zoneStatus} />
              <MetricTrendSparkline points={metricTrend} />
            </div>
          </div>
          <VideoPlayer
            apiOnline={apiOnline}
            cameras={cameras}
            camera={selectedCamera}
            selectedCameraId={selectedCamera?.camera_id || selectedCameraId}
            onSelectCamera={setSelectedCameraId}
          />
        </section>
      ) : (
        <AssistanceView cameras={cameras} />
      )}
    </DashboardLayout>
  );
}
