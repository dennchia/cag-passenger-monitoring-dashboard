import { useEffect, useState } from "react";
import AssistanceView from "./components/AssistanceView.jsx";
import DashboardLayout from "./components/DashboardLayout.jsx";
import OperationsSidebarTabs from "./components/OperationsSidebarTabs.jsx";
import OperationsStatusPills from "./components/OperationsStatusPills.jsx";
import VideoPlayer from "./components/VideoPlayer.jsx";
import { endpoints, fetchJson } from "./lib/api.js";

const POLL_MS = 3000;

export default function App() {
  const [activeTab, setActiveTab] = useState("assistance");
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [apiOnline, setApiOnline] = useState(false);

  useEffect(() => {
    let isMounted = true;

    async function poll() {
      try {
        const [nextCameras, nextMetrics, nextAlerts] = await Promise.all([
          fetchJson(endpoints.cameras),
          fetchJson(endpoints.metrics),
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
          <OperationsStatusPills apiOnline={apiOnline} cameras={cameras} status={selectedCamera} />
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
