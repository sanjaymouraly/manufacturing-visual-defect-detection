import { lazy, Suspense, type ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { Spinner } from "./components/ui";
import { useAuth } from "./lib/auth";
import type { Role } from "./lib/types";
import Login from "./pages/Login";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const Inspect = lazy(() => import("./pages/Inspect"));
const InspectionDetail = lazy(() => import("./pages/InspectionDetail"));
const Batch = lazy(() => import("./pages/Batch"));
const CameraPage = lazy(() => import("./pages/Camera"));
const ReviewQueue = lazy(() => import("./pages/ReviewQueue"));
const ReviewDetail = lazy(() => import("./pages/ReviewDetail"));
const HistoryPage = lazy(() => import("./pages/History"));
const Analytics = lazy(() => import("./pages/Analytics"));
const ModelLab = lazy(() => import("./pages/ModelLab"));
const ModelDetail = lazy(() => import("./pages/ModelDetail"));
const Robustness = lazy(() => import("./pages/Robustness"));
const Datasets = lazy(() => import("./pages/Datasets"));
const Jobs = lazy(() => import("./pages/Jobs"));
const SettingsPage = lazy(() => import("./pages/Settings"));

function RequireRole({ roles, children }: { roles: Role[]; children: ReactNode }) {
  const { can, user } = useAuth();
  return can(...roles) ? <>{children}</> : (
    <div className="card p-6 text-sm" role="status">
      <p className="font-semibold">Your role ({user?.role}) does not have access to this page.</p>
      <p className="muted mt-1">It requires: {roles.join(" or ")}. Ask an administrator if you need access.</p>
    </div>
  );
}

export default function App() {
  const { user, loading } = useAuth();
  if (loading) return <Spinner label="Starting DefectLens" />;
  if (!user) return <Login />;
  const ops: Role[] = ["admin", "inspector"];
  return (
    <Suspense fallback={<Spinner />}>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="inspect" element={<RequireRole roles={ops}><Inspect /></RequireRole>} />
          <Route path="inspections/:id" element={<InspectionDetail />} />
          <Route path="batch" element={<RequireRole roles={ops}><Batch /></RequireRole>} />
          <Route path="camera" element={<RequireRole roles={ops}><CameraPage /></RequireRole>} />
          <Route path="reviews" element={<ReviewQueue />} />
          <Route path="reviews/:id" element={<ReviewDetail />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="models" element={<ModelLab />} />
          <Route path="models/:id" element={<ModelDetail />} />
          <Route path="robustness" element={<Robustness />} />
          <Route path="datasets" element={<Datasets />} />
          <Route path="jobs" element={<Jobs />} />
          <Route path="settings" element={<RequireRole roles={["admin"]}><SettingsPage /></RequireRole>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
