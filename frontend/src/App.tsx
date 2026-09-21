import { lazy, Suspense, useEffect, useState } from "react";
import { api } from "./api";
import { AppShell } from "./components/AppShell";
import { LoadingState } from "./components/EmptyState";
import type { AppView, HealthResponse, MetricsResponse } from "./types";

const OverviewPage = lazy(() =>
  import("./pages/OverviewPage").then((module) => ({
    default: module.OverviewPage,
  })),
);
const QueryPage = lazy(() =>
  import("./pages/QueryPage").then((module) => ({ default: module.QueryPage })),
);
const ExperimentsPage = lazy(() =>
  import("./pages/ExperimentsPage").then((module) => ({
    default: module.ExperimentsPage,
  })),
);

export function App() {
  const [view, setView] = useState<AppView>("overview");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [dataError, setDataError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    Promise.all([api.health(), api.metrics()])
      .then(([healthResponse, metricsResponse]) => {
        if (!active) return;
        setHealth(healthResponse);
        setMetrics(metricsResponse);
      })
      .catch((error: unknown) => {
        if (!active) return;
        setDataError(error instanceof Error ? error.message : "Não foi possível carregar os dados do laboratório.");
      });
    return () => { active = false; };
  }, []);

  return (
    <AppShell activeView={view} health={health} onNavigate={setView}>
      <Suspense fallback={<LoadingState label="Preparando a visualização…" />}>
        {view === "overview" && <OverviewPage health={health} metrics={metrics} error={dataError} onNavigate={setView} />}
        {view === "ask" && <QueryPage />}
        {view === "experiments" && <ExperimentsPage metrics={metrics} error={dataError} />}
      </Suspense>
    </AppShell>
  );
}
