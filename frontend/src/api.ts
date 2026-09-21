import type { HealthResponse, MetricsResponse, QueryRequest, QueryResponse } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Falha na requisição (${response.status}).`);
  }

  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  metrics: () => request<MetricsResponse>("/api/metrics"),
  query: (payload: QueryRequest) =>
    request<QueryResponse>("/api/query", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
