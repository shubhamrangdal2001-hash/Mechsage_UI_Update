// frontend/lib/api-client.ts
// Typed fetch wrapper for all MechSage backend API calls.

import type {
  AgentRunRequest,
  AgentRunResponse,
  CyclePayload,
  FleetSnapshotResponse,
  HITLActionRequest,
  MetricsResponse,
  RAGQueryRequest,
  RAGQueryResponse,
  WorkOrder,
  WorkOrderListResponse,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? "dev-secret-key";

export const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/fleet";

async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
      ...(options.headers ?? {}),
    },
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail ?? `API error ${res.status}`);
  }

  return res.json() as Promise<T>;
}

// ── Inference ────────────────────────────────────────────────────────────────

export const api = {
  getFleetSnapshot: () =>
    apiFetch<FleetSnapshotResponse>("/api/v1/infer/fleet/snapshot"),

  getMetrics: () => apiFetch<MetricsResponse>("/api/v1/metrics"),

  pauseSimulation: () =>
    apiFetch<{ status: string }>("/api/v1/infer/simulation/pause", { method: "POST" }),

  resumeSimulation: () =>
    apiFetch<{ status: string }>("/api/v1/infer/simulation/resume", { method: "POST" }),

  getHistory: (datasetId: string) =>
    apiFetch<CyclePayload[]>(`/api/v1/infer/history/${datasetId}`),

  // ── Agent ────────────────────────────────────────────────────────────────
  runAgent: (payload: AgentRunRequest) =>
    apiFetch<AgentRunResponse>("/api/v1/agent/run", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // ── RAG ──────────────────────────────────────────────────────────────────
  queryRAG: (payload: RAGQueryRequest) =>
    apiFetch<RAGQueryResponse>("/api/v1/rag/query", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // ── Work Orders ───────────────────────────────────────────────────────────
  listWorkOrders: (params?: { status?: string; asset_id?: string }) => {
    const qs = new URLSearchParams();
    if (params?.status) qs.set("status", params.status);
    if (params?.asset_id) qs.set("asset_id", params.asset_id);
    return apiFetch<WorkOrderListResponse>(`/api/v1/workorders?${qs}`);
  },

  getWorkOrder: (id: number) =>
    apiFetch<WorkOrder>(`/api/v1/workorders/${id}`),

  hitlAction: (id: number, payload: HITLActionRequest) =>
    apiFetch<WorkOrder>(`/api/v1/workorders/${id}/action`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
};

// ── Type re-exports (avoid circular imports in components) ────────────────────
export type {
  FleetSnapshotResponse,
  MetricsResponse,
  AgentRunRequest,
  AgentRunResponse,
  RAGQueryRequest,
  RAGQueryResponse,
  WorkOrder,
  WorkOrderListResponse,
  HITLActionRequest,
} from "@/types";
