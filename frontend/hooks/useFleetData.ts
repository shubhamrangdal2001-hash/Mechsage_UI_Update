"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import type { AgentRunRequest, HITLActionRequest, RAGQueryRequest } from "@/types";

export function useMetrics() {
  return useQuery({ queryKey: ["metrics"], queryFn: api.getMetrics, refetchInterval: 2000, staleTime: 1500 });
}

export function useEngineHistory(datasetId: string) {
  return useQuery({
    queryKey: ["history", datasetId],
    queryFn: () => api.getHistory(datasetId),
    refetchInterval: 1500,
    enabled: !!datasetId,
  });
}

export function useWorkOrders(params?: { status?: string; asset_id?: string }) {
  return useQuery({
    queryKey: ["workorders", params],
    queryFn: () => api.listWorkOrders(params),
    refetchInterval: 3000,
    staleTime: 1000,
  });
}

export function useWorkOrder(id: number | null) {
  return useQuery({
    queryKey: ["workorder", id],
    queryFn: () => api.getWorkOrder(id!),
    enabled: id != null,
    refetchInterval: 2000,
  });
}

export function useHITLAction() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, payload }: { id: number; payload: HITLActionRequest }) => api.hitlAction(id, payload),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["workorders"] });
      qc.invalidateQueries({ queryKey: ["workorder"] });
    },
  });
}

export function useAgentRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: AgentRunRequest) => api.runAgent(payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["workorders"] }),
  });
}

export function useRAGQuery() {
  return useMutation({ mutationFn: (payload: RAGQueryRequest) => api.queryRAG(payload) });
}

export function useSimulationControl() {
  return { pause: api.pauseSimulation, resume: api.resumeSimulation };
}
