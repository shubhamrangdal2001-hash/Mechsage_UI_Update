// frontend/stores/fleet-store.ts
// Zustand store for client-side UI state.
// Only pure UI state lives here — server data goes through TanStack Query.

import { create } from "zustand";
import type { AgentAlert, CyclePayload } from "@/types";

interface FleetState {
  // Selected engine
  selectedDataset: string;
  setSelectedDataset: (ds: string) => void;

  // WebSocket connection state
  wsConnected: boolean;
  setWsConnected: (v: boolean) => void;

  // Simulation control
  simulationPaused: boolean;
  setSimulationPaused: (v: boolean) => void;

  // Real-time engine data (latest payload per engine)
  liveEngines: Record<string, CyclePayload>;
  updateLiveEngine: (payload: CyclePayload) => void;
  updateLiveEngines: (payloads: CyclePayload[]) => void;

  // Alert feed (last 50)
  alerts: AgentAlert[];
  addAlert: (alert: AgentAlert) => void;
  clearAlerts: () => void;

  // Global simulation cycle counter
  globalCycle: number;
  setGlobalCycle: (n: number) => void;
}

export const useFleetStore = create<FleetState>((set) => ({
  selectedDataset: "FD001",
  setSelectedDataset: (ds) => set({ selectedDataset: ds }),

  wsConnected: false,
  setWsConnected: (v) => set({ wsConnected: v }),

  simulationPaused: false,
  setSimulationPaused: (v) => set({ simulationPaused: v }),

  liveEngines: {},
  updateLiveEngine: (payload) =>
    set((s) => {
      const current = s.liveEngines[payload.dataset_variant];
      if (current && payload.cycle < current.cycle) return s;
      return {
        liveEngines: {
          ...s.liveEngines,
          [payload.dataset_variant]: payload,
        },
      };
    }),
  updateLiveEngines: (payloads) =>
    set((s) => {
      const updated = { ...s.liveEngines };
      for (const p of payloads) {
        const current = updated[p.dataset_variant];
        if (!p.warming_up && (!current || p.cycle >= current.cycle)) {
          updated[p.dataset_variant] = p;
        }
      }
      return { liveEngines: updated };
    }),

  alerts: [],
  addAlert: (alert) =>
    set((s) => {
      const duplicate = s.alerts.some(
        (item) => item.asset_id === alert.asset_id && item.timestamp === alert.timestamp
      );
      return duplicate ? s : { alerts: [alert, ...s.alerts].slice(0, 50) };
    }),
  clearAlerts: () => set({ alerts: [] }),

  globalCycle: 0,
  setGlobalCycle: (n) => set({ globalCycle: n }),
}));
