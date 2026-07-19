import { beforeEach, describe, expect, it } from "vitest";
import { useFleetStore } from "@/stores/fleet-store";
import { cyclePayload } from "./fixtures";

describe("fleet store", () => {
  beforeEach(() => {
    useFleetStore.setState({ liveEngines: {}, alerts: [], globalCycle: 0, wsConnected: false });
  });

  it("updates live telemetry and ignores stale reconnection payloads", () => {
    useFleetStore.getState().updateLiveEngine(cyclePayload);
    useFleetStore.getState().updateLiveEngine({ ...cyclePayload, cycle: 24, rul: { ...cyclePayload.rul, prediction_cycles: 999 } });
    expect(useFleetStore.getState().liveEngines.FD001.cycle).toBe(25);
    expect(useFleetStore.getState().liveEngines.FD001.rul.prediction_cycles).toBe(42);
  });

  it("ignores warm-up rows and accepts newer rows", () => {
    useFleetStore.getState().updateLiveEngines([{ ...cyclePayload, warming_up: true }]);
    expect(useFleetStore.getState().liveEngines).toEqual({});
    useFleetStore.getState().updateLiveEngines([cyclePayload, { ...cyclePayload, cycle: 26 }]);
    expect(useFleetStore.getState().liveEngines.FD001.cycle).toBe(26);
  });

  it("deduplicates repeated transition alerts", () => {
    const alert = { asset_id: "ISM-CNC-001", severity: "CRITICAL", rul: 12, anomaly_score: 0.8, timestamp: "2026-07-19T00:00:00Z" };
    useFleetStore.getState().addAlert(alert);
    useFleetStore.getState().addAlert(alert);
    expect(useFleetStore.getState().alerts).toHaveLength(1);
  });
});
