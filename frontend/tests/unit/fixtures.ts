import type { CyclePayload, WorkOrder } from "@/types";

export const cyclePayload: CyclePayload = {
  timestamp: "2026-07-19T00:00:00Z",
  machine_id: "ISM-CNC-001",
  dataset_variant: "FD001",
  cycle: 25,
  rul: { prediction_cycles: 42, alert: false, severity: "NORMAL", threshold_cycles: 30 },
  anomaly: { score: 0.21, alert: false, severity: "NORMAL", threshold: 0.54, model_type: "IsolationForest" },
  trigger_agent: false,
  agent_instruction: "Continue monitoring the Ironside unit.",
  raw_features: { sensor_11: 2.1 },
};

export const pendingWorkOrder: WorkOrder = {
  id: 1,
  asset_id: "ISM-CNC-001",
  failure_mode: "bearing wear",
  recommended_action: "Inspect spindle bearings.",
  shap_top_features: [],
  manual_refs: ["SOP-PM-001"],
  parts: [],
  priority: "high",
  estimated_duration_hrs: 4,
  diagnosis_confidence: 0.9,
  auto_generated: true,
  status: "pending_approval",
  created_at: "2026-07-19T00:00:00Z",
  updated_at: "2026-07-19T00:00:00Z",
};
