// frontend/lib/synthetic-engine.ts
// ─────────────────────────────────────────────────────────────────────────────
// CMAPSS-faithful synthetic data generator.
// Runs entirely in the browser — no backend required.
// Produces realistic RUL degradation + Isolation Forest anomaly scores
// for all 4 engine variants (FD001–FD004).
// ─────────────────────────────────────────────────────────────────────────────

import type { CyclePayload, AgentAlert, WorkOrder, RAGQueryResponse, AgentRunResponse } from "@/types";

// ── Engine Profiles ───────────────────────────────────────────────────────────
// Each engine variant has its own life span, degradation rate, and anomaly profile.
export const ENGINE_CONFIGS: Record<string, {
  maxCycles: number;
  threshold: number;
  anomalyThreshold: number;
  degradationRate: number; // RUL reduction per cycle (approx)
}> = {
  FD001: { maxCycles: 362,  threshold: 30, anomalyThreshold: 0.52, degradationRate: 0.28 },
  FD002: { maxCycles: 378,  threshold: 30, anomalyThreshold: 0.55, degradationRate: 0.26 },
  FD003: { maxCycles: 525,  threshold: 30, anomalyThreshold: 0.50, degradationRate: 0.19 },
  FD004: { maxCycles: 543,  threshold: 30, anomalyThreshold: 0.48, degradationRate: 0.18 },
};

// ── Sensor baseline values (CMAPSS FD001 nominal) ─────────────────────────────
const SENSOR_BASELINES: Record<string, number> = {
  s2:  642.0,  s3: 1585.0, s4: 1400.0, s7:  554.0,
  s8: 2388.0,  s9: 9050.0, s11:  47.2, s12:  521.7,
  s13: 2388.0, s14: 8130.0, s15:  8.4, s17:  392.0,
  s20:  39.06, s21:  23.42,
};

// ── Internal engine state ─────────────────────────────────────────────────────
interface EngineState {
  cycle: number;
  rul: number;
  maxCycles: number;
}

const engineStates: Record<string, EngineState> = {
  FD001: { cycle: 1, rul: 362, maxCycles: 362 },
  FD002: { cycle: 1, rul: 378, maxCycles: 378 },
  FD003: { cycle: 1, rul: 525, maxCycles: 525 },
  FD004: { cycle: 1, rul: 543, maxCycles: 543 },
};

let globalCycle = 1;

// ── Noise helpers ─────────────────────────────────────────────────────────────
function gaussianNoise(std = 1): number {
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2) * std;
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

// ── Degraded sensor values ─────────────────────────────────────────────────────
// As cycle progresses, sensors shift away from nominal, modelling wear.
function generateSensorValues(dataset: string, cycle: number): Record<string, number> {
  const cfg = ENGINE_CONFIGS[dataset];
  const degradationFraction = Math.min(cycle / cfg.maxCycles, 1.0);

  return Object.fromEntries(
    Object.entries(SENSOR_BASELINES).map(([key, base]) => {
      // Degrade sensors that typically change under wear
      const degradeFactor = ["s3", "s4", "s11", "s15"].includes(key) ? 1 : -1;
      const drift = degradationFraction * degradeFactor * base * 0.05;
      const noise = gaussianNoise(base * 0.002);
      return [key, parseFloat((base + drift + noise).toFixed(3))];
    })
  );
}

// ── Anomaly Score ─────────────────────────────────────────────────────────────
// Rises with degradation, plus occasional spikes.
function computeAnomalyScore(dataset: string, cycle: number): number {
  const cfg = ENGINE_CONFIGS[dataset];
  const degradationFraction = Math.min(cycle / cfg.maxCycles, 1.0);
  const baseScore = degradationFraction * 0.85;
  const spike = Math.random() < 0.05 ? Math.random() * 0.2 : 0; // 5% chance of spike
  const noise = gaussianNoise(0.02);
  return clamp(baseScore + spike + noise, 0.01, 0.99);
}

// ── RUL Prediction ────────────────────────────────────────────────────────────
// True CMAPSS RUL is capped at 130 (piece-wise linear degradation model).
function computeRUL(dataset: string, cycle: number): number {
  const cfg = ENGINE_CONFIGS[dataset];
  const trueRUL = Math.max(cfg.maxCycles - cycle, 0);
  // Clip at 130 (piece-wise linear)
  const clipped = Math.min(trueRUL, 130);
  // Add small prediction noise to simulate ML model error
  const noise = gaussianNoise(2.5);
  return clamp(clipped + noise, 0, 135);
}

// ── Single cycle payload ──────────────────────────────────────────────────────
export function generateCyclePayload(dataset: string): CyclePayload {
  const state = engineStates[dataset];
  const cfg = ENGINE_CONFIGS[dataset];

  const rul = computeRUL(dataset, state.cycle);
  const anomalyScore = computeAnomalyScore(dataset, state.cycle);
  const sensors = generateSensorValues(dataset, state.cycle);

  const rulAlert = rul < cfg.threshold;
  const anomalyAlert = anomalyScore >= cfg.anomalyThreshold;
  const triggerAgent = rulAlert || anomalyAlert;

  let rulSeverity: CyclePayload["rul"]["severity"] = "NORMAL";
  if (rul < 15) rulSeverity = "EMERGENCY";
  else if (rul < cfg.threshold) rulSeverity = "CRITICAL";

  const anomalySeverity = anomalyAlert ? "ANOMALY DETECTED" : "NORMAL";

  const agentInstruction = triggerAgent
    ? rulSeverity === "EMERGENCY"
      ? "IMMEDIATE SHUTDOWN REQUIRED — schedule emergency maintenance."
      : "Degradation threshold exceeded — diagnostic agent queued."
    : "System nominal. Continue monitoring.";

  // Advance engine state
  state.cycle += 1;
  if (state.cycle > state.maxCycles) {
    // Reset engine (simulate new unit being deployed)
    state.cycle = 1;
  }

  return {
    timestamp: new Date().toISOString(),
    machine_id: `SIM-${dataset}`,
    dataset_variant: dataset,
    cycle: state.cycle - 1,
    rul: {
      prediction_cycles: rul,
      alert: rulAlert,
      severity: rulSeverity,
      threshold_cycles: cfg.threshold,
    },
    anomaly: {
      score: anomalyScore,
      alert: anomalyAlert,
      severity: anomalySeverity,
      threshold: cfg.anomalyThreshold,
      model_type: dataset === "FD002" || dataset === "FD004" ? "LightGBM" : "IsolationForest",
    },
    trigger_agent: triggerAgent,
    agent_instruction: agentInstruction,
    raw_features: sensors,
  };
}

// ── Fleet snapshot (all 4 engines) ────────────────────────────────────────────
export function generateFleetSnapshot() {
  globalCycle += 1;
  const engines = Object.keys(engineStates).map((ds) => generateCyclePayload(ds));
  return { engines, globalCycle };
}

// ── Generate agent alert when threshold is breached ───────────────────────────
export function generateAgentAlert(payload: CyclePayload): AgentAlert {
  return {
    asset_id: payload.machine_id,
    severity: payload.rul.severity,
    rul: payload.rul.prediction_cycles,
    anomaly_score: payload.anomaly.score,
    timestamp: new Date().toISOString(),
  };
}

// ── Mock Metrics response ─────────────────────────────────────────────────────
export function generateMetrics() {
  const engines = Object.keys(engineStates).map((ds) => {
    const state = engineStates[ds];
    const cfg = ENGINE_CONFIGS[ds];
    const rul = computeRUL(ds, state.cycle);
    const anomalyScore = computeAnomalyScore(ds, state.cycle);
    return {
      machine_id: `SIM-${ds}`,
      dataset_variant: ds,
      cycle: state.cycle,
      rul_cycles: rul,
      rul_severity: rul < 15 ? "EMERGENCY" : rul < 30 ? "CRITICAL" : "NORMAL",
      rul_alert: rul < cfg.threshold,
      anomaly_score: anomalyScore,
      anomaly_severity: anomalyScore >= cfg.anomalyThreshold ? "ANOMALY DETECTED" : "NORMAL",
      anomaly_alert: anomalyScore >= cfg.anomalyThreshold,
      trigger_agent: rul < cfg.threshold || anomalyScore >= cfg.anomalyThreshold,
      timestamp: new Date().toISOString(),
    };
  });

  const alertCount = engines.filter((e) => e.rul_alert).length;
  const emergencyCount = engines.filter((e) => e.rul_severity === "EMERGENCY").length;

  return {
    snapshot_at: new Date().toISOString(),
    simulation: { running: true, global_cycle: globalCycle, connections: 1 },
    fleet: {
      total_engines: 4,
      active_engines: 4,
      alert_count: alertCount,
      emergency_count: emergencyCount,
      worst_rul_cycles: Math.min(...engines.map((e) => e.rul_cycles)),
    },
    engines,
  };
}

// ── Mock history (last N cycles) ─────────────────────────────────────────────
export function generateHistory(dataset: string, n = 80): CyclePayload[] {
  const cfg = ENGINE_CONFIGS[dataset];
  const currentCycle = engineStates[dataset].cycle;
  const startCycle = Math.max(1, currentCycle - n);

  return Array.from({ length: currentCycle - startCycle }, (_, i) => {
    const c = startCycle + i;
    const rul = clamp(Math.min(cfg.maxCycles - c, 130) + gaussianNoise(2), 0, 135);
    const anomalyScore = clamp((c / cfg.maxCycles) * 0.85 + gaussianNoise(0.02), 0.01, 0.99);
    const sensors = generateSensorValues(dataset, c);
    return {
      timestamp: new Date(Date.now() - (currentCycle - c) * 500).toISOString(),
      machine_id: `SIM-${dataset}`,
      dataset_variant: dataset,
      cycle: c,
      rul: {
        prediction_cycles: rul,
        alert: rul < cfg.threshold,
        severity: rul < 15 ? "EMERGENCY" : rul < 30 ? "CRITICAL" : "NORMAL",
        threshold_cycles: cfg.threshold,
      },
      anomaly: {
        score: anomalyScore,
        alert: anomalyScore >= cfg.anomalyThreshold,
        severity: anomalyScore >= cfg.anomalyThreshold ? "ANOMALY DETECTED" : "NORMAL",
        threshold: cfg.anomalyThreshold,
        model_type: "IsolationForest",
      },
      trigger_agent: rul < cfg.threshold || anomalyScore >= cfg.anomalyThreshold,
      agent_instruction: rul < cfg.threshold ? "Diagnostic agent queued." : "Nominal.",
      raw_features: sensors,
    } as CyclePayload;
  });
}

// ── Mock Work Orders ──────────────────────────────────────────────────────────
let woIdCounter = 1;
const mockWorkOrders: WorkOrder[] = [];

export function createMockWorkOrder(payload: CyclePayload): WorkOrder {
  const wo: WorkOrder = {
    id: woIdCounter++,
    asset_id: payload.machine_id,
    failure_mode: payload.rul.severity === "EMERGENCY"
      ? "High-pressure turbine thermal degradation — critical bearing wear"
      : "LPT coolant flow anomaly — progressive seal erosion",
    recommended_action: payload.rul.severity === "EMERGENCY"
      ? "IMMEDIATE: Replace HPT blade set and turbine seals. Inspect combustion liner."
      : "Schedule borescope inspection. Replace HPT/LPT seals within 2 maintenance windows.",
    rul_estimate: payload.rul.prediction_cycles,
    anomaly_score: payload.anomaly.score,
    shap_top_features: ["s11", "s4", "s15", "s3"],
    rag_context: "Per Section 4.3 of Ironside SOS: HPT coolant flow reduction >8% requires immediate borescope inspection and potential blade replacement. Reference FMEA Case 11-B.",
    manual_refs: ["SOS Section 4.3", "FMEA Case 11-B"],
    parts: ["HPT Blade Seal Kit", "Turbine Nozzle Guide Vane", "Cooling Duct Assembly"],
    priority: payload.rul.severity === "EMERGENCY" ? "critical" : payload.rul.severity === "CRITICAL" ? "high" : "medium",
    estimated_duration_hrs: payload.rul.severity === "EMERGENCY" ? 16 : 6.5,
    diagnosis_confidence: 0.89,
    auto_generated: true,
    status: "pending_approval",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
  mockWorkOrders.push(wo);
  return wo;
}

export function getMockWorkOrders(): WorkOrder[] {
  return [...mockWorkOrders].reverse();
}

export function getMockWorkOrder(id: number): WorkOrder | undefined {
  return mockWorkOrders.find((w) => w.id === id);
}

export function hitlAction(id: number, action: "approve" | "reject", notes?: string, technicianId?: string, proposedStart?: string): WorkOrder | null {
  const wo = mockWorkOrders.find((w) => w.id === id);
  if (!wo) return null;
  wo.status = action === "approve" ? "approved" : "rejected";
  wo.rejection_reason = action === "reject" ? notes : undefined;
  wo.technician_id = technicianId;
  wo.proposed_start = proposedStart;
  wo.updated_at = new Date().toISOString();
  return wo;
}

// ── Mock RAG response ─────────────────────────────────────────────────────────
const RAG_KNOWLEDGE_BASE: Record<string, RAGQueryResponse> = {
  default: {
    query: "",
    answer: "Based on Ironside SOS maintenance manual, Section 4.3: When HPT coolant flow (Sensor 11) shows a reduction greater than 8% from nominal baseline (47.2 units), immediate borescope inspection is required. Progressive seal erosion is the primary failure mode. Recommended corrective action: Replace HPT/LPT seals within the next maintenance window and perform a full blade inspection.",
    abstained: false,
    relevance_score: 0.88,
    threshold_used: 0.45,
    sources: [
      { doc_id: "ironside-sos-section-4-3", text: "HPT coolant flow reduction >8% requires immediate borescope inspection and potential blade seal replacement. Cross-reference with FMEA Case 11-B.", relevance_score: 0.91, source_file: "Ironside_SOS_Manual_v3.2.pdf", page: 47 },
      { doc_id: "fmea-case-11b", text: "FMEA Case 11-B: Sensor 11 anomaly correlated with turbine stage 2 degradation. Failure probability: 78% within 30 operating cycles if unaddressed.", relevance_score: 0.85, source_file: "FMEA_Turbofan_Engine.pdf", page: 103 },
      { doc_id: "cmapss-fd001-rul-guide", text: "For FD001 engines, RUL below 30 cycles indicates advanced degradation stage. Standard maintenance window is insufficient; schedule emergency slot.", relevance_score: 0.80, source_file: "NASA_CMAPSS_Maintenance_Guide.pdf", page: 12 },
    ],
    corpus_version: "synthetic-fixture-only",
    retrieval_mode: "synthetic-fixture-only",
  },
  abstained: {
    query: "",
    answer: "",
    abstained: true,
    abstain_reason: "Query relevance score (0.12) below safety threshold (0.45). No reliable maintenance procedure found for this query.",
    relevance_score: 0.12,
    threshold_used: 0.45,
    sources: [],
    corpus_version: "synthetic-fixture-only",
    retrieval_mode: "synthetic-fixture-only",
  }
};

export function mockRAGQuery(query: string): RAGQueryResponse {
  const offDomainKeywords = ["weather", "stock", "recipe", "football", "movie"];
  const isOffDomain = offDomainKeywords.some(k => query.toLowerCase().includes(k));

  const resp = isOffDomain ? { ...RAG_KNOWLEDGE_BASE.abstained } : { ...RAG_KNOWLEDGE_BASE.default };
  resp.query = query;
  resp.latency_ms = 280 + Math.round(Math.random() * 400);
  return resp;
}

// ── Mock Agent Run ────────────────────────────────────────────────────────────
export function mockAgentRun(payload: { asset_id: string; dataset_id: string; cycle?: number; rul_estimate?: number; anomaly_score?: number; anomaly_flag?: boolean }): AgentRunResponse {
  const wo = createMockWorkOrder({
    machine_id: payload.asset_id,
    dataset_variant: payload.dataset_id,
    cycle: payload.cycle ?? 145,
    rul: {
      prediction_cycles: payload.rul_estimate ?? 12.5,
      alert: true,
      severity: (payload.rul_estimate ?? 12.5) < 15 ? "EMERGENCY" : "CRITICAL",
      threshold_cycles: 30,
    },
    anomaly: {
      score: payload.anomaly_score ?? 0.68,
      alert: payload.anomaly_flag ?? true,
      severity: "ANOMALY DETECTED",
      threshold: 0.52,
      model_type: "IsolationForest",
    },
    trigger_agent: true,
    agent_instruction: "Agent triggered by RUL threshold breach.",
    raw_features: generateSensorValues(payload.dataset_id, payload.cycle ?? 145),
    timestamp: new Date().toISOString(),
  });

  const startedAt = new Date();
  const completedAt = new Date(startedAt.getTime() + 1850);

  return {
    run_id: `run-${Date.now()}`,
    asset_id: payload.asset_id,
    started_at: startedAt.toISOString(),
    completed_at: completedAt.toISOString(),
    duration_ms: 1850,
    final_status: "work_order_drafted",
    trace: [
      {
        node: "supervisor",
        status: "success",
        output: { asset_id: payload.asset_id, dataset: payload.dataset_id, cycle: payload.cycle, routing_decision: "dispatch_to_monitor" },
        duration_ms: 120,
      },
      {
        node: "monitor",
        status: "success",
        output: { rul_estimate: payload.rul_estimate, anomaly_score: payload.anomaly_score, anomaly_flag: payload.anomaly_flag, threshold_breached: true, severity: (payload.rul_estimate ?? 12.5) < 15 ? "EMERGENCY" : "CRITICAL" },
        duration_ms: 85,
      },
      {
        node: "diagnostics",
        status: "success",
        output: { diagnosis: "HPT coolant flow degradation — Stage 2 blade seal erosion", confidence: 0.89, failure_mode: "Thermal wear", top_sensors: ["s11", "s4", "s15"], rag_context_retrieved: true },
        duration_ms: 620,
      },
      {
        node: "work_order",
        status: "success",
        output: { work_order_id: wo.id, priority: wo.priority, estimated_duration_hrs: wo.estimated_duration_hrs, parts_required: wo.parts, auto_drafted: true },
        duration_ms: 780,
      },
      {
        node: "scheduling",
        status: "success",
        output: { schedule_proposal: "Priority window: within 48h. Recommended slot: Next available 16-hour maintenance block. Technician clearance required.", maintenance_slot: "2026-07-21T06:00:00Z" },
        duration_ms: 245,
      },
    ],
    work_order_id: wo.id,
    work_order: { id: wo.id, priority: wo.priority, status: "pending_approval" },
    schedule_proposal: "Priority window: within 48h. Next available 16-hour maintenance block.",
    approval_status: "pending_approval",
    messages: ["Agent pipeline completed successfully.", `Work order #${wo.id} drafted and pending HITL review.`],
  };
}
