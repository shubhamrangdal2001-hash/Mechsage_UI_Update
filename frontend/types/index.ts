// frontend/types/index.ts
// Canonical TypeScript types mirroring backend Pydantic schemas.

export interface RULResult {
  prediction_cycles: number;
  alert: boolean;
  severity: "NORMAL" | "CRITICAL" | "EMERGENCY";
  threshold_cycles: number;
}

export interface AnomalyResult {
  score: number;
  alert: boolean;
  severity: "NORMAL" | "ANOMALY DETECTED";
  threshold: number;
  model_type: string;
}

export interface CyclePayload {
  timestamp: string;
  machine_id: string;
  dataset_variant: string;
  cycle: number;
  rul: RULResult;
  anomaly: AnomalyResult;
  trigger_agent: boolean;
  agent_instruction: string;
  raw_features: Record<string, number>;
  warming_up?: boolean;
  warmup_progress?: number;
}

export interface FleetSnapshotResponse {
  engines: CyclePayload[];
  alert_count: number;
  emergency_count: number;
  snapshot_at: string;
}

export interface MetricsResponse {
  snapshot_at: string;
  simulation: {
    running: boolean;
    global_cycle: number;
    connections: number;
  };
  fleet: {
    total_engines: number;
    active_engines: number;
    alert_count: number;
    emergency_count: number;
    worst_rul_cycles: number | null;
  };
  engines: EngineSummary[];
}

export interface EngineSummary {
  machine_id: string;
  dataset_variant: string;
  cycle: number;
  rul_cycles: number;
  rul_severity: string;
  rul_alert: boolean;
  anomaly_score: number;
  anomaly_severity: string;
  anomaly_alert: boolean;
  trigger_agent: boolean;
  timestamp: string;
  status?: "warming_up";
  warmup_progress?: number;
}

// ── WebSocket messages ────────────────────────────────────────────────────────

export type WSMessageType =
  | "cycle_update"
  | "agent_alert"
  | "simulation_status"
  | "initial_snapshot";

export interface WSMessage {
  type: WSMessageType;
  payload: unknown;
  global_cycle?: number;
}

export interface AgentAlert {
  asset_id: string;
  severity: string;
  rul: number;
  anomaly_score: number;
  timestamp: string;
}

// ── Agent ─────────────────────────────────────────────────────────────────────

export interface AgentNodeTrace {
  node: string;
  status: "success" | "abstained" | "error";
  output: Record<string, unknown>;
  duration_ms?: number;
}

export interface AgentRunRequest {
  asset_id: string;
  dataset_id: string;
  cycle?: number;
  rul_estimate?: number;
  anomaly_score?: number;
  anomaly_flag?: boolean;
}

export interface AgentRunResponse {
  run_id: string;
  asset_id: string;
  started_at: string;
  completed_at: string;
  duration_ms: number;
  final_status: string;
  trace: AgentNodeTrace[];
  work_order_id?: number;
  work_order?: Record<string, unknown>;
  schedule_proposal?: string;
  approval_status: string;
  messages: string[];
}

// ── RAG ───────────────────────────────────────────────────────────────────────

export interface RAGQueryRequest {
  query: string;
  top_k?: number;
  dataset_context?: string;
}

export interface RAGSource {
  doc_id: string;
  text: string;
  relevance_score: number;
  source_file?: string;
  page?: number;
}

export interface RAGQueryResponse {
  query: string;
  answer: string;
  abstained: boolean;
  abstain_reason?: string;
  relevance_score: number;
  threshold_used: number;
  sources: RAGSource[];
  latency_ms?: number;
  corpus_version: string;
  retrieval_mode: string;
}

// ── Work Orders ───────────────────────────────────────────────────────────────

export type WorkOrderStatus =
  | "pending_approval"
  | "approved"
  | "rejected"
  | "in_progress"
  | "completed";

export type Priority = "low" | "medium" | "high" | "critical";
export type HITLActionType = "approve" | "reject";

export interface WorkOrder {
  id: number;
  asset_id: string;
  failure_mode: string;
  recommended_action: string;
  rul_estimate?: number;
  anomaly_score?: number;
  shap_top_features: string[];
  rag_context?: string;
  manual_refs: string[];
  parts: string[];
  priority: Priority;
  estimated_duration_hrs: number;
  diagnosis_confidence: number;
  auto_generated: boolean;
  status: WorkOrderStatus;
  rejection_reason?: string;
  proposed_start?: string;
  technician_id?: string;
  created_at: string;
  updated_at: string;
}

export interface WorkOrderListResponse {
  items: WorkOrder[];
  total: number;
  pending: number;
  approved: number;
  rejected: number;
}

export interface HITLActionRequest {
  action: HITLActionType;
  notes?: string;
  technician_id?: string;
  proposed_start?: string;
}
