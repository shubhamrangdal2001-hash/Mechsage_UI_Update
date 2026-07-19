// frontend/app/engines/[id]/page.tsx
"use client";

import React, { useState } from "react";
import Link from "next/link";
import { use } from "react";
import { useFleetStore } from "@/stores/fleet-store";
import { useEngineHistory, useAgentRun } from "@/hooks/useFleetData";
import { RULChart } from "@/components/organisms/RULChart";
import { AnomalyChart } from "@/components/organisms/AnomalyChart";
import { SHAPWaterfall } from "@/components/organisms/SHAPWaterfall";
import { Badge } from "@/components/atoms/Badge";
import { Spinner } from "@/components/atoms/Spinner";
import { SeverityPill } from "@/components/molecules/SeverityPill";
import {
  ArrowLeft,
  Activity,
  Cpu,
  ShieldAlert,
  Wrench,
  Sparkles,
  BookOpen,
  Gauge,
  GaugeCircle,
} from "lucide-react";
import toast from "react-hot-toast";

interface EngineDetailPageProps {
  params: Promise<{ id: string }>;
}

export default function EngineDetailPage({ params }: EngineDetailPageProps) {
  const { id } = use(params);
  const engineId = id.toUpperCase();
  
  // Get live data from WebSocket
  const { liveEngines } = useFleetStore();
  const livePayload = liveEngines[engineId];

  // Fetch prediction history
  const { data: history = [], isLoading: historyLoading } = useEngineHistory(engineId);

  // Agent run mutation
  const agentRunMutation = useAgentRun();
  const [isAgentRunning, setIsAgentRunning] = useState(false);

  // Determine active payload: live WebSocket payload or last item in history
  const activePayload = livePayload ?? (history.length > 0 ? history[history.length - 1] : null);

  const handleRunDiagnostics = async () => {
    if (!activePayload) return;
    setIsAgentRunning(true);
    toast.loading("Invoking LangGraph Agentic Pipeline...", { id: "agent-run" });

    agentRunMutation.mutate(
      {
        asset_id: activePayload.machine_id,
        dataset_id: activePayload.dataset_variant,
        cycle: activePayload.cycle,
        rul_estimate: activePayload.rul.prediction_cycles,
        anomaly_score: activePayload.anomaly.score,
        anomaly_flag: activePayload.anomaly.alert,
      },
      {
        onSuccess: (data) => {
          setIsAgentRunning(false);
          toast.success(
            `Agent execution completed: ${data.final_status}. Work Order created: #${data.work_order_id ?? "N/A"}`,
            { id: "agent-run", duration: 6000 }
          );
        },
        onError: (err: Error) => {
          setIsAgentRunning(false);
          toast.error(`Agent invocation failed: ${err.message}`, { id: "agent-run" });
        },
      }
    );
  };

  if (historyLoading && !activePayload) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-3">
        <Spinner size="lg" />
        <span className="text-sm text-text-muted">Loading telemetry history for {engineId}...</span>
      </div>
    );
  }

  if (!activePayload) {
    return (
      <div className="max-w-xl mx-auto text-center py-16 card space-y-6 mt-10">
        <ShieldAlert className="h-12 w-12 text-accent-amber mx-auto" />
        <div>
          <h3 className="text-lg font-semibold text-text-primary">Ironside Unit Not Found</h3>
          <p className="text-xs text-text-muted mt-1">
            Ironside unit {engineId} has no active stream or historical records in the database.
          </p>
        </div>
        <Link
          href="/fleet"
          className="btn btn-secondary inline-flex items-center gap-2 text-xs"
        >
          <ArrowLeft className="h-4 w-4" /> Back to Fleet overview
        </Link>
      </div>
    );
  }

  const sensors = activePayload.raw_features ?? {};

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Top Breadcrumb Nav */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-center gap-3">
          <Link
            href="/fleet"
            className="p-2 bg-white/5 hover:bg-white/10 border border-white/5 rounded-lg transition-all text-text-muted hover:text-text-primary"
          >
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-bold text-text-primary tracking-tight">
                {activePayload.machine_id}
              </h1>
              <Badge variant="blue">{activePayload.dataset_variant}</Badge>
            </div>
            <p className="text-[11px] text-text-muted mt-0.5">
              Diagnostic Deep-Dive — Cycle {activePayload.cycle}
            </p>
          </div>
        </div>

        {/* Manual Actions Panel */}
        <div className="flex items-center gap-3">
          <button
            onClick={handleRunDiagnostics}
            disabled={isAgentRunning}
            className="flex items-center gap-2 px-4 py-2 bg-accent-teal hover:bg-accent-teal/90 disabled:opacity-50 text-bg-base font-semibold text-xs rounded-lg transition-all cursor-pointer shadow-[0_0_12px_rgba(0,212,170,0.2)]"
          >
            {isAgentRunning ? (
              <Spinner size="sm" className="border-bg-base" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            Run Agent Diagnostics
          </button>
          
          <Link
            href="/rag"
            className="flex items-center gap-2 px-4 py-2 border border-white/10 bg-white/5 hover:bg-white/10 text-text-primary font-semibold text-xs rounded-lg transition-all"
          >
            <BookOpen className="h-4 w-4 text-accent-blue" />
            Search Tech Manuals
          </Link>
        </div>
      </div>

      {/* Hero Stats Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* RUL Card */}
        <div className="card p-6 flex flex-col justify-between h-36">
          <div className="flex items-center justify-between text-text-muted text-xs">
            <span>Estimated Remaining Useful Life</span>
            <Cpu className="h-4 w-4 text-accent-teal" />
          </div>
          <div>
            <span className="text-3xl font-extrabold font-mono text-text-primary">
              {activePayload.rul.prediction_cycles.toFixed(1)}
            </span>
            <span className="text-xs text-text-muted ml-2">cycles</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-text-muted">RUL Prognosis severity:</span>
            <SeverityPill severity={activePayload.rul.severity} />
          </div>
        </div>

        {/* Anomaly Card */}
        <div className="card p-6 flex flex-col justify-between h-36">
          <div className="flex items-center justify-between text-text-muted text-xs">
            <span>Isolation Forest Anomaly Score</span>
            <Activity className="h-4 w-4 text-accent-blue" />
          </div>
          <div>
            <span className="text-3xl font-extrabold font-mono text-text-primary">
              {activePayload.anomaly.score.toFixed(3)}
            </span>
            <span className="text-xs text-text-muted ml-2">/ 1.000</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-text-muted">Anomaly status:</span>
            <SeverityPill severity={activePayload.anomaly.severity} />
          </div>
        </div>

        {/* System Health Status */}
        <div className="card p-6 flex flex-col justify-between h-36">
          <div className="flex items-center justify-between text-text-muted text-xs">
            <span>Agent Trigger Alarm Flag</span>
            <ShieldAlert className="h-4 w-4 text-accent-amber" />
          </div>
          <div>
            <span className={`text-xl font-bold ${activePayload.trigger_agent ? "text-accent-red" : "text-accent-teal"}`}>
              {activePayload.trigger_agent ? "ALARM ACTIVE" : "NOMINAL STATE"}
            </span>
            <p className="text-[10px] text-text-muted mt-1 leading-relaxed truncate">
              {activePayload.agent_instruction}
            </p>
          </div>
          <div className="flex items-center justify-between text-[10px] text-text-muted">
            <span>Trigger threshold: RUL &lt; 30 cyc</span>
            {activePayload.trigger_agent && (
              <Link href="/agent" className="text-accent-amber hover:underline flex items-center gap-1 font-semibold">
                <Wrench className="h-3 w-3" /> View Agent Trace
              </Link>
            )}
          </div>
        </div>
      </div>

      {/* Main Charts & Telemetry Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Timeline Charts (Left columns) */}
        <div className="lg:col-span-2 space-y-6">
          <RULChart history={history} />
          <AnomalyChart history={history} />
        </div>

        {/* SHAP Waterfall Explainability (Right column) */}
        <div className="lg:col-span-1">
          <SHAPWaterfall payload={activePayload} />
        </div>
      </div>

      {/* Telemetry Sensor Matrix Table */}
      <div className="card p-6">
        <div className="flex items-center gap-2 border-b border-white/5 pb-4 mb-4">
          <Gauge className="h-5 w-5 text-accent-blue" />
          <h3 className="text-base font-semibold text-text-primary">Real-time Telemetry Sensor Matrix</h3>
        </div>

        {Object.keys(sensors).length === 0 ? (
          <div className="py-8 text-center text-xs text-text-muted">No telemetry values loaded.</div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {Object.entries(sensors).map(([sensorName, val]) => (
              <div
                key={sensorName}
                className="bg-white/[0.02] border border-white/5 hover:border-white/10 rounded-lg p-3 flex items-center justify-between transition-all"
              >
                <div className="flex flex-col">
                  <span className="text-[10px] text-text-muted font-medium uppercase">{sensorName}</span>
                  <span className="text-sm font-bold font-mono text-text-primary mt-0.5">
                    {val.toFixed(2)}
                  </span>
                </div>
                <GaugeCircle className="h-4 w-4 text-slate-600" />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
