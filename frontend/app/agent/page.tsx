// frontend/app/agent/page.tsx
"use client";

import React, { useState } from "react";
import { useAgentRun } from "@/hooks/useFleetData";
import { AgentTraceViewer } from "@/components/organisms/AgentTraceViewer";
import { Spinner } from "@/components/atoms/Spinner";
import { Badge } from "@/components/atoms/Badge";
import { Sparkles, Terminal, Activity, Clipboard, ExternalLink } from "lucide-react";
import Link from "next/link";
import type { AgentRunResponse } from "@/types";

export default function AgentPage() {
  const [assetId, setAssetId] = useState("ISM-CNC-001");
  const [datasetId, setDatasetId] = useState("FD001");
  const [cycle, setCycle] = useState(145);
  const [rulEstimate, setRulEstimate] = useState(12.5);
  const [anomalyScore, setAnomalyScore] = useState(0.685);
  const [anomalyFlag, setAnomalyFlag] = useState(true);

  const [response, setResponse] = useState<AgentRunResponse | null>(null);
  const agentMutation = useAgentRun();

  const handleRunAgent = (e: React.FormEvent) => {
    e.preventDefault();
    setResponse(null);

    agentMutation.mutate(
      {
        asset_id: assetId,
        dataset_id: datasetId,
        cycle,
        rul_estimate: rulEstimate,
        anomaly_score: anomalyScore,
        anomaly_flag: anomalyFlag,
      },
      {
        onSuccess: (data) => {
          setResponse(data);
        },
      }
    );
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-text-primary flex items-center gap-2">
          Agent Execution Panel
        </h1>
        <p className="text-sm text-text-muted mt-1">
          Trigger and trace the 5-node LangGraph agentic pipeline mapping telemetry inputs to maintenance directives.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Trigger form (Left column) */}
        <div className="lg:col-span-1 space-y-6">
          <div className="card p-6 space-y-4">
            <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider flex items-center gap-2">
              <Terminal className="h-4 w-4 text-accent-blue" />
              Execution Parameters
            </h3>

            <form onSubmit={handleRunAgent} className="space-y-4 text-xs">
              <div className="space-y-1.5">
                <label className="text-[10px] text-text-muted uppercase font-bold">Asset ID</label>
                <input
                  type="text"
                  value={assetId}
                  onChange={(e) => setAssetId(e.target.value)}
                  className="w-full bg-bg-surface border border-white/10 rounded-lg px-3 py-2 text-text-primary focus:outline-none focus:border-accent-blue"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-[10px] text-text-muted uppercase font-bold">Dataset Variant</label>
                  <select
                    value={datasetId}
                    onChange={(e) => {
                      const value = e.target.value;
                      setDatasetId(value);
                      setAssetId({
                        FD001: "ISM-CNC-001",
                        FD002: "ISM-HYD-002",
                        FD003: "ISM-GBX-003",
                        FD004: "ISM-CMR-004",
                      }[value] ?? "ISM-CNC-001");
                    }}
                    className="w-full bg-bg-surface border border-white/10 rounded-lg px-3 py-2 text-text-primary focus:outline-none focus:border-accent-blue cursor-pointer"
                  >
                    <option value="FD001">ISM-CNC-001 · CNC Machining Center</option>
                    <option value="FD002">ISM-HYD-002 · Hydraulic Press</option>
                    <option value="FD003">ISM-GBX-003 · Helical Gearbox</option>
                    <option value="FD004">ISM-CMR-004 · Screw Compressor</option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-[10px] text-text-muted uppercase font-bold">Current Cycle</label>
                  <input
                    type="number"
                    value={cycle}
                    onChange={(e) => setCycle(parseInt(e.target.value, 10))}
                    className="w-full bg-bg-surface border border-white/10 rounded-lg px-3 py-2 font-mono text-text-primary focus:outline-none focus:border-accent-blue"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-[10px] text-text-muted uppercase font-bold">RUL Estimate (cyc)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={rulEstimate}
                    onChange={(e) => setRulEstimate(parseFloat(e.target.value))}
                    className="w-full bg-bg-surface border border-white/10 rounded-lg px-3 py-2 font-mono text-text-primary focus:outline-none focus:border-accent-blue"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-[10px] text-text-muted uppercase font-bold">Anomaly Score</label>
                  <input
                    type="number"
                    step="0.001"
                    value={anomalyScore}
                    onChange={(e) => setAnomalyScore(parseFloat(e.target.value))}
                    className="w-full bg-bg-surface border border-white/10 rounded-lg px-3 py-2 font-mono text-text-primary focus:outline-none focus:border-accent-blue"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between py-2 border-y border-white/5">
                <label className="text-[10px] text-text-muted uppercase font-bold cursor-pointer select-none" htmlFor="anomaly-flag">
                  Anomaly Flag Triggered
                </label>
                <input
                  id="anomaly-flag"
                  type="checkbox"
                  checked={anomalyFlag}
                  onChange={(e) => setAnomalyFlag(e.target.checked)}
                  className="h-4 w-4 accent-accent-blue border-white/10 bg-bg-surface rounded cursor-pointer"
                />
              </div>

              <button
                type="submit"
                disabled={agentMutation.isPending}
                className="w-full py-2 bg-accent-blue hover:bg-accent-blue/90 disabled:opacity-50 text-text-primary font-bold text-xs rounded-lg transition-all flex items-center justify-center gap-2 cursor-pointer shadow-[0_0_12px_rgba(59,130,246,0.2)]"
              >
                {agentMutation.isPending ? (
                  <Spinner size="sm" className="border-text-primary" />
                ) : (
                  <Sparkles className="h-4 w-4" />
                )}
                Trigger Diagnostics Agent
              </button>
            </form>
          </div>
        </div>

        {/* Execution Trace (Right columns) */}
        <div className="lg:col-span-2 space-y-6">
          {agentMutation.isPending && (
            <div className="card p-12 text-center flex flex-col items-center justify-center gap-3">
              <Spinner size="lg" />
              <span className="text-xs text-text-muted">LangGraph pipeline running: Supervisor initializing...</span>
            </div>
          )}

          {!agentMutation.isPending && response && (
            <div className="space-y-6 animate-fade-in">
              {/* Executive Summary */}
              <div className="card p-6 border-white/5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                  <h3 className="text-sm font-bold text-text-primary">LangGraph Run Successful</h3>
                  <div className="flex flex-wrap items-center gap-4 mt-2 text-[10px] text-text-muted font-mono">
                    <span>ID: <span className="text-text-secondary">{response.run_id}</span></span>
                    <span>Duration: <span className="text-text-secondary">{response.duration_ms} ms</span></span>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Badge variant={response.final_status.includes("order_drafted") ? "amber" : "teal"}>
                    {response.final_status.toUpperCase().replace("_", " ")}
                  </Badge>
                  {response.work_order_id && (
                    <Link
                      href={`/workorders/${response.work_order_id}`}
                      className="inline-flex items-center gap-1 text-xs text-accent-blue hover:underline font-semibold"
                    >
                      <Clipboard className="h-3.5 w-3.5" /> WO #{response.work_order_id} <ExternalLink className="h-3 w-3" />
                    </Link>
                  )}
                </div>
              </div>

              {/* Node Trace Timeline */}
              <div className="space-y-3">
                <h4 className="text-xs font-bold text-text-primary uppercase tracking-wider">Node Trace logs</h4>
                <AgentTraceViewer trace={response.trace} />
              </div>
            </div>
          )}

          {!agentMutation.isPending && !response && (
            <div className="card p-12 text-center text-text-muted">
              <Activity className="h-12 w-12 mx-auto mb-4 text-slate-600 animate-pulse" />
              <h4 className="text-sm font-semibold text-text-secondary">Awaiting Trigger</h4>
              <p className="text-xs mt-1">Select parameters on the left and trigger the diagnostics agent to run simulation.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
