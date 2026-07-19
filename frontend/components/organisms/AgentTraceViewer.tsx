// frontend/components/organisms/AgentTraceViewer.tsx
"use client";

import React, { useState } from "react";
import type { AgentNodeTrace } from "@/types";
import { Badge } from "@/components/atoms/Badge";
import { ChevronDown, ChevronUp, PlayCircle, CheckCircle, HelpCircle, AlertCircle, Clock } from "lucide-react";

interface AgentTraceViewerProps {
  trace: AgentNodeTrace[];
}

export function AgentTraceViewer({ trace }: AgentTraceViewerProps) {
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({
    supervisor: true, // expand first step by default
  });

  const toggleExpand = (node: string) => {
    setExpandedNodes((prev) => ({ ...prev, [node]: !prev[node] }));
  };

  const getNodeIcon = (nodeName: string, status: string) => {
    if (status === "error") return <AlertCircle className="h-5 w-5 text-accent-red" />;
    if (status === "abstained") return <HelpCircle className="h-5 w-5 text-accent-amber" />;
    
    switch (nodeName.toLowerCase()) {
      case "supervisor":
        return <PlayCircle className="h-5 w-5 text-accent-teal" />;
      default:
        return <CheckCircle className="h-5 w-5 text-accent-teal" />;
    }
  };

  const getNodeTitle = (nodeName: string) => {
    const titles: Record<string, string> = {
      supervisor: "Supervisor (Routing & Task Allocation)",
      monitor: "Telemetry Monitor (RUL & Anomaly Threshold Check)",
      diagnostics: "Diagnostics (Failure Mode & Confidence Classification)",
      work_order: "Work Order Creator (Auto-Drafting Database Record)",
      scheduling: "Scheduling Proposal (Technician & Start Time Planning)",
    };
    return titles[nodeName.toLowerCase()] || nodeName;
  };

  return (
    <div className="space-y-4">
      {/* Trace Timeline */}
      <div className="relative border-l border-white/10 pl-6 ml-3 space-y-6">
        {trace.length === 0 ? (
          <div className="text-xs text-text-muted py-6">No execution trace data available.</div>
        ) : (
          trace.map((step, idx) => {
            const isExpanded = !!expandedNodes[step.node];
            const hasOutput = Object.keys(step.output || {}).length > 0;
            const statusColor =
              step.status === "error"
                ? "red"
                : step.status === "abstained"
                ? "amber"
                : "teal";

            return (
              <div key={step.node || idx} className="relative group">
                {/* Timeline Bullet */}
                <div className="absolute -left-[37px] top-1 p-1 bg-bg-base border border-white/10 rounded-full transition-all group-hover:scale-110">
                  {getNodeIcon(step.node, step.status)}
                </div>

                {/* Timeline Step Card */}
                <div className="card p-4 border-white/5 hover:border-white/10 bg-white/[0.01] transition-all">
                  <div
                    onClick={() => hasOutput && toggleExpand(step.node)}
                    className={`flex items-center justify-between gap-4 select-none ${
                      hasOutput ? "cursor-pointer" : ""
                    }`}
                  >
                    <div>
                      <h4 className="text-xs font-bold text-text-primary">
                        {getNodeTitle(step.node)}
                      </h4>
                      <div className="flex items-center gap-3 mt-1 text-[9px] text-text-muted font-mono">
                        <span className="flex items-center gap-0.5">
                          Node: <span className="text-text-secondary font-semibold">{step.node}</span>
                        </span>
                        {step.duration_ms !== undefined && (
                          <span className="flex items-center gap-0.5">
                            <Clock className="h-3 w-3" />
                            {step.duration_ms} ms
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <Badge variant={statusColor}>
                        {step.status.toUpperCase()}
                      </Badge>
                      {hasOutput && (
                        <div>
                          {isExpanded ? (
                            <ChevronUp className="h-4 w-4 text-text-muted hover:text-text-primary" />
                          ) : (
                            <ChevronDown className="h-4 w-4 text-text-muted hover:text-text-primary" />
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Collapsible raw JSON output */}
                  {isExpanded && hasOutput && (
                    <div className="mt-3 border-t border-white/5 pt-3 animate-fade-in">
                      <div className="flex items-center justify-between text-[9px] text-text-muted uppercase font-bold tracking-wider mb-1.5 font-sans">
                        <span>Raw Node Output</span>
                      </div>
                      <pre className="text-[10px] font-mono bg-black/40 border border-white/5 rounded-lg p-3 text-accent-teal overflow-x-auto max-h-60 leading-relaxed scrollbar-thin">
                        {JSON.stringify(step.output, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
