// frontend/app/workorders/[id]/page.tsx
"use client";

import React, { use } from "react";
import Link from "next/link";
import { useWorkOrder } from "@/hooks/useFleetData";
import { HITLApprovalCard } from "@/components/organisms/HITLApprovalCard";
import { Badge } from "@/components/atoms/Badge";
import { Spinner } from "@/components/atoms/Spinner";
import {
  ArrowLeft,
  Wrench,
  Cpu,
  Activity,
  Calendar,
  User,
  BookOpen,
  FileText,
  AlertTriangle,
} from "lucide-react";
import { formatTimestamp } from "@/lib/utils";

interface WorkOrderDetailPageProps {
  params: Promise<{ id: string }>;
}

export default function WorkOrderDetailPage({ params }: WorkOrderDetailPageProps) {
  const { id } = use(params);
  const woId = parseInt(id, 10);

  const { data: wo, isLoading, refetch } = useWorkOrder(isNaN(woId) ? null : woId);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-3">
        <Spinner size="lg" />
        <span className="text-sm text-text-muted">Fetching work order dossier...</span>
      </div>
    );
  }

  if (!wo) {
    return (
      <div className="max-w-xl mx-auto text-center py-16 card space-y-6 mt-10">
        <AlertTriangle className="h-12 w-12 text-accent-red mx-auto" />
        <div>
          <h3 className="text-lg font-semibold text-text-primary">Work Order Not Found</h3>
          <p className="text-xs text-text-muted mt-1">
            Work order #{id} does not exist in the database.
          </p>
        </div>
        <Link
          href="/workorders"
          className="btn btn-secondary inline-flex items-center gap-2 text-xs"
        >
          <ArrowLeft className="h-4 w-4" /> Back to Work Orders List
        </Link>
      </div>
    );
  }

  const getPriorityColor = (priority: string) => {
    switch (priority.toLowerCase()) {
      case "critical":
        return "text-accent-red";
      case "high":
        return "text-accent-amber";
      case "medium":
        return "text-accent-blue";
      default:
        return "text-text-muted";
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Navigation Header */}
      <div className="flex items-center gap-3">
        <Link
          href="/workorders"
          className="p-2 bg-white/5 hover:bg-white/10 border border-white/5 rounded-lg transition-all text-text-muted hover:text-text-primary"
        >
          <ArrowLeft className="h-4 w-4" />
        </Link>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold text-text-primary tracking-tight">
              Work Order #{wo.id}
            </h1>
            <Badge variant={wo.status === "approved" ? "teal" : wo.status === "rejected" ? "red" : "amber"}>
              {wo.status.toUpperCase().replace("_", " ")}
            </Badge>
          </div>
          <p className="text-[11px] text-text-muted mt-0.5">
            Asset ID: <span className="font-bold text-accent-blue font-mono">{wo.asset_id}</span> • Diagnostics generated on {formatTimestamp(wo.created_at)}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Diagnostic Dossier (Left columns) */}
        <div className="lg:col-span-2 space-y-6">
          {/* Main Card */}
          <div className="card p-6 space-y-6">
            <div className="flex items-center gap-2 border-b border-white/5 pb-4">
              <Wrench className="h-5 w-5 text-accent-blue" />
              <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider">Maintenance Report Summary</h3>
            </div>

            {/* Diagnosis and Action */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-1">
                <span className="text-[10px] text-text-muted uppercase font-bold">Failure Mode / Diagnosis</span>
                <p className="text-sm font-semibold text-text-primary">{wo.failure_mode}</p>
              </div>
              <div className="space-y-1">
                <span className="text-[10px] text-text-muted uppercase font-bold">Recommended Mitigation Action</span>
                <p className="text-sm text-text-secondary">{wo.recommended_action}</p>
              </div>
            </div>

            {/* Parameters */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 bg-white/[0.01] border border-white/5 p-4 rounded-xl text-xs font-mono">
              <div>
                <span className="text-[9px] text-text-muted block font-sans font-bold uppercase mb-1">RUL Prognosis</span>
                <span className="font-bold text-accent-teal">
                  {wo.rul_estimate != null ? `${wo.rul_estimate.toFixed(1)} cyc` : "N/A"}
                </span>
              </div>
              <div>
                <span className="text-[9px] text-text-muted block font-sans font-bold uppercase mb-1">Anomaly Score</span>
                <span className="font-bold text-accent-blue">
                  {wo.anomaly_score != null ? wo.anomaly_score.toFixed(3) : "N/A"}
                </span>
              </div>
              <div>
                <span className="text-[9px] text-text-muted block font-sans font-bold uppercase mb-1">Confidence</span>
                <span className="font-bold text-text-primary">
                  {Math.round(wo.diagnosis_confidence * 100)}%
                </span>
              </div>
              <div>
                <span className="text-[9px] text-text-muted block font-sans font-bold uppercase mb-1">Severity / priority</span>
                <span className={`font-bold uppercase ${getPriorityColor(wo.priority)}`}>
                  {wo.priority}
                </span>
              </div>
            </div>

            {/* Top SHAP Features contributing */}
            {wo.shap_top_features && wo.shap_top_features.length > 0 && (
              <div className="space-y-2">
                <span className="text-[10px] text-text-muted uppercase font-bold flex items-center gap-1">
                  <Activity className="h-3.5 w-3.5" /> Contributing Anomalous Sensors (SHAP Top)
                </span>
                <div className="flex flex-wrap gap-2">
                  {wo.shap_top_features.map((feat) => (
                    <span key={feat} className="px-2.5 py-1 bg-accent-red/10 border border-accent-red/20 text-accent-red rounded text-[10px] font-semibold font-mono">
                      {feat.toUpperCase()}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Bill of Materials / Parts */}
            {wo.parts && wo.parts.length > 0 && (
              <div className="space-y-2">
                <span className="text-[10px] text-text-muted uppercase font-bold flex items-center gap-1">
                  <Cpu className="h-3.5 w-3.5" /> Bill of Materials (Required Spares)
                </span>
                <div className="flex flex-wrap gap-2">
                  {wo.parts.map((part) => (
                    <span key={part} className="px-2.5 py-1 bg-white/5 border border-white/10 text-text-primary rounded text-[10px] font-semibold">
                      {part}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* RAG Context */}
            {wo.rag_context && (
              <div className="space-y-2">
                <span className="text-[10px] text-text-muted uppercase font-bold flex items-center gap-1.5">
                  <BookOpen className="h-3.5 w-3.5 text-accent-blue" />
                  RAG Technical Manual Retrieval
                </span>
                <div className="bg-bg-elevated p-4 rounded-xl border border-white/5 text-[11px] leading-relaxed text-text-secondary whitespace-pre-wrap italic font-serif">
                  &ldquo;{wo.rag_context}&rdquo;
                </div>
              </div>
            )}
          </div>
        </div>

        {/* HITL Action Section (Right column) */}
        <div className="lg:col-span-1 space-y-6">
          <HITLApprovalCard
            workOrderId={wo.id}
            currentStatus={wo.status}
            onSuccess={refetch}
          />

          {/* Schedule proposal log details */}
          <div className="card p-6 space-y-4">
            <div className="flex items-center gap-2 border-b border-white/5 pb-3">
              <FileText className="h-4 w-4 text-text-muted" />
              <h4 className="text-xs font-bold text-text-primary uppercase tracking-wider">Proposal Details</h4>
            </div>
            <div className="space-y-3 text-[11px]">
              <div className="flex justify-between">
                <span className="text-text-muted">Est. Work Duration:</span>
                <span className="font-semibold text-text-primary">{wo.estimated_duration_hrs} hours</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Workflow Engine:</span>
                <span className="font-semibold text-text-primary">{wo.auto_generated ? "Agent Automated Draft" : "Manual Log"}</span>
              </div>
              {wo.technician_id && (
                <div className="flex justify-between border-t border-white/5 pt-2">
                  <span className="text-text-muted flex items-center gap-1"><User className="h-3.5 w-3.5" /> Assigned Tech:</span>
                  <span className="font-semibold text-accent-blue">{wo.technician_id}</span>
                </div>
              )}
              {wo.proposed_start && (
                <div className="flex justify-between">
                  <span className="text-text-muted flex items-center gap-1"><Calendar className="h-3.5 w-3.5" /> Scheduled Start:</span>
                  <span className="font-semibold text-text-primary">{formatTimestamp(wo.proposed_start)}</span>
                </div>
              )}
              {wo.rejection_reason && (
                <div className="bg-accent-red/10 border border-accent-red/20 rounded p-3 text-accent-red">
                  <span className="font-bold block mb-0.5">Rejection Log:</span>
                  {wo.rejection_reason}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
