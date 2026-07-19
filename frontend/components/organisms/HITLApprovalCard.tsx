// frontend/components/organisms/HITLApprovalCard.tsx
"use client";

import React, { useState } from "react";
import { useHITLAction } from "@/hooks/useFleetData";
import { Badge } from "@/components/atoms/Badge";
import { Spinner } from "@/components/atoms/Spinner";
import { CheckCircle2, XCircle, AlertCircle, Calendar, User } from "lucide-react";
import toast from "react-hot-toast";

interface HITLApprovalCardProps {
  workOrderId: number;
  currentStatus: string;
  onSuccess?: () => void;
}

export function HITLApprovalCard({ workOrderId, currentStatus, onSuccess }: HITLApprovalCardProps) {
  const [action, setAction] = useState<"approve" | "reject">("approve");
  const [notes, setNotes] = useState("");
  const [technicianId, setTechnicianId] = useState("TECH-ALPHA");
  const [proposedStart, setProposedStart] = useState("");
  const hitlMutation = useHITLAction();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (action === "reject" && !notes.trim()) {
      toast.error("Rejection reasons/notes are required when rejecting a work order.");
      return;
    }

    toast.loading("Submitting decision...", { id: "hitl-action" });

    hitlMutation.mutate(
      {
        id: workOrderId,
        payload: {
          action,
          notes,
          technician_id: technicianId || undefined,
          proposed_start: proposedStart || undefined,
        },
      },
      {
        onSuccess: () => {
          toast.success(
            action === "approve" ? "Work order approved & scheduled!" : "Work order rejected.",
            { id: "hitl-action" }
          );
          if (onSuccess) onSuccess();
        },
        onError: (err: Error) => {
          toast.error(`Submission failed: ${err.message}`, { id: "hitl-action" });
        },
      }
    );
  };

  const isPending = currentStatus.toLowerCase() === "pending_approval";

  if (!isPending) {
    return (
      <div className="card p-6 border-white/5 bg-white/[0.01] flex items-center gap-3">
        {currentStatus.toLowerCase() === "approved" ? (
          <>
            <CheckCircle2 className="h-6 w-6 text-accent-teal" />
            <div>
              <h4 className="text-sm font-semibold text-text-primary">Decision Logged</h4>
              <p className="text-[10px] text-text-muted mt-0.5">This work order has been approved and moved to active queue.</p>
            </div>
          </>
        ) : (
          <>
            <XCircle className="h-6 w-6 text-accent-red" />
            <div>
              <h4 className="text-sm font-semibold text-text-primary">Decision Logged</h4>
              <p className="text-[10px] text-text-muted mt-0.5">This work order was rejected and will not be executed.</p>
            </div>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="card p-6 border-accent-amber/20 shadow-[0_0_15px_rgba(245,158,11,0.05)]">
      <div className="flex items-center gap-2 mb-4">
        <AlertCircle className="h-5 w-5 text-accent-amber animate-pulse" />
        <h3 className="text-sm font-bold text-text-primary">Human-in-the-Loop Review</h3>
        <Badge variant="amber">Needs Review</Badge>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Approve / Reject Select */}
        <div className="grid grid-cols-2 gap-4">
          <button
            type="button"
            onClick={() => setAction("approve")}
            className={`py-2 px-4 rounded-lg flex items-center justify-center gap-2 text-xs font-semibold cursor-pointer border transition-all ${
              action === "approve"
                ? "bg-accent-teal/15 text-accent-teal border-accent-teal/50 shadow-[0_0_8px_rgba(0,212,170,0.15)]"
                : "bg-white/5 text-text-muted border-white/5 hover:bg-white/10"
            }`}
          >
            <CheckCircle2 className="h-4 w-4" /> Approve &amp; Schedule
          </button>
          <button
            type="button"
            onClick={() => setAction("reject")}
            className={`py-2 px-4 rounded-lg flex items-center justify-center gap-2 text-xs font-semibold cursor-pointer border transition-all ${
              action === "reject"
                ? "bg-accent-red/15 text-accent-red border-accent-red/50 shadow-[0_0_8px_rgba(239,68,68,0.15)]"
                : "bg-white/5 text-text-muted border-white/5 hover:bg-white/10"
            }`}
          >
            <XCircle className="h-4 w-4" /> Reject Order
          </button>
        </div>

        {/* Technician ID */}
        {action === "approve" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-fade-in">
            <div className="space-y-1.5">
              <label htmlFor="technician-id" className="text-[10px] text-text-muted uppercase font-bold flex items-center gap-1">
                <User className="h-3 w-3" /> Assign Technician
              </label>
              <input
                id="technician-id"
                type="text"
                value={technicianId}
                onChange={(e) => setTechnicianId(e.target.value)}
                placeholder="Enter Technician ID"
                className="w-full bg-bg-surface border border-white/10 rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-accent-teal"
              />
            </div>
            <div className="space-y-1.5">
              <label htmlFor="proposed-start" className="text-[10px] text-text-muted uppercase font-bold flex items-center gap-1">
                <Calendar className="h-3 w-3" /> Proposed Start
              </label>
              <input
                id="proposed-start"
                type="datetime-local"
                value={proposedStart}
                onChange={(e) => setProposedStart(e.target.value)}
                className="w-full bg-bg-surface border border-white/10 rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-accent-teal cursor-pointer"
              />
            </div>
          </div>
        )}

        {/* Comments/Notes */}
        <div className="space-y-1.5">
          <label htmlFor="decision-notes" className="text-[10px] text-text-muted uppercase font-bold">
            {action === "approve" ? "Approval Notes" : "Reason for Rejection"}
          </label>
          <textarea
            id="decision-notes"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={3}
            placeholder={
              action === "approve"
                ? "Enter comments or special instructions for the technician..."
                : "Explain why this work order is being rejected (required)..."
            }
            className="w-full bg-bg-surface border border-white/10 rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-accent-teal resize-none"
          />
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={hitlMutation.isPending}
          className="w-full py-2 bg-text-primary text-bg-base font-bold text-xs rounded-lg hover:bg-text-primary/90 disabled:opacity-50 transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md"
        >
          {hitlMutation.isPending && <Spinner size="sm" className="border-bg-base" />}
          Submit Decision
        </button>
      </form>
    </div>
  );
}
