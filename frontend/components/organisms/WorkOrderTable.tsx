// frontend/components/organisms/WorkOrderTable.tsx
"use client";

import React, { useState } from "react";
import Link from "next/link";
import { formatTimestamp } from "@/lib/utils";
import type { WorkOrder } from "@/types";
import { Badge } from "@/components/atoms/Badge";
import { Filter, AlertTriangle, Eye } from "lucide-react";

interface WorkOrderTableProps {
  workOrders: WorkOrder[];
}

export function WorkOrderTable({ workOrders }: WorkOrderTableProps) {
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [priorityFilter, setPriorityFilter] = useState<string>("ALL");

  // Filtering logic
  const filtered = workOrders.filter((wo) => {
    const statusMatch = statusFilter === "ALL" || wo.status.toUpperCase() === statusFilter;
    const priorityMatch = priorityFilter === "ALL" || wo.priority.toUpperCase() === priorityFilter;
    return statusMatch && priorityMatch;
  });

  const getPriorityBadge = (priority: string) => {
    switch (priority.toLowerCase()) {
      case "critical":
        return <Badge variant="red">CRITICAL</Badge>;
      case "high":
        return <Badge variant="amber">HIGH</Badge>;
      case "medium":
        return <Badge variant="blue">MEDIUM</Badge>;
      default:
        return <Badge variant="slate">LOW</Badge>;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status.toLowerCase()) {
      case "approved":
        return <Badge variant="teal">APPROVED</Badge>;
      case "rejected":
        return <Badge variant="red">REJECTED</Badge>;
      case "in_progress":
        return <Badge variant="blue">IN PROGRESS</Badge>;
      case "completed":
        return <Badge variant="slate">COMPLETED</Badge>;
      default:
        return <Badge variant="amber">PENDING APPROVAL</Badge>;
    }
  };

  return (
    <div className="space-y-4">
      {/* Table Filters */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-bg-surface border border-white/5 p-4 rounded-xl">
        <div className="flex items-center gap-2 text-xs font-semibold text-text-muted">
          <Filter className="h-4 w-4" />
          <span>Filters:</span>
        </div>
        <div className="flex flex-wrap gap-3">
          {/* Status Select */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-bg-elevated border border-white/10 text-xs text-text-primary rounded-lg px-3 py-1.5 focus:outline-none focus:border-accent-teal cursor-pointer"
          >
            <option value="ALL">All Statuses</option>
            <option value="PENDING_APPROVAL">Pending Approval</option>
            <option value="APPROVED">Approved</option>
            <option value="REJECTED">Rejected</option>
            <option value="IN_PROGRESS">In Progress</option>
            <option value="COMPLETED">Completed</option>
          </select>

          {/* Priority Select */}
          <select
            value={priorityFilter}
            onChange={(e) => setPriorityFilter(e.target.value)}
            className="bg-bg-elevated border border-white/10 text-xs text-text-primary rounded-lg px-3 py-1.5 focus:outline-none focus:border-accent-teal cursor-pointer"
          >
            <option value="ALL">All Priorities</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>
        </div>
      </div>

      {/* Grid / Table Container */}
      <div className="card overflow-x-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="border-b border-white/5 text-text-muted font-semibold uppercase bg-white/[0.01]">
              <th className="py-3 px-4">WO ID</th>
              <th className="py-3 px-4">Asset ID</th>
              <th className="py-3 px-4">Diagnosis</th>
              <th className="py-3 px-4">Priority</th>
              <th className="py-3 px-4">Est. Duration</th>
              <th className="py-3 px-4">Created At</th>
              <th className="py-3 px-4">Status</th>
              <th className="py-3 px-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5 text-text-secondary font-medium">
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={8} className="py-12 text-center text-text-muted">
                  <AlertTriangle className="h-8 w-8 mx-auto mb-2 text-slate-600" />
                  No work orders matching filter criteria.
                </td>
              </tr>
            ) : (
              filtered.map((wo) => (
                <tr key={wo.id} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-3.5 px-4 font-bold font-mono text-text-primary">
                    #{wo.id}
                  </td>
                  <td className="py-3.5 px-4 font-bold text-accent-blue">
                    {wo.asset_id}
                  </td>
                  <td className="py-3.5 px-4 max-w-xs truncate">
                    <span className="text-text-primary block font-semibold">
                      {wo.failure_mode}
                    </span>
                    <span className="text-text-muted block text-[10px] truncate">
                      {wo.recommended_action}
                    </span>
                  </td>
                  <td className="py-3.5 px-4">{getPriorityBadge(wo.priority)}</td>
                  <td className="py-3.5 px-4 font-mono">
                    {wo.estimated_duration_hrs} hrs
                  </td>
                  <td className="py-3.5 px-4 text-text-muted">
                    {formatTimestamp(wo.created_at)}
                  </td>
                  <td className="py-3.5 px-4">{getStatusBadge(wo.status)}</td>
                  <td className="py-3.5 px-4 text-right">
                    <Link
                      href={`/workorders/${wo.id}`}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-white/5 hover:bg-white/10 hover:text-text-primary text-text-muted border border-white/5 hover:border-white/10 rounded transition-all font-semibold"
                    >
                      <Eye className="h-3.5 w-3.5" /> Details
                    </Link>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
