"use client";

import React from "react";
import Link from "next/link";
import { cn, formatCycles } from "@/lib/utils";
import type { CyclePayload } from "@/types";
import { Badge } from "@/components/atoms/Badge";
import { StatusDot } from "@/components/atoms/StatusDot";
import { ArrowUpRight, Gauge, TrendingDown } from "lucide-react";

interface EngineCardProps {
  payload: CyclePayload;
  className?: string;
}

const IRONSIDE_UNIT_NAMES: Record<string, string> = {
  FD001: "5-Axis CNC Machining Center",
  FD002: "500-Tonne Hydraulic Press",
  FD003: "Helical Gearbox Drive Train",
  FD004: "Rotary Screw Compressor",
};

export function EngineCard({ payload, className }: EngineCardProps) {
  const isEmergency = payload.rul.severity === "EMERGENCY";
  const isCritical = payload.rul.severity === "CRITICAL";
  const anomalyLoad =
    payload.anomaly.threshold > 0
      ? Math.min(
          100,
          Math.round((payload.anomaly.score / payload.anomaly.threshold) * 100)
        )
      : 0;

  let statusText = "Nominal";
  let statusVariant: "teal" | "amber" | "red" | "blue" = "teal";

  if (isEmergency) {
    statusText = "Emergency";
    statusVariant = "red";
  } else if (isCritical) {
    statusText = "Critical";
    statusVariant = "amber";
  } else if (payload.anomaly.alert) {
    statusText = "Anomaly";
    statusVariant = "blue";
  }

  const status =
    payload.rul.severity === "NORMAL" && payload.anomaly.alert
      ? "ANOMALY DETECTED"
      : payload.rul.severity;

  return (
    <Link
      href={"/engines/" + payload.dataset_variant}
      className="group block"
      aria-label={"Open diagnostics for " + payload.machine_id}
    >
      <article
        className={cn(
          "card h-[272px] cursor-pointer p-5 transition-transform duration-200 group-hover:-translate-y-0.5",
          isEmergency && "card-glow-red",
          isCritical && "card-glow-amber",
          !isEmergency &&
            !isCritical &&
            "group-hover:border-accent-teal/25",
          className
        )}
      >
        <span
          className={cn(
            "absolute inset-x-0 top-0 h-0.5",
            isEmergency
              ? "bg-accent-red"
              : isCritical
                ? "bg-accent-amber"
                : payload.anomaly.alert
                  ? "bg-accent-blue"
                  : "bg-accent-teal"
          )}
        />

        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <StatusDot status={status} />
              <h3 className="font-mono text-sm font-bold text-text-primary">
                {payload.machine_id}
              </h3>
            </div>
            <p className="mt-1.5 pl-5 text-[9px] font-semibold uppercase tracking-[0.14em] text-text-muted">
              {IRONSIDE_UNIT_NAMES[payload.dataset_variant] ?? "Ironside industrial unit"}
            </p>
          </div>
          <Badge variant={statusVariant}>{statusText}</Badge>
        </div>

        <div className="mt-5 grid grid-cols-2 border border-white/[0.07] bg-black/20">
          <div className="border-r border-white/[0.07] p-3">
            <div className="flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-wider text-text-muted">
              <TrendingDown className="h-3 w-3" />
              RUL
            </div>
            <div className="mt-2 flex items-end gap-1">
              <span
                className={cn(
                  "font-mono text-[25px] font-semibold leading-none tracking-[-0.04em]",
                  isEmergency
                    ? "text-accent-red"
                    : isCritical
                      ? "text-accent-amber"
                      : "text-text-primary"
                )}
              >
                {formatCycles(payload.rul.prediction_cycles)}
              </span>
              <span className="pb-0.5 text-[9px] text-text-muted">cycles</span>
            </div>
          </div>

          <div className="p-3">
            <div className="flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-wider text-text-muted">
              <Gauge className="h-3 w-3" />
              Anomaly
            </div>
            <div className="mt-2 flex items-end gap-1">
              <span
                className={cn(
                  "font-mono text-[25px] font-semibold leading-none tracking-[-0.04em]",
                  payload.anomaly.alert
                    ? "text-accent-orange"
                    : "text-text-primary"
                )}
              >
                {payload.anomaly.score.toFixed(3)}
              </span>
            </div>
          </div>
        </div>

        <div className="mt-4">
          <div className="flex items-center justify-between text-[9px] font-semibold uppercase tracking-wider">
            <span className="text-text-muted">Anomaly threshold load</span>
            <span
              className={
                payload.anomaly.alert ? "text-accent-orange" : "text-text-secondary"
              }
            >
              {anomalyLoad}%
            </span>
          </div>
          <div className="mt-2 h-1 overflow-hidden bg-white/[0.07]">
            <div
              className={cn(
                "h-full transition-all duration-300",
                payload.anomaly.alert ? "bg-accent-orange" : "bg-accent-teal"
              )}
              style={{ width: anomalyLoad + "%" }}
            />
          </div>
        </div>

        <div className="mt-4 flex items-center justify-between border-t border-white/[0.06] pt-3 text-[10px]">
          <span className="font-mono text-text-muted">Cycle {payload.cycle}</span>
          <span className="flex items-center gap-1 font-semibold text-text-secondary transition-colors group-hover:text-accent-teal">
            Open diagnostics
            <ArrowUpRight className="h-3 w-3" />
          </span>
        </div>
      </article>
    </Link>
  );
}
