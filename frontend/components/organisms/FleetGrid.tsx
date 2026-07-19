"use client";

import React from "react";
import { EngineCard } from "./EngineCard";
import { Skeleton } from "@/components/atoms/Skeleton";
import { Badge } from "@/components/atoms/Badge";
import type { CyclePayload } from "@/types";

interface FleetGridProps {
  engines: CyclePayload[];
  loading?: boolean;
}

export function FleetGrid({ engines, loading = false }: FleetGridProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {[1, 2, 3, 4].map((id) => (
          <Skeleton key={id} className="h-[272px] w-full rounded-[10px]" />
        ))}
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {engines.map((engine) => {
          if (engine.warming_up) {
            const percent = Math.round((engine.warmup_progress ?? 0) * 100);
            return (
              <div
                key={engine.dataset_variant}
                className="card panel-grid flex h-[272px] flex-col justify-between p-5"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm font-bold text-text-primary">
                      {engine.machine_id}
                    </span>
                    <Badge variant="slate">Calibrating</Badge>
                  </div>
                  <p className="mt-4 max-w-[220px] text-[11px] leading-5 text-text-muted">
                    Fitting signal scalers and buffering the initial operating
                    cycles.
                  </p>
                </div>
                <div>
                  <div className="flex justify-between text-[10px] font-semibold uppercase tracking-wider text-text-secondary">
                    <span>Signal calibration</span>
                    <span className="font-mono">{percent}%</span>
                  </div>
                  <div className="mt-2 h-1 overflow-hidden bg-white/[0.07]">
                    <div
                      className="h-full bg-accent-teal transition-all duration-300"
                      style={{ width: percent + "%" }}
                    />
                  </div>
                </div>
              </div>
            );
          }

          return <EngineCard key={engine.dataset_variant} payload={engine} />;
        })}
      </div>
    </div>
  );
}
