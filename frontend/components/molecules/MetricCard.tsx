import React from "react";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  title: string;
  value: string | number | React.ReactNode;
  icon?: React.ReactNode;
  description?: string;
  trend?: {
    value: string | number;
    direction: "up" | "down" | "neutral";
  };
  className?: string;
  glow?: "teal" | "red" | "amber" | "none";
}

export function MetricCard({
  title,
  value,
  icon,
  description,
  trend,
  className,
  glow = "none",
}: MetricCardProps) {
  const glowStyles = {
    teal: "card-glow-teal",
    red: "card-glow-red",
    amber: "card-glow-amber",
    none: "",
  };

  const accentStyles = {
    teal: "bg-accent-teal",
    red: "bg-accent-red",
    amber: "bg-accent-amber",
    none: "bg-white/20",
  };

  const trendColors = {
    up: "text-accent-teal",
    down: "text-accent-red",
    neutral: "text-text-muted",
  };

  return (
    <div
      className={cn(
        "card min-h-[132px] p-5",
        glowStyles[glow],
        className
      )}
    >
      <span className={cn("absolute inset-x-0 top-0 h-px", accentStyles[glow])} />
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-text-muted">
            {title}
          </p>
          <div className="mt-2.5 font-mono text-[30px] font-semibold leading-none tracking-[-0.04em] text-text-primary">
            {value}
          </div>
        </div>
        {icon && (
          <div className="grid h-9 w-9 shrink-0 place-items-center border border-white/[0.07] bg-black/20 text-text-muted">
            {icon}
          </div>
        )}
      </div>

      {(description || trend) && (
        <div className="mt-4 flex items-center gap-2 border-t border-white/[0.06] pt-3 text-[10px]">
          {trend && (
            <span className={cn("font-bold", trendColors[trend.direction])}>
              {trend.direction === "up" ? "↑" : trend.direction === "down" ? "↓" : "•"}{" "}
              {trend.value}
            </span>
          )}
          {description && <span className="truncate text-text-muted">{description}</span>}
        </div>
      )}
    </div>
  );
}
