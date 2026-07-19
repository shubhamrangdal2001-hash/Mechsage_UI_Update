// frontend/lib/utils.ts
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function formatCycles(cycles: number | null | undefined): string {
  if (cycles == null) return "—";
  return Math.round(cycles).toLocaleString();
}

export function formatTimestamp(ts: string | null | undefined): string {
  if (!ts) return "—";
  return new Date(ts).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function severityColor(severity: string): string {
  switch (severity) {
    case "EMERGENCY":
      return "text-red-400";
    case "CRITICAL":
      return "text-amber-400";
    case "ANOMALY DETECTED":
      return "text-orange-400";
    case "NORMAL":
      return "text-teal-400";
    default:
      return "text-slate-400";
  }
}

export function severityBg(severity: string): string {
  switch (severity) {
    case "EMERGENCY":
      return "bg-red-500/15 border-red-500/30";
    case "CRITICAL":
      return "bg-amber-500/15 border-amber-500/30";
    case "ANOMALY DETECTED":
      return "bg-orange-500/15 border-orange-500/30";
    default:
      return "bg-teal-500/15 border-teal-500/30";
  }
}
