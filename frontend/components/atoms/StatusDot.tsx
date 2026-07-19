// frontend/components/atoms/StatusDot.tsx
import { cn } from "@/lib/utils";

interface StatusDotProps {
  status: "NORMAL" | "CRITICAL" | "EMERGENCY" | "ANOMALY DETECTED" | "offline";
  pulse?: boolean;
  className?: string;
}

export function StatusDot({ status, pulse = true, className }: StatusDotProps) {
  const statusStyles = {
    NORMAL: "bg-accent-teal shadow-accent-teal/20",
    CRITICAL: "bg-accent-amber shadow-accent-amber/20",
    EMERGENCY: "bg-accent-red shadow-accent-red/20",
    "ANOMALY DETECTED": "bg-accent-orange shadow-accent-orange/20",
    offline: "bg-text-muted shadow-transparent",
  };

  const pulseAnimations = {
    NORMAL: "bg-accent-teal/40",
    CRITICAL: "bg-accent-amber/40 pulse-critical",
    EMERGENCY: "bg-accent-red/40 pulse-emergency",
    "ANOMALY DETECTED": "bg-accent-orange/40",
    offline: "bg-text-muted/40",
  };

  return (
    <div className={cn("relative flex h-3 w-3", className)}>
      {pulse && status !== "offline" && (
        <span
          className={cn(
            "animate-ping absolute inline-flex h-full w-full rounded-full opacity-75",
            pulseAnimations[status]
          )}
        />
      )}
      <span
        className={cn(
          "relative inline-flex rounded-full h-3 w-3 shadow-sm border border-black/10",
          statusStyles[status]
        )}
      />
    </div>
  );
}
