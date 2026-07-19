// frontend/components/molecules/SeverityPill.tsx
import { cn, severityColor, severityBg } from "@/lib/utils";

interface SeverityPillProps {
  severity: "NORMAL" | "CRITICAL" | "EMERGENCY" | "ANOMALY DETECTED" | string;
  className?: string;
}

export function SeverityPill({ severity, className }: SeverityPillProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border",
        severityBg(severity),
        severityColor(severity),
        className
      )}
    >
      {severity}
    </span>
  );
}
