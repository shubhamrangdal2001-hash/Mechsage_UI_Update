// frontend/components/molecules/AlertBanner.tsx
import React from "react";
import { cn } from "@/lib/utils";

interface AlertBannerProps {
  message: string;
  type?: "info" | "warning" | "error";
  onActionClick?: () => void;
  actionText?: string;
  className?: string;
}

export function AlertBanner({
  message,
  type = "warning",
  onActionClick,
  actionText,
  className,
}: AlertBannerProps) {
  const typeStyles = {
    info: "bg-accent-blue/10 border-accent-blue/20 text-accent-blue",
    warning: "bg-accent-amber/10 border-accent-amber/20 text-accent-amber",
    error: "bg-accent-red/10 border-accent-red/20 text-accent-red",
  };

  return (
    <div
      className={cn(
        "flex items-center justify-between px-4 py-3 rounded-lg border text-sm animate-fade-in",
        typeStyles[type],
        className
      )}
    >
      <div className="flex items-center gap-2">
        <span>{type === "error" ? "🚨" : type === "warning" ? "⚠️" : "ℹ️"}</span>
        <span className="font-medium">{message}</span>
      </div>
      {actionText && onActionClick && (
        <button
          onClick={onActionClick}
          className="text-xs font-semibold underline hover:no-underline transition-all"
        >
          {actionText}
        </button>
      )}
    </div>
  );
}
