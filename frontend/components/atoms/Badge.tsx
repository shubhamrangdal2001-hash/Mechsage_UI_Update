import React from "react";
import { cn } from "@/lib/utils";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "teal" | "amber" | "red" | "blue" | "slate";
  className?: string;
}

export function Badge({ children, variant = "slate", className }: BadgeProps) {
  const variantStyles = {
    teal: "bg-accent-teal/[0.08] text-accent-teal border-accent-teal/25",
    amber: "bg-accent-amber/[0.08] text-accent-amber border-accent-amber/25",
    red: "bg-accent-red/[0.08] text-accent-red border-accent-red/25",
    blue: "bg-accent-blue/[0.08] text-accent-blue border-accent-blue/25",
    slate: "bg-white/[0.035] text-text-secondary border-white/[0.08]",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center border px-2 py-1 text-[9px] font-bold uppercase leading-none tracking-[0.1em]",
        variantStyles[variant],
        className
      )}
    >
      {children}
    </span>
  );
}
