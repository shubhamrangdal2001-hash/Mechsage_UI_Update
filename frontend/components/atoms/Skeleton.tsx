// frontend/components/atoms/Skeleton.tsx
import { cn } from "@/lib/utils";

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={cn(
        "animate-pulse rounded bg-slate-800 border border-slate-700/50",
        className
      )}
    />
  );
}
