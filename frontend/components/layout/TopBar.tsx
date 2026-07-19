"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useFleetStore } from "@/stores/fleet-store";
import {
  BookOpenCheck,
  ClipboardCheck,
  LayoutDashboard,
  Menu,
  Pause,
  Play,
  Radio,
  Search,
  Workflow,
  X,
} from "lucide-react";
import { useMetrics } from "@/hooks/useFleetData";
import { getSimControl } from "@/lib/sim-control";
import { cn } from "@/lib/utils";

const mobileNavItems = [
  { name: "Fleet overview", href: "/fleet", icon: LayoutDashboard },
  { name: "Work orders", href: "/workorders", icon: ClipboardCheck },
  { name: "Knowledge console", href: "/rag", icon: BookOpenCheck },
  { name: "Agent operations", href: "/agent", icon: Workflow },
];

const pageLabels = [
  { prefix: "/fleet", label: "Fleet overview" },
  { prefix: "/engines", label: "Asset diagnostics" },
  { prefix: "/workorders", label: "Work orders" },
  { prefix: "/rag", label: "Knowledge console" },
  { prefix: "/agent", label: "Agent operations" },
];

export default function TopBar() {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);
  const { wsConnected, globalCycle, simulationPaused, setSimulationPaused } =
    useFleetStore();
  const { data: metrics } = useMetrics();

  const pageLabel =
    pageLabels.find((item) => pathname.startsWith(item.prefix))?.label ??
    "Operations";

  const handleTogglePlay = () => {
    const action = simulationPaused ? "resume" : "pause";
    getSimControl()?.(action);
    setSimulationPaused(!simulationPaused);
  };

  return (
    <header className="relative z-40 flex h-[72px] shrink-0 items-center justify-between border-b border-white/[0.07] bg-[#090e16]/95 px-4 backdrop-blur-xl sm:px-6 xl:px-8">
      <div className="flex min-w-0 items-center gap-3">
        <button
          type="button"
          onClick={() => setMenuOpen((open) => !open)}
          className="grid h-9 w-9 shrink-0 place-items-center border border-white/[0.09] text-text-secondary transition-colors hover:bg-white/[0.04] hover:text-text-primary lg:hidden"
          aria-label={menuOpen ? "Close navigation" : "Open navigation"}
          aria-expanded={menuOpen}
        >
          {menuOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </button>

        <div className="min-w-0">
          <p className="hidden text-[9px] font-bold uppercase tracking-[0.16em] text-text-muted sm:block">
            Operations control
          </p>
          <h2 className="truncate text-sm font-semibold text-text-primary sm:mt-0.5">
            {pageLabel}
          </h2>
        </div>

        <div className="hidden h-7 w-px bg-white/[0.07] sm:block" />
        <div
          className={cn(
            "hidden items-center gap-2 text-[10px] font-semibold uppercase tracking-wider sm:flex",
            wsConnected ? "text-accent-teal" : "text-accent-red"
          )}
        >
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              wsConnected
                ? "bg-accent-teal shadow-[0_0_7px_rgba(53,216,178,0.55)]"
                : "bg-accent-red"
            )}
          />
          {wsConnected ? "Live telemetry" : "Feed offline"}
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-3">
        {metrics && (
          <div className="hidden items-center gap-5 pr-2 xl:flex">
            <div className="text-right">
              <p className="text-[9px] uppercase tracking-wider text-text-muted">Priority</p>
              <p className="mt-0.5 text-[11px] font-semibold text-text-secondary">
                <span className="text-accent-red">{metrics.fleet.emergency_count}</span> critical
                <span className="mx-1.5 text-text-muted">/</span>
                <span className="text-accent-amber">{metrics.fleet.alert_count}</span> alerts
              </p>
            </div>
            <div className="h-7 w-px bg-white/[0.07]" />
          </div>
        )}

        <Link
          href="/rag"
          className="hidden h-9 items-center gap-2 border border-white/[0.09] bg-white/[0.025] px-3 text-[11px] font-semibold text-text-secondary transition-colors hover:border-white/[0.16] hover:text-text-primary md:flex"
        >
          <Search className="h-3.5 w-3.5" />
          Search manuals
        </Link>

        <div className="flex h-9 items-center border border-white/[0.09] bg-black/20">
          <div className="hidden items-center gap-2 px-3 sm:flex">
            <Radio className="h-3 w-3 text-text-muted" />
            <span className="font-mono text-[10px] font-semibold uppercase tracking-wider text-text-secondary">
              Cycle {globalCycle}
            </span>
          </div>
          <button
            type="button"
            onClick={handleTogglePlay}
            className="grid h-full w-9 place-items-center border-l border-white/[0.08] text-text-secondary transition-colors hover:bg-white/[0.05] hover:text-text-primary sm:w-10"
            title={simulationPaused ? "Resume simulation" : "Pause simulation"}
            aria-label={simulationPaused ? "Resume simulation" : "Pause simulation"}
          >
            {simulationPaused ? (
              <Play className="h-3.5 w-3.5 fill-accent-teal/20 text-accent-teal" />
            ) : (
              <Pause className="h-3.5 w-3.5 fill-accent-amber/20 text-accent-amber" />
            )}
          </button>
        </div>
      </div>

      {menuOpen && (
        <div className="fixed inset-x-0 top-[72px] border-b border-white/[0.09] bg-[#090e16] p-3 shadow-2xl lg:hidden">
          <nav className="grid gap-1" aria-label="Mobile navigation">
            {mobileNavItems.map((item) => {
              const active = pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setMenuOpen(false)}
                  className={cn(
                    "flex items-center gap-3 px-3 py-3 text-xs font-semibold",
                    active
                      ? "bg-accent-teal/[0.08] text-accent-teal"
                      : "text-text-secondary hover:bg-white/[0.04]"
                  )}
                >
                  <item.icon className="h-4 w-4" />
                  {item.name}
                </Link>
              );
            })}
          </nav>
        </div>
      )}
    </header>
  );
}
