"use client";

import React from "react";
import Link from "next/link";
import { useFleetStore } from "@/stores/fleet-store";
import { useMetrics } from "@/hooks/useFleetData";
import { MetricCard } from "@/components/molecules/MetricCard";
import { FleetGrid } from "@/components/organisms/FleetGrid";
import {
  Activity,
  AlertOctagon,
  ArrowUpRight,
  ClipboardCheck,
  Cpu,
  Radio,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";
import { Badge } from "@/components/atoms/Badge";
import { formatTimestamp } from "@/lib/utils";

export default function FleetOverviewPage() {
  const {
    liveEngines,
    alerts,
    globalCycle,
    wsConnected,
    simulationPaused,
  } = useFleetStore();
  const { isLoading } = useMetrics();

  const activeEngines = Object.values(liveEngines);
  const activeCount = activeEngines.length;
  const emergencyCount = activeEngines.filter(
    (engine) => engine.rul?.severity === "EMERGENCY"
  ).length;
  const attentionCount = activeEngines.filter(
    (engine) =>
      engine.rul?.severity !== "NORMAL" ||
      engine.anomaly?.alert ||
      engine.trigger_agent
  ).length;
  const nominalCount = Math.max(0, activeCount - attentionCount);
  const nominalRatio =
    activeCount > 0 ? Math.round((nominalCount / activeCount) * 100) : 0;

  return (
    <div className="mx-auto max-w-[1600px] space-y-6">
      <section className="card panel-grid p-5 sm:p-7">
        <div className="relative z-10 flex flex-col justify-between gap-7 xl:flex-row xl:items-end">
          <div className="max-w-3xl">
            <p className="eyebrow">Unified asset intelligence</p>
            <h1 className="gradient-text mt-3 text-3xl font-semibold tracking-[-0.035em] sm:text-[38px]">
              Fleet command center
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-text-secondary">
              Real-time health, remaining useful life, and anomaly intelligence
              across the Ironside Manufacturing fleet—with human approval at every maintenance
              decision.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-2">
              <Link
                href="/workorders"
                className="inline-flex h-9 items-center gap-2 bg-accent-teal px-3.5 text-[11px] font-bold text-[#07110e] transition-colors hover:bg-[#62e4c5]"
              >
                <ClipboardCheck className="h-3.5 w-3.5" />
                Review work orders
                <ArrowUpRight className="h-3.5 w-3.5" />
              </Link>
              <Link
                href="/rag"
                className="inline-flex h-9 items-center gap-2 border border-white/[0.1] bg-black/20 px-3.5 text-[11px] font-semibold text-text-secondary transition-colors hover:border-white/[0.18] hover:text-text-primary"
              >
                Search technical guidance
              </Link>
            </div>
          </div>

          <div className="grid min-w-full grid-cols-2 border border-white/[0.08] bg-black/20 sm:min-w-[370px]">
            <div className="border-r border-white/[0.08] p-4">
              <p className="text-[9px] font-bold uppercase tracking-[0.15em] text-text-muted">
                Telemetry link
              </p>
              <div className="mt-2 flex items-center gap-2">
                <Radio
                  className={
                    wsConnected
                      ? "h-4 w-4 text-accent-teal"
                      : "h-4 w-4 text-accent-red"
                  }
                />
                <span className="text-xs font-semibold text-text-primary">
                  {wsConnected ? "Streaming" : "Disconnected"}
                </span>
              </div>
              <p className="mt-1 text-[10px] text-text-muted">
                {simulationPaused ? "Simulation paused" : "Live cycle ingestion"}
              </p>
            </div>
            <div className="p-4">
              <p className="text-[9px] font-bold uppercase tracking-[0.15em] text-text-muted">
                Fleet posture
              </p>
              <div className="mt-2 flex items-baseline gap-1.5">
                <span className="font-mono text-xl font-semibold text-text-primary">
                  {nominalRatio}%
                </span>
                <span className="text-[10px] text-text-muted">nominal</span>
              </div>
              <p className="mt-1 text-[10px] text-text-muted">
                {nominalCount} of {activeCount} assets healthy
              </p>
            </div>
          </div>
        </div>
      </section>

      <section
        className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4"
        aria-label="Fleet performance indicators"
      >
        <MetricCard
          title="Assets online"
          value={activeCount > 0 ? activeCount : isLoading ? "—" : "0"}
          icon={<Cpu className="h-4.5 w-4.5 text-accent-teal" />}
          description="Units reporting telemetry"
          glow="teal"
        />
        <MetricCard
          title="Current cycle"
          value={globalCycle > 0 ? globalCycle : isLoading ? "—" : "0"}
          icon={<Activity className="h-4.5 w-4.5 text-accent-blue" />}
          description="Synchronized simulation index"
        />
        <MetricCard
          title="Attention required"
          value={attentionCount}
          icon={<ShieldAlert className="h-4.5 w-4.5 text-accent-amber" />}
          description="Threshold or anomaly events"
          glow={attentionCount > 0 && emergencyCount === 0 ? "amber" : "none"}
        />
        <MetricCard
          title="Critical assets"
          value={emergencyCount}
          icon={<AlertOctagon className="h-4.5 w-4.5 text-accent-red" />}
          description="Immediate intervention"
          glow={emergencyCount > 0 ? "red" : "none"}
        />
      </section>

      <section className="space-y-3" aria-labelledby="asset-grid-title">
        <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
          <div>
            <p className="eyebrow">Asset health matrix</p>
            <h2
              id="asset-grid-title"
              className="mt-1.5 text-lg font-semibold tracking-tight text-text-primary"
            >
              Live Ironside systems
            </h2>
          </div>
          <div className="flex items-center gap-4 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
            <span className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-accent-teal" />
              Nominal
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-accent-amber" />
              Attention
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-accent-red" />
              Critical
            </span>
          </div>
        </div>

        {activeCount === 0 && isLoading ? (
          <FleetGrid engines={[]} loading />
        ) : activeCount === 0 ? (
          <div className="card panel-grid flex min-h-64 flex-col items-center justify-center p-8 text-center">
            <div className="grid h-12 w-12 place-items-center border border-white/[0.08] bg-black/20">
              <Activity className="h-5 w-5 animate-pulse text-text-muted" />
            </div>
            <p className="mt-4 text-sm font-semibold text-text-secondary">
              Awaiting telemetry
            </p>
            <p className="mt-1 max-w-md text-xs leading-5 text-text-muted">
              Start the FastAPI service and simulation stream to populate the
              fleet health matrix.
            </p>
          </div>
        ) : (
          <FleetGrid engines={activeEngines} />
        )}
      </section>

      <section className="card" aria-labelledby="incident-log-title">
        <div className="flex items-center justify-between border-b border-white/[0.07] px-4 py-4 sm:px-5">
          <div className="flex items-center gap-3">
            <div className="grid h-8 w-8 place-items-center border border-accent-amber/20 bg-accent-amber/[0.06]">
              <ShieldCheck className="h-4 w-4 text-accent-amber" />
            </div>
            <div>
              <p className="text-[9px] font-bold uppercase tracking-[0.15em] text-text-muted">
                SCADA event stream
              </p>
              <h2 id="incident-log-title" className="mt-0.5 text-sm font-semibold text-text-primary">
                Incident log
              </h2>
            </div>
          </div>
          <Badge variant={alerts.length > 0 ? "amber" : "teal"}>
            {alerts.length} events
          </Badge>
        </div>

        <div className="max-h-[320px] overflow-auto" aria-live="polite">
          {alerts.length === 0 ? (
            <div className="flex min-h-36 flex-col items-center justify-center px-5 py-8 text-center">
              <ShieldCheck className="h-5 w-5 text-accent-teal" />
              <p className="mt-2 text-xs font-semibold text-text-secondary">
                No active incidents
              </p>
              <p className="mt-1 text-[10px] text-text-muted">
                All reporting assets are within configured operating thresholds.
              </p>
            </div>
          ) : (
            <>
              <div className="hidden grid-cols-[100px_120px_1fr_110px] gap-4 border-b border-white/[0.06] bg-black/15 px-5 py-2.5 text-[9px] font-bold uppercase tracking-[0.14em] text-text-muted sm:grid">
                <span>Time</span>
                <span>Asset</span>
                <span>Diagnostic event</span>
                <span className="text-right">Severity</span>
              </div>
              <div className="divide-y divide-white/[0.055]">
                {alerts.slice(0, 12).map((alert, index) => (
                  <div
                    key={alert.asset_id + alert.timestamp + index}
                    className="grid gap-2 px-4 py-3 transition-colors hover:bg-white/[0.02] sm:grid-cols-[100px_120px_1fr_110px] sm:items-center sm:gap-4 sm:px-5"
                  >
                    <span className="font-mono text-[10px] text-text-muted">
                      {formatTimestamp(alert.timestamp)}
                    </span>
                    <span className="font-mono text-[10px] font-semibold text-accent-blue">
                      {alert.asset_id}
                    </span>
                    <span className="text-[11px] leading-5 text-text-secondary">
                      RUL reached {Math.round(alert.rul)} cycles; anomaly score{" "}
                      {alert.anomaly_score.toFixed(3)}
                    </span>
                    <div className="sm:text-right">
                      <Badge
                        variant={
                          alert.severity === "EMERGENCY" ? "red" : "amber"
                        }
                      >
                        {alert.severity}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  );
}
