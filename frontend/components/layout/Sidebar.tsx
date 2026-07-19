"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  Activity,
  BookOpenCheck,
  ClipboardCheck,
  LayoutDashboard,
  ShieldCheck,
  Workflow,
} from "lucide-react";

const navItems = [
  {
    name: "Fleet overview",
    description: "Live asset health",
    href: "/fleet",
    icon: LayoutDashboard,
  },
  {
    name: "Work orders",
    description: "Maintenance queue",
    href: "/workorders",
    icon: ClipboardCheck,
  },
  {
    name: "Knowledge console",
    description: "Manuals and guidance",
    href: "/rag",
    icon: BookOpenCheck,
  },
  {
    name: "Agent operations",
    description: "Execution trace",
    href: "/agent",
    icon: Workflow,
  },
];

interface SidebarProps {
  className?: string;
}

export default function Sidebar({ className }: SidebarProps) {
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        "hidden h-screen w-[272px] shrink-0 flex-col border-r border-white/[0.07] bg-[#090e16] lg:flex",
        className
      )}
    >
      <div className="flex h-[72px] items-center gap-3 border-b border-white/[0.07] px-5">
        <div className="relative grid h-9 w-9 place-items-center border border-accent-teal/30 bg-accent-teal/[0.08]">
          <Activity className="h-5 w-5 text-accent-teal" strokeWidth={1.8} />
          <span className="absolute -right-px -top-px h-1.5 w-1.5 bg-accent-teal" />
        </div>
        <div className="min-w-0">
          <h1 className="text-[15px] font-bold leading-tight tracking-[0.01em] text-text-primary">
            MECHSAGE
          </h1>
          <p className="mt-0.5 text-[9px] font-semibold uppercase tracking-[0.18em] text-text-muted">
            Asset intelligence
          </p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-5">
        <p className="px-3 text-[9px] font-bold uppercase tracking-[0.18em] text-text-muted">
          Operations
        </p>
        <nav className="mt-3 space-y-1" aria-label="Primary navigation">
          {navItems.map((item) => {
            const isActive = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={isActive ? "page" : undefined}
                className={cn(
                  "group relative flex items-center gap-3 px-3 py-3 transition-colors",
                  isActive
                    ? "bg-white/[0.055] text-text-primary"
                    : "text-text-secondary hover:bg-white/[0.03] hover:text-text-primary"
                )}
              >
                {isActive && (
                  <span className="absolute inset-y-2 left-0 w-0.5 bg-accent-teal" />
                )}
                <div
                  className={cn(
                    "grid h-8 w-8 shrink-0 place-items-center border transition-colors",
                    isActive
                      ? "border-accent-teal/25 bg-accent-teal/[0.08] text-accent-teal"
                      : "border-white/[0.06] bg-white/[0.02] text-text-muted group-hover:text-text-secondary"
                  )}
                >
                  <item.icon className="h-4 w-4" strokeWidth={1.8} />
                </div>
                <div className="min-w-0">
                  <p className="text-[13px] font-semibold leading-tight">{item.name}</p>
                  <p className="mt-1 text-[10px] leading-tight text-text-muted">
                    {item.description}
                  </p>
                </div>
              </Link>
            );
          })}
        </nav>

        <div className="mx-3 mt-7 border-t border-white/[0.07] pt-6">
          <p className="text-[9px] font-bold uppercase tracking-[0.18em] text-text-muted">
            Platform status
          </p>
          <div className="mt-3 border border-white/[0.07] bg-white/[0.018] p-3.5">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-[11px] font-semibold text-text-secondary">
                <ShieldCheck className="h-3.5 w-3.5 text-accent-teal" />
                Guardrails
              </span>
              <span className="text-[9px] font-bold uppercase tracking-wider text-accent-teal">
                Active
              </span>
            </div>
            <p className="mt-2 text-[10px] leading-relaxed text-text-muted">
              Human approval is enforced for all maintenance actions.
            </p>
          </div>
        </div>
      </div>

      <div className="border-t border-white/[0.07] px-5 py-4">
        <div className="flex items-center gap-3">
          <div className="grid h-8 w-8 place-items-center bg-[#17212e] text-[10px] font-bold text-text-secondary">
            IM
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-[11px] font-semibold text-text-secondary">
              Ironside Manufacturing
            </p>
            <p className="mt-0.5 text-[9px] uppercase tracking-wider text-text-muted">
              Plant 04 · Production
            </p>
          </div>
          <span className="h-2 w-2 rounded-full bg-accent-teal shadow-[0_0_8px_rgba(53,216,178,0.5)]" />
        </div>
      </div>
    </aside>
  );
}
