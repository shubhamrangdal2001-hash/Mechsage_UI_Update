"use client";

import React from "react";
import Sidebar from "./Sidebar";
import TopBar from "./TopBar";

interface DashboardShellProps {
  children: React.ReactNode;
}

export default function DashboardShell({ children }: DashboardShellProps) {
  return (
    <div className="flex h-screen overflow-hidden bg-bg-base">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <TopBar />
        <main className="flex-1 overflow-y-auto px-4 py-5 sm:px-6 sm:py-6 xl:px-8 xl:py-7">
          <div className="animate-fade-in">{children}</div>
        </main>
      </div>
    </div>
  );
}
