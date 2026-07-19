// frontend/app/layout.tsx
"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "react-hot-toast";
import { useState } from "react";
import "@/app/globals.css";
import DashboardShell from "@/components/layout/DashboardShell";
import { useFleetWebSocket } from "@/hooks/useFleetWebSocket";

function WebSocketInitializer({ children }: { children: React.ReactNode }) {
  // Initialise WebSocket connection globally
  useFleetWebSocket();
  return <>{children}</>;
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            refetchOnWindowFocus: false,
            retry: 1,
          },
        },
      })
  );

  return (
    <html lang="en">
      <body className="bg-bg-base text-text-primary">
        <QueryClientProvider client={queryClient}>
          <WebSocketInitializer>
            <DashboardShell>{children}</DashboardShell>
            <Toaster
              position="top-right"
              toastOptions={{
                style: {
                  background: "#1F2937",
                  color: "#F9FAFB",
                  border: "1px solid rgba(255, 255, 255, 0.07)",
                },
              }}
            />
          </WebSocketInitializer>
        </QueryClientProvider>
      </body>
    </html>
  );
}
