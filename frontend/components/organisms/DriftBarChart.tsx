// frontend/components/organisms/DriftBarChart.tsx
"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from "recharts";

// Simulated drift metrics (PSI - Population Stability Index) for CMAPSS telemetry
const DRIFT_DATA = [
  { sensor: "Sensor 11 (HPT Coolant Flow)", psi: 0.045 },
  { sensor: "Sensor 4 (LPT Outlet Temp)", psi: 0.125 },
  { sensor: "Sensor 3 (HPT Outlet Temp)", psi: 0.082 },
  { sensor: "Sensor 15 (Bypass Ratio)", psi: 0.264 },
  { sensor: "Sensor 2 (LPT Inlet Temp)", psi: 0.031 },
  { sensor: "Sensor 21 (Turbine Inlet Press)", psi: 0.195 },
  { sensor: "Sensor 12 (LPT Coolant Flow)", psi: 0.057 },
  { sensor: "Sensor 17 (Bleed Enthalpy)", psi: 0.110 },
];

export function DriftBarChart() {
  const getBarColor = (psi: number) => {
    if (psi >= 0.25) return "#EF4444"; // red: high drift
    if (psi >= 0.1) return "#F59E0B";  // amber: moderate drift
    return "#00D4AA";                  // teal: low drift
  };

  return (
    <div className="w-full h-80 bg-bg-surface border border-white/5 rounded-xl p-5 flex flex-col justify-between shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="text-sm font-semibold text-text-primary">Data Drift Diagnostics (PSI)</h4>
          <p className="text-[10px] text-text-muted">PSI values mapping telemetry data distribution shifts.</p>
        </div>
        <div className="flex gap-4 text-[9px] font-mono text-text-muted">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded bg-accent-teal" /> &lt; 0.10 (Low)
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded bg-accent-amber" /> 0.10 - 0.25 (Mod)
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded bg-accent-red" /> &gt; 0.25 (High)
          </span>
        </div>
      </div>

      <div className="flex-1 w-full text-[10px] font-mono">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={DRIFT_DATA}
            margin={{ top: 5, right: 5, left: -25, bottom: 5 }}
            layout="vertical"
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" horizontal={false} />
            <XAxis
              type="number"
              stroke="#6B7280"
              tickLine={false}
              axisLine={false}
              domain={[0, 0.35]}
            />
            <YAxis
              dataKey="sensor"
              type="category"
              stroke="#6B7280"
              tickLine={false}
              axisLine={false}
              width={140}
              tickFormatter={(val) => {
                const parts = val.split(" (");
                return parts[0] || val;
              }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const dataPoint = payload[0].payload;
                  return (
                    <div className="bg-bg-elevated border border-white/10 p-3 rounded-lg shadow-xl font-sans text-xs">
                      <p className="font-semibold text-text-primary mb-1">
                        {dataPoint.sensor}
                      </p>
                      <div className="flex items-center justify-between gap-6">
                        <span className="text-text-muted">Population Stability Index (PSI):</span>
                        <span className="font-mono font-bold" style={{ color: getBarColor(dataPoint.psi) }}>
                          {dataPoint.psi.toFixed(3)}
                        </span>
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <ReferenceLine x={0.1} stroke="#F59E0B" strokeDasharray="3 3" />
            <ReferenceLine x={0.25} stroke="#EF4444" strokeDasharray="3 3" />
            <Bar dataKey="psi" radius={[0, 4, 4, 0]} barSize={12}>
              {DRIFT_DATA.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getBarColor(entry.psi)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
