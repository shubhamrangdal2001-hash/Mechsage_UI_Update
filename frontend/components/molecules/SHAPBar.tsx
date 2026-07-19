// frontend/components/molecules/SHAPBar.tsx
import React from "react";
import { cn } from "@/lib/utils";

interface SHAPBarProps {
  feature: string;
  value: number; // raw value of the sensor
  shapValue: number; // SHAP contribution value (e.g., -5.2 or +3.1)
  maxShap: number; // max absolute SHAP value for scaling
  unit?: string;
}

export function SHAPBar({ feature, value, shapValue, maxShap, unit = "" }: SHAPBarProps) {
  const isPositive = shapValue >= 0;
  const percentage = maxShap > 0 ? (Math.abs(shapValue) / maxShap) * 50 : 0; // 50% max width per side

  // Format sensor name for better readability
  const formatFeatureName = (name: string) => {
    const sensorMap: Record<string, string> = {
      s2: "Sensor 2 (LPT Inlet Temp)",
      s3: "Sensor 3 (HPT Outlet Temp)",
      s4: "Sensor 4 (LPT Outlet Temp)",
      s7: "Sensor 7 (Fan Inlet Press)",
      s8: "Sensor 8 (Bypass Ratio Press)",
      s9: "Sensor 9 (HPC Outlet Press)",
      s11: "Sensor 11 (HPT Coolant Flow)",
      s12: "Sensor 12 (LPT Coolant Flow)",
      s13: "Sensor 13 (HPC Speed)",
      s14: "Sensor 14 (LPT Speed)",
      s15: "Sensor 15 (Bypass Ratio)",
      s17: "Sensor 17 (Bleed Enthalpy)",
      s20: "Sensor 20 (Combustion Press)",
      s21: "Sensor 21 (Turbine Inlet Press)",
    };
    return sensorMap[name.toLowerCase()] || name;
  };

  return (
    <div className="grid grid-cols-12 items-center gap-3 py-2 text-xs border-b border-white/5 last:border-b-0 hover:bg-white/[0.02] px-2 rounded-md transition-all">
      {/* Feature Name & Value */}
      <div className="col-span-4 flex flex-col">
        <span className="font-medium text-text-primary truncate">
          {formatFeatureName(feature)}
        </span>
        <span className="text-text-muted text-[10px]">
          Value: <span className="text-text-secondary font-mono">{value.toFixed(2)}{unit}</span>
        </span>
      </div>

      {/* Bi-directional SHAP Bar */}
      <div className="col-span-6 relative h-4 bg-white/5 rounded overflow-hidden flex">
        {/* Left Side (Negative/Beneficial SHAP: extends left from center) */}
        <div className="w-1/2 flex justify-end border-r border-white/10 relative">
          {!isPositive && (
            <div
              className="h-full bg-accent-teal/80 border-r-2 border-accent-teal shadow-[0_0_8px_rgba(0,212,170,0.3)] transition-all duration-500"
              style={{ width: `${percentage * 2}%` }}
            />
          )}
        </div>

        {/* Right Side (Positive/Degrading SHAP: extends right from center) */}
        <div className="w-1/2 flex justify-start relative">
          {isPositive && (
            <div
              className="h-full bg-accent-red/80 border-l-2 border-accent-red shadow-[0_0_8px_rgba(239,68,68,0.3)] transition-all duration-500"
              style={{ width: `${percentage * 2}%` }}
            />
          )}
        </div>
      </div>

      {/* SHAP Contribution Value */}
      <div className="col-span-2 text-right font-mono font-semibold">
        <span className={cn(isPositive ? "text-accent-red" : "text-accent-teal")}>
          {isPositive ? "+" : ""}
          {shapValue.toFixed(2)}
        </span>
      </div>
    </div>
  );
}
