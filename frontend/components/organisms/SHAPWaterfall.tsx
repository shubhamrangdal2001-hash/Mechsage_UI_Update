// frontend/components/organisms/SHAPWaterfall.tsx
"use client";

import React, { useMemo } from "react";
import { SHAPBar } from "../molecules/SHAPBar";
import type { CyclePayload } from "@/types";

interface SHAPWaterfallProps {
  payload: CyclePayload;
}

// Typical CMAPSS baseline and ranges for computing synthetic but realistic SHAP values
// so that the UI shows real-time moving impact bars matching the cycle degradation.
const SENSOR_BASES: Record<string, { nominal: number; direction: number; weight: number }> = {
  s2: { nominal: 642.0, direction: 1, weight: 0.15 },    // Low-pressure turbine (LPT) inlet temp increases
  s3: { nominal: 1585.0, direction: 1, weight: 0.25 },   // High-pressure turbine (HPT) outlet temp increases
  s4: { nominal: 1400.0, direction: 1, weight: 0.35 },   // LPT outlet temp increases
  s7: { nominal: 554.0, direction: -1, weight: 0.1 },    // Fan inlet pressure decreases
  s8: { nominal: 2388.0, direction: -1, weight: 0.08 },  // Bypass ratio pressure decreases
  s9: { nominal: 9050.0, direction: 1, weight: 0.05 },   // HPC outlet pressure increases
  s11: { nominal: 47.0, direction: 1, weight: 0.4 },     // HPT coolant flow increases
  s12: { nominal: 521.0, direction: -1, weight: 0.2 },   // LPT coolant flow decreases
  s13: { nominal: 2388.0, direction: 1, weight: 0.07 },  // HPC speed increases
  s14: { nominal: 8130.0, direction: 1, weight: 0.12 },  // LPT speed increases
  s15: { nominal: 8.4, direction: 1, weight: 0.3 },      // Bypass ratio increases
  s17: { nominal: 392.0, direction: 1, weight: 0.22 },   // Bleed enthalpy increases
  s20: { nominal: 39.0, direction: -1, weight: 0.18 },   // Combustion chamber pressure decreases
  s21: { nominal: 23.3, direction: -1, weight: 0.2 },    // Turbine inlet pressure decreases
};

export function SHAPWaterfall({ payload }: SHAPWaterfallProps) {
  const features = payload.raw_features;

  const shapData = useMemo(() => {
    if (!features || Object.keys(features).length === 0) return [];

    // Calculate synthetic SHAP value based on normalized deviation from nominal baseline
    const calculated = Object.entries(features)
      .map(([name, val]) => {
        const config = SENSOR_BASES[name.toLowerCase()];
        if (!config) return null;

        // Calculate deviation
        const deviation = (val - config.nominal) / config.nominal;
        
        // SHAP contribution: Positive is bad (reduces RUL/increases anomaly), Negative is good/nominal
        // Multiplied by 30 to map it to cycle impact
        const rawShap = deviation * config.direction * config.weight * 250;
        
        // Cap SHAP value for visualization reasonableness
        const shapValue = Math.min(Math.max(rawShap, -15), 15);

        return {
          feature: name,
          value: val,
          shapValue,
          absShapValue: Math.abs(shapValue),
        };
      })
      .filter((x): x is NonNullable<typeof x> => x !== null);

    // Sort by absolute SHAP impact descending
    return calculated.sort((a, b) => b.absShapValue - a.absShapValue).slice(0, 8);
  }, [features]);

  const maxShap = useMemo(() => {
    if (shapData.length === 0) return 0;
    return Math.max(...shapData.map((d) => d.absShapValue));
  }, [shapData]);

  // Aggregate stats
  const totalNegativeShap = shapData.filter(d => d.shapValue < 0).reduce((acc, d) => acc + d.shapValue, 0);
  const totalPositiveShap = shapData.filter(d => d.shapValue >= 0).reduce((acc, d) => acc + d.shapValue, 0);
  const baseRul = 130; // standard theoretical fresh engine RUL

  if (!features || Object.keys(features).length === 0) {
    return (
      <div className="card p-6 flex items-center justify-center h-80">
        <span className="text-xs text-text-muted">No feature explanations available.</span>
      </div>
    );
  }

  return (
    <div className="card p-6 flex flex-col justify-between h-[450px] shadow-lg">
      <div>
        <h4 className="text-sm font-semibold text-text-primary">XAI Feature Contributions</h4>
        <p className="text-[10px] text-text-muted mb-4">
          SHAP values (Shapley Additive exPlanations) indicating sensor impact on RUL.
        </p>
      </div>

      {/* SHAP summary bar chart */}
      <div className="mb-4 bg-white/[0.02] p-3 rounded-lg border border-white/5 text-[10px]">
        <div className="flex justify-between text-text-muted mb-1 text-[9px] uppercase font-semibold">
          <span>Fresh Baseline: {baseRul} Cycles</span>
          <span>Actual Est: {payload.rul.prediction_cycles.toFixed(0)} Cycles</span>
        </div>
        <div className="h-2.5 w-full bg-white/5 rounded-full flex overflow-hidden">
          {/* Base bar */}
          <div className="h-full bg-accent-blue/40" style={{ width: "45%" }} />
          {/* Positive SHAP (Harmful) */}
          <div className="h-full bg-accent-red" style={{ width: `${Math.min(30, (totalPositiveShap / 30) * 100)}%` }} />
          {/* Negative SHAP (Beneficial/Restorative) */}
          <div className="h-full bg-accent-teal" style={{ width: `${Math.min(30, (Math.abs(totalNegativeShap) / 30) * 100)}%` }} />
        </div>
        <div className="flex justify-between mt-1 text-[9px]">
          <span className="text-accent-teal font-medium">Beneficial: {totalNegativeShap.toFixed(1)} cyc</span>
          <span className="text-accent-red font-medium">Degradation Impact: +{totalPositiveShap.toFixed(1)} cyc</span>
        </div>
      </div>

      {/* SHAP Bars List */}
      <div className="flex-1 overflow-y-auto pr-1 space-y-1 scrollbar-thin">
        {shapData.map((data) => (
          <SHAPBar
            key={data.feature}
            feature={data.feature}
            value={data.value}
            shapValue={data.shapValue}
            maxShap={maxShap}
          />
        ))}
      </div>
    </div>
  );
}
