// frontend/components/organisms/RULChart.tsx
"use client";

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { CyclePayload } from "@/types";

interface RULChartProps {
  history: CyclePayload[];
}

export function RULChart({ history }: RULChartProps) {
  // Map history to chart data
  const data = history.map((item) => ({
    cycle: item.cycle,
    predictedRUL: item.rul.prediction_cycles,
    threshold: item.rul.threshold_cycles,
  }));

  const latestPrediction = history[history.length - 1]?.rul.prediction_cycles ?? 0;
  const isEmergency = latestPrediction <= 15;
  const isCritical = latestPrediction <= 30 && latestPrediction > 15;

  const lineColor = isEmergency
    ? "#EF4444" // emergency red
    : isCritical
    ? "#F59E0B" // critical amber
    : "#00D4AA"; // normal teal

  return (
    <div className="w-full h-80 bg-bg-surface border border-white/5 rounded-xl p-5 flex flex-col justify-between shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="text-sm font-semibold text-text-primary">RUL Projection Timeline</h4>
          <p className="text-[10px] text-text-muted">Estimated Remaining Useful Life vs Actual Cycle</p>
        </div>
        <div className="text-right">
          <span className="text-[10px] text-text-muted block font-medium">Latest Estimate</span>
          <span className="text-lg font-bold font-mono" style={{ color: lineColor }}>
            {latestPrediction.toFixed(1)} <span className="text-xs">cycles</span>
          </span>
        </div>
      </div>

      <div className="flex-1 w-full text-[10px] font-mono">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 5, right: 5, left: -25, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />
            <XAxis
              dataKey="cycle"
              stroke="#6B7280"
              tickLine={false}
              axisLine={false}
              dy={10}
            />
            <YAxis
              stroke="#6B7280"
              tickLine={false}
              axisLine={false}
              domain={[0, "auto"]}
              dx={-5}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const dataPoint = payload[0].payload;
                  return (
                    <div className="bg-bg-elevated border border-white/10 p-3 rounded-lg shadow-xl font-sans text-xs">
                      <p className="font-semibold text-text-primary mb-1">
                        Cycle {dataPoint.cycle}
                      </p>
                      <div className="space-y-1">
                        <div className="flex items-center justify-between gap-6">
                          <span className="text-text-muted">Predicted RUL:</span>
                          <span className="font-mono font-bold text-accent-teal">
                            {Number(dataPoint.predictedRUL).toFixed(1)} cycles
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-6">
                          <span className="text-text-muted">Threshold:</span>
                          <span className="font-mono font-semibold text-accent-amber">
                            {dataPoint.threshold} cycles
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend
              verticalAlign="bottom"
              height={36}
              content={() => (
                <div className="flex justify-center gap-6 text-[10px] text-text-muted font-sans mt-3">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-0.5 rounded" style={{ backgroundColor: lineColor }} />
                    <span>Predicted RUL</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-0.5 border-t border-dashed border-accent-amber" />
                    <span>Threshold ({history[0]?.rul.threshold_cycles ?? 30} cycles)</span>
                  </div>
                </div>
              )}
            />
            <ReferenceLine
              y={history[0]?.rul.threshold_cycles ?? 30}
              stroke="#F59E0B"
              strokeDasharray="5 5"
              label={{
                value: "Alert Limit",
                fill: "#F59E0B",
                fontSize: 9,
                position: "insideBottomLeft",
                offset: 5,
              }}
            />
            <Line
              type="monotone"
              dataKey="predictedRUL"
              stroke={lineColor}
              strokeWidth={2.5}
              dot={data.length < 50 ? { r: 3, strokeWidth: 1 } : false}
              activeDot={{ r: 5, strokeWidth: 0 }}
              animationDuration={500}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
