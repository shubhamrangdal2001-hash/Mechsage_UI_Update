// frontend/components/organisms/AnomalyChart.tsx
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

interface AnomalyChartProps {
  history: CyclePayload[];
}

export function AnomalyChart({ history }: AnomalyChartProps) {
  const data = history.map((item) => ({
    cycle: item.cycle,
    score: item.anomaly.score,
    threshold: item.anomaly.threshold,
    modelType: item.anomaly.model_type,
  }));

  const latestScore = history[history.length - 1]?.anomaly.score ?? 0;
  const threshold = history[0]?.anomaly.threshold ?? 0.52;
  const isAnomaly = latestScore >= threshold;
  const modelType = history[0]?.anomaly.model_type ?? "IsolationForest";

  const lineColor = isAnomaly ? "#EF4444" : "#3B82F6"; // Red for anomaly, blue for normal

  return (
    <div className="w-full h-80 bg-bg-surface border border-white/5 rounded-xl p-5 flex flex-col justify-between shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="text-sm font-semibold text-text-primary">Anomaly Score Profile</h4>
          <p className="text-[10px] text-text-muted">Algorithm: <span className="font-semibold text-text-secondary">{modelType}</span></p>
        </div>
        <div className="text-right">
          <span className="text-[10px] text-text-muted block font-medium">Latest Score</span>
          <span className="text-lg font-bold font-mono" style={{ color: lineColor }}>
            {latestScore.toFixed(3)}
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
              domain={[0, 1]}
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
                          <span className="text-text-muted">Anomaly Score:</span>
                          <span className="font-mono font-bold text-accent-blue">
                            {Number(dataPoint.score).toFixed(3)}
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-6">
                          <span className="text-text-muted">Threshold:</span>
                          <span className="font-mono font-semibold text-accent-red">
                            {dataPoint.threshold.toFixed(3)}
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
                    <span>Anomaly Score</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-0.5 border-t border-dashed border-accent-red" />
                    <span>Threshold ({threshold.toFixed(2)})</span>
                  </div>
                </div>
              )}
            />
            <ReferenceLine
              y={threshold}
              stroke="#EF4444"
              strokeDasharray="5 5"
              label={{
                value: "Anomaly Threshold",
                fill: "#EF4444",
                fontSize: 9,
                position: "insideBottomLeft",
                offset: 5,
              }}
            />
            <Line
              type="monotone"
              dataKey="score"
              stroke={lineColor}
              strokeWidth={2}
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
