// frontend/components/organisms/RAGResponseCard.tsx
"use client";

import React from "react";
import type { RAGQueryResponse } from "@/types";
import { Badge } from "@/components/atoms/Badge";
import { ShieldCheck, ShieldAlert, Clock, BarChart3, BookMarked } from "lucide-react";

interface RAGResponseCardProps {
  response: RAGQueryResponse;
}

export function RAGResponseCard({ response }: RAGResponseCardProps) {
  const { answer, abstained, relevance_score, threshold_used, sources = [], latency_ms } = response;

  const scorePercent = Math.round(relevance_score * 100);
  const threshPercent = Math.round(threshold_used * 100);
  
  // Dynamic color for relevance score
  const scoreColor = relevance_score >= threshold_used
    ? "text-accent-teal"
    : "text-accent-red";

  return (
    <div className="space-y-6">
      {/* Response Status Card */}
      <div className="card p-6 border-white/5 space-y-4 shadow-md">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-white/5 pb-4 gap-4">
          <div className="flex items-center gap-2">
            {abstained ? (
              <ShieldAlert className="h-5 w-5 text-accent-red animate-pulse" />
            ) : (
              <ShieldCheck className="h-5 w-5 text-accent-teal" />
            )}
            <h3 className="text-sm font-bold text-text-primary">
              {abstained ? "Guardrail Triggered" : "Engineered Response"}
            </h3>
            <Badge variant={abstained ? "red" : "teal"}>
              {abstained ? "ABSTAINED" : "NOMINAL"}
            </Badge>
          </div>

          <div className="flex flex-wrap items-center gap-4 text-[10px] font-mono text-text-muted">
            <span className="flex items-center gap-1">
              <Clock className="h-3.5 w-3.5" />
              Latency: <span className="text-text-primary font-bold">{latency_ms ?? "—"} ms</span>
            </span>
            <span className="flex items-center gap-1">
              <BarChart3 className="h-3.5 w-3.5" />
              Relevance: <span className={`font-bold ${scoreColor}`}>{scorePercent}%</span> (Threshold: {threshPercent}%)
            </span>
          </div>
        </div>

        {/* Abstain Warning */}
        {abstained ? (
          <div className="bg-accent-red/10 border border-accent-red/20 rounded-xl p-4 text-xs text-text-secondary leading-relaxed">
            <span className="font-bold text-accent-red block mb-1">Safety Guardrail Active:</span>
            The retrieval relevance score ({scorePercent}%) was below the system guardrail threshold ({threshPercent}%). 
            To prevent generation of hallucinated, incorrect, or unsafe maintenance advice, the system has abstained from answering this query.
            {response.abstain_reason && (
              <p className="mt-2 text-text-muted italic">Reason: {response.abstain_reason}</p>
            )}
          </div>
        ) : (
          /* Actual Answer */
          <div className="text-xs leading-relaxed text-text-secondary whitespace-pre-wrap font-sans">
            {answer}
          </div>
        )}
      </div>

      {/* Sources / Citations */}
      {!abstained && sources.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <BookMarked className="h-4 w-4 text-accent-blue" />
            <h4 className="text-xs font-bold text-text-primary uppercase tracking-wider">Citations &amp; Reference Material</h4>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {sources.map((source, index) => (
              <div
                key={source.doc_id || index}
                className="card p-4 border-white/5 hover:border-white/10 bg-white/[0.01] flex flex-col justify-between gap-3 transition-all"
              >
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="font-bold text-accent-blue truncate max-w-[70%]">
                      {source.doc_id}
                    </span>
                    <span className="font-mono text-text-muted">
                      Match: <span className="text-text-primary font-bold">{Math.round(source.relevance_score * 100)}%</span>
                    </span>
                  </div>
                  <p className="text-[11px] text-text-muted leading-relaxed line-clamp-4 italic">
                    &ldquo;{source.text}&rdquo;
                  </p>
                </div>
                
                {source.source_file && (
                  <div className="border-t border-white/5 pt-2 flex items-center justify-between text-[9px] text-text-muted font-mono">
                    <span className="truncate max-w-[80%]">Ref: {source.source_file}</span>
                    {source.page !== undefined && <span>Page {source.page}</span>}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
