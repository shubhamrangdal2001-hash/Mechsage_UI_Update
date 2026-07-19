// frontend/app/rag/page.tsx
"use client";

import React, { useState } from "react";
import { useRAGQuery } from "@/hooks/useFleetData";
import { RAGResponseCard } from "@/components/organisms/RAGResponseCard";
import { DriftBarChart } from "@/components/organisms/DriftBarChart";
import { Spinner } from "@/components/atoms/Spinner";
import { Search, Sliders, BookOpen, Sparkles, HelpCircle } from "lucide-react";
import type { RAGQueryResponse } from "@/types";

const SUGGESTED_QUERIES = [
  "What is the bearing-wear procedure for ISM-CNC-001 when s11 vibration rises?",
  "How should ISM-HYD-002 respond to hydraulic pump cavitation and unstable s7 pressure?",
  "What action is required for gear pitting on ISM-GBX-003?",
  "Which spares are required for ISM-CMR-004 air-end bearing fatigue?",
];

export default function RAGPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(3);
  const [datasetContext, setDatasetContext] = useState("");
  const [response, setResponse] = useState<RAGQueryResponse | null>(null);

  const ragMutation = useRAGQuery();

  const handleSearch = (e?: React.FormEvent, queryOverride?: string) => {
    if (e) e.preventDefault();
    const searchQuery = queryOverride ?? query;
    if (!searchQuery.trim()) return;

    ragMutation.mutate(
      {
        query: searchQuery,
        top_k: topK,
        dataset_context: datasetContext || undefined,
      },
      {
        onSuccess: (data) => {
          setResponse(data);
        },
      }
    );
  };

  const handleSuggestedClick = (suggested: string) => {
    setQuery(suggested);
    handleSearch(undefined, suggested);
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-text-primary flex items-center gap-2">
          RAG Search Console
        </h1>
        <p className="text-sm text-text-muted mt-1">
          Query maintenance manuals, technical guidelines, and historical logs with safety guardrails.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Search & Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {/* Query Input Card */}
          <div className="card p-6">
            <form onSubmit={handleSearch} className="space-y-4">
              <div className="relative">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter manual query e.g., sensor 11 degradation checklist..."
                  className="w-full bg-bg-surface border border-white/10 rounded-xl pl-11 pr-24 py-3 text-xs text-text-primary focus:outline-none focus:border-accent-blue"
                />
                <Search className="absolute left-4 top-3.5 h-4 w-4 text-text-muted" />
                <button
                  type="submit"
                  disabled={ragMutation.isPending || !query.trim()}
                  className="absolute right-2 top-2 px-4 py-1.5 bg-accent-blue hover:bg-accent-blue/90 disabled:opacity-50 text-text-primary text-[10px] font-bold rounded-lg transition-all flex items-center gap-1.5 cursor-pointer"
                >
                  {ragMutation.isPending ? (
                    <Spinner size="sm" className="border-text-primary" />
                  ) : (
                    <Sparkles className="h-3 w-3" />
                  )}
                  Query
                </button>
              </div>

              {/* Advanced Parameters */}
              <div className="flex flex-wrap items-center gap-6 border-t border-white/5 pt-4">
                <div className="flex items-center gap-2 text-[10px] font-semibold text-text-muted uppercase">
                  <Sliders className="h-3.5 w-3.5" />
                  <span>Retrieve parameters:</span>
                </div>

                {/* Top K */}
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-text-muted">Top K documents:</span>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                    className="w-12 bg-bg-surface border border-white/10 rounded px-1.5 py-0.5 text-center font-mono focus:outline-none focus:border-accent-blue text-xs text-text-primary"
                  />
                </div>

                {/* Dataset context */}
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-text-muted">Domain context:</span>
                  <select
                    value={datasetContext}
                    onChange={(e) => setDatasetContext(e.target.value)}
                    className="bg-bg-surface border border-white/10 rounded px-2 py-0.5 focus:outline-none focus:border-accent-blue text-xs text-text-primary cursor-pointer"
                  >
                    <option value="">None (Global)</option>
                    <option value="ISM-CNC-001">ISM-CNC-001 (CNC Machining Center)</option>
                    <option value="ISM-HYD-002">ISM-HYD-002 (Hydraulic Press)</option>
                    <option value="ISM-GBX-003">ISM-GBX-003 (Helical Gearbox)</option>
                    <option value="ISM-CMR-004">ISM-CMR-004 (Screw Compressor)</option>
                  </select>
                </div>
              </div>
            </form>
          </div>

          {/* Response Mount */}
          {ragMutation.isPending && (
            <div className="card p-12 text-center flex flex-col items-center justify-center gap-3">
              <Spinner size="lg" />
              <span className="text-xs text-text-muted">Scanning manuals database and computing similarity metrics...</span>
            </div>
          )}

          {!ragMutation.isPending && response && (
            <RAGResponseCard response={response} />
          )}

          {!ragMutation.isPending && !response && (
            <div className="card p-12 text-center text-text-muted">
              <BookOpen className="h-12 w-12 mx-auto mb-4 text-slate-600" />
              <h4 className="text-sm font-semibold text-text-secondary">Awaiting Search Input</h4>
              <p className="text-xs mt-1">Submit a search query above or select one from the quick prompts.</p>
            </div>
          )}
        </div>

        {/* Sidebar Suggestions & Drift Charts */}
        <div className="lg:col-span-1 space-y-6">
          {/* Quick suggestions */}
          <div className="card p-6 space-y-4">
            <h4 className="text-xs font-bold text-text-primary uppercase tracking-wider flex items-center gap-1.5">
              <HelpCircle className="h-4 w-4 text-accent-teal" /> Suggested Queries
            </h4>
            <div className="space-y-2 text-xs">
              {SUGGESTED_QUERIES.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSuggestedClick(q)}
                  className="w-full text-left p-2.5 bg-white/[0.01] hover:bg-white/5 border border-white/5 hover:border-white/10 rounded-lg transition-all text-text-secondary leading-relaxed cursor-pointer block font-medium"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>

          {/* Drift Chart */}
          <DriftBarChart />
        </div>
      </div>
    </div>
  );
}
