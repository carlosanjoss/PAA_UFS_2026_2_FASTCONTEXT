import type { Algorithm, PerformanceRow, QualityRow } from "../types";

export const ALGORITHM_LABELS: Record<Algorithm, string> = {
  linear: "Linear",
  indexed: "Indexado",
  optimized: "Otimizado",
  semantic: "Semântico",
};

export const ALGORITHM_COLORS: Record<Algorithm, string> = {
  linear: "#e11d48",
  indexed: "#3b82f6",
  optimized: "#10b981",
  semantic: "#8b5cf6",
};

export function formatMs(value: number | null | undefined): string {
  if (value == null) return "N/D";
  if (value < 1) return `${value.toLocaleString("pt-BR", { maximumFractionDigits: 3 })} ms`;
  return `${value.toLocaleString("pt-BR", { maximumFractionDigits: 2 })} ms`;
}

export function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value == null) return "N/D";
  return value.toLocaleString("pt-BR", { maximumFractionDigits: digits });
}

export function formatPercent(value: number | null | undefined): string {
  if (value == null) return "N/D";
  return `${(value * 100).toLocaleString("pt-BR", { maximumFractionDigits: 1 })}%`;
}

export function fullCorpusPerformance(rows: PerformanceRow[]): PerformanceRow[] {
  const maxFraction = Math.max(...rows.map((row) => row.corpus_fraction), 0);
  return rows.filter((row) => row.corpus_fraction === maxFraction);
}

export function fullCorpusQuality(rows: QualityRow[], k: number): QualityRow[] {
  const maxFraction = Math.max(...rows.map((row) => row.corpus_fraction), 0);
  return rows.filter((row) => row.corpus_fraction === maxFraction && row.k === k);
}
