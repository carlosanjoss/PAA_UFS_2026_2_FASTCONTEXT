export type Algorithm = "linear" | "indexed" | "optimized" | "semantic";
export type AppView = "overview" | "ask" | "experiments";

export interface HealthComponent {
  name: string;
  status: "healthy" | "degraded" | "unavailable";
  message: string;
}

export interface HealthResponse {
  status: HealthComponent["status"];
  corpus_size: number | null;
  algorithms: Algorithm[];
  components: HealthComponent[];
}

export interface PerformanceRow {
  algorithm: Algorithm;
  corpus_fraction: number;
  corpus_chunks: number;
  query_count: number;
  retrieval_time_ms_mean: number | null;
  total_time_ms_mean: number | null;
  sorting_time_ms_mean: number | null;
  comparisons_mean: number | null;
  chunks_scored_mean: number | null;
  candidates_found_mean: number | null;
  peak_memory_mb_mean: number | null;
}

export interface QualityRow {
  algorithm: Algorithm;
  corpus_fraction: number;
  corpus_chunks: number;
  k: number;
  query_count: number;
  precision_mean: number | null;
  recall_mean: number | null;
  mrr_mean: number | null;
  hit_rate_mean: number | null;
}

export interface SortingRow {
  algorithm: "merge" | "quick";
  scenario: string;
  item_count: number;
  sample_count: number;
  time_ms_mean: number | null;
  comparisons_mean: number | null;
}

export interface MetricsResponse {
  performance: PerformanceRow[];
  quality: QualityRow[];
  sorting: SortingRow[];
  generated_from: string[];
}

export interface RetrievedChunk {
  rank: number;
  chunk_id: string;
  score: number;
  section_title: string;
  source_path: string;
  token_count: number | null;
  content: string;
}

export interface RuntimeMetrics {
  retrieval_time_ms: number;
  sorting_time_ms: number | null;
  index_build_time_ms: number | null;
  generation_time_ms: number | null;
  total_time_ms: number;
  comparisons: number | null;
  chunks_scored: number | null;
  candidates_found: number | null;
  peak_memory_mb: number | null;
}

export interface TraceStep {
  title: string;
  detail: string;
}

export interface QueryResponse {
  query: string;
  algorithm: Algorithm;
  top_k: number;
  mode: "retrieval" | "rag";
  answer: string | null;
  provider: string | null;
  model: string | null;
  citations: string[];
  citation_valid: boolean | null;
  fallback_reason: string | null;
  metrics: RuntimeMetrics;
  trace: TraceStep[];
  chunks: RetrievedChunk[];
}

export interface QueryRequest {
  query: string;
  algorithm: Algorithm;
  top_k: number;
  use_rag: boolean;
}
