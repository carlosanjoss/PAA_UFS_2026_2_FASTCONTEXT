export type Algorithm = "linear" | "indexed" | "optimized" | "semantic";
export type StringMatchingAlgorithm = "naive" | "kmp" | "boyer-moore" | "rabin-karp";
export type AppView = "overview" | "ask" | "experiments" | "string-matching" | "av2-experiments";

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
  temperature?: number;
}

export interface StringMatchingAlgorithmsResponse {
  algorithms: StringMatchingAlgorithm[];
  corpus_size: number;
}

export interface StringMatchingChunk {
  chunk_id: string;
  source_path: string;
  section_title: string;
  content: string;
  positions: number[];
  occurrences: number;
  comparisons: number;
}

export interface StringMatchingResponse {
  algorithm: StringMatchingAlgorithm;
  pattern: string;
  total_chunks: number;
  matching_chunks: number;
  total_occurrences: number;
  total_comparisons: number;
  preprocessing_time_ns: number;
  search_time_ns: number;
  total_time_ns: number;
  returned_chunks: number;
  matches: StringMatchingChunk[];
  llm?: StringMatchingLLMResult;
}

export interface StringMatchingRequest {
  pattern: string;
  algorithm: StringMatchingAlgorithm;
  max_results: number;
  use_llm: boolean;
  question?: string;
  llm_context_limit: number;
  temperature?: number;
}

export interface StringMatchingLLMResult {
  status: "disabled" | "generated" | "unavailable" | "no_context" | "error";
  question: string | null;
  answer: string | null;
  provider: string | null;
  model: string | null;
  citations: string[];
  citation_valid: boolean | null;
  generation_time_ns: number | null;
  context_chunk_ids: string[];
  fallback_reason: string | null;
}

export interface AV2ExperimentMetadata {
  algorithm_count: number;
  pattern_count: number;
  scenario_count: number;
  timing_measurements: number;
  timing_combinations: number;
  memory_combinations: number;
  corpus_chunks: number;
  repetitions: number;
  generated_from: string[];
}

export interface AV2Pattern {
  pattern_id: string;
  pattern: string;
  category: string;
  description: string;
  scope: "corpus" | "synthetic";
  corpus_occurrences: number;
}

export interface AV2Scenario {
  scenario_id: string;
  pattern_id: string;
  pattern: string;
  pattern_category: string;
  scale: string;
  corpus_fraction: number;
  total_characters: number;
  expected_occurrences: number;
  description: string;
}

export interface AV2SummaryRow {
  scenario_id: string;
  pattern_id: string;
  pattern_category: string;
  scale: string;
  corpus_fraction: number;
  algorithm: StringMatchingAlgorithm;
  repetitions: number;
  input_count: number;
  total_characters: number;
  occurrences: number;
  mean_time_ms: number;
  median_time_ms: number;
  std_time_ms: number;
  min_time_ms: number;
  max_time_ms: number;
  mean_comparisons: number;
  mean_preprocessing_ms: number;
  mean_search_ms: number;
}

export interface AV2MemoryRow {
  scenario_id: string;
  pattern_id: string;
  pattern_category: string;
  scale: string;
  corpus_fraction: number;
  algorithm: StringMatchingAlgorithm;
  total_characters: number;
  peak_memory_mib: number;
}

export interface AV2SpeedupRow {
  scenario_id: string;
  pattern_id: string;
  pattern_category: string;
  scale: string;
  corpus_fraction: number;
  algorithm: StringMatchingAlgorithm;
  repetitions: number;
  total_characters: number;
  occurrences: number;
  mean_time_ms: number;
  std_time_ms: number;
  speedup_vs_naive: number;
}

export interface AV2ExperimentsResponse {
  metadata: AV2ExperimentMetadata;
  patterns: AV2Pattern[];
  scenarios: AV2Scenario[];
  summary: AV2SummaryRow[];
  memory: AV2MemoryRow[];
  speedups: AV2SpeedupRow[];
  environment: {
    project: string;
    captured_at: string;
    python: Record<string, string | boolean>;
    system: Record<string, string | number>;
    packages: Record<string, string>;
    experimental_matrix: Record<string, unknown>;
    notes: string[];
  };
}
