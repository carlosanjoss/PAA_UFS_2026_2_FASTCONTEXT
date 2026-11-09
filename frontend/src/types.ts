export type Algorithm = "linear" | "indexed" | "optimized" | "semantic";
export type StringMatchingAlgorithm = "naive" | "kmp" | "boyer-moore" | "rabin-karp";
export type AppView = "overview" | "ask" | "experiments" | "string-matching" | "av2-experiments" | "rag-models";

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
  retrieval_measurements: number;
  retrieval_combinations: number;
  rag_generations: number;
  human_evaluation_ratings: number;
  generated_from: string[];
}

export type AV2RetrievalCondition = "semantic" | "literal" | "hybrid";

export interface AV2RetrievalConditionRow {
  condition: AV2RetrievalCondition;
  queries: number;
  repetitions_per_query: number;
  mean_time_ms: number;
  pooled_precision_at_k: number;
  pooled_recall_at_k: number;
  known_mrr_at_k: number;
  known_hit_rate_at_k: number;
  mean_precision_lower_bound: number;
  mean_precision_upper_bound: number;
  unjudged_results: number;
  queries_with_unjudged_results: number;
}

export interface AV2RetrievalQueryRow {
  query_id: string;
  category: string;
  condition: AV2RetrievalCondition;
  repetitions: number;
  mean_time_ms: number;
  std_time_ms: number;
  pooled_precision_at_k: number;
  pooled_recall_at_k: number;
  known_mrr_at_k: number;
  known_hit_rate_at_k: number;
  precision_lower_bound: number;
  precision_upper_bound: number;
  unjudged_count: number;
  judged_count: number;
  known_positive_count: number;
}

export interface AV2RAGConditionRow {
  condition: AV2RetrievalCondition;
  generations: number;
  mean_generation_time_ms: number;
  std_generation_time_ms: number;
  citation_valid_count: number;
  citation_valid_rate: number;
  mean_citation_count: number;
  mean_known_relevant_citations: number;
  citation_retry_count: number;
}

export interface AV2RAGQueryRow {
  query_id: string;
  category: string;
  condition: AV2RetrievalCondition;
  generation_time_ms: number;
  citation_valid: boolean;
  citation_count: number;
  citation_retry_count: number;
  known_relevant_citations: number;
  answer_length_chars: number;
}

export interface AV2HumanConditionRow {
  condition: AV2RetrievalCondition;
  responses: number;
  quality_ratings: number;
  groundedness_ratings: number;
  mean_correctness: number;
  mean_completeness: number;
  mean_clarity: number;
  hallucination_rate: number;
  mean_groundedness: number;
}

export interface AV2HumanAgreementRow {
  metric: string;
  exact_agreement_rate: number;
  mean_absolute_difference: number;
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
  retrieval: {
    metadata: {
      query_count: number;
      top_k: number;
      candidate_depth: number;
      repetitions: number;
      warmup: number;
      raw_measurements: number;
      conditions: AV2RetrievalCondition[];
      semantic_index_source: string;
      quality_interpretation: string;
      latency_interpretation: string;
    };
    conditions: AV2RetrievalConditionRow[];
    queries: AV2RetrievalQueryRow[];
  };
  rag: {
    metadata: {
      query_count: number;
      generation_count: number;
      model: string;
      top_k: number;
      temperature: number;
      seed: number;
      conditions: AV2RetrievalCondition[];
      evaluation_note: string;
    };
    conditions: AV2RAGConditionRow[];
    queries: AV2RAGQueryRow[];
  };
  human_evaluation: {
    status: "complete";
    scale_min: number;
    scale_max: number;
    quality_ratings: number;
    groundedness_ratings: number;
    effective_quality_judgments: number;
    effective_groundedness_judgments: number;
    unique_responses: number;
    ratings_per_response: number;
    duplicate_score_pairs: number;
    duplicate_note_pairs: number;
    independent_ratings_confirmed: boolean;
    conditions: AV2HumanConditionRow[];
    agreement: AV2HumanAgreementRow[];
    source_archive_sha256: string;
    methodology_note: string;
  };
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

export type RAGModelRole = "baseline" | "candidate";
export type RAGCondition = "no_rag" | "optimized" | "semantic";
export type RAGHumanCriterion = "correctness" | "completeness" | "clarity" | "groundedness" | "hallucination_present";

export interface RAGModelDescriptor {
  role: RAGModelRole;
  model: string;
  parameter_size: string;
  quantization: string;
  context_length: number;
}

export interface RAGAutomaticMetricRow {
  role: RAGModelRole;
  model: string;
  condition: RAGCondition;
  planned: number;
  success: number;
  truncated: number;
  success_rate: number;
  generation_mean_ms: number;
  generation_median_ms: number;
  citation_accepted_n: number | null;
  citation_compliance_rate: number | null;
  with_valid_citation_n: number | null;
  valid_citation_presence_rate: number | null;
  citation_retry_mean: number | null;
}

export interface RAGHumanMetricRow {
  role: RAGModelRole;
  model: string;
  condition: RAGCondition;
  criterion: RAGHumanCriterion;
  responses: number;
  mean_score: number;
  exact_agreement_n: number;
  exact_agreement_rate: number;
  mean_absolute_disagreement: number;
  discordant_items: number;
}

export interface RAGPairedHumanMetricRow {
  condition: RAGCondition;
  criterion: RAGHumanCriterion;
  planned_pairs: number;
  both_complete: number;
  missing_pair: number;
  mean_candidate_minus_baseline: number;
  candidate_better: number;
  baseline_better: number;
  ties: number;
}

export interface RAGAgreementMetricRow {
  dimension: "qualidade" | "fundamentacao";
  criterion: RAGHumanCriterion;
  item_pairs: number;
  exact_agreement_n: number;
  exact_agreement_rate: number;
  mean_absolute_difference: number;
}

export interface RAGModelComparisonResponse {
  metadata: {
    attempts: number;
    complete_answers: number;
    truncated_answers: number;
    frozen_prompts: number;
    human_judgments: number;
    valid_criterion_comparisons: number;
    planned_criterion_comparisons: number;
    missing_criterion_comparisons: number;
    reviewers: number;
    descriptive_only: boolean;
    generated_from: string[];
  };
  models: RAGModelDescriptor[];
  conditions: RAGCondition[];
  retrieval: {
    model: string;
    embedding_dimension: number;
    normalized: boolean;
    index_type: string;
    top_k: number;
    reference: string;
    reference_url: string;
    faiss_reference_url: string;
  };
  generation: {
    provider: string;
    temperature: number;
    initial_max_tokens: number;
    max_retry_tokens: number;
    max_truncation_retries: number;
    seed: number;
    think: boolean;
  };
  automatic_metrics: RAGAutomaticMetricRow[];
  human_metrics: RAGHumanMetricRow[];
  paired_human_metrics: RAGPairedHumanMetricRow[];
  agreement_metrics: RAGAgreementMetricRow[];
  automatic_methodology_note: string;
  human_methodology_note: string;
}
