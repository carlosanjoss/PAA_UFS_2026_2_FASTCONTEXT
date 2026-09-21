import {
  BookOpen,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Cpu,
  Database,
  LoaderCircle,
  MemoryStick,
  MessageSquareText,
  Search,
  Sparkles,
} from "lucide-react";
import { useState, type FormEvent } from "react";
import { api } from "../api";
import { ErrorState } from "../components/EmptyState";
import { KpiCard } from "../components/KpiCard";
import { PageHeader } from "../components/PageHeader";
import type { Algorithm, QueryResponse } from "../types";
import { ALGORITHM_LABELS, formatMs, formatNumber } from "../utils/format";

const strategyDescriptions: Record<Algorithm, string> = {
  linear: "TF-IDF · varredura sequencial · Merge Sort",
  indexed: "índice invertido · busca binária · Merge Sort",
  optimized: "índice invertido · busca binária · heap top-k",
  semantic: "embeddings BGE-small · FAISS IndexFlatIP",
};

const examples = [
  "Como configurar CORS no FastAPI?",
  "Como funciona a injeção de dependências?",
  "Como declarar parâmetros de consulta?",
];

export function QueryPage() {
  const [query, setQuery] = useState(examples[0]);
  const [algorithm, setAlgorithm] = useState<Algorithm>("indexed");
  const [topK, setTopK] = useState(5);
  const [useRag, setUseRag] = useState(true);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!query.trim()) {
      setError("Escreva uma pergunta antes de executar.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setResult(await api.query({ query: query.trim(), algorithm, top_k: topK, use_rag: useRag }));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Não foi possível concluir a consulta.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <PageHeader eyebrow="Consulta interativa" title="Pergunte, veja a resposta e entenda o caminho" description="O resultado aparece primeiro. Depois você pode abrir as métricas, as etapas do algoritmo e cada evidência recuperada." />

      <form className="query-panel" onSubmit={submit}>
        <div className="query-main">
          <label htmlFor="query">Sua pergunta</label>
          <div className="query-input-wrap">
            <Search size={21} />
            <textarea id="query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Pergunte sobre a documentação FastAPI…" rows={2} />
            <button className="primary-button query-submit" disabled={loading} type="submit">
              {loading ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}
              {loading ? "Analisando…" : useRag ? "Gerar resposta" : "Buscar contexto"}
            </button>
          </div>
          <div className="example-row"><span>Experimente:</span>{examples.map((example) => <button type="button" key={example} onClick={() => setQuery(example)}>{example}</button>)}</div>
        </div>

        <div className="query-controls">
          <label>Estratégia<select value={algorithm} onChange={(event) => setAlgorithm(event.target.value as Algorithm)}>{Object.entries(ALGORITHM_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          <label>Quantidade de contextos<select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>{[1, 3, 5, 10].map((value) => <option key={value} value={value}>Top {value}</option>)}</select></label>
          <label className="toggle-label"><span><strong>Resposta com LLM</strong><small>mantém as evidências visíveis</small></span><input type="checkbox" checked={useRag} onChange={(event) => setUseRag(event.target.checked)} /><span className="toggle" /></label>
        </div>
        <div className="strategy-note"><Cpu size={17} /><span><strong>{ALGORITHM_LABELS[algorithm]}</strong> — {strategyDescriptions[algorithm]}</span></div>
      </form>

      {error && <ErrorState message={error} />}
      {result && <QueryResult result={result} />}
      {!result && !error && !loading && (
        <section className="query-empty">
          <div className="query-empty-icon"><MessageSquareText size={28} /></div>
          <h2>A resposta será exibida aqui</h2>
          <p>Execute uma pergunta para comparar tempo, trabalho computacional e relevância dos trechos.</p>
          <div className="learning-steps"><span><b>1</b>Pergunte</span><i /><span><b>2</b>Leia a resposta</span><i /><span><b>3</b>Inspecione as evidências</span></div>
        </section>
      )}
    </>
  );
}

function QueryResult({ result }: { result: QueryResponse }) {
  return (
    <div className="result-stack">
      <section className="answer-card">
        <div className="answer-topline"><span><Sparkles size={17} />Resposta fundamentada</span><span className={`validation-pill ${result.citation_valid === false ? "validation-pill--warn" : ""}`}><CheckCircle2 size={15} />{result.citation_valid === false ? "Citação pendente" : result.mode === "rag" ? "Citações verificadas" : "Modo recuperação"}</span></div>
        {result.answer ? <p className="answer-text">{result.answer}</p> : <p className="answer-placeholder">A geração não foi solicitada. Os contextos recuperados estão disponíveis abaixo.</p>}
        {result.fallback_reason && <div className="fallback-note">A geração não estava disponível; a recuperação foi concluída normalmente. {result.fallback_reason}</div>}
        <div className="answer-meta"><span>{ALGORITHM_LABELS[result.algorithm]}</span><span>Top {result.top_k}</span>{result.provider && <span>{result.provider} · {result.model}</span>}</div>
      </section>

      <section className="kpi-grid kpi-grid--result">
        <KpiCard label="Recuperação" value={formatMs(result.metrics.retrieval_time_ms)} detail="tempo para localizar e ordenar" icon={Clock3} tone="blue" />
        <KpiCard label="Tempo total" value={formatMs(result.metrics.total_time_ms)} detail={result.mode === "rag" ? "recuperação + geração" : "somente recuperação"} icon={Sparkles} tone="coral" />
        <KpiCard label="Comparações" value={formatNumber(result.metrics.comparisons)} detail={result.metrics.comparisons == null ? "não instrumentado nesta estratégia" : "operações de chave medidas"} icon={Cpu} tone="green" />
        <KpiCard label="Memória de pico" value={result.metrics.peak_memory_mb == null ? "N/D" : `${result.metrics.peak_memory_mb.toFixed(2)} MB`} detail="medição disponível nesta execução" icon={MemoryStick} tone="amber" />
      </section>

      <section className="panel trace-panel">
        <div className="panel-heading"><div><h2>Caminho do algoritmo</h2><p>Cada etapa abaixo foi construída a partir dos metadados desta execução.</p></div><span className="panel-badge">execução real</span></div>
        <div className="trace-grid">{result.trace.map((step, index) => <div className="trace-step" key={`${step.title}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><strong>{step.title}</strong><small>{step.detail}</small></div>)}</div>
      </section>

      <section className="panel evidence-panel">
        <div className="panel-heading"><div><h2>Evidências recuperadas</h2><p>Trechos ordenados por relevância; abra um item para ler o conteúdo usado na resposta.</p></div><span className="panel-badge">{result.chunks.length} contextos</span></div>
        <div className="evidence-list">
          {result.chunks.map((chunk) => (
            <details key={chunk.chunk_id} className="evidence-item">
              <summary><span className="rank-badge">#{chunk.rank}</span><span className="evidence-title"><strong>{chunk.section_title}</strong><small>{chunk.source_path}</small></span><span className="score-pill">score {chunk.score.toFixed(4)}</span><ChevronDown size={18} /></summary>
              <div className="evidence-content"><div className="evidence-meta"><span><BookOpen size={15} />{chunk.chunk_id}</span><span><Database size={15} />{chunk.token_count ?? "N/D"} tokens</span></div><p>{chunk.content}</p></div>
            </details>
          ))}
        </div>
      </section>
    </div>
  );
}
