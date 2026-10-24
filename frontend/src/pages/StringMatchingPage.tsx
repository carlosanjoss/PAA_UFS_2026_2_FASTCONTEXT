import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Files,
  Gauge,
  Hash,
  LoaderCircle,
  Search,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api";
import { ErrorState } from "../components/EmptyState";
import { KpiCard } from "../components/KpiCard";
import { PageHeader } from "../components/PageHeader";
import type {
  StringMatchingAlgorithm,
  StringMatchingChunk,
  StringMatchingResponse,
} from "../types";

const fallbackAlgorithms: StringMatchingAlgorithm[] = ["naive", "kmp", "boyer-moore", "rabin-karp"];
const examples = ["OAuth2", "WebSocket", "dependency"];
const labels: Record<StringMatchingAlgorithm, string> = {
  naive: "Naive",
  kmp: "KMP",
  "boyer-moore": "Boyer-Moore",
  "rabin-karp": "Rabin-Karp",
};

export function StringMatchingPage() {
  const [pattern, setPattern] = useState("OAuth2");
  const [algorithm, setAlgorithm] = useState<StringMatchingAlgorithm>("boyer-moore");
  const [algorithms, setAlgorithms] = useState<StringMatchingAlgorithm[]>(fallbackAlgorithms);
  const [maxResults, setMaxResults] = useState(10);
  const [compareAll, setCompareAll] = useState(false);
  const [results, setResults] = useState<StringMatchingResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.stringMatchingAlgorithms()
      .then((response) => setAlgorithms(response.algorithms))
      .catch((requestError: unknown) => {
        setError(requestError instanceof Error ? requestError.message : "Não foi possível listar os algoritmos da AV2.");
      });
  }, []);

  const primary = results.find((result) => result.algorithm === algorithm) ?? results[0] ?? null;
  const equivalent = useMemo(() => results.length > 1 ? haveEquivalentPositions(results) : null, [results]);

  const runSearch = async () => {
    const normalizedPattern = pattern.trim();
    if (!normalizedPattern) {
      setError("Digite um padrão não vazio.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const selected = compareAll ? algorithms : [algorithm];
      const responses = await Promise.all(selected.map((name) => api.stringMatchingSearch({
        pattern: normalizedPattern,
        algorithm: name,
        max_results: maxResults,
      })));
      setResults(responses);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "A busca textual não pôde ser concluída.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <PageHeader
        eyebrow="Processamento de cadeias"
        title="Busca textual — AV2"
        description="Encontre ocorrências exatas no corpus FastAPI preservando maiúsculas, minúsculas e sobreposições."
      />

      <section className="query-panel av2-search-panel">
        <div className="query-main">
          <label htmlFor="string-pattern">Padrão exato</label>
          <div className="query-input-wrap">
            <Search size={20} />
            <input
              id="string-pattern"
              value={pattern}
              onChange={(event) => setPattern(event.target.value)}
              onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); void runSearch(); } }}
              placeholder="Ex.: OAuth2"
            />
            <button className="primary-button query-submit" onClick={() => void runSearch()} disabled={loading}>
              {loading ? <LoaderCircle className="spin" size={17} /> : <Search size={17} />}
              {loading ? "Analisando…" : "Buscar no corpus"}
            </button>
          </div>
          <div className="example-row"><span>Exemplos:</span>{examples.map((example) => <button key={example} onClick={() => setPattern(example)}>{example}</button>)}</div>
        </div>

        <div className="query-controls av2-query-controls">
          <label>Algoritmo<select value={algorithm} onChange={(event) => setAlgorithm(event.target.value as StringMatchingAlgorithm)} disabled={compareAll}>{algorithms.map((name) => <option value={name} key={name}>{labels[name]}</option>)}</select></label>
          <label>Chunks exibidos<select value={maxResults} onChange={(event) => setMaxResults(Number(event.target.value))}>{[5, 10, 20, 50].map((value) => <option value={value} key={value}>{value}</option>)}</select></label>
          <label className="toggle-label"><span><strong>Comparar os quatro</strong><small>execução interativa, fora do benchmark</small></span><input type="checkbox" checked={compareAll} onChange={(event) => setCompareAll(event.target.checked)} /><span className="toggle" /></label>
        </div>
        <div className="strategy-note"><AlertTriangle size={17} /><span>O limite controla somente os cards retornados. Totais de chunks, ocorrências e comparações são calculados sobre todo o corpus.</span></div>
      </section>

      {error && <ErrorState message={error} />}

      {primary && (
        <div className="result-stack av2-result-stack">
          <section className="kpi-grid kpi-grid--result">
            <KpiCard label="Algoritmo" value={labels[primary.algorithm]} detail="casamento exato e sensível a caixa" icon={Gauge} tone="coral" />
            <KpiCard label="Chunks analisados" value={primary.total_chunks.toLocaleString("pt-BR")} detail={`${primary.matching_chunks.toLocaleString("pt-BR")} com correspondências`} icon={Files} tone="blue" />
            <KpiCard label="Ocorrências" value={primary.total_occurrences.toLocaleString("pt-BR")} detail={`${primary.returned_chunks} chunks exibidos`} icon={Hash} tone="green" />
            <KpiCard label="Comparações" value={primary.total_comparisons.toLocaleString("pt-BR")} detail="comparações diretas de caracteres" icon={Clock3} tone="amber" />
          </section>

          <section className="panel runtime-breakdown">
            <div><span>Pré-processamento</span><strong>{formatNs(primary.preprocessing_time_ns)}</strong></div>
            <div><span>Busca</span><strong>{formatNs(primary.search_time_ns)}</strong></div>
            <div><span>Tempo total externo</span><strong>{formatNs(primary.total_time_ns)}</strong></div>
            <div><span>Padrão</span><strong className="mono-value">{primary.pattern}</strong></div>
          </section>

          {results.length > 1 && <InteractiveComparison results={results} equivalent={Boolean(equivalent)} />}

          <section className="panel evidence-panel">
            <div className="panel-heading"><div><h2>Chunks encontrados</h2><p>As posições são offsets Python no conteúdo original; o destaque converte o texto por pontos de código Unicode.</p></div><span className="panel-badge">{primary.returned_chunks} de {primary.matching_chunks}</span></div>
            {primary.matches.length ? <div className="string-match-list">{primary.matches.map((match) => <MatchCard match={match} pattern={primary.pattern} key={match.chunk_id} />)}</div> : <div className="chart-unavailable">Nenhuma ocorrência exata foi encontrada.</div>}
          </section>
        </div>
      )}
    </>
  );
}

function InteractiveComparison({ results, equivalent }: { results: StringMatchingResponse[]; equivalent: boolean }) {
  return <section className="panel comparison-table-panel"><div className="panel-heading"><div><h2>Comparação interativa</h2><p>Estas medidas vieram desta interação e não substituem o benchmark controlado.</p></div><span className={`validation-pill ${equivalent ? "" : "validation-pill--warn"}`}>{equivalent ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}{equivalent ? "Posições idênticas" : "Divergência detectada"}</span></div><div className="table-scroll"><table><thead><tr><th>Algoritmo</th><th>Ocorrências</th><th>Chunks</th><th>Comparações</th><th>Pré-processamento</th><th>Busca</th><th>Total</th></tr></thead><tbody>{results.map((result) => <tr key={result.algorithm}><td><strong>{labels[result.algorithm]}</strong></td><td>{result.total_occurrences.toLocaleString("pt-BR")}</td><td>{result.matching_chunks.toLocaleString("pt-BR")}</td><td>{result.total_comparisons.toLocaleString("pt-BR")}</td><td>{formatNs(result.preprocessing_time_ns)}</td><td>{formatNs(result.search_time_ns)}</td><td>{formatNs(result.total_time_ns)}</td></tr>)}</tbody></table></div></section>;
}

function MatchCard({ match, pattern }: { match: StringMatchingChunk; pattern: string }) {
  return <details className="string-match-card"><summary><span className="rank-badge">{match.occurrences}</span><div className="evidence-title"><strong>{match.section_title}</strong><small>{match.source_path} · {match.chunk_id}</small></div><span className="score-pill">{match.positions.length} posição(ões)</span><ChevronDown size={18} /></summary><div className="string-match-content"><div className="position-list"><strong>Offsets:</strong> {match.positions.join(", ")}</div><pre><HighlightedText content={match.content} pattern={pattern} positions={match.positions} /></pre></div></details>;
}

function HighlightedText({ content, pattern, positions }: { content: string; pattern: string; positions: number[] }) {
  const codePoints = Array.from(content);
  const mask = new Array<boolean>(codePoints.length).fill(false);
  const patternLength = Array.from(pattern).length;
  positions.forEach((position) => {
    for (let index = position; index < Math.min(position + patternLength, mask.length); index += 1) mask[index] = true;
  });
  const parts: ReactNode[] = [];
  let start = 0;
  while (start < codePoints.length) {
    const highlighted = mask[start];
    let end = start + 1;
    while (end < codePoints.length && mask[end] === highlighted) end += 1;
    const text = codePoints.slice(start, end).join("");
    parts.push(highlighted ? <mark key={start}>{text}</mark> : <span key={start}>{text}</span>);
    start = end;
  }
  return <>{parts}</>;
}

function haveEquivalentPositions(results: StringMatchingResponse[]): boolean {
  const canonical = (result: StringMatchingResponse) => JSON.stringify(result.matches.map((match) => [match.chunk_id, match.positions]));
  return results.every((result) => result.total_occurrences === results[0].total_occurrences && canonical(result) === canonical(results[0]));
}

function formatNs(value: number): string {
  return `${(value / 1_000_000).toLocaleString("pt-BR", { minimumFractionDigits: 3, maximumFractionDigits: 3 })} ms`;
}
