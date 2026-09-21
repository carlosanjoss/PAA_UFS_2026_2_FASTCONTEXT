import { Filter, Info, Scale } from "lucide-react";
import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartCard } from "../components/ChartCard";
import { ErrorState, LoadingState } from "../components/EmptyState";
import { PageHeader } from "../components/PageHeader";
import type { MetricsResponse, PerformanceRow, QualityRow } from "../types";
import { ALGORITHM_COLORS, ALGORITHM_LABELS, formatMs, formatNumber, formatPercent } from "../utils/format";

interface ExperimentsPageProps {
  metrics: MetricsResponse | null;
  error: string | null;
}

export function ExperimentsPage({ metrics, error }: ExperimentsPageProps) {
  const [fraction, setFraction] = useState(1);
  const [topK, setTopK] = useState(5);
  const [scenario, setScenario] = useState("random");

  const fractions = useMemo(() => metrics ? [...new Set(metrics.performance.map((row) => row.corpus_fraction))].sort((a, b) => a - b) : [], [metrics]);
  const performance = metrics?.performance.filter((row) => row.corpus_fraction === fraction) ?? [];
  const quality = metrics?.quality.filter((row) => row.corpus_fraction === fraction && row.k === topK) ?? [];
  const sorting = metrics?.sorting.filter((row) => row.scenario === scenario) ?? [];

  return (
    <>
      <PageHeader eyebrow="Comparações experimentais" title="Cada métrica no contexto certo" description="Explore eficiência, qualidade de recuperação e comportamento da ordenação sem misturar conclusões de experimentos diferentes." />

      {error && <ErrorState message={error} />}
      {!metrics && !error && <LoadingState />}
      {metrics && (
        <>
          <section className="filter-bar">
            <div className="filter-title"><Filter size={18} /><span><strong>Recorte do experimento</strong><small>os gráficos abaixo usam o mesmo corpus e top-k</small></span></div>
            <label>Corpus<select value={fraction} onChange={(event) => setFraction(Number(event.target.value))}>{fractions.map((value) => <option key={value} value={value}>{Math.round(value * 100)}% · {metrics.performance.find((row) => row.corpus_fraction === value)?.corpus_chunks.toLocaleString("pt-BR")} trechos</option>)}</select></label>
            <label>Profundidade<select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>{[1, 3, 5, 10].map((value) => <option key={value} value={value}>Top {value}</option>)}</select></label>
          </section>

          <section className="section-intro"><div className="section-index">01</div><div><span>Eficiência</span><h2>Quanto custa recuperar o contexto?</h2><p>Tempo, comparações e volume processado medem trabalho computacional. Valores menores são preferíveis.</p></div></section>
          <section className="chart-grid">
            <MetricBarChart rows={performance} metric="total_time_ms_mean" title="Tempo total médio" description="Custo completo da recuperação, incluindo descoberta e ranking." badge="Menor é melhor" formatter={formatMs} />
            <MetricBarChart rows={performance} metric="comparisons_mean" title="Comparações de chave" description="Quantidade média de comparações registradas pelo algoritmo." badge="Menor é melhor" formatter={(value) => formatNumber(value)} />
            <MetricBarChart rows={performance} metric="chunks_scored_mean" title="Trechos pontuados" description="Quantos documentos precisaram receber score para cada consulta." badge="Trabalho medido" formatter={(value) => formatNumber(value)} />
            <MetricBarChart rows={performance} metric="peak_memory_mb_mean" title="Memória de pico" description="Uso médio máximo de memória observado durante a recuperação." badge="Menor é melhor" formatter={(value) => value == null ? "N/D" : `${value.toFixed(2)} MB`} />
          </section>

          <section className="section-intro"><div className="section-index section-index--blue">02</div><div><span>Qualidade</span><h2>Os resultados relevantes aparecem cedo?</h2><p>Precisão mede pureza; recall mede cobertura; MRR valoriza o primeiro acerto; hit rate indica se houve ao menos um acerto.</p></div></section>
          <section className="chart-grid">
            <QualityChart rows={quality} />
            <QualityExplanation rows={quality} topK={topK} />
          </section>

          <section className="panel comparison-table-panel">
            <div className="panel-heading"><div><h2>Tabela comparativa</h2><p>Mesma configuração experimental, lado a lado. N/D significa que a estratégia não instrumenta aquela medida.</p></div><span className="panel-badge">{Math.round(fraction * 100)}% · top {topK}</span></div>
            <div className="table-scroll"><table><thead><tr><th>Estratégia</th><th>Tempo total</th><th>Comparações</th><th>Memória</th><th>Precisão</th><th>Recall</th><th>MRR</th><th>Hit rate</th></tr></thead><tbody>{performance.map((row) => {
              const q = quality.find((item) => item.algorithm === row.algorithm);
              return <tr key={row.algorithm}><td><span className="algorithm-cell"><i style={{ background: ALGORITHM_COLORS[row.algorithm] }} />{ALGORITHM_LABELS[row.algorithm]}</span></td><td>{formatMs(row.total_time_ms_mean)}</td><td>{formatNumber(row.comparisons_mean)}</td><td>{row.peak_memory_mb_mean == null ? "N/D" : `${row.peak_memory_mb_mean.toFixed(2)} MB`}</td><td>{formatPercent(q?.precision_mean)}</td><td>{formatPercent(q?.recall_mean)}</td><td>{formatPercent(q?.mrr_mean)}</td><td>{formatPercent(q?.hit_rate_mean)}</td></tr>;
            })}</tbody></table></div>
          </section>

          <section className="section-intro"><div className="section-index section-index--green">03</div><div><span>Ordenação</span><h2>Merge Sort e Quick Sort reagem igual à entrada?</h2><p>Este benchmark é separado da recuperação. Ele evidencia sensibilidade à ordem e ao tamanho da entrada.</p></div></section>
          <section className="filter-bar filter-bar--compact"><div className="filter-title"><Scale size={18} /><span><strong>Cenário de entrada</strong><small>compare as curvas sob a mesma distribuição</small></span></div><label>Cenário<select value={scenario} onChange={(event) => setScenario(event.target.value)}>{[...new Set(metrics.sorting.map((row) => row.scenario))].sort().map((value) => <option key={value} value={value}>{scenarioLabel(value)}</option>)}</select></label></section>
          <section className="chart-grid">
            <SortingChart rows={sorting} metric="time_ms_mean" title="Tempo por tamanho da entrada" unit="ms" />
            <SortingChart rows={sorting} metric="comparisons_mean" title="Comparações por tamanho da entrada" unit="" />
          </section>

          <div className="method-note"><Info size={18} /><p><strong>Leitura responsável:</strong> eficiência e qualidade respondem perguntas diferentes. Os gráficos são gerados a partir dos CSVs persistidos em <code>reports/tables</code>; abrir esta página não executa nem altera experimentos.</p></div>
        </>
      )}
    </>
  );
}

type PerformanceMetric = keyof Pick<PerformanceRow, "total_time_ms_mean" | "comparisons_mean" | "chunks_scored_mean" | "peak_memory_mb_mean">;

function MetricBarChart({ rows, metric, title, description, badge, formatter }: { rows: PerformanceRow[]; metric: PerformanceMetric; title: string; description: string; badge: string; formatter: (value: number | null) => string }) {
  const data = rows.filter((row) => row[metric] != null).map((row) => ({ algorithm: ALGORITHM_LABELS[row.algorithm], key: row.algorithm, value: row[metric] }));
  return <ChartCard title={title} description={description} badge={badge}>{data.length ? <ResponsiveContainer width="100%" height={265}><BarChart data={data} margin={{ top: 12, right: 12, left: 2, bottom: 0 }}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="algorithm" tickLine={false} axisLine={false} /><YAxis tickLine={false} axisLine={false} width={62} /><Tooltip formatter={(value) => formatter(Number(value))} cursor={{ fill: "#f5f6f9" }} /><Bar dataKey="value" radius={[7, 7, 0, 0]} maxBarSize={58}>{data.map((entry) => <Cell key={entry.key} fill={ALGORITHM_COLORS[entry.key]} />)}</Bar></BarChart></ResponsiveContainer> : <div className="chart-unavailable">Esta métrica não foi instrumentada neste recorte.</div>}</ChartCard>;
}

function QualityChart({ rows }: { rows: QualityRow[] }) {
  const data = rows.map((row) => ({
    algorithm: ALGORITHM_LABELS[row.algorithm],
    Precisão: row.precision_mean == null ? null : row.precision_mean * 100,
    Recall: row.recall_mean == null ? null : row.recall_mean * 100,
    MRR: row.mrr_mean == null ? null : row.mrr_mean * 100,
    "Hit rate": row.hit_rate_mean == null ? null : row.hit_rate_mean * 100,
  }));
  return <ChartCard title="Qualidade por estratégia" description="Quatro perspectivas complementares para o mesmo top-k." badge="Maior é melhor"><ResponsiveContainer width="100%" height={300}><BarChart data={data} margin={{ top: 12, right: 12, left: 0, bottom: 0 }}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="algorithm" tickLine={false} axisLine={false} /><YAxis domain={[0, 100]} unit="%" tickLine={false} axisLine={false} width={48} /><Tooltip formatter={(value) => `${Number(value).toFixed(1)}%`} cursor={{ fill: "#f5f6f9" }} /><Legend iconType="circle" /><Bar dataKey="Precisão" fill="#e11d48" radius={[4, 4, 0, 0]} /><Bar dataKey="Recall" fill="#3b82f6" radius={[4, 4, 0, 0]} /><Bar dataKey="MRR" fill="#10b981" radius={[4, 4, 0, 0]} /><Bar dataKey="Hit rate" fill="#8b5cf6" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></ChartCard>;
}

function QualityExplanation({ rows, topK }: { rows: QualityRow[]; topK: number }) {
  const metrics: Array<{ label: string; key: keyof Pick<QualityRow, "precision_mean" | "recall_mean" | "mrr_mean" | "hit_rate_mean">; text: string }> = [
    { label: "Precisão", key: "precision_mean", text: `Proporção dos ${topK} resultados que é relevante.` },
    { label: "Recall", key: "recall_mean", text: "Parcela de todos os itens relevantes que foi recuperada." },
    { label: "MRR", key: "mrr_mean", text: "Premia estratégias que colocam o primeiro acerto no topo." },
    { label: "Hit rate", key: "hit_rate_mean", text: "Percentual de consultas com pelo menos um resultado relevante." },
  ];
  return <article className="panel metric-guide"><div className="panel-heading"><div><h2>Guia de leitura</h2><p>O melhor valor de cada métrica neste recorte.</p></div></div><div className="metric-guide-list">{metrics.map((metric) => {
    const leader = [...rows].sort((a, b) => (b[metric.key] ?? -1) - (a[metric.key] ?? -1))[0];
    return <div key={metric.key}><span>{metric.label}</span><strong>{leader ? formatPercent(leader[metric.key]) : "N/D"} <small>{leader ? ALGORITHM_LABELS[leader.algorithm] : ""}</small></strong><p>{metric.text}</p></div>;
  })}</div></article>;
}

function SortingChart({ rows, metric, title, unit }: { rows: MetricsResponse["sorting"]; metric: "time_ms_mean" | "comparisons_mean"; title: string; unit: string }) {
  const sizes = [...new Set(rows.map((row) => row.item_count))].sort((a, b) => a - b);
  const data = sizes.map((size) => ({ size, Merge: rows.find((row) => row.item_count === size && row.algorithm === "merge")?.[metric], Quick: rows.find((row) => row.item_count === size && row.algorithm === "quick")?.[metric] }));
  return <ChartCard title={title} description="Eixo horizontal: quantidade de itens." badge="Curva experimental"><ResponsiveContainer width="100%" height={280}><LineChart data={data} margin={{ top: 12, right: 18, left: 4, bottom: 0 }}><CartesianGrid strokeDasharray="4 4" stroke="#e7eaf0" /><XAxis dataKey="size" tickLine={false} /><YAxis tickLine={false} width={62} /><Tooltip formatter={(value) => `${Number(value).toLocaleString("pt-BR", { maximumFractionDigits: 2 })}${unit ? ` ${unit}` : ""}`} /><Legend iconType="circle" /><Line type="monotone" dataKey="Merge" stroke="#e11d48" strokeWidth={3} dot={{ r: 3 }} /><Line type="monotone" dataKey="Quick" stroke="#3b82f6" strokeWidth={3} dot={{ r: 3 }} /></LineChart></ResponsiveContainer></ChartCard>;
}

function scenarioLabel(value: string): string {
  const labels: Record<string, string> = { already_sorted: "Já ordenada", reverse_sorted: "Ordem inversa", random: "Aleatória", many_ties: "Muitos empates" };
  return labels[value] ?? value.replaceAll("_", " ");
}
