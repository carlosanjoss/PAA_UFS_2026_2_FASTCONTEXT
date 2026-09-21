import { ArrowRight, Boxes, Gauge, SearchCheck, Trophy } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartCard } from "../components/ChartCard";
import { ErrorState, LoadingState } from "../components/EmptyState";
import { KpiCard } from "../components/KpiCard";
import { PageHeader } from "../components/PageHeader";
import type { AppView, HealthResponse, MetricsResponse } from "../types";
import {
  ALGORITHM_COLORS,
  ALGORITHM_LABELS,
  formatMs,
  formatPercent,
  fullCorpusPerformance,
  fullCorpusQuality,
} from "../utils/format";

interface OverviewPageProps {
  health: HealthResponse | null;
  metrics: MetricsResponse | null;
  error: string | null;
  onNavigate: (view: AppView) => void;
}

export function OverviewPage({ health, metrics, error, onNavigate }: OverviewPageProps) {
  const performance = metrics ? fullCorpusPerformance(metrics.performance) : [];
  const quality = metrics ? fullCorpusQuality(metrics.quality, 5) : [];
  const fastest = [...performance].filter((row) => row.total_time_ms_mean != null)
    .sort((a, b) => (a.total_time_ms_mean ?? Infinity) - (b.total_time_ms_mean ?? Infinity))[0];
  const bestMrr = [...quality].filter((row) => row.mrr_mean != null)
    .sort((a, b) => (b.mrr_mean ?? -1) - (a.mrr_mean ?? -1))[0];

  const timeData = performance.map((row) => ({
    algorithm: ALGORITHM_LABELS[row.algorithm],
    key: row.algorithm,
    tempo: row.total_time_ms_mean,
  }));
  const qualityData = quality.map((row) => ({
    algorithm: ALGORITHM_LABELS[row.algorithm],
    Precisão: row.precision_mean == null ? null : row.precision_mean * 100,
    Recall: row.recall_mean == null ? null : row.recall_mean * 100,
    MRR: row.mrr_mean == null ? null : row.mrr_mean * 100,
  }));

  return (
    <>
      <PageHeader
        eyebrow="Visão geral"
        title="O experimento, explicado por dados"
        description="Compare custo computacional e qualidade de recuperação sem perder o vínculo com as medições do projeto."
        action={<button className="primary-button" onClick={() => onNavigate("ask")}>Fazer uma pergunta <ArrowRight size={17} /></button>}
      />

      <section className="kpi-grid" aria-label="Indicadores principais">
        <KpiCard label="Corpus preparado" value={health?.corpus_size?.toLocaleString("pt-BR") ?? "—"} detail="trechos da documentação FastAPI" icon={Boxes} tone="blue" />
        <KpiCard label="Estratégias avaliadas" value={String(health?.algorithms.length ?? 4)} detail="mesma coleção e protocolo" icon={SearchCheck} tone="coral" />
        <KpiCard label="Menor tempo médio" value={fastest ? formatMs(fastest.total_time_ms_mean) : "—"} detail={fastest ? `${ALGORITHM_LABELS[fastest.algorithm]} · corpus completo` : "aguardando relatório"} icon={Gauge} tone="green" />
        <KpiCard label="Melhor MRR @ 5" value={bestMrr ? formatPercent(bestMrr.mrr_mean) : "—"} detail={bestMrr ? ALGORITHM_LABELS[bestMrr.algorithm] : "aguardando relatório"} icon={Trophy} tone="amber" />
      </section>

      {error && <ErrorState message={error} />}
      {!metrics && !error && <LoadingState />}

      {metrics && (
        <section className="chart-grid">
          <ChartCard title="Tempo médio por estratégia" description="Tempo total no corpus completo. Barras menores indicam respostas mais rápidas." badge="Menor é melhor">
            <ResponsiveContainer width="100%" height={290}>
              <BarChart data={timeData} margin={{ top: 12, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" />
                <XAxis dataKey="algorithm" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} unit=" ms" width={66} />
                <Tooltip formatter={(value) => formatMs(Number(value))} cursor={{ fill: "#f5f6f9" }} />
                <Bar dataKey="tempo" radius={[7, 7, 0, 0]} maxBarSize={58}>
                  {timeData.map((entry) => <Cell key={entry.key} fill={ALGORITHM_COLORS[entry.key]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Qualidade da recuperação @ 5" description="Precisão, recall e MRR no corpus completo. Aqui, barras maiores indicam melhor resultado." badge="Maior é melhor">
            <ResponsiveContainer width="100%" height={290}>
              <BarChart data={qualityData} margin={{ top: 12, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" />
                <XAxis dataKey="algorithm" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} domain={[0, 100]} unit="%" width={48} />
                <Tooltip formatter={(value) => `${Number(value).toFixed(1)}%`} cursor={{ fill: "#f5f6f9" }} />
                <Legend iconType="circle" />
                <Bar dataKey="Precisão" fill="#e11d48" radius={[5, 5, 0, 0]} />
                <Bar dataKey="Recall" fill="#3b82f6" radius={[5, 5, 0, 0]} />
                <Bar dataKey="MRR" fill="#10b981" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </section>
      )}

      <section className="insight-strip">
        <div className="insight-number">01</div>
        <div><strong>Como interpretar</strong><p>Tempo mede eficiência. Precisão, recall e MRR medem qualidade. Uma estratégia só é “melhor” quando o critério da comparação está explícito.</p></div>
        <button className="text-button" onClick={() => onNavigate("experiments")}>Abrir análise completa <ArrowRight size={16} /></button>
      </section>
    </>
  );
}
