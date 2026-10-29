import { Bot, CheckCircle2, DatabaseZap, Info, MessagesSquare, ShieldAlert } from "lucide-react";
import { useMemo, useState } from "react";
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
import type {
  AV2ExperimentsResponse,
  AV2RetrievalCondition,
  AV2RetrievalQueryRow,
} from "../types";
import { ChartCard } from "./ChartCard";
import { KpiCard } from "./KpiCard";

const conditionOrder: AV2RetrievalCondition[] = ["semantic", "literal", "hybrid"];
const conditionLabels: Record<AV2RetrievalCondition, string> = {
  semantic: "Semântica",
  literal: "Literal",
  hybrid: "Híbrida",
};
const conditionColors: Record<AV2RetrievalCondition, string> = {
  semantic: "#2563eb",
  literal: "#f97316",
  hybrid: "#16a34a",
};

interface AV2RAGDashboardProps {
  data: AV2ExperimentsResponse;
}

export function AV2RAGDashboard({ data }: AV2RAGDashboardProps) {
  const [category, setCategory] = useState("all");
  const categories = useMemo(
    () => [...new Set(data.retrieval.queries.map((row) => row.category))].sort(),
    [data.retrieval.queries],
  );
  const categoryRows = useMemo(
    () => aggregateRetrievalQueries(
      data.retrieval.queries.filter((row) => category === "all" || row.category === category),
    ),
    [category, data.retrieval.queries],
  );

  const qualityData = data.retrieval.conditions.map((row) => ({
    condition: conditionLabels[row.condition],
    key: row.condition,
    "Precisão@5": row.pooled_precision_at_k * 100,
    "Recall@5": row.pooled_recall_at_k * 100,
    "MRR@5": row.known_mrr_at_k * 100,
    "Hit rate@5": row.known_hit_rate_at_k * 100,
  }));
  const boundData = data.retrieval.conditions.map((row) => ({
    condition: conditionLabels[row.condition],
    key: row.condition,
    "Limite inferior": row.mean_precision_lower_bound * 100,
    "Faixa não julgada": Math.max(0, row.mean_precision_upper_bound - row.mean_precision_lower_bound) * 100,
    unjudged: row.unjudged_results,
  }));
  const ragData = data.rag.conditions.map((row) => ({
    condition: conditionLabels[row.condition],
    key: row.condition,
    tempo: row.mean_generation_time_ms,
    validade: row.citation_valid_rate * 100,
    citações: row.mean_citation_count,
    relevantes: row.mean_known_relevant_citations,
  }));
  const humanScores = data.human_evaluation.conditions.map((row) => ({
    condition: conditionLabels[row.condition],
    key: row.condition,
    Correção: row.mean_correctness,
    Completude: row.mean_completeness,
    Clareza: row.mean_clarity,
    Fundamentação: row.mean_groundedness,
  }));
  const hallucinationData = data.human_evaluation.conditions.map((row) => ({
    condition: conditionLabels[row.condition],
    key: row.condition,
    taxa: row.hallucination_rate * 100,
  }));
  const fastestRetrieval = [...data.retrieval.conditions].sort((a, b) => a.mean_time_ms - b.mean_time_ms)[0];
  const fastestGeneration = [...data.rag.conditions].sort((a, b) => a.mean_generation_time_ms - b.mean_generation_time_ms)[0];
  const bestGroundedness = [...data.human_evaluation.conditions].sort((a, b) => b.mean_groundedness - a.mean_groundedness)[0];
  const lowestHallucination = [...data.human_evaluation.conditions].sort((a, b) => a.hallucination_rate - b.hallucination_rate)[0];
  const validCitations = data.rag.conditions.reduce((total, row) => total + row.citation_valid_count, 0);

  return (
    <>
      <section className="section-intro"><div className="section-index section-index--green">03</div><div><span>Recuperação para RAG</span><h2>Qual estratégia encontra contexto relevante com menor custo?</h2><p>Trinta perguntas foram executadas com recuperação semântica, literal e híbrida, sempre com top-5.</p></div></section>

      <section className="kpi-grid av2-kpi-grid rag-kpi-grid">
        <KpiCard label="Perguntas" value={String(data.retrieval.metadata.query_count)} detail="mesmo conjunto nas 3 condições" icon={MessagesSquare} tone="blue" />
        <KpiCard label="Medições" value={data.retrieval.metadata.raw_measurements.toLocaleString("pt-BR")} detail={`${data.retrieval.metadata.repetitions} repetições por consulta`} icon={DatabaseZap} tone="green" />
        <KpiCard label="Resultados por consulta" value={`top-${data.retrieval.metadata.top_k}`} detail={`profundidade candidata ${data.retrieval.metadata.candidate_depth}`} icon={CheckCircle2} tone="amber" />
      </section>

      <section className="rag-guided-summary panel">
        <div><span>Mais rápida</span><strong>{conditionLabels[fastestRetrieval.condition]} · {formatMs(fastestRetrieval.mean_time_ms)}</strong><p>Tempo médio de recuperação no conjunto completo.</p></div>
        <div><span>Maior precisão observada</span><strong>{conditionLabels[bestBy(data.retrieval.conditions, "pooled_precision_at_k").condition]} · {formatPercent(bestBy(data.retrieval.conditions, "pooled_precision_at_k").pooled_precision_at_k)}</strong><p>Precisão agrupada sobre a relevância conhecida.</p></div>
        <div><span>Maior cobertura conhecida</span><strong>{conditionLabels[bestBy(data.retrieval.conditions, "pooled_recall_at_k").condition]} · {formatPercent(bestBy(data.retrieval.conditions, "pooled_recall_at_k").pooled_recall_at_k)}</strong><p>Recall@5 não considera documentos ainda não julgados como negativos.</p></div>
      </section>

      <section className="chart-grid">
        <ChartCard title="Qualidade da recuperação" description="Métricas agregadas no top-5; quanto maior, melhor." badge="30 perguntas">
          <ResponsiveContainer width="100%" height={315}><BarChart data={qualityData} margin={{ top: 10, right: 8, left: 0, bottom: 0 }}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="condition" /><YAxis domain={[0, 100]} unit="%" width={54} /><Tooltip formatter={(value) => formatPercent(Number(value) / 100)} /><Legend /><Bar dataKey="Precisão@5" fill="#2563eb" radius={[3, 3, 0, 0]} /><Bar dataKey="Recall@5" fill="#8b5cf6" radius={[3, 3, 0, 0]} /><Bar dataKey="MRR@5" fill="#f97316" radius={[3, 3, 0, 0]} /><Bar dataKey="Hit rate@5" fill="#16a34a" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Leitura didática</strong><p>Precisão indica quanto do top-5 era relevante; recall mostra quanto da relevância conhecida foi recuperada; MRR premia o primeiro acerto bem posicionado; hit rate informa em quantas perguntas houve ao menos um acerto.</p></div>
        </ChartCard>
        <ChartCard title="Latência de recuperação" description="Inclui a execução completa de cada estratégia, conforme o manifesto." badge="Menor é melhor">
          <ResponsiveContainer width="100%" height={315}><BarChart data={data.retrieval.conditions.map((row) => ({ condition: conditionLabels[row.condition], key: row.condition, tempo: row.mean_time_ms }))}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="condition" /><YAxis unit=" ms" width={68} /><Tooltip formatter={(value) => formatMs(Number(value))} /><Bar dataKey="tempo" radius={[7, 7, 0, 0]}>{data.retrieval.conditions.map((row) => <Cell key={row.condition} fill={conditionColors[row.condition]} />)}</Bar></BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Resultado observado</strong><p>A recuperação <b>{conditionLabels[fastestRetrieval.condition]}</b> teve o menor tempo médio ({formatMs(fastestRetrieval.mean_time_ms)}). A híbrida agrega resultados semânticos e literais; por isso sua latência inclui as duas recuperações e a fusão.</p></div>
        </ChartCard>
      </section>

      <section className="chart-grid">
        <ChartCard title="Incerteza por itens não julgados" description="A parte clara mostra quanto a precisão poderia aumentar no melhor caso." badge="Faixa de precisão">
          <ResponsiveContainer width="100%" height={295}><BarChart data={boundData}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="condition" /><YAxis domain={[0, 100]} unit="%" width={54} /><Tooltip formatter={(value, name) => [formatPercent(Number(value) / 100), name]} /><Legend /><Bar dataKey="Limite inferior" stackId="bounds" fill="#2563eb" /><Bar dataKey="Faixa não julgada" stackId="bounds" fill="#bfdbfe" radius={[6, 6, 0, 0]} /></BarChart></ResponsiveContainer>
          <div className="chart-analysis chart-analysis--warning"><strong>Cuidado metodológico</strong><p>Itens não julgados não são automaticamente irrelevantes. A condição literal retornou {data.retrieval.conditions.find((row) => row.condition === "literal")?.unjudged_results ?? 0} itens sem julgamento; por isso o dashboard exibe limites em vez de transformar ausência de anotação em erro.</p></div>
        </ChartCard>
        <ChartCard title="Análise por categoria" description="O filtro recalcula as médias a partir das 90 combinações por consulta." badge={category === "all" ? "Todas as categorias" : category} action={<label className="compact-select">Categoria<select value={category} onChange={(event) => setCategory(event.target.value)}><option value="all">Todas</option>{categories.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>}>
          <ResponsiveContainer width="100%" height={295}><BarChart data={categoryRows}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="condition" /><YAxis yAxisId="quality" domain={[0, 100]} unit="%" width={54} /><YAxis yAxisId="time" orientation="right" unit=" ms" width={68} /><Tooltip formatter={(value, name) => name === "Tempo médio" ? formatMs(Number(value)) : formatPercent(Number(value) / 100)} /><Legend /><Bar yAxisId="quality" dataKey="Precisão@5" fill="#8b5cf6" radius={[4, 4, 0, 0]} /><Bar yAxisId="time" dataKey="Tempo médio" fill="#14b8a6" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Filtro efetivo</strong><p>{category === "all" ? "As barras resumem as 30 perguntas." : `As barras usam somente as perguntas da categoria “${category}”.`} Isso permite verificar se uma média global esconde diferenças entre tipos de pergunta.</p></div>
        </ChartCard>
      </section>

      <section className="section-intro"><div className="section-index section-index--blue">04</div><div><span>Geração com contexto</span><h2>Como a estratégia de recuperação afeta a resposta da LLM?</h2><p>As 90 respostas foram geradas com o mesmo modelo e configuração; muda apenas a origem do contexto recuperado.</p></div></section>

      <section className="kpi-grid av2-kpi-grid rag-kpi-grid">
        <KpiCard label="Respostas geradas" value={String(data.rag.metadata.generation_count)} detail={`${data.rag.metadata.query_count} por condição`} icon={Bot} tone="coral" />
        <KpiCard label="Citações válidas" value={`${validCitations}/${data.rag.metadata.generation_count}`} detail="IDs citados pertencem ao contexto" icon={CheckCircle2} tone="green" />
        <KpiCard label="Modelo" value={data.rag.metadata.model} detail={`temperatura ${data.rag.metadata.temperature} · seed ${data.rag.metadata.seed}`} icon={DatabaseZap} tone="blue" />
      </section>

      <section className="chart-grid">
        <ChartCard title="Tempo de geração" description="Tempo médio da resposta da LLM após a recuperação." badge="Menor é melhor">
          <ResponsiveContainer width="100%" height={295}><BarChart data={ragData}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="condition" /><YAxis unit=" ms" width={78} /><Tooltip formatter={(value) => formatMs(Number(value))} /><Bar dataKey="tempo" radius={[7, 7, 0, 0]}>{data.rag.conditions.map((row) => <Cell key={row.condition} fill={conditionColors[row.condition]} />)}</Bar></BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Resultado observado</strong><p>A condição <b>{conditionLabels[fastestGeneration.condition]}</b> apresentou a menor média de geração: {formatMs(fastestGeneration.mean_generation_time_ms)}. Esses tempos dependem do modelo, máquina e comprimento das respostas.</p></div>
        </ChartCard>
        <ChartCard title="Validade das citações" description="Percentual de respostas que citaram somente IDs presentes no contexto fornecido." badge="Validação estrutural">
          <ResponsiveContainer width="100%" height={295}><BarChart data={ragData}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="condition" /><YAxis domain={[0, 100]} unit="%" width={54} /><Tooltip formatter={(value) => formatPercent(Number(value) / 100)} /><Bar dataKey="validade" radius={[7, 7, 0, 0]}>{data.rag.conditions.map((row) => <Cell key={row.condition} fill={conditionColors[row.condition]} />)}</Bar></BarChart></ResponsiveContainer>
          <div className="chart-analysis chart-analysis--warning"><strong>O que isto não mede</strong><p>Citação válida significa apenas que o identificador existe no contexto. Não comprova que a afirmação está correta, que o trecho realmente sustenta a resposta ou que não houve alucinação.</p></div>
        </ChartCard>
      </section>

      <section className="chart-grid chart-grid--single">
        <ChartCard title="Citações produzidas e relevância conhecida" description="Médias por resposta; a diferença evidencia citações sem relevância conhecida no conjunto anotado." badge="Por resposta">
          <ResponsiveContainer width="100%" height={280}><BarChart data={ragData}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="condition" /><YAxis width={48} /><Tooltip formatter={(value) => Number(value).toLocaleString("pt-BR", { maximumFractionDigits: 3 })} /><Legend /><Bar dataKey="citações" name="Citações emitidas" fill="#2563eb" radius={[5, 5, 0, 0]} /><Bar dataKey="relevantes" name="Citações com relevância conhecida" fill="#16a34a" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Como interpretar</strong><p>Uma citação pode ser estruturalmente válida e ainda não estar marcada como relevante. A relevância conhecida depende do pool de julgamentos; portanto, ausência de marcação também não prova irrelevância.</p></div>
        </ChartCard>
      </section>

      <section className="section-intro"><div className="section-index section-index--amber">05</div><div><span>Avaliação humana anônima</span><h2>As respostas foram corretas, completas, claras e fundamentadas?</h2><p>Notas de 1 a 5 e indicador binário de alucinação, agregados sem expor identidade, textos ou observações individuais.</p></div></section>

      <section className="chart-grid">
        <ChartCard title="Qualidade percebida" description="Médias na escala ordinal de 1 a 5; maior é melhor." badge="Avaliação anônima">
          <ResponsiveContainer width="100%" height={315}><BarChart data={humanScores}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="condition" /><YAxis domain={[1, 5]} width={44} /><Tooltip formatter={(value) => Number(value).toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} /><Legend /><Bar dataKey="Correção" fill="#2563eb" /><Bar dataKey="Completude" fill="#8b5cf6" /><Bar dataKey="Clareza" fill="#f97316" /><Bar dataKey="Fundamentação" fill="#16a34a" /></BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Resultado observado</strong><p>A maior fundamentação média foi da condição <b>{conditionLabels[bestGroundedness.condition]}</b> ({bestGroundedness.mean_groundedness.toFixed(2)}/5). As diferenças são descritivas e não demonstram significância estatística.</p></div>
        </ChartCard>
        <ChartCard title="Respostas marcadas com alucinação" description="Proporção de avaliações com hallucination_present = 1; menor é melhor." badge="Indicador binário">
          <ResponsiveContainer width="100%" height={315}><BarChart data={hallucinationData}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="condition" /><YAxis domain={[0, 100]} unit="%" width={54} /><Tooltip formatter={(value) => formatPercent(Number(value) / 100)} /><Bar dataKey="taxa" radius={[7, 7, 0, 0]}>{data.human_evaluation.conditions.map((row) => <Cell key={row.condition} fill={conditionColors[row.condition]} />)}</Bar></BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Resultado observado</strong><p>A menor taxa foi da condição <b>{conditionLabels[lowestHallucination.condition]}</b>: {formatPercent(lowestHallucination.hallucination_rate)}. A marcação é humana e não substitui verificação factual externa.</p></div>
        </ChartCard>
      </section>

      <section className="panel comparison-table-panel rag-comparison-table">
        <div className="panel-heading"><div><h2>Visão integrada por condição</h2><p>Recuperação, geração e avaliação humana lado a lado.</p></div></div>
        <div className="table-scroll"><table><thead><tr><th>Condição</th><th>Recuperação</th><th>Precisão@5</th><th>Recall@5</th><th>Geração</th><th>Citação válida</th><th>Correção</th><th>Fundamentação</th><th>Alucinação</th></tr></thead><tbody>{conditionOrder.map((condition) => { const retrieval = data.retrieval.conditions.find((row) => row.condition === condition)!; const rag = data.rag.conditions.find((row) => row.condition === condition)!; const human = data.human_evaluation.conditions.find((row) => row.condition === condition)!; return <tr key={condition}><td><span className="condition-cell"><i style={{ background: conditionColors[condition] }} />{conditionLabels[condition]}</span></td><td>{formatMs(retrieval.mean_time_ms)}</td><td>{formatPercent(retrieval.pooled_precision_at_k)}</td><td>{formatPercent(retrieval.pooled_recall_at_k)}</td><td>{formatMs(rag.mean_generation_time_ms)}</td><td>{formatPercent(rag.citation_valid_rate)}</td><td>{human.mean_correctness.toFixed(2)}/5</td><td>{human.mean_groundedness.toFixed(2)}/5</td><td>{formatPercent(human.hallucination_rate)}</td></tr>; })}</tbody></table></div>
      </section>

      <section className="panel evaluation-audit">
        <div className="panel-heading"><div><h2>Auditoria do pacote de avaliações</h2><p>Por que o número de linhas submetidas não deve ser confundido com avaliações independentes.</p></div><ShieldAlert size={20} /></div>
        <div className="evaluation-audit__numbers">
          <div><span>Linhas recebidas</span><strong>{data.metadata.human_evaluation_ratings}</strong><p>{data.human_evaluation.quality_ratings} de qualidade + {data.human_evaluation.groundedness_ratings} de fundamentação</p></div>
          <div><span>Julgamentos efetivamente distintos</span><strong>{data.human_evaluation.effective_quality_judgments + data.human_evaluation.effective_groundedness_judgments}</strong><p>{data.human_evaluation.effective_quality_judgments} respostas de qualidade + {data.human_evaluation.effective_groundedness_judgments} de fundamentação</p></div>
          <div><span>Pares idênticos</span><strong>{data.human_evaluation.duplicate_score_pairs}/{data.human_evaluation.unique_responses * 2}</strong><p>Mesmas notas e mesmas observações nos dois registros de cada item</p></div>
        </div>
        <div className="method-note evaluation-audit__note"><Info size={18} /><p><strong>Limitação obrigatória:</strong> {data.human_evaluation.methodology_note} Consequentemente, os 100% de coincidência não são apresentados como concordância entre avaliadores independentes.</p></div>
      </section>

      <section className="scientific-interpretation panel rag-interpretation">
        <div className="panel-heading"><div><h2>Interpretação integrada</h2><p>Conclusões sustentadas pelos artefatos atuais, sem extrapolar o desenho experimental.</p></div><Info size={19} /></div>
        <div className="interpretation-grid">
          <div><strong>Semântica: melhor equilíbrio entre rapidez e qualidade</strong><p>Foi a recuperação mais rápida ({formatMs(fastestRetrieval.mean_time_ms)}), teve precisão@5 de {formatPercent(data.retrieval.conditions.find((row) => row.condition === "semantic")!.pooled_precision_at_k)} e a menor média de geração ({formatMs(data.rag.conditions.find((row) => row.condition === "semantic")!.mean_generation_time_ms)}).</p></div>
          <div><strong>Híbrida: melhor avaliação humana neste conjunto</strong><p>Obteve correção média de {data.human_evaluation.conditions.find((row) => row.condition === "hybrid")!.mean_correctness.toFixed(2)}/5, fundamentação de {data.human_evaluation.conditions.find((row) => row.condition === "hybrid")!.mean_groundedness.toFixed(2)}/5 e a menor marcação de alucinação ({formatPercent(data.human_evaluation.conditions.find((row) => row.condition === "hybrid")!.hallucination_rate)}), com maior latência.</p></div>
          <div><strong>Literal: resultados sensíveis ao julgamento incompleto</strong><p>Apresentou mais itens não julgados e menor cobertura conhecida. Isso não autoriza classificá-los automaticamente como irrelevantes; os limites de precisão mostram a incerteza.</p></div>
          <div><strong>Escopo das conclusões</strong><p>Os resultados valem para estas 30 perguntas, este corpus, este modelo local e estas implementações. Não há teste de significância, verificação factual externa nem confirmação de avaliações humanas independentes.</p></div>
        </div>
      </section>
    </>
  );
}

function aggregateRetrievalQueries(rows: AV2RetrievalQueryRow[]) {
  return conditionOrder.map((condition) => {
    const group = rows.filter((row) => row.condition === condition);
    return {
      condition: conditionLabels[condition],
      "Precisão@5": average(group.map((row) => row.pooled_precision_at_k)) * 100,
      "Tempo médio": average(group.map((row) => row.mean_time_ms)),
    };
  });
}

function bestBy<T extends { condition: AV2RetrievalCondition }>(rows: T[], key: keyof T) {
  return [...rows].sort((left, right) => Number(right[key]) - Number(left[key]))[0];
}

function average(values: number[]) {
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : 0;
}

function formatMs(value: number) {
  return `${value.toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ms`;
}

function formatPercent(value: number) {
  return `${(value * 100).toLocaleString("pt-BR", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%`;
}
