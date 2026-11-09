import {
  BrainCircuit,
  CheckCircle2,
  ClipboardCheck,
  DatabaseZap,
  FileWarning,
  Info,
  MessagesSquare,
  Scale,
  Settings2,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import { ChartCard } from "../components/ChartCard";
import { ErrorState, LoadingState } from "../components/EmptyState";
import { KpiCard } from "../components/KpiCard";
import { PageHeader } from "../components/PageHeader";
import type {
  RAGCondition,
  RAGHumanCriterion,
  RAGModelComparisonResponse,
  RAGModelRole,
} from "../types";

const roles: RAGModelRole[] = ["baseline", "candidate"];
const conditions: RAGCondition[] = ["no_rag", "optimized", "semantic"];
const roleColors: Record<RAGModelRole, string> = {
  baseline: "#e11d48",
  candidate: "#2563eb",
};
const conditionLabels: Record<RAGCondition, string> = {
  no_rag: "Sem RAG",
  optimized: "RAG otimizado",
  semantic: "RAG semântico",
};
const criterionLabels: Record<RAGHumanCriterion, string> = {
  correctness: "Correção",
  completeness: "Completude",
  clarity: "Clareza",
  groundedness: "Groundedness",
  hallucination_present: "Alucinação",
};

export function RAGModelComparisonPage() {
  const [data, setData] = useState<RAGModelComparisonResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedRoles, setSelectedRoles] = useState<RAGModelRole[]>(roles);
  const [selectedConditions, setSelectedConditions] = useState<RAGCondition[]>(conditions);
  const [criterion, setCriterion] = useState<RAGHumanCriterion>("correctness");

  useEffect(() => {
    api.ragModelComparison()
      .then(setData)
      .catch((requestError: unknown) => setError(
        requestError instanceof Error
          ? requestError.message
          : "A comparação de modelos está indisponível.",
      ));
  }, []);

  const modelLabels = useMemo(() => Object.fromEntries(
    data?.models.map((model) => [model.role, shortModel(model.model)]) ?? [],
  ) as Record<RAGModelRole, string>, [data]);

  if (error) return <><PageHeader eyebrow="Avaliação · LLMs" title="Qwen2.5 3B × Llama 3.2 3B" description="Comparação controlada de geração, citações e avaliação humana." /><ErrorState message={error} /></>;
  if (!data) return <><PageHeader eyebrow="Avaliação · LLMs" title="Qwen2.5 3B × Llama 3.2 3B" description="Comparação controlada de geração, citações e avaliação humana." /><LoadingState label="Validando os artefatos auditados…" /></>;

  const automatic = data.automatic_metrics.filter((row) => selectedRoles.includes(row.role) && selectedConditions.includes(row.condition));
  const human = data.human_metrics.filter((row) => selectedRoles.includes(row.role) && selectedConditions.includes(row.condition));
  const selectedHuman = human.filter((row) => row.criterion === criterion);
  const hallucination = human.filter((row) => row.criterion === "hallucination_present");
  const latencyData = pivotByCondition(automatic, (row) => row.generation_mean_ms);
  const citationData = pivotByCondition(
    automatic.filter((row) => row.condition !== "no_rag"),
    (row) => row.citation_compliance_rate === null ? null : row.citation_compliance_rate * 100,
  );
  const selectedMetricData = pivotHuman(selectedHuman);
  const hallucinationData = pivotHuman(hallucination, 100);
  const pairedRows = data.paired_human_metrics.filter((row) => selectedConditions.includes(row.condition) && row.criterion === criterion);
  const fastest = [...automatic].sort((left, right) => left.generation_mean_ms - right.generation_mean_ms)[0];
  const bestSelected = [...selectedHuman].sort((left, right) => {
    if (criterion === "hallucination_present") return left.mean_score - right.mean_score;
    return right.mean_score - left.mean_score;
  })[0];

  const toggleRole = (role: RAGModelRole) => setSelectedRoles((current) => (
    current.includes(role)
      ? current.length === 1 ? current : current.filter((item) => item !== role)
      : roles.filter((item) => current.includes(item) || item === role)
  ));
  const toggleCondition = (condition: RAGCondition) => setSelectedConditions((current) => (
    current.includes(condition)
      ? current.length === 1 ? current : current.filter((item) => item !== condition)
      : conditions.filter((item) => current.includes(item) || item === condition)
  ));

  return (
    <>
      <PageHeader
        eyebrow="Avaliação · LLMs"
        title="Qwen2.5 3B × Llama 3.2 3B"
        description="Latência, citações e avaliação humana em 90 prompts congelados e três condições experimentais."
      />

      <section className="kpi-grid model-comparison-kpis">
        <KpiCard label="Tentativas" value={String(data.metadata.attempts)} detail="2 modelos × 90 prompts" icon={BrainCircuit} tone="coral" />
        <KpiCard label="Respostas completas" value={String(data.metadata.complete_answers)} detail="usadas nas médias de geração" icon={CheckCircle2} tone="green" />
        <KpiCard label="Truncamento" value={String(data.metadata.truncated_answers)} detail="Qwen · optimized · q29" icon={FileWarning} tone="amber" />
        <KpiCard label="Prompts congelados" value={String(data.metadata.frozen_prompts)} detail="30 por condição" icon={MessagesSquare} tone="blue" />
        <KpiCard label="Julgamentos humanos" value={String(data.metadata.human_judgments)} detail={`${data.metadata.reviewers} avaliadores cegos`} icon={ClipboardCheck} tone="green" />
        <KpiCard label="Comparações válidas" value={String(data.metadata.valid_criterion_comparisons)} detail={`${data.metadata.missing_criterion_comparisons} ausentes de ${data.metadata.planned_criterion_comparisons}`} icon={Scale} tone="blue" />
      </section>

      <section className="model-cards" aria-label="Modelos comparados">
        {data.models.map((model) => <article className="panel model-card" key={model.role} style={{ borderTopColor: roleColors[model.role] }}><span>{model.role === "baseline" ? "Baseline" : "Candidato"}</span><h2>{shortModel(model.model)}</h2><p>{model.parameter_size} · {model.quantization}</p><small>Contexto declarado: {model.context_length.toLocaleString("pt-BR")} tokens</small></article>)}
        <article className="panel model-card model-card--method"><span>Protocolo comum</span><h2>Ollama · temperatura {data.generation.temperature}</h2><p>Seed {data.generation.seed} · limite inicial {data.generation.initial_max_tokens} tokens</p><small>Prompts e contexto congelados; sem troca automática de modelo.</small></article>
      </section>

      <section className="filter-bar model-comparison-filters">
        <div className="filter-title"><Settings2 size={18} /><span><strong>Filtros da comparação</strong><small>gráficos e tabelas respondem às seleções</small></span></div>
        <fieldset><legend>Modelos</legend><div className="filter-chip-row">{roles.map((role) => <label className={selectedRoles.includes(role) ? "algorithm-chip algorithm-chip--active" : "algorithm-chip"} key={role}><input type="checkbox" checked={selectedRoles.includes(role)} onChange={() => toggleRole(role)} /><i style={{ background: roleColors[role] }} />{modelLabels[role]}</label>)}</div></fieldset>
        <fieldset><legend>Condições</legend><div className="filter-chip-row">{conditions.map((condition) => <label className={selectedConditions.includes(condition) ? "algorithm-chip algorithm-chip--active" : "algorithm-chip"} key={condition}><input type="checkbox" checked={selectedConditions.includes(condition)} onChange={() => toggleCondition(condition)} />{conditionLabels[condition]}</label>)}</div></fieldset>
        <label>Métrica humana<select value={criterion} onChange={(event) => setCriterion(event.target.value as RAGHumanCriterion)}>{Object.entries(criterionLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      </section>

      <section className="analysis-overview panel model-overview">
        <div className="analysis-overview__heading"><span>Leitura guiada</span><h2>O que os filtros atuais mostram?</h2><p>As médias são descritivas. Diferenças visuais não constituem evidência de significância estatística.</p></div>
        <div className="analysis-overview__facts">
          <div><span>Menor latência média</span><strong>{fastest ? `${modelLabels[fastest.role]} · ${conditionLabels[fastest.condition]}` : "Sem dados"}</strong><p>{fastest ? formatMs(fastest.generation_mean_ms) : "Ative uma combinação para comparar."}</p></div>
          <div><span>Melhor resultado em {criterionLabels[criterion]}</span><strong>{bestSelected ? `${modelLabels[bestSelected.role]} · ${conditionLabels[bestSelected.condition]}` : "Sem dados"}</strong><p>{bestSelected ? criterion === "hallucination_present" ? formatPercent(bestSelected.mean_score) : `${bestSelected.mean_score.toFixed(2)}/5` : "A combinação não possui essa métrica."}</p></div>
          <div><span>Escopo da inferência</span><strong>Descrição do conjunto observado</strong><p>Sem teste inferencial, adjudicação ou generalização para outros modelos e prompts.</p></div>
        </div>
      </section>

      <section className="section-intro"><div className="section-index">01</div><div><span>Métricas automáticas</span><h2>Tempo de geração e conformidade estrutural</h2><p>Calculadas diretamente dos 180 registros de execução; latência usa somente respostas completas.</p></div></section>
      <section className="chart-grid">
        <ChartCard title="Latência média de geração" description="Milissegundos por resposta completa; menor é melhor." badge="Métrica automática">
          <ResponsiveContainer width="100%" height={310}><BarChart data={latencyData}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="condition" /><YAxis unit=" ms" width={82} tickFormatter={compactNumber} /><Tooltip formatter={(value) => formatMs(Number(value))} /><Legend formatter={(value) => modelLabels[value as RAGModelRole] ?? value} />{selectedRoles.map((role) => <Bar key={role} dataKey={role} fill={roleColors[role]} radius={[6, 6, 0, 0]} />)}</BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Como interpretar</strong><p>O tempo inclui somente a geração. No-RAG é mais curto porque não recebe os cinco chunks; a comparação entre modelos deve ser feita dentro da mesma condição.</p></div>
        </ChartCard>
        <ChartCard title="Conformidade das citações" description="Respostas cujo conjunto de citações foi aceito pelo validador; apenas condições RAG." badge="Métrica automática">
          <ResponsiveContainer width="100%" height={310}><BarChart data={citationData}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="condition" /><YAxis domain={[0, 100]} unit="%" width={56} /><Tooltip formatter={(value) => `${Number(value).toFixed(1)}%`} /><Legend formatter={(value) => modelLabels[value as RAGModelRole] ?? value} />{selectedRoles.map((role) => <Bar key={role} dataKey={role} fill={roleColors[role]} radius={[6, 6, 0, 0]} />)}</BarChart></ResponsiveContainer>
          <div className="chart-analysis chart-analysis--warning"><strong>Limite da métrica</strong><p>Conformidade verifica se as citações respeitam o contrato do contexto. Não comprova correção factual, completude ou groundedness.</p></div>
        </ChartCard>
      </section>

      <section className="section-intro"><div className="section-index section-index--blue">02</div><div><span>Avaliação humana</span><h2>Qualidade percebida nas respostas completas</h2><p>Dois julgamentos por resposta e critério, preservados sem substituição das notas originais.</p></div></section>
      <section className="chart-grid">
        <ChartCard title={criterionLabels[criterion]} description={criterion === "hallucination_present" ? "Média do indicador binário 0/1; menor é melhor." : "Média dos dois julgamentos na escala ordinal de 1 a 5."} badge="Métrica selecionada">
          <ResponsiveContainer width="100%" height={310}><BarChart data={selectedMetricData}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="condition" /><YAxis domain={criterion === "hallucination_present" ? [0, 100] : [0, 5]} unit={criterion === "hallucination_present" ? "%" : ""} width={56} /><Tooltip formatter={(value) => criterion === "hallucination_present" ? `${Number(value).toFixed(1)}%` : `${Number(value).toFixed(2)}/5`} /><Legend formatter={(value) => modelLabels[value as RAGModelRole] ?? value} />{selectedRoles.map((role) => <Bar key={role} dataKey={role} fill={roleColors[role]} radius={[6, 6, 0, 0]} />)}</BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Filtro de métrica</strong><p>{criterion === "groundedness" && selectedConditions.includes("no_rag") ? "Groundedness não foi aplicado ao No-RAG porque não existe contexto recuperado a fundamentar. " : ""}Uma média ordinal resume o conjunto, mas não demonstra distância intervalar nem significância.</p></div>
        </ChartCard>
        <ChartCard title="Alucinação marcada pelos avaliadores" description="Média dos indicadores 0/1; 0,5 em um item significa discordância entre os dois julgamentos." badge="Menor é melhor">
          <ResponsiveContainer width="100%" height={310}><BarChart data={hallucinationData}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="condition" /><YAxis domain={[0, 100]} unit="%" width={56} /><Tooltip formatter={(value) => `${Number(value).toFixed(1)}%`} /><Legend formatter={(value) => modelLabels[value as RAGModelRole] ?? value} />{selectedRoles.map((role) => <Bar key={role} dataKey={role} fill={roleColors[role]} radius={[6, 6, 0, 0]} />)}</BarChart></ResponsiveContainer>
          <div className="chart-analysis chart-analysis--warning"><strong>Cuidado</strong><p>A marcação humana é descritiva e não substitui checagem factual externa. Empates entre avaliadores não foram adjudicados.</p></div>
        </ChartCard>
      </section>

      <section className="chart-grid chart-grid--single">
        <ChartCard title="Concordância exata entre avaliadores" description="Agregada no protocolo cego, sem revelar pares ou identificadores." badge="596 julgamentos">
          <ResponsiveContainer width="100%" height={290}><BarChart data={data.agreement_metrics.map((row) => ({ criterion: criterionLabels[row.criterion], agreement: row.exact_agreement_rate * 100, pairs: row.item_pairs }))} margin={{ left: 4, right: 12 }}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="criterion" /><YAxis domain={[0, 100]} unit="%" width={56} /><Tooltip formatter={(value, name) => name === "agreement" ? `${Number(value).toFixed(1)}%` : String(value)} /><Bar dataKey="agreement" name="Concordância exata" radius={[7, 7, 0, 0]}>{data.agreement_metrics.map((row) => <Cell key={row.criterion} fill={row.criterion === "hallucination_present" ? "#16a34a" : "#8b5cf6"} />)}</Bar></BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>O que mede</strong><p>É a proporção de pares que atribuíram exatamente a mesma nota. Não comprova independência, qualidade factual ou confiabilidade geral; não foram calculados kappa ou alpha.</p></div>
        </ChartCard>
      </section>

      <section className="panel comparison-table-panel model-comparison-table">
        <div className="panel-heading"><div><h2>Comparação pareada · {criterionLabels[criterion]}</h2><p>Llama menos Qwen para a mesma pergunta e condição; somente pares completos.</p></div></div>
        <div className="table-scroll"><table><thead><tr><th>Condição</th><th>Pares válidos</th><th>Ausentes</th><th>Δ médio Llama − Qwen</th><th>Llama melhor</th><th>Qwen melhor</th><th>Empates</th></tr></thead><tbody>{pairedRows.map((row) => <tr key={`${row.condition}-${row.criterion}`}><td>{conditionLabels[row.condition]}</td><td>{row.both_complete}/{row.planned_pairs}</td><td>{row.missing_pair}</td><td>{row.mean_candidate_minus_baseline.toFixed(3)}</td><td>{row.candidate_better}</td><td>{row.baseline_better}</td><td>{row.ties}</td></tr>)}</tbody></table></div>
        <div className="method-note"><Info size={18} /><p>Para alucinação, delta positivo favorece o Qwen porque valores menores são melhores. Nos demais critérios, delta positivo favorece o Llama. A tabela é descritiva e não contém teste de hipótese.</p></div>
      </section>

      <section className="section-intro"><div className="section-index section-index--green">03</div><div><span>Recuperação semântica</span><h2>Como o contexto semântico foi produzido?</h2><p>Configuração lida do projeto e preservada durante o experimento.</p></div></section>
      <section className="semantic-method panel">
        <div className="semantic-method__flow"><div><DatabaseZap size={20} /><span>Modelo</span><strong>{data.retrieval.model}</strong></div><i>→</i><div><span>Embedding</span><strong>{data.retrieval.embedding_dimension} dimensões</strong><small>{data.retrieval.normalized ? "vetores normalizados" : "sem normalização"}</small></div><i>→</i><div><span>Índice</span><strong>FAISS {data.retrieval.index_type}</strong><small>busca por produto interno</small></div><i>→</i><div><span>Seleção</span><strong>Top-{data.retrieval.top_k}</strong><small>mesmo contexto congelado para os modelos</small></div></div>
        <div className="semantic-method__reference"><strong>Referência bibliográfica</strong><p>{data.retrieval.reference}</p><a href={data.retrieval.reference_url} target="_blank" rel="noreferrer">Artigo BGE</a><a href={data.retrieval.faiss_reference_url} target="_blank" rel="noreferrer">Documentação IndexFlatIP</a></div>
      </section>

      <section className="scientific-interpretation panel model-methodology">
        <div className="panel-heading"><div><h2>Metodologia e limites</h2><p>Métricas automáticas e humanas permanecem separadas.</p></div><Info size={19} /></div>
        <div className="interpretation-grid"><div><strong>Automático</strong><p>{data.automatic_methodology_note}</p></div><div><strong>Humano</strong><p>{data.human_methodology_note}</p></div><div><strong>Privacidade</strong><p>A API envia somente agregados por modelo, condição e critério. Chaves cegas, alocações, observações e identificadores de avaliadores permanecem nos artefatos administrativos locais.</p></div><div><strong>Leitura correta</strong><p>Os resultados descrevem este corpus, estes 90 prompts e estas versões quantizadas. Não autorizam alegar superioridade universal entre os modelos.</p></div></div>
      </section>
    </>
  );
}

function pivotByCondition<T extends { role: RAGModelRole; condition: RAGCondition }>(
  rows: T[],
  value: (row: T) => number | null,
) {
  return conditions.filter((condition) => rows.some((row) => row.condition === condition)).map((condition) => {
    const point: Record<string, string | number | null> = { condition: conditionLabels[condition] };
    rows.filter((row) => row.condition === condition).forEach((row) => { point[row.role] = value(row); });
    return point;
  });
}

function pivotHuman(
  rows: RAGModelComparisonResponse["human_metrics"],
  multiplier = 1,
) {
  return pivotByCondition(rows, (row) => row.mean_score * multiplier);
}

function shortModel(model: string) {
  if (model === "qwen2.5:3b") return "Qwen2.5 3B";
  if (model === "llama3.2:3b") return "Llama 3.2 3B";
  return model;
}

function formatMs(value: number) {
  return `${value.toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ms`;
}

function formatPercent(value: number) {
  return `${(value * 100).toLocaleString("pt-BR", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%`;
}

function compactNumber(value: number) {
  return Intl.NumberFormat("pt-BR", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}
