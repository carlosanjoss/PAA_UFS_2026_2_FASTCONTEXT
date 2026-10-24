import {
  Cpu,
  Database,
  Download,
  Filter,
  FlaskConical,
  Gauge,
  Info,
  MemoryStick,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  ErrorBar,
  Legend,
  Line,
  LineChart,
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
  AV2ExperimentsResponse,
  AV2SummaryRow,
  StringMatchingAlgorithm,
} from "../types";

const algorithmOrder: StringMatchingAlgorithm[] = ["naive", "kmp", "boyer-moore", "rabin-karp"];
const algorithmLabels: Record<StringMatchingAlgorithm, string> = {
  naive: "Naive",
  kmp: "KMP",
  "boyer-moore": "Boyer-Moore",
  "rabin-karp": "Rabin-Karp",
};
const algorithmColors: Record<StringMatchingAlgorithm, string> = {
  naive: "#64748b",
  kmp: "#2563eb",
  "boyer-moore": "#e11d48",
  "rabin-karp": "#d97706",
};

export function AV2ExperimentsPage() {
  const [data, setData] = useState<AV2ExperimentsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scope, setScope] = useState<"corpus" | "synthetic">("corpus");
  const [patternId, setPatternId] = useState("P-OAUTH2");
  const [fraction, setFraction] = useState(1);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<StringMatchingAlgorithm[]>(algorithmOrder);
  const [logScale, setLogScale] = useState(false);

  useEffect(() => {
    api.av2Experiments()
      .then(setData)
      .catch((requestError: unknown) => setError(requestError instanceof Error ? requestError.message : "Os experimentos da AV2 estão indisponíveis."));
  }, []);

  const patterns = useMemo(() => data?.patterns.filter((pattern) => pattern.scope === scope) ?? [], [data, scope]);
  useEffect(() => {
    if (patterns.length && !patterns.some((pattern) => pattern.pattern_id === patternId)) setPatternId(patterns[0].pattern_id);
  }, [patternId, patterns]);

  if (error) return <><PageHeader eyebrow="AV2 · Benchmark" title="Experimentos — Processamento de Cadeias" description="Naive, KMP, Boyer-Moore e Rabin-Karp em 24 cenários experimentais." /><ErrorState message={error} /></>;
  if (!data) return <><PageHeader eyebrow="AV2 · Benchmark" title="Experimentos — Processamento de Cadeias" description="Naive, KMP, Boyer-Moore e Rabin-Karp em 24 cenários experimentais." /><LoadingState label="Lendo os artefatos congelados da AV2…" /></>;

  const selectedPattern = data.patterns.find((pattern) => pattern.pattern_id === patternId) ?? patterns[0];
  const visibleRows = data.summary.filter((row) => row.pattern_id === selectedPattern?.pattern_id && row.corpus_fraction === fraction && selectedAlgorithms.includes(row.algorithm));
  const fractions = [...new Set(data.summary.map((row) => row.corpus_fraction))].sort((a, b) => a - b);
  const speedupRows = data.speedups.filter((row) => row.pattern_id === selectedPattern?.pattern_id && row.corpus_fraction === fraction && selectedAlgorithms.includes(row.algorithm));
  const memoryRows = data.memory.filter((row) => row.pattern_id === selectedPattern?.pattern_id && row.corpus_fraction === fraction && selectedAlgorithms.includes(row.algorithm));
  const scalability = buildScalabilityData(data, selectedPattern?.pattern_id ?? "", selectedAlgorithms);
  const patternTimes = buildPatternTimes(data, scope, fraction, selectedAlgorithms);
  const memoryData = memoryRows.map((row) => ({ algorithm: algorithmLabels[row.algorithm], key: row.algorithm, memory: row.peak_memory_mib }));
  const componentData = visibleRows.map((row) => ({ algorithm: algorithmLabels[row.algorithm], key: row.algorithm, "Pré-processamento": row.mean_preprocessing_ms, Busca: row.mean_search_ms }));
  const oauthBoyerMoore = data.summary.find((row) => row.pattern_id === "P-OAUTH2" && row.corpus_fraction === 1 && row.algorithm === "boyer-moore");
  const oauthRows = data.summary
    .filter((row) => row.pattern_id === "P-OAUTH2" && row.corpus_fraction === 1)
    .sort((left, right) => left.mean_time_ms - right.mean_time_ms);
  const rankedVisibleRows = [...visibleRows].sort((left, right) => left.mean_time_ms - right.mean_time_ms);
  const fastestRow = rankedVisibleRows[0];
  const secondRow = rankedVisibleRows[1];
  const naiveRow = visibleRows.find((row) => row.algorithm === "naive");
  const fastestAdvantage = fastestRow && naiveRow && fastestRow.algorithm !== "naive"
    ? Math.max(0, (1 - fastestRow.mean_time_ms / naiveRow.mean_time_ms) * 100)
    : 0;
  const bestSpeedup = [...speedupRows].sort((left, right) => right.speedup_vs_naive - left.speedup_vs_naive)[0];
  const directComparisonWinner = [...visibleRows]
    .filter((row) => row.algorithm !== "rabin-karp")
    .sort((left, right) => left.mean_comparisons - right.mean_comparisons)[0];
  const memoryRanking = [...memoryRows].sort((left, right) => left.peak_memory_mib - right.peak_memory_mib);
  const lowestMemory = memoryRanking[0];
  const highestMemory = memoryRanking.at(-1);
  const componentReference = fastestRow;
  const componentTotal = componentReference
    ? componentReference.mean_preprocessing_ms + componentReference.mean_search_ms
    : 0;
  const searchShare = componentReference && componentTotal > 0
    ? componentReference.mean_search_ms / componentTotal * 100
    : 0;
  const scenarioWins = countScenarioWins(data.summary);

  const toggleAlgorithm = (algorithm: StringMatchingAlgorithm) => {
    setSelectedAlgorithms((current) => current.includes(algorithm)
      ? current.length === 1 ? current : current.filter((item) => item !== algorithm)
      : algorithmOrder.filter((item) => current.includes(item) || item === algorithm));
  };

  return (
    <>
      <PageHeader eyebrow="AV2 · Benchmark" title="Experimentos — Processamento de Cadeias" description="Naive, KMP, Boyer-Moore e Rabin-Karp em 24 cenários experimentais." />

      <section className="kpi-grid av2-kpi-grid">
        <KpiCard label="Algoritmos" value={String(data.metadata.algorithm_count)} detail="implementações manuais avaliadas" icon={Cpu} tone="coral" />
        <KpiCard label="Padrões" value={String(data.metadata.pattern_count)} detail="reais e sintéticos" icon={Filter} tone="blue" />
        <KpiCard label="Cenários" value={String(data.metadata.scenario_count)} detail={`${data.metadata.repetitions} repetições temporais`} icon={FlaskConical} tone="green" />
        <KpiCard label="Medições temporais" value={data.metadata.timing_measurements.toLocaleString("pt-BR")} detail={`${data.metadata.timing_combinations} combinações resumidas`} icon={Gauge} tone="amber" />
        <KpiCard label="Combinações de memória" value={String(data.metadata.memory_combinations)} detail="medidas separadamente" icon={MemoryStick} tone="blue" />
        <KpiCard label="Corpus" value={data.metadata.corpus_chunks.toLocaleString("pt-BR")} detail="chunks FastAPI 0.141.0" icon={Database} tone="green" />
      </section>

      <section className="filter-bar av2-filter-bar">
        <div className="filter-title"><Filter size={18} /><span><strong>Recorte experimental</strong><small>todos os gráficos respondem aos filtros</small></span></div>
        <label>Tipo de entrada<select value={scope} onChange={(event) => setScope(event.target.value as "corpus" | "synthetic")}><option value="corpus">Corpus real</option><option value="synthetic">Microbenchmarks sintéticos</option></select></label>
        <label>Padrão<select value={selectedPattern?.pattern_id} onChange={(event) => setPatternId(event.target.value)}>{patterns.map((pattern) => <option value={pattern.pattern_id} key={pattern.pattern_id}>{pattern.pattern}{pattern.scope === "synthetic" ? " · sintético" : ""}</option>)}</select></label>
        <label>Escala<select value={fraction} onChange={(event) => setFraction(Number(event.target.value))}>{fractions.map((value) => <option value={value} key={value}>{Math.round(value * 100)}%</option>)}</select></label>
      </section>
      <section className="algorithm-filter" aria-label="Algoritmos exibidos">{algorithmOrder.map((algorithm) => <label key={algorithm} className={selectedAlgorithms.includes(algorithm) ? "algorithm-chip algorithm-chip--active" : "algorithm-chip"}><input type="checkbox" checked={selectedAlgorithms.includes(algorithm)} onChange={() => toggleAlgorithm(algorithm)} /><i style={{ background: algorithmColors[algorithm] }} />{algorithmLabels[algorithm]}</label>)}</section>

      <section className="analysis-overview panel" aria-label="Resumo do recorte selecionado">
        <div className="analysis-overview__heading"><span>Leitura guiada do recorte</span><h2>{selectedPattern?.pattern} · {Math.round(fraction * 100)}% · {scope === "corpus" ? "corpus real" : "entrada sintética"}</h2><p>{selectedPattern?.description}</p></div>
        <div className="analysis-overview__facts">
          <div><span>O que foi procurado</span><strong>{selectedPattern?.corpus_occurrences.toLocaleString("pt-BR")} ocorrência(s) no corpus integral</strong><p>A busca é exata e sensível a maiúsculas e minúsculas.</p></div>
          <div><span>Melhor tempo neste filtro</span><strong>{fastestRow ? `${algorithmLabels[fastestRow.algorithm]} · ${formatMs(fastestRow.mean_time_ms)}` : "Sem dados"}</strong><p>{fastestRow && secondRow ? `${formatPercent((1 - fastestRow.mean_time_ms / secondRow.mean_time_ms) * 100)} menor que o segundo colocado, ${algorithmLabels[secondRow.algorithm]}.` : "Selecione ao menos dois algoritmos para comparar."}</p></div>
          <div><span>Comparação com a referência</span><strong>{fastestRow && naiveRow ? `${formatPercent(fastestAdvantage)} menos tempo que o Naive` : "Ative o Naive para comparar"}</strong><p>Diferença descritiva das médias observadas; não é teste de significância.</p></div>
        </div>
      </section>

      <section className="section-intro"><div className="section-index">01</div><div><span>Tempo e escala</span><h2>Como o custo muda entre algoritmos e entradas?</h2><p>Tempo em milissegundos. Barras e curvas menores indicam execução mais rápida.</p></div></section>
      <section className="chart-grid">
        <ChartCard title="Tempo médio por algoritmo" description={`${selectedPattern?.pattern ?? "Padrão"} · ${Math.round(fraction * 100)}% da escala`} badge="Menor é melhor">
          <ResponsiveContainer width="100%" height={290}><BarChart data={visibleRows.map(toAlgorithmBar)} margin={{ top: 12, right: 12, left: 3, bottom: 0 }}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="algorithm" tickLine={false} axisLine={false} /><YAxis unit=" ms" width={68} tickLine={false} axisLine={false} /><Tooltip formatter={(value) => formatMs(Number(value))} /><Bar dataKey="tempo" radius={[7, 7, 0, 0]}>{visibleRows.map((row) => <Cell fill={algorithmColors[row.algorithm]} key={row.algorithm} />)}</Bar></BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Como interpretar</strong><p>A altura representa a média de cinco execuções. Neste recorte, {fastestRow ? <><b>{algorithmLabels[fastestRow.algorithm]}</b> foi o mais rápido com <b>{formatMs(fastestRow.mean_time_ms)}</b>{naiveRow && fastestRow.algorithm !== "naive" ? `, usando ${formatPercent(fastestAdvantage)} menos tempo que o Naive` : ""}.</> : <>não há dados para a combinação selecionada.</>}</p></div>
        </ChartCard>
        <ChartCard title="Escalabilidade" description="Caracteres efetivamente analisados; barras de erro mostram ±1 desvio-padrão." badge="5 repetições">
          <ResponsiveContainer width="100%" height={290}><LineChart data={scalability} margin={{ top: 16, right: 24, left: 5, bottom: 0 }}><CartesianGrid strokeDasharray="4 4" stroke="#e7eaf0" /><XAxis dataKey="characters" tickFormatter={compactNumber} tickLine={false} /><YAxis unit=" ms" width={68} tickLine={false} /><Tooltip formatter={(value, name) => { const key = String(name ?? "").replace("Mean", "") as StringMatchingAlgorithm; return [formatMs(Number(value)), algorithmLabels[key] ?? String(name ?? "")]; }} labelFormatter={(value) => `${Number(value).toLocaleString("pt-BR")} caracteres`} /><Legend formatter={(value) => algorithmLabels[value.replace("Mean", "") as StringMatchingAlgorithm] ?? value} />{selectedAlgorithms.map((algorithm) => <Line key={algorithm} type="monotone" dataKey={`${algorithm}Mean`} stroke={algorithmColors[algorithm]} strokeWidth={2.5} dot={{ r: 3 }} connectNulls><ErrorBar dataKey={`${algorithm}Std`} width={4} strokeWidth={1.2} stroke={algorithmColors[algorithm]} /></Line>)}</LineChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Como interpretar</strong><p>{buildScalingExplanation(data, selectedPattern?.pattern_id ?? "", selectedAlgorithms)} As barras de erro mostram a dispersão, não um intervalo de confiança; sobreposição visual não prova igualdade estatística.</p></div>
        </ChartCard>
      </section>

      <section className="chart-grid">
        <ChartCard title="Speedup em relação ao Naive" description="Razão entre o tempo médio do Naive e o tempo do algoritmo." badge="Maior que 1 é melhor">
          <ResponsiveContainer width="100%" height={285}><ComposedChart data={speedupRows.map((row) => ({ algorithm: algorithmLabels[row.algorithm], key: row.algorithm, speedup: row.speedup_vs_naive }))}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="algorithm" tickLine={false} /><YAxis unit="×" width={50} tickLine={false} /><Tooltip formatter={(value) => `${Number(value).toFixed(3)}×`} /><ReferenceLine y={1} stroke="#475569" strokeDasharray="5 5" label={{ value: "Naive 1×", fill: "#64748b", fontSize: 11 }} /><Bar dataKey="speedup" radius={[7, 7, 0, 0]}>{speedupRows.map((row) => <Cell fill={algorithmColors[row.algorithm]} key={row.algorithm} />)}</Bar></ComposedChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Como interpretar</strong><p>Um valor de 2× significa metade do tempo médio do Naive; 0,5× significa o dobro do tempo. {bestSpeedup ? <>O maior ganho do recorte é de <b>{bestSpeedup.speedup_vs_naive.toFixed(3)}×</b> para <b>{algorithmLabels[bestSpeedup.algorithm]}</b>.</> : <>Não há dados para os filtros atuais.</>}</p></div>
        </ChartCard>
        <ChartCard title="Comparações diretas de caracteres" description="Rabin-Karp não contabiliza as operações do rolling hash; os contadores não representam custo total equivalente." badge="Leitura metodológica">
          <ResponsiveContainer width="100%" height={285}><BarChart data={visibleRows.map((row) => ({ algorithm: algorithmLabels[row.algorithm], key: row.algorithm, comparisons: row.mean_comparisons }))}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="algorithm" tickLine={false} /><YAxis tickFormatter={compactNumber} width={62} tickLine={false} /><Tooltip formatter={(value) => Number(value).toLocaleString("pt-BR", { maximumFractionDigits: 0 })} /><Bar dataKey="comparisons" radius={[7, 7, 0, 0]}>{visibleRows.map((row) => <Cell fill={algorithmColors[row.algorithm]} key={row.algorithm} />)}</Bar></BarChart></ResponsiveContainer>
          <div className="chart-analysis chart-analysis--warning"><strong>O que este contador mede</strong><p>Entre Naive, KMP e Boyer-Moore, {directComparisonWinner ? <><b>{algorithmLabels[directComparisonWinner.algorithm]}</b> fez menos comparações explícitas de caracteres ({directComparisonWinner.mean_comparisons.toLocaleString("pt-BR", { maximumFractionDigits: 0 })}).</> : <>não há dados comparáveis.</>} O valor baixo de Rabin-Karp não significa custo total menor, pois cálculo e atualização do hash ficam fora desse contador.</p></div>
        </ChartCard>
      </section>

      <section className="section-intro"><div className="section-index section-index--blue">02</div><div><span>Padrões e memória</span><h2>O formato da entrada muda o comportamento?</h2><p>Casos sintéticos são isolados dos padrões reais porque representam distribuições artificiais.</p></div></section>
      <section className="chart-grid">
        <ChartCard title="Tempo por padrão" description={`${scope === "corpus" ? "Padrões do corpus real" : "Entradas sintéticas"} · ${Math.round(fraction * 100)}%`} badge={logScale ? "Escala log" : "Escala linear"} action={<label className="chart-toggle"><input type="checkbox" checked={logScale} onChange={(event) => setLogScale(event.target.checked)} /> Eixo log</label>}>
          <ResponsiveContainer width="100%" height={310}><BarChart data={patternTimes} margin={{ top: 12, right: 12, left: 3, bottom: 35 }}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="pattern" angle={-18} textAnchor="end" interval={0} height={60} /><YAxis scale={logScale ? "log" : "auto"} domain={logScale ? ["auto", "auto"] : [0, "auto"]} unit=" ms" width={68} /><Tooltip formatter={(value) => formatMs(Number(value))} /><Legend />{selectedAlgorithms.map((algorithm) => <Bar key={algorithm} dataKey={algorithmLabels[algorithm]} fill={algorithmColors[algorithm]} radius={[4, 4, 0, 0]} />)}</BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Como interpretar</strong><p>O tamanho e a estrutura do padrão alteram quantas posições podem ser descartadas. A escala logarítmica ajuda a comparar valores muito diferentes, mas comprime visualmente as distâncias; use a escala linear para discutir diferenças absolutas.</p></div>
        </ChartCard>
        <ChartCard title="Pico de alocações rastreadas" description="tracemalloc em MiB; inclui objetos de saída e não representa RAM total nem memória auxiliar isolada." badge="Medição separada">
          <ResponsiveContainer width="100%" height={310}><BarChart data={memoryData} margin={{ top: 12, right: 12, left: 3, bottom: 0 }}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="algorithm" /><YAxis unit=" MiB" width={72} /><Tooltip formatter={(value) => `${Number(value).toFixed(6)} MiB`} /><Bar dataKey="memory" radius={[7, 7, 0, 0]}>{memoryRows.map((row) => <Cell fill={algorithmColors[row.algorithm]} key={row.algorithm} />)}</Bar></BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Como interpretar</strong><p>{lowestMemory && highestMemory ? <>No recorte, o menor pico rastreado foi de <b>{lowestMemory.peak_memory_mib.toFixed(6)} MiB</b> em {algorithmLabels[lowestMemory.algorithm]}, e o maior foi de <b>{highestMemory.peak_memory_mib.toFixed(6)} MiB</b> em {algorithmLabels[highestMemory.algorithm]}.</> : <>Não há medições para os filtros atuais.</>} Compare tendências, não trate estes valores como consumo total de RAM.</p></div>
        </ChartCard>
      </section>

      <section className="chart-grid chart-grid--single">
        <ChartCard title="Pré-processamento versus busca" description="Componentes internos do algoritmo; a soma não equivale ao mean_time_ms externo, que inclui sobrecarga adicional." badge="Tempos componentes">
          <ResponsiveContainer width="100%" height={300}><BarChart data={componentData} margin={{ top: 12, right: 12, left: 4, bottom: 0 }}><CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#e7eaf0" /><XAxis dataKey="algorithm" /><YAxis unit=" ms" width={68} /><Tooltip formatter={(value) => formatMs(Number(value))} /><Legend /><Bar dataKey="Pré-processamento" stackId="components" fill="#8b5cf6" /><Bar dataKey="Busca" stackId="components" fill="#0ea5e9" radius={[6, 6, 0, 0]} /></BarChart></ResponsiveContainer>
          <div className="chart-analysis"><strong>Como interpretar</strong><p>{componentReference ? <>No algoritmo mais rápido do recorte, {algorithmLabels[componentReference.algorithm]}, a busca representa <b>{formatPercent(searchShare)}</b> da soma dos dois componentes instrumentados.</> : <>Não há dados para o recorte.</>} O tempo médio externo deve continuar sendo a medida principal, pois também inclui iteração, agregação e criação dos resultados.</p></div>
        </ChartCard>
      </section>

      <section className="panel comparison-table-panel av2-table-panel">
        <div className="panel-heading"><div><h2>Tabela completa do recorte</h2><p>Tempo, comparações, speedup e memória unidos pelo cenário e algoritmo.</p></div><button className="secondary-action" onClick={() => exportRows(visibleRows, data)}>Exportar CSV <Download size={15} /></button></div>
        <div className="table-scroll"><table><thead><tr><th>Algoritmo</th><th>Padrão</th><th>Escala</th><th>Caracteres</th><th>Ocorrências</th><th>Tempo médio</th><th>Desvio</th><th>Comparações</th><th>Speedup</th><th>Pico MiB</th></tr></thead><tbody>{visibleRows.map((row) => { const speedup = data.speedups.find((item) => item.scenario_id === row.scenario_id && item.algorithm === row.algorithm); const memory = data.memory.find((item) => item.scenario_id === row.scenario_id && item.algorithm === row.algorithm); return <tr key={`${row.scenario_id}-${row.algorithm}`}><td><span className="algorithm-cell"><i style={{ background: algorithmColors[row.algorithm] }} />{algorithmLabels[row.algorithm]}</span></td><td>{selectedPattern?.pattern}</td><td>{Math.round(row.corpus_fraction * 100)}%</td><td>{row.total_characters.toLocaleString("pt-BR")}</td><td>{row.occurrences.toLocaleString("pt-BR")}</td><td>{formatMs(row.mean_time_ms)}</td><td>± {formatMs(row.std_time_ms)}</td><td>{row.mean_comparisons.toLocaleString("pt-BR", { maximumFractionDigits: 0 })}</td><td>{speedup?.speedup_vs_naive.toFixed(3)}×</td><td>{memory?.peak_memory_mib.toFixed(6)}</td></tr>; })}</tbody></table></div>
      </section>

      <section className="scientific-interpretation panel"><div className="panel-heading"><div><h2>Interpretação científica</h2><p>O que os 24 cenários sustentam, por que o comportamento é plausível e onde a conclusão termina.</p></div><Info size={19} /></div><div className="interpretation-summary"><div><span>Vitórias por menor tempo médio</span><strong>Boyer-Moore {scenarioWins["boyer-moore"] ?? 0} · KMP {scenarioWins.kmp ?? 0}</strong><p>Boyer-Moore venceu a maioria dos padrões reais; KMP venceu todos os cenários repetitivos e sobrepostos.</p></div><div><span>Conferência OAuth2 · 100%</span><strong>{oauthBoyerMoore ? `${formatMs(oauthBoyerMoore.mean_time_ms)} · Boyer-Moore` : "N/D"}</strong><p>Os quatro valores abaixo são lidos do summary.csv, não inseridos manualmente.</p></div></div><div className="interpretation-grid"><div><strong>Boyer-Moore: vantagem por saltos</strong><p>Venceu {scenarioWins["boyer-moore"] ?? 0} de 24 cenários. A heurística bad-character pode avançar várias posições após uma incompatibilidade, o que ajuda em padrões reais médios, longos e ausentes. Isso explica o resultado observado, mas não garante que vencerá em qualquer texto.</p></div><div><strong>KMP: estabilidade em repetição</strong><p>Venceu {scenarioWins.kmp ?? 0} cenários, incluindo as seis escalas sintéticas de AAAA e ABA. Sua tabela de prefixos evita voltar no texto e reaproveita informação quando prefixos e sufixos se repetem ou quando há ocorrências sobrepostas.</p></div><div><strong>Rabin-Karp: contador não é tempo</strong><p>As poucas comparações registradas acontecem apenas quando hashes coincidem. O rolling hash é calculado em Python para cada janela e não aparece nesse contador; por isso é possível exibir poucas comparações e ainda ter o maior tempo.</p></div><div><strong>Naive: linha de base necessária</strong><p>Seu percurso simples facilita a auditoria da corretude e fornece o denominador do speedup. Ser referência não significa ser sempre inadequado: para entradas pequenas, simplicidade, constantes e legibilidade também importam.</p></div></div><div className="interpretation-table"><h3>Estudo de caso: OAuth2 no corpus integral</h3><p>Mesmas ocorrências e posições, mas custos instrumentados diferentes.</p><div className="table-scroll"><table><thead><tr><th>Posição</th><th>Algoritmo</th><th>Tempo médio</th><th>Desvio-padrão</th><th>Speedup vs. Naive</th></tr></thead><tbody>{oauthRows.map((row, index) => { const speedup = data.speedups.find((item) => item.scenario_id === row.scenario_id && item.algorithm === row.algorithm); return <tr key={row.algorithm}><td>{index + 1}º</td><td><span className="algorithm-cell"><i style={{ background: algorithmColors[row.algorithm] }} />{algorithmLabels[row.algorithm]}</span></td><td>{formatMs(row.mean_time_ms)}</td><td>± {formatMs(row.std_time_ms)}</td><td>{speedup?.speedup_vs_naive.toFixed(3)}×</td></tr>; })}</tbody></table></div></div><div className="method-note"><Info size={18} /><p><strong>Limites da inferência:</strong> são cinco repetições por combinação e a análise é descritiva, sem teste de significância ou generalização para outras linguagens, máquinas e distribuições de texto. Memória foi medida separadamente. Microbenchmarks sintéticos isolam comportamentos, mas não representam documentação real. Busca exata não substitui recuperação semântica nem RAG.</p></div></section>

      <details className="environment-panel panel"><summary>Ambiente experimental <span>ver configuração</span></summary><div className="environment-grid"><div><strong>Sistema</strong><p>{String(data.environment.system.os)} {String(data.environment.system.release)} · {String(data.environment.system.architecture)}</p></div><div><strong>CPU</strong><p>{String(data.environment.system.cpu)} · {String(data.environment.system.logical_processors)} processadores lógicos</p></div><div><strong>Memória</strong><p>{String(data.environment.system.ram_gib)} GiB</p></div><div><strong>Python</strong><p>{String(data.environment.python.implementation)} {String(data.environment.python.version)}</p></div></div><div className="package-list">{Object.entries(data.environment.packages).map(([name, version]) => <span key={name}>{name} {version}</span>)}</div></details>
    </>
  );
}

function toAlgorithmBar(row: AV2SummaryRow) { return { algorithm: algorithmLabels[row.algorithm], tempo: row.mean_time_ms }; }

function buildScalabilityData(data: AV2ExperimentsResponse, patternId: string, algorithms: StringMatchingAlgorithm[]) {
  const rows = data.summary.filter((row) => row.pattern_id === patternId && algorithms.includes(row.algorithm));
  return [...new Set(rows.map((row) => row.total_characters))].sort((a, b) => a - b).map((characters) => {
    const point: Record<string, number> = { characters };
    rows.filter((row) => row.total_characters === characters).forEach((row) => { point[`${row.algorithm}Mean`] = row.mean_time_ms; point[`${row.algorithm}Std`] = row.std_time_ms; });
    return point;
  });
}

function buildPatternTimes(data: AV2ExperimentsResponse, scope: "corpus" | "synthetic", fraction: number, algorithms: StringMatchingAlgorithm[]) {
  return data.patterns.filter((pattern) => pattern.scope === scope).map((pattern) => {
    const row: Record<string, string | number> = { pattern: pattern.pattern.length > 18 ? `${pattern.pattern.slice(0, 15)}…` : pattern.pattern };
    data.summary.filter((item) => item.pattern_id === pattern.pattern_id && item.corpus_fraction === fraction && algorithms.includes(item.algorithm)).forEach((item) => { row[algorithmLabels[item.algorithm]] = item.mean_time_ms; });
    return row;
  });
}

function buildScalingExplanation(data: AV2ExperimentsResponse, patternId: string, algorithms: StringMatchingAlgorithm[]) {
  const comparisons = algorithms.map((algorithm) => {
    const rows = data.summary
      .filter((row) => row.pattern_id === patternId && row.algorithm === algorithm)
      .sort((left, right) => left.total_characters - right.total_characters);
    const first = rows[0];
    const last = rows.at(-1);
    if (!first || !last || first.mean_time_ms <= 0 || first.total_characters <= 0) return null;
    return {
      algorithm,
      characterGrowth: last.total_characters / first.total_characters,
      timeGrowth: last.mean_time_ms / first.mean_time_ms,
    };
  }).filter((item): item is { algorithm: StringMatchingAlgorithm; characterGrowth: number; timeGrowth: number } => item !== null);
  if (!comparisons.length) return "Não há pontos suficientes para analisar o crescimento.";
  const mostStable = [...comparisons].sort((left, right) => left.timeGrowth - right.timeGrowth)[0];
  return `Entre a menor e a maior escala, o texto cresceu ${mostStable.characterGrowth.toFixed(2)}× e ${algorithmLabels[mostStable.algorithm]} apresentou o menor crescimento temporal entre os algoritmos visíveis: ${mostStable.timeGrowth.toFixed(2)}×.`;
}

function countScenarioWins(rows: AV2SummaryRow[]) {
  const grouped = new Map<string, AV2SummaryRow[]>();
  rows.forEach((row) => grouped.set(row.scenario_id, [...(grouped.get(row.scenario_id) ?? []), row]));
  const counts: Partial<Record<StringMatchingAlgorithm, number>> = {};
  grouped.forEach((scenarioRows) => {
    const winner = [...scenarioRows].sort((left, right) => left.mean_time_ms - right.mean_time_ms)[0];
    counts[winner.algorithm] = (counts[winner.algorithm] ?? 0) + 1;
  });
  return counts;
}

function exportRows(rows: AV2SummaryRow[], data: AV2ExperimentsResponse) {
  const headers = ["algorithm", "pattern_id", "corpus_fraction", "total_characters", "occurrences", "mean_time_ms", "std_time_ms", "mean_comparisons", "speedup_vs_naive", "peak_memory_mib"];
  const lines = rows.map((row) => { const speedup = data.speedups.find((item) => item.scenario_id === row.scenario_id && item.algorithm === row.algorithm); const memory = data.memory.find((item) => item.scenario_id === row.scenario_id && item.algorithm === row.algorithm); return [row.algorithm, row.pattern_id, row.corpus_fraction, row.total_characters, row.occurrences, row.mean_time_ms, row.std_time_ms, row.mean_comparisons, speedup?.speedup_vs_naive ?? "", memory?.peak_memory_mib ?? ""].join(","); });
  const url = URL.createObjectURL(new Blob([[headers.join(","), ...lines].join("\n")], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = "fastcontext-av2-recorte.csv"; anchor.click(); URL.revokeObjectURL(url);
}

function formatMs(value: number) { return `${value.toLocaleString("pt-BR", { minimumFractionDigits: 3, maximumFractionDigits: 3 })} ms`; }
function formatPercent(value: number) { return `${value.toLocaleString("pt-BR", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%`; }
function compactNumber(value: number) { return Intl.NumberFormat("pt-BR", { notation: "compact", maximumFractionDigits: 1 }).format(value); }
