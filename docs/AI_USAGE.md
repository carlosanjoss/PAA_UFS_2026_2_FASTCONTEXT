# Declaração de Uso de Inteligência Artificial — FastContext

## 1. Objetivo

Este documento registra o uso de ferramentas de Inteligência Artificial durante o desenvolvimento do FastContext.

A IA foi utilizada como **ferramenta de apoio** em engenharia, revisão, documentação e organização metodológica. A responsabilidade final sobre código, resultados, anotações, conclusões e apresentação permanece com a equipe.

---

## 2. Princípio adotado

Saídas de IA não foram tratadas como fonte automática de verdade.

Quando aplicável, sugestões foram verificadas por:

- inspeção do código real;
- execução local;
- testes automatizados;
- análise estática;
- comparação com contratos e configurações;
- validação humana.

---

## 3. Tipos de uso

A IA foi utilizada como apoio em tarefas como:

- brainstorming arquitetural;
- revisão de código;
- organização de módulos;
- geração e revisão de testes;
- diagnóstico de erros;
- documentação técnica;
- infraestrutura experimental;
- revisão de metodologia;
- consistência entre implementação e documentação;
- planejamento e revisão da interface Streamlit;
- preparação de material para relatório e apresentação.

---

## 4. Código

Código sugerido por IA não foi aceito automaticamente.

As alterações relevantes foram submetidas aos gates do projeto:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src
```

---

## 5. Algoritmos clássicos

Os algoritmos avaliados permanecem explícitos no repositório:

- Linear Search;
- Binary Search;
- Inverted Index;
- Merge Sort;
- Quick Sort;
- Top-k Heap.

A IA pôde auxiliar em revisão, testes e documentação, mas os algoritmos medidos não foram substituídos por funções prontas que executassem integralmente a tarefa avaliada.

---

## 6. Corretude e complexidade

IA foi utilizada como apoio na estruturação e revisão de argumentos de corretude e complexidade.

Documento principal:

```text
docs/CORRECTNESS.md
```

As justificativas finais foram vinculadas ao código efetivamente presente no repositório.

---

## 7. Retrieval e RAG

IA foi utilizada para auxiliar na integração e revisão das camadas de retrieval e RAG.

A arquitetura preserva a separação entre:

```text
retrieval
```

e:

```text
LLM generation
```

Isso permite avaliar os algoritmos sem depender do comportamento do modelo generativo.

---

## 8. Streamlit

A implementação da interface Streamlit recebeu assistência de IA por meio do Codex.

A ferramenta foi instruída a:

- auditar o repositório antes de escrever código;
- reutilizar serviços existentes;
- não duplicar algoritmos;
- não alterar corpus;
- não alterar ground truth;
- não alterar resultados experimentais;
- criar testes;
- executar validações;
- produzir um handoff técnico.

Handoff:

```text
docs/STREAMLIT_IMPLEMENTATION_HANDOFF.md
```

---

## 9. Experimentos

IA auxiliou na criação e revisão da infraestrutura experimental, incluindo:

- métricas;
- estatísticas;
- runner;
- pooling;
- adjudicação;
- reporting;
- benchmark Merge Sort × Quick Sort.

Os resultados científicos devem vir da execução dos scripts do repositório, e não de valores sugeridos por IA.

---

## 10. Ground truth

A IA não define os rótulos finais de relevância.

O protocolo utiliza:

```text
2 primary annotators per query
```

Em caso de discordância:

```text
third-person adjudication
```

A IA pode auxiliar na preparação e validação estrutural dos arquivos, mas não substitui o julgamento humano.

---

## 11. Pooling

A IA auxiliou na infraestrutura de pooling.

Os arquivos destinados aos anotadores ocultam:

- algoritmo de origem;
- score;
- rank de origem.

---

## 12. Resultados experimentais

Nenhum resultado deve ser preenchido a partir de expectativa gerada por IA.

Métricas como tempo, memória, comparações, Precision, Recall, MRR e Hit Rate devem ser produzidas pela infraestrutura experimental.

---

## 13. Interpretação

IA pode auxiliar na interpretação dos resultados, mas toda interpretação deve ser confrontada com:

- resultados brutos;
- tabelas;
- gráficos;
- código;
- configurações;
- teoria algorítmica.

---

## 14. Documentação

IA foi utilizada como apoio editorial em documentos Markdown e poderá auxiliar na preparação de relatório e slides.

O material final deve permanecer consistente com código, configuração, testes, resultados e histórico Git.

---

## 15. Registros específicos

Registros adicionais podem existir por integrante.

Exemplo:

```text
docs/mateus/AI_USAGE_LOG.md
```

---

## 16. Responsabilidade humana

A equipe permanece responsável por:

- aceitar ou rejeitar sugestões;
- revisar código;
- executar testes;
- validar resultados;
- verificar referências;
- explicar os algoritmos;
- defender decisões metodológicas;
- interpretar os experimentos;
- apresentar o trabalho.

---

## 17. Limitações conhecidas

Ferramentas de IA podem:

- sugerir APIs inexistentes;
- interpretar contratos incorretamente;
- gerar código incompatível;
- produzir análises genéricas;
- declarar incorretamente que algo foi testado;
- gerar fatos ou referências erradas.

Por isso, nenhuma saída deve ser aceita apenas por ter sido produzida com alta confiança.

---

## 18. Fluxo de validação

```text
AI suggestion
     |
     v
Human/code review
     |
     v
Execution
     |
     v
Automated tests
     |
     v
Static validation
     |
     v
Scientific consistency review
     |
     v
Acceptance
```

---

## 19. Declaração final

O FastContext utilizou IA generativa como ferramenta de apoio ao desenvolvimento, revisão e documentação.

As implementações, medições, anotações humanas, resultados e conclusões da entrega final devem permanecer verificáveis e reproduzíveis a partir do repositório.

O uso de IA não substitui a responsabilidade técnica e científica da equipe.
