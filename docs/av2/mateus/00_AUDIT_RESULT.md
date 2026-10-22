# AV2 — Relatório de Auditoria (Fase 0)

**Responsável:** Mateus Almeida
**Escopo:** Boyer–Moore (bad-character) e Rabin–Karp (rolling hash)
**Data da auditoria:** 22/10/2026
**Regra aplicada:** somente leitura — nenhum código de algoritmo foi modificado nesta fase.

---

## 1. Estado do Git

| Item | Valor observado |
|---|---|
| Remoto | `https://github.com/carlosanjoss/PAA_UFS_2026_2_FASTCONTEXT` |
| Branch local atual (no momento da auditoria) | `feature/tests` (base da AV1) |
| Working tree | limpo |
| HEAD da `origin/main` | `b1b8f15` — *Merge pull request #7 from carlosanjoss/av2-string-matching* |
| Branch da AV2 | `origin/av2-string-matching` (já mergeada na `main`) |

### Branches relevantes
- `origin/main` — contém **toda a AV2 de string matching já integrada** (PRs #5, #6, #7).
- `origin/av2-string-matching` — branch de desenvolvimento da AV2.
- `feature/tests`, `feature/classical-algorithms` — artefatos locais da AV1.

**Conclusão Git:** a base de trabalho correta para a AV2 é a **`origin/main` atualizada**, e não a `feature/tests` local (que é AV1). Boyer–Moore e Rabin–Karp **ainda não existem** em nenhuma ref.

---

## 2. Estrutura AV2 encontrada (em `origin/main`)

```
src/string_matching/
  __init__.py
  base.py          # contrato StringMatcher (ABC)
  models.py        # StringMatchResult (dataclass frozen)
  naive.py         # baseline Wilson
  kmp.py           # KMP Wilson
  registry.py      # StringMatcherRegistry + create_default_registry()
  api.py           # rotas FastAPI (Carlos)
src/services/
  string_matching_service.py   # serviço sobre o corpus (Carlos)
data/av2/
  patterns.json
  pattern_candidates.json
  scenarios.json
scripts/av2/
  prepare_inputs.py            # Yann
docs/av2/
  DATA_CONTRACT.md
  DATA_PROTOCOL.md
  NAIVE_KMP_ANALYSIS.md
tests/
  test_av2_base.py
  test_av2_models.py
  test_av2_naive.py
  test_av2_kmp.py
  test_av2_registry.py
  test_av2_api.py
  test_av2_string_matching_service.py
  test_av2_prepare_inputs.py
```

### Divergências em relação ao guia `KIRO_MATEUS_FASTCONTEXT_AV2.md`
A estrutura real prevalece (seção 5 do contrato). Diferenças confirmadas:

| Guia sugeria | Repositório real (convenção do Carlos) |
|---|---|
| `tests/av2/test_boyer_moore_rabin_karp.py` | `tests/test_av2_*.py` (prefixo, sem subpasta) |
| `docs/av2/mateus/...` | docs técnicos ficam em `docs/av2/` (ex.: `NAIVE_KMP_ANALYSIS.md`) |
| `metadata: dict` no resultado | **não existe** campo `metadata` no `StringMatchResult` |

---

## 3. Contrato real (definido por Carlos)

### 3.1 Interface `StringMatcher` (`src/string_matching/base.py`)
- Classe abstrata (`ABC`).
- Propriedade abstrata `name -> str` (nome canônico do algoritmo).
- Método abstrato `search(text: str, pattern: str) -> StringMatchResult`.
- `validate_input(text, pattern)` estático:
  - `text` não-str → `TypeError`
  - `pattern` não-str → `TypeError`
  - `pattern` vazio → **`ValueError`**

### 3.2 `StringMatchResult` (`src/string_matching/models.py`)
`@dataclass(frozen=True, slots=True)` com os campos:

| Campo | Tipo | Observação |
|---|---|---|
| `algorithm` | `str` | nome canônico |
| `positions` | `tuple[int, ...]` | índices base-zero, ordem crescente |
| `comparisons` | `int` | comparações reais de caracteres |
| `preprocessing_time_ns` | `int` | tempo de pré-processamento |
| `search_time_ns` | `int` | tempo do núcleo de busca |
| `text_length` | `int` | |
| `pattern_length` | `int` | |

Propriedades derivadas: `occurrences` (= `len(positions)`), `total_time_ns` (= pré + busca).

**Não há campo `metadata`.** Isso significa que `hash_candidates`/`collisions` do Rabin–Karp **não cabem no contrato público atual**. Opções (a decidir com a equipe): (a) não expor essas métricas no resultado e validar colisões apenas via teste controlado; (b) solicitar ao Carlos a adição de um campo opcional. Recomendação: **opção (a)** para não alterar o contrato do Carlos.

### 3.3 Registro de algoritmos (`registry.py`)
- `create_default_registry()` registra hoje apenas `"naive"` e `"kmp"`.
- Regra de validação: `matcher.name` deve ser igual ao nome normalizado (lower/trim) usado no registro.
- **Ponto de integração de Mateus:** registrar `"boyer-moore"` e `"rabin-karp"` aqui (nome canônico a confirmar; o service expõe `available_names()` na API).

### 3.4 Padrão de implementação observado (Naive e KMP)
- `self.validate_input(text, pattern)` no início.
- Pré-processamento cronometrado com `perf_counter_ns()` **isoladamente** (KMP mede a construção da LPS separada da busca; Naive tem `preprocessing_time_ns=0`).
- `comparisons` incrementado a cada comparação real `text[i] == pattern[j]`.
- Sobreposições preservadas (KMP usa `lps[j-1]` após match; Naive percorre todo `i`).
- Retorno via `StringMatchResult(...)`.

Este é exatamente o molde a ser seguido por Boyer–Moore e Rabin–Karp.

---

## 4. Baseline de Wilson (oráculo de equivalência)

| Arquivo | Estado |
|---|---|
| `src/string_matching/naive.py` | **PRESENTE** — `NaiveMatcher`, `name="naive"` |
| `src/string_matching/kmp.py` | **PRESENTE** — `KMPMatcher`, `name="kmp"` |
| `tests/test_av2_naive.py` | **PRESENTE** — cobre único/múltiplo/ausente/vazio/sobreposição/métricas |
| `tests/test_av2_kmp.py` | **PRESENTE** |

**Naive está disponível e validado** → servirá como **oráculo de equivalência** para Boyer–Moore e Rabin–Karp (Fase 4). Dependência externa satisfeita.

Casos canônicos confirmados no teste do Naive:
- `search("ABABABA", "ABA") == (0, 2, 4)` (sobreposição)
- `search("AAAA", "AA")` → posições `(0,1,2)`, `comparisons == 6`
- pattern vazio → `ValueError`; texto vazio → `()`; pattern > texto → `()`.

---

## 5. Cenários de Yann

| Arquivo | Estado |
|---|---|
| `data/av2/patterns.json` | **PRESENTE** — status `ready` |
| `data/av2/scenarios.json` | **PRESENTE** — status `ready` |
| `scripts/av2/prepare_inputs.py` | **PRESENTE** |

- Corpus: FastAPI Documentation `0.141.0`, 153 documentos, 1305 chunks.
- `fingerprint == known_av1_fingerprint` → `matches_known_av1: true` (**corpus da AV1 preservado**).
- Padrões verificados com contagens reais (ex.: `"OAuth2"` → 103 ocorrências; `"dependency"` → 220; `"a"` → 43612).
- Cenários com frações `0.25 / 0.5 / 1.0`, seleção determinística por `chunk_id` ascendente, `expected_case` registrado.

Cenários disponíveis e reproduzíveis. **Não inventar benchmark** — o agregado final é do Carlos.

---

## 6. Dependências e ambiente

| Item | Valor |
|---|---|
| Python (sistema e `.venv`) | 3.13.1 |
| pytest | 9.1.1 |
| Sistema Operacional | Windows (win32), shell PowerShell |
| Docker | **NÃO EXECUTADO** (não testado nesta fase) |

---

## 7. Testes baseline

- Comando padrão do projeto: `python -m pytest`
- Suíte AV2 existente: `tests/test_av2_*.py` (8 arquivos).
- **Ainda NÃO EXECUTADA** nesta fase (auditoria somente leitura, e o working tree está em `feature/tests`/AV1). Será executada no início da Fase 1, após alinhar a branch de trabalho com a `main`.

> Nenhuma afirmação de "testes passaram" é feita neste relatório.

---

## 8. Riscos identificados

1. **Branch de trabalho desalinhada:** o working tree local está em `feature/tests` (AV1). É preciso decidir a branch da AV2 antes de implementar (ver seção 10).
2. **Ausência de `metadata` no contrato:** métricas específicas do Rabin–Karp (`hash_candidates`, `collisions`) não têm onde morar no `StringMatchResult`. Decisão recomendada: validar colisão via teste controlado, sem alterar o contrato do Carlos.
3. **Nome canônico dos algoritmos:** definir `"boyer-moore"` e `"rabin-karp"` (com hífen) de forma consistente com o service/API. A confirmar antes de registrar.
4. **AV1 congelada:** não tocar corpus, ground truth, retrievers, resultados nem docs da AV1.

---

## 9. Arquivos que Mateus deverá criar (plano, sujeito a aprovação)

**Código (núcleo):**
- `src/string_matching/boyer_moore.py` — `BoyerMooreMatcher` (bad-character).
- `src/string_matching/rabin_karp.py` — `RabinKarpMatcher` (rolling hash).

**Integração (mínima, no arquivo do Carlos):**
- `src/string_matching/registry.py` — registrar os dois novos matchers em `create_default_registry()`. *(Alteração pontual; confirmar com Carlos.)*

**Testes (convenção real `tests/test_av2_*.py`):**
- `tests/test_av2_boyer_moore.py`
- `tests/test_av2_rabin_karp.py`
- (equivalência contra Naive, possivelmente em um dos arquivos acima ou em `tests/test_av2_matching_equivalence.py`.)

**Documentação:**
- `docs/av2/BM_RK_ANALYSIS.md` — corretude e complexidade (convenção `docs/av2/`).
- `docs/av2/mateus/00_AUDIT_RESULT.md` (este arquivo), `01_BOYER_MOORE_RESULT.md`, `02_RABIN_KARP_RESULT.md`, `03_TESTS_RESULT.md`, `AI_USAGE_LOG.md`, `FINAL_HANDOFF.md`.

---

## 10. Decisão pendente antes da Fase 1 (requer Mateus)

**Estratégia de branch.** A AV2 já está na `main`. Opções:
- **(A) Recomendada:** criar branch nova a partir de `origin/main`, ex. `feature/av2-boyer-moore-rabin-karp`, implementar lá e abrir PR. Não fazer merge na `main` sem autorização.
- (B) Reaproveitar `av2-string-matching` — desaconselhado, pois é a branch integradora do Carlos.

Confirmar também: nome canônico dos algoritmos e tratamento das métricas do RK (seção 8, riscos 2 e 3).

---

## 11. Estado por item (honestidade)

| Item | Estado |
|---|---|
| Contrato real localizado | **CONCLUÍDO** |
| Baseline Naive disponível | **CONCLUÍDO** (oráculo pronto) |
| Cenários do Yann disponíveis | **CONCLUÍDO** |
| Ambiente (Python/pytest) | **CONCLUÍDO** |
| Docker | **NÃO EXECUTADO** |
| Suíte baseline executada | **PENDENTE** (início da Fase 1) |
| Boyer–Moore | **PENDENTE** |
| Rabin–Karp | **PENDENTE** |

---

## FIM DA FASE 0 — aguardando revisão humana (Mateus) antes de implementar Boyer–Moore / Rabin–Karp.
