# AV2 — Handoff Final (Mateus → Carlos / material para Gabriella)

**Responsável:** Mateus Almeida
**Branch:** `feature/av2-boyer-moore-rabin-karp` (a partir de `origin/main`)
**Escopo entregue:** Boyer–Moore (bad-character) e Rabin–Karp (rolling hash)

---

## 1. Link (preencher após push/PR)

```
branch : feature/av2-boyer-moore-rabin-karp
commit : <HEAD após push>
PR     : <URL do Pull Request>
```

## 2. Arquivos entregues

**Núcleo:**
- `src/string_matching/boyer_moore.py` — `BoyerMooreMatcher` (`name="boyer-moore"`)
- `src/string_matching/rabin_karp.py` — `RabinKarpMatcher` (`name="rabin-karp"`)

**Integração (pontual):**
- `src/string_matching/registry.py` — registra `"boyer-moore"` e `"rabin-karp"` em `create_default_registry()`
- `tests/test_av2_registry.py` — asserção de `available_names()` atualizada

**Testes:**
- `tests/test_av2_boyer_moore.py`
- `tests/test_av2_rabin_karp.py`
- `tests/test_av2_matching_equivalence.py`

**Documentação:**
- `docs/av2/BM_RK_ANALYSIS.md` (corretude + complexidade)
- `docs/av2/mateus/00_AUDIT_RESULT.md` ... `03_TESTS_RESULT.md`, `AI_USAGE_LOG.md`, este arquivo.

## 3. Comando de reprodução

```bash
python -m pytest tests/test_av2_boyer_moore.py tests/test_av2_rabin_karp.py tests/test_av2_matching_equivalence.py -v
```

Suíte de string matching completa:

```bash
python -m pytest tests/test_av2_base.py tests/test_av2_models.py \
  tests/test_av2_naive.py tests/test_av2_kmp.py tests/test_av2_registry.py \
  tests/test_av2_boyer_moore.py tests/test_av2_rabin_karp.py \
  tests/test_av2_matching_equivalence.py -q
```

## 4. Exemplo mínimo

```python
from src.string_matching.registry import create_default_registry

reg = create_default_registry()
reg.create("boyer-moore").search("abababab", "abab").positions  # (0, 2, 4)
reg.create("rabin-karp").search("abababab", "abab").positions   # (0, 2, 4)
```

## 5. Evidência de testes

- Testes de Mateus (BM/RK/equivalência): **60 PASSOU**.
- Suíte AV2 de string matching: **101 PASSOU**.
- Equivalência com a baseline Naive (Wilson): **PASSOU** (canônicos + 800
  casos de fuzzing com seed fixa).
- Colisão RK sem falso positivo: **PASSOU**.
- `test_av2_api.py` / `test_av2_string_matching_service.py`: **BLOQUEADO**
  por dependências ausentes no ambiente (`dotenv`, `fastapi`, `pydantic`)
  — ver `03_TESTS_RESULT.md`. Não relacionado a BM/RK.
- Docker: **NÃO EXECUTADO**.

## 6. Integração com a API/serviço

Os dois algoritmos já aparecem em `registry.available_names()`, portanto o
endpoint `GET /api/av2/string-matching/algorithms` passa a listá-los e o
`POST /search` aceita `algorithm = "boyer-moore"` ou `"rabin-karp"` sem
alteração adicional no service do Carlos.

## 7. Limitações (explícitas)

- **Boyer–Moore** usa somente bad-character (não é a versão completa);
  pior caso `O(n·m)`.
- **Rabin–Karp** pode degradar para `O(n·m)` com muitas colisões/alta
  repetição; métricas internas de hash não são expostas (contrato sem
  `metadata`).
- **Benchmark final, gráficos e métricas agregadas** pertencem ao Carlos.
- **Corpus/cenários** pertencem ao Yann; **baseline Naive/KMP** ao Wilson.

## 8. Material para Gabriella (relatório)

- Pseudocódigo BM e RK: `docs/av2/BM_RK_ANALYSIS.md`, seções 1.3 e 2.2.
- Variante BM: bad-character (sem good-suffix).
- Estratégia de hash RK: polinomial base 256, módulo primo, rolling O(1).
- Corretude: seções 1.4 e 2.3.
- Complexidade: seções 1.5 e 2.4.
- Limitações: seção 6 do mesmo documento.
- Evidência de testes: `docs/av2/mateus/03_TESTS_RESULT.md`.
- Contribuição individual: Boyer–Moore, Rabin–Karp, testes unitários,
  testes de equivalência e documentação técnica correspondente.

> Não há números de benchmark fornecidos por Mateus (não executados);
> esses dados são responsabilidade do Carlos.
