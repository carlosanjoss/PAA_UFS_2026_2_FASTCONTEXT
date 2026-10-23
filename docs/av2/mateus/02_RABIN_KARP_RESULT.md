# AV2 — Fase 2: Rabin–Karp (rolling hash)

**Responsável:** Mateus Almeida
**Branch:** `feature/av2-boyer-moore-rabin-karp`
**Data:** 22/10/2026

---

## 1. Variante implementada

Rabin–Karp com **rolling hash polinomial manual** e **verificação
obrigatória caractere a caractere** sempre que o hash da janela coincide
com o hash do padrão.

## 2. Arquivos

| Arquivo | Ação |
|---|---|
| `src/string_matching/rabin_karp.py` | **CRIADO** — `RabinKarpMatcher` |
| `src/string_matching/registry.py` | **ALTERADO** — registra `"rabin-karp"` |
| `tests/test_av2_rabin_karp.py` | **CRIADO** — 15 testes (inclui colisão T10) |
| `tests/test_av2_matching_equivalence.py` | **CRIADO** — equivalência BM/RK vs Naive (Fase 4 antecipada) |
| `tests/test_av2_registry.py` | **ALTERADO** — `available_names()` inclui `"rabin-karp"` |

## 3. Assinatura e contrato

```python
class RabinKarpMatcher(StringMatcher):
    def __init__(self, base: int = 256, modulus: int = 1_000_000_007) -> None: ...

    @property
    def name(self) -> str: ...           # "rabin-karp"

    def search(self, text: str, pattern: str) -> StringMatchResult: ...
```

- `base`/`modulus` configuráveis **apenas** para permitir forçar colisões
  em teste; os padrões (base 256, módulo primo grande) são usados em
  produção. Isso **não altera** o contrato público `search(text, pattern)`.
- `validate_input` herdado: pattern vazio → `ValueError`.
- `comparisons` conta **apenas** comparações reais de caractere feitas na
  verificação. **Igualdade de hash NÃO é contada** como comparação de
  caractere (conforme seção 9.1 do contrato de Mateus).

## 4. Rolling hash (manual)

1. `high_order = base^(m-1) mod modulus` (pré-computado).
2. hash do padrão e hash da primeira janela computados caractere a
   caractere, módulo `modulus`.
3. a cada passo, se `window_hash == pattern_hash` → **verificação char a
   char**;
4. roll para a próxima janela:
   `h = ((h - code(saindo)*high_order) * base + code(entrando)) mod modulus`.

Como Python usa inteiros de precisão arbitrária e `%` retorna resultado
não-negativo para módulo positivo, não há problema de hash negativo.

## 5. Verificação obrigatória (sem falso positivo)

`window_hash == pattern_hash` **nunca** é tratado como prova de match.
Sempre há verificação caractere a caractere. Logo, uma colisão de hash
(hashes iguais, conteúdos diferentes) é rejeitada na verificação.

### Teste de colisão controlada (T10)
Com `base=10, modulus=11`:
- hash("19") = (1·10 + 9) mod 11 = 19 mod 11 = **8**
- hash("08") = (0·10 + 8) mod 11 = **8**
- `"19" != "08"`, mas os hashes colidem.

`search("08", "19")` → `positions == ()` (nenhum falso positivo) e
`comparisons >= 1` (a verificação ocorreu). Match verdadeiro permanece
detectável: `search("xx19yy19zz", "19")` → `(2, 6)`.

## 6. Exemplo mínimo reproduzível

```python
from src.string_matching.rabin_karp import RabinKarpMatcher

m = RabinKarpMatcher()
print(m.search("abababab", "abab").positions)  # (0, 2, 4)
```

## 7. Complexidade

| Aspecto | Custo |
|---|---|
| Pré-processamento (hashes + high_order) | `O(m)` |
| Rolling hash por janela | `O(1)` amortizado |
| Caso esperado/médio | `O(n + m)` (poucas verificações completas) |
| **Pior caso** | `O(n·m)` quando muitas janelas colidem ou o texto é muito repetitivo |
| Espaço auxiliar | `O(1)` (fora da lista de posições) |

> **Honestidade:** o pior caso `O(n·m)` **não é escondido** — ocorre quando
> a verificação caractere a caractere é acionada com frequência.

## 8. Decisão de contrato — métricas do RK

O `StringMatchResult` do Carlos **não possui campo `metadata`**. Portanto
`hash_candidates` e `collisions` **não são expostos** no resultado para não
alterar o contrato. A ausência de falso positivo por colisão é garantida e
comprovada **via teste controlado** (seção 5), não via metadados.

## 9. Testes (evidência)

Comandos:

```
python -m pytest tests/test_av2_rabin_karp.py -q
python -m pytest tests/test_av2_matching_equivalence.py -q
```

- `test_av2_rabin_karp.py`: **15 PASSOU** (T01–T08, padrão=texto, 1 char,
  início/fim, case-sensitive, métricas, colisão T10 sem falso positivo).
- `test_av2_matching_equivalence.py`: **23 PASSOU** — BM e RK retornam as
  mesmas posições que o Naive em casos canônicos e em **800 casos
  aleatórios** com seed fixa (fuzzing diferencial).
- Suíte de domínio AV2 completa (naive, kmp, base, models, registry,
  boyer-moore, rabin-karp, equivalência): **101 PASSOU**.

### Bloqueios ambientais (honestidade)
- `tests/test_av2_api.py`, `tests/test_av2_string_matching_service.py`:
  **BLOQUEADO** — `ModuleNotFoundError: No module named 'dotenv'` (não
  relacionado ao RK).
- Docker: **NÃO EXECUTADO**.

## 10. Limitações e pendências

- Pior caso `O(n·m)` com colisões/alta repetição.
- Documento consolidado de corretude/complexidade (`docs/av2/BM_RK_ANALYSIS.md`)
  será a Fase 5.

---

## FIM DA FASE 2 — aguardando revisão de Mateus antes da Fase 5 (documentação consolidada) e Git/handoff.
