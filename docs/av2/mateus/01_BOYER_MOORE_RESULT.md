# AV2 — Fase 1: Boyer–Moore (bad-character)

**Responsável:** Mateus Almeida
**Branch:** `feature/av2-boyer-moore-rabin-karp`
**Data:** 22/10/2026

---

## 1. Variante implementada

Boyer–Moore **com heurística bad-character apenas**.

> **Importante (honestidade):** esta implementação **não** possui a heurística *good-suffix*. Portanto **não** deve ser descrita como "Boyer–Moore completo". O pior caso de busca permanece `O(n·m)`.

## 2. Arquivos

| Arquivo | Ação |
|---|---|
| `src/string_matching/boyer_moore.py` | **CRIADO** — `BoyerMooreMatcher` |
| `src/string_matching/registry.py` | **ALTERADO** — registra `"boyer-moore"` em `create_default_registry()` |
| `tests/test_av2_boyer_moore.py` | **CRIADO** — 25 testes |
| `tests/test_av2_registry.py` | **ALTERADO** — asserção de `available_names()` atualizada para incluir `"boyer-moore"` (ponto de integração; alteração pontual em teste do Carlos, a confirmar no PR) |

## 3. Assinatura e contrato

Segue o contrato real do Carlos (`StringMatcher` / `StringMatchResult`):

```python
class BoyerMooreMatcher(StringMatcher):
    @property
    def name(self) -> str: ...           # "boyer-moore"

    def search(self, text: str, pattern: str) -> StringMatchResult: ...
```

- `validate_input` herdado: pattern vazio → `ValueError`; tipos errados → `TypeError`.
- `preprocessing_time_ns` mede **apenas** a construção da tabela de última ocorrência.
- `search_time_ns` mede **apenas** o núcleo de busca.
- `comparisons` conta **apenas** comparações reais `text[...] == pattern[...]`.
- `positions`: índices base-zero, ordem crescente, **com sobreposições**.

## 4. Pré-processamento — tabela de última ocorrência

Mapa `char -> índice da ocorrência mais à direita no padrão`. Caracteres
ausentes são tratados como `-1` pela busca (via `.get(char, -1)`).

Exemplo `pattern = "ABCA"`:

```
A -> 3
B -> 1
C -> 2
```

## 5. Busca (direita para a esquerda)

Para cada alinhamento do padrão sobre o texto:

1. compara do último caractere do padrão para o primeiro;
2. a cada comparação, incrementa `comparisons`;
3. se todos baterem → registra a posição e **avança 1** (preserva sobreposição);
4. se houver *mismatch* no caractere `c` do texto na posição `j` do padrão:
   - `shift = j - last_occurrence.get(c, -1)`;
   - avança `shift` se `shift > 0`, senão avança `1` (garante progresso).

## 6. Exemplo mínimo reproduzível

```python
from src.string_matching.boyer_moore import BoyerMooreMatcher

m = BoyerMooreMatcher()
print(m.search("abababab", "abab").positions)  # (0, 2, 4)
```

## 7. Corretude (argumento)

- **Não perde ocorrências:** o deslocamento bad-character nunca pula um
  alinhamento que poderia conter um match, pois move o padrão no máximo
  até alinhar a última ocorrência do caractere conflitante; quando esse
  cálculo seria ≤ 0, o avanço é forçado para 1.
- **Progresso/terminação:** o alinhamento aumenta em pelo menos 1 a cada
  iteração do laço externo, que termina em `alignment <= n - m`.
- **Todas as ocorrências / sobreposição:** após um match completo o avanço
  é de exatamente 1, preservando ocorrências sobrepostas (ex.: `"aaaaa"` /
  `"aaa"` → `(0, 1, 2)`).

## 8. Complexidade

| Aspecto | Custo |
|---|---|
| Pré-processamento (tabela) | `O(m)` tempo, `O(k)` espaço (`k` = caracteres distintos do padrão) |
| Busca — melhor/caso bom | sublinear: pode pular trechos do texto |
| Busca — **pior caso** | `O(n·m)` (sem good-suffix) |
| Espaço auxiliar total | `O(k)` |

`n = len(text)`, `m = len(pattern)`.

## 9. Testes (evidência)

Comando:

```
python -m pytest tests/test_av2_boyer_moore.py -q
```

Resultado: **25 PASSOU** (cobrindo T01–T08 do contrato + padrão = texto,
padrão de 1 caractere, match só no início, match só no fim, case-sensitive,
caracteres especiais, Unicode `café`, e verificação das métricas).

Suíte de domínio AV2 (naive, kmp, base, models, registry, boyer-moore):
**59 PASSOU**.

### Bloqueios ambientais (honestidade)
- `tests/test_av2_api.py` e `tests/test_av2_string_matching_service.py`:
  **BLOQUEADO / NÃO EXECUTADO** — erro de coleta por `ModuleNotFoundError:
  No module named 'dotenv'` (dependência `python-dotenv` ausente no
  ambiente, puxada pela cadeia RAG em `src/rag/settings.py`). **Não é
  relacionado ao Boyer–Moore.**
- Docker: **NÃO EXECUTADO**.

## 10. Limitações e pendências

- Sem good-suffix (pior caso `O(n·m)`).
- Equivalência formal contra a baseline Naive será consolidada na Fase 4
  (já há cenários canônicos cobertos nos testes unitários).
- Não há campo `metadata` no contrato → nenhuma métrica extra exposta.

---

## FIM DA FASE 1 — aguardando revisão de Mateus antes da Fase 2 (Rabin–Karp).
