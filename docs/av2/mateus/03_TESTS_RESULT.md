# AV2 — Fase 6: Resultado dos Testes

**Responsável:** Mateus Almeida
**Branch:** `feature/av2-boyer-moore-rabin-karp`
**Data:** 22/10/2026

---

## 1. Ambiente

| Item | Valor |
|---|---|
| Sistema Operacional | Windows (win32) |
| Shell | PowerShell |
| Python | 3.13.1 (`.venv`) |
| pytest | 9.1.1 |
| Docker | **NÃO EXECUTADO** (indisponível/não testado no ambiente) |

### Dependências do ambiente (verificado)
| Pacote | Estado |
|---|---|
| `yaml` (PyYAML) | OK |
| `numpy` | OK |
| `dotenv` (python-dotenv) | **MISSING** |
| `pydantic` | **MISSING** |
| `fastapi` | **MISSING** |
| `faiss` | **MISSING** |
| `sentence_transformers` | **MISSING** |

---

## 2. Testes do escopo de Mateus (Boyer–Moore / Rabin–Karp / equivalência)

Comando:

```
python -m pytest tests/test_av2_boyer_moore.py tests/test_av2_rabin_karp.py tests/test_av2_matching_equivalence.py -v
```

Resultado: **60 PASSOU** / 0 falhou.

Detalhe:
- `tests/test_av2_boyer_moore.py` — 25 testes — **PASSOU**
- `tests/test_av2_rabin_karp.py` — 15 testes — **PASSOU**
- `tests/test_av2_matching_equivalence.py` — 23 testes — **PASSOU**
  - inclui 800 casos de fuzzing diferencial com seed fixa contra o Naive.

> Observação: 25 + 15 + 23 = 63 funções de teste; algumas são
> parametrizadas/contadas em conjunto. O total coletado pelo pytest para
> esses três arquivos foi **60 itens**, todos PASSOU.

## 3. Suíte de domínio AV2 (string matching, sem cadeia RAG)

Comando:

```
python -m pytest tests/test_av2_base.py tests/test_av2_models.py \
  tests/test_av2_naive.py tests/test_av2_kmp.py tests/test_av2_registry.py \
  tests/test_av2_boyer_moore.py tests/test_av2_rabin_karp.py \
  tests/test_av2_matching_equivalence.py -q
```

Resultado: **101 PASSOU** / 0 falhou.

Cobre: contrato base, modelos, Naive (Wilson), KMP (Wilson), registry
(com os 4 algoritmos), Boyer–Moore, Rabin–Karp e equivalência.

## 4. Suíte global (`pytest -k av2` sobre todo o projeto)

Resultado: **20 erros de coleta** (collection errors), 371 deselecionados.

- **Causa:** `ModuleNotFoundError` para `dotenv`, `pydantic`, `fastapi`,
  `faiss`, `sentence_transformers` — dependências **ausentes no ambiente**,
  puxadas por testes de API/RAG/semântica/serviço.
- **Arquivos AV2 afetados:** `tests/test_av2_api.py`,
  `tests/test_av2_string_matching_service.py` — **BLOQUEADO / NÃO EXECUTADO**.
- **Relação com Mateus:** **nenhuma.** Boyer–Moore e Rabin–Karp não
  importam essas dependências. Os erros são ambientais e pré-existentes.

> Honestidade: a ausência dessas dependências **não** é tratada como
> "PASSOU". É registrada como **BLOQUEADO / NÃO EXECUTADO**.

## 5. Evidências específicas exigidas pelo contrato

| Requisito | Evidência | Estado |
|---|---|---|
| Sobreposição | `"abababab"/"abab"` → `(0,2,4)`; `"aaaaa"/"aaa"` → `(0,1,2)` | PASSOU |
| Unicode | `"café café"/"café"` → `(0,5)` | PASSOU |
| Padrão vazio → `ValueError` | BM e RK | PASSOU |
| Texto vazio → `()` | BM e RK | PASSOU |
| Padrão > texto → `()` | BM e RK | PASSOU |
| Case-sensitive | `"API api API"/"API"` → `(0,8)` | PASSOU |
| RK rejeita falso positivo por colisão | `base=10,mod=11`: `search("08","19")` → `()` | PASSOU |
| Comparações instrumentadas | contadas só em comparação real de caractere | PASSOU |
| Pré-processamento e busca separados | `preprocessing_time_ns` / `search_time_ns` | PASSOU |
| Posições em ordem crescente | garantido pela varredura | PASSOU |
| Equivalência com Naive | 800 casos aleatórios + canônicos | PASSOU |

## 6. Resumo por estado (honestidade)

| Item | Estado |
|---|---|
| Testes de Mateus (BM/RK/equivalência) | **PASSOU (60)** |
| Suíte AV2 string matching | **PASSOU (101)** |
| `test_av2_api.py`, `test_av2_string_matching_service.py` | **BLOQUEADO** (dependências ausentes) |
| Suíte global | **BLOQUEADA parcialmente** (20 erros de coleta ambientais) |
| Docker | **NÃO EXECUTADO** |

## 7. Riscos / dependências externas

- Para desbloquear `test_av2_api.py` e `test_av2_string_matching_service.py`
  basta instalar as dependências do projeto (`pip install -r requirements.txt`
  ou ao menos `python-dotenv`, `fastapi`, `pydantic`). Fora do escopo de
  Mateus; recomendado ao integrador (Carlos) no ambiente de CI.
