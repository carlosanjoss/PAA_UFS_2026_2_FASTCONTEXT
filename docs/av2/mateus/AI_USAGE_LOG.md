# AV2 — Registro de Uso de IA (Mateus)

**Ferramenta:** Kiro (assistente de desenvolvimento no editor).
**Modelo/configuração:** a ferramenta não expõe o modelo exato utilizado;
portanto não é afirmado aqui qual modelo específico foi empregado.
**Responsável humano pela verificação:** Mateus Almeida.

---

## 1. Objetivo do uso

Auxílio na implementação, testes e documentação da parte de Mateus na AV2
(Boyer–Moore bad-character e Rabin–Karp rolling hash), seguindo o contrato
`KIRO_MATEUS_FASTCONTEXT_AV2.md` e o contrato real do repositório definido
pela equipe.

## 2. Prompts relevantes (resumo)

1. "Leia o contrato da AV2 e execute apenas a Fase 0 (auditoria), sem
   modificar código."
2. "Pode seguir" → implementar Fase 1 (Boyer–Moore bad-character).
3. "Pode seguir" → implementar Fase 2 (Rabin–Karp rolling hash + colisão).
4. "Pode seguir" → Fases 5–7 (documentação de complexidade, suíte final,
   Git e preparação do PR, sem merge na main).

## 3. Sugestões aceitas

- Seguir o **contrato real do Carlos** (`StringMatcher` / `StringMatchResult`)
  em vez do contrato apenas sugerido no guia (que previa um campo
  `metadata` inexistente no código real).
- Adotar a **convenção real** de nomes de arquivo de teste
  (`tests/test_av2_*.py`) e docs (`docs/av2/`), em vez de `tests/av2/`.
- Registrar Boyer–Moore e Rabin–Karp em `create_default_registry()` como
  ponto de integração, atualizando o teste do registry.
- Validar colisões de Rabin–Karp por **teste controlado**
  (`base=10, modulus=11`), sem alterar o contrato público.

## 4. Sugestões rejeitadas / corrigidas

- **Não** descrever o Boyer–Moore como "completo" (não há good-suffix).
- **Não** usar `str.find`, `re` ou bibliotecas prontas no núcleo.
- **Não** expor `hash_candidates`/`collisions` no resultado, pois o
  contrato `StringMatchResult` não tem `metadata`.
- **Não** marcar como PASSOU testes que apenas não foram executados por
  dependências ausentes (`dotenv`, `fastapi`, etc.) — registrados como
  BLOQUEADO.

## 5. Erros encontrados e tratados

- Teste do Carlos `test_default_registry` quebrou ao adicionar os novos
  algoritmos → asserção de `available_names()` atualizada de forma
  consciente e documentada.
- Erros de coleta na suíte global por dependências ausentes no ambiente →
  diagnosticados e registrados como bloqueio ambiental, não como falha do
  código de Mateus.

## 6. Método de verificação humana

- Revisão da lógica de deslocamento (BM) e do rolling hash (RK).
- Execução de pytest localmente (Python 3.13.1 / pytest 9.1.1).
- **Equivalência contra a baseline Naive** (oráculo do Wilson) em casos
  canônicos e 800 casos de fuzzing com seed fixa.
- Teste explícito de colisão para comprovar ausência de falso positivo.

## 7. Testes usados para validar o código assistido por IA

- `tests/test_av2_boyer_moore.py`
- `tests/test_av2_rabin_karp.py`
- `tests/test_av2_matching_equivalence.py`
- `tests/test_av2_registry.py` (integração)
