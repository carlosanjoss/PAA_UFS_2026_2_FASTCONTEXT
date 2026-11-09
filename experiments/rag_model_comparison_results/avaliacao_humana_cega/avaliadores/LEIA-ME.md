# Instrumentos cegos — comparacao Qwen × Llama

As duas planilhas foram produzidas das 179 respostas completas e preservam integralmente os textos das respostas. O modelo e a condicao nao aparecem nas planilhas dos avaliadores.

## Rubrica ja documentada na avaliacao AV2

- correctness: 1 (incorreto) a 5 (tecnicamente correto).
- completeness: 1 (nao responde) a 5 (atende de forma completa).
- clarity: 1 (muito confuso) a 5 (claro e organizado).
- hallucination_present: 0 (nao ha afirmacao inventada evidente) ou 1 (afirmacao tecnica nao fundamentada ou API inventada).
- groundedness: 1 (sem suporte no contexto) a 5 (totalmente alinhada aos excertos recuperados).
- notes: justificativa curta e especifica.

A escala acima reproduz os campos da avaliacao AV2 existente. Nenhuma nota foi atribuida automaticamente.

## Escopo

- Qualidade: 179 itens (inclui No-RAG).
- Fundamentacao: 119 itens RAG completos; No-RAG fica fora desta dimensao por nao possuir contexto recuperado.
- Uma tentativa truncada nao integra as planilhas de respostas completas; consta na auditoria experimental.
- Os textos dos chunks seguem os IDs congelados e a ordem original.
- As colunas de notas e justificativas estao intencionalmente vazias.

## Regras de integridade

Mantenha `administrativo/chave_cego_admin.csv` fora do alcance dos avaliadores. Nao compartilhe o manifesto administrativo com eles. As planilhas sao masters cegos: este programa nao define o numero de avaliadores, nao aloca itens, nao calcula concordancia e nao simula julgamentos. Defina o plano de distribuicao antes da coleta.
