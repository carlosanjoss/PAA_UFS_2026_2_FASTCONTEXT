# FastContext — Administração da avaliação humana

## Fonte congelada

`generations.csv` SHA-256:

`77160C5A3B2F10EA7E76C01B755B289CED4C73ABC325141791400A6EA9B092B4`

Seed de preparação e cegamento: `42`.

## Carga

- 90 itens de qualidade geral × 2 avaliadores = 180 julgamentos.
- 60 itens de groundedness × 2 avaliadores = 120 julgamentos.
- Total = 300 julgamentos.
- 5 avaliadores.
- Cada avaliador recebe 36 itens de qualidade + 24 de groundedness = 60 julgamentos.

Avaliadores:

- carlos
- yann
- wilson
- matheus
- gabriela

## Arquivos que podem ser entregues

Entregue a cada avaliador apenas:

- `evaluation_codebook.md`
- `assignments/<nome>_quality.csv`
- `assignments/<nome>_groundedness.csv`

## Arquivos administrativos — não compartilhar

Não compartilhe durante a fase cega:

- `hidden_manifest.csv`
- `assignment_summary.csv`
- os arquivos originais de `experiments/rag_results/`

## Após a coleta

Reúna as planilhas preenchidas sem alterar `blind_id`. A próxima etapa será:

1. validação dos valores;
2. junção pelo `blind_id`;
3. cálculo de concordância;
4. identificação das divergências;
5. adjudicação por terceiro avaliador;
6. abertura do `hidden_manifest.csv` somente para a análise final por condição.
