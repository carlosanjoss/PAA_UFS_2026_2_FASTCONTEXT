AVALIAÇÕES ANÔNIMAS — LEIA-ME

Este ZIP contém 10 arquivos CSV, organizados em 5 pares anônimos (conjunto_01 a conjunto_05):
- qualidade: 36 registros, com correctness, completeness, clarity, hallucination_present e notes preenchidos.
- fundamentacao: 36 registros, com groundedness e notes preenchidos.
Total: 360 linhas de avaliação; 180 de qualidade e 180 de fundamentação.

CRITÉRIOS ADOTADOS
correctness: 1 (incorreto) a 5 (tecnicamente correto).
completeness: 1 (não responde) a 5 (atende de forma completa).
clarity: 1 (muito confuso) a 5 (claro e organizado).
hallucination_present: 0 (não há afirmação inventada evidente) e 1 (há afirmação técnica não fundamentada ou API inventada).
groundedness: 1 (sem suporte no contexto) a 5 (alinhamento total com os excertos recuperados). Abstenção fiel quando não há contexto pode receber 5, ainda que a resposta tenha baixa completude.
notes: breve justificativa específica da avaliação.

As avaliações são julgamento técnico baseado nas respostas e nos trechos fornecidos; não representam validação por execução do código.
Os arquivos originais possuíam aspas e quebras de linha irregulares; os 36 registros de cada arquivo foram reconstruídos e os CSVs gerados foram validados com leitor CSV estrito.
Perguntas, respostas, trechos recuperados e identificadores cegos originais foram mantidos para rastreabilidade dos exemplos.
Os nomes originais, assinaturas e informações de autoria dos avaliadores não aparecem no ZIP; os números dos conjuntos são pseudônimos de arquivo, sem identificação de pessoa.
