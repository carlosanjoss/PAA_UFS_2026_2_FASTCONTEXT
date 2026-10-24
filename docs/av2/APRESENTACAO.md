# Roteiro de apresentação — FastContext AV1 + AV2

Este roteiro foi preparado para uma demonstração de aproximadamente 12 a 15 minutos. Os valores temporais abaixo pertencem ao benchmark oficial persistido; os tempos exibidos na busca ao vivo variam entre execuções e devem ser apresentados como medições interativas.

## 1. Preparação antes de apresentar

Execute na raiz do repositório:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
cd frontend
npm ci
npm run build
cd ..
.venv\Scripts\python.exe -m uvicorn src.web_api:app --host 127.0.0.1 --port 8000
```

Abra `http://127.0.0.1:8000/` e confirme:

- as cinco abas aparecem no menu;
- `http://127.0.0.1:8000/docs` mostra as rotas da AV1 e da AV2;
- a aba **Busca textual** encontra `OAuth2`;
- a aba **Experimentos AV2** carrega os gráficos;
- o navegador está com zoom em 100%;
- nenhum serviço de outro projeto está ocupando a porta 8000.

Se o LLM local não estiver disponível, a busca exata e os experimentos da AV2 continuam funcionando. A interface mostrará a indisponibilidade sem ocultar as métricas algorítmicas.

## 2. Abertura — 1 minuto

### O que dizer

> O FastContext começou na AV1 como um laboratório de recuperação de informações sobre a documentação do FastAPI. Na AV2, o mesmo corpus foi preservado e adicionamos processamento de cadeias com quatro algoritmos de casamento exato. A aplicação permite separar duas coisas: uma busca interativa para observar a execução e um dashboard que lê o benchmark oficial congelado.

### O que mostrar

1. Menu lateral com os grupos **AV1 · Recuperação** e **AV2 · Cadeias**.
2. Aba **Visão geral**.
3. Indicador do corpus com 1.305 chunks.

## 3. Preservação da AV1 — 1 minuto

### O que dizer

> A integração foi aditiva. As rotas `/api/health`, `/api/metrics` e `/api/query`, os quatro métodos de recuperação e as três páginas da AV1 foram preservados. Os algoritmos da AV2 possuem tipos e contratos separados porque casamento exato não é a mesma operação que recuperação ou RAG.

### O que mostrar

1. Passe rapidamente pelas abas **Perguntar** e **Comparações**.
2. Abra `/docs` em outra aba e aponte os grupos de endpoints.
3. Não gaste tempo executando RAG se o provedor local estiver indisponível.

## 4. Busca textual da AV2 — 4 minutos

Abra **Busca textual**.

### Demonstração principal

1. Selecione `OAuth2`.
2. Selecione **Boyer-Moore**.
3. Escolha 5 chunks exibidos.
4. Mantenha **Resposta com LLM** ativada.
5. Use a pergunta `Como OAuth2 é utilizado no FastAPI?`.
6. Clique em **Buscar no corpus**.

### Resultado esperado

- 1.305 chunks analisados;
- 42 chunks com correspondências;
- 103 ocorrências no total;
- 139.057 comparações diretas para Boyer-Moore;
- somente 5 cards retornados, sem alterar os totais;
- primeiro chunk: `advanced/security/oauth2-scopes-0`;
- posições iniciais no primeiro chunk: `2, 48, 212, 291, 607, 718, 896, 986, 1222`.

Os tempos desta tela variam, pois são produzidos durante a interação.

Quando o provedor estiver disponível, mostre também:

- a resposta fundamentada nos chunks que contêm `OAuth2`;
- o provedor e o modelo utilizados;
- o tempo de geração separado do tempo do algoritmo;
- as citações e o selo de validação;
- os IDs dos chunks usados como contexto.

### O que explicar

> O backend percorre o corpus completo e calcula todas as ocorrências. O limite atua somente na quantidade de cards enviados para a interface. A busca preserva caixa, encontra sobreposições e retorna offsets do texto original. O destaque é construído com componentes React, sem injetar HTML, e converte o texto em pontos de código para manter compatibilidade com os offsets Unicode do Python.

> A LLM entra somente depois da busca exata. Os chunks que possuem correspondência tornam-se o contexto do RAG. Portanto, a LLM não encontra posições, não modifica comparações e não participa do benchmark. Se ela falhar, os resultados de Boyer-Moore, KMP, Naive ou Rabin-Karp continuam válidos e visíveis.

### Comparação interativa

1. Ative **Comparar os quatro**.
2. Execute novamente `OAuth2`.
3. Mostre o selo **Posições idênticas**.

Todos devem informar 103 ocorrências e as mesmas posições. Explique:

> A igualdade das posições verifica a corretude relativa. Os tempos desta tabela são exploratórios e não devem ser confundidos com o benchmark oficial, que tem ambiente, cenários e repetições controlados.

### Outras perguntas/buscas que podem ser demonstradas

- `WebSocket`: termo real e sensível a caixa, com 60 ocorrências no corpus integral.
- `dependency`: termo real em minúsculas, com 220 ocorrências.
- `websocket`: útil para mostrar que a busca diferencia maiúsculas e minúsculas.
- `FASTCONTEXT_AV2_ABSENT_SENTINEL_7F3C91`: padrão ausente, deve retornar zero.
- Um padrão vazio: a API deve rejeitar a requisição.

Para o padrão ausente, destaque que a LLM não é chamada: sem correspondência exata, não existe contexto autorizado para a resposta.

## 5. Metodologia experimental — 1 minuto

Abra **Experimentos AV2** e mostre os indicadores superiores.

### O que dizer

> O benchmark possui quatro algoritmos, oito padrões e três escalas, formando 24 cenários. Cada combinação de cenário e algoritmo foi repetida cinco vezes: 24 vezes 4 vezes 5 resulta em 480 medições temporais. As 96 combinações temporais e as 96 medições de memória foram resumidas separadamente. A página apenas lê os artefatos persistidos; abrir os gráficos não executa novamente o benchmark.

Mostre:

- 4 algoritmos;
- 8 padrões;
- 24 cenários;
- 480 medições temporais;
- 96 combinações temporais;
- 96 combinações de memória;
- 1.305 chunks.

## 6. Leitura dos gráficos — 4 minutos

Mantenha `OAuth2`, escala de 100% e os quatro algoritmos ativos.

### Tempo médio

Leia os valores carregados do CSV:

| Posição | Algoritmo | Tempo médio |
|---:|---|---:|
| 1 | Boyer-Moore | 27,78014 ms |
| 2 | KMP | 64,40984 ms |
| 3 | Naive | 148,276 ms |
| 4 | Rabin-Karp | 322,02624 ms |

### O que dizer

> Para OAuth2 no corpus integral, Boyer-Moore foi aproximadamente 5,337 vezes mais rápido que o Naive. O resultado é compatível com a possibilidade de saltar posições usando bad-character, mas a conclusão vale para estas entradas e esta implementação.

### Escalabilidade

> O eixo horizontal usa o número real de caracteres. Cada curva mostra como o tempo cresceu nas escalas de 25%, 50% e 100%. As barras representam mais ou menos um desvio-padrão entre cinco repetições; não são intervalos de confiança e não demonstram significância estatística.

### Speedup

> Speedup é o tempo médio do Naive dividido pelo tempo do algoritmo. Acima de 1 significa mais rápido; exatamente 1 é a referência; abaixo de 1 significa mais lento.

### Comparações de caracteres

> O contador é diretamente comparável entre Naive, KMP e Boyer-Moore. Em Rabin-Karp, ele conta verificações de caracteres após colisões ou coincidências de hash, mas não conta as operações do rolling hash. Portanto, poucas comparações não significam necessariamente menor custo total.

### Memória e componentes temporais

> O gráfico de memória usa MiB e mede o pico rastreado pelo `tracemalloc`, incluindo objetos de saída. Não representa RAM total nem memória auxiliar isolada. Pré-processamento e busca são componentes internos; sua soma não substitui o tempo externo, que também inclui iteração e agregação dos resultados.

## 7. Cenários sintéticos — 1 minuto

Altere **Tipo de entrada** para **Microbenchmarks sintéticos**.

Mostre `AAAA` e depois `ABA`.

### O que dizer

> Os casos sintéticos isolam repetição e sobreposição. KMP venceu as três escalas de `AAAA` e as três escalas de `ABA`. Isso é coerente com o reaproveitamento da tabela de prefixos. Esses textos artificiais ajudam a explicar comportamento algorítmico, mas não representam a distribuição da documentação real.

No conjunto completo dos 24 cenários:

- Boyer-Moore obteve o menor tempo médio em 15 cenários;
- KMP obteve o menor tempo médio em 9 cenários;
- KMP venceu todos os seis cenários sintéticos repetitivos/sobrepostos.

## 8. Conclusão — 1 minuto

### O que dizer

> Não existe um vencedor universal. Boyer-Moore foi mais forte em vários padrões reais, principalmente quando seus saltos evitaram comparações. KMP foi mais forte nas entradas com repetição e sobreposição. Rabin-Karp mostrou que um contador isolado não representa todo o custo, pois o hash teve sobrecarga relevante em Python. O Naive permaneceu essencial como referência de corretude e para o cálculo de speedup.

> As conclusões são descritivas e específicas deste corpus, desta máquina e destas implementações. Busca exata responde onde uma cadeia aparece; recuperação semântica e RAG respondem outra classe de problema.

## 9. Perguntas prováveis da banca

### Por que não comparar diretamente AV1 e AV2?

Porque possuem objetivos e contratos diferentes. A AV1 ranqueia chunks relevantes para uma consulta; a AV2 procura ocorrências literais e devolve posições exatas.

### Por que Boyer-Moore usa apenas bad-character?

Essa é a variante definida para a atividade. Ela mantém a implementação auditável e permite observar ganhos por saltos sem introduzir a heurística good-suffix.

### Por que Rabin-Karp foi lento apesar de poucas comparações?

O contador registra comparações diretas de caracteres, não todas as operações de hash. Nesta implementação Python, atualizar o rolling hash em cada janela introduziu custo que aparece no tempo, mas não nesse contador.

### Por que apenas cinco repetições?

Foi a metodologia definida para produzir estatísticas descritivas. Ela permite média e desvio-padrão, mas não sustenta afirmações de significância estatística.

### O limite de resultados pode esconder ocorrências?

Não nas métricas. O corpus inteiro é processado antes do corte; o limite reduz somente os chunks enviados para visualização.

### Os gráficos executam o benchmark novamente?

Não. O endpoint lê e valida arquivos CSV e JSON congelados. Se um artefato obrigatório estiver ausente, retorna erro 503 informativo.

### A LLM faz parte do algoritmo da AV2?

Não. Ela é uma etapa posterior e opcional. O algoritmo encontra os chunks e as posições exatas; o RAG usa uma quantidade limitada desses chunks para responder à pergunta com citações. As métricas de string matching são calculadas antes da geração.

### Como a corretude foi verificada?

Os quatro algoritmos seguem o mesmo contrato e devem produzir as mesmas posições. Há testes automatizados para ocorrências reais, padrões vazios, algoritmos desconhecidos, sobreposição, Unicode, artefatos experimentais e preservação das rotas da AV1.

## 10. Plano de contingência

- Se o LLM estiver indisponível, mostre o aviso da própria AV2 e explique que a busca exata permanece operacional.
- Se a porta 8000 estiver ocupada, encerre apenas o processo identificado ou execute o build pela porta 8010 e abra `http://127.0.0.1:8010/`.
- Se a rede estiver indisponível, a demonstração local continua funcionando porque os dados e o frontend compilado estão no repositório.
- Não execute `run_benchmarks.py` durante a apresentação.
- Mantenha uma captura ou PDF dos gráficos como contingência visual, sem apresentá-la como execução ao vivo.
