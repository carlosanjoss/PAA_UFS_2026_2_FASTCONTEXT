# Análise dos algoritmos Naive e KMP

## 1. Objetivo

Este documento descreve os algoritmos de casamento exato de cadeias Naive e Knuth-Morris-Pratt (KMP) utilizados na AV2. A análise apresenta pseudocódigos, funcionamento da tabela LPS, argumento de corretude, complexidades, instrumentação e estratégia de testes.

Considere:

- `n`: comprimento do texto;
- `m`: comprimento do padrão;
- índices baseados em zero;
- busca sensível a maiúsculas e minúsculas;
- ocorrências sobrepostas permitidas.

## 2. Contrato utilizado

Os algoritmos implementam o contrato `StringMatcher` e expõem:

```python
search(text: str, pattern: str) -> StringMatchResult
```

O resultado informa o algoritmo, as posições iniciais das ocorrências, o total de comparações de caracteres, os tempos de pré-processamento e busca e os comprimentos do texto e do padrão.

Um padrão vazio é rejeitado com `ValueError`. Um texto vazio ou menor que o padrão produz uma tupla vazia de posições.

## 3. Algoritmo Naive

### 3.1 Ideia

O algoritmo Naive testa cada posição inicial possível do texto. Para cada alinhamento, compara os caracteres do padrão da esquerda para a direita e interrompe o alinhamento assim que encontra uma diferença. Se todos os caracteres coincidirem, registra a posição inicial.

### 3.2 Pseudocódigo

```text
NAIVE(texto, padrão):
    posições <- lista vazia

    para início de 0 até tamanho(texto) - tamanho(padrão):
        índice_padrão <- 0

        enquanto índice_padrão < tamanho(padrão):
            contabilizar uma comparação

            se texto[início + índice_padrão] != padrão[índice_padrão]:
                interromper o alinhamento

            índice_padrão <- índice_padrão + 1

        se índice_padrão == tamanho(padrão):
            adicionar início em posições

    retornar posições
```

### 3.3 Corretude

Toda ocorrência do padrão no texto precisa começar em alguma posição entre `0` e `n - m`. O algoritmo verifica todas essas posições. Em cada alinhamento, uma posição é registrada somente quando os `m` caracteres do padrão coincidem com os caracteres correspondentes do texto. Portanto, o algoritmo não omite ocorrências e não registra falsos positivos.

### 3.4 Complexidade

- Tempo no pior caso: `O(nm)`.
- Espaço auxiliar, desconsiderando a saída: `O(1)`.

O pior caso ocorre em textos e padrões repetitivos, nos quais muitos alinhamentos exigem quase `m` comparações.

## 4. Algoritmo KMP

### 4.1 Ideia

O KMP evita recomeçar a comparação do padrão do índice zero após uma diferença. O algoritmo usa informações do próprio padrão para determinar qual prefixo já conhecido também pode funcionar como sufixo da parte comparada. Essas informações são armazenadas na tabela LPS.

### 4.2 Tabela LPS

`LPS[i]` representa o comprimento do maior prefixo próprio de `pattern[0:i+1]` que também é sufixo dessa substring. Prefixo próprio significa que a substring inteira não pode ser usada como prefixo.

Exemplo para o padrão `ABAB`:

```text
índice:     0  1  2  3
caractere:  A  B  A  B
LPS:        0  0  1  2
```

No último índice, `AB` é simultaneamente prefixo e sufixo de `ABAB`, por isso o valor é 2.

### 4.3 Pseudocódigo da construção da LPS

```text
CONSTRUIR-LPS(padrão):
    lps <- vetor de zeros com tamanho m
    comprimento <- 0
    índice <- 1

    enquanto índice < m:
        se padrão[índice] == padrão[comprimento]:
            comprimento <- comprimento + 1
            lps[índice] <- comprimento
            índice <- índice + 1
        senão se comprimento > 0:
            comprimento <- lps[comprimento - 1]
        senão:
            lps[índice] <- 0
            índice <- índice + 1

    retornar lps
```

### 4.4 Pseudocódigo da busca

```text
KMP(texto, padrão):
    lps <- CONSTRUIR-LPS(padrão)
    posições <- lista vazia
    i <- 0
    j <- 0

    enquanto i < n:
        contabilizar uma comparação

        se texto[i] == padrão[j]:
            i <- i + 1
            j <- j + 1

            se j == m:
                adicionar i - m em posições
                j <- lps[j - 1]

        senão se j > 0:
            j <- lps[j - 1]
        senão:
            i <- i + 1

    retornar posições
```

### 4.5 Ocorrências sobrepostas

Depois de encontrar uma ocorrência, o índice do padrão recebe `lps[m - 1]` em vez de voltar a zero. Isso preserva a parte final da ocorrência que também pode iniciar uma nova ocorrência.

```text
texto:   abababab
padrão:  abab
saída:   (0, 2, 4)
```

### 4.6 Corretude

Durante a busca, `j` representa a quantidade de caracteres iniciais do padrão que coincidem com o sufixo do trecho do texto já processado. Quando ocorre uma diferença e `j > 0`, a LPS seleciona o maior prefixo do padrão que também é sufixo da parte já reconhecida. Assim, o KMP elimina apenas alinhamentos que não podem produzir uma ocorrência e mantém todo alinhamento ainda possível.

Quando `j == m`, todos os caracteres do padrão coincidiram, logo `i - m` é uma posição válida. A atualização para `lps[m - 1]` mantém a invariante e permite encontrar sobreposições. Portanto, toda posição registrada é correta e nenhuma ocorrência possível é descartada.

### 4.7 Complexidade

- Construção da LPS: `O(m)` em tempo e `O(m)` em espaço.
- Busca: `O(n)` em tempo.
- Tempo total: `O(n + m)`.
- Espaço auxiliar total: `O(m)`.

Na busca, o índice do texto nunca retrocede. Embora o índice do padrão possa diminuir por meio da LPS, o progresso combinado dos índices limita o número de operações a uma quantidade linear.

## 5. Instrumentação

### 5.1 Contagem de comparações

O campo `comparisons` é incrementado exatamente uma vez sempre que um caractere do texto é comparado com um caractere do padrão durante a fase de busca.

As comparações realizadas durante a construção da LPS não entram em `comparisons`. O custo dessa etapa é representado separadamente por `preprocessing_time_ns`.

### 5.2 Tempos

- `preprocessing_time_ns`: tempo para construir a tabela LPS;
- `search_time_ns`: tempo gasto na busca pelo padrão no texto;
- `total_time_ns`: soma dos tempos de pré-processamento e busca, calculada pelo modelo de resultado.

Os tempos são medidos em nanossegundos com `perf_counter_ns`. Pequenas variações entre execuções são esperadas devido ao sistema operacional, ao interpretador e a outros processos em execução.

## 6. Validação e testes

A implementação foi validada com:

- ocorrência única e ocorrências múltiplas;
- padrão ausente;
- texto vazio;
- rejeição de padrão vazio;
- padrão maior que o texto;
- sensibilidade a maiúsculas e minúsculas;
- caracteres Unicode;
- entradas repetitivas;
- ocorrências sobrepostas;
- verificação direta de tabelas LPS;
- comparação entre as posições retornadas por KMP e Naive;
- 250 casos pseudoaleatórios reproduzíveis com semente fixa.

Exemplo mínimo:

```python
from src.string_matching.kmp import KMPMatcher

matcher = KMPMatcher()
result = matcher.search("abababab", "abab")

assert result.positions == (0, 2, 4)
```

Teste específico:

```bat
python -m pytest tests\test_av2_kmp.py -q
```

Resultado observado:

```text
18 passed
```

Suíte completa:

```bat
python -m pytest -q
```

Resultado observado durante a validação:

```text
545 passed in 32.12s
```

## 7. Comparação resumida

O Naive tem implementação simples, não requer pré-processamento e usa espaço auxiliar constante, mas pode executar `O(nm)` comparações no pior caso. O KMP requer a tabela LPS e espaço `O(m)`, porém garante tempo total `O(n + m)` e tende a ser especialmente vantajoso em entradas longas ou repetitivas.

Os dois algoritmos seguem o mesmo contrato e devem retornar exatamente as mesmas posições para a mesma entrada. A implementação usa a Naive como oráculo de referência nos testes de corretude do KMP.

## 8. Limitações e observações

- A busca é exata e sensível a maiúsculas e minúsculas.
- Não há normalização Unicode, remoção de acentos ou busca aproximada.
- O padrão vazio é inválido por contrato.
- A medição de tempo deve ser analisada em várias repetições durante os experimentos; uma execução isolada não é suficiente para comparar desempenho.
- A comparação experimental definitiva depende da mesma máquina, do mesmo corpus e dos mesmos cenários para todos os algoritmos.
