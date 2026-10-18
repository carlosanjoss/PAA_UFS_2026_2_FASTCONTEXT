# AV2 Data Protocol

## Corpus provenance and preservation

The experiment uses the frozen AV1 chunks at
`data/chunks/chunks.jsonl`. The source is the English FastAPI documentation at
`https://github.com/fastapi/fastapi/tree/0.141.0/docs/en/docs`, version
`0.141.0`. The expected corpus contains 153 source documents and 1,305 chunks.
Its known AV1 fingerprint is
`e08b47b55d7237f38bcaba48c79a468a40b92e4f3120e968ae9ece999f5e8c2d`.

`scripts/av2/prepare_inputs.py` loads the existing JSONL and verifies the
fingerprint, version, document count, and chunk count against
`config/corpus.yaml`. It does not download, regenerate, rewrite, or normalize
the AV1 chunks. A different corpus is rejected unless mismatch approval is
explicitly requested for a test or a documented corpus change.

## Matching and text policy

Patterns are exact, case-sensitive substrings. Text and patterns are passed
without additional normalization. Matching positions are zero-based, and
overlapping occurrences count as separate matches. Each algorithm receives
the same original chunk text and exact pattern for each corpus scenario.

## Pattern selection

The catalog has eight cases, selected to cover the PDF matrix and its requested
input types:

| Case | Pattern | Purpose |
|---|---|---|
| Short/frequent | `a` | A short pattern with many corpus occurrences |
| Real term | `OAuth2` | Authentication term present in the documentation |
| Real term | `dependency` | Lowercase technical term present in the documentation |
| Real term | `WebSocket` | Feature term present in the documentation |
| Long real phrase | `background tasks` | Longest multiword real pattern among the examples named in the PDF |
| Absent | `FASTCONTEXT_AV2_ABSENT_SENTINEL_7F3C91` | Sentinel whose absence is verified against the full corpus |
| Repetitive synthetic | `AAAA` | Repeated-character stress input |
| Overlap synthetic | `ABA` | Artificial overlap case, with matches at 0, 2, and 4 in `ABABABA` |

The preparation process verifies real-pattern occurrence counts before
publishing the catalog. The absent sentinel must occur zero times. Synthetic
expected positions are explicit test data in `pattern_candidates.json` and
are copied into the catalog for the algorithm tests; the preparation script
does not execute a matcher to verify them. Wilson's and the team's algorithm
tests verify the actual implementations against those expected positions.
Pattern counts are case-sensitive and include overlaps.

## Scale and deterministic sampling

The corpus scales are 25%, 50%, and 100%, configured in
`config/experiments.yaml`. Chunks are ordered by ascending `chunk_id`; each
corpus scale is the first `ceil(chunk_count × fraction)` chunks in that order.
The resulting subsets are nested and repeatable. Each scenario records the
selected chunk IDs and the total number of original content characters.

All eight patterns have the three scale labels: 25%, 50%, and 100%, yielding
24 scenarios. For corpus patterns, each scale selects the deterministic chunk
prefix described above. For the two synthetic patterns, each scale is an
isolated microbenchmark input whose character length equals the total content
characters in the corresponding corpus prefix. The seed in
`pattern_candidates.json` is repeated and truncated to that length. Synthetic
scenarios have no corpus chunk IDs and use `scale: "synthetic_microbenchmark"`
to distinguish generated input from real documentation; their
`corpus_fraction` names only the reference size. The generated text itself,
its seed, its size, and the reference size are recorded in each scenario.

The synthetic inputs remain artificial; repeating a seed is a declared
microbenchmark construction, not a claim that the text comes from FastAPI.
Occurrences are counted on the exact generated text, including matches that
may cross a repeated-seed boundary. The base overlap example `ABA` in
`ABABABA` has positions `[0, 2, 4]`.

The recommended matrix is `4 algorithms × 3 scales × 8 patterns/cases × 5
repetitions = 480` scenario/repetition/algorithm combinations. For corpus
scenarios, each combination searches selected chunks independently; the
scenario-level run can aggregate chunk results, but must not concatenate chunk
texts and create cross-chunk matches. Synthetic scenarios search their one
generated text per combination.

## Correctness checks and boundaries

Run the preparation with:

```bash
python -m scripts.av2.prepare_inputs
```

Run the automated input tests with:

```bash
python -m pytest -q tests/test_av2_prepare_inputs.py
```

These tests verify that the corpus, patterns, and scenarios load, that the
shared iterator reconstructs deterministic inputs from those files, and that
the available Naive implementation checks the expected positions for the
prepared scenarios. A dispatch-contract test feeds the iterator's ordered
`(text, pattern)` sequence to four mocked consumer slots (Naive, KMP,
Boyer–Moore, and Rabin–Karp) and asserts that every slot receives exactly the
same calls. This verifies the prepared-input interface and the fan-out
contract; it does not exercise production dispatch or execute KMP,
Boyer–Moore, or Rabin–Karp. Carlos's integration and experiment work must
connect the actual implementations, check cross-algorithm position agreement,
and run the five benchmark repetitions.

This protocol establishes literal matching correctness only. It does not
measure semantic relevance, and it does not authorize changing the AV1 corpus,
its metadata, or the existing AV1 tables.
