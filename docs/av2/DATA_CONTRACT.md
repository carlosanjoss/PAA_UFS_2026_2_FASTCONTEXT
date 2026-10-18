
# FastContext AV2 — Experimental Data Contract

## 1. Objective

Prepare reproducible datasets and experimental scenarios
for exact string matching algorithms.

Algorithms:
- Naive String Matching
- Knuth-Morris-Pratt (KMP)
- Boyer-Moore
- Rabin-Karp

## 2. Original Corpus

Source: FastAPI official documentation
Version: 0.141.0
Language: English

Expected dataset:
- 153 documents
- 1305 chunks

Source file:
data/chunks/chunks.jsonl

Existing loader:
src.ingestion.chunk_loader.load_chunks_jsonl

The AV1 corpus must remain unchanged.

## 3. Matching Rules

All algorithms must use:

- Exact substring matching
- Case-sensitive comparison
- Zero-based indexing
- Overlapping occurrences
- Original text content
- No additional text normalization

Empty patterns must raise ValueError.

A pattern longer than the text must return no matches.

## 4. Pattern Dataset

Output file:
data/av2/patterns.json

Editable candidate definitions are read from data/av2/pattern_candidates.json.
The preparation script validates these candidates and generates the output
catalog; it does not modify the candidate definitions.

Required fields:

- pattern_id: unique identifier
- pattern: exact string to search
- category: experimental category
- description: purpose of the pattern

The catalog contains exactly eight verified patterns, matching the eight cases
in the experimental matrix. The long pattern is an explicit exact phrase from
the corpus; the generator does not append an extra pattern automatically.

Required categories:

- short
- medium
- long
- absent
- repetitive
- overlapping

Every pattern must have a unique identifier.

## 5. Experimental Scenarios

Output file:
data/av2/scenarios.json

Required fields:

- scenario_id
- pattern
- pattern_type
- corpus_fraction
- expected_case
- pattern_id
- pattern_category
- scale
- chunk_ids
- total_characters
- description

Every pattern has one scenario per scale: eight patterns × three scales = 24
scenarios. Corpus patterns use the deterministic chunk prefixes. Synthetic
patterns are isolated microbenchmark inputs: the seed text declared in
`pattern_candidates.json` is repeated and truncated to the same character
length as the corresponding corpus scale. Their `chunk_ids` remain empty,
`scale` is `synthetic_microbenchmark`, and `corpus_fraction` identifies the
reference scale, not a sample of corpus chunks. The scenario records both the
synthetic text and the reference character count. For the overlap seed,
`ABABABA` searched for `ABA` has seed positions `[0, 2, 4]`; a scaled synthetic
text may also contain matches across repeated-seed boundaries.

Corpus fractions:

- 0.25
- 0.50
- 1.00

These AV2 scales are configured separately from the general experiment corpus
sizes in `config/experiments.yaml`.

Corpus subsets must be deterministic and nested:
each smaller subset must be contained in the next one.

Use a stable ordering of chunk IDs.

## 6. Preparation Script

File:
scripts/av2/prepare_inputs.py

Responsibilities:

1. Load the existing corpus.
2. Validate chunk IDs.
3. Preserve the original text.
4. Create deterministic corpus subsets.
5. Prepare the search patterns.
6. Generate experimental scenarios.
7. Validate the output files.

Do not modify the original corpus.

Do not implement matching algorithms in this script.

## 7. Validation

The preparation process must verify:

- Unique pattern IDs
- Unique scenario IDs
- Existing chunk IDs
- Deterministic subset selection
- Correct corpus sizes
- Valid pattern references
- Reproducible outputs
- Exactly eight patterns and 24 scenarios (eight patterns × three scales)
- One deterministic shared input stream from the prepared corpus, pattern, and
  scenario files; algorithm correctness and benchmark repetitions are validated
  during Carlos's integration and experiments

## 8. Delivery

Yann must provide:
- data/av2/patterns.json
- data/av2/scenarios.json
- scripts/av2/prepare_inputs.py
- docs/av2/DATA_PROTOCOL.md

Automated tests for the prepared inputs are in `tests/test_av2_prepare_inputs.py`.
Cross-algorithm correctness and the five benchmark repetitions belong to the
integration and experiment work, not this data-preparation component.

Submit changes using a dedicated branch
and a Pull Request targeting av2-string-matching.
