
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

Required fields:

- pattern_id: unique identifier
- pattern: exact string to search
- category: experimental category
- description: purpose of the pattern

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
- corpus_fraction
- pattern_ids
- chunk_ids
- total_characters
- description

Corpus fractions:

- 0.25
- 0.50
- 0.75
- 1.00

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

## 8. Delivery

Yann must provide:

- data/av2/patterns.json
- data/av2/scenarios.json
- scripts/av2/prepare_inputs.py
- docs/av2/DATA_PROTOCOL.md

Submit changes using a dedicated branch
and a Pull Request targeting av2-string-matching.
