
# AV2 — Data Contract

## 1. Objective

Prepare reproducible text-matching experiments for the
FastContext AV2 project.

## 2. Corpus

Source: FastAPI official documentation, version 0.141.0.

Expected corpus:
- 153 documents
- 1305 chunks
- English language

Source file:
data/chunks/chunks.jsonl

Use the existing loader:
src.ingestion.chunk_loader.load_chunks_jsonl

The original corpus must not be modified.

## 3. Matching rules

All algorithms must follow the same rules:

- Exact substring matching
- Case-sensitive comparison
- Zero-based positions
- Overlapping matches allowed
- Empty patterns are invalid
- No additional lowercasing or text normalization
- Preserve original chunk IDs and source metadata

## 4. Pattern dataset

Create:
data/av2/patterns.json

Each pattern must contain:

- pattern_id
- pattern
- category
- description

Categories:
- short
- medium
- long
- absent
- repetitive
- overlapping

## 5. Experimental scenarios

Create:
data/av2/scenarios.json

Each scenario must identify:

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

Chunk selection must be deterministic.

Record the actual number of selected chunks.

## 6. Preparation script

Create:
scripts/av2/prepare_inputs.py

Responsibilities:

1. Load the existing corpus.
2. Validate chunk IDs.
3. Select deterministic corpus subsets.
4. Prepare and validate search patterns.
5. Generate scenarios.json.
6. Preserve the original corpus.

Do not implement string-matching algorithms
inside this script.

## 7. Acceptance criteria

- Original corpus remains unchanged.
- Every pattern has a unique ID.
- Every scenario has a unique ID.
- Every referenced chunk exists.
- All four algorithms can use the same inputs.
- Repeated preparation produces identical outputs.
- Dataset preparation does not depend on an LLM.

## 8. Delivery

Submit the files through a dedicated Git branch
and open a Pull Request targeting the AV2 branch.

Include instructions to reproduce the dataset.
