# AV2 Amendment — generation retry budget

Date: 2026-10-27
Scope: `config/rag_experiments.yaml`, `src/rag/pipeline.py`,
`experiments/av2/run_rag_comparison.py`

## Defect

The frozen generation block set `max_retry_tokens` equal to `max_tokens`
(both `256`). `RAGPipeline._generate_with_retry` escalates a truncated
generation with `next_max_tokens = min(max_tokens * 2, max_retry_tokens)`;
with both values at `256` the result can never exceed the current budget, so
the pipeline raised `LLMTruncatedResponseError` ("Generation reached the
maximum retry token limit") on the *first* truncation, without performing any
retry. The AV2 RAG comparison aborted at `q02/literal`.

## Evidence

Direct provider probe (`qwen2.5:3b`, `temperature=0`, `seed=42`, frozen
`q02/literal` context, 1,341 prompt tokens):

| `max_tokens` | `done_reason` | `eval_count` | answer |
|---|---|---|---|
| 256 | `length` | 256 | cut mid-sentence |
| 512 | `stop` | 229 | complete |
| 1024 | `stop` | 229 | complete |

A single escalation to 512 tokens is sufficient for this case.

## Change

- `config/rag_experiments.yaml`: `max_truncation_retries` `1 → 2`,
  `max_retry_tokens` `256 → 1024`. `max_tokens` stays `256`, preserving the
  base budget of the frozen protocol; the retry path now escalates
  `256 → 512 → 1024`.
- `src/rag/pipeline.py`: both truncation errors now report `max_tokens`,
  `max_retry_tokens`, and the retry count; the no-headroom case states the
  misconfiguration explicitly instead of a generic message.
- `experiments/av2/run_rag_comparison.py`: startup validation rejecting
  `max_retry_tokens <= max_tokens` whenever truncation retries are enabled,
  so this configuration cannot silently abort a long run again.

Verified against the live provider: base budget 64 tokens → two escalations →
`effective_max_tokens=256`, `truncation_retries=2`, complete answer returned.

## Impact on AV1

AV1 (`experiments/rag_results/generations.csv`) is complete: 90/90 rows, all
`done_reason = "stop"`, maximum `eval_count` 130 with
`requested_max_tokens = 256`. No AV1 generation ever truncated, so the retry
parameters cannot change any AV1 output; the AV1 dataset does not need
regeneration. The config file is hashed into AV1's `experiment_signature`
(`67797253ce57…`), which therefore changes; the stored rows keep the previous
signature, so a future `--resume` of AV1 will refuse and require a fresh run.

## Impact on AV2

The AV2 RAG dataset was regenerated from scratch under the amended config. The
three rows produced before the fix are preserved at
`experiments/av2/rag_comparison/final.old-256tokens/` and are superseded.

- Manifest signature: `5c1834b1cf247c66672f585e6d5c5e3bdaf35813c9a1a13bb2d1159a7eb14700`
- 90/90 generations, one signature, all `done_reason = "stop"`, no empty answers
- 13 of 90 rows required exactly one escalation (`effective_max_tokens = 512`,
  `truncation_retries = 1`) — each of them would have aborted the run before
  this fix
- 77 of 90 rows completed within the base budget (`effective_max_tokens = 256`)

## Reproduction

```bash
python -m pytest tests -q                          # 684 passed
python experiments/av2/run_rag_comparison.py       # 90/90 generations
```
