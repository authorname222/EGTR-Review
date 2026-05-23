# Data Format

This page documents the JSON / JSONL schemas used throughout EGTR-Review. All
field names follow the agent prompts in [`docs/prompts/`](prompts/).

## 1. Input paper (`data/samples/train_papers/paper_*.json`)

```json
{
  "paper_id": "iclr2019_0001",
  "title": "string",
  "abstract": "string",
  "sections": [
    { "heading": "string", "paragraphs": ["string", "string", "..."] }
  ],
  "citations": [
    { "ref_id": "string", "raw": "string" }
  ]
}
```

## 2. Teacher output per paper (`data/samples/distilled/sample_*.json`)

The teacher produces three artefacts per paper that together form one
distillation training instance:

- `L_prime` — list of evidence-enhanced paper units (schema in prompts/ §3)
- `T`      — list of reasoning-trajectory units    (schema in prompts/ §4)
- `Y`      — structured final review object        (schema in prompts/ §5)
- `alpha`  — sample weight aggregated from per-unit `Flag_i_n`

### 2.1 `L_prime[*]` (per prompts/ §3)

```json
{
  "unit_id":   "1",
  "P_i_n":     "Section 3 Method / Paragraph 2",
  "C_i_n":     "...verbatim fragment text...",
  "Q_i_n":     "retrieval query text",
  "E_i_n": [
    {
      "source_title":     "...",
      "source_metadata":  { "source": "arxiv|serpapi|semantic_scholar", "url": "..." },
      "evidence_snippet": "...",
      "evidence_relevance": "support | refute | weak | contextual"
    }
  ],
  "Flag_i_n":  "Strong Evidence-Supports | Strong Evidence-Refutes | Weak Evidence-Metadata Only | No Evidence | Non-verifiable Item",
  "labeling_rationale": "...",
  "feedback_to_key_element_extractor_agent": "None | Need query rewriting | Need query expansion",
  "feedback_rationale": "..."
}
```

The five `Flag_i_n` values are the canonical labels. The separator between
"Evidence" and the suffix is the ASCII hyphen-minus `-` (U+002D), matching
the paper PDF, the README, and the distillation-data example in the
appendix.

### 2.2 `T[*]` (per prompts/ §4)

```json
{
  "unit_id":             "1",
  "P_i_n":               "Section 3 Method / Paragraph 2",
  "C_i_n":               "...",
  "E_i_n":               [/* same item schema as §2.1 */],
  "Flag_i_n":            "...",
  "verification_path":   "Support Verification | Refutation Verification | Weak Evidence Probing | Internal Consistency Checking | Non-verifiable Quality Assessment",
  "verification_question": "...",
  "reasoning_process":   "...",
  "preliminary_review_point": "...",
  "review_point_type":   "Strength | Weakness | Question | Limitation | Clarification Request",
  "traceability_basis":  "P_i_n | C_i_n | E_i_n | Flag_i_n | Reasoning Process"
}
```

### 2.3 `Y` (per prompts/ §5)

```json
{
  "summary":     "string",
  "strengths":   [{"comment": "...", "impact": "High|Medium|Low", "evidence_grounding": "...", "trace_ids": ["T_i-...", "..."]}],
  "weaknesses":  [{"comment": "...", "impact": "...", "evidence_grounding": "...", "trace_ids": ["..."]}],
  "questions":   [{"question": "...", "type": "Clarification | Missing Information | Ambiguity | Limitation | Reproducibility", "trace_ids": ["..."]}],
  "suggestions": [{"suggestion": "...", "related_to": "Weakness ID | Question ID", "trace_ids": ["..."]}],
  "traceability_notes": [
    {
      "review_point_id":      "S1 / W1 / Q1 / Sug1 / ...",
      "source_position":      "Section / Figure / Table / Equation / Appendix / Paragraph ...",
      "paper_fragment":       "short excerpt",
      "evidence_set":         "...",
      "evidence_state_label": "Strong Evidence-Supports | ...",
      "reasoning_unit_id":    "T_i-...",
      "rationale":            "..."
    }
  ]
}
```

## 3. Distillation dataset (`data/full/D_distill.jsonl`)

One JSON object per line, identical to the per-paper teacher output above
plus the aggregated sample weight `alpha`. Built by
`python -m src.distillation.build_distill_data`.

## 4. Inference output (`outputs/<paper>_review.json`)

```json
{
  "paper_id": "iclr2019_demo",
  "review":  {/* a Y object as above, or {"raw_text": "..."} if JSON parsing failed */}
}
```

## 5. Evaluation reference (`data/samples/eval/references.json`)

A JSON list of objects; each entry pairs with a prediction by `paper_id`:

```json
{
  "paper_id": "iclr2019_demo",
  "review":   {/* a Y object */}
}
```

## 5b. Migrating prior teacher outputs

If you have a legacy `distill_triplets.jsonl` (one record per paper with
`input` / `reasoning` / `review` as plain text / Markdown), convert it with:

```bash
python scripts/migrate_from_distill_project.py \
    --in_file /path/to/distill_triplets.jsonl \
    --out_dir data/full/distilled \
    [--rerun_retrieval]
```

Each record is reshaped into the per-paper schema in §2. Without
`--rerun_retrieval` the migrator sets `Flag_i_n` to `"No Evidence"` (for
verifiable claims) or `"Non-verifiable Item"` (for writing-style claims);
`--rerun_retrieval` re-invokes the EGTR-Review retriever to populate
`E_i_n` and overwrite `Flag_i_n` per the labelling rule in prompts/ §3.

## 6. Cached LLM-as-Judge results (`data/samples/cached_llm_judgments.json`)

```json
{
  "gemini:iclr2019_demo":   {"pertinency": 8, "usefulness": 9, "...": "..."},
  "deepseek:iclr2019_demo": {"...": "..."},
  "claude:iclr2019_demo":   {"...": "..."}
}
```

Keys are `<judge_name>:<paper_id>` so reviewers can replay metric
computation without re-issuing API calls.
