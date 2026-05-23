# Leakage Control

EGTR-Review explicitly prevents the teacher and student from observing
ground-truth peer reviews during data construction, training, and
evaluation. This file documents every leakage source we considered and
the corresponding mitigation.

## 1. External evidence retrieval

The Evidence Retriever Agent queries SerpApi, the arXiv API, and the
Semantic Scholar Graph API. Three classes of records are filtered out
before any record reaches the LLM:

| Category                                            | Mitigation                                                                                          |
|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| OpenReview-hosted reviews / rebuttals / decisions   | `BLOCKED_DOMAINS = ("openreview.net",)` in `src/utils/leak_filter.py`. Any record whose `url` contains an `openreview.net` host is dropped before LLM ingestion. |
| Review-like text snippets                           | `BLOCKED_KEYWORDS` substring filter on the candidate's `title + abstract` (case-insensitive): `rebuttal`, `official review`, `meta-review`, `decision: accept`, `decision: reject`, `reviewer 1/2/3`. |
| Author rebuttals, ratings, confidence scores        | The Evidence Retriever's system prompt (`docs/prompts/03_evidence_retriever.md`) explicitly forbids using OpenReview reviews, author responses, decisions, ratings, or confidence scores as evidence. |

The filter runs after every retrieval (and after every query-refinement
round) so that re-ranking by the LLM cannot bring filtered records back.

The filter switch is enforced via `configs/teacher_pipeline.yaml`:

```yaml
teacher:
  evidence_retriever:
    leak_filter: true   # must remain true; see docs/leakage_control.md
```

Reviewers can audit the runtime behaviour by importing
`src.utils.leak_filter.filter_review_leakage` and feeding it a candidate
list — every dropped entry is silently filtered, never mutated.

## 2. Distillation data

The teacher generates `(L'_i, T_i, Y_i)` for each training paper without
ever loading the ground-truth review. Specifically:

- The Structure Parser Agent and Key Element Extractor Agent only see the
  paper text (`title`, `abstract`, `sections`, `citations`) from the input
  JSON. No `review`, `decision`, or `rating` field is exposed.
- The Evidence Retriever Agent only sees the retrieval query `Q_i_n` and
  the leak-filtered external snippets — never the human review for the
  same paper.
- The Verification Reasoner Agent and Review Synthesizer Agent only see
  `L'_i` and `T_i`. They do not have read access to any per-paper review
  reference.

The student is trained on `D_distill = {(L'_i, T_i, Y_i)}` where `Y_i`
is the teacher-side synthesized review — **not** the human review. Human
references are only loaded at evaluation time.

## 3. Evaluation references

`data/samples/eval/references.json` (and the full-set equivalent) is
loaded by `src.evaluation.run_eval.main` strictly to score predictions
via ROUGE / BERTScore / SN / ITF-IDF / LLM-as-Judge. References are
never broadcast back into training or retrieval:

- `run_eval` does not write to the teacher's `--output` directory.
- The LLM-as-Judge prompt (`src/evaluation/llm_judge.py :: JUDGE_PROMPT`)
  includes both reference and candidate review texts because that's the
  defined task of the judge. The judge runs only at scoring time and is
  isolated from any training-time code path.

## 4. Identity leakage in distilled data

`src/utils/anonymizer.py` provides a best-effort scrub for free-text
fields written into `L'_i.C_i_n`. The teacher pipeline calls it before
saving each per-paper JSON:

- E-mails (`[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`) → `[EMAIL]`
- ORCIDs (`\d{4}-\d{4}-\d{4}-\d{3}[\dX]`) → `[ORCID]`
- Affiliations / "Author Information" / "Correspondence" headers →
  `[AFFILIATION REMOVED]`

This is a defence-in-depth scrub; the primary mitigation is to strip
these fields during paper-source preprocessing.

## 5. Anonymous repository

While this repository is hosted for double-blind review:

- `assets/README.md` mandates PNG metadata removal via `exiftool` for
  every figure exported by the authors.
- `.gitignore` excludes `.env`, `*.key`, `secrets.json`, `.identity_notes`,
  and the local-only `anonymity_check.sh`.
- The README's BibTeX and acknowledgements blocks are anonymized
  placeholders; they will be populated only after acceptance.
- Per-paper teacher outputs do not include the paper's
  OpenReview / arXiv ID in any path that points back to a non-anonymous
  source.

## 6. Auditing a run

```bash
# 1. The leak-filter is on (must print "leak_filter: true").
grep "leak_filter" configs/teacher_pipeline.yaml

# 2. No openreview.net URL appears in the teacher's per-paper outputs.
grep -RIl "openreview.net" data/full/distilled/ 2>/dev/null || \
  echo "(no openreview.net references — leakage filter working)"

# 3. Per-paper resume log shows leak-filter is engaged for every retrieval.
python -m src.teacher_pipeline --output data/full/distilled --status
```
