# Agent Prompts

This directory is the **single source of truth** for the five teacher-side
agent prompts. Each prompt is loaded verbatim from one of the files below by
`src/agents/prompt_loader.py`, which the corresponding agent class imports.

| Agent                                  | File                              |
|----------------------------------------|-----------------------------------|
| 1. Structure Parser Agent              | `01_structure_parser.md`          |
| 2. Key Element Extractor Agent         | `02_key_element_extractor.md`     |
| 3. Evidence Retriever Agent            | `03_evidence_retriever.md`        |
| 4. Verification Reasoner Agent         | `04_verification_reasoner.md`     |
| 5. Review Synthesizer Agent            | `05_review_synthesizer.md`        |

The canonical evidence-state labels used across all five prompts are listed
in [`flag_labels.md`](flag_labels.md).

The LLM-as-Judge prompt used during evaluation lives in
`src/evaluation/llm_judge.py` (loaded as `JUDGE_PROMPT`).
