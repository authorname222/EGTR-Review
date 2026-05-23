# Data

This directory contains **small samples** of training papers, distilled supervision,
and evaluation references. The full data is reconstructed locally from public
sources (PeerRead, OpenReview) — see `data_card.md` and
`scripts/build_full_dataset.sh` (will be released upon acceptance).

```
data/
├── README.md                ← this file
├── data_card.md             ← provenance, licensing, statistics
└── samples/
    ├── train_papers/        ← 5–10 papers (content only; NO human reviews)
    ├── distilled/           ← 5–10 (L', T, Y) triples produced by the teacher
    └── eval/                ← demo papers + cached references for quick start
```

⚠️ The samples are intentionally small. They are sufficient to exercise the
pipeline end-to-end but cannot reproduce the paper numbers; full reproduction
requires the data construction step described in `data_card.md`.

## Anonymity notice

The sample papers are public ICLR submissions selected to **avoid any link
to the authors of this submission**: none of them are authored by the authors,
their advisors, or known collaborators.
