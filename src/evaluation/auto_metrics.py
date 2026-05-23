"""Auto metrics: ROUGE-1/2/L, BERTScore, sentence-level SN-P/R/F1, ITF-IDF."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable


def _split_sents(text: str) -> list[str]:
    import re
    sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [s for s in sents if s]


def _rouge(preds: list[str], refs: list[str]) -> dict[str, float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)
        for k in agg:
            agg[k] += scores[k].fmeasure
    n = max(1, len(preds))
    return {k: v / n for k, v in agg.items()}


def _bertscore(preds: list[str], refs: list[str], model: str, lang: str) -> float:
    from bert_score import score as bs_score
    _, _, f1 = bs_score(preds, refs, model_type=model, lang=lang, verbose=False)
    return float(f1.mean().item())


def _sn_f1(preds: list[str], refs: list[str], threshold: float = 0.55) -> dict[str, float]:
    """Sentence-level matching F1 using TF-IDF cosine similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    precisions, recalls = [], []
    for p, r in zip(preds, refs):
        ps, rs = _split_sents(p), _split_sents(r)
        if not ps or not rs:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        vec = TfidfVectorizer().fit(ps + rs)
        sim = cosine_similarity(vec.transform(ps), vec.transform(rs))
        precisions.append(float((sim.max(axis=1) >= threshold).mean()))
        recalls.append(float((sim.max(axis=0) >= threshold).mean()))
    p_avg = sum(precisions) / max(1, len(precisions))
    r_avg = sum(recalls) / max(1, len(recalls))
    f1 = 2 * p_avg * r_avg / (p_avg + r_avg) if (p_avg + r_avg) else 0.0
    return {"SN-P": p_avg, "SN-R": r_avg, "SN-F1": f1}


def _itf_idf(preds: list[str], refs: list[str]) -> float:
    """Inverse-token-frequency * inverse-document-frequency overlap."""
    df: Counter = Counter()
    for r in refs:
        df.update(set(r.split()))
    N = max(1, len(refs))
    scores = []
    for p, r in zip(preds, refs):
        p_tokens, r_tokens = p.split(), r.split()
        if not r_tokens:
            scores.append(0.0)
            continue
        overlap = set(p_tokens) & set(r_tokens)
        score = 0.0
        for tok in overlap:
            idf = math.log(N / max(1, df[tok]))
            score += idf
        scores.append(score / max(1, len(r_tokens)))
    return sum(scores) / max(1, len(scores))


def compute_auto_metrics(preds: list[str], refs: list[str], cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    rouge = _rouge(preds, refs)
    bert = _bertscore(
        preds,
        refs,
        model=cfg.get("bertscore_model", "roberta-large"),
        lang=cfg.get("bertscore_lang", "en"),
    )
    sn = _sn_f1(preds, refs, threshold=cfg.get("sn_threshold", 0.55))
    itf = _itf_idf(preds, refs)
    return {**rouge, "BERTScore": bert, **sn, "ITF-IDF": itf}
