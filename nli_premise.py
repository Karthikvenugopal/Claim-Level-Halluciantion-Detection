"""
Shared premise construction for the RoBERTa NLI classifier — used identically by
training (finetuning/roberta_nli_finetune.py) and evaluation (benchmark.py) so the
two never drift.

Two modes (PREMISE_MODE):

  "abstract"  — premise = the full cited abstract. Honest "retrieve the context"
                setting; noisier, lower F1.

  "rationale" — the standard SciFact *label-prediction-given-evidence* setting:
                  * SUPPORT / CONTRADICT -> the gold evidence (rationale) sentences
                  * NEI                  -> a length-matched RANDOM sample of
                                            non-evidence sentences from the cited
                                            abstract (NEI claims have no rationale)
                Using length-matched NEI premises is deliberate: it prevents the
                model from cheating on premise length (short=>has-verdict,
                long=>NEI) and forces it to actually reason about the content.
                This uses gold evidence at inference time (oracle), so it measures
                the classifier in isolation, not an end-to-end retrieval pipeline.
"""

import random

SUPPORT, CONTRADICT, NEI = "SUPPORT", "CONTRADICT", "NEI"
NEI_K_RANGE = (1, 3)   # sentences sampled for NEI premises (≈ rationale length)


def _abstract(corpus: dict, did) -> list | None:
    doc = corpus.get(did)
    if doc is None:
        return None
    return doc["abstract"] if isinstance(doc, dict) else doc


def _full_abstract(evidence: dict, cited_doc_ids: list, corpus: dict) -> str | None:
    order = [int(k) for k in evidence.keys()] + list(cited_doc_ids)
    for did in order:
        ab = _abstract(corpus, did)
        if ab:
            return " ".join(ab)
    return None


def _rationale_sentences(evidence: dict, corpus: dict) -> str | None:
    """Concatenate the gold evidence (rationale) sentences from the evidence doc."""
    for k, groups in (evidence or {}).items():
        ab = _abstract(corpus, int(k))
        if not ab:
            continue
        groups = groups if isinstance(groups, list) else [groups]
        idxs = sorted({i for g in groups for i in (g.get("sentences") or [])})
        sents = [ab[i] for i in idxs if 0 <= i < len(ab)]
        if sents:
            return " ".join(sents)
    return None


def _sampled_nei(cited_doc_ids: list, corpus: dict, seed_key) -> str | None:
    """Deterministically sample 1–3 non-evidence sentences as an NEI premise."""
    for did in cited_doc_ids:
        ab = _abstract(corpus, did)
        if ab:
            rng = random.Random(f"nei::{seed_key}")
            k = min(rng.randint(*NEI_K_RANGE), len(ab))
            idxs = sorted(rng.sample(range(len(ab)), k))
            return " ".join(ab[i] for i in idxs)
    return None


def build_premise(evidence: dict, cited_doc_ids: list, label: str, corpus: dict,
                  mode: str = "abstract", seed_key=0) -> str | None:
    if mode == "rationale":
        if label in (SUPPORT, CONTRADICT):
            return _rationale_sentences(evidence, corpus) \
                or _full_abstract(evidence, cited_doc_ids, corpus)
        return _sampled_nei(cited_doc_ids, corpus, seed_key) \
            or _full_abstract(evidence, cited_doc_ids, corpus)
    return _full_abstract(evidence, cited_doc_ids, corpus)
