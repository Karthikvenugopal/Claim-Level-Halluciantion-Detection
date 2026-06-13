"""
Claim-level NLI benchmark harness.

Implements the two resume deliverables:

  (1) Benchmarks the fine-tuned RoBERTa NLI classifier against prompted-GPT-3.5
      and GPT-3.5-as-judge baselines on the held-out SciFact dev split, reporting
      per-class / macro F1 and accuracy for each backend.

  (2) Claim-level evaluation: decomposes each claim into atomic sub-claims,
      retrieves supporting context per atom, scores every atom with every backend,
      and emits a per-claim failure-attribution report (which atoms each backend
      flagged). Enable with BENCHMARK_ATOMIC=1.

RoBERTa runs locally (no API). The GPT-3.5 backends require OPENAI_API_KEY; if it
is unset they are SKIPPED with a clear message rather than faked.

Premise per claim is built by nli_premise.build_premise (PREMISE_MODE):
  rationale = gold evidence sentences (standard SciFact label-prediction)
  abstract  = full cited abstract (harder retrieval-style setting)

Usage:
    python benchmark.py                      # all available backends, claim-level
    BENCHMARK_ATOMIC=1 python benchmark.py   # also run atomic attribution
"""

import os
import sys
import json
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "level3"))
sys.path.insert(0, str(ROOT / "FActScore"))

from sklearn.metrics import accuracy_score, classification_report, f1_score
from dotenv import load_dotenv

from benchmark_backends import GPT35NLIBackend, LLMJudgeBackend, RobertaBackend
from nli_premise import build_premise

load_dotenv()

LABELS       = ["SUPPORT", "CONTRADICT", "NEI"]
DATA_DIR     = ROOT / os.environ.get("DATA_DIR", "data/scifact/data")
RESULTS_DIR  = ROOT / os.environ.get("RESULTS_DIR", "results")
DEV_FILE     = os.environ.get("BENCHMARK_DATA", "claims_dev.jsonl")
ATOMIC       = os.environ.get("BENCHMARK_ATOMIC", "0") == "1"
ATOMIC_N     = int(os.environ.get("BENCHMARK_ATOMIC_N", 30))
TOP_K        = int(os.environ.get("TOP_K_ABSTRACTS", 3))
# Premise for the claim-level benchmark: "abstract" (retrieve full context) or
# "rationale" (gold evidence sentences; oracle). Must match how RoBERTa was trained.
PREMISE_MODE = os.environ.get("PREMISE_MODE", "abstract")


# ── Data ────────────────────────────────────────────────────────────────────────

def claim_label(evidence: dict) -> str:
    if not evidence:
        return "NEI"
    labels = set()
    for v in evidence.values():
        for item in (v if isinstance(v, list) else [v]):
            if isinstance(item, dict):
                labels.add(item.get("label", ""))
    if "SUPPORT" in labels:
        return "SUPPORT"
    if "CONTRADICT" in labels:
        return "CONTRADICT"
    return "NEI"


def load_corpus() -> dict:
    corpus = {}
    for line in (DATA_DIR / "corpus.jsonl").open():
        doc = json.loads(line)
        corpus[doc["doc_id"]] = {"title": doc["title"], "abstract": doc["abstract"]}
    return corpus


def load_dev_claims() -> list:
    rows = [json.loads(l) for l in (DATA_DIR / DEV_FILE).open()]
    return [{
        "id": r["id"],
        "claim": r["claim"],
        "ground_truth": claim_label(r.get("evidence", {})),
        "cited_doc_ids": r.get("cited_doc_ids", []),
        "evidence": r.get("evidence", {}),
    } for r in rows]


# ── Metrics / reporting ─────────────────────────────────────────────────────────

def evaluate(name: str, gt: list, pred: list) -> dict:
    acc = accuracy_score(gt, pred)
    macro = f1_score(gt, pred, labels=LABELS, average="macro", zero_division=0)
    per_class = f1_score(gt, pred, labels=LABELS, average=None, zero_division=0)
    print(f"\n{'='*70}\n  {name}  vs SciFact dev ground truth\n{'='*70}")
    print(classification_report(gt, pred, labels=LABELS, zero_division=0))
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "per_class_f1": {l: float(f) for l, f in zip(LABELS, per_class)},
    }


def print_table(metrics: dict):
    print(f"\n{'='*70}\n  BENCHMARK SUMMARY — claim-level NLI on SciFact dev\n{'='*70}")
    print(f"  {'Backend':<28} {'Accuracy':>10} {'Macro F1':>10}")
    print("  " + "-" * 50)
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["macro_f1"], reverse=True)
    for i, (name, m) in enumerate(ranked):
        star = "  ★ best" if i == 0 and len(ranked) > 1 else ""
        print(f"  {name:<28} {m['accuracy']*100:>9.1f}% {m['macro_f1']:>10.3f}{star}")
    print("  " + "=" * 50)


# ── Atomic decomposition (GPT-3.5) ───────────────────────────────────────────────

async def decompose_claim(llm, claim: str) -> list[str]:
    from langchain_core.messages import SystemMessage, HumanMessage
    sys_p = (
        "Break the scientific claim into atomic, independently checkable factual "
        "statements. Output each atomic statement on its own line, with no numbering "
        "or extra text. If the claim is already atomic, output it unchanged."
    )
    try:
        resp = await llm.ainvoke(
            [SystemMessage(content=sys_p), HumanMessage(content=f"Claim: {claim}")]
        )
        atoms = [ln.strip(" -*•\t") for ln in resp.content.splitlines() if ln.strip()]
        return atoms or [claim]
    except Exception as e:
        print(f"    [decompose] error: {e}")
        return [claim]


async def run_atomic(claims, corpus, backends):
    """Decompose -> retrieve per atom -> score per backend -> attribution report."""
    from langchain_openai import ChatOpenAI
    from bm25_retriever import CorpusRetriever

    print(f"\n{'='*70}\n  ATOMIC MODE — decompose / retrieve / score (first {ATOMIC_N} claims)"
          f"\n{'='*70}")
    key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(model=os.environ.get("GPT35_MODEL", "gpt-3.5-turbo"),
                     api_key=key, temperature=0.0)
    retriever = CorpusRetriever(corpus)

    subset = claims[:ATOMIC_N]
    api_backends = [b for _, b in backends.items() if b.available()]

    report = []
    for n, c in enumerate(subset):
        atoms = await decompose_claim(llm, c["claim"])
        atom_items = []
        for a in atoms:
            top = retriever.retrieve(a, k=TOP_K)
            premise = " ".join(s for doc in top for s in doc["abstract"]) if top else ""
            atom_items.append({"id": f"{c['id']}::{len(atom_items)}", "claim": a,
                               "premise": premise})

        # Score every atom with every available backend
        per_backend = {}
        for b in api_backends:
            verdicts = await b.classify(atom_items)
            per_backend[b.name] = [verdicts[it["id"]] for it in atom_items]

        atoms_out = []
        for j, a in enumerate(atoms):
            atoms_out.append({
                "atom": a,
                "verdicts": {bn: vs[j] for bn, vs in per_backend.items()},
            })
        # Failure attribution: atoms any backend flagged as non-SUPPORT
        flagged = [ao for ao in atoms_out
                   if any(v != "SUPPORT" for v in ao["verdicts"].values())]
        report.append({
            "id": c["id"], "claim": c["claim"], "ground_truth": c["ground_truth"],
            "n_atoms": len(atoms), "atoms": atoms_out,
            "flagged_atoms": [ao["atom"] for ao in flagged],
        })
        print(f"  [{n+1:2d}/{len(subset)}] {c['ground_truth']:<10} "
              f"{len(atoms)} atoms, {len(flagged)} flagged  ({c['claim'][:45]})")

    out = RESULTS_DIR / "benchmark_attribution.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved per-claim failure attribution -> {out}")


# ── Orchestration ────────────────────────────────────────────────────────────────

def build_roberta_backend(corpus):
    try:
        from roberta_runner import RobertaNLIRunner
        runner = RobertaNLIRunner(corpus=corpus)
        return RobertaBackend(runner)
    except FileNotFoundError as e:
        print(f"\n[RoBERTa] {e}")
        return None
    except Exception as e:
        print(f"\n[RoBERTa] could not load model: {e}")
        return None


async def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    claims = load_dev_claims()
    gt = [c["ground_truth"] for c in claims]
    dist = {l: gt.count(l) for l in LABELS}
    print(f"Loaded {len(claims)} dev claims from {DEV_FILE}  dist={dist}")
    print(f"Premise mode: {PREMISE_MODE}")

    # Resolve one premise per claim (shared with training via nli_premise.py)
    items = []
    for c in claims:
        premise = build_premise(
            c.get("evidence", {}), c.get("cited_doc_ids", []), c["ground_truth"],
            corpus, mode=PREMISE_MODE, seed_key=c["id"],
        )
        items.append({"id": c["id"], "claim": c["claim"], "premise": premise})
    n_no_prem = sum(1 for it in items if not it["premise"])
    if n_no_prem:
        print(f"  [!] {n_no_prem} claims had no resolvable premise (scored NEI)")

    # Assemble backends; skip the ones that cannot run (no model / no API key)
    backends = {}
    roberta = build_roberta_backend(corpus)
    if roberta and roberta.available():
        backends["roberta"] = roberta
    backends["gpt35_nli"] = GPT35NLIBackend()
    backends["llm_judge"] = LLMJudgeBackend()

    metrics, all_preds = {}, {}
    for key, b in backends.items():
        if not b.available():
            reason = ("OPENAI_API_KEY not set" if key != "roberta"
                      else "model not found")
            print(f"\n[skip] {b.name}: {reason} — run it later once available.")
            continue
        print(f"\nRunning backend: {b.name} ...")
        verdicts = await b.classify(items)
        pred = [verdicts[c["id"]] for c in claims]
        all_preds[b.name] = pred
        metrics[b.name] = evaluate(b.name, gt, pred)

    if metrics:
        print_table(metrics)
    else:
        print("\nNo backends ran. Train RoBERTa and/or set OPENAI_API_KEY.")

    # Persist per-claim verdicts + metrics
    records = []
    for i, c in enumerate(claims):
        row = {"id": c["id"], "claim": c["claim"], "ground_truth": c["ground_truth"]}
        for name, pred in all_preds.items():
            row[name] = pred[i]
        records.append(row)
    (RESULTS_DIR / "benchmark_results.json").write_text(json.dumps(records, indent=2))
    (RESULTS_DIR / "benchmark_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {RESULTS_DIR/'benchmark_results.json'}"
          f"\nSaved -> {RESULTS_DIR/'benchmark_metrics.json'}")

    if ATOMIC:
        key = os.environ.get("OPENAI_API_KEY", "")
        if key and not key.startswith("#"):
            await run_atomic(claims, corpus, backends)
        else:
            print("\n[atomic] skipped — needs OPENAI_API_KEY for claim decomposition.")


if __name__ == "__main__":
    asyncio.run(main())
