"""
NLI inference with the fine-tuned RoBERTa sequence classifier.

Drop-in replacement for the previous TinyLlama runner: same public surface
(`run_nli(records) -> [{id, nli_verdict, nli_note}]`) so it plugs straight into
the Level 3 ensemble. The model is produced by
finetuning/roberta_nli_finetune.py and read from ROBERTA_MODEL.

Premise = the cited abstract (oracle via cited_doc_ids); falls back to the BM25
top-1 abstract when a retriever is supplied and no cited doc resolves.
"""

import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

SUPPORT, CONTRADICT, NEI = "SUPPORT", "CONTRADICT", "NEI"


def _canonical(label: str) -> str:
    """Map a model's raw label string onto SUPPORT / CONTRADICT / NEI."""
    u = label.upper()
    if u in (SUPPORT, CONTRADICT, NEI):
        return u
    if "ENTAIL" in u or "SUPPORT" in u:
        return SUPPORT
    if "CONTRA" in u:
        return CONTRADICT
    return NEI


class RobertaNLIRunner:
    def __init__(self, corpus: dict, retriever=None):
        self.corpus = corpus
        self.retriever = retriever
        self.model_path = os.environ.get("ROBERTA_MODEL", "level3/finetuned_roberta")
        self.max_length = int(os.environ.get("ROBERTA_MAX_LENGTH", 256))

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"ROBERTA_MODEL path '{self.model_path}' not found. "
                f"Train it first:  python finetuning/roberta_nli_finetune.py"
            )

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading fine-tuned RoBERTa: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        try:
            self.model.eval().to(self.device)
        except Exception as e:                       # pragma: no cover - MPS edge cases
            print(f"  [!] {self.device} load failed ({e}); using cpu")
            self.device = "cpu"
            self.model.eval().to("cpu")
        # idx -> canonical verdict, read from the saved config
        self.idx2verdict = {
            int(i): _canonical(lbl) for i, lbl in self.model.config.id2label.items()
        }
        print(f"  Labels: {self.model.config.id2label}  on {self.device}")

    # ── Premise resolution ──────────────────────────────────────────────────────

    def _premise(self, record: dict) -> str | None:
        for cid in record.get("cited_doc_ids", []):
            if cid in self.corpus:
                doc = self.corpus[cid]
                abstract = doc["abstract"] if isinstance(doc, dict) else doc
                return " ".join(abstract)
        if self.retriever is not None:
            top = self.retriever.retrieve(record["claim"], k=1)
            if top:
                return " ".join(top[0]["abstract"])
        return None

    # ── Inference ───────────────────────────────────────────────────────────────

    def predict(self, premise: str, claim: str) -> str:
        """Public single-pair inference (used by the benchmark backend)."""
        enc = self.tokenizer(
            premise, claim, return_tensors="pt",
            truncation=True, max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return self.idx2verdict.get(int(logits.argmax(-1).item()), NEI)

    # backwards-compatible alias
    _infer = predict

    def run_nli(self, records: list) -> list:
        print(f"\nRunning RoBERTa NLI on {len(records)} claims...")
        results = []
        for i, r in enumerate(records):
            premise = self._premise(r)
            if not premise:
                print(f"  [{i+1:2d}/{len(records)}] NEI  (no abstract)")
                results.append({"id": r["id"], "nli_verdict": NEI, "nli_note": "NO_ABSTRACT"})
                continue
            verdict = self._infer(premise[:2000], r["claim"])
            print(f"  [{i+1:2d}/{len(records)}] {verdict:<12}  {r['claim'][:55]}")
            results.append({"id": r["id"], "nli_verdict": verdict, "nli_note": "ok"})
        return results
