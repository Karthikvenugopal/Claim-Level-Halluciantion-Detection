"""
Fine-tune a RoBERTa sequence classifier as the project's NLI signal.

Builds (premise, claim) -> {SUPPORT, CONTRADICT, NEI} pairs from the SciFact
training split and fine-tunes `roberta-large-mnli` (or any RoBERTa checkpoint
in ROBERTA_BASE_MODEL) into a 3-class claim-verification model. The fine-tuned
model is saved to ROBERTA_MODEL (default: level3/finetuned_roberta) and is
consumed at inference time by level3/roberta_runner.py and benchmark.py.

Premise  = the cited abstract (joined sentences).
Label    = the SciFact claim-level verdict (empty evidence -> NEI).

Train split = claims_train.jsonl, MINUS the deterministic claims that the
Level 3 pipeline samples for evaluation (so the ensemble stays leakage-free).
A held-out slice of the training pairs is used for the in-training validation
metric; the headline benchmark (benchmark.py) evaluates on claims_dev.jsonl.

Usage:
    python finetuning/roberta_nli_finetune.py
    ROBERTA_BASE_MODEL=roberta-base python finetuning/roberta_nli_finetune.py   # lighter

Runs on Apple Silicon (MPS), CUDA, or CPU — device is auto-detected.
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from dotenv import load_dotenv

load_dotenv()

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nli_premise import build_premise

# ── Config (env-driven, with defaults so a bare checkout still runs) ────────────
DATA_DIR        = ROOT / os.environ.get("DATA_DIR", "data/scifact/data")
BASE_MODEL      = os.environ.get("ROBERTA_BASE_MODEL", "roberta-large-mnli")
OUTPUT_DIR      = ROOT / os.environ.get("ROBERTA_MODEL", "level3/finetuned_roberta")
MAX_LENGTH      = int(os.environ.get("ROBERTA_MAX_LENGTH", 256))
EPOCHS          = float(os.environ.get("ROBERTA_EPOCHS", 3))
BATCH_SIZE      = int(os.environ.get("ROBERTA_BATCH_SIZE", 8))
LR              = float(os.environ.get("ROBERTA_LR", 2e-5))
SEED            = int(os.environ.get("RAND_SEED", 42))
N_CLAIMS        = int(os.environ.get("N_CLAIMS", 10))   # to mirror the level3 holdout
VAL_FRACTION    = 0.15
# "abstract" (full cited abstract) or "rationale" (gold evidence sentences;
# length-matched random sentences for NEI). See nli_premise.py.
PREMISE_MODE    = os.environ.get("PREMISE_MODE", "abstract")
# Memory knobs — set ROBERTA_OPTIM=adafactor + ROBERTA_GRAD_CHECKPOINT=1 to fit
# roberta-large on a small (8 GB) GPU; AdamW's optimizer states otherwise OOM.
OPTIM           = os.environ.get("ROBERTA_OPTIM", "adamw_torch")
GRAD_CKPT       = os.environ.get("ROBERTA_GRAD_CHECKPOINT", "0") == "1"
# Up-weight rare classes (CONTRADICT) in the loss to counter label imbalance.
CLASS_WEIGHTS   = os.environ.get("ROBERTA_CLASS_WEIGHTS", "0") == "1"

def resolve_label_scheme(base_model: str):
    """
    Align our 3 classes to the base model's existing NLI head when it has one
    (contradiction/entailment/neutral), so the pretrained NLI weights transfer
    in the right slots. Falls back to a fixed order for non-NLI checkpoints.
    Returns (id2label, label2id).
    """
    from transformers import AutoConfig
    default = {0: "CONTRADICT", 1: "NEI", 2: "SUPPORT"}
    try:
        cfg = AutoConfig.from_pretrained(base_model)
        scheme = {}
        for i, lab in (cfg.id2label or {}).items():
            u = str(lab).upper()
            if "CONTRA" in u:
                scheme[int(i)] = "CONTRADICT"
            elif "ENTAIL" in u:
                scheme[int(i)] = "SUPPORT"
            elif "NEUTRAL" in u:
                scheme[int(i)] = "NEI"
        if len(scheme) == 3 and set(scheme.values()) == {"CONTRADICT", "NEI", "SUPPORT"}:
            id2label = scheme
            print(f"  label scheme : aligned to NLI head -> {scheme}")
        else:
            id2label = default
            print(f"  label scheme : default (non-NLI base) -> {default}")
    except Exception:
        id2label = default
    return id2label, {v: k for k, v in id2label.items()}


# ── Data ────────────────────────────────────────────────────────────────────────

def load_corpus() -> dict:
    corpus = {}
    for line in (DATA_DIR / "corpus.jsonl").open():
        doc = json.loads(line)
        corpus[doc["doc_id"]] = doc["abstract"]
    return corpus


def claim_label(evidence: dict) -> str:
    """SciFact claim-level verdict (matches level3.get_ground_truth)."""
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


def level3_eval_ids() -> set:
    """
    Reproduce the exact set of claim ids that level3.Level3.load_claims samples
    for evaluation, so we can exclude them from training (leakage-free ensemble).
    Mirrors that function's RNG usage step for step.
    """
    rows = [json.loads(l) for l in (DATA_DIR / "claims_train.jsonl").open()]
    by_label = {"SUPPORT": [], "CONTRADICT": [], "NEI": []}
    for ex in rows:
        gt = claim_label(ex.get("evidence", {}))
        by_label[gt].append({"id": ex["id"]})

    rng = random.Random(SEED)
    per_class = N_CLAIMS // 3
    sampled = []
    for i, label in enumerate(["SUPPORT", "CONTRADICT", "NEI"]):
        take = per_class + (1 if i < N_CLAIMS % 3 else 0)
        pool = by_label[label][:]
        rng.shuffle(pool)
        sampled.extend(pool[:take])
    rng.shuffle(sampled)
    return {c["id"] for c in sampled}


def build_pairs(corpus: dict, exclude_ids: set, label2id: dict) -> list[dict]:
    rows = [json.loads(l) for l in (DATA_DIR / "claims_train.jsonl").open()]
    pairs = []
    skipped = 0
    for ex in rows:
        if ex["id"] in exclude_ids:
            continue
        label = claim_label(ex.get("evidence", {}))
        premise = build_premise(
            ex.get("evidence", {}), ex.get("cited_doc_ids", []), label, corpus,
            mode=PREMISE_MODE, seed_key=ex["id"],
        )
        if not premise:
            skipped += 1
            continue
        pairs.append({"premise": premise, "claim": ex["claim"], "label": label2id[label]})
    if skipped:
        print(f"  [build_pairs] skipped {skipped} claims with no resolvable abstract")
    return pairs


# ── Torch dataset ────────────────────────────────────────────────────────────────

class NLIPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: list[dict], tokenizer):
        self.pairs = pairs
        self.tok = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        # Pad to a FIXED MAX_LENGTH so every batch has identical shape. On MPS,
        # variable-length batches (dynamic padding) fragment the allocator cache
        # and OOM mid-run; fixed shapes reuse one buffer. MAX_LENGTH is small in
        # rationale mode, so this is cheap.
        enc = self.tok(
            p["premise"], p["claim"],
            truncation=True, max_length=MAX_LENGTH, padding="max_length",
        )
        enc["labels"] = p["label"]
        return {k: torch.tensor(v) for k, v in enc.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy to counter label imbalance."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self._class_weights.to(outputs.logits.device) if self._class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


# ── Train ─────────────────────────────────────────────────────────────────────

def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    device = pick_device()
    print(f"{'='*70}\n  RoBERTa NLI fine-tuning\n{'='*70}")
    print(f"  base model : {BASE_MODEL}")
    print(f"  premise    : {PREMISE_MODE}")
    print(f"  device     : {device}")
    print(f"  output dir : {OUTPUT_DIR}")

    id2label, label2id = resolve_label_scheme(BASE_MODEL)
    labels = [id2label[i] for i in range(3)]

    corpus = load_corpus()
    print(f"  corpus     : {len(corpus)} docs")

    holdout = level3_eval_ids()
    pairs = build_pairs(corpus, exclude_ids=holdout, label2id=label2id)
    print(f"  pairs      : {len(pairs)}  (excluded {len(holdout)} level3 eval ids)")
    dist = {l: sum(1 for p in pairs if p["label"] == label2id[l]) for l in labels}
    print(f"  label dist : {dist}")

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRACTION))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"  split      : {len(train_pairs)} train / {len(val_pairs)} val")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    train_ds = NLIPairDataset(train_pairs, tokenizer)
    val_ds = NLIPairDataset(val_pairs, tokenizer)

    args = TrainingArguments(
        output_dir=str(ROOT / "results" / "roberta_train_ckpts"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=SEED,
        fp16=False,   # MPS/CPU train in fp32
        bf16=False,
        optim=OPTIM,
        gradient_checkpointing=GRAD_CKPT,
        gradient_checkpointing_kwargs={"use_reentrant": False} if GRAD_CKPT else None,
        report_to="none",
        # Device (MPS/CUDA/CPU) is auto-detected by Trainer/accelerate in transformers 5.x.
    )

    if CLASS_WEIGHTS:
        counts = np.bincount([p["label"] for p in train_pairs], minlength=3).astype(float)
        weights = counts.sum() / (3 * np.clip(counts, 1, None))
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"  class weights: {dict(zip(labels, weights.round(3)))}")
        trainer = WeightedTrainer(
            model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
            compute_metrics=compute_metrics, class_weights=class_weights,
        )
    else:
        trainer = Trainer(
            model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

    trainer.train()

    # Final validation report
    pred = trainer.predict(val_ds)
    y_pred = np.argmax(pred.predictions, axis=-1)
    y_true = pred.label_ids
    target_names = labels
    print(f"\n{'='*70}\n  VALIDATION REPORT (held-out slice of train)\n{'='*70}")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    print(f"  Macro F1 : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nSaved fine-tuned RoBERTa -> {OUTPUT_DIR}")
    print("Next: run `python benchmark.py` to evaluate on held-out claims_dev.")


if __name__ == "__main__":
    main()
