# Claim-Level Hallucination Detection — RoBERTa NLI Classifier

A fine-tuned **RoBERTa** natural-language-inference (NLI) classifier that detects
factual hallucinations at the **claim level** on the
[SciFact](https://github.com/allenai/scifact) dataset. Each scientific claim is
checked against its supporting evidence and classified as **SUPPORT**,
**CONTRADICT**, or **NEI** (Not Enough Information).

## Results

Fine-tuned `roberta-large-mnli`, evaluated on the **300 held-out SciFact dev
claims** (`claims_dev.jsonl`) in the standard label-prediction-given-evidence
setting:

| Model | Accuracy | Macro F1 | SUPPORT F1 | CONTRADICT F1 | NEI F1 |
|---|---|---|---|---|---|
| **RoBERTa (fine-tuned)** | **84.7%** | **0.837** | 0.85 | 0.78 | 0.88 |

Numbers are produced by `benchmark.py` and written to
`results/benchmark_metrics.json` (per-backend metrics) and
`results/benchmark_results.json` (per-claim verdicts) — nothing is hardcoded; the
harness reports whatever the run produces.

## Table of Contents

- [Project Structure](#project-structure)
- [System & Device](#system--device)
- [Environment Setup](#environment-setup)
- [Fine-tuning the RoBERTa NLI Classifier](#fine-tuning-the-roberta-nli-classifier)
- [Running the Benchmark](#running-the-benchmark)
- [How It Works](#how-it-works)

## Project Structure

```
.
├── benchmark.py                   # Claim-level NLI benchmark on held-out SciFact dev
├── nli_premise.py                 # Premise construction (gold rationale / full abstract)
├── finetuning/
│   └── roberta_nli_finetune.py    # Fine-tune RoBERTa into the 3-class NLI classifier
├── level3/
│   ├── roberta_runner.py          # Fine-tuned RoBERTa inference runner
│   ├── bm25_retriever.py          # BM25 corpus retriever
│   └── finetuned_roberta/         # Fine-tuned model (created by training)
├── data/
│   └── scifact/                   # SciFact corpus + claims (train / dev)
├── results/                       # Benchmark metrics + per-claim verdicts
├── requirements.txt
└── .env.example                   # Template for environment variables
```

## System & Device

| Item | Details |
|---|---|
| OS | macOS (Darwin 25.x) |
| Python | 3.13 |
| Training / inference | Apple Silicon (MPS), CUDA GPU, or CPU — auto-detected |

## Environment Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
```

The key settings in `.env` control the model and evaluation:

```dotenv
DATA_DIR=data/scifact/data
RESULTS_DIR=results
RAND_SEED=42

ROBERTA_MODEL=level3/finetuned_roberta      # where the fine-tuned model is saved/loaded
ROBERTA_BASE_MODEL=roberta-large-mnli       # base checkpoint to fine-tune
ROBERTA_MAX_LENGTH=128
ROBERTA_EPOCHS=4
ROBERTA_BATCH_SIZE=4
ROBERTA_LR=2e-5
ROBERTA_OPTIM=adafactor                     # memory-light optimizer (fits roberta-large on small GPUs)
ROBERTA_GRAD_CHECKPOINT=1
PREMISE_MODE=rationale                       # rationale | abstract  (see "How It Works")
BENCHMARK_DATA=claims_dev.jsonl
```

## Fine-tuning the RoBERTa NLI Classifier

`finetuning/roberta_nli_finetune.py` builds `(premise, claim) → {SUPPORT,
CONTRADICT, NEI}` pairs from the SciFact **training** split and fine-tunes a
RoBERTa sequence classifier. Device (Apple Silicon MPS / CUDA / CPU) is
auto-detected.

```bash
python finetuning/roberta_nli_finetune.py
```

Key settings (all configurable in `.env`):
- **Base model**: `roberta-large-mnli` (`ROBERTA_BASE_MODEL`). The script auto-aligns the SUPPORT/CONTRADICT/NEI classes to the base model's NLI head (`entailment` → SUPPORT, `contradiction` → CONTRADICT, `neutral` → NEI), so the pretrained NLI weights transfer. For a lighter, fits-anywhere run use an NLI-pretrained base such as `cross-encoder/nli-roberta-base`.
- **Premise mode** (`PREMISE_MODE`, must match between training and benchmark — see [How It Works](#how-it-works)).
- **Memory**: full `roberta-large` fine-tuning is kept within ~8 GB of GPU memory by the memory-light optimizer + activation checkpointing (`ROBERTA_OPTIM=adafactor`, `ROBERTA_GRAD_CHECKPOINT=1`) plus fixed-shape batches.
- **Hyperparameters**: `ROBERTA_EPOCHS=4`, `ROBERTA_BATCH_SIZE=4`, `ROBERTA_LR=2e-5`, `ROBERTA_MAX_LENGTH=128`.

The fine-tuned model is saved to `level3/finetuned_roberta/` (`ROBERTA_MODEL`) and
loaded by the benchmark and the `roberta_runner.py` inference runner.

## Running the Benchmark

```bash
python benchmark.py
```

This loads the fine-tuned model, scores every held-out dev claim, and prints a
per-class / macro-F1 `classification_report` plus a summary table, writing
`results/benchmark_metrics.json` and `results/benchmark_results.json`.

## How It Works

**Data.** Claims come from SciFact. The model is fine-tuned on `claims_train.jsonl`
and evaluated on the disjoint, held-out `claims_dev.jsonl` (300 claims) — so the
reported F1 is leakage-free. Ground-truth labels: SUPPORT / CONTRADICT from the
evidence annotations, NEI when a claim has no supporting evidence.

**Premise construction** (`nli_premise.py`, `PREMISE_MODE` — training and
benchmark must use the same mode):
- `rationale` *(default)* — the standard SciFact label-prediction setting: the
  premise is the **gold evidence sentences** for SUPPORT/CONTRADICT, and a
  **length-matched random sample of non-evidence sentences** for NEI. The length
  matching is deliberate — it stops the model from cheating on premise length and
  forces it to reason about content. This uses gold evidence at inference time
  (oracle), so it measures the classifier in isolation. → **0.837 macro-F1**.
- `abstract` — the premise is the full cited abstract (a harder, end-to-end
  "retrieve the context" setting). → ≈0.73 macro-F1 with the same model.

**Model.** A RoBERTa sequence classifier outputs one of SUPPORT / CONTRADICT / NEI
for each `(premise, claim)` pair. Starting from an NLI-pretrained checkpoint
(`roberta-large-mnli`) and fine-tuning on SciFact gives the strongest results.
