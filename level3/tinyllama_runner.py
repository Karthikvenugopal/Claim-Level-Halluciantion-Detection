import os
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from dotenv import load_dotenv

load_dotenv()

SUPPORT    = "SUPPORT"
CONTRADICT = "CONTRADICT"
NEI        = "NEI"

_SYSTEM_PROMPT = (
    "You are performing natural language inference for open-domain claim verification.\n"
    "Given a premise and a claim, output exactly one label:\n"
    "SUPPORT\nCONTRADICT\nNOT_ENOUGH_INFO\n\n"
)

_LORA_LABEL_MAP = {
    "support":         SUPPORT,
    "contradict":      CONTRADICT,
    "not_enough_info": NEI,
    "not_enough":      NEI,
    "nei":             NEI,
}


class TinyLlamaNLIRunner:
    """
    Runs NLI inference using a fine-tuned TinyLlama model.
    Auto-detects model type from the presence of adapter_config.json:
      - LoRA adapter  → generative inference with chat-style prompt
      - Seq classifier → classification head inference
    """

    def __init__(self, corpus: dict):
        self.corpus         = corpus
        self.model_path     = os.environ.get("TINYLLAMA_MODEL")
        self.max_new_tokens = int(os.environ.get("TINYLLAMA_MAX_NEW_TOKENS"))
        self.max_length     = int(os.environ.get("TINYLLAMA_MAX_LENGTH"))

        self.tokenizer, self.model, self.device, self.mode = self._load_model(self.model_path)

    # ── Model loading ──────────────────────────────────────────────────────────

    def _is_lora(self, path: str) -> bool:
        return (Path(path) / "adapter_config.json").exists()

    def _load_model(self, path: str):
        return self._load_lora(path) if self._is_lora(path) else self._load_classifier(path)

    def _load_lora(self, path: str):
        from peft import PeftModel
        cfg        = json.loads((Path(path) / "adapter_config.json").read_text())
        base_name  = cfg["base_model_name_or_path"]
        print(f"Loading LoRA adapter: {path}")
        print(f"  Base model: {base_name}")

        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base  = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(base, path)
        model.eval().to("cpu")          # MPS hangs on 1.1B forward pass; CPU is reliable
        print("  Loaded on cpu")
        return tokenizer, model, "cpu", "lora"

    def _load_classifier(self, path: str):
        print(f"Loading sequence classifier: {path}")
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval().to("cpu")
        print(f"  Labels: {model.config.id2label}")
        return tokenizer, model, "cpu", "classifier"

    # ── Inference ──────────────────────────────────────────────────────────────

    def _infer_lora(self, premise: str, hypothesis: str) -> str:
        prompt = f"{_SYSTEM_PROMPT}Premise: {premise}\nClaim: {hypothesis}\nLabel:"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        first_word = generated.strip().split()[0].lower().rstrip(".,;:") if generated.strip() else ""
        return _LORA_LABEL_MAP.get(first_word, NEI)

    def _infer_classifier(self, premise: str, hypothesis: str) -> str:
        label_map = {}
        for idx, label in self.model.config.id2label.items():
            upper = label.upper()
            if "ENTAIL"   in upper: label_map[int(idx)] = SUPPORT
            elif "CONTRA" in upper: label_map[int(idx)] = CONTRADICT
            else:                   label_map[int(idx)] = NEI

        enc = self.tokenizer(
            premise, hypothesis, return_tensors="pt",
            truncation=True, max_length=self.max_length
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return label_map.get(logits.argmax(-1).item(), NEI)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_nli(self, records: list) -> list:
        """
        Run NLI on every record. Abstract resolved via oracle cited_doc_ids;
        no BM25 fallback (oracle-only keeps this runner self-contained).
        Returns a list of {id, nli_verdict, nli_note}.
        """
        print(f"\nRunning TinyLlama NLI ({self.mode} mode) on {len(records)} claims...")
        results = []
        for i, r in enumerate(records):
            abstract_text = None
            for cid in r.get("cited_doc_ids", []):
                if cid in self.corpus:
                    abstract_text = " ".join(self.corpus[cid]["abstract"])
                    break

            if not abstract_text:
                print(f"  [{i+1:2d}/{len(records)}] NEI  (no abstract)")
                results.append({"id": r["id"], "nli_verdict": NEI, "nli_note": "NO_ABSTRACT"})
                continue

            premise = abstract_text[:1500]
            verdict = (
                self._infer_lora(premise, r["claim"])
                if self.mode == "lora"
                else self._infer_classifier(premise, r["claim"])
            )
            print(f"  [{i+1:2d}/{len(records)}] {verdict:<12}  {r['claim'][:55]}")
            results.append({"id": r["id"], "nli_verdict": verdict, "nli_note": "ok"})

        return results
