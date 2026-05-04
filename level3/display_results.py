from collections import Counter

from sklearn.metrics import classification_report, accuracy_score, f1_score
import os
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv()

class DisplayResults:
    def __init__(self):
        pass

    def load_level2_results(self) -> list | None:
        p = Path(os.environ.get("RESULTS_DIR")) / "level2_results.json"
        if p.exists():
            return json.loads(p.read_text())
        return None

    def print_report(self, label: str, gt: list, pred: list):
        print(f"\n{'='*70}")
        print(f"{label}")
        print(f"{'='*70}")
        print(
            classification_report(
                gt, pred, labels=["SUPPORT", "CONTRADICT", "NEI"], zero_division=0
            )
        )

    def evaluate_and_print(self, combined: list) -> None:
        gt = [c["ground_truth"] for c in combined]
        fs_pred = [c["fs_verdict"] for c in combined]
        uq_pred = [c["uqlm_verdict"] for c in combined]
        nli_pred = [c.get("nli_verdict") for c in combined]
        ens_pred = [c["ensemble_verdict"] for c in combined]

        self.print_report("FACTSCORE L3  vs SciFact Ground Truth", gt, fs_pred)
        self.print_report("UQLM L3       vs SciFact Ground Truth", gt, uq_pred)
        if any(v is not None for v in nli_pred):
            self.print_report("NLI L3        vs SciFact Ground Truth", gt, nli_pred)
        self.print_report("ENSEMBLE L3   vs SciFact Ground Truth", gt, ens_pred)

        agree_n = sum(1 for c in combined if c["fs_verdict"] == c["uqlm_verdict"])
        print(
            f"\nFActScore ↔ uqlm agreement: {agree_n}/{len(combined)} ({100*agree_n/len(combined):.1f}%)"
        )

        method_counts = Counter(c.get("ensemble_method", "") for c in combined)
        print(f"Ensemble methods: {dict(method_counts)}")

        # Display side-by-side Level 2 vs Level 3 results
        l2 = self.load_level2_results()
        if l2:
            l2_by_id = {r["id"]: r for r in l2}
            l3_by_id = {r["id"]: r for r in combined}
            ids = [r["id"] for r in combined]

            l2_gt = [l2_by_id[i]["ground_truth"] for i in ids]
            l2_fs = [l2_by_id[i]["fs_verdict"] for i in ids]
            l2_uq = [l2_by_id[i]["uqlm_verdict"] for i in ids]
            l3_fs = [l3_by_id[i]["fs_verdict"] for i in ids]
            l3_uq = [l3_by_id[i]["uqlm_verdict"] for i in ids]
            l3_nli = [l3_by_id[i].get("nli_verdict") for i in ids]
            l3_ens = [l3_by_id[i]["ensemble_verdict"] for i in ids]

            def macro_f1(gt, pred):
                return f1_score(
                    gt,
                    pred,
                    labels=["SUPPORT", "CONTRADICT", "NEI"],
                    average="macro",
                    zero_division=0,
                )

            def acc(gt, pred):
                return accuracy_score(gt, pred)

            print(f"\n{'='*70}")
            print(f"LEVEL 2 vs LEVEL 3 SUMMARY")
            print(f"{'='*70}")
            print(f"{'System':<22} {'Accuracy':>10} {'Macro F1':>10}")
            print(f"{'-'*44}")
            print(
                f"{'L2 FActScore':<22} {acc(l2_gt,l2_fs):>10.1%} {macro_f1(l2_gt,l2_fs):>10.3f}"
            )
            print(
                f"{'L2 uqlm':<22} {acc(l2_gt,l2_uq):>10.1%} {macro_f1(l2_gt,l2_uq):>10.3f}"
            )
            print(f"{'-'*44}")
            print(
                f"{'L3 FActScore':<22} {acc(l2_gt,l3_fs):>10.1%} {macro_f1(l2_gt,l3_fs):>10.3f}"
            )
            print(
                f"{'L3 uqlm':<22} {acc(l2_gt,l3_uq):>10.1%} {macro_f1(l2_gt,l3_uq):>10.3f}"
            )
            if any(v is not None for v in l3_nli):
                print(
                    f"{'L3 NLI':<22} {acc(l2_gt,l3_nli):>10.1%} {macro_f1(l2_gt,l3_nli):>10.3f}"
                )
            print(
                f"{'L3 Ensemble':<22} {acc(l2_gt,l3_ens):>10.1%} {macro_f1(l2_gt,l3_ens):>10.3f}"
            )
            print(f"{'='*70}")

        # Disagreement table
        disagree = [c for c in combined if c["fs_verdict"] != c["uqlm_verdict"]]
        print(f"\nDisagreements ({len(disagree)} cases):")
        print(f"  {'GT':<12} {'FActScore':<13} {'uqlm':<13} {'Ensemble':<12} Claim")
        print("  " + "-" * 72)
        for c in disagree[:12]:
            fs_m = "✓" if c["fs_verdict"] == c["ground_truth"] else "✗"
            uq_m = "✓" if c["uqlm_verdict"] == c["ground_truth"] else "✗"
            en_m = "✓" if c["ensemble_verdict"] == c["ground_truth"] else "✗"
            print(
                f"  {c['ground_truth']:<12} "
                f"{c['fs_verdict']}{fs_m:<12} "
                f"{c['uqlm_verdict']}{uq_m:<12} "
                f"{c['ensemble_verdict']}{en_m:<11} "
                f"{c['claim'][:42]}"
            )
    
    def serial(self, o):
        if isinstance(o, dict): return {k: self.serial(v) for k, v in o.items()}
        if isinstance(o, list): return [self.serial(v) for v in o]
        if hasattr(o, "item"):  return o.item()
        return o

    def save_results(self, combined: list):
        path = Path(os.environ.get("RESULTS_DIR")) / "level3_results.json"
        path.write_text(json.dumps(self.serial(combined), indent=2))
        print(f"\nResults saved → {path}")
