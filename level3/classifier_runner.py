import os
import csv
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

LABELS = ["SUPPORT", "CONTRADICT", "NEI"]

FEATURE_NAMES = [
    "fs_max_sent_score",
    "fs_ratio",
    "fs_gated",
    "uqlm_confidence",
    "uqlm_avg_entailment",
    "uqlm_support_count",
    "uqlm_contradict_count",
    "uqlm_nei_count",
    "uqlm_support_frac",
    "uqlm_contradict_frac",
    "uqlm_nei_frac",
    "fs_num_atoms",
    "fs_supported_atoms",
    "fs_contradicted_atoms",
    "fs_neither_atoms",
    "claim_word_count",
    "claim_char_count",
    "fs_support",
    "fs_contradict",
    "fs_nei",
    "uqlm_support",
    "uqlm_contradict",
    "uqlm_nei",
    "ensemble_support",
    "ensemble_contradict",
    "ensemble_nei",
]

NLI_FEATURE_NAMES = ["nli_support", "nli_contradict", "nli_nei"]


class Classifier:
    def __init__(self):
        self.results_dir = Path(os.environ.get("RESULTS_DIR"))
        self.n_estimators = int(os.environ.get("RF_N_ESTIMATORS"))
        self.max_depth = int(os.environ.get("RF_MAX_DEPTH"))
        self.random_state = int(os.environ.get("RF_RANDOM_STATE"))
        self.cv_n_splits = int(os.environ.get("CV_N_SPLITS"))

    def run(self, records: list, nli_by_id: dict = None) -> list:
        """
        Build feature matrix, train RandomForest with stratified CV,
        add classifier_verdict to each record, return the updated records.
        """
        feature_names = FEATURE_NAMES + (NLI_FEATURE_NAMES if nli_by_id else [])
        X = self._build_feature_matrix(records, nli_by_id)
        y = np.array([LABELS.index(r["ground_truth"]) for r in records], dtype=int)

        clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight="balanced",
                random_state=self.random_state,
            ),
        )

        counts = np.bincount(y)
        n_splits = min(self.cv_n_splits, len(records), int(counts[counts > 0].min()))
        if n_splits >= 2:
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
            y_pred = cross_val_predict(clf, X, y, cv=cv)
            clf.fit(X, y)
        else:
            print("Not enough samples for cross-validation; fitting on all data.")
            clf.fit(X, y)
            y_pred = clf.predict(X)

        method = "extended_score_classifier" + ("_nli" if nli_by_id else "")
        for record, pred, features in zip(records, y_pred, X.tolist()):
            record["classifier_verdict"] = LABELS[pred]
            record["classifier_method"] = method
            record["classifier_features"] = dict(zip(feature_names, features))

        return records

    def print_report(self, label: str, records: list):
        y_true = [r["ground_truth"] for r in records]
        y_pred = [r["classifier_verdict"] for r in records]
        print(f"\n{'='*72}")
        print(label)
        print(f"{'='*72}")
        print(
            classification_report(y_true, y_pred, target_names=LABELS, zero_division=0)
        )
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    def save_results(self, records: list, path: Path = None):
        path = path or (self.results_dir / "level3_classifier_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Saved classifier results → {path}")
        self._save_csv(records, path)

    def _save_csv(self, records: list, json_path: Path):
        csv_path = json_path.with_suffix(".csv")
        has_nli = (
            bool(
                (records[0].get("classifier_features") or {}).get("nli_support")
                is not None
            )
            if records
            else False
        )
        names = FEATURE_NAMES + (NLI_FEATURE_NAMES if has_nli else [])
        fieldnames = ["id", "ground_truth", "classifier_verdict"] + names
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                row = {
                    "id": r.get("id"),
                    "ground_truth": r.get("ground_truth"),
                    "classifier_verdict": r.get("classifier_verdict"),
                }
                for name in names:
                    row[name] = (r.get("classifier_features") or {}).get(name)
                writer.writerow(row)
        print(f"Saved CSV → {csv_path}")

    def _build_feature_matrix(
        self, records: list, nli_by_id: dict = None
    ) -> np.ndarray:
        X = []
        for r in records:
            counts = r.get("uqlm_label_counts") or {}
            total = max(sum(counts.values()), 1)
            n_atoms, sup_a, con_a, nei_a = self._count_atoms(r.get("fs_decisions"))
            fv = r.get("fs_verdict", "NEI")
            uv = r.get("uqlm_verdict", "NEI")
            ev = r.get("ensemble_verdict", "NEI")
            row = [
                float(r.get("fs_max_sent_score") or 0),
                float(r.get("fs_ratio") or 0),
                int(bool(r.get("fs_gated"))),
                float(r.get("uqlm_confidence") or 0),
                float(r.get("uqlm_avg_entailment") or 0.5),
                int(counts.get("SUPPORT", 0)),
                int(counts.get("CONTRADICT", 0)),
                int(counts.get("NEI", 0)),
                counts.get("SUPPORT", 0) / total,
                counts.get("CONTRADICT", 0) / total,
                counts.get("NEI", 0) / total,
                n_atoms,
                sup_a,
                con_a,
                nei_a,
                len(r.get("claim", "").split()),
                len(r.get("claim", "")),
                int(fv == "SUPPORT"),
                int(fv == "CONTRADICT"),
                int(fv == "NEI"),
                int(uv == "SUPPORT"),
                int(uv == "CONTRADICT"),
                int(uv == "NEI"),
                int(ev == "SUPPORT"),
                int(ev == "CONTRADICT"),
                int(ev == "NEI"),
            ]
            if nli_by_id is not None:
                nv = nli_by_id.get(r.get("id"), {}).get("nli_verdict", "NEI")
                row += [int(nv == "SUPPORT"), int(nv == "CONTRADICT"), int(nv == "NEI")]
            X.append(row)
        return np.array(X, dtype=float)

    @staticmethod
    def _count_atoms(decisions):
        d = decisions or []
        sup = sum(1 for x in d if x.get("is_supported") is True)
        con = sum(1 for x in d if x.get("is_supported") is False)
        nei = sum(1 for x in d if x.get("is_supported") is None)
        return len(d), sup, con, nei
