#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from veprad_gmm.labels import DEFAULT_IGNORE_LABELS, normalize_label
from veprad_gmm.model import GMMPhonemeClassifier
from veprad_gmm.plotting import save_confusion_matrix


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluacija GMM fonemskog klasifikatora.")
    ap.add_argument("--features_index", required=True)
    ap.add_argument("--model_path", default="models/gmm_mfcc39.joblib")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--include_silence", action="store_true")
    ap.add_argument("--save_predictions", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clf = GMMPhonemeClassifier.load(args.model_path)
    idx = pd.read_csv(args.features_index)
    idx = idx[idx["split"] == args.split]
    if idx.empty:
        raise SystemExit(f"Nema feature datoteka za split={args.split}")

    ignore = set() if args.include_silence else {v.lower() for v in DEFAULT_IGNORE_LABELS}
    y_true_all = []
    y_pred_all = []
    pred_rows = []

    for _, row in tqdm(idx.iterrows(), total=len(idx), desc="Evaluacija"):
        data = np.load(row["feature_path"], allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(str)
        times = data["times"].astype(float)
        keep = np.array([(normalize_label(v).lower() not in ignore) and (normalize_label(v) in clf.labels) for v in y], dtype=bool)
        if not keep.any():
            continue
        Xk = X[keep]
        yk = np.array([normalize_label(v) for v in y[keep]], dtype=object)
        pred, score = clf.predict(Xk, return_scores=True)
        y_true_all.extend(yk.tolist())
        y_pred_all.extend(pred.tolist())
        if args.save_predictions:
            for t, true, pr, sc in zip(times[keep], yk, pred, score):
                pred_rows.append({
                    "utterance_id": row["utterance_id"],
                    "speaker_id": row.get("speaker_id", ""),
                    "time_sec": t,
                    "true": true,
                    "pred": pr,
                    "score": float(sc),
                })

    if not y_true_all:
        raise SystemExit("Nema evaluacijskih frameova nakon filtriranja.")

    labels = sorted(clf.labels)
    acc = accuracy_score(y_true_all, y_pred_all)
    report = classification_report(y_true_all, y_pred_all, labels=labels, zero_division=0)
    print(f"Accuracy ({args.split}): {acc:.4f}")
    print(report)

    (out_dir / f"classification_report_{args.split}.txt").write_text(
        f"Accuracy: {acc:.6f}\n\n{report}\n", encoding="utf-8"
    )
    save_confusion_matrix(y_true_all, y_pred_all, labels, out_dir / f"confusion_matrix_{args.split}.png", normalize="true")
    print(f"Spremljeno: {out_dir / f'classification_report_{args.split}.txt'}")
    print(f"Spremljeno: {out_dir / f'confusion_matrix_{args.split}.png'}")

    if args.save_predictions:
        pred_csv = out_dir / f"frame_predictions_{args.split}.csv"
        pd.DataFrame(pred_rows).to_csv(pred_csv, index=False)
        print(f"Spremljeno: {pred_csv}")


if __name__ == "__main__":
    main()
