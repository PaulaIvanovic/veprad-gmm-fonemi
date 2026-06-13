#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def moving_average_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(scores) <= 1:
        return scores
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(scores, ((pad, pad), (0, 0)), mode="edge")
    csum = np.vstack([np.zeros((1, scores.shape[1]), dtype=np.float64), np.cumsum(padded, axis=0, dtype=np.float64)])
    out = (csum[window:window + len(scores)] - csum[:len(scores)]) / float(window)
    return out.astype(np.float32)


def compute_log_priors(features_index: str, labels: list[str], split: str = "train", include_silence: bool = False, smoothing: float = 1.0):
    idx = pd.read_csv(features_index)
    idx = idx[idx["split"].astype(str) == split]
    ignore = set() if include_silence else {v.lower() for v in DEFAULT_IGNORE_LABELS}
    counts = {lab: float(smoothing) for lab in labels}
    for _, row in tqdm(idx.iterrows(), total=len(idx), desc="Brojim priore", leave=False):
        data = np.load(row["feature_path"], allow_pickle=True)
        y = data["y"].astype(str)
        for v in y:
            ph = normalize_label(v)
            if ph.lower() in ignore or ph not in counts:
                continue
            counts[ph] += 1.0
    total = sum(counts.values())
    priors = np.array([counts[lab] / total for lab in labels], dtype=np.float32)
    return np.log(priors + 1e-12).astype(np.float32), counts


def evaluate_once(features_index: str, model_path: str, split: str, prior_weight: float, smooth_window: int,
                  include_silence: bool = False, save_predictions: bool = False, out_dir: str | Path | None = None,
                  priors_from_split: str = "train"):
    clf = GMMPhonemeClassifier.load(model_path)
    labels = sorted(clf.labels)
    label_to_i = {lab: i for i, lab in enumerate(labels)}
    log_priors = None
    if abs(prior_weight) > 1e-12:
        log_priors, _ = compute_log_priors(features_index, labels, split=priors_from_split, include_silence=include_silence)

    idx = pd.read_csv(features_index)
    idx = idx[idx["split"].astype(str) == split]
    if idx.empty:
        raise SystemExit(f"Nema feature datoteka za split={split}")
    ignore = set() if include_silence else {v.lower() for v in DEFAULT_IGNORE_LABELS}

    y_true_all, y_pred_all, pred_rows = [], [], []
    for _, row in tqdm(idx.iterrows(), total=len(idx), desc=f"Evaluacija {split}"):
        data = np.load(row["feature_path"], allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(str)
        times = data["times"].astype(float)
        keep = np.array([(normalize_label(v).lower() not in ignore) and (normalize_label(v) in label_to_i) for v in y], dtype=bool)
        if not keep.any():
            continue
        Xk = X[keep]
        yk = np.array([normalize_label(v) for v in y[keep]], dtype=object)
        scores = clf.score_matrix(Xk)
        if log_priors is not None:
            scores = scores + float(prior_weight) * log_priors[None, :]
        scores = moving_average_scores(scores, smooth_window)
        pred_idx = np.argmax(scores, axis=1)
        pred = np.array([labels[i] for i in pred_idx], dtype=object)
        best_score = scores[np.arange(len(pred_idx)), pred_idx]
        y_true_all.extend(yk.tolist())
        y_pred_all.extend(pred.tolist())
        if save_predictions:
            for t, true, pr, sc in zip(times[keep], yk, pred, best_score):
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
    acc = accuracy_score(y_true_all, y_pred_all)
    err = 1.0 - acc
    report = classification_report(y_true_all, y_pred_all, labels=labels, zero_division=0)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        txt = out_dir / f"classification_report_{split}.txt"
        txt.write_text(
            f"Accuracy: {acc:.6f}\nError rate: {err:.6f}\nPrior weight: {prior_weight}\nSmooth window: {smooth_window}\n\n{report}\n",
            encoding="utf-8",
        )
        save_confusion_matrix(y_true_all, y_pred_all, labels, out_dir / f"confusion_matrix_{split}.png", normalize="true")
        if save_predictions:
            pd.DataFrame(pred_rows).to_csv(out_dir / f"frame_predictions_{split}.csv", index=False)
    return acc, err, report


def main():
    ap = argparse.ArgumentParser(description="Napredna evaluacija: accuracy, error rate, priors, smoothing i grid search na validation skupu.")
    ap.add_argument("--features_index", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", default="results_advanced")
    ap.add_argument("--prior_weight", type=float, default=0.0)
    ap.add_argument("--smooth_window", type=int, default=1)
    ap.add_argument("--grid_search", action="store_true", help="Isprobaj više prior_weight/smooth_window kombinacija na zadanom splitu, najčešće val.")
    ap.add_argument("--prior_grid", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--smooth_grid", default="1,3,5,7,9,11")
    ap.add_argument("--best_params_json", default="")
    ap.add_argument("--include_silence", action="store_true")
    ap.add_argument("--save_predictions", action="store_true")
    args = ap.parse_args()

    if args.best_params_json:
        params = json.loads(Path(args.best_params_json).read_text(encoding="utf-8"))
        args.prior_weight = float(params.get("prior_weight", args.prior_weight))
        args.smooth_window = int(params.get("smooth_window", args.smooth_window))
        print(f"Učitani najbolji postprocess parametri: prior_weight={args.prior_weight}, smooth_window={args.smooth_window}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.grid_search:
        rows = []
        best = None
        for pw in parse_float_list(args.prior_grid):
            for sw in parse_int_list(args.smooth_grid):
                acc, err, _ = evaluate_once(
                    args.features_index, args.model_path, args.split, pw, sw,
                    include_silence=args.include_silence, save_predictions=False, out_dir=None,
                )
                row = {"split": args.split, "prior_weight": pw, "smooth_window": sw, "accuracy": acc, "error_rate": err}
                rows.append(row)
                print(f"VAL/GRID prior={pw:>4} smooth={sw:>2} -> acc={acc:.6f}, err={err:.6f}")
                if best is None or acc > best["accuracy"]:
                    best = row
        pd.DataFrame(rows).sort_values("accuracy", ascending=False).to_csv(out_dir / "validation_grid.csv", index=False)
        Path(out_dir / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(f"Najbolje na validation skupu: {best}")
        print(f"Spremljeno: {out_dir / 'validation_grid.csv'}")
        print(f"Spremljeno: {out_dir / 'best_params.json'}")
        return

    acc, err, report = evaluate_once(
        args.features_index, args.model_path, args.split, args.prior_weight, args.smooth_window,
        include_silence=args.include_silence, save_predictions=args.save_predictions, out_dir=out_dir,
    )
    print(f"Accuracy ({args.split}): {acc:.6f}")
    print(f"Error rate ({args.split}): {err:.6f}")
    print(report)
    print(f"Spremljeno u: {out_dir}")


if __name__ == "__main__":
    main()
