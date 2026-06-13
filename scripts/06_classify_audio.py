#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from veprad_gmm.features import FeatureConfig, extract_features_from_file
from veprad_gmm.model import GMMPhonemeClassifier
from veprad_gmm.plotting import save_phone_timeline


def parse_args():
    ap = argparse.ArgumentParser(description="Klasificiraj novi audio u foneme po kratkotrajnim segmentima.")
    ap.add_argument("--audio_path", required=True)
    ap.add_argument("--model_path", default="models/gmm_mfcc39.joblib")
    ap.add_argument("--out_csv", default="results/classified_segments.csv")
    ap.add_argument("--out_png", default="results/classified_segments.png")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--frame_length_ms", type=float, default=25.0)
    ap.add_argument("--hop_length_ms", type=float, default=10.0)
    ap.add_argument("--n_mfcc", type=int, default=13)
    return ap.parse_args()


def collapse_frames(times: np.ndarray, pred: np.ndarray, scores: np.ndarray, hop_sec: float):
    if len(pred) == 0:
        return []
    segments = []
    start_time = max(0.0, float(times[0] - hop_sec / 2))
    cur = pred[0]
    sc = [float(scores[0])]
    for i in range(1, len(pred)):
        if pred[i] != cur:
            end_time = float(times[i - 1] + hop_sec / 2)
            segments.append((start_time, end_time, str(cur), float(np.mean(sc))))
            start_time = max(0.0, float(times[i] - hop_sec / 2))
            cur = pred[i]
            sc = [float(scores[i])]
        else:
            sc.append(float(scores[i]))
    end_time = float(times[-1] + hop_sec / 2)
    segments.append((start_time, end_time, str(cur), float(np.mean(sc))))
    return segments


def main():
    args = parse_args()
    cfg = FeatureConfig(
        sample_rate=args.sample_rate,
        frame_length_ms=args.frame_length_ms,
        hop_length_ms=args.hop_length_ms,
        n_mfcc=args.n_mfcc,
        use_delta=True,
        use_delta_delta=True,
    )
    clf = GMMPhonemeClassifier.load(args.model_path)
    X, times, duration = extract_features_from_file(args.audio_path, cfg)
    pred, scores = clf.predict(X, return_scores=True)
    segments = collapse_frames(times, pred, scores, hop_sec=cfg.hop_length / cfg.sample_rate)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(segments, columns=["start", "end", "phone", "mean_log_likelihood"]).to_csv(out, index=False)
    print(f"Spremljeni segmenti: {out}")
    if args.out_png:
        save_phone_timeline(segments, args.out_png)
        print(f"Spremljena slika: {args.out_png}")


if __name__ == "__main__":
    main()
