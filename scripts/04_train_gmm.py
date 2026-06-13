#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from veprad_gmm.labels import DEFAULT_IGNORE_LABELS, normalize_label
from veprad_gmm.model import GMMTrainingConfig, train_gmm_models


def parse_args():
    ap = argparse.ArgumentParser(description="Treniraj jedan GMM po fonemu.")
    ap.add_argument("--features_index", required=True)
    ap.add_argument("--out_model", default="models/gmm_mfcc39.joblib")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--n_components", type=int, default=8)
    ap.add_argument("--covariance_type", choices=["diag", "full", "tied", "spherical"], default="diag")
    ap.add_argument("--max_frames_per_phone", type=int, default=30000, help="Ograničenje memorije i trajanja treninga po fonemu")
    ap.add_argument("--min_frames_per_phone", type=int, default=50)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--n_init", type=int, default=2)
    ap.add_argument("--reg_covar", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_silence", action="store_true", help="Ako se postavi, ne izbacuje silence/noise oznake")
    return ap.parse_args()


def compact_bucket(buckets, counts, phone, limit, rng):
    if not limit or counts[phone] <= limit * 2:
        return
    merged = np.vstack(buckets[phone]).astype(np.float32)
    idx = rng.choice(len(merged), size=limit, replace=False)
    buckets[phone] = [merged[idx]]
    counts[phone] = limit


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    idx = pd.read_csv(args.features_index)
    idx = idx[idx["split"] == args.train_split]
    if idx.empty:
        raise SystemExit(f"Nema feature datoteka za split={args.train_split}")

    ignore = set() if args.include_silence else {v.lower() for v in DEFAULT_IGNORE_LABELS}
    buckets = defaultdict(list)
    counts = defaultdict(int)

    for _, row in tqdm(idx.iterrows(), total=len(idx), desc="Učitavam train feature"):
        data = np.load(row["feature_path"], allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(str)
        for ph in np.unique(y):
            ph_norm = normalize_label(ph)
            if ph_norm.lower() in ignore:
                continue
            mask = y == ph
            if not mask.any():
                continue
            part = X[mask]
            buckets[ph_norm].append(part)
            counts[ph_norm] += len(part)
            compact_bucket(buckets, counts, ph_norm, args.max_frames_per_phone, rng)

    phone_to_features = {}
    for ph, parts in buckets.items():
        if not parts:
            continue
        X = np.vstack(parts).astype(np.float32)
        if args.max_frames_per_phone and len(X) > args.max_frames_per_phone:
            sel = rng.choice(len(X), size=args.max_frames_per_phone, replace=False)
            X = X[sel]
        phone_to_features[ph] = X

    print(f"Broj fonema za trening: {len(phone_to_features)}")
    for ph, X in sorted(phone_to_features.items()):
        print(f"  {ph:>8s}: {len(X)} frameova")

    cfg = GMMTrainingConfig(
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        reg_covar=args.reg_covar,
        max_iter=args.max_iter,
        n_init=args.n_init,
        random_state=args.seed,
        min_frames_per_phone=args.min_frames_per_phone,
    )
    clf = train_gmm_models(phone_to_features, cfg, max_frames_per_phone=args.max_frames_per_phone)
    clf.save(args.out_model)
    print(f"Model spremljen: {args.out_model}")


if __name__ == "__main__":
    main()
