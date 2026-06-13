#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from veprad_gmm.features import FeatureConfig, extract_features_from_file
from veprad_gmm.io_utils import read_manifest, safe_stem
from veprad_gmm.labels import labels_for_frame_centers, read_labels


def parse_args():
    ap = argparse.ArgumentParser(description="Ekstrahiraj MFCC(+delta+delta-delta) i poravnaj s fonemskim oznakama.")
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--out_dir", default="data/features")
    ap.add_argument("--index_csv", default="data/features/features_index.csv")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--frame_length_ms", type=float, default=25.0)
    ap.add_argument("--hop_length_ms", type=float, default=10.0)
    ap.add_argument("--n_mfcc", type=int, default=13)
    ap.add_argument("--no_delta", action="store_true")
    ap.add_argument("--no_delta_delta", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    df = read_manifest(args.manifest_csv)
    cfg = FeatureConfig(
        sample_rate=args.sample_rate,
        frame_length_ms=args.frame_length_ms,
        hop_length_ms=args.hop_length_ms,
        n_mfcc=args.n_mfcc,
        use_delta=not args.no_delta,
        use_delta_delta=not args.no_delta_delta,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        label_path = str(row.get("label_path", "") or "")
        if not label_path or not Path(label_path).exists():
            skipped += 1
            continue
        try:
            X, times, duration = extract_features_from_file(row["audio_path"], cfg)
            intervals = read_labels(label_path)
            y = labels_for_frame_centers(times, intervals)
            keep = np.array([v != "" for v in y], dtype=bool)
            X = X[keep]
            times = times[keep]
            y = y[keep].astype(str)
            if len(y) == 0:
                skipped += 1
                continue
            split_dir = out_dir / str(row["split"])
            split_dir.mkdir(parents=True, exist_ok=True)
            npz_path = split_dir / f"{safe_stem(row['audio_path'])}.npz"
            np.savez_compressed(
                npz_path,
                X=X.astype(np.float32),
                y=y,
                times=times.astype(np.float32),
                audio_path=str(row["audio_path"]),
                label_path=label_path,
                utterance_id=str(row["utterance_id"]),
                speaker_id=str(row.get("speaker_id", "")),
                split=str(row["split"]),
                duration=np.float32(duration),
            )
            rows.append({
                "utterance_id": row["utterance_id"],
                "speaker_id": row.get("speaker_id", ""),
                "split": row["split"],
                "audio_path": row["audio_path"],
                "label_path": label_path,
                "feature_path": str(npz_path),
                "n_frames": len(y),
                "duration": duration,
            })
        except Exception as e:
            skipped += 1
            print(f"[WARN] preskačem {row['audio_path']}: {e}")

    index = pd.DataFrame(rows)
    out_index = Path(args.index_csv)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(out_index, index=False)
    print(f"Spremljen feature index: {out_index}")
    print(f"Uspješno: {len(index)}; preskočeno: {skipped}")
    if not index.empty:
        print(index.groupby("split")["n_frames"].sum().to_string())
        print(f"Dimenzija značajki: {np.load(index.iloc[0]['feature_path'])['X'].shape[1]}")


if __name__ == "__main__":
    main()
