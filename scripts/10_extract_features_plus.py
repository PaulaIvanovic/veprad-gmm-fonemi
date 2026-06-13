#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from veprad_gmm.io_utils import read_manifest, safe_stem
from veprad_gmm.labels import labels_for_frame_centers, read_labels


def parse_args():
    ap = argparse.ArgumentParser(description="Ekstrahiraj proširene MFCC značajke za SM04: MFCC + delta + delta-delta + energija/spektralne značajke.")
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--out_dir", default="data/features_sm04_plus")
    ap.add_argument("--index_csv", default="data/features_sm04_plus/features_index.csv")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--frame_length_ms", type=float, default=20.0)
    ap.add_argument("--hop_length_ms", type=float, default=8.0)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--no_delta", action="store_true")
    ap.add_argument("--no_delta_delta", action="store_true")
    ap.add_argument("--extra_spectral", action="store_true", help="Dodaj RMS/log-energiju, ZCR, centroid, bandwidth, rolloff i flatness.")
    ap.add_argument("--delta_all", action="store_true", help="Ako je postavljeno, delta i delta-delta se računaju nad svim statičkim značajkama, ne samo MFCC.")
    return ap.parse_args()


def _safe_delta(mat: np.ndarray, order: int) -> np.ndarray:
    n_frames = mat.shape[1]
    if n_frames < 3:
        return np.zeros_like(mat)
    width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
    width = max(3, width)
    return librosa.feature.delta(mat, order=order, width=width, mode="nearest")


def _fix_len(mats: list[np.ndarray]) -> list[np.ndarray]:
    n = min(m.shape[1] for m in mats)
    return [m[:, :n] for m in mats]


def extract_plus(path: str | Path, sr: int, frame_ms: float, hop_ms: float, n_mfcc: int,
                 use_delta: bool, use_delta_delta: bool, extra_spectral: bool, delta_all: bool):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Prazan audio zapis: {path}")
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = (y / peak).astype(np.float32)
    win_length = int(round(sr * frame_ms / 1000.0))
    hop_length = int(round(sr * hop_ms / 1000.0))
    if len(y) < win_length:
        y = np.pad(y, (0, win_length - len(y)), mode="constant")
    n_fft = int(2 ** np.ceil(np.log2(max(2, win_length))))

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window="hamming", center=False,
    )
    statics = [mfcc]

    if extra_spectral:
        rms = librosa.feature.rms(y=y, frame_length=win_length, hop_length=hop_length, center=False)
        log_rms = np.log(rms + 1e-8)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=win_length, hop_length=hop_length, center=False)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hamming", center=False)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hamming", center=False)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hamming", center=False)
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hamming", center=False)
        statics.extend([log_rms, zcr, centroid / sr, bandwidth / sr, rolloff / sr, flatness])

    statics = _fix_len(statics)
    static_mat = np.vstack(statics)

    feats = [static_mat]
    base_for_delta = static_mat if delta_all else mfcc[:, :static_mat.shape[1]]
    if use_delta:
        d = _safe_delta(base_for_delta, 1)
        feats.append(d)
    if use_delta_delta:
        dd = _safe_delta(base_for_delta, 2)
        feats.append(dd)

    mats = _fix_len(feats)
    X = np.vstack(mats).T.astype(np.float32)
    times = (np.arange(X.shape[0]) * hop_length + win_length / 2.0) / sr
    duration = float(len(y) / sr)
    return X, times.astype(np.float32), duration


def main():
    args = parse_args()
    df = read_manifest(args.manifest_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="PLUS značajke"):
        label_path = str(row.get("label_path", "") or "")
        if not label_path or not Path(label_path).exists():
            skipped += 1
            continue
        try:
            X, times, duration = extract_plus(
                row["audio_path"], args.sample_rate, args.frame_length_ms, args.hop_length_ms,
                args.n_mfcc, not args.no_delta, not args.no_delta_delta, args.extra_spectral, args.delta_all,
            )
            intervals = read_labels(label_path)
            y = labels_for_frame_centers(times, intervals)
            keep = np.array([v != "" for v in y], dtype=bool)
            X, times, y = X[keep], times[keep], y[keep].astype(str)
            if len(y) == 0:
                skipped += 1
                continue
            split_dir = out_dir / str(row["split"])
            split_dir.mkdir(parents=True, exist_ok=True)
            npz_path = split_dir / f"{safe_stem(row['audio_path'])}.npz"
            np.savez_compressed(
                npz_path,
                X=X.astype(np.float32), y=y, times=times.astype(np.float32),
                audio_path=str(row["audio_path"]), label_path=label_path,
                utterance_id=str(row["utterance_id"]), speaker_id=str(row.get("speaker_id", "")),
                split=str(row["split"]), duration=np.float32(duration),
            )
            rows.append({
                "utterance_id": row["utterance_id"], "speaker_id": row.get("speaker_id", ""),
                "split": row["split"], "audio_path": row["audio_path"], "label_path": label_path,
                "feature_path": str(npz_path), "n_frames": len(y), "duration": duration,
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
