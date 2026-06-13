#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def parse_args():
    ap = argparse.ArgumentParser(description="Pipeline koji koristi audio_m/audio_z + text transkripte i približne uniformne fonemske oznake.")
    ap.add_argument("--max_files", type=int, default=0, help="0 = sve; za brzi test stavi npr. 200")
    ap.add_argument("--n_components", type=int, default=8)
    ap.add_argument("--max_frames_per_phone", type=int, default=30000)
    return ap.parse_args()


def run(args):
    print("\n$", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    args = parse_args()
    run([PYTHON, "scripts/00_validate_dataset.py"])
    cmd = [PYTHON, "scripts/01_build_manifest.py",
           "--wav_dir", "data/raw/audio_m", "data/raw/audio_z",
           "--transcript_dir", "data/transcripts/text",
           "--out_csv", "data/manifests/manifest_audio_mz_text.csv",
           "--split_by", "speaker", "--test_size", "0.2", "--val_size", "0.1",
           "--only_with_transcripts"]
    if args.max_files and args.max_files > 0:
        cmd += ["--max_files", str(args.max_files)]
    run(cmd)
    run([PYTHON, "scripts/02_make_uniform_phone_labels.py",
         "--manifest_csv", "data/manifests/manifest_audio_mz_text.csv",
         "--dict_path", "data/dict/VEPRAD_W.DCT.txt",
         "--out_label_dir", "data/labels/uniform_audio_mz",
         "--out_manifest_csv", "data/manifests/manifest_audio_mz_uniform.csv"])
    run([PYTHON, "scripts/03_extract_features.py",
         "--manifest_csv", "data/manifests/manifest_audio_mz_uniform.csv",
         "--out_dir", "data/features_audio_mz_weak",
         "--index_csv", "data/features_audio_mz_weak/features_index.csv",
         "--frame_length_ms", "20", "--hop_length_ms", "8", "--n_mfcc", "13"])
    run([PYTHON, "scripts/04_train_gmm.py",
         "--features_index", "data/features_audio_mz_weak/features_index.csv",
         "--out_model", "models/gmm_mfcc39_audio_mz_weak.joblib",
         "--n_components", str(args.n_components),
         "--max_frames_per_phone", str(args.max_frames_per_phone),
         "--min_frames_per_phone", "50", "--max_iter", "200", "--n_init", "2"])
    run([PYTHON, "scripts/05_evaluate.py",
         "--features_index", "data/features_audio_mz_weak/features_index.csv",
         "--model_path", "models/gmm_mfcc39_audio_mz_weak.joblib",
         "--split", "test", "--out_dir", "results_audio_mz_weak", "--save_predictions"])
    print("\nAudio_m/audio_z weak pipeline gotov.")
    print("NAPOMENA: rezultati su na približnim uniformnim oznakama, ne na ručno/forced-alignment fonemskim granicama.")


if __name__ == "__main__":
    main()
