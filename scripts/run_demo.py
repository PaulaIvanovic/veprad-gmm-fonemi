#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(args):
    print("\n$", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    run([PYTHON, "scripts/00_validate_data.py", "--wav_dir", "data/raw/wav_sm04_demo", "--label_dir", "data/labels/lab_sm04_demo", "--transcript_dir", "data/transcripts/txt_sm04_demo"])
    run([PYTHON, "scripts/01_build_manifest.py", "--wav_dir", "data/raw/wav_sm04_demo", "--label_dir", "data/labels/lab_sm04_demo", "--transcript_dir", "data/transcripts/txt_sm04_demo", "--out_csv", "data/manifests/manifest_demo.csv", "--split_by", "utterance", "--speaker_regex", "^(sm\\d{2})", "--test_size", "0.2", "--val_size", "0.1"])
    run([PYTHON, "scripts/03_extract_features.py", "--manifest_csv", "data/manifests/manifest_demo.csv", "--out_dir", "data/features_demo", "--index_csv", "data/features_demo/features_index.csv", "--frame_length_ms", "20", "--hop_length_ms", "8", "--n_mfcc", "13"])
    run([PYTHON, "scripts/04_train_gmm.py", "--features_index", "data/features_demo/features_index.csv", "--out_model", "models/gmm_mfcc39_demo.joblib", "--n_components", "2", "--max_frames_per_phone", "1000", "--min_frames_per_phone", "20", "--max_iter", "20", "--n_init", "1"])
    run([PYTHON, "scripts/05_evaluate.py", "--features_index", "data/features_demo/features_index.csv", "--model_path", "models/gmm_mfcc39_demo.joblib", "--split", "test", "--out_dir", "results_demo", "--save_predictions"])
    sample = sorted((ROOT / "data/raw/wav_sm04_demo").glob("*.wav"))[0]
    run([PYTHON, "scripts/06_classify_audio.py", "--audio_path", str(sample.relative_to(ROOT)), "--model_path", "models/gmm_mfcc39_demo.joblib", "--out_csv", "results_demo/classified_segments_sample.csv", "--out_png", "results_demo/classified_segments_sample.png", "--frame_length_ms", "20", "--hop_length_ms", "8"])
    print("\nDEMO gotov. Pogledaj mapu results_demo/ i model models/gmm_mfcc39_demo.joblib")


if __name__ == "__main__":
    main()
