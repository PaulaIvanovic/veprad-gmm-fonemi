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
    # Glavni, najispravniji pipeline: koristi prave .lab oznake iz lab_sm04.
    run([PYTHON, "scripts/00_validate_dataset.py"])
    run([PYTHON, "scripts/01_build_manifest.py",
         "--wav_dir", "data/raw/wav_sm04",
         "--label_dir", "data/labels/lab_sm04",
         "--transcript_dir", "data/transcripts/txt_sm04",
         "--out_csv", "data/manifests/manifest_sm04.csv",
         "--split_by", "utterance", "--speaker_regex", "^(sm\\d{2})",
         "--test_size", "0.2", "--val_size", "0.1", "--only_with_labels"])
    run([PYTHON, "scripts/03_extract_features.py",
         "--manifest_csv", "data/manifests/manifest_sm04.csv",
         "--out_dir", "data/features_sm04",
         "--index_csv", "data/features_sm04/features_index.csv",
         "--frame_length_ms", "20", "--hop_length_ms", "8", "--n_mfcc", "13"])
    run([PYTHON, "scripts/04_train_gmm.py",
         "--features_index", "data/features_sm04/features_index.csv",
         "--out_model", "models/gmm_mfcc39_sm04.joblib",
         "--n_components", "8", "--max_frames_per_phone", "30000",
         "--min_frames_per_phone", "50", "--max_iter", "200", "--n_init", "2"])
    run([PYTHON, "scripts/05_evaluate.py",
         "--features_index", "data/features_sm04/features_index.csv",
         "--model_path", "models/gmm_mfcc39_sm04.joblib",
         "--split", "test", "--out_dir", "results_sm04", "--save_predictions"])
    samples = sorted((ROOT / "data/raw/wav_sm04").glob("*.wav"))
    if samples:
        sample = samples[0]
        run([PYTHON, "scripts/06_classify_audio.py",
             "--audio_path", str(sample.relative_to(ROOT)),
             "--model_path", "models/gmm_mfcc39_sm04.joblib",
             "--out_csv", "results_sm04/classified_segments_sample.csv",
             "--out_png", "results_sm04/classified_segments_sample.png",
             "--frame_length_ms", "20", "--hop_length_ms", "8"])
    print("\nSM04 supervised pipeline gotov. Pogledaj: models/gmm_mfcc39_sm04.joblib i results_sm04/")


if __name__ == "__main__":
    main()
