#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(args):
    print("\n$", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    print("=== BRZI TEST: provjera da projekt radi ===")
    print("Koristi mali demo podskup iz ZIP-a, ne treba nista dodatno raspakirati.")
    run([PYTHON, "scripts/01_build_manifest.py",
         "--wav_dir", "data/raw/wav_sm04_demo",
         "--label_dir", "data/labels/lab_sm04_demo",
         "--transcript_dir", "data/transcripts/txt_sm04_demo",
         "--out_csv", "data/manifests/manifest_quick.csv",
         "--split_by", "utterance", "--speaker_regex", "^(sm\\d{2})",
         "--test_size", "0.25", "--val_size", "0",
         "--only_with_labels", "--max_files", "16"])
    run([PYTHON, "scripts/03_extract_features.py",
         "--manifest_csv", "data/manifests/manifest_quick.csv",
         "--out_dir", "data/features_quick",
         "--index_csv", "data/features_quick/features_index.csv",
         "--frame_length_ms", "20", "--hop_length_ms", "8", "--n_mfcc", "13"])
    run([PYTHON, "scripts/04_train_gmm.py",
         "--features_index", "data/features_quick/features_index.csv",
         "--out_model", "models/gmm_mfcc39_QUICK_TEST.joblib",
         "--n_components", "1",
         "--max_frames_per_phone", "300",
         "--min_frames_per_phone", "5",
         "--max_iter", "10", "--n_init", "1"])
    run([PYTHON, "scripts/05_evaluate.py",
         "--features_index", "data/features_quick/features_index.csv",
         "--model_path", "models/gmm_mfcc39_QUICK_TEST.joblib",
         "--split", "test", "--out_dir", "results_QUICK_TEST", "--save_predictions"])
    sample = sorted((ROOT / "data/raw/wav_sm04_demo").glob("*.wav"))[0]
    run([PYTHON, "scripts/06_classify_audio.py",
         "--audio_path", str(sample.relative_to(ROOT)),
         "--model_path", "models/gmm_mfcc39_QUICK_TEST.joblib",
         "--out_csv", "results_QUICK_TEST/classified_segments_sample.csv",
         "--out_png", "results_QUICK_TEST/classified_segments_sample.png",
         "--frame_length_ms", "20", "--hop_length_ms", "8"])
    print("\n✅ BRZI TEST JE GOTOV.")
    print("Ako postoje ove datoteke, projekt radi:")
    print("  models/gmm_mfcc39_QUICK_TEST.joblib")
    print("  results_QUICK_TEST/classification_report_test.txt")
    print("  results_QUICK_TEST/confusion_matrix_test.png")
    print("  results_QUICK_TEST/classified_segments_sample.png")


if __name__ == "__main__":
    main()
