#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS = {
    "QUICK TEST": [
        "models/gmm_mfcc39_QUICK_TEST.joblib",
        "results_QUICK_TEST/classification_report_test.txt",
        "results_QUICK_TEST/confusion_matrix_test.png",
        "results_QUICK_TEST/classified_segments_sample.png",
    ],
    "SM04 SUPERVISED": [
        "models/gmm_mfcc39_sm04.joblib",
        "results_sm04/classification_report_test.txt",
        "results_sm04/confusion_matrix_test.png",
        "results_sm04/frame_predictions_test.csv",
    ],
    "AUDIO_MZ WEAK": [
        "models/gmm_mfcc39_audio_mz_weak.joblib",
        "results_audio_mz_weak/classification_report_test.txt",
        "results_audio_mz_weak/confusion_matrix_test.png",
    ],
}

print("=== PROVJERA REZULTATA ===")
for name, files in CHECKS.items():
    print(f"\n{name}")
    ok = True
    for f in files:
        p = ROOT / f
        exists = p.exists()
        ok = ok and exists
        print(("  OK   " if exists else "  FALI ") + f)
    print("  STATUS:", "GOTOVO" if ok else "NIJE GOTOVO")

for report in [ROOT / "results_sm04/classification_report_test.txt", ROOT / "results_QUICK_TEST/classification_report_test.txt", ROOT / "results_audio_mz_weak/classification_report_test.txt"]:
    if report.exists():
        print(f"\n--- {report.relative_to(ROOT)} ---")
        text = report.read_text(encoding="utf-8", errors="replace")
        print(text[:1500])
