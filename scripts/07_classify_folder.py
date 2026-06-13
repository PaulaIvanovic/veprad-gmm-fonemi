#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def parse_args():
    ap = argparse.ArgumentParser(description="Klasificiraj sve WAV datoteke u mapi istreniranim GMM modelom.")
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--model_path", default="models/gmm_mfcc39_sm04.joblib")
    ap.add_argument("--out_dir", default="results_classified_folder")
    ap.add_argument("--max_files", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    wavs = sorted(Path(args.wav_dir).rglob("*.wav"))
    if args.max_files and args.max_files > 0:
        wavs = wavs[: args.max_files]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not wavs:
        raise SystemExit(f"Nema WAV datoteka u {args.wav_dir}")
    for wav in wavs:
        csv = out / f"{wav.stem}_segments.csv"
        png = out / f"{wav.stem}_segments.png"
        cmd = [PYTHON, "scripts/06_classify_audio.py", "--audio_path", str(wav),
               "--model_path", args.model_path, "--out_csv", str(csv), "--out_png", str(png),
               "--frame_length_ms", "20", "--hop_length_ms", "8"]
        print("$", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"Gotovo. Rezultati su u {out}")


if __name__ == "__main__":
    main()
