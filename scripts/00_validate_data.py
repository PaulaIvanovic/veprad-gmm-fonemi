#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import wave
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from veprad_gmm.labels import read_labels, DEFAULT_IGNORE_LABELS


def parse_args():
    ap = argparse.ArgumentParser(description="Provjeri VEPRAD wav/lab/txt datoteke prije treninga.")
    ap.add_argument("--wav_dir", default="data/raw/wav_sm04_demo")
    ap.add_argument("--label_dir", default="data/labels/lab_sm04_demo")
    ap.add_argument("--transcript_dir", default="data/transcripts/txt_sm04_demo")
    ap.add_argument("--max_examples", type=int, default=5)
    return ap.parse_args()


def wav_info(path: Path):
    with wave.open(str(path), "rb") as w:
        return {
            "channels": w.getnchannels(),
            "sample_width_bytes": w.getsampwidth(),
            "sample_rate": w.getframerate(),
            "frames": w.getnframes(),
            "duration": w.getnframes() / w.getframerate(),
        }


def main():
    args = parse_args()
    wav_dir = Path(args.wav_dir)
    label_dir = Path(args.label_dir)
    transcript_dir = Path(args.transcript_dir)

    wavs = sorted(wav_dir.rglob("*.wav"))
    labs = sorted(label_dir.rglob("*.lab"))
    txts = sorted(transcript_dir.rglob("*.txt")) if transcript_dir.exists() else []

    wav_stems = {p.stem for p in wavs}
    lab_stems = {p.stem for p in labs}
    txt_stems = {p.stem for p in txts}
    matched = sorted(wav_stems & lab_stems)

    print("=== VEPRAD provjera ===")
    print(f"WAV datoteka:      {len(wavs)}")
    print(f"LAB datoteka:      {len(labs)}")
    print(f"TXT transkripata:  {len(txts)}")
    print(f"WAV+LAB parova:    {len(matched)}")
    print(f"WAV bez LAB:       {len(wav_stems - lab_stems)}")
    print(f"LAB bez WAV:       {len(lab_stems - wav_stems)}")
    if txts:
        print(f"WAV bez TXT:       {len(wav_stems - txt_stems)}")

    label_counts = Counter()
    duration_mismatch = 0
    total_duration = 0.0
    bad_wav = 0
    for stem in matched:
        wp = next(wav_dir.rglob(f"{stem}.wav"))
        lp = next(label_dir.rglob(f"{stem}.lab"))
        try:
            wi = wav_info(wp)
            total_duration += wi["duration"]
            if wi["sample_rate"] != 16000 or wi["sample_width_bytes"] != 2:
                bad_wav += 1
            df = read_labels(lp)
            if not df.empty:
                for lab in df["label"].astype(str):
                    label_counts[lab] += 1
                lab_end = float(df["end"].max())
                if abs(lab_end - wi["duration"]) > 0.10:
                    duration_mismatch += 1
        except Exception as e:
            print(f"[WARN] problem s {stem}: {e}")

    print(f"Ukupno trajanje WAV+LAB: {total_duration/60:.2f} min")
    print(f"WAV nije 16 kHz / 16-bit: {bad_wav}")
    print(f"LAB/WAV trajanje odstupa >0.10s: {duration_mismatch}")
    print("\nNajčešće oznake u LAB:")
    for lab, n in label_counts.most_common(20):
        tag = "(ignorira se)" if lab.lower() in {x.lower() for x in DEFAULT_IGNORE_LABELS} else ""
        print(f"  {lab:>8s}: {n:6d} {tag}")

    print("\nPrimjeri:")
    for stem in matched[: args.max_examples]:
        print(f"  {stem}.wav + {stem}.lab")


if __name__ == "__main__":
    main()
