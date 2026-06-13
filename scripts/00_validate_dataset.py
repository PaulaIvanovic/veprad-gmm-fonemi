#!/usr/bin/env python3
from __future__ import annotations

import argparse
import wave
from collections import Counter
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Provjeri koje VEPRAD datoteke imaš nakon raspakiravanja.")
    ap.add_argument("--data_root", default="data")
    return ap.parse_args()


def count_files(root: Path, suffix: str) -> int:
    return len(list(root.rglob(f"*{suffix}"))) if root.exists() else 0


def stems(root: Path, suffix: str) -> set[str]:
    return {p.stem for p in root.rglob(f"*{suffix}")} if root.exists() else set()


def wav_info_one(path: Path):
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getsampwidth(), w.getnchannels(), w.getnframes() / w.getframerate()


def main():
    args = parse_args()
    d = Path(args.data_root)
    paths = {
        "SM04 WAV": d / "raw" / "wav_sm04",
        "SM04 LAB": d / "labels" / "lab_sm04",
        "SM04 TXT": d / "transcripts" / "txt_sm04",
        "audio_m WAV": d / "raw" / "audio_m",
        "audio_z WAV": d / "raw" / "audio_z",
        "text TXT": d / "transcripts" / "text",
    }
    print("=== VEPRAD provjera datoteka ===")
    for name, root in paths.items():
        suffix = ".wav" if "WAV" in name else ".lab" if "LAB" in name else ".txt"
        print(f"{name:12s}: {count_files(root, suffix):6d}  ({root})")

    sm_wav = stems(paths["SM04 WAV"], ".wav")
    sm_lab = stems(paths["SM04 LAB"], ".lab")
    sm_txt = stems(paths["SM04 TXT"], ".txt")
    print("\nSM04 supervised skup:")
    print(f"  WAV+LAB parova: {len(sm_wav & sm_lab)}")
    print(f"  WAV bez LAB:    {len(sm_wav - sm_lab)}")
    print(f"  LAB bez WAV:    {len(sm_lab - sm_wav)}")
    print(f"  WAV+TXT parova: {len(sm_wav & sm_txt)}")

    all_audio_stems = stems(paths["audio_m WAV"], ".wav") | stems(paths["audio_z WAV"], ".wav")
    all_text_stems = stems(paths["text TXT"], ".txt")
    print("\naudio_m/audio_z weak-labeled skup:")
    print(f"  WAV+TXT parova: {len(all_audio_stems & all_text_stems)}")
    print(f"  WAV bez TXT:    {len(all_audio_stems - all_text_stems)}")
    print(f"  TXT bez WAV:    {len(all_text_stems - all_audio_stems)}")

    # Kratka provjera WAV formata.
    bad = 0
    duration = 0.0
    sp_counts = Counter()
    wav_roots = [paths["SM04 WAV"], paths["audio_m WAV"], paths["audio_z WAV"]]
    for root in wav_roots:
        if not root.exists():
            continue
        for wav in root.rglob("*.wav"):
            try:
                sr, sw, ch, dur = wav_info_one(wav)
                duration += dur
                sp_counts[wav.parent.name if wav.parent.name not in {"wav_sm04", "audio_m", "audio_z"} else wav.stem[:4]] += 1
                if sr != 16000 or sw != 2:
                    bad += 1
            except Exception:
                bad += 1
    print(f"\nUkupno trajanje svih pronađenih WAV datoteka: {duration/60:.1f} min")
    print(f"WAV nije 16 kHz/16-bit ili se ne može pročitati: {bad}")
    print("\nBroj datoteka po govorniku/mapi:")
    for sp, n in sorted(sp_counts.items()):
        print(f"  {sp:>8s}: {n}")


if __name__ == "__main__":
    main()
