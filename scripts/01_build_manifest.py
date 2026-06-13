#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from veprad_gmm.io_utils import AUDIO_EXTS, LABEL_EXTS, TEXT_EXTS, find_by_stem, infer_speaker_id


def parse_args():
    ap = argparse.ArgumentParser(description="Izgradi manifest za VEPRAD wav + label/transkript datoteke.")
    ap.add_argument("--wav_dir", nargs="+", required=True, help="Jedna ili više mapa s WAV datotekama")
    ap.add_argument("--label_dir", nargs="*", default=[], help="Jedna ili više mapa s .lab/.TextGrid oznakama")
    ap.add_argument("--transcript_dir", nargs="*", default=[], help="Jedna ili više mapa s .txt transkriptima")
    ap.add_argument("--out_csv", default="data/manifests/manifest.csv")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1, help="Udio cijelog skupa za validaciju; 0 za bez validacije")
    ap.add_argument("--split_by", choices=["speaker", "utterance"], default="speaker")
    ap.add_argument("--speaker_regex", default=None, help="Regex za speaker_id iz imena datoteke; prva grupa se koristi ako postoji")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only_with_labels", action="store_true", help="Zadrži samo WAV koji imaju label_path")
    ap.add_argument("--only_with_transcripts", action="store_true", help="Zadrži samo WAV koji imaju transcript_path")
    ap.add_argument("--max_files", type=int, default=0, help="Za brzi test; 0 znači bez limita")
    return ap.parse_args()


def assign_splits(df: pd.DataFrame, split_by: str, test_size: float, val_size: float, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["split"] = "train"
    if len(df) < 3:
        return df

    if split_by == "speaker" and df["speaker_id"].nunique() > 2:
        units = df["speaker_id"].drop_duplicates().to_numpy()
        train_units, test_units = train_test_split(units, test_size=test_size, random_state=seed)
        if val_size > 0 and len(train_units) > 2:
            val_fraction_of_train = val_size / max(1e-9, (1.0 - test_size))
            train_units, val_units = train_test_split(train_units, test_size=val_fraction_of_train, random_state=seed)
        else:
            val_units = []
        df.loc[df["speaker_id"].isin(test_units), "split"] = "test"
        df.loc[df["speaker_id"].isin(val_units), "split"] = "val"
    else:
        idx = df.index.to_numpy()
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)
        if val_size > 0 and len(train_idx) > 2:
            val_fraction_of_train = val_size / max(1e-9, (1.0 - test_size))
            train_idx, val_idx = train_test_split(train_idx, test_size=val_fraction_of_train, random_state=seed)
        else:
            val_idx = []
        df.loc[test_idx, "split"] = "test"
        df.loc[val_idx, "split"] = "val"
    return df


def main():
    args = parse_args()
    wav_dirs = [Path(p) for p in args.wav_dir]
    missing = [str(p) for p in wav_dirs if not p.exists()]
    if missing:
        raise SystemExit(f"Ne postoje --wav_dir mape: {missing}")

    audio_files = []
    for wav_root in wav_dirs:
        audio_files.extend(sorted([p for p in wav_root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]))
    if not audio_files:
        raise SystemExit(f"Nema audio datoteka u: {wav_dirs}")
    if args.max_files and args.max_files > 0:
        audio_files = audio_files[: args.max_files]

    rows = []
    for audio in audio_files:
        lab = find_by_stem(args.label_dir, audio.stem, LABEL_EXTS)
        trn = find_by_stem(args.transcript_dir, audio.stem, TEXT_EXTS)
        if args.only_with_labels and not lab:
            continue
        if args.only_with_transcripts and not trn:
            continue
        rows.append({
            "utterance_id": audio.stem,
            "speaker_id": infer_speaker_id(audio, speaker_regex=args.speaker_regex),
            "audio_path": str(audio),
            "label_path": lab,
            "transcript_path": trn,
        })
    if not rows:
        raise SystemExit("Nema datoteka nakon filtriranja. Provjeri label_dir/transcript_dir.")

    df = pd.DataFrame(rows).drop_duplicates(subset=["audio_path"])
    df = assign_splits(df, args.split_by, args.test_size, args.val_size, args.seed)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Spremljen manifest: {out}")
    print(df["split"].value_counts().to_string())
    print(f"Audio datoteka: {len(df)}")
    print(f"S labelama: {(df['label_path'].astype(str) != '').sum()}")
    print(f"S transkriptima: {(df['transcript_path'].astype(str) != '').sum()}")
    print("Govornici:", ", ".join(sorted(df["speaker_id"].unique())[:30]))


if __name__ == "__main__":
    main()
