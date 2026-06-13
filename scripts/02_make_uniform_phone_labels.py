#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from veprad_gmm.g2p_hr import load_pronunciation_dict, read_transcript, text_to_phones
from veprad_gmm.io_utils import read_manifest


def parse_args():
    ap = argparse.ArgumentParser(description="Napravi približne uniformne fonemske .lab oznake iz transkripata i VEPRAD_W.DCT rječnika.")
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--dict_path", default="data/dict/VEPRAD_W.DCT.txt")
    ap.add_argument("--out_label_dir", default="data/labels/uniform")
    ap.add_argument("--out_manifest_csv", default="data/manifests/manifest_uniform.csv")
    ap.add_argument("--insert_short_pause_between_words", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    df = read_manifest(args.manifest_csv)
    pron = load_pronunciation_dict(args.dict_path)
    print(f"Učitano riječi iz rječnika: {len(pron)} ({args.dict_path})")
    out_dir = Path(args.out_label_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    skipped = 0
    unknown_tokens_total = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Uniform labels"):
        transcript_path = str(row.get("transcript_path", "") or "")
        if not transcript_path or not Path(transcript_path).exists():
            skipped += 1
            continue
        text = read_transcript(transcript_path)
        # Za statistiku: koliko riječi nije u rječniku.
        toks = text.lower().split()
        unknown_tokens_total += sum(1 for t in toks if t not in pron and not (t.startswith("<") and t.endswith(">")))
        phones = text_to_phones(text, pronunciation_dict=pron, insert_short_pause_between_words=args.insert_short_pause_between_words)
        if not phones:
            skipped += 1
            continue
        duration = float(librosa.get_duration(path=row["audio_path"]))
        step = duration / len(phones)
        lab_path = out_dir / f"{Path(row['audio_path']).stem}.lab"
        with open(lab_path, "w", encoding="utf-8") as f:
            for k, ph in enumerate(phones):
                start = k * step
                end = duration if k == len(phones) - 1 else (k + 1) * step
                f.write(f"{start:.6f} {end:.6f} {ph}\n")
        df.loc[i, "label_path"] = str(lab_path)
        made += 1

    out_manifest = Path(args.out_manifest_csv)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)
    print(f"Uniformne oznake napravljene za {made} datoteka; preskočeno {skipped}.")
    print(f"Novi manifest: {out_manifest}")
    print(f"Nepoznati tokeni za rječnik, fallback G2P korišten: {unknown_tokens_total}")
    print("NAPOMENA: uniformne oznake su približne. Za glavnu ocjenu koristi realne .lab oznake ili forced alignment.")


if __name__ == "__main__":
    main()
