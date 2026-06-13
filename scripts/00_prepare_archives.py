#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser(description="Raspakiraj VEPRAD RAR arhive u ispravnu strukturu mapa.")
    ap.add_argument("--archive_dir", default="data/archives")
    ap.add_argument("--out_root", default="data")
    return ap.parse_args()


def run(cmd):
    print("$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def extract_with_available_tool(archive: Path, out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Windows/Linux/macOS: najčešće radi 7-Zip.
    for exe in ["7z", "7zz", "7za"]:
        if shutil.which(exe):
            run([exe, "x", "-y", f"-o{out_dir}", str(archive)])
            return True
    # Linux alternativa.
    for exe in ["unrar", "unrar-free"]:
        if shutil.which(exe):
            run([exe, "x", "-o+", str(archive), str(out_dir)])
            return True
    return False


def find_archive(archive_dir: Path, keywords: list[str]) -> Path | None:
    files = list(archive_dir.glob("*.rar")) + list(archive_dir.glob("*.RAR"))
    for kw in keywords:
        for f in files:
            if kw.lower() in f.name.lower():
                return f
    return None


def main():
    args = parse_args()
    archive_dir = Path(args.archive_dir)
    out_root = Path(args.out_root)
    if not archive_dir.exists():
        raise SystemExit(f"Ne postoji mapa s arhivama: {archive_dir}")

    mapping = [
        (["wav_sm04"], out_root / "raw" / "wav_sm04", "SM04 WAV audio s pravim LAB oznakama"),
        (["lab_sm04"], out_root / "labels" / "lab_sm04", "SM04 fonemske LAB oznake"),
        (["txt_sm04"], out_root / "transcripts" / "txt_sm04", "SM04 TXT transkripti"),
        (["audio_m"], out_root / "raw" / "audio_m", "muški WAV govornici m01-m11"),
        (["audio_z"], out_root / "raw" / "audio_z", "ženski WAV govornici z01-z14"),
        (["text"], out_root / "transcripts" / "text", "TXT transkripti za audio_m/audio_z"),
    ]

    print("Tražim arhive u:", archive_dir)
    missing = []
    not_extracted = []
    for keywords, out_dir, desc in mapping:
        archive = find_archive(archive_dir, keywords)
        if archive is None:
            missing.append((keywords[0], out_dir, desc))
            continue
        print(f"\nRaspakiravam {archive.name} -> {out_dir}  ({desc})")
        if not extract_with_available_tool(archive, out_dir):
            not_extracted.append((archive, out_dir))

    print("\n=== Sažetak ===")
    if missing:
        print("Nisam našao ove arhive; to nije nužno problem ako radiš samo dio projekta:")
        for kw, out_dir, desc in missing:
            print(f"  - *{kw}*.rar -> {out_dir}  ({desc})")
    if not_extracted:
        print("\nNije pronađen 7-Zip/unrar. Ručno raspakiraj ovako:")
        for archive, out_dir in not_extracted:
            print(f"  {archive.name} -> {out_dir}")
        raise SystemExit("Instaliraj 7-Zip ili ručno raspakiraj arhive pa ponovno pokreni pipeline.")

    print("\nGotovo. Sljedeće: python scripts/00_validate_dataset.py")


if __name__ == "__main__":
    main()
