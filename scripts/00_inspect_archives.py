#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Prikaži sadržaj RAR arhiva bez raspakiravanja.")
    ap.add_argument("--archive_dir", default="data/archives")
    return ap.parse_args()


def list_archive(archive: Path) -> list[str]:
    for exe in ["7z", "7zz", "7za"]:
        if shutil.which(exe):
            p = subprocess.run([exe, "l", "-ba", str(archive)], capture_output=True, text=True, errors="ignore")
            # Zadnja kolona je često ime datoteke; fallback ispisuje cijelu liniju.
            names = []
            for line in p.stdout.splitlines():
                parts = line.split()
                if parts and (parts[-1].lower().endswith((".wav", ".lab", ".txt"))):
                    names.append(parts[-1])
            return names
    print("Instaliraj 7-Zip ako želiš automatski pregled arhiva.")
    return []


def main():
    args = parse_args()
    archive_dir = Path(args.archive_dir)
    for archive in sorted(archive_dir.glob("*.rar")) + sorted(archive_dir.glob("*.RAR")):
        names = list_archive(archive)
        print(f"\n{archive.name}: {len(names)} datoteka")
        speakers = {}
        for name in names:
            first = name.split("/")[0] if "/" in name else "(root)"
            speakers[first] = speakers.get(first, 0) + 1
        for sp, n in sorted(speakers.items()):
            print(f"  {sp:>8s}: {n}")
        for name in names[:5]:
            print("    primjer:", name)


if __name__ == "__main__":
    main()
