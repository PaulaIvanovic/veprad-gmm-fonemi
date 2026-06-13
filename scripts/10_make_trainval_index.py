#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Napravi novi feature index u kojem se train i val koriste za finalni trening.")
    ap.add_argument("--features_index", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.features_index)
    before = df["split"].value_counts().to_dict()
    df2 = df.copy()
    df2.loc[df2["split"].astype(str).str.lower().eq("val"), "split"] = "train"
    after = df2["split"].value_counts().to_dict()
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(args.out_csv, index=False)
    print(f"Spremljen train+val index: {args.out_csv}")
    print("Prije:", before)
    print("Poslije:", after)


if __name__ == "__main__":
    main()
