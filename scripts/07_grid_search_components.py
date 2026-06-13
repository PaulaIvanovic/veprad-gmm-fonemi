#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Jednostavna usporedba broja GMM komponenti.")
    ap.add_argument("--features_index", required=True)
    ap.add_argument("--components", default="2,4,8,16")
    ap.add_argument("--out_dir", default="results/grid_components")
    ap.add_argument("--covariance_type", default="diag")
    ap.add_argument("--max_frames_per_phone", type=int, default=20000)
    return ap.parse_args()


def read_accuracy(report_path: Path) -> float:
    text = report_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("Accuracy:"):
            return float(line.split(":", 1)[1].strip())
    return float("nan")


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in [int(x.strip()) for x in args.components.split(",") if x.strip()]:
        model = out_dir / f"gmm_{c}.joblib"
        res = out_dir / f"eval_{c}"
        print(f"\n=== n_components={c} ===")
        subprocess.check_call([
            sys.executable, str(root / "scripts" / "04_train_gmm.py"),
            "--features_index", args.features_index,
            "--out_model", str(model),
            "--n_components", str(c),
            "--covariance_type", args.covariance_type,
            "--max_frames_per_phone", str(args.max_frames_per_phone),
        ])
        subprocess.check_call([
            sys.executable, str(root / "scripts" / "05_evaluate.py"),
            "--features_index", args.features_index,
            "--model_path", str(model),
            "--out_dir", str(res),
            "--split", "test",
        ])
        acc = read_accuracy(res / "classification_report_test.txt")
        rows.append({"n_components": c, "accuracy": acc, "model": str(model), "result_dir": str(res)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    print(df.to_string(index=False))
    print(f"Sažetak: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
