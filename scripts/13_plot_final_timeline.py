import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def choose_utterance(df, utterance_id=None):
    if utterance_id:
        if utterance_id not in set(df["utterance_id"].astype(str)):
            raise SystemExit(f"Utterance ID nije pronaden u CSV-u: {utterance_id}")
        return utterance_id
    counts = df.groupby("utterance_id").size().sort_values(ascending=False)
    return str(counts.index[0])


def majority_smooth(labels, window=5):
    if window <= 1:
        return labels
    half = window // 2
    labels = list(labels)
    out = []
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        vals, cnts = np.unique(labels[lo:hi], return_counts=True)
        out.append(vals[np.argmax(cnts)])
    return out


def make_plot(csv_path, out_png, utterance_id=None, column="pred", smooth_window=1, max_points=900):
    csv_path = Path(csv_path)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {"utterance_id", "time_sec", "true", "pred"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV nema potrebne stupce: {missing}. Nadeni stupci: {list(df.columns)}")

    uid = choose_utterance(df, utterance_id)
    one = df[df["utterance_id"].astype(str) == uid].copy()
    one = one.sort_values("time_sec").reset_index(drop=True)

    labels_all = sorted(set(df["true"].astype(str)) | set(df["pred"].astype(str)))
    label_to_y = {lab: i for i, lab in enumerate(labels_all)}

    pred = one["pred"].astype(str).tolist()
    true = one["true"].astype(str).tolist()
    pred_smooth = majority_smooth(pred, smooth_window)

    # Downsample only for readability if too many frame points
    step = max(1, int(np.ceil(len(one) / max_points)))
    plot_idx = np.arange(0, len(one), step)

    t = one.loc[plot_idx, "time_sec"].to_numpy()
    yp = [label_to_y[pred_smooth[i]] for i in plot_idx]
    yt = [label_to_y[true[i]] for i in plot_idx]

    plt.figure(figsize=(13, 8))

    if column == "pred":
        plt.scatter(t, yp, s=12, marker="s", alpha=0.85)
        title = f"Predviđeni fonemi kroz vrijeme — finalni model ({uid})"
        legend = None
    elif column == "true":
        plt.scatter(t, yt, s=12, marker="s", alpha=0.85)
        title = f"Stvarni fonemi kroz vrijeme — LAB oznake ({uid})"
        legend = None
    else:
        plt.scatter(t, yt, s=12, marker="s", alpha=0.55, label="Stvarni fonem")
        plt.scatter(t, yp, s=10, marker="x", alpha=0.75, label="Predviđeni fonem")
        title = f"Stvarni i predviđeni fonemi kroz vrijeme — finalni model ({uid})"
        legend = True

    plt.yticks(range(len(labels_all)), labels_all)
    plt.xlabel("Vrijeme [s]")
    plt.ylabel("Fonem")
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    if legend:
        plt.legend(loc="upper right")
    plt.savefig(out_png, dpi=180)
    plt.close()

    print(f"Odabrani audio zapis: {uid}")
    print(f"Broj frameova u zapisu: {len(one)}")
    print(f"Prikazan svaki {step}. frame radi citljivosti.")
    print(f"Spremljeno: {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Napravi graf fonemskih predikcija kroz vrijeme iz finalnog frame_predictions_test.csv.")
    ap.add_argument("--csv", default="results_sm04_plus_TRAINVAL_LONG16_OWNVAL/frame_predictions_test.csv")
    ap.add_argument("--out_png", default="results_sm04_plus_TRAINVAL_LONG16_OWNVAL/final_predicted_phonemes_timeline.png")
    ap.add_argument("--utterance_id", default=None)
    ap.add_argument("--column", choices=["pred", "true", "both"], default="pred")
    ap.add_argument("--smooth_window", type=int, default=5)
    ap.add_argument("--max_points", type=int, default=900)
    args = ap.parse_args()

    make_plot(
        csv_path=args.csv,
        out_png=args.out_png,
        utterance_id=args.utterance_id,
        column=args.column,
        smooth_window=args.smooth_window,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
