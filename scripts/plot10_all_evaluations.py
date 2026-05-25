"""Generate the Phase 2 full-size evaluation history plot.

The plot is deliberately descriptive rather than a reproducibility claim:
historical archived peaks are shown as logs, while current reproducible
numbers are reported in the README/blog tables.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs" / "phase2"
OUT = ROOT / "docs" / "phase2" / "plots" / "10_all_evaluations.png"


def load_points() -> tuple[list[dict], list[dict]]:
    train = []
    val = []
    for path in sorted(LOGS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            if data.get("n_samples") != 2000:
                continue
            tag = data.get("tag", "")
            if "knn" in tag.lower() or "bear_gr_val" in tag.lower():
                continue
            ts = datetime.strptime(data["timestamp"][:19], "%Y-%m-%d_%H-%M-%S")
            point = {
                "dt": ts,
                "acc": 100 * float(data["top1_accuracy"]),
                "tag": tag,
            }
            if "val" in tag.lower():
                val.append(point)
            else:
                train.append(point)
        except Exception:
            continue
    train.sort(key=lambda p: p["dt"])
    val.sort(key=lambda p: p["dt"])
    return train, val


def best_so_far(points: list[dict]) -> tuple[list[datetime], list[float]]:
    dates = []
    best = []
    current = float("-inf")
    for p in points:
        current = max(current, p["acc"])
        dates.append(p["dt"])
        best.append(current)
    return dates, best


def main() -> None:
    train, val = load_points()
    all_points = train + val
    all_points.sort(key=lambda p: p["dt"])
    if not all_points:
        raise RuntimeError("No full-size Phase 2 eval logs found")

    fig, ax = plt.subplots(figsize=(18, 8.8))

    ax.scatter(
        [p["dt"] for p in train],
        [p["acc"] for p in train],
        s=12,
        alpha=0.35,
        color="#2f73c7",
        label=f"Train/unknown full evals ({len(train)})",
    )
    ax.scatter(
        [p["dt"] for p in val],
        [p["acc"] for p in val],
        s=22,
        alpha=0.75,
        marker="s",
        color="#e85c33",
        label=f"Validation full evals ({len(val)})",
    )

    b_dates, b_accs = best_so_far(train)
    ax.plot(b_dates, b_accs, color="#235aa6", linewidth=2.2, label="Train best-so-far")

    ax.axhline(71.8, color="#777777", linestyle="-.", linewidth=1.2, alpha=0.55)
    ax.text(
        all_points[-1]["dt"],
        72.8,
        "Simple CNN validation reference (71.8%)",
        ha="right",
        va="bottom",
        color="#666666",
        fontsize=9,
    )

    ax.set_title(
        "Phase 2: Full-Size Evaluation History\n"
        "Archived train peak shown from logs; current reproducible results are reported separately",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.96,
        "Non-neural symbolic pipeline\n"
        "No gradient descent or learned embedding model\n"
        "No stored training images at prediction time",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#999999", alpha=0.9),
    )

    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_xlabel("Time (2026)")
    ax.set_ylim(18, 105)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=30, ha="right")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT.relative_to(ROOT)} with {len(train)} train/unknown and {len(val)} val evals")


if __name__ == "__main__":
    main()
