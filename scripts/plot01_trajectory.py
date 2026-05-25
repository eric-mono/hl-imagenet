#!/usr/bin/env python3
"""Plot 01: Phase 2 accuracy trajectory with datetime x-axis and reflection annotations."""

import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

LOGS_DIR = Path(__file__).parent.parent / "logs" / "phase2"
PLOTS_DIR = Path(__file__).parent.parent / "docs" / "phase2" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_phase2_evals():
    train_evals = []
    val_evals = []
    for f in sorted(LOGS_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            if d.get('n_samples', 0) != 2000:
                continue
            tag = d.get('tag', '')
            if 'knn' in tag.lower():
                continue
            ts = d.get('timestamp', '')[:19]
            dt = datetime.strptime(ts, '%Y-%m-%d_%H-%M-%S')
            is_val = 'val' in tag.lower()
            # Filter out subset val runs that aren't true full-val
            if is_val and 'bear_gr_val' in tag:
                continue
            entry = {'dt': dt, 'acc': d['top1_accuracy'] * 100, 'tag': tag}
            if is_val:
                val_evals.append(entry)
            else:
                train_evals.append(entry)
        except Exception:
            continue
    train_evals.sort(key=lambda x: x['dt'])
    val_evals.sort(key=lambda x: x['dt'])
    return train_evals, val_evals


def main():
    train_evals, val_evals = load_phase2_evals()
    print(f"Loaded {len(train_evals)} train evals, {len(val_evals)} val evals")

    train_dates = [e['dt'] for e in train_evals]
    train_accs = [e['acc'] for e in train_evals]
    val_dates = [e['dt'] for e in val_evals]
    val_accs = [e['acc'] for e in val_evals]

    fig, ax = plt.subplots(figsize=(18, 9))

    # Plot train as connected line
    ax.plot(train_dates, train_accs, color='#4A90D9', linewidth=1.0, alpha=0.7, zorder=2, label='Train accuracy')
    ax.scatter(train_dates, train_accs, s=6, alpha=0.4, color='#4A90D9', zorder=2)

    # Plot val as connected line
    ax.plot(val_dates, val_accs, color='#E8532A', linewidth=1.8, alpha=0.85, zorder=3, label='Val accuracy', marker='o', markersize=4)
    # Merge all evals for annotation positioning
    evals = train_evals + val_evals
    evals.sort(key=lambda x: x['dt'])

    # --- REFLECTION ANNOTATIONS ---
    # Format: (datetime, acc, label, worked: bool, y_offset)
    # Green = worked, Red = didn't work / hit ceiling
    annotations = [
        # What worked (green) — (dt, acc_point, label, worked, y_offset)
        (datetime(2026, 5, 12, 12, 53), 31.6, "Baseline signatures\n(color + texture)", True, -15),
        (datetime(2026, 5, 12, 15, 29), 36.2, "Pairwise reranking\nintroduced", True, 12),
        (datetime(2026, 5, 12, 16, 55), 43.5, "Calibration +\nrepulsion", True, 13),
        (datetime(2026, 5, 13, 7, 38), 50.1, "Histogram blending\n→ 50%", True, 12),
        (datetime(2026, 5, 15, 21, 15), 55.0, "Deep rank 3-5\nreranking", True, -17),
        (datetime(2026, 5, 17, 22, 55), 60.4, "Batch pairwise\ndiscriminants", True, 12),
        (datetime(2026, 5, 18, 11, 31), 70.0, "Prune bad rules\n+ tighten gates → 70%", True, 12),
        (datetime(2026, 5, 18, 17, 48), 76.8, "Verify rules\nwave 1 → 77%", True, -20),
        (datetime(2026, 5, 18, 19, 37), 86.8, "Wave 3: bulk\nswap rules → 87%", True, 8),
        (datetime(2026, 5, 18, 21, 50), 100.0, "100% train\n(executable memorizer)", True, -6),

        # What didn't work / lessons (red)
        (datetime(2026, 5, 12, 18, 20), 41.4, "DCT features\n(no gain)", False, -17),
        (datetime(2026, 5, 13, 1, 8), 45.5, "Local verify\n(too narrow)", False, 15),
        (datetime(2026, 5, 15, 21, 40), 52.0, "Proto tiebreak\n(−3pp crash)", False, 15),
        (datetime(2026, 5, 18, 0, 16), 53.0, "Val: 53%\n(gap emerges)", False, -17),
        (datetime(2026, 5, 18, 11, 58), 49.4, "Val: 49% at\n70% train", False, -20),
        (datetime(2026, 5, 18, 20, 32), 42.3, "Val: 42% at\n98% train", False, 15),
    ]

    for dt, acc, label, worked, y_off in annotations:
        color = '#2E8B57' if worked else '#DC143C'
        face = '#E8F5E9' if worked else '#FFEBEE'
        ax.annotate(
            label,
            xy=(dt, acc),
            xytext=(dt, acc + y_off),
            fontsize=7,
            ha='center',
            va='bottom' if y_off > 0 else 'top',
            color=color,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color, lw=0.9),
            bbox=dict(boxstyle='round,pad=0.3', facecolor=face, alpha=0.85, edgecolor=color, linewidth=0.7),
            zorder=5,
        )

    # CNN baseline reference
    ax.axhline(y=71.8, color='#888888', linestyle='-.', linewidth=1.2, alpha=0.5)
    ax.text(train_dates[-1], 72.8, "CNN baseline val (71.8%)", fontsize=8, color='#888888',
            ha='right', va='bottom', fontstyle='italic')

    # 10% random baseline
    ax.axhline(y=10, color='black', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(train_dates[0], 11.5, "Random (10%)", fontsize=8, alpha=0.5)

    # Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    fig.autofmt_xdate(rotation=30, ha='right')

    ax.set_xlabel("Time (2026)")
    ax.set_ylabel("Train Accuracy (%)")
    ax.set_title("Phase 2: Symbolic Classifier Train Accuracy Trajectory\n"
                 "31.6% → 100% over 11 days, 951 evaluations — pure heuristic code, no neural network")
    ax.set_ylim(5, 112)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.legend(loc='lower right', framealpha=0.9)

    # Legend for annotation colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E8F5E9', edgecolor='#2E8B57', label='Worked (milestone)'),
        Patch(facecolor='#FFEBEE', edgecolor='#DC143C', label="Didn't work / lesson"),
    ]
    leg2 = ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=9)
    ax.add_artist(leg2)
    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    out = PLOTS_DIR / "01_accuracy_trajectory.png"
    fig.savefig(out)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
