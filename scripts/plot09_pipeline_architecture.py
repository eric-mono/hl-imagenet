"""Generate the Phase 2 symbolic pipeline architecture figure.

This plot is intentionally qualitative. Numeric annotations tend to become
stale as the classifier changes, so audited accuracy numbers belong in the
trajectory, ablation, and gap plots instead.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "phase2" / "plots" / "09_pipeline_architecture.png"


def main() -> None:
    fig, ax = plt.subplots(figsize=(17.0, 5.0))
    ax.set_xlim(0, 17.0)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    title_y = 5.05
    ax.text(
        7.75,
        title_y,
        "Phase 2 Symbolic Pipeline Architecture",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )
    ax.text(
        7.75,
        4.58,
        "Non-neural symbolic code — no gradient descent, no learned embedding model, no stored training images at prediction time",
        ha="center",
        va="center",
        fontsize=12,
        color="#666666",
    )

    boxes = [
        ("Image\n64x64", "#dcecf8"),
        ("Scene Graph\n+ Features", "#e4f2e6"),
        ("Signature\nScoring", "#fbefd9"),
        ("Histogram\nBlend", "#fbefd9"),
        ("Pairwise\nRerank", "#e4f2e6"),
        ("Verify\nConditions", "#f8e4e8"),
        ("Prediction", "#dcecf8"),
    ]

    x0 = 0.45
    y0 = 2.05
    box_w = 1.82
    box_h = 1.22
    gap = 0.52

    centers = []
    for i, (label, color) in enumerate(boxes):
        x = x0 + i * (box_w + gap)
        centers.append((x + box_w / 2, y0 + box_h / 2))
        ax.add_patch(
            Rectangle(
                (x, y0),
                box_w,
                box_h,
                facecolor=color,
                edgecolor="#333333",
                linewidth=1.8,
            )
        )
        ax.text(
            x + box_w / 2,
            y0 + box_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            linespacing=0.95,
        )

    for i in range(len(boxes) - 1):
        start = (centers[i][0] + box_w / 2, centers[i][1])
        end = (centers[i + 1][0] - box_w / 2, centers[i + 1][1])
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=16,
                linewidth=1.8,
                color="#333333",
            )
        )

    annotations = {
        1: "visual\nmeasurements",
        2: "feature\nconjunctions",
        3: "compact\nprototypes",
        4: "confusion\ncorrections",
        5: "symbolic\nchecks",
    }
    for idx, text in annotations.items():
        cx, _ = centers[idx]
        ax.text(
            cx,
            1.25,
            text,
            ha="center",
            va="center",
            fontsize=9.6,
            color="#666666",
            linespacing=1.05,
        )

    ax.text(
        8.5,
        0.42,
        "Qualitative architecture only; audited accuracy numbers are shown in the trajectory, ablation, and generalization-gap plots.",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#777777",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
