"""Mine zero-risk post-pipeline verify conditions - correct version.

For each rank N error (true class at rank N), finds features where error images
are outside the range of correct predictions for the SAME predicted class.
Ensures the swap is to the correct rank position.
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from hlinet.classifier.predict import predict
from hlinet.features.compounds.phase2_signatures import _stats
from hlinet.scene.builder import SceneGraphBuilder


def run_mining():
    train_dir = Path("data/phase2/train")
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    builder = SceneGraphBuilder()

    print("Step 1: Running pipeline on all train images...")
    sys.stdout.flush()

    results = []

    for ci, cls in enumerate(classes):
        cls_dir = train_dir / cls
        images = sorted(cls_dir.glob("*.JPEG")) + sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        for img_path in images[:200]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            pred = predict(img, mode="full")
            graph = builder.build(img)
            feats = _stats(graph)

            all_labels = [pred.label] + [a[0] for a in (pred.alternatives or [])]
            if cls in all_labels:
                rank = all_labels.index(cls) + 1
            else:
                rank = 99

            results.append({
                "path": str(img_path),
                "true": cls,
                "predicted": pred.label,
                "rank": rank,
                "features": feats,
                "all_labels": all_labels[:5],
            })

        print(f"  [{ci+1}/{len(classes)}] {cls} done")
        sys.stdout.flush()

    correct = [r for r in results if r["predicted"] == r["true"]]
    correct_by_pred = defaultdict(list)
    for r in correct:
        correct_by_pred[r["predicted"]].append(r)

    # For rank-N errors, we need to check against correct predictions
    # where candidates[N-1] happens to be the same class
    # But simpler: the risk pool is ALL correct predictions for `predicted` class
    # because a swap from position 0 to position N-1 would break a correct prediction

    print(f"\nTotal correct: {len(correct)}")

    all_conditions = []
    feature_names = list(results[0]["features"].keys()) if results else []

    for rank in range(2, 11):
        errors = [r for r in results if r["rank"] == rank]
        if not errors:
            continue

        # Group by (predicted, true, rank-position label in alternatives)
        # The key constraint: for a rank-N swap, we need:
        # 1. top_label == predicted class
        # 2. candidates[N-1] == true class
        # This is already ensured by rank == N (true class IS at position N)

        pair_errors = defaultdict(list)
        for r in errors:
            pair_errors[(r["predicted"], r["true"])].append(r)

        for (pred_cls, true_cls), err_list in sorted(pair_errors.items(), key=lambda x: -len(x[1])):
            if len(err_list) < 3:
                continue

            # Risk pool for rank-N swap: ALL images correctly predicted as pred_cls
            # If condition fires on a correct image, swapping position 0 with N-1 makes it wrong
            risk_pool = correct_by_pred[pred_cls]
            if not risk_pool:
                continue

            for feat in feature_names:
                err_vals = [(r, r["features"].get(feat)) for r in err_list]
                err_vals = [(r, v) for r, v in err_vals if v is not None and not np.isnan(v)]
                if len(err_vals) < 3:
                    continue

                risk_vals = [r["features"].get(feat) for r in risk_pool]
                risk_vals = [v for v in risk_vals if v is not None and not np.isnan(v)]
                if not risk_vals:
                    continue

                risk_min = min(risk_vals)
                risk_max = max(risk_vals)

                err_above = [(r, v) for r, v in err_vals if v > risk_max]
                if len(err_above) >= 3:
                    all_conditions.append({
                        "rank": rank,
                        "pred": pred_cls,
                        "true": true_cls,
                        "feat": feat,
                        "direction": ">",
                        "threshold": risk_max,
                        "fix": len(err_above),
                        "risk": 0,
                        "total_errors": len(err_list),
                        "fixed_paths": [r["path"] for r, _ in err_above],
                    })

                err_below = [(r, v) for r, v in err_vals if v < risk_min]
                if len(err_below) >= 3:
                    all_conditions.append({
                        "rank": rank,
                        "pred": pred_cls,
                        "true": true_cls,
                        "feat": feat,
                        "direction": "<",
                        "threshold": risk_min,
                        "fix": len(err_below),
                        "risk": 0,
                        "total_errors": len(err_list),
                        "fixed_paths": [r["path"] for r, _ in err_below],
                    })

    # Greedy selection maximizing unique fixes
    all_conditions.sort(key=lambda c: (-c["fix"], c["rank"]))
    selected = []
    fixed_global = set()

    for c in all_conditions:
        new_fixes = set(c["fixed_paths"]) - fixed_global
        if len(new_fixes) >= 2:
            c["unique_new"] = len(new_fixes)
            selected.append(c)
            fixed_global.update(c["fixed_paths"])

    print(f"\nTotal raw conditions: {len(all_conditions)}")
    print(f"Selected (greedy): {len(selected)}")
    print(f"Total unique fixes: {len(fixed_global)}")
    print(f"Expected improvement: +{len(fixed_global)/20:.1f}pp")

    print(f"\n{'Rk':<4} {'Pair':<35} {'Feature':<25} {'Dir':<3} {'Thresh':<14} {'Fix':<4} {'New':<4}")
    print("-" * 90)
    for c in selected:
        pair = f"{c['pred']}→{c['true']}"
        print(f"{c['rank']:<4} {pair:<35} {c['feat']:<25} {c['direction']:<3} {c['threshold']:<14.8f} {c['fix']:<4} {c['unique_new']:<4}")

    # Save
    save_data = [{k: v for k, v in c.items() if k != "fixed_paths"} for c in selected]
    Path("logs/final_verify_mining_v3.json").write_text(json.dumps(save_data, indent=2))
    print(f"\nSaved to logs/final_verify_mining_v3.json")


if __name__ == "__main__":
    run_mining()
