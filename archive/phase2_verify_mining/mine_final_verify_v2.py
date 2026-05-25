"""Mine zero-risk post-pipeline verify conditions for ranks 2-5.

Improved version: tracks specific images fixed per condition to avoid double-counting.
Also mines rank 3-5 conditions (not just rank 2).
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


def run_mining():
    train_dir = Path("data/phase2/train")
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

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
            from hlinet.scene.builder import SceneGraphBuilder
            builder = SceneGraphBuilder()
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
                "all_labels": all_labels,
            })

        print(f"  [{ci+1}/{len(classes)}] {cls} done")
        sys.stdout.flush()

    # Group errors by rank
    errors_by_rank = defaultdict(list)
    for r in results:
        if r["predicted"] != r["true"] and 2 <= r["rank"] <= 5:
            errors_by_rank[r["rank"]].append(r)

    correct = [r for r in results if r["predicted"] == r["true"]]
    correct_by_pred = defaultdict(list)
    for r in correct:
        correct_by_pred[r["predicted"]].append(r)

    print(f"\nErrors by rank: {dict((k, len(v)) for k, v in sorted(errors_by_rank.items()))}")
    print(f"Total correct: {len(correct)}")

    # Mine for each rank
    all_conditions = []
    feature_names = list(results[0]["features"].keys())

    for rank in [2, 3, 4, 5]:
        errors = errors_by_rank[rank]
        if not errors:
            continue
        print(f"\n--- Mining rank {rank} ({len(errors)} errors) ---")
        sys.stdout.flush()

        # Group by (predicted, true)
        pair_errors = defaultdict(list)
        for r in errors:
            pair_errors[(r["predicted"], r["true"])].append(r)

        for (pred_cls, true_cls), err_list in sorted(pair_errors.items(), key=lambda x: -len(x[1])):
            if len(err_list) < 3:
                continue

            # Risk pool: correct predictions as pred_cls (can't break these)
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

                # Try "feat > threshold": fix errors above risk_max
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
                        "fixed_images": [r["path"] for r, v in err_above],
                    })

                # Try "feat < threshold": fix errors below risk_min
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
                        "fixed_images": [r["path"] for r, v in err_below],
                    })

    # Deduplicate: for each pair+rank, pick conditions that fix different images
    print(f"\n\nTotal raw conditions: {len(all_conditions)}")

    # Greedy selection: pick conditions that maximize unique fixes
    all_conditions.sort(key=lambda c: -c["fix"])
    selected = []
    fixed_images_global = set()

    for c in all_conditions:
        new_fixes = set(c["fixed_images"]) - fixed_images_global
        if len(new_fixes) >= 2:  # Only deploy if it adds at least 2 new fixes
            selected.append(c)
            fixed_images_global.update(c["fixed_images"])

    print(f"Selected conditions (greedy, new_fixes >= 2): {len(selected)}")
    print(f"Total unique images fixed: {len(fixed_images_global)}")
    print(f"Expected accuracy improvement: +{len(fixed_images_global)/2000*100:.1f}pp")

    # Report
    print(f"\n{'Rank':<5} {'Pair':<30} {'Feature':<25} {'Dir':<3} {'Threshold':<14} {'Fix':<5}")
    print("-" * 85)
    for c in selected:
        pair = f"{c['pred']}→{c['true']}"
        print(f"{c['rank']:<5} {pair:<30} {c['feat']:<25} {c['direction']:<3} {c['threshold']:<14.8f} {c['fix']:<5}")

    # Save
    output = {
        "total_images": len(results),
        "total_correct": len(correct),
        "selected_conditions": [{k: v for k, v in c.items() if k != "fixed_images"} for c in selected],
        "unique_fixes": len(fixed_images_global),
    }
    Path("logs/final_verify_mining_v2.json").write_text(json.dumps(output, indent=2))
    print(f"\nSaved to logs/final_verify_mining_v2.json")


if __name__ == "__main__":
    run_mining()
