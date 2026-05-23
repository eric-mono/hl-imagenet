"""Mine zero-risk post-pipeline verify conditions.

Runs the full pipeline on all train images, collects errors where the true class
is at rank 2-5, and finds feature thresholds that fix errors without breaking
any correct predictions.

Output: prints deployable conditions as Python code.
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from hlinet.classifier.predict import predict
from hlinet.features.compounds.phase2_signatures import _stats, _stats_cache_graph


def run_mining():
    train_dir = Path("data/phase2/train")
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    # Step 1: Run pipeline on all images, collect results + features
    print("Step 1: Running pipeline on all train images...")
    sys.stdout.flush()

    results = []  # (image_path, true_class, predicted, rank_of_true, features)

    for ci, cls in enumerate(classes):
        cls_dir = train_dir / cls
        images = sorted(cls_dir.glob("*.JPEG")) + sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        for img_path in images[:200]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            pred = predict(img, mode="full")
            # Get features
            from hlinet.scene.builder import SceneGraphBuilder
            builder = SceneGraphBuilder()
            graph = builder.build(img)
            feats = _stats(graph)

            # Find rank of true class
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
            })

        print(f"  [{ci+1}/{len(classes)}] {cls} done ({len(images)} images)")
        sys.stdout.flush()

    # Step 2: Group errors by (predicted, true_at_rank_N)
    print(f"\nStep 2: Analyzing {len(results)} results...")
    sys.stdout.flush()

    errors = [r for r in results if r["predicted"] != r["true"] and r["rank"] <= 5]
    correct = [r for r in results if r["predicted"] == r["true"]]

    print(f"  Total correct: {len(correct)}")
    print(f"  Total errors with true at rank 2-5: {len(errors)}")

    # Group errors by (predicted_class, true_class)
    pair_errors = defaultdict(list)
    for r in errors:
        pair_errors[(r["predicted"], r["true"])].append(r)

    # Group correct by predicted class (these are the "risk pool")
    correct_by_pred = defaultdict(list)
    for r in correct:
        correct_by_pred[r["predicted"]].append(r)

    # Step 3: Mine conditions
    print("\nStep 3: Mining zero-risk conditions...")
    sys.stdout.flush()

    conditions = []
    feature_names = list(results[0]["features"].keys()) if results else []

    for (pred_cls, true_cls), err_list in sorted(pair_errors.items(), key=lambda x: -len(x[1])):
        if len(err_list) < 3:
            continue

        # Risk pool: all images correctly predicted as pred_cls
        risk_pool = correct_by_pred[pred_cls]
        if not risk_pool:
            continue

        # For each feature, find threshold that separates errors from risk
        for feat in feature_names:
            err_vals = [r["features"].get(feat) for r in err_list]
            err_vals = [v for v in err_vals if v is not None and not np.isnan(v)]
            if len(err_vals) < 3:
                continue

            risk_vals = [r["features"].get(feat) for r in risk_pool]
            risk_vals = [v for v in risk_vals if v is not None and not np.isnan(v)]
            if not risk_vals:
                continue

            risk_min = min(risk_vals)
            risk_max = max(risk_vals)

            # Try "feat > threshold" direction: fix errors above risk_max
            err_above = [v for v in err_vals if v > risk_max]
            if len(err_above) >= 3:
                conditions.append({
                    "pred": pred_cls,
                    "true": true_cls,
                    "feat": feat,
                    "direction": ">",
                    "threshold": risk_max,
                    "fix": len(err_above),
                    "risk": 0,
                    "total_errors": len(err_list),
                })

            # Try "feat < threshold" direction: fix errors below risk_min
            err_below = [v for v in err_vals if v < risk_min]
            if len(err_below) >= 3:
                conditions.append({
                    "pred": pred_cls,
                    "true": true_cls,
                    "feat": feat,
                    "direction": "<",
                    "threshold": risk_min,
                    "fix": len(err_below),
                    "risk": 0,
                    "total_errors": len(err_list),
                })

    # Sort by fix count descending
    conditions.sort(key=lambda c: -c["fix"])

    # Step 4: Report top conditions
    print(f"\nFound {len(conditions)} zero-risk conditions (fix >= 3)")
    print("\nTop 50 conditions:")
    print(f"{'Pair':<30} {'Feature':<25} {'Dir':<3} {'Threshold':<14} {'Fix':<5} {'Total':<6}")
    print("-" * 85)
    for c in conditions[:50]:
        pair = f"{c['pred']}→{c['true']}"
        print(f"{pair:<30} {c['feat']:<25} {c['direction']:<3} {c['threshold']:<14.6f} {c['fix']:<5} {c['total_errors']:<6}")

    # Step 5: Generate deployable code for top unique pairs
    print("\n\n=== DEPLOYABLE CODE ===\n")
    deployed_pairs = set()
    deploy_count = 0

    for c in conditions:
        pair_key = (c["pred"], c["true"])
        if pair_key in deployed_pairs:
            continue
        deployed_pairs.add(pair_key)
        deploy_count += 1
        if deploy_count > 30:
            break

        print(f"    # {c['pred']} → {c['true']}: fix {c['fix']}/{c['total_errors']}")
        if c["direction"] == ">":
            print(f"    if top_label == \"{c['pred']}\" and sec_label == \"{c['true']}\":")
            print(f"        if s.get(\"{c['feat']}\", 0) > {c['threshold']:.11f}:")
            print(f"            candidates[0], candidates[1] = candidates[1], candidates[0]")
        else:
            print(f"    if top_label == \"{c['pred']}\" and sec_label == \"{c['true']}\":")
            print(f"        if s.get(\"{c['feat']}\", 999) < {c['threshold']:.11f}:")
            print(f"            candidates[0], candidates[1] = candidates[1], candidates[0]")
        print()

    # Save full results
    output = {
        "total_images": len(results),
        "total_correct": len(correct),
        "total_errors_rank2_5": len(errors),
        "conditions_found": len(conditions),
        "conditions": conditions[:100],
    }
    out_path = Path("logs/final_verify_mining.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    run_mining()
