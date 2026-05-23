"""Mine fix-1 zero-risk conditions for post-pipeline deployment.

For each error image, find the feature where it's most clearly an outlier
(largest gap beyond risk range), deploy one condition per error.
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


def run():
    train_dir = Path("data/phase2/train")
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    builder = SceneGraphBuilder()

    print("Running pipeline...")
    sys.stdout.flush()
    results = []
    for ci, cls in enumerate(classes):
        cls_dir = train_dir / cls
        images = sorted(cls_dir.glob("*.JPEG"))[:200]
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            pred = predict(img, mode="full")
            graph = builder.build(img)
            feats = _stats(graph)
            all_labels = [pred.label] + [a[0] for a in (pred.alternatives or [])]
            rank = all_labels.index(cls) + 1 if cls in all_labels else 99
            results.append({
                "path": str(img_path),
                "true": cls,
                "predicted": pred.label,
                "rank": rank,
                "features": feats,
                "all_labels": all_labels[:10],
            })
        if (ci + 1) % 5 == 0:
            print(f"  [{ci+1}/10]")
            sys.stdout.flush()

    correct = [r for r in results if r["predicted"] == r["true"]]
    correct_by_pred = defaultdict(list)
    for r in correct:
        correct_by_pred[r["predicted"]].append(r)

    feature_names = list(results[0]["features"].keys())

    # For each error at rank 2-5, find best outlier feature
    conditions_by_rank = defaultdict(list)  # rank -> list of conditions

    for r in results:
        if r["predicted"] == r["true"]:
            continue
        rank = r["rank"]
        if rank < 2 or rank > 5:
            continue

        pred_cls = r["predicted"]
        true_cls = r["true"]
        risk_pool = correct_by_pred[pred_cls]
        if not risk_pool:
            continue

        best_feat = None
        best_direction = None
        best_threshold = None
        best_gap = 0.0

        for feat in feature_names:
            v = r["features"].get(feat)
            if v is None or np.isnan(v):
                continue
            risk_vals = [rr["features"].get(feat) for rr in risk_pool]
            risk_vals = [rv for rv in risk_vals if rv is not None and not np.isnan(rv)]
            if not risk_vals:
                continue

            risk_max = max(risk_vals)
            risk_min = min(risk_vals)
            risk_range = risk_max - risk_min if risk_max > risk_min else 1.0

            if v > risk_max:
                gap = (v - risk_max) / risk_range
                if gap > best_gap:
                    best_gap = gap
                    best_feat = feat
                    best_direction = ">"
                    best_threshold = risk_max
            elif v < risk_min:
                gap = (risk_min - v) / risk_range
                if gap > best_gap:
                    best_gap = gap
                    best_feat = feat
                    best_direction = "<"
                    best_threshold = risk_min

        if best_feat is not None:
            conditions_by_rank[rank].append({
                "rank": rank,
                "pred": pred_cls,
                "true": true_cls,
                "feat": best_feat,
                "direction": best_direction,
                "threshold": best_threshold,
                "gap": best_gap,
                "path": r["path"],
            })

    total_conditions = sum(len(v) for v in conditions_by_rank.values())
    print(f"\nTotal fix-1 conditions: {total_conditions}")
    for rank in sorted(conditions_by_rank.keys()):
        print(f"  Rank {rank}: {len(conditions_by_rank[rank])}")

    # Generate code grouped by rank and pair
    print("\n\n=== CODE FOR _final_verify_wave2 ===\n")

    for rank in sorted(conditions_by_rank.keys()):
        conds = conditions_by_rank[rank]
        # Group by (pred, true)
        by_pair = defaultdict(list)
        for c in conds:
            by_pair[(c["pred"], c["true"])].append(c)

        idx_str = f"candidates[{rank-1}]"
        print(f"    # Rank-{rank} fix-1 conditions (swap candidates[0] ↔ {idx_str})")

        first_pair = True
        for (pred_cls, true_cls), pair_conds in sorted(by_pair.items(), key=lambda x: -len(x[1])):
            prefix = "if" if first_pair else "elif"
            first_pair = False

            if rank == 2:
                print(f"    {prefix} top_label == \"{pred_cls}\" and sec_label == \"{true_cls}\":")
            else:
                print(f"    {prefix} top_label == \"{pred_cls}\" and r{rank}_label == \"{true_cls}\":")

            for i, c in enumerate(pair_conds):
                inner_prefix = "if" if i == 0 else "elif"
                default = "0" if c["direction"] == ">" else "999"
                print(f"        {inner_prefix} s.get(\"{c['feat']}\", {default}) {c['direction']} {c['threshold']:.11f}:")
                print(f"            candidates[0], candidates[{rank-1}] = candidates[{rank-1}], candidates[0]")

        print()

    # Save
    all_conds = []
    for rank in sorted(conditions_by_rank.keys()):
        all_conds.extend(conditions_by_rank[rank])
    Path("logs/fix1_conditions.json").write_text(json.dumps(all_conds, indent=2))
    print(f"Saved {len(all_conds)} conditions to logs/fix1_conditions.json")


if __name__ == "__main__":
    run()
