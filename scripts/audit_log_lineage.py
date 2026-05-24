#!/usr/bin/env python3
"""Classify historical eval logs by experiment lineage.

Directory names are not reliable in this repository: many later Phase 2 runs
were saved under logs/phase1 because the evaluator used a tag-prefix router.
This script classifies logs from their contents and writes an inventory that
plotting/reporting code can use as a filter.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS_ROOT = ROOT / "logs"

PHASE2_CLASSES = {
    "golden_retriever", "mushroom", "teapot", "school_bus", "banana",
    "orange", "brown_bear", "king_penguin", "jellyfish", "sports_car",
}
PHASE1_CLASSES = {
    "zebra", "school_bus", "golden_retriever", "bicycle", "mushroom",
    "teapot", "piano", "eagle", "laptop", "banana",
}


@dataclass
class LogRecord:
    path: Path
    directory: str
    tag: str
    timestamp: str
    top1: float | None
    top3: float | None
    n_samples: int | None
    class_set: str
    lineage: str
    split: str
    comparable_phase2: bool
    reason: str


def _class_set(per_class: dict) -> str:
    keys = set(per_class.keys())
    if keys == PHASE2_CLASSES:
        return "phase2_10class"
    if keys == PHASE1_CLASSES:
        return "phase1_10class"
    if not keys:
        return "none"
    if keys <= PHASE2_CLASSES:
        return "phase2_subset"
    if keys <= PHASE1_CLASSES:
        return "phase1_subset"
    return "mixed_or_unknown"


def _guess_split(path: Path, tag: str) -> str:
    text = f"{path.stem} {tag}".lower()
    if "inner_train" in text:
        return "inner_train"
    if "inner_dev" in text:
        return "inner_dev"
    tokens = {t for t in re.split(r"[^a-z0-9]+", text) if t}
    if "test" in tokens:
        return "test"
    if "val" in tokens or "validation" in tokens:
        return "val"
    if "train" in tokens:
        return "train"
    return "unknown"


def _guess_lineage(path: Path, tag: str, class_set: str) -> tuple[str, str]:
    text = f"{path.as_posix()} {tag}".lower()
    if class_set.startswith("phase2"):
        return "phase2", "per-class labels match Phase 2 classes"
    if class_set.startswith("phase1"):
        return "phase1", "per-class labels match Phase 1 classes"
    if "logs/phase2/" in text or "phase2" in text:
        return "phase2", "path/tag says phase2"
    if "logs/phase1/" in text or "phase1" in text:
        return "phase1", "path/tag says phase1"
    return "unknown", "no class keys or phase marker"


def load_records() -> list[LogRecord]:
    records: list[LogRecord] = []
    for path in sorted(LOGS_ROOT.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if "top1_accuracy" not in data:
            continue

        per_class = data.get("per_class_accuracy") or {}
        class_set = _class_set(per_class)
        tag = str(data.get("tag", ""))
        lineage, lineage_reason = _guess_lineage(path, tag, class_set)
        split = _guess_split(path, tag)
        n_samples = data.get("n_samples")
        comparable_phase2 = lineage == "phase2" and class_set == "phase2_10class" and n_samples == 2000

        reason = lineage_reason
        if comparable_phase2:
            reason += "; 2000-sample full Phase 2 eval"
        elif lineage == "phase2" and n_samples != 2000:
            reason += f"; excluded from comparable trajectory because n={n_samples}"

        records.append(LogRecord(
            path=path.relative_to(ROOT),
            directory=path.parent.relative_to(LOGS_ROOT).as_posix(),
            tag=tag,
            timestamp=str(data.get("timestamp", "")),
            top1=data.get("top1_accuracy"),
            top3=data.get("top3_accuracy"),
            n_samples=n_samples,
            class_set=class_set,
            lineage=lineage,
            split=split,
            comparable_phase2=comparable_phase2,
            reason=reason,
        ))
    return records


def write_csv(records: list[LogRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path", "directory", "tag", "timestamp", "top1", "top3", "n_samples",
        "class_set", "lineage", "split", "comparable_phase2", "reason",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "path": r.path.as_posix(),
                "directory": r.directory,
                "tag": r.tag,
                "timestamp": r.timestamp,
                "top1": "" if r.top1 is None else f"{r.top1:.6f}",
                "top3": "" if r.top3 is None else f"{r.top3:.6f}",
                "n_samples": "" if r.n_samples is None else r.n_samples,
                "class_set": r.class_set,
                "lineage": r.lineage,
                "split": r.split,
                "comparable_phase2": r.comparable_phase2,
                "reason": r.reason,
            })


def write_markdown(records: list[LogRecord], path: Path) -> None:
    counts = Counter(r.lineage for r in records)
    class_counts = Counter(r.class_set for r in records)
    comparable = [r for r in records if r.comparable_phase2]
    misplaced = [
        r for r in comparable
        if r.directory == "phase1"
    ]
    top_phase2 = sorted(comparable, key=lambda r: (r.top1 or 0), reverse=True)[:20]

    by_dir_lineage: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        by_dir_lineage[r.directory][r.lineage] += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Log Lineage Inventory\n\n")
        f.write("Generated by `scripts/audit_log_lineage.py`.\n\n")
        f.write("## Why This Exists\n\n")
        f.write(
            "`logs/phase1/` is not a clean Phase 1 directory. Older evaluator "
            "routing sent every tag not starting with `phase2` into `logs/phase1`, "
            "so many later Phase 2 train-only, verify-wave, loop, and repro runs "
            "live there. Use lineage/classification metadata, not directory names, "
            "when building plots.\n\n"
        )
        f.write("## Summary\n\n")
        f.write(f"- Eval JSON records: {len(records)}\n")
        f.write(f"- Comparable full Phase 2 records: {len(comparable)}\n")
        f.write(f"- Comparable Phase 2 records currently under `logs/phase1/`: {len(misplaced)}\n\n")

        f.write("### By Inferred Lineage\n\n")
        for key, count in sorted(counts.items()):
            f.write(f"- `{key}`: {count}\n")
        f.write("\n### By Class Set\n\n")
        for key, count in sorted(class_counts.items()):
            f.write(f"- `{key}`: {count}\n")

        f.write("\n### Directory vs Inferred Lineage\n\n")
        f.write("| Directory | Phase 1 | Phase 2 | Unknown |\n")
        f.write("|---|---:|---:|---:|\n")
        for directory, counter in sorted(by_dir_lineage.items()):
            f.write(
                f"| `{directory}` | {counter.get('phase1', 0)} | "
                f"{counter.get('phase2', 0)} | {counter.get('unknown', 0)} |\n"
            )

        f.write("\n## Plotting Rules\n\n")
        f.write("- For the Phase 2 main trajectory, include only `comparable_phase2 == true`.\n")
        f.write("- Do not use raw `logs/phase1/` or `logs/phase2/` directory membership as a filter.\n")
        f.write("- Treat archived 100% train logs as historical endpoints unless the exact code is present in current `HEAD`.\n")
        f.write("- Keep Phase 1 dev results separate from Phase 2 2000-sample train/val results.\n")

        f.write("\n## Highest Comparable Phase 2 Records By Top-1\n\n")
        f.write(
            "These are ranked only by recorded accuracy. `split` is a filename/tag "
            "guess, so unknown split rows should not be treated as validation results.\n\n"
        )
        f.write("| Top-1 | Split guess | Tag | Path |\n")
        f.write("|---:|---|---|---|\n")
        for r in top_phase2:
            f.write(
                f"| {100 * (r.top1 or 0):.2f}% | {r.split} | `{r.tag}` | "
                f"`{r.path.as_posix()}` |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit eval log lineage.")
    parser.add_argument("--csv", type=Path, default=LOGS_ROOT / "log_inventory.csv")
    parser.add_argument("--md", type=Path, default=LOGS_ROOT / "README.md")
    parser.add_argument("--write", action="store_true", help="Write inventory files.")
    args = parser.parse_args()

    records = load_records()
    comparable = [r for r in records if r.comparable_phase2]
    misplaced = [r for r in comparable if r.directory == "phase1"]

    print(f"Eval JSON records: {len(records)}")
    print(f"Comparable full Phase 2 records: {len(comparable)}")
    print(f"Comparable Phase 2 records under logs/phase1: {len(misplaced)}")

    if args.write:
        write_csv(records, args.csv)
        write_markdown(records, args.md)
        print(f"Wrote {args.csv}")
        print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
