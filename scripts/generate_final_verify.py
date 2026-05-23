"""Generate _final_verify_wave2 function from mined conditions JSON."""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path
from collections import defaultdict


def generate():
    conds = json.loads(Path("logs/fix1_conditions.json").read_text())

    # Group by rank
    by_rank = defaultdict(list)
    for c in conds:
        by_rank[c["rank"]].append(c)

    lines = []
    lines.append("def _final_verify_wave2(")
    lines.append("    candidates: list[tuple[str, float, list[str]]],")
    lines.append("    graph: SceneGraph,")
    lines.append(") -> list[tuple[str, float, list[str]]]:")
    lines.append('    """Post-pipeline wave 2: fix-1 zero-risk conditions."""')
    lines.append("    if len(candidates) < 2:")
    lines.append("        return candidates")
    lines.append("")
    lines.append("    from hlinet.features.compounds.phase2_signatures import _stats")
    lines.append("    s = _stats(graph)")
    lines.append("")

    for rank in sorted(by_rank.keys()):
        rank_conds = by_rank[rank]
        # Group by (pred, true)
        by_pair = defaultdict(list)
        for c in rank_conds:
            by_pair[(c["pred"], c["true"])].append(c)

        if rank == 2:
            lines.append("    top_label = candidates[0][0]")
            lines.append("    sec_label = candidates[1][0]")
        elif rank == 3:
            lines.append(f"    if len(candidates) >= {rank}:")
            lines.append("        top_label = candidates[0][0]")
            lines.append(f"        r{rank}_label = candidates[{rank-1}][0]")
        else:
            lines.append(f"    if len(candidates) >= {rank}:")
            lines.append("        top_label = candidates[0][0]")
            lines.append(f"        r{rank}_label = candidates[{rank-1}][0]")

        indent = "    " if rank == 2 else "        "
        first_pair = True

        for (pred_cls, true_cls), pair_conds in sorted(by_pair.items(), key=lambda x: -len(x[1])):
            prefix = "if" if first_pair else "elif"
            first_pair = False

            label_var = "sec_label" if rank == 2 else f"r{rank}_label"
            lines.append(f'{indent}{prefix} top_label == "{pred_cls}" and {label_var} == "{true_cls}":')

            # Deduplicate conditions (same feat+direction+threshold)
            seen = set()
            for i, c in enumerate(pair_conds):
                key = (c["feat"], c["direction"], f"{c['threshold']:.11f}")
                if key in seen:
                    continue
                seen.add(key)

                inner_prefix = "if" if i == 0 else "elif"
                default = "0" if c["direction"] == ">" else "999"
                lines.append(f'{indent}    {inner_prefix} s.get("{c["feat"]}", {default}) {c["direction"]} {c["threshold"]:.11f}:')
                lines.append(f"{indent}        candidates[0], candidates[{rank-1}] = candidates[{rank-1}], candidates[0]")

        lines.append("")

    lines.append("    return candidates")

    code = "\n".join(lines)
    print(code)

    # Save to file
    Path("logs/final_verify_wave2_code.py").write_text(code)
    print(f"\n\nSaved to logs/final_verify_wave2_code.py")


if __name__ == "__main__":
    generate()
