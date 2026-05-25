# Phase 2 Docs

Phase 2 is the main hand-built symbolic pipeline on 10 real Tiny ImageNet classes at 64x64.

## Files

- [phase2_eda.md](phase2_eda.md) — Dataset and class statistics.
- [blog.md](blog.md) — Phase 2 public writeup using the latest reproducible symbolic results.
- [phase2_experiment_log.md](phase2_experiment_log.md) — Early chronological experiment notes.
- [phase2_local_vision.md](phase2_local_vision.md) — Proposal to move from global statistics to local perception.
- [lessons.md](lessons.md) — Lessons from the hand-built symbolic pipeline.
- [theory.md](theory.md) — Theoretical framing for next-generation heuristic learning.
- [understanding/](understanding/) — Distilled Session 1-26 knowledge about cascade dynamics, overfitting, hard confusions, and patch safety.
- [reflections/](reflections/) — Phase 2 analysis, X drafts, blog outline, and next-step research strategy.
- [plots/](plots/) — Phase 2 trajectory plot.
- [logs/composition_architecture.md](logs/composition_architecture.md) — Composition analysis of the symbolic feature/scoring stack.

## Most Important Boundary

The current reproducible Phase 2 symbolic results are:

| System | Train | Val | Reading |
|---|---:|---:|---|
| `base_rerank` | 55.4% | 51.9% | Best generalizing symbolic core |
| `full` verify-rule system | 84.0% | 50.5% | Higher train, larger generalization gap |
| archived historical endpoint | 100.0% train | not current ground truth | Reached in logs, exact code state not currently reproducible |

The archived 100% train endpoint is also Phase 2, but it should be treated as historical evidence of executable memorization rather than the current reproducible artifact.

## Reflection Strategy

Keep [understanding/](understanding/) as the durable working memory for Phase 2. It should record hard-won facts, failed patterns, patch safety, generalization gaps, and feature inventory updates after every serious run.

Keep [reflections/](reflections/) as the Phase 2 public/research layer. It should turn the understanding files into X drafts, blog structure, and the next research program. It should not be a separate global synthesis track.
