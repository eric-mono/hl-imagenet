# HL-ImageNet Docs

The docs are organized by experiment lineage. The main story is the hand-built symbolic pipeline; the anycode/tree material is a side branch and should not be used to explain the symbolic result.

## Map

- [phase1/](phase1/) — Exploratory 4-real + 6-synthetic setup. Useful for the first proof that the heuristic-learning loop can grow a symbolic visual system, but not an honest 10-class benchmark.
- [phase2/](phase2/) — Main hand-built symbolic pipeline on 10 real Tiny ImageNet classes. This is the core heuristic-learning story: signatures, histogram blending, pairwise reranking, verify rules, cascade dynamics, overfitting, and the Phase 2 reflection/writing strategy.
- [anycode/](anycode/) — Side experiment that removes architecture constraints. Useful as background only; not part of the main symbolic result.
- [phase3/](phase3/) — Forward plan for higher-resolution local perception.

## Key Numbers

| Track | Train | Val | Status |
|---|---:|---:|---|
| Phase 1 dev system | 86.1% dev | 51-54% held-out subset | Historical proof-of-loop |
| Phase 2 `base_rerank` symbolic core | 55.4% | 51.9% | Current reproducible best-generalizing symbolic system |
| Phase 2 `full` verify-rule system | 84.0% | 50.5% | Current reproducible high-train symbolic system |
| Phase 2 archived full verify-wave endpoint | 100.0% train | not current ground truth | Historical overfit endpoint, exact code state not currently reproducible |
| Small CNN baseline | 76.0% | 71.8% | Learned-representation reference |

## Current Research Frame

The central lesson is not that symbolic vision reaches high validation accuracy. It does not. The current reproducible symbolic core reaches 51.9% validation, while the current full verify system reaches 84.0% train but only 50.5% validation. A historical endpoint did reach 100.0% train accuracy by stacking verify waves, but that exact code state is not currently reproducible from `HEAD`. The central lesson is that **train-only heuristic learning over code can memorize just like train-only parameter learning**.

For reporting, use `base_rerank` at 55.4% train / 51.9% val as the best-generalizing symbolic core, `full` verify at 84.0% train / 50.5% val as the current high-train symbolic system, and the archived 100.0% train endpoint only as historical evidence of executable memorization.

In this project, the codebase is the model, patches are actions, and evaluation accuracy is reward. When the reward is train accuracy, the agent eventually writes a memorizer made of thresholds and special cases. The next research step is a heuristic-learning loop with a generalization reward, explicit regularization, and representation-level feature invention.
