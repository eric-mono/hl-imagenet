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
| Phase 2 symbolic pipeline, Session 20/21 | 70.0% | 49.4% | Main no-tree symbolic result: base scoring + rerank + in-pipeline verify |
| Phase 2 archived full verify-wave endpoint | 100.0% | 41.35% | Historical Session 26 overfit endpoint, not the active restarted loop |
| Small CNN baseline | 76.0% | 71.8% | Learned-representation reference |

## Current Research Frame

The central lesson is not that symbolic vision reaches high validation accuracy. It does not. A historical Session 26 endpoint did reach 100.0% train accuracy by stacking verify waves, but that was an overfit artifact recorded in the logs, not the current restarted loop state. The central lesson is that **train-only heuristic learning over code can memorize just like train-only parameter learning**.

For reporting, freeze two symbolic artifacts: the Session 20/21 no-tree symbolic pipeline at 70.0% train / 49.4% val, and the archived Session 26 verify-wave endpoint at 100.0% train / 41.35% val. The contrast between them is the main result.

In this project, the codebase is the model, patches are actions, and evaluation accuracy is reward. When the reward is train accuracy, the agent eventually writes a memorizer made of thresholds and special cases. The next research step is a heuristic-learning loop with a generalization reward, explicit regularization, and representation-level feature invention.
