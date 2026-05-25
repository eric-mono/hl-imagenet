# Heuristic Learning Analysis

## The Core Lesson

Train-only heuristic search learns code that memorizes, just like train-only gradient descent learns parameters that memorize.

In this project, the codebase is the model. A patch is a policy update. The evaluation score is reward. The session logs are memory. Once the reward became train accuracy, the agent optimized toward executable memorization: hundreds of narrow threshold conditions that recover individual training images but do not transfer.

That is not a failure of the heuristic-learning frame. It is evidence that heuristic learning needs the same theory of generalization that neural learning needs.

## The Honest Phase Boundary

After the latest reproducibility audit, the current hand-built symbolic Phase 2 systems are:

| System | Train | Val | Reading |
|---|---:|---:|---|
| `base_rerank` | 55.4% | 51.9% | Best generalizing symbolic core |
| `full` verify-rule system | 84.0% | 50.5% | Higher train, larger generalization gap |
| archived historical endpoint | 100.0% train | not current ground truth | Reached in logs; exact code state is not currently reproducible |

The `base_rerank` result comes from:

- sigmoid class signatures over hand-crafted visual statistics;
- histogram prototype blending;
- pairwise discriminant reranking;

The `full` result adds local verify rules and rank-3/4/5/final verification waves. No trees are involved in either hand-built symbolic result.

The later archived **100% train** endpoint is the overfitting endpoint of the same Phase 2 pipeline. It is historical evidence that code can memorize as effectively as parameters when the loop rewards training accuracy, but it should not be used as the current reproducible headline.

Anycode is separate and should not be used to explain the hand-built Phase 2 symbolic result.

## What Actually Generalized

The evidence points to three symbolic layers:

- Base visual statistics generalize moderately.
- Pairwise reranking generalizes best among the post-base interventions.
- Verify rules overfit, especially when they fire on only a few training images.

For public reporting, the canonical hand-built symbolic systems are:

- `base_rerank`: **55.4% train / 51.9% val**, the best generalizing symbolic core;
- `full`: **84.0% train / 50.5% val**, the current high-train verify-rule system.

The old recommendation to headline the Session 20/21 **70.0% train / 49.4% val** pipeline is superseded by the current reproducibility audit.

## What To Do Next

Stop optimizing train accuracy. Use train for proposing rules, a dev split for accepting or rejecting them, and keep val/test untouched. The HL loop needs a generalization reward, not a train reward.

Freeze `base_rerank` as the main generalizing no-tree artifact and `full` as the high-train diagnostic artifact. Treat the archived 100% train system as historical evidence of executable memorization unless the exact code state is recovered.

Move from per-image verify rules to reusable heuristics. Accept rules only if they fire on enough examples, with support around 10-20, precision on held-out dev, and no class collapse. No more fix-1 thresholds.

Improve representation, not thresholds. The current features are too global: mean color, edge density, texture stats. The next useful HL direction is object/region-centered features: foreground masks, contour descriptors, local patch pooling, "pattern exists somewhere" detectors, and part relations.

Maintain [../understanding/](../understanding/) as the reflection memory. Every accepted or rejected direction should update the relevant understanding file before it becomes a public claim.

## Why The System Became Fragile

The pipeline is sequential:

`signature score -> histogram blend -> calibration -> repulsion -> sort -> rerank -> verify`

A small early change changes the score gaps, which changes which pairs get checked, which changes which verify conditions fire. By Session 21, 496 post-processing rescues depended on the exact shape of the ranking. Changes that were locally sensible caused global regressions.

This is cascade dynamics: the codebase behaves like an evolved system under selection pressure, not like a cleanly separable design.

## The Research Claim

The useful claim is not "symbolic image classification beats neural networks." It does not.

The useful claim is:

> Heuristic learning is program induction under feedback. If its reward is train accuracy, it overfits by writing code that memorizes. The next theory needs to explain how to regularize patches, measure heuristic generality, and select updates by held-out transfer.

That is the opening for a new theory of heuristic learning.
