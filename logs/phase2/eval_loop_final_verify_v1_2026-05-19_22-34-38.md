# Eval Run: 2026-05-19_22-34-38

**Tag:** loop_final_verify_v1
**Samples:** 2000
**Top-1 Accuracy:** 0.710
**Top-3 Accuracy:** 0.814
**Mean Latency:** 92 ms

## Per-Class Accuracy

| Class | Accuracy | Correct/Total |
|-------|----------|---------------|
| banana | 0.705 | 141/200 |
| brown_bear | 0.740 | 148/200 |
| golden_retriever | 0.635 | 127/200 |
| jellyfish | 0.775 | 155/200 |
| king_penguin | 0.715 | 143/200 |
| mushroom | 0.660 | 132/200 |
| orange | 0.720 | 144/200 |
| school_bus | 0.825 | 165/200 |
| sports_car | 0.735 | 147/200 |
| teapot | 0.585 | 117/200 |

## Top Confusions

- teapot → banana: 21
- sports_car → school_bus: 20
- orange → banana: 16
- teapot → golden_retriever: 15
- golden_retriever → king_penguin: 13
- mushroom → brown_bear: 13
- mushroom → golden_retriever: 13
- brown_bear → golden_retriever: 13
- golden_retriever → brown_bear: 12
- golden_retriever → teapot: 11

## Feature Reuse

- phase2_golden_retriever_signature: used by 10 classes
- golden_fur_in_nature: used by 10 classes
- quadruped_like: used by 10 classes
- phase2_jellyfish_signature: used by 10 classes
- phase2_mushroom_signature: used by 10 classes
- phase2_teapot_signature: used by 10 classes
- phase2_school_bus_signature: used by 10 classes
- phase2_banana_signature: used by 10 classes
- yellow_dominant: used by 10 classes
- phase2_orange_signature: used by 10 classes
