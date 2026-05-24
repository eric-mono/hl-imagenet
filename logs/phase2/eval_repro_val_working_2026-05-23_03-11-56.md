# Eval Run: 2026-05-23_03-11-56

**Tag:** repro_val_working
**Samples:** 2000
**Top-1 Accuracy:** 0.505
**Top-3 Accuracy:** 0.742
**Mean Latency:** 91 ms

## Per-Class Accuracy

| Class | Accuracy | Correct/Total |
|-------|----------|---------------|
| banana | 0.485 | 97/200 |
| brown_bear | 0.425 | 85/200 |
| golden_retriever | 0.355 | 71/200 |
| jellyfish | 0.710 | 142/200 |
| king_penguin | 0.495 | 99/200 |
| mushroom | 0.450 | 90/200 |
| orange | 0.575 | 115/200 |
| school_bus | 0.685 | 137/200 |
| sports_car | 0.540 | 108/200 |
| teapot | 0.335 | 67/200 |

## Top Confusions

- golden_retriever → mushroom: 32
- orange → banana: 32
- brown_bear → golden_retriever: 31
- brown_bear → mushroom: 28
- teapot → banana: 27
- banana → orange: 27
- sports_car → school_bus: 27
- teapot → golden_retriever: 25
- golden_retriever → brown_bear: 23
- golden_retriever → king_penguin: 22

## Feature Reuse

- phase2_golden_retriever_signature: used by 10 classes
- golden_fur_in_nature: used by 10 classes
- quadruped_like: used by 10 classes
- phase2_jellyfish_signature: used by 10 classes
- phase2_mushroom_signature: used by 10 classes
- blob_textured_interior: used by 10 classes
- phase2_teapot_signature: used by 10 classes
- distinct_background: used by 10 classes
- phase2_school_bus_signature: used by 10 classes
- horizontal_window_pattern: used by 10 classes
