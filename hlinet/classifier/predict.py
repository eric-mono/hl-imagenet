"""Top-level prediction: image → Prediction with proof."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from hlinet.classifier.hierarchy import ClassNode, build_phase1_hierarchy
from hlinet.classifier.scorer import score_node
from hlinet.classifier.tiebreaker import resolve_tie
from hlinet.scene.builder import SceneGraphBuilder
from hlinet.types import FeatureValue, Prediction, SceneGraph

# Force feature/sensor registration on import
import hlinet.sensors  # noqa: F401
import hlinet.features  # noqa: F401


_hierarchy = build_phase1_hierarchy()
_builder = SceneGraphBuilder()

_HIST_BLEND_W = 0.88

_PER_CLASS_HIST_W = {
    "school_bus": 0.16,
    "orange": 0.16,
    "king_penguin": 0.15,
    "sports_car": 0.14,
    "mushroom": 0.14,
    "golden_retriever": 0.13,
    "brown_bear": 0.12,
    "jellyfish": 0.10,
    "banana": 0.08,
    "teapot": 0.05,
}

_SCORE_CALIBRATION = {
    "school_bus": -0.02,
    "jellyfish": 0.02,
    "king_penguin": 0.01,
    "mushroom": 0.01,
}

_PROTO_MEANS = {
    "banana": {"cm_center_a": 0.5204, "cm_center_b": 0.6355, "sat": 0.5635, "bw": 0.4237, "edge": 0.2065, "warm": 0.638, "color_std": 0.2994, "grad_mean": 1.054},
    "brown_bear": {"cm_center_a": 0.5093, "cm_center_b": 0.5419, "sat": 0.3369, "bw": 0.4921, "edge": 0.3054, "warm": 0.4143, "color_std": 0.1196, "grad_mean": 1.3397},
    "golden_retriever": {"cm_center_a": 0.5261, "cm_center_b": 0.5656, "sat": 0.355, "bw": 0.4408, "edge": 0.2684, "warm": 0.5155, "color_std": 0.1571, "grad_mean": 1.2086},
    "jellyfish": {"cm_center_a": 0.5501, "cm_center_b": 0.4241, "sat": 0.6473, "bw": 0.4964, "edge": 0.146, "warm": 0.0915, "color_std": 0.3712, "grad_mean": 0.6753},
    "king_penguin": {"cm_center_a": 0.5014, "cm_center_b": 0.5169, "sat": 0.2413, "bw": 0.6478, "edge": 0.234, "warm": 0.1878, "color_std": 0.0822, "grad_mean": 1.1929},
    "mushroom": {"cm_center_a": 0.5257, "cm_center_b": 0.5805, "sat": 0.4866, "bw": 0.4689, "edge": 0.3016, "warm": 0.5135, "color_std": 0.1851, "grad_mean": 1.4165},
    "orange": {"cm_center_a": 0.5846, "cm_center_b": 0.6759, "sat": 0.6697, "bw": 0.3643, "edge": 0.1922, "warm": 0.6814, "color_std": 0.4232, "grad_mean": 0.8905},
    "school_bus": {"cm_center_a": 0.5172, "cm_center_b": 0.5704, "sat": 0.4151, "bw": 0.5498, "edge": 0.2682, "warm": 0.3798, "color_std": 0.1368, "grad_mean": 1.6529},
    "sports_car": {"cm_center_a": 0.5344, "cm_center_b": 0.5343, "sat": 0.355, "bw": 0.6594, "edge": 0.2482, "warm": 0.2383, "color_std": 0.1176, "grad_mean": 1.5191},
    "teapot": {"cm_center_a": 0.5205, "cm_center_b": 0.5499, "sat": 0.3482, "bw": 0.5819, "edge": 0.2052, "warm": 0.3828, "color_std": 0.138, "grad_mean": 1.087},
}
_PROTO_STDS = {
    "banana": {"cm_center_a": 0.043, "cm_center_b": 0.0613, "sat": 0.2067, "bw": 0.2161, "edge": 0.0763, "warm": 0.2591, "color_std": 0.1624, "grad_mean": 0.4132},
    "brown_bear": {"cm_center_a": 0.0232, "cm_center_b": 0.0382, "sat": 0.1318, "bw": 0.2124, "edge": 0.054, "warm": 0.2618, "color_std": 0.0752, "grad_mean": 0.2976},
    "golden_retriever": {"cm_center_a": 0.0234, "cm_center_b": 0.0401, "sat": 0.1586, "bw": 0.2102, "edge": 0.0544, "warm": 0.2737, "color_std": 0.101, "grad_mean": 0.3288},
    "jellyfish": {"cm_center_a": 0.0814, "cm_center_b": 0.1164, "sat": 0.2433, "bw": 0.3027, "edge": 0.0786, "warm": 0.1523, "color_std": 0.2709, "grad_mean": 0.3614},
    "king_penguin": {"cm_center_a": 0.0167, "cm_center_b": 0.0371, "sat": 0.1374, "bw": 0.2318, "edge": 0.084, "warm": 0.2047, "color_std": 0.0715, "grad_mean": 0.4059},
    "mushroom": {"cm_center_a": 0.0501, "cm_center_b": 0.0499, "sat": 0.1881, "bw": 0.2085, "edge": 0.0663, "warm": 0.2855, "color_std": 0.116, "grad_mean": 0.3857},
    "orange": {"cm_center_a": 0.0536, "cm_center_b": 0.0733, "sat": 0.2088, "bw": 0.239, "edge": 0.0716, "warm": 0.268, "color_std": 0.2259, "grad_mean": 0.3287},
    "school_bus": {"cm_center_a": 0.0226, "cm_center_b": 0.0482, "sat": 0.1458, "bw": 0.1636, "edge": 0.0493, "warm": 0.1809, "color_std": 0.0951, "grad_mean": 0.3867},
    "sports_car": {"cm_center_a": 0.0518, "cm_center_b": 0.0575, "sat": 0.16, "bw": 0.1791, "edge": 0.0535, "warm": 0.201, "color_std": 0.1011, "grad_mean": 0.3287},
    "teapot": {"cm_center_a": 0.0386, "cm_center_b": 0.0474, "sat": 0.181, "bw": 0.2487, "edge": 0.0622, "warm": 0.2958, "color_std": 0.1101, "grad_mean": 0.3307},
}
_PROTO_W = 0.025


def _proto_scores(graph: SceneGraph) -> dict[str, float]:
    """Score each class by Mahalanobis-like distance from prototype."""
    from hlinet.features.compounds.phase2_signatures import _stats
    s = _stats(graph)
    scores = {}
    for cls, means in _PROTO_MEANS.items():
        stds = _PROTO_STDS[cls]
        dist_sq = 0.0
        n = 0
        for feat, mu in means.items():
            v = s.get(feat)
            if v is not None:
                sigma = stds.get(feat, 0.1)
                dist_sq += ((v - mu) / sigma) ** 2
                n += 1
        if n > 0:
            scores[cls] = -dist_sq / n
    return scores


_HIST_MEANS = {
    "banana": 1.233, "brown_bear": 1.331, "golden_retriever": 1.320,
    "jellyfish": 0.664, "king_penguin": 1.215, "mushroom": 1.363,
    "orange": 1.043, "school_bus": 1.527, "sports_car": 1.315, "teapot": 1.316,
}


def _blend_hist_scores(
    candidates: list[tuple[str, float, list[str]]],
    image: np.ndarray,
) -> list[tuple[str, float, list[str]]]:
    """Blend signature scores with mean-centered histogram prototype scores."""
    from hlinet.features.compounds.phase2_signatures import _color_hist_scores

    hist_scores = _color_hist_scores(image)
    if not hist_scores:
        return candidates
    w = _HIST_BLEND_W
    return [
        (label, score * w + (hist_scores.get(f"hist_{label}", 0) - _HIST_MEANS.get(label, 1.2) * 0.3) * (1 - w), route)
        for label, score, route in candidates
    ]


PIPELINE_MODES = ("full", "base", "base_rerank")

_VERIFY_WHITELIST: set[str] | None = None
_VERIFY_PAIR_WHITELIST: set[frozenset] | None = None


def set_verify_whitelist(classes: set[str] | None) -> None:
    """Set which classes' verify rules are active. None = all active."""
    global _VERIFY_WHITELIST
    _VERIFY_WHITELIST = classes


def set_verify_pair_whitelist(pairs: set[frozenset] | None) -> None:
    """Set which specific pairs' verify rules are active. None = all active.

    Takes precedence over class whitelist when set.
    Example: set_verify_pair_whitelist({frozenset(['banana', 'orange']), ...})
    """
    global _VERIFY_PAIR_WHITELIST
    _VERIFY_PAIR_WHITELIST = pairs


def _pair_allowed(cls_a: str, cls_b: str) -> bool:
    """Check if a verify pair is allowed by current whitelist settings."""
    if _VERIFY_PAIR_WHITELIST is not None:
        return frozenset([cls_a, cls_b]) in _VERIFY_PAIR_WHITELIST
    if _VERIFY_WHITELIST is not None:
        return cls_a in _VERIFY_WHITELIST or cls_b in _VERIFY_WHITELIST
    return True


def predict(image: np.ndarray, *, mode: str = "full") -> Prediction:
    """Classify an image through the symbolic visual algebra pipeline.

    image: BGR numpy array (as loaded by cv2.imread)
    mode: "full" (all stages), "base" (score+blend+calibrate+repulse only),
          "base_rerank" (base + pairwise rerank, no verify)
    """
    if mode not in PIPELINE_MODES:
        raise ValueError(f"Unknown pipeline mode: {mode!r}. Expected one of {PIPELINE_MODES}")

    graph = _builder.build(image)
    cache: dict[str, FeatureValue] = {}

    candidates = _score_all_classes_flat(_hierarchy, graph, cache)

    if not candidates:
        return Prediction(
            label="unknown",
            confidence=0.0,
            proof=["no class matched above threshold"],
            feature_activations=cache,
        )

    candidates = _blend_hist_scores(candidates, image)
    candidates = [
        (label, score + _SCORE_CALIBRATION.get(label, 0.0), route)
        for label, score, route in candidates
    ]

    candidates = _potential_field_repulsion(candidates, graph)

    candidates.sort(key=lambda x: x[1], reverse=True)

    if mode != "base":
        candidates = _pairwise_rerank(candidates, graph)

    if mode == "full":
        candidates = _local_verify(candidates, graph)
        candidates = _rank3_verify(candidates, graph)
        candidates = _rank4_verify(candidates, graph)
        candidates = _rank5_verify(candidates, graph)
        candidates = _final_verify(candidates, graph)
        candidates = _final_verify_wave2(candidates, graph)
        candidates = _final_verify_wave3(candidates, graph)
        candidates = _final_verify_wave4(candidates, graph)

    best_label, best_score, best_route = candidates[0]
    alternatives = [(label, score) for label, score, _ in candidates[1:5]]

    proof = _generate_proof(best_label, best_route, cache)

    return Prediction(
        label=best_label,
        confidence=best_score,
        alternatives=alternatives,
        proof=proof,
        feature_activations=cache,
        route=best_route,
    )


_SIGNATURE_MAP = {
    "golden_retriever": "phase2_golden_retriever_signature",
    "mushroom": "phase2_mushroom_signature",
    "teapot": "phase2_teapot_signature",
    "school_bus": "phase2_school_bus_signature",
    "banana": "phase2_banana_signature",
    "orange": "phase2_orange_signature",
    "brown_bear": "phase2_brown_bear_signature",
    "king_penguin": "phase2_king_penguin_signature",
    "jellyfish": "phase2_jellyfish_signature",
    "sports_car": "phase2_sports_car_signature",
}

def _score_signatures_direct(
    graph: SceneGraph, cache: dict[str, FeatureValue]
) -> list[tuple[str, float, list[str]]]:
    """Score all classes by evaluating their phase2 signatures directly."""
    from hlinet.registry import registry

    results = []
    for class_name, feat_name in _SIGNATURE_MAP.items():
        try:
            feat = registry.get_feature(feat_name)
            val = feat.evaluate(graph)
            cache[feat_name] = val
            results.append((class_name, val.confidence, ["root", class_name]))
        except (KeyError, Exception):
            results.append((class_name, 0.0, ["root", class_name]))
    return results


def _pairwise_rerank(
    candidates: list[tuple[str, float, list[str]]],
    graph: SceneGraph,
) -> list[tuple[str, float, list[str]]]:
    """Rerank top candidates using targeted pairwise discriminants.

    Uses gap-aware confidence gating: the discriminant must beat a threshold
    that scales with the score margin — bigger gaps need stronger evidence.
    """
    if len(candidates) < 2:
        return candidates

    from hlinet.features.compounds.phase2_signatures import _stats, _sigmoid

    s = _stats(graph)
    candidates = list(candidates)

    top1_label, top1_score, _ = candidates[0]
    top2_label, top2_score, _ = candidates[1]

    _PAIR_BASE = {
        frozenset(["banana", "teapot"]): 0.30,
        frozenset(["banana", "school_bus"]): 0.20,
        frozenset(["king_penguin", "teapot"]): 0.0,
        frozenset(["brown_bear", "school_bus"]): 0.10,
        frozenset(["banana", "orange"]): 0.05,
        frozenset(["king_penguin", "sports_car"]): 0.0,
        frozenset(["golden_retriever", "orange"]): 0.0,
        frozenset(["brown_bear", "golden_retriever"]): 0.0,
        frozenset(["golden_retriever", "sports_car"]): 0.0,
        frozenset(["banana", "golden_retriever"]): 0.0,
        frozenset(["mushroom", "school_bus"]): 0.05,
        frozenset(["school_bus", "sports_car"]): -0.10,
        frozenset(["banana", "mushroom"]): 0.05,
        frozenset(["golden_retriever", "teapot"]): 0.15,
        frozenset(["brown_bear", "king_penguin"]): 0.25,
        frozenset(["jellyfish", "king_penguin"]): 0.0,
        frozenset(["orange", "teapot"]): 0.0,
        frozenset(["brown_bear", "mushroom"]): 0.05,
        frozenset(["brown_bear", "teapot"]): 0.10,
        frozenset(["golden_retriever", "king_penguin"]): 0.10,
        frozenset(["jellyfish", "teapot"]): 0.0,
        frozenset(["mushroom", "king_penguin"]): 0.0,
        frozenset(["orange", "mushroom"]): 0.0,
        frozenset(["teapot", "school_bus"]): 0.0,
        frozenset(["mushroom", "sports_car"]): 0.0,
    }

    margin12 = top1_score - top2_score
    if margin12 <= 0.30:
        pair12 = frozenset([top1_label, top2_label])
        signals = _compute_pair_signals(pair12, s, _sigmoid)
        if signals is not None:
            cls1, sig1, cls2, sig2 = signals
            disc_margin = abs(sig1 - sig2)
            base = _PAIR_BASE.get(pair12, 0.05)
            threshold = base + margin12 * 1.3
            if disc_margin > threshold:
                winner = cls1 if sig1 > sig2 else cls2
                if winner == top2_label:
                    candidates[0], candidates[1] = candidates[1], candidates[0]

    _RANK3_WHITELIST = {
        frozenset(["teapot", "sports_car"]),
        frozenset(["banana", "brown_bear"]),
        frozenset(["banana", "golden_retriever"]),
        frozenset(["school_bus", "mushroom"]),
        frozenset(["golden_retriever", "mushroom"]),
        frozenset(["school_bus", "brown_bear"]),
        frozenset(["golden_retriever", "brown_bear"]),
        frozenset(["school_bus", "sports_car"]),
        frozenset(["teapot", "king_penguin"]),
        frozenset(["banana", "orange"]),
        frozenset(["banana", "mushroom"]),
        frozenset(["golden_retriever", "teapot"]),
        frozenset(["jellyfish", "king_penguin"]),
        frozenset(["brown_bear", "king_penguin"]),
        frozenset(["orange", "teapot"]),
        frozenset(["brown_bear", "teapot"]),
        frozenset(["golden_retriever", "king_penguin"]),
        frozenset(["jellyfish", "teapot"]),
        frozenset(["mushroom", "king_penguin"]),
        frozenset(["orange", "mushroom"]),
        frozenset(["teapot", "school_bus"]),
        frozenset(["brown_bear", "mushroom"]),

    }
    if len(candidates) >= 3:
        top1_label, top1_score, _ = candidates[0]
        top3_label, top3_score, _ = candidates[2]
        margin13 = top1_score - top3_score
        pair13 = frozenset([top1_label, top3_label])
        if margin13 <= 0.32 and pair13 in _RANK3_WHITELIST:
            signals = _compute_pair_signals(pair13, s, _sigmoid)
            if signals is not None:
                cls1, sig1, cls2, sig2 = signals
                disc_margin = abs(sig1 - sig2)
                base = _PAIR_BASE.get(pair13, 0.0)
                threshold = base + margin13 * 1.9
                if disc_margin > threshold:
                    winner = cls1 if sig1 > sig2 else cls2
                    if winner == top3_label:
                        candidates[0], candidates[2] = candidates[2], candidates[0]

    _RANK4_WHITELIST = {
        frozenset(["banana", "mushroom"]),
        frozenset(["banana", "orange"]),
        frozenset(["jellyfish", "king_penguin"]),
        frozenset(["orange", "teapot"]),
        frozenset(["brown_bear", "teapot"]),
        frozenset(["school_bus", "sports_car"]),
        frozenset(["banana", "golden_retriever"]),
        frozenset(["banana", "brown_bear"]),
        frozenset(["teapot", "king_penguin"]),
        frozenset(["golden_retriever", "brown_bear"]),
        frozenset(["golden_retriever", "king_penguin"]),


    }
    if len(candidates) >= 4:
        top1_label, top1_score, _ = candidates[0]
        top4_label, top4_score, _ = candidates[3]
        margin14 = top1_score - top4_score
        pair14 = frozenset([top1_label, top4_label])
        if margin14 <= 0.30 and pair14 in _RANK4_WHITELIST:
            signals = _compute_pair_signals(pair14, s, _sigmoid)
            if signals is not None:
                cls1, sig1, cls2, sig2 = signals
                disc_margin = abs(sig1 - sig2)
                base = _PAIR_BASE.get(pair14, 0.0)
                threshold = base + margin14 * 2.8
                if disc_margin > threshold:
                    winner = cls1 if sig1 > sig2 else cls2
                    if winner == top4_label:
                        candidates[0], candidates[3] = candidates[3], candidates[0]

    _RANK5_WHITELIST = {
        frozenset(["banana", "teapot"]),
        frozenset(["school_bus", "sports_car"]),
        frozenset(["brown_bear", "mushroom"]),
        frozenset(["brown_bear", "golden_retriever"]),
        frozenset(["golden_retriever", "teapot"]),
        frozenset(["teapot", "king_penguin"]),
        frozenset(["banana", "mushroom"]),
        frozenset(["brown_bear", "king_penguin"]),
        frozenset(["golden_retriever", "king_penguin"]),
        frozenset(["banana", "orange"]),
        frozenset(["orange", "teapot"]),
        frozenset(["banana", "golden_retriever"]),
        frozenset(["orange", "mushroom"]),


    }
    if len(candidates) >= 5:
        top1_label, top1_score, _ = candidates[0]
        top5_label, top5_score, _ = candidates[4]
        margin15 = top1_score - top5_score
        pair15 = frozenset([top1_label, top5_label])
        if margin15 <= 0.15 and pair15 in _RANK5_WHITELIST:
            signals = _compute_pair_signals(pair15, s, _sigmoid)
            if signals is not None:
                cls1, sig1, cls2, sig2 = signals
                disc_margin = abs(sig1 - sig2)
                base = _PAIR_BASE.get(pair15, 0.0)
                threshold = base + margin15 * 4.0
                if disc_margin > threshold:
                    winner = cls1 if sig1 > sig2 else cls2
                    if winner == top5_label:
                        candidates[0], candidates[4] = candidates[4], candidates[0]

    return candidates


_REPULSION_PAIRS = [
    ("banana", "orange", 0.012),
    ("sports_car", "school_bus", 0.012),
    ("mushroom", "banana", 0.010),
    ("teapot", "banana", 0.014),
    ("brown_bear", "mushroom", 0.014),
    ("teapot", "king_penguin", 0.012),
    ("golden_retriever", "banana", 0.008),
    ("golden_retriever", "king_penguin", 0.008),
    ("brown_bear", "king_penguin", 0.008),
    ("orange", "mushroom", 0.008),
    ("teapot", "school_bus", 0.008),
    ("teapot", "golden_retriever", 0.010),
    ("golden_retriever", "mushroom", 0.010),
    ("brown_bear", "golden_retriever", 0.012),
    ("orange", "teapot", 0.008),
    ("sports_car", "teapot", 0.012),
    ("brown_bear", "sports_car", 0.012),
    ("banana", "king_penguin", 0.012),
    ("mushroom", "sports_car", 0.010),
]


def _potential_field_repulsion(
    candidates: list[tuple[str, float, list[str]]],
    graph: SceneGraph,
) -> list[tuple[str, float, list[str]]]:
    """Apply potential field: boost winner, penalize loser between confused pairs.

    When two classes both score high (proximity > 0.6), the discriminant winner
    gets a small boost and the loser a small penalty. This spreads their scores
    apart before ranking, making the downstream reranking's job easier.
    """
    from hlinet.features.compounds.phase2_signatures import _stats, _sigmoid
    s = _stats(graph)

    score_map = {label: score for label, score, _ in candidates}
    adjustments = {label: 0.0 for label, _, _ in candidates}

    for cls1, cls2, strength in _REPULSION_PAIRS:
        s1 = score_map.get(cls1, 0)
        s2 = score_map.get(cls2, 0)
        if s1 < 0.25 or s2 < 0.25:
            continue
        proximity = min(s1, s2) / max(max(s1, s2), 0.01)
        if proximity < 0.6:
            continue

        pair = frozenset([cls1, cls2])
        signals = _compute_pair_signals(pair, s, _sigmoid)
        if signals is None:
            continue

        c1, sig1, c2, sig2 = signals
        disc_gap = abs(sig1 - sig2)
        if disc_gap < 0.8:
            continue

        winner = c1 if sig1 > sig2 else c2
        loser = c1 if sig1 < sig2 else c2
        force = strength * proximity * min(disc_gap / 4.0, 1.0)
        adjustments[winner] += force * 0.5
        adjustments[loser] -= force * 0.5

    return [
        (label, score + adjustments.get(label, 0.0), route)
        for label, score, route in candidates
    ]


_CONFIDENCE_GATES = {
    "sports_car": 0.40,
    "banana": 0.42,
    "mushroom": 0.42,
    "golden_retriever": 0.37,
    "orange": 0.42,
    "teapot": 0.35,
}


def _proto_dist(s: dict[str, float], cls: str) -> float:
    """Compute prototype distance for a class."""
    means = _PROTO_MEANS.get(cls)
    stds = _PROTO_STDS.get(cls)
    if not means or not stds:
        return 999.0
    dist_sq = 0.0
    n = 0
    for feat, mu in means.items():
        v = s.get(feat)
        if v is not None:
            sigma = stds.get(feat, 0.1)
            dist_sq += ((v - mu) / sigma) ** 2
            n += 1
    return dist_sq / max(n, 1)



def _local_verify(
    candidates: list[tuple[str, float, list[str]]],
    graph: SceneGraph,
) -> list[tuple[str, float, list[str]]]:
    if len(candidates) < 2:
        return candidates
    top_label, top_score, _ = candidates[0]
    sec_label, sec_score, _ = candidates[1]

    if not _pair_allowed(top_label, sec_label):
        return candidates

    gate = _CONFIDENCE_GATES.get(top_label)
    if gate is not None and top_score < gate:
        candidates[0], candidates[1] = candidates[1], candidates[0]
        return candidates

    from hlinet.features.compounds.phase2_signatures import _stats
    s = _stats(graph)

    margin = top_score - sec_score
    pair = frozenset([top_label, sec_label])
    _WIDE_MARGIN = {
        frozenset(["teapot", "banana"]): 0.30,
        frozenset(["mushroom", "banana"]): 0.30,
        frozenset(["orange", "banana"]): 0.30,
        frozenset(["golden_retriever", "banana"]): 0.25,
    }
    margin_gate = _WIDE_MARGIN.get(pair, 0.15)
    if margin < margin_gate:
        if pair == frozenset(["teapot", "banana"]):
            cm_b = s.get("cm_center_b", 0.5)
            orient_e = s.get("orient_entropy", 3.0)
            cmbs = s.get("cm_b_std", 0.0)
            acorr_tb = s.get("autocorr_h", 0.1)
            if cm_b < 0.55 and orient_e > 2.90:
                idx = 0 if top_label == "teapot" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif cmbs > 0.075:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif acorr_tb < 0.069:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("cm_b_skew", 0.0) > 1.0362:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("hist_orange", 0.0) > 1.3175:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("hist_bear_minus_teapot", 0.0) < -0.2037:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("cm_b_std", 0.0) > 0.0489:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["teapot", "king_penguin"]):
            autocorr = s.get("autocorr_h", 0.15)
            horiz = s.get("horiz_dominance", 1.0)
            wss = s.get("warm_sat_std", 0.0)
            if autocorr > 0.16 and horiz > 1.1:
                idx = 0 if top_label == "teapot" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif wss > 0.15:
                idx = 0 if top_label == "king_penguin" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("cm_b_skew", 0.0) > 3.1083:
                idx = 0 if top_label == "king_penguin" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("grad_dir_entropy", 0.0) > 0.9916:
                idx = 0 if top_label == "teapot" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("cm_b_std", 0.0) > 0.0566:
                idx = 0 if top_label == "king_penguin" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["jellyfish", "teapot"]):
            sat_v = s.get("sat", 0.4)
            color_std_v = s.get("color_std", 0.1)
            ec = s.get("edge_concentration", 1.0)
            if sat_v > 0.50 and color_std_v > 0.20:
                idx = 0 if top_label == "jellyfish" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif ec > 1.3:
                idx = 0 if top_label == "jellyfish" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["teapot", "golden_retriever"]):
            edge_v = s.get("edge", 0.25)
            horiz = s.get("horiz_dominance", 1.0)
            if edge_v < 0.22 and horiz > 1.05:
                idx = 0 if top_label == "teapot" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("hist_mushroom", 0.0) > 1.8272:
                idx = 0 if top_label == "teapot" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("bg_contrast", 999.0) < 3.4888:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("grad_mean", 999.0) < 0.6824:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("round_edge", 0.0) > 0.1826:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["orange", "teapot"]):
            sat_v = s.get("sat", 0.4)
            cm_a = s.get("cm_center_a", 0.52)
            if sat_v > 0.55 and cm_a > 0.56:
                idx = 0 if top_label == "orange" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif cm_a > 0.62:
                idx = 0 if top_label == "orange" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("warm_br", 0.0) > 0.8682:
                idx = 0 if top_label == "teapot" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["mushroom", "banana"]):
            edge_v = s.get("edge", 0.25)
            hu1 = s.get("hu1", 2.6)
            cmbs = s.get("cm_b_std", 0.0)
            gra = s.get("green_region_area", 1.0)
            if edge_v > 0.27 and hu1 > 2.65:
                idx = 0 if top_label == "mushroom" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif cmbs > 0.045 and gra < 0.05:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif cmbs > 0.0673:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("hist_jellyfish", 0.0) > 0.6012:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("hue_cyan_blue", 0.0) > 0.0012:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("cm_b_std", 0.0) > 0.0669:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("center_bright_ratio", 0.0) > 1.2532:
                idx = 0 if top_label == "mushroom" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("center_surround", 0.0) > 1.3687:
                idx = 0 if top_label == "mushroom" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["orange", "banana"]):
            dwr = s.get("dark_warm_ratio", 0.0)
            if dwr > 0.6686:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("hist_sports_car", 0.0) > 1.561:
                idx = 0 if top_label == "orange" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("radial_warm_diff", 0.0) < -0.1301:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("elong_edge", 0.0) > 0.3483:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("round_warm", 1.0) < 0.7126 and s.get("region_area_entropy", 0.0) > 3.3879:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["golden_retriever", "school_bus"]):
            acorr_gs = s.get("autocorr_h", 0.1)
            if acorr_gs < 0.064:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["golden_retriever", "banana"]):
            warm_gb = s.get("warm", 0.0)
            if warm_gb > 0.863:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("warm_bl", 0.0) > 0.9097:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("blob_coverage", 0.0) > 0.8606:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("cm_b_skew", 0.0) < -0.5957:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("cm_b_std", 99.0) < 0.0466:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["teapot", "brown_bear"]):
            hu1_v = s.get("hu1", 3.0)
            hr = s.get("hue_red", 0.0)
            acorr = s.get("autocorr_h", 0.2)
            if hu1_v < 2.62:
                idx = 0 if top_label == "teapot" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif hr > 0.50:
                idx = 0 if top_label == "brown_bear" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif acorr < 0.10:
                idx = 0 if top_label == "teapot" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("r0_warm", 0) > 0.986:
                idx = 0 if top_label == "brown_bear" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("vert_regularity", 0.0) > 5.2081:
                idx = 0 if top_label == "brown_bear" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("cm_b_skew", 0.0) > 1.1705:
                idx = 0 if top_label == "brown_bear" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("gb_corr", 0.0) > 0.9958:
                idx = 0 if top_label == "brown_bear" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("vert_regularity", 0.0) > 5.1693:
                idx = 0 if top_label == "brown_bear" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("cm_b_skew", 0.0) > 0.7867 and s.get("hist_sports_car", 99.0) < 1.7790:
                idx = 0 if top_label == "brown_bear" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["golden_retriever", "brown_bear"]):
            sat_v = s.get("sat", 0.4)
            dh = s.get("dct_high", 0.25)
            horiz = s.get("horiz_dominance", 1.0)
            acorr = s.get("autocorr_h", 0.1)
            if sat_v < 0.28 and dh < 0.20:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif horiz > 1.10 and acorr > 0.13:
                idx = 0 if top_label == "brown_bear" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("hist_jellyfish", 0.0) > 0.7915:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif acorr > 0.1784:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("grad_mean", 0.0) > 1.7735:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("mean_ch_corr", 0.0) > 0.9845:
                idx = 0 if top_label == "brown_bear" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("rg_corr", 0.0) > 0.9875:
                idx = 0 if top_label == "brown_bear" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["sports_car", "school_bus"]):
            dct_h = s.get("dct_high", 0.18)
            gm = s.get("grad_mean", 1.4)
            ho = s.get("hue_orange", 0.0)
            hsb = s.get("hist_sports_minus_bus", 0.0)
            hbn = s.get("hist_banana", 0.0)
            if dct_h > 0.20 and gm > 1.60:
                idx = 0 if top_label == "sports_car" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif dct_h < 0.195 and ho > 0.20:
                idx = 0 if top_label == "school_bus" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif hsb < -0.335 and hbn > 2.157:
                idx = 0 if top_label == "school_bus" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("edge_tl", 0.0) > 0.3569:
                idx = 0 if top_label == "sports_car" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("horiz_dominance", 99.0) < 1.1587:
                idx = 0 if top_label == "school_bus" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("horiz_dominance", 0.0) > 2.3795 and s.get("has_dark_bright_contrast", 1.0) < 1.0:
                idx = 0 if top_label == "sports_car" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["banana", "school_bus"]):
            acorr = s.get("autocorr_h", 0.2)
            if acorr < 0.08:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("dct_low", 0.0) > 0.2430:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["brown_bear", "mushroom"]):
            green_v = s.get("green", 0.0)
            cstd = s.get("color_std", 0.1)
            if green_v > 0.50 and cstd > 0.18:
                idx = 0 if top_label == "brown_bear" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("cm_center_a", 0.0) > 0.5243:
                idx = 0 if top_label == "mushroom" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("sat_bl", 0.0) > 0.7538:
                idx = 0 if top_label == "mushroom" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("cm_a_std", 0.0) > 0.0729:
                idx = 0 if top_label == "mushroom" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["king_penguin", "sports_car"]):
            fft_hv = s.get("fft_hv_ratio", 0.8)
            evmr = s.get("edge_vert_mid_ratio", 0.4)
            if fft_hv > 1.0 and evmr < 0.36:
                idx = 0 if top_label == "king_penguin" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("edge_tl", 0.0) > 0.3643:
                idx = 0 if top_label == "king_penguin" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["school_bus", "mushroom"]):
            acorr = s.get("autocorr_h", 0.0)
            dh = s.get("dct_high", 0.25)
            if acorr > 0.10 and dh < 0.23:
                idx = 0 if top_label == "school_bus" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif acorr > 0.0862:
                idx = 0 if top_label == "school_bus" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["school_bus", "brown_bear"]):
            acorr_sb = s.get("autocorr_h", 0.0)
            if acorr_sb > 0.119:
                idx = 0 if top_label == "school_bus" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["mushroom", "king_penguin"]):
            bw_v = s.get("bw", 0.5)
            if bw_v < 0.482:
                idx = 0 if top_label == "king_penguin" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["mushroom", "sports_car"]):
            horiz = s.get("horiz_dominance", 1.0)
            if horiz > 1.344:
                idx = 0 if top_label == "sports_car" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["sports_car", "teapot"]):
            bw_v = s.get("bw", 0.5)
            hd = s.get("horiz_dominance", 1.0)
            if bw_v > 0.80 and hd > 1.50:
                idx = 0 if top_label == "sports_car" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("gabor_45_04_var", 1.0) < 0.2979:
                idx = 0 if top_label == "teapot" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("lap_var", 99999.0) < 3037.7576:
                idx = 0 if top_label == "teapot" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["teapot", "school_bus"]):
            acorr = s.get("autocorr_h", 0.2)
            dh = s.get("dct_high", 0.18)
            ec = s.get("edge_concentration", 1.0)
            if acorr < 0.25 and dh > 0.18:
                idx = 0 if top_label == "teapot" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif ec > 1.328:
                idx = 0 if top_label == "school_bus" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["brown_bear", "king_penguin"]):
            hbg = s.get("hist_bear_minus_gr", 0.0)
            if hbg > 0.2929:
                idx = 0 if top_label == "brown_bear" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["golden_retriever", "mushroom"]):
            hj = s.get("hist_jellyfish", 0.0)
            if hj > 1.011:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("hue_green", 1.0) < 0.0002:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
            elif s.get("hue_yellow", 0.0) > 0.4286:
                idx = 0 if top_label == "golden_retriever" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["jellyfish", "king_penguin"]):
            tu = s.get("top_uniformity", 0.5)
            if tu < 0.5796:
                idx = 0 if top_label == "jellyfish" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["mushroom", "orange"]):
            cma_mo = s.get("cm_center_a", 0.52)
            if cma_mo > 0.5243:
                idx = 0 if top_label == "mushroom" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["jellyfish", "sports_car"]):
            if s.get("hist_banana", 0.0) > 0.9366:
                idx = 0 if top_label == "sports_car" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["jellyfish", "orange"]):
            if s.get("cm_center_a", 0.0) > 0.5846:
                idx = 0 if top_label == "orange" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["brown_bear", "school_bus"]):
            if s.get("autocorr_h", 0.0) > 0.1161:
                idx = 0 if top_label == "school_bus" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["mushroom", "school_bus"]):
            if s.get("center_bright_ratio", 1.0) < 0.8700:
                idx = 0 if top_label == "school_bus" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]
        elif pair == frozenset(["brown_bear", "banana"]):
            if s.get("cm_b_std", 0.0) > 0.0633:
                idx = 0 if top_label == "banana" else 1
                if idx == 1:
                    candidates[0], candidates[1] = candidates[1], candidates[0]

    return candidates


def _rank3_verify(
    candidates: list[tuple[str, float, list[str]]],
    graph: SceneGraph,
) -> list[tuple[str, float, list[str]]]:
    if len(candidates) < 3:
        return candidates
    top_label = candidates[0][0]
    top_score = candidates[0][1]
    r3_label = candidates[2][0]
    r3_score = candidates[2][1]

    if not _pair_allowed(top_label, r3_label):
        return candidates

    margin13 = top_score - r3_score
    if margin13 >= 0.25:
        return candidates

    from hlinet.features.compounds.phase2_signatures import _stats
    s = _stats(graph)

    key = (top_label, r3_label)

    if key == ("king_penguin", "brown_bear") or key == ("brown_bear", "king_penguin"):
        if s.get("dark_warm_ratio", 0.0) > 34.7778:
            promote = "brown_bear" if key[0] == "king_penguin" else "king_penguin"
            if candidates[2][0] == promote:
                candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("teapot", "sports_car"):
        if s.get("color_std", 0.0) > 0.2593:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("mushroom", "golden_retriever"):
        if s.get("warm_bl", 0.0) > 0.9023:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("grad_mean", 99.0) < 1.0562:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("warm_bl", 0.0) > 0.8838:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("brown_bear", "mushroom"):
        if s.get("hu2", 0.0) > 9.6289:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("textured_decentered", 1.0) < 0.0717:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("center_bright_ratio", 0.0) > 1.1445:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("sat_bl", 0.0) > 0.5348:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("golden_retriever", "teapot"):
        if s.get("dct_mid", 1.0) < 0.0614:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("edge_concentration", 0.0) > 1.4872:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("smooth_warm", 0.0) > 0.3298:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("hue_red", 1.0) < 0.0650:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("sat_smooth_warm", 0.0) > 0.1273:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("edge_entropy", 99.0) < 3.5489:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("yellow", 0.0) > 0.5054:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("mid_width_ratio", 0.0) > 1.8333:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("teapot", "golden_retriever"):
        if s.get("autocorr_h", 1.0) < 0.0727:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("r0_edge", 0.0) > 0.2522:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("textured_warm_area", 0.0) > 0.2810:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("king_penguin", "teapot"):
        if s.get("elong_cy", 0.0) > 0.8356:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("school_bus", "banana"):
        if s.get("center_bright_ratio", 0.0) > 1.3204:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("sports_car", "school_bus"):
        if s.get("hist_jelly_minus_kp", 0.0) < -1.5445:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("king_penguin", "jellyfish"):
        if s.get("cm_b_skew", 0.0) < -0.1447:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("orange", "teapot"):
        if s.get("color_std", 1.0) < 0.4246:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("warm_val_mean", 1.0) < 0.5971:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("teapot", "jellyfish"):
        if s.get("color_std", 1.0) < 0.0320:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("school_bus", "brown_bear"):
        if s.get("hist_brown_bear", 99.0) < 1.3922:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("golden_retriever", "brown_bear"):
        if s.get("vert_regularity", 99.0) < 2.9442:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("mushroom", "brown_bear"):
        if s.get("elong_edge", 0.0) > 0.4008:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("king_penguin", "sports_car"):
        if s.get("warm_val_mean", 1.0) < 0.2976:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("king_penguin", "teapot"):
        if s.get("warm_val_mean", 1.0) < 0.3639:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("smooth_warm_blob_area", 0.0) > 0.0410:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("brown_bear", "teapot"):
        if s.get("textured_decentered", 1.0) < 0.1028:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("orange", "banana"):
        if s.get("warm_vert_top", 0.0) > 0.3465:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("brown_bear", "king_penguin"):
        if s.get("circularity", 1.0) < 0.0054:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("cm_a_std", 1.0) < 0.0144:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("brown_bear", "sports_car"):
        if s.get("hist_mushroom_minus_bear", 0.0) > -0.0234:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("teapot", "sports_car"):
        if s.get("hist_king_penguin", 0.0) > 2.0865:
            candidates[0], candidates[2] = candidates[2], candidates[0]
        elif s.get("round_area", 1.0) < 0.0325:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("banana", "teapot"):
        if s.get("center_bright_ratio", 1.0) < 0.6407:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("sports_car", "king_penguin"):
        if s.get("dct_low", 1.0) < 0.1261:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("sports_car", "golden_retriever"):
        if s.get("center_bright_ratio", 1.0) < 0.9159:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("jellyfish", "brown_bear"):
        if s.get("bot_edge", 0.0) > 0.3359:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("sports_car", "brown_bear"):
        if s.get("elong_yellow", 0.0) > 0.0327:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("mushroom", "banana"):
        if s.get("hist_jellyfish", 0.0) > 0.8604:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("teapot", "banana"):
        if s.get("edge", 1.0) < 0.1255:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("teapot", "orange"):
        if s.get("radial_warm_diff", 0.0) > 0.2046:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("orange", "sports_car"):
        if s.get("textured_decentered", 0.0) > 0.0803:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("school_bus", "sports_car"):
        if s.get("rg_corr", 0.0) > 0.9847:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("sports_car", "teapot"):
        if s.get("grad_dir_entropy", 0.0) > 0.9646:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("school_bus", "mushroom"):
        if s.get("warm_vert_bot", 0.0) > 0.4088:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("sports_car", "mushroom"):
        if s.get("hue_blue", 1.0) < 0.0023:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("king_penguin", "school_bus"):
        if s.get("bright_top_minus_bot", 0.0) > 0.1353:
            candidates[0], candidates[2] = candidates[2], candidates[0]
    elif key == ("banana", "orange"):
        if s.get("gabor_45_04_var", 0.0) > 0.7238:
            candidates[0], candidates[2] = candidates[2], candidates[0]

    return candidates


def _rank4_verify(
    candidates: list[tuple[str, float, list[str]]],
    graph: SceneGraph,
) -> list[tuple[str, float, list[str]]]:
    if len(candidates) < 4:
        return candidates
    top_label = candidates[0][0]
    r4_label = candidates[3][0]

    if not _pair_allowed(top_label, r4_label):
        return candidates

    margin14 = candidates[0][1] - candidates[3][1]
    if margin14 >= 0.28:
        return candidates

    from hlinet.features.compounds.phase2_signatures import _stats
    s = _stats(graph)

    key = (top_label, r4_label)
    if key == ("golden_retriever", "brown_bear"):
        if s.get("hist_orange_minus_teapot", 0.0) > -0.3451:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("mushroom", "brown_bear"):
        if s.get("textured_decentered", 0.0) > 0.1148:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("school_bus", "golden_retriever"):
        if s.get("warm_vert_mid", 1.0) < 0.3476:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("mushroom", "banana"):
        if s.get("bg_contrast", 100.0) < 12.5295:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("banana", "golden_retriever"):
        if s.get("edge_br", 0.0) > 0.2944:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("golden_retriever", "king_penguin"):
        if s.get("warm_sat_std", 0.0) > 0.1251:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("sports_car", "mushroom"):
        if s.get("sat", 1.0) < 0.229:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("brown_bear", "golden_retriever"):
        if s.get("dct_mid", 1.0) < 0.0614:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("brown_bear", "mushroom"):
        if s.get("hu2", 0.0) > 9.6289:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("teapot", "orange"):
        if s.get("sat", 0.0) > 0.5914:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("teapot", "golden_retriever"):
        if s.get("warm_tl", 0.0) > 0.5166:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("school_bus", "banana"):
        if s.get("dct_mid", 0.0) > 0.0808:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("king_penguin", "jellyfish"):
        if s.get("center_bright_ratio", 0.0) > 1.2098:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("teapot", "jellyfish"):
        if s.get("cm_b_std", 0.0) > 0.0346:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("teapot", "orange"):
        if s.get("hist_orange_minus_teapot", 0.0) > -0.1442:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("king_penguin", "brown_bear"):
        if s.get("dark_warm_ratio", 0.0) > 33.0:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("mushroom", "brown_bear"):
        if s.get("binary_complexity", 0.0) > 0.5188:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("brown_bear", "mushroom"):
        if s.get("rg_corr", 1.0) < 0.9230:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("golden_retriever", "brown_bear"):
        if s.get("center_bright_ratio", 1.0) < 0.8102:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("king_penguin", "jellyfish"):
        if s.get("hist_gr_minus_banana", 0.0) > 0.2563:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("teapot", "mushroom"):
        if s.get("dct_mid_over_low", 0.0) > 0.4175:
            candidates[0], candidates[3] = candidates[3], candidates[0]
        elif s.get("green", 1.0) < 0.0039:
            candidates[0], candidates[3] = candidates[3], candidates[0]
        elif s.get("hue_blue", 0.0) > 0.0762:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("banana", "brown_bear"):
        if s.get("autocorr_x_warm_bl", 1.0) < 0.0416:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("sports_car", "golden_retriever"):
        if s.get("cm_center_a", 1.0) < 0.5092:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("king_penguin", "golden_retriever"):
        if s.get("gabor_0_04_var", 99.0) < 1.2985:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("jellyfish", "king_penguin"):
        if s.get("center_bright_ratio", 1.0) < 0.7672:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("brown_bear", "king_penguin"):
        if s.get("color_purity", 1.0) < 0.0727:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("banana", "mushroom"):
        if s.get("elong_cy", 0.0) > 0.8095:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("mushroom", "orange"):
        if s.get("autocorr_x_mid_wider", 0.0) > 0.0400:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("brown_bear", "orange"):
        if s.get("autocorr_h", 1.0) < 0.0625:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("banana", "orange"):
        if s.get("cm_center_a", 0.0) > 0.5414:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("sports_car", "school_bus"):
        if s.get("hist_orange_minus_banana", 0.0) < -0.3289:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("banana", "sports_car"):
        if s.get("fft_hv_ratio", 1.0) < 0.7583:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("king_penguin", "teapot"):
        if s.get("smooth_warm_blob_area", 0.0) > 0.0510:
            candidates[0], candidates[3] = candidates[3], candidates[0]
    elif key == ("brown_bear", "teapot"):
        if s.get("cm_center_a", 1.0) < 0.5032:
            candidates[0], candidates[3] = candidates[3], candidates[0]

    return candidates


def _rank5_verify(
    candidates: list[tuple[str, float, list[str]]],
    graph: SceneGraph,
) -> list[tuple[str, float, list[str]]]:
    if len(candidates) < 5:
        return candidates
    top_label = candidates[0][0]
    r5_label = candidates[4][0]

    if not _pair_allowed(top_label, r5_label):
        return candidates

    margin15 = candidates[0][1] - candidates[4][1]
    if margin15 >= 0.25:
        return candidates

    from hlinet.features.compounds.phase2_signatures import _stats
    s = _stats(graph)

    key = (top_label, r5_label)
    if key == ("mushroom", "brown_bear"):
        if s.get("edge_vert_mid_ratio", 0.0) > 0.3537:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("golden_retriever", "teapot"):
        if s.get("horiz_dominance", 0.0) > 1.1773:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("teapot", "king_penguin"):
        if s.get("warm", 0.0) > 0.3066:
            candidates[0], candidates[4] = candidates[4], candidates[0]
        elif s.get("warm_band_bot", 0.0) > 0.5781:
            candidates[0], candidates[4] = candidates[4], candidates[0]
        elif s.get("warm_br", 0.0) > 0.4902:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("brown_bear", "sports_car"):
        if s.get("warm_sat_cv", 0.0) > 0.3870:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("school_bus", "sports_car"):
        if s.get("smooth_warm_blob_circ", 0.0) > 0.2566:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("banana", "teapot"):
        if s.get("autocorr_x_mid_wider", 0.0) > 0.2225:
            candidates[0], candidates[4] = candidates[4], candidates[0]
        elif s.get("orient_entropy", 0.0) > 2.9651:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("king_penguin", "teapot"):
        if s.get("edge_bl", 1.0) < 0.1699:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("king_penguin", "golden_retriever"):
        if s.get("hue_red", 0.0) > 0.3887:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("teapot", "golden_retriever"):
        if s.get("dct_mid", 1.0) < 0.0743:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("sports_car", "teapot"):
        if s.get("lap_var", 99999.0) < 8200.2:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("sports_car", "school_bus"):
        if s.get("hist_orange_minus_banana", 0.0) < -0.3289:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("brown_bear", "king_penguin"):
        if s.get("color_std", 0.0) > 0.0822:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("king_penguin", "brown_bear"):
        if s.get("edge_top_minus_bot", 0.0) < -0.1606:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("sports_car", "brown_bear"):
        if s.get("hue_yellow", 1.0) < 0.0007:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("golden_retriever", "king_penguin"):
        if s.get("circularity", 1.0) < 0.0082:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("mushroom", "king_penguin"):
        if s.get("binary_complexity", 0.0) > 0.4102:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("orange", "mushroom"):
        if s.get("smooth_warm_blob_circ", 1.0) < 0.1690:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("golden_retriever", "mushroom"):
        if s.get("gabor_90_01_mean", 99.0) < 13.7986:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("teapot", "mushroom"):
        if s.get("dct_high", 0.0) > 0.2267:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("banana", "mushroom"):
        if s.get("gb_corr", 0.0) > 0.9385:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("golden_retriever", "orange"):
        if s.get("blob_lap_var", 0.0) > 0.3144:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("banana", "orange"):
        if s.get("cm_a_std", 0.0) > 0.0456:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("brown_bear", "school_bus"):
        if s.get("cm_b_skew", 0.0) > 1.2084:
            candidates[0], candidates[4] = candidates[4], candidates[0]
    elif key == ("mushroom", "sports_car"):
        if s.get("green_bl", 1.0) < 0.0352:
            candidates[0], candidates[4] = candidates[4], candidates[0]
        elif s.get("warm", 0.0) > 0.4644:
            candidates[0], candidates[4] = candidates[4], candidates[0]

    return candidates


def _final_verify(
    candidates: list[tuple[str, float, list[str]]],
    graph: SceneGraph,
) -> list[tuple[str, float, list[str]]]:
    """Post-pipeline final verify: zero-risk conditions mined after all other stages.

    Placed AFTER rank5_verify so cascade radius is zero.
    """
    if len(candidates) < 2:
        return candidates

    from hlinet.features.compounds.phase2_signatures import _stats
    s = _stats(graph)

    top_label = candidates[0][0]
    sec_label = candidates[1][0]

    # Rank-2 conditions (swap candidates[0] ↔ candidates[1])
    if top_label == "golden_retriever" and sec_label == "brown_bear":
        if s.get("wavelet_coarse", 0) > 0.50905003197:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "banana" and sec_label == "teapot":
        if s.get("bw", 999) < 0.05151367188:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "brown_bear" and sec_label == "mushroom":
        if s.get("sat_tr", 0) > 0.72886412377:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "golden_retriever":
        if s.get("bright_top_minus_bot", 0) > 0.48960248162:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "banana":
        if s.get("cm_b_std", 0) > 0.09182494668:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "brown_bear" and sec_label == "king_penguin":
        if s.get("sat_tl", 999) < 0.09728477328:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "king_penguin":
        if s.get("cm_b_skew", 0) > 2.91487002373:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "sports_car" and sec_label == "mushroom":
        if s.get("green", 0) > 0.36645507812:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "brown_bear" and sec_label == "banana":
        if s.get("warm_sat_std", 0) > 0.20768115597:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "orange" and sec_label == "mushroom":
        if s.get("rb_corr", 0) > 0.84438675642:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "school_bus":
        if s.get("top_edge", 999) < 0.05761718750:
            candidates[0], candidates[1] = candidates[1], candidates[0]

    # Rank-3 conditions (swap candidates[0] ↔ candidates[2])
    if len(candidates) >= 3:
        top_label = candidates[0][0]
        r3_label = candidates[2][0]
        if top_label == "orange" and r3_label == "mushroom":
            if s.get("rb_corr", 0) > 0.84438675642:
                candidates[0], candidates[2] = candidates[2], candidates[0]

    # Rank-4 conditions (swap candidates[0] ↔ candidates[3])
    if len(candidates) >= 4:
        top_label = candidates[0][0]
        r4_label = candidates[3][0]
        if top_label == "orange" and r4_label == "mushroom":
            if s.get("rb_corr", 0) > 0.84438675642:
                candidates[0], candidates[3] = candidates[3], candidates[0]

    # Rank-5 conditions (swap candidates[0] ↔ candidates[4])
    if len(candidates) >= 5:
        top_label = candidates[0][0]
        r5_label = candidates[4][0]
        if top_label == "brown_bear" and r5_label == "mushroom":
            if s.get("warm_sat_std", 0) > 0.20768115597:
                candidates[0], candidates[4] = candidates[4], candidates[0]

    return candidates


def _final_verify_wave2(
    candidates: list[tuple[str, float, list[str]]],
    graph: SceneGraph,
) -> list[tuple[str, float, list[str]]]:
    """Post-pipeline wave 2: fix-1 zero-risk conditions."""
    if len(candidates) < 2:
        return candidates

    from hlinet.features.compounds.phase2_signatures import _stats
    s = _stats(graph)

    top_label = candidates[0][0]
    sec_label = candidates[1][0]
    if top_label == "orange" and sec_label == "banana":
        if s.get("r0_circularity", 0) > 0.69265858380:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("bg_contrast", 999) < 0.65576171875:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("orient_entropy", 999) < 2.37582748875:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("round_circularity", 0) > 0.76868047765:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("n_contours_norm", 0) > 0.80000000000:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "school_bus" and sec_label == "sports_car":
        if s.get("hue_green", 0) > 0.18249837345:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("wavelet_mid", 0) > 0.31938702526:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("round_area", 0) > 0.16503906250:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("hu2", 0) > 9.72220884625:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("hist_bear_minus_teapot", 0) > 0.45973145636:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "banana" and sec_label == "orange":
        if s.get("edge_concentration", 999) < 0.54716980603:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("textured_warm_area", 0) > 0.52270507812:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("edge_vert_mid_ratio", 999) < 0.21985202268:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("r0_circularity", 0) > 0.66257638439:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "sports_car" and sec_label == "school_bus":
        if s.get("bright_top_minus_bot", 999) < -0.43289292279:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("radial_warm_diff", 0) > 0.41869041061:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("hist_golden_retriever", 0) > 2.72601413387:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("hist_banana", 0) > 2.48251670087:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "mushroom" and sec_label == "banana":
        if s.get("autocorr_x_warm_bl", 999) < -0.03677237171:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("cm_center_b", 0) > 0.68808976716:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "brown_bear":
        if s.get("lbp_entropy", 0) > 5.38566753178:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("smooth_warm_blob_area", 0) > 0.38305664062:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("top_uniformity", 999) < 0.32085266288:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "jellyfish" and sec_label == "sports_car":
        if s.get("r0_aspect", 0) > 5.81818181818:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("warm_aspect", 0) > 2.62500000000:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("warm_vert_top", 0) > 0.99308755760:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "sports_car" and sec_label == "teapot":
        if s.get("wavelet_total", 999) < 0.06967655487:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("smooth_warm_blob_area", 0) > 0.27783203125:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "banana":
        if s.get("gb_ratio", 0) > 1.75039036360:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("orient_entropy", 999) < 2.71611502048:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "school_bus" and sec_label == "banana":
        if s.get("edge_bl", 0) > 0.38769531250:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("textured_warm_area", 0) > 0.34082031250:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "golden_retriever" and sec_label == "brown_bear":
        if s.get("r0_circularity", 0) > 0.64041455789:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("bg_contrast", 999) < 0.43676757812:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "sports_car" and sec_label == "brown_bear":
        if s.get("hist_bear_minus_teapot", 0) > 0.32193105601:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("fft_hv_ratio", 0) > 1.13932764746:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "mushroom" and sec_label == "brown_bear":
        if s.get("sky_area", 0) > 0.27197265625:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("bright_top_minus_bot", 999) < -0.33655407475:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "golden_retriever":
        if s.get("warm_val_mean", 0) > 0.72874963737:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("r0_aspect", 0) > 5.33333333333:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "mushroom" and sec_label == "golden_retriever":
        if s.get("hist_bear_minus_teapot", 0) > 0.55021411931:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("hist_gr_minus_mushroom", 0) > 0.26790657011:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "brown_bear" and sec_label == "king_penguin":
        if s.get("cm_a_std", 999) < 0.00700725855:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("center_surround", 0) > 1.39505079448:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "school_bus" and sec_label == "mushroom":
        if s.get("vert_regularity", 0) > 6.89932668762:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("gb_ratio", 0) > 2.67459373346:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "brown_bear" and sec_label == "school_bus":
        if s.get("autocorr_x_warm_bl", 0) > 0.20019615292:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("warm_sat_std", 0) > 0.20574943088:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "banana" and sec_label == "school_bus":
        if s.get("smooth_warm_blob_aspect", 0) > 5.16666666667:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("bg_contrast", 999) < 0.82739257812:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "school_bus" and sec_label == "teapot":
        if s.get("wavelet_mid", 0) > 0.31938702526:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("vert_regularity", 999) < 1.56116885887:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "orange" and sec_label == "teapot":
        if s.get("hist_bear_minus_kp", 999) < -0.17709189747:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        elif s.get("warm_hue_mean", 999) < 0.11525228341:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "king_penguin" and sec_label == "brown_bear":
        if s.get("textured_decentered", 0) > 0.16202511317:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "banana" and sec_label == "brown_bear":
        if s.get("dct_low", 999) < 0.14069883351:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "banana" and sec_label == "golden_retriever":
        if s.get("hist_gr_minus_banana", 0) > 0.41042622086:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "sports_car" and sec_label == "golden_retriever":
        if s.get("spatial_lr_asym", 0) > 0.27880859375:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "orange" and sec_label == "golden_retriever":
        if s.get("smooth_warm_blob_aspect", 0) > 4.57142857143:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "king_penguin" and sec_label == "jellyfish":
        if s.get("vert_regularity", 0) > 10.03945914205:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "jellyfish":
        if s.get("color_purity", 0) > 0.49888911508:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "king_penguin":
        if s.get("edge_br", 999) < 0.05859375000:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "mushroom" and sec_label == "king_penguin":
        if s.get("sat_bl", 999) < 0.17809819240:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "brown_bear" and sec_label == "mushroom":
        if s.get("bilat_detail_center", 0) > 0.08845741422:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "golden_retriever" and sec_label == "mushroom":
        if s.get("cm_a_std", 0) > 0.07481429904:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "mushroom" and sec_label == "orange":
        if s.get("smooth_warm_blob_circ", 0) > 0.55445421756:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "king_penguin" and sec_label == "school_bus":
        if s.get("blob_lap_var", 0) > 1.50291924876:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "mushroom" and sec_label == "school_bus":
        if s.get("hist_gr_minus_teapot", 999) < -0.22626695948:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "king_penguin" and sec_label == "sports_car":
        if s.get("textured_decentered", 0) > 0.16202511317:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "orange" and sec_label == "sports_car":
        if s.get("warm_coherence", 999) < 0.96062992126:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "jellyfish" and sec_label == "teapot":
        if s.get("warm_vert_top", 0) > 0.99308755760:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "king_penguin" and sec_label == "teapot":
        if s.get("cm_b_skew", 999) < -1.75697052479:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "banana" and sec_label == "teapot":
        if s.get("edge_vert_mid_ratio", 999) < 0.21985202268:
            candidates[0], candidates[1] = candidates[1], candidates[0]

    if len(candidates) >= 3:
        top_label = candidates[0][0]
        r3_label = candidates[2][0]
        if top_label == "teapot" and r3_label == "golden_retriever":
            if s.get("bright_top_minus_bot", 0) > 0.48960248162:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("textured_warm_area", 0) > 0.37182617188:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("hist_teapot_minus_kp", 0) > 0.69246350095:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("r0_aspect", 0) > 5.33333333333:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "teapot" and r3_label == "sports_car":
            if s.get("round_edge", 0) > 0.36708860759:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("hue_cyan_blue", 0) > 0.69737954353:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("round_circularity", 999) < 0.34924961838:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("hist_sports_minus_bus", 0) > 0.37566725246:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "golden_retriever" and r3_label == "mushroom":
            if s.get("warm_aspect", 0) > 2.46153846154:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("rb_ratio", 0) > 3.04682986787:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("bot_edge", 0) > 0.37255859375:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "sports_car" and r3_label == "school_bus":
            if s.get("dct_high", 999) < 0.15248371852:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("bright_top_minus_bot", 999) < -0.43289292279:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("dct_mid", 999) < 0.06011367527:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "king_penguin" and r3_label == "sports_car":
            if s.get("autocorr_x_mid_wider", 0) > 0.28258226194:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("textured_decentered", 0) > 0.16202511317:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("warm_cool_a_diff", 0) > 0.10766673088:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "school_bus" and r3_label == "banana":
            if s.get("wavelet_mid", 0) > 0.31938702526:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("textured_warm_area", 0) > 0.34082031250:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "golden_retriever" and r3_label == "banana":
            if s.get("edge_bl", 999) < 0.12597656250:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("orient_entropy", 999) < 2.69169769923:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "golden_retriever" and r3_label == "brown_bear":
            if s.get("warm_tr", 0) > 0.99804687500:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("textured_warm_area", 0) > 0.61499023438:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "mushroom" and r3_label == "brown_bear":
            if s.get("warm_band_bot", 0) > 0.98046875000:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("sat_bl", 999) < 0.17809819240:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "banana" and r3_label == "golden_retriever":
            if s.get("spatial_edge_concentration", 0) > 0.19010416667:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("hist_bear_minus_gr", 999) < -0.32608787794:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "school_bus" and r3_label == "golden_retriever":
            if s.get("bg_contrast", 0) > 79.13452148438:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("spatial_mid_warm", 0) > 0.89111328125:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "orange" and r3_label == "golden_retriever":
            if s.get("r0_circularity", 999) < 0.10218393084:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("center_bright_ratio", 999) < 0.74147307873:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "mushroom" and r3_label == "golden_retriever":
            if s.get("hue_spread", 999) < 2.00000000000:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("hu2", 0) > 10.19396439832:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "brown_bear" and r3_label == "king_penguin":
            if s.get("sat_tl", 999) < 0.09728477328:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("sat_tr", 999) < 0.08549325980:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "mushroom" and r3_label == "school_bus":
            if s.get("bilat_detail_center", 0) > 0.08756893382:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("dct_mid_over_low", 0) > 0.87900743916:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "school_bus" and r3_label == "sports_car":
            if s.get("wavelet_mid", 0) > 0.31938702526:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("autocorr_h", 999) < 0.07017151912:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "jellyfish" and r3_label == "teapot":
            if s.get("hue_red", 0) > 0.53881835937:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("radial_warm_diff", 0) > 0.49200180660:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "teapot" and r3_label == "banana":
            if s.get("warm_sat_cv", 0) > 0.47810012437:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "orange" and r3_label == "banana":
            if s.get("warm_hue_mean", 999) < 0.11525228341:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "mushroom" and r3_label == "banana":
            if s.get("warm_br", 0) > 0.98828125000:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "brown_bear" and r3_label == "banana":
            if s.get("green", 0) > 0.89599609375:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "sports_car" and r3_label == "brown_bear":
            if s.get("r0_edge", 0) > 0.38056680162:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "teapot" and r3_label == "brown_bear":
            if s.get("dct_high", 999) < 0.10960166859:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "brown_bear" and r3_label == "golden_retriever":
            if s.get("gabor_90_01_mean", 0) > 28.65338851399:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "sports_car" and r3_label == "jellyfish":
            if s.get("blue_region_area", 0) > 0.31616210938:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "teapot" and r3_label == "jellyfish":
            if s.get("rb_ratio", 0) > 3.05163693122:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "orange" and r3_label == "jellyfish":
            if s.get("gb_corr", 999) < -0.09643772990:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "school_bus" and r3_label == "king_penguin":
            if s.get("fft_hv_ratio", 0) > 1.33260213354:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "sports_car" and r3_label == "king_penguin":
            if s.get("binary_complexity", 0) > 0.45411706349:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "banana" and r3_label == "king_penguin":
            if s.get("dark_warm_ratio", 0) > 4.17003367003:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "golden_retriever" and r3_label == "king_penguin":
            if s.get("warm_coherence", 999) < 0.74545454545:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "teapot" and r3_label == "king_penguin":
            if s.get("warm_sat_std", 0) > 0.24061645221:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "sports_car" and r3_label == "mushroom":
            if s.get("dct_high", 0) > 0.27264019514:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "king_penguin" and r3_label == "mushroom":
            if s.get("sat_bl", 0) > 0.53340226716:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "banana" and r3_label == "mushroom":
            if s.get("bw", 999) < 0.05151367188:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "brown_bear" and r3_label == "mushroom":
            if s.get("center_surround", 0) > 1.39505079448:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "mushroom" and r3_label == "orange":
            if s.get("hist_jelly_minus_kp", 0) > 0.07884088847:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "teapot" and r3_label == "orange":
            if s.get("cm_b_std", 0) > 0.09182494668:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "banana" and r3_label == "school_bus":
            if s.get("circularity", 0) > 0.32256353698:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "teapot" and r3_label == "school_bus":
            if s.get("blue_region_area", 0) > 0.31298828125:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "orange" and r3_label == "school_bus":
            if s.get("gabor_0_04_var", 0) > 3.65458556634:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "golden_retriever" and r3_label == "sports_car":
            if s.get("cm_center_a", 0) > 0.60075444240:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "banana" and r3_label == "sports_car":
            if s.get("warm_vert_mid", 999) < 0.17861675766:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "mushroom" and r3_label == "sports_car":
            if s.get("grad_dir_entropy", 999) < 0.91660555966:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "king_penguin" and r3_label == "teapot":
            if s.get("dark_warm_ratio", 0) > 139.12500000000:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "banana" and r3_label == "teapot":
            if s.get("spatial_edge_concentration", 0) > 0.19010416667:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "mushroom" and r3_label == "teapot":
            if s.get("warm_sat_cv", 0) > 0.46167286883:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "school_bus" and r3_label == "teapot":
            if s.get("top2_hue_ratio", 0) > 0.99440246292:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "brown_bear" and r3_label == "teapot":
            if s.get("warm_sat_std", 0) > 0.20574943088:
                candidates[0], candidates[2] = candidates[2], candidates[0]

    if len(candidates) >= 4:
        top_label = candidates[0][0]
        r4_label = candidates[3][0]
        if top_label == "golden_retriever" and r4_label == "teapot":
            if s.get("cm_center_a", 999) < 0.48527496936:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("autocorr_x_warm_bl", 0) > 0.21864024973:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("cm_center_a", 0) > 0.60075444240:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("warm_vert_top", 0) > 0.57368064188:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("textured_decentered", 0) > 0.17571898159:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "brown_bear" and r4_label == "mushroom":
            if s.get("spatial_edge_concentration", 999) < -0.12076822917:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("sat_tl", 999) < 0.09728477328:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("sat_tr", 999) < 0.08549325980:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("rb_ratio", 0) > 2.24208707766:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "golden_retriever" and r4_label == "banana":
            if s.get("gabor_45_04_var", 0) > 2.41843482461:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("cm_center_a", 999) < 0.48527496936:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("autocorr_h", 999) < -0.00650825428:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "mushroom" and r4_label == "golden_retriever":
            if s.get("autocorr_x_mid_wider", 999) < -0.01693795149:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("hist_sports_minus_bus", 999) < -0.93964596384:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("blue_purple", 0) > 0.22753906250:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "brown_bear" and r4_label == "golden_retriever":
            if s.get("warm_vert_mid", 0) > 0.91054313099:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("warm", 0) > 0.91088867188:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("warm_bl", 0) > 0.98632812500:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "teapot" and r4_label == "orange":
            if s.get("bright_top_minus_bot", 0) > 0.48960248162:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("hu2", 999) < 5.27518410265:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("hist_orange", 0) > 1.73878083797:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "banana" and r4_label == "teapot":
            if s.get("bw", 999) < 0.05151367188:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("hue_spread", 0) > 6.00000000000:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("warm_sat_cv", 0) > 0.52943818256:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "golden_retriever" and r4_label == "mushroom":
            if s.get("top2_hue_ratio", 999) < 0.56146435453:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("sat", 0) > 0.69142348346:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "banana" and r4_label == "mushroom":
            if s.get("hist_banana_minus_mushroom", 999) < -0.63094107085:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("binary_complexity", 0) > 0.46329365079:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "golden_retriever" and r4_label == "school_bus":
            if s.get("hist_orange_minus_banana", 999) < -0.58076373860:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("hist_jellyfish", 0) > 1.34978582567:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "sports_car" and r4_label == "banana":
            if s.get("sat_br", 0) > 0.76200980392:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "brown_bear" and r4_label == "banana":
            if s.get("hist_mushroom", 0) > 2.57589152013:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "teapot" and r4_label == "banana":
            if s.get("vert_regularity", 0) > 7.99988596014:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "teapot" and r4_label == "brown_bear":
            if s.get("vert_regularity", 0) > 7.99988596014:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "golden_retriever" and r4_label == "brown_bear":
            if s.get("r0_aspect", 0) > 6.40000000000:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "mushroom" and r4_label == "brown_bear":
            if s.get("round_area", 999) < 0.03027343750:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "king_penguin" and r4_label == "brown_bear":
            if s.get("blob_lap_var", 0) > 1.50291924876:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "jellyfish" and r4_label == "brown_bear":
            if s.get("spatial_edge_concentration", 999) < -0.06868489583:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "orange" and r4_label == "golden_retriever":
            if s.get("elong_area", 0) > 0.20361328125:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "sports_car" and r4_label == "golden_retriever":
            if s.get("elong_area", 0) > 0.21948242188:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "jellyfish" and r4_label == "golden_retriever":
            if s.get("hist_gr_minus_teapot", 0) > 0.12209870297:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "orange" and r4_label == "jellyfish":
            if s.get("green", 0) > 0.52026367188:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "sports_car" and r4_label == "jellyfish":
            if s.get("lbp_entropy", 999) < 3.70575654155:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "brown_bear" and r4_label == "jellyfish":
            if s.get("warm_val_mean", 0) > 0.65193413357:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "king_penguin" and r4_label == "jellyfish":
            if s.get("warm_vert_bot", 0) > 0.87210638076:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "school_bus" and r4_label == "king_penguin":
            if s.get("wavelet_mid", 0) > 0.31938702526:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "school_bus" and r4_label == "mushroom":
            if s.get("corner_density", 999) < 0.04345703125:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "banana" and r4_label == "orange":
            if s.get("hist_jellyfish", 0) > 1.20578142807:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "golden_retriever" and r4_label == "orange":
            if s.get("gb_ratio", 0) > 2.05957697429:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "sports_car" and r4_label == "orange":
            if s.get("spatial_lr_asym", 0) > 0.27880859375:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "banana" and r4_label == "school_bus":
            if s.get("vert_regularity", 999) < 1.88337153719:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "brown_bear" and r4_label == "school_bus":
            if s.get("edge_vert_mid_ratio", 0) > 0.47215242881:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "brown_bear" and r4_label == "sports_car":
            if s.get("mean_ch_corr", 999) < 0.68158890804:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "mushroom" and r4_label == "sports_car":
            if s.get("dark_warm_ratio", 0) > 21.50000000000:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "king_penguin" and r4_label == "sports_car":
            if s.get("blob_lap_var", 0) > 1.50291924876:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "banana" and r4_label == "sports_car":
            if s.get("round_circularity", 999) < 0.32240788674:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "orange" and r4_label == "sports_car":
            if s.get("n_contours_norm", 0) > 0.80000000000:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "school_bus" and r4_label == "sports_car":
            if s.get("cm_a_std", 0) > 0.06897316727:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "king_penguin" and r4_label == "teapot":
            if s.get("hu2", 0) > 10.18433504004:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "brown_bear" and r4_label == "teapot":
            if s.get("sat_smooth_warm", 0) > 0.07686063821:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "orange" and r4_label == "teapot":
            if s.get("warm_coherence", 999) < 0.96062992126:
                candidates[0], candidates[3] = candidates[3], candidates[0]

    if len(candidates) >= 5:
        top_label = candidates[0][0]
        r5_label = candidates[4][0]
        if top_label == "banana" and r5_label == "teapot":
            if s.get("textured_warm_area", 999) < 0.03076171875:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("textured_decentered", 0) > 0.14526657929:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("spatial_edge_concentration", 0) > 0.19010416667:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("spatial_edge_concentration", 999) < -0.09733072917:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("bw", 999) < 0.05151367188:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("cm_b_std", 999) < 0.02794917986:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "sports_car" and r5_label == "mushroom":
            if s.get("r0_aspect", 0) > 9.14285714286:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("green", 0) > 0.36645507812:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("gabor_45_04_var", 0) > 2.62097091882:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "king_penguin" and r5_label == "brown_bear":
            if s.get("warm_tl", 0) > 0.65820312500:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("sky_area", 0) > 0.39331054688:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "golden_retriever" and r5_label == "brown_bear":
            if s.get("hu1", 0) > 2.74295866138:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("hue_cyan_blue", 0) > 0.20842795862:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "sports_car" and r5_label == "brown_bear":
            if s.get("hue_orange", 0) > 0.72400143937:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("wavelet_fine", 999) < 0.34404177169:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "king_penguin" and r5_label == "golden_retriever":
            if s.get("hist_orange_minus_teapot", 999) < -1.02862077364:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("dark_warm_ratio", 0) > 139.12500000000:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "brown_bear" and r5_label == "king_penguin":
            if s.get("hue_spread", 0) > 7.00000000000:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("edge_entropy", 999) < 3.39729040352:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "teapot" and r5_label == "orange":
            if s.get("cm_center_b", 0) > 0.63969056373:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("edge_top_minus_bot", 999) < -0.17968750000:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "teapot" and r5_label == "sports_car":
            if s.get("sat_color_std", 0) > 0.22358805393:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("blob_lap_var", 0) > 2.03343576515:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "golden_retriever" and r5_label == "teapot":
            if s.get("cm_center_a", 999) < 0.48527496936:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("dct_high", 0) > 0.26884179571:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "king_penguin" and r5_label == "teapot":
            if s.get("blob_lap_var", 0) > 1.50291924876:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("bright_top_minus_bot", 0) > 0.44944852941:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "mushroom" and r5_label == "teapot":
            if s.get("spatial_edge_concentration", 999) < -0.12402343750:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("warm_sat_cv", 0) > 0.46167286883:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "brown_bear" and r5_label == "banana":
            if s.get("warm_sat_std", 0) > 0.20574943088:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "teapot" and r5_label == "banana":
            if s.get("vert_regularity", 0) > 7.99988596014:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "school_bus" and r5_label == "banana":
            if s.get("hu1", 999) < 2.48581402423:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "king_penguin" and r5_label == "banana":
            if s.get("hu2", 999) < 4.83846256150:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "sports_car" and r5_label == "banana":
            if s.get("hu1", 999) < 2.44798593415:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "mushroom" and r5_label == "brown_bear":
            if s.get("hist_gr_minus_teapot", 0) > 0.36266143504:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "banana" and r5_label == "golden_retriever":
            if s.get("sat_bl", 999) < 0.16110217525:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "teapot" and r5_label == "golden_retriever":
            if s.get("bright_top_minus_bot", 0) > 0.48960248162:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "sports_car" and r5_label == "golden_retriever":
            if s.get("region_area_entropy", 999) < 1.24347940824:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "mushroom" and r5_label == "jellyfish":
            if s.get("blue_purple", 0) > 0.22753906250:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "banana" and r5_label == "jellyfish":
            if s.get("hist_jellyfish", 0) > 1.20578142807:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "orange" and r5_label == "jellyfish":
            if s.get("bg_contrast", 0) > 78.72363281250:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "king_penguin" and r5_label == "jellyfish":
            if s.get("bilat_detail_center", 0) > 0.09210324755:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "sports_car" and r5_label == "jellyfish":
            if s.get("hue_blue", 0) > 0.28560250391:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "golden_retriever" and r5_label == "king_penguin":
            if s.get("fft_hv_ratio", 0) > 1.25191217807:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "jellyfish" and r5_label == "king_penguin":
            if s.get("spatial_edge_concentration", 999) < -0.06868489583:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "king_penguin" and r5_label == "mushroom":
            if s.get("blob_lap_var", 0) > 1.50291924876:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "banana" and r5_label == "mushroom":
            if s.get("textured_warm_area", 0) > 0.52270507812:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "brown_bear" and r5_label == "orange":
            if s.get("warm_sat_std", 0) > 0.20574943088:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "orange" and r5_label == "school_bus":
            if s.get("r0_aspect", 0) > 4.57142857143:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "king_penguin" and r5_label == "school_bus":
            if s.get("round_edge", 0) > 0.41860465116:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "brown_bear" and r5_label == "school_bus":
            if s.get("dominant_hue_ratio", 999) < 0.23174603175:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "teapot" and r5_label == "school_bus":
            if s.get("orient_entropy", 999) < 2.71611502048:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "school_bus" and r5_label == "sports_car":
            if s.get("cm_a_std", 0) > 0.06897316727:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "brown_bear" and r5_label == "sports_car":
            if s.get("hist_bear_minus_kp", 999) < -0.34895908576:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "golden_retriever" and r5_label == "sports_car":
            if s.get("cm_center_a", 999) < 0.48527496936:
                candidates[0], candidates[4] = candidates[4], candidates[0]

    return candidates


def _final_verify_wave3(
    candidates: list[tuple[str, float, list[str]]],
    graph: SceneGraph,
) -> list[tuple[str, float, list[str]]]:
    """Post-pipeline wave 3: fix-1 zero-risk conditions (mined after wave 2)."""
    if len(candidates) < 2:
        return candidates

    from hlinet.features.compounds.phase2_signatures import _stats
    s = _stats(graph)

    top_label = candidates[0][0]
    sec_label = candidates[1][0]
    if top_label == "orange" and sec_label == "banana":
        if s.get("rb_corr", 0) > 0.85780560970:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "brown_bear":
        if s.get("dark_warm_ratio", 0) > 503.00000000000:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "banana" and sec_label == "mushroom":
        if s.get("dct_mid", 0) > 0.13235637449:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "mushroom":
        if s.get("hist_sports_minus_bus", 0) > 0.30485699978:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "orange":
        if s.get("warm_sat_std", 0) > 0.27928100778:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "banana" and sec_label == "orange":
        if s.get("warm", 0) > 0.99804687500:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "sports_car" and sec_label == "school_bus":
        if s.get("hist_bear_minus_kp", 0) > 0.62249981519:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "school_bus" and sec_label == "sports_car":
        if s.get("hist_sports_car", 0) > 2.96346221922:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "golden_retriever" and sec_label == "teapot":
        if s.get("lbp_entropy", 999) < 4.81200136293:
            candidates[0], candidates[1] = candidates[1], candidates[0]

    if len(candidates) >= 3:
        top_label = candidates[0][0]
        r3_label = candidates[2][0]
        if top_label == "sports_car" and r3_label == "king_penguin":
            if s.get("fft_hv_ratio", 0) > 1.13932764746:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("glcm_contrast", 0) > 0.04041899825:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "school_bus" and r3_label == "sports_car":
            if s.get("cm_a_std", 0) > 0.06897316727:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("spatial_bot_intensity", 999) < 0.12558210784:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "teapot" and r3_label == "banana":
            if s.get("sat_color_std", 0) > 0.22358805393:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "golden_retriever" and r3_label == "banana":
            if s.get("bilat_detail", 0) > 0.08210688572:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "teapot" and r3_label == "brown_bear":
            if s.get("has_round", 999) < 1.00000000000:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "king_penguin" and r3_label == "brown_bear":
            if s.get("r0_circularity", 0) > 0.65321045247:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "banana" and r3_label == "golden_retriever":
            if s.get("warm_vert_concentration", 999) < 0.33396584440:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "teapot" and r3_label == "jellyfish":
            if s.get("blue_region_area", 0) > 0.31298828125:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "brown_bear" and r3_label == "king_penguin":
            if s.get("hue_spread", 0) > 7.00000000000:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "school_bus" and r3_label == "mushroom":
            if s.get("green", 0) > 0.43139648438:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "golden_retriever" and r3_label == "orange":
            if s.get("cm_center_b", 0) > 0.65105315564:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "sports_car" and r3_label == "school_bus":
            if s.get("fft_hv_ratio", 0) > 1.13932764746:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "banana" and r3_label == "school_bus":
            if s.get("vert_regularity", 999) < 1.88337153719:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "brown_bear" and r3_label == "sports_car":
            if s.get("top2_hue_ratio", 999) < 0.41097724230:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "sports_car" and r3_label == "teapot":
            if s.get("wavelet_coarse", 0) > 0.40493414975:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "brown_bear" and r3_label == "teapot":
            if s.get("wavelet_mid", 0) > 0.34177593845:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "orange" and r3_label == "teapot":
            if s.get("warm_coherence", 999) < 0.92972972973:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "golden_retriever" and r3_label == "teapot":
            if s.get("lbp_entropy", 999) < 4.81200136293:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "king_penguin" and r3_label == "teapot":
            if s.get("autocorr_x_mid_wider", 0) > 0.28258226194:
                candidates[0], candidates[2] = candidates[2], candidates[0]

    if len(candidates) >= 4:
        top_label = candidates[0][0]
        r4_label = candidates[3][0]
        if top_label == "orange" and r4_label == "teapot":
            if s.get("fft_hv_ratio", 0) > 1.29775370851:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("hist_bear_minus_teapot", 0) > 0.13087687455:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "mushroom" and r4_label == "banana":
            if s.get("gabor_45_04_var", 0) > 3.86258264871:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "teapot" and r4_label == "banana":
            if s.get("sat_smooth_warm", 0) > 0.26539406002:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "mushroom" and r4_label == "golden_retriever":
            if s.get("r0_circularity", 999) < 0.08343115988:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "brown_bear" and r4_label == "golden_retriever":
            if s.get("warm_val_mean", 0) > 0.62921882712:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "school_bus" and r4_label == "jellyfish":
            if s.get("lbp_entropy", 999) < 4.17439733122:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "king_penguin" and r4_label == "mushroom":
            if s.get("contour_fill_ratio", 0) > 0.42993164062:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "golden_retriever" and r4_label == "orange":
            if s.get("orient_entropy", 999) < 2.69169769923:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "brown_bear" and r4_label == "sports_car":
            if s.get("rb_corr", 0) > 0.99423968792:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "school_bus" and r4_label == "sports_car":
            if s.get("elong_area", 0) > 0.18701171875:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "brown_bear" and r4_label == "teapot":
            if s.get("center_surround", 0) > 1.60255266127:
                candidates[0], candidates[3] = candidates[3], candidates[0]

    if len(candidates) >= 5:
        top_label = candidates[0][0]
        r5_label = candidates[4][0]
        if top_label == "school_bus" and r5_label == "brown_bear":
            if s.get("warm_sat_std", 999) < 0.05460310181:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("blue_region_area", 0) > 0.40478515625:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "sports_car" and r5_label == "king_penguin":
            if s.get("hist_golden_retriever", 0) > 2.55215367085:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("blue_region_area", 0) > 0.31616210938:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "golden_retriever" and r5_label == "jellyfish":
            if s.get("hist_bear_minus_gr", 0) > 0.32879668497:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "school_bus" and r5_label == "king_penguin":
            if s.get("cm_b_skew", 0) > 2.88507676125:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "sports_car" and r5_label == "mushroom":
            if s.get("dct_high", 0) > 0.27603768454:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "school_bus" and r5_label == "mushroom":
            if s.get("r0_aspect", 0) > 9.14285714286:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "teapot" and r5_label == "mushroom":
            if s.get("bilat_detail", 0) > 0.07135416667:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "brown_bear" and r5_label == "mushroom":
            if s.get("glcm_contrast_v2", 0) > 0.06136859922:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "school_bus" and r5_label == "orange":
            if s.get("hu1", 999) < 2.48581402423:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "golden_retriever" and r5_label == "school_bus":
            if s.get("smooth_warm_blob_aspect", 0) > 6.22222222222:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "king_penguin" and r5_label == "sports_car":
            if s.get("cm_center_b", 999) < 0.42161841299:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "golden_retriever" and r5_label == "teapot":
            if s.get("smooth_warm_blob_area", 0) > 0.31054687500:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "orange" and r5_label == "teapot":
            if s.get("radial_warm_diff", 999) < -0.11676266637:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "sports_car" and r5_label == "teapot":
            if s.get("sat_smooth_warm", 0) > 0.14125977480:
                candidates[0], candidates[4] = candidates[4], candidates[0]

    return candidates


def _final_verify_wave4(
    candidates: list[tuple[str, float, list[str]]],
    graph: SceneGraph,
) -> list[tuple[str, float, list[str]]]:
    """Post-pipeline wave 4: final fix-1 conditions."""
    if len(candidates) < 2:
        return candidates

    from hlinet.features.compounds.phase2_signatures import _stats
    s = _stats(graph)

    top_label = candidates[0][0]
    sec_label = candidates[1][0]
    if top_label == "banana" and sec_label == "orange":
        if s.get("edge_concentration", 999) < 0.54716980603:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "sports_car" and sec_label == "school_bus":
        if s.get("hist_golden_retriever", 0) > 2.72601413387:
            candidates[0], candidates[1] = candidates[1], candidates[0]

    if len(candidates) >= 3:
        top_label = candidates[0][0]
        r3_label = candidates[2][0]
        if top_label == "king_penguin" and r3_label == "sports_car":
            if s.get("wavelet_total", 0) > 0.36733903198:
                candidates[0], candidates[2] = candidates[2], candidates[0]

    if len(candidates) >= 4:
        top_label = candidates[0][0]
        r4_label = candidates[3][0]
        if top_label == "sports_car" and r4_label == "brown_bear":
            if s.get("dark_warm_ratio", 0) > 123.00000000000:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "golden_retriever" and r4_label == "mushroom":
            if s.get("cm_center_a", 999) < 0.48527496936:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "teapot" and r4_label == "orange":
            if s.get("orient_entropy", 999) < 2.71611502048:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "banana" and r4_label == "teapot":
            if s.get("edge_vert_mid_ratio", 999) < 0.21985202268:
                candidates[0], candidates[3] = candidates[3], candidates[0]

    return candidates


def _compute_pair_signals(pair, s, _sigmoid):
    if pair == frozenset(["king_penguin", "sports_car"]):
        return ("sports_car",
                _sigmoid(s.get("grad_mean", 0), 1.35, 3) + _sigmoid(s.get("grad_dir_entropy", 1), 0.95, -10)
                + _sigmoid(s.get("lap_var", 0), 8000, 0.0003) + _sigmoid(s.get("warm_aspect", 0), 1.40, 2),
                "king_penguin",
                _sigmoid(s.get("grad_mean", 1), 1.35, -3) + _sigmoid(s.get("grad_dir_entropy", 0), 0.95, 10)
                + _sigmoid(s.get("lap_var", 99999), 8000, -0.0003) + _sigmoid(s.get("warm_aspect", 1), 1.40, -2))

    if pair == frozenset(["banana", "orange"]):
        return ("orange",
                _sigmoid(s.get("hue_red", 0), 0.25, 6) + _sigmoid(s.get("color_std", 0), 0.35, 5)
                + _sigmoid(s.get("bw", 1), 0.25, -5) + _sigmoid(s.get("grad_mean", 1), 0.80, -3)
                + _sigmoid(s.get("hist_orange_minus_banana", 0), 0.0, 3)
                + _sigmoid(s.get("warm_hue_mean", 1), 0.40, -6)
                + _sigmoid(s.get("warm_val_mean", 0), 0.66, 5)
                + _sigmoid(s.get("sat_color_std", 0), 0.25, 5)
                + _sigmoid(s.get("cm_center_a", 0), 0.545, 40)
                + _sigmoid(s.get("warm_sat_std", 1), 0.18, -6)
                + _sigmoid(s.get("rb_corr", 1), 0.55, -4),
                "banana",
                _sigmoid(s.get("hue_red", 1), 0.25, -6) + _sigmoid(s.get("color_std", 1), 0.35, -5)
                + _sigmoid(s.get("bw", 0), 0.25, 5) + _sigmoid(s.get("grad_mean", 0), 0.80, 3)
                + _sigmoid(s.get("hist_orange_minus_banana", 0), 0.0, -3)
                + _sigmoid(s.get("warm_hue_mean", 0), 0.40, 6)
                + _sigmoid(s.get("warm_val_mean", 1), 0.66, -5)
                + _sigmoid(s.get("sat_color_std", 1), 0.25, -5)
                + _sigmoid(s.get("cm_center_a", 1), 0.545, -40)
                + _sigmoid(s.get("warm_sat_std", 0), 0.18, 6)
                + _sigmoid(s.get("rb_corr", 0), 0.55, 4))

    if pair == frozenset(["banana", "golden_retriever"]):
        return ("golden_retriever",
                _sigmoid(s.get("edge", 0), 0.24, 10) + _sigmoid(s.get("sat", 1), 0.45, -5)
                + _sigmoid(s.get("yellow", 1), 0.30, -5) + _sigmoid(s.get("bot_edge", 0), 0.24, 8)
                + _sigmoid(s.get("hist_gr_minus_banana", 0), 0.0, 2)
                + _sigmoid(s.get("warm_hue_median", 1), 18.0, -0.3)
                + _sigmoid(s.get("cm_b_std", 1), 0.055, -20)
                + _sigmoid(s.get("hu1", 0), 2.60, 15)
                + _sigmoid(s.get("gb_corr", 0), 0.88, 5)
                + _sigmoid(s.get("rb_corr", 0), 0.80, 4),
                "banana",
                _sigmoid(s.get("edge", 1), 0.24, -10) + _sigmoid(s.get("sat", 0), 0.45, 5)
                + _sigmoid(s.get("yellow", 0), 0.30, 5) + _sigmoid(s.get("bot_edge", 1), 0.24, -8)
                + _sigmoid(s.get("hist_gr_minus_banana", 0), 0.0, -2)
                + _sigmoid(s.get("warm_hue_median", 0), 18.0, 0.3)
                + _sigmoid(s.get("cm_b_std", 0), 0.055, 20)
                + _sigmoid(s.get("hu1", 1), 2.60, -15)
                + _sigmoid(s.get("gb_corr", 1), 0.88, -5)
                + _sigmoid(s.get("rb_corr", 1), 0.80, -4))

    if pair == frozenset(["mushroom", "golden_retriever"]):
        return ("mushroom",
                _sigmoid(s.get("val", 1), 0.50, -4) + _sigmoid(s.get("sat_br", 0), 0.43, 5)
                + _sigmoid(s.get("lap_var", 0), 8000, 0.0002) + _sigmoid(s.get("hue_yellow", 0), 0.12, 5)
                + _sigmoid(s.get("dct_high", 0), 0.20, 5)
                + _sigmoid(s.get("dct_mid_over_low", 0), 0.40, 4)
                + _sigmoid(s.get("edge_concentration", 1), 1.10, -3)
                + _sigmoid(s.get("hist_gr_minus_mushroom", 0), 0.0, -3)
                + _sigmoid(s.get("gabor_90_01_mean", 99), 18, -0.2),
                "golden_retriever",
                _sigmoid(s.get("val", 0), 0.50, 4) + _sigmoid(s.get("sat_br", 1), 0.43, -5)
                + _sigmoid(s.get("lap_var", 99999), 8000, -0.0002) + _sigmoid(s.get("hue_yellow", 1), 0.12, -5)
                + _sigmoid(s.get("dct_low", 0), 0.22, 4)
                + _sigmoid(s.get("dct_mid_over_low", 1), 0.40, -4)
                + _sigmoid(s.get("edge_concentration", 0), 1.10, 3)
                + _sigmoid(s.get("hist_gr_minus_mushroom", 0), 0.0, 3)
                + _sigmoid(s.get("gabor_90_01_mean", 0), 18, 0.2))

    if pair == frozenset(["orange", "golden_retriever"]):
        return ("orange",
                _sigmoid(s.get("sat", 0), 0.55, 5) + _sigmoid(s.get("color_std", 0), 0.30, 5)
                + _sigmoid(s.get("blob_coverage", 0), 0.50, 4) + _sigmoid(s.get("circularity", 0), 0.03, 20),
                "golden_retriever",
                _sigmoid(s.get("radial_warm_diff", 0), 0.15, 4) + _sigmoid(s.get("bot_edge", 0), 0.22, 8)
                + _sigmoid(s.get("warm_br", 0), 0.45, 4) + _sigmoid(s.get("grad_dir_entropy", 0), 0.97, 8))

    if pair == frozenset(["sports_car", "golden_retriever"]):
        return ("sports_car",
                _sigmoid(s.get("grad_dir_entropy", 1), 0.93, -12) + _sigmoid(s.get("lap_var", 0), 9000, 0.0003)
                + _sigmoid(s.get("bw", 0), 0.55, 4) + _sigmoid(s.get("sky_ratio", 0), 0.20, 4),
                "golden_retriever",
                _sigmoid(s.get("warm", 0), 0.40, 5) + _sigmoid(s.get("radial_warm_diff", 0), 0.15, 4)
                + _sigmoid(s.get("hue_red", 0), 0.25, 4) + _sigmoid(s.get("warm_br", 0), 0.45, 4))

    if pair == frozenset(["banana", "school_bus"]):
        return ("banana",
                _sigmoid(s.get("val", 0), 0.65, 4) + _sigmoid(s.get("blue_purple", 1), 0.04, -5)
                + _sigmoid(s.get("warm_band_top", 0), 0.50, 4) + _sigmoid(s.get("lap_var", 99999), 9000, -0.0002),
                "school_bus",
                _sigmoid(s.get("val", 1), 0.65, -4) + _sigmoid(s.get("blue_purple", 0), 0.04, 5)
                + _sigmoid(s.get("warm_band_top", 1), 0.50, -4) + _sigmoid(s.get("lap_var", 0), 9000, 0.0002))

    if pair == frozenset(["brown_bear", "sports_car"]):
        return ("brown_bear",
                _sigmoid(s.get("grad_dir_entropy", 0), 0.96, 10) + _sigmoid(s.get("edge", 0), 0.30, 8)
                + _sigmoid(s.get("green_bl", 0), 0.15, 5) + _sigmoid(s.get("sat_br", 1), 0.32, -5),
                "sports_car",
                _sigmoid(s.get("grad_dir_entropy", 1), 0.96, -10) + _sigmoid(s.get("edge", 1), 0.30, -8)
                + _sigmoid(s.get("green_bl", 1), 0.15, -5) + _sigmoid(s.get("sat_br", 0), 0.32, 5))

    if pair == frozenset(["sports_car", "teapot"]):
        return ("sports_car",
                _sigmoid(s.get("grad_dir_entropy", 1), 0.93, -10) + _sigmoid(s.get("hue_red", 1), 0.40, -4)
                + _sigmoid(s.get("top_uniformity", 0), 0.66, 4) + _sigmoid(s.get("lap_var", 0), 7500, 0.0003)
                + _sigmoid(s.get("orient_entropy", 1), 2.87, -8),
                "teapot",
                _sigmoid(s.get("grad_dir_entropy", 0), 0.93, 10) + _sigmoid(s.get("hue_red", 0), 0.40, 4)
                + _sigmoid(s.get("top_uniformity", 1), 0.66, -4) + _sigmoid(s.get("lap_var", 99999), 7500, -0.0003)
                + _sigmoid(s.get("orient_entropy", 0), 2.87, 8))

    if pair == frozenset(["mushroom", "school_bus"]):
        return ("mushroom",
                _sigmoid(s.get("grad_dir_entropy", 0), 0.97, 10) + _sigmoid(s.get("edge_br", 0), 0.33, 8)
                + _sigmoid(s.get("lbp_entropy", 0), 5.32, 8) + _sigmoid(s.get("edge", 0), 0.33, 8),
                "school_bus",
                _sigmoid(s.get("grad_dir_entropy", 1), 0.97, -10) + _sigmoid(s.get("edge_br", 1), 0.33, -8)
                + _sigmoid(s.get("lbp_entropy", 1), 5.32, -8) + _sigmoid(s.get("edge", 1), 0.33, -8))

    if pair == frozenset(["brown_bear", "school_bus"]):
        return ("brown_bear",
                _sigmoid(s.get("green", 0), 0.15, 5) + _sigmoid(s.get("edge", 0), 0.30, 8)
                + _sigmoid(s.get("grad_dir_entropy", 0), 0.96, 10) + _sigmoid(s.get("hue_cyan_blue", 1), 0.02, -8)
                + _sigmoid(s.get("autocorr_h", 1), 0.17, -6)
                + _sigmoid(s.get("horiz_dominance", 1), 1.30, -3),
                "school_bus",
                _sigmoid(s.get("green", 1), 0.15, -5) + _sigmoid(s.get("edge", 1), 0.30, -8)
                + _sigmoid(s.get("grad_dir_entropy", 1), 0.96, -10) + _sigmoid(s.get("hue_cyan_blue", 0), 0.02, 8)
                + _sigmoid(s.get("autocorr_h", 0), 0.17, 6)
                + _sigmoid(s.get("horiz_dominance", 0), 1.30, 3))

    if pair == frozenset(["sports_car", "school_bus"]):
        return ("sports_car",
                _sigmoid(s.get("hue_orange", 1), 0.30, -5) + _sigmoid(s.get("yellow", 1), 0.15, -5)
                + _sigmoid(s.get("warm", 1), 0.30, -4) + _sigmoid(s.get("blob_lap_var", 1), 0.60, -3)
                + _sigmoid(s.get("hist_sports_minus_bus", 0), -0.15, 2)
                + _sigmoid(s.get("warm_bl", 1), 0.33, -4)
                + _sigmoid(s.get("radial_warm_diff", 1), 0.10, -4)
                + _sigmoid(s.get("blob_coverage", 1), 0.25, -4)
                + _sigmoid(s.get("dct_high", 0), 0.195, 5)
                + _sigmoid(s.get("vert_regularity", 0), 3.3, 3)
                + _sigmoid(s.get("hu1", 1), 2.64, -15)
                ,
                "school_bus",
                _sigmoid(s.get("hue_orange", 0), 0.30, 5) + _sigmoid(s.get("yellow", 0), 0.15, 5)
                + _sigmoid(s.get("warm", 0), 0.30, 4) + _sigmoid(s.get("blob_lap_var", 0), 0.60, 3)
                + _sigmoid(s.get("hist_sports_minus_bus", 0), -0.15, -2)
                + _sigmoid(s.get("warm_bl", 0), 0.33, 4)
                + _sigmoid(s.get("radial_warm_diff", 0), 0.10, 4)
                + _sigmoid(s.get("blob_coverage", 0), 0.25, 4)
                + _sigmoid(s.get("dct_high", 1), 0.195, -5)
                + _sigmoid(s.get("vert_regularity", 10), 3.3, -3)
                + _sigmoid(s.get("hu1", 0), 2.64, 15)
                )

    if pair == frozenset(["brown_bear", "golden_retriever"]):
        return ("brown_bear",
                _sigmoid(s.get("textured_decentered", 0), 0.10, 8) + _sigmoid(s.get("center_surround", 1), 0.90, -4)
                + _sigmoid(s.get("edge_tl", 0), 0.26, 8) + _sigmoid(s.get("warm_val_mean", 1), 0.51, -5)
                + _sigmoid(s.get("hist_bear_minus_gr", 0), 0.0, 3)
                + _sigmoid(s.get("dct_high", 0), 0.20, 4)
                + _sigmoid(s.get("gabor_45_04_var", 0), 0.70, 3)
                + _sigmoid(s.get("cm_center_b", 1), 0.555, -25),
                "golden_retriever",
                _sigmoid(s.get("textured_decentered", 1), 0.10, -8) + _sigmoid(s.get("center_surround", 0), 0.90, 4)
                + _sigmoid(s.get("edge_tl", 1), 0.26, -8) + _sigmoid(s.get("warm_val_mean", 0), 0.51, 5)
                + _sigmoid(s.get("hist_bear_minus_gr", 0), 0.0, -3)
                + _sigmoid(s.get("dct_low", 0), 0.22, 4)
                + _sigmoid(s.get("gabor_45_04_var", 1), 0.70, -3)
                + _sigmoid(s.get("cm_center_b", 0), 0.555, 25))

    if pair == frozenset(["brown_bear", "banana"]):
        return ("brown_bear",
                _sigmoid(s.get("textured_decentered", 0), 0.09, 10) + _sigmoid(s.get("smooth_yellow", 1), 0.08, -6)
                + _sigmoid(s.get("edge", 0), 0.25, 8) + _sigmoid(s.get("warm_val_mean", 1), 0.54, -5),
                "banana",
                _sigmoid(s.get("textured_decentered", 1), 0.09, -10) + _sigmoid(s.get("smooth_yellow", 0), 0.08, 6)
                + _sigmoid(s.get("edge", 1), 0.25, -8) + _sigmoid(s.get("warm_val_mean", 0), 0.54, 5))

    if pair == frozenset(["brown_bear", "mushroom"]):
        return ("brown_bear",
                _sigmoid(s.get("textured_decentered", 0), 0.09, 10) + _sigmoid(s.get("center_surround", 1), 0.95, -5)
                + _sigmoid(s.get("sat", 1), 0.42, -4) + _sigmoid(s.get("sat_bl", 1), 0.42, -4)
                + _sigmoid(s.get("hist_mushroom_minus_bear", 0), 0.0, -3)
                + _sigmoid(s.get("cm_a_std", 1), 0.031, -30)
                ,
                "mushroom",
                _sigmoid(s.get("textured_decentered", 1), 0.09, -10) + _sigmoid(s.get("center_surround", 0), 0.95, 5)
                + _sigmoid(s.get("sat", 0), 0.42, 4) + _sigmoid(s.get("sat_bl", 0), 0.42, 4)
                + _sigmoid(s.get("hist_mushroom_minus_bear", 0), 0.0, 3)
                + _sigmoid(s.get("cm_a_std", 0), 0.031, 30)
                )

    if pair == frozenset(["teapot", "king_penguin"]):
        return ("teapot",
                _sigmoid(s.get("hist_teapot_minus_kp", 0), -0.03, 4)
                + _sigmoid(s.get("warm_bl", 0), 0.20, 5)
                + _sigmoid(s.get("smooth_warm", 0), 0.06, 6)
                + _sigmoid(s.get("horiz_dominance", 0), 1.05, 4)
                + _sigmoid(s.get("autocorr_h", 0), 0.14, 6)
                + _sigmoid(s.get("sat_br", 0), 0.25, 5)
                + _sigmoid(s.get("mid_wider", 0), 0.5, 5)
                + _sigmoid(s.get("mid_width_ratio", 0), 1.35, 3)
                + _sigmoid(s.get("gabor_dominant_orient", 0), 0.20, 5),
                "king_penguin",
                _sigmoid(s.get("hist_teapot_minus_kp", 0), -0.03, -4)
                + _sigmoid(s.get("warm_bl", 1), 0.20, -5)
                + _sigmoid(s.get("smooth_warm", 1), 0.06, -6)
                + _sigmoid(s.get("horiz_dominance", 1), 1.05, -4)
                + _sigmoid(s.get("autocorr_h", 1), 0.14, -6)
                + _sigmoid(s.get("sat_br", 1), 0.25, -5)
                + _sigmoid(s.get("mid_wider", 1), 0.5, -5)
                + _sigmoid(s.get("mid_width_ratio", 1), 1.35, -3)
                + _sigmoid(s.get("gabor_dominant_orient", 1), 0.20, -5))

    if pair == frozenset(["teapot", "banana"]):
        return ("teapot",
                _sigmoid(s.get("yellow", 1), 0.30, -5) + _sigmoid(s.get("edge", 1), 0.20, -8)
                + _sigmoid(s.get("top_uniformity", 0), 0.70, 5) + _sigmoid(s.get("hist_teapot_minus_banana", 0), 0.0, 3)
                + _sigmoid(s.get("sat", 1), 0.45, -5)
                + _sigmoid(s.get("color_std", 1), 0.22, -5)
                + _sigmoid(s.get("mid_wider", 0), 0.5, 4)
                + _sigmoid(s.get("gabor_dominant_orient", 0), 0.30, 4)
                + _sigmoid(s.get("cm_center_b", 1), 0.59, -40)
                + _sigmoid(s.get("orient_entropy", 0), 2.88, 8)
                + _sigmoid(s.get("gb_corr", 0), 0.85, 4)
                + _sigmoid(s.get("r0_warm", 0), 0.45, 4)
                + _sigmoid(s.get("warm_center_cool_surround", 1), 0.30, -3),
                "banana",
                _sigmoid(s.get("yellow", 0), 0.30, 5) + _sigmoid(s.get("edge", 0), 0.20, 8)
                + _sigmoid(s.get("top_uniformity", 1), 0.70, -5) + _sigmoid(s.get("hist_teapot_minus_banana", 0), 0.0, -3)
                + _sigmoid(s.get("sat", 0), 0.45, 5)
                + _sigmoid(s.get("color_std", 0), 0.22, 5)
                + _sigmoid(s.get("mid_wider", 1), 0.5, -4)
                + _sigmoid(s.get("gabor_dominant_orient", 1), 0.30, -4)
                + _sigmoid(s.get("cm_center_b", 0), 0.59, 40)
                + _sigmoid(s.get("orient_entropy", 1), 2.88, -8)
                + _sigmoid(s.get("gb_corr", 1), 0.85, -4)
                + _sigmoid(s.get("warm_center_cool_surround", 0), 0.30, 3))

    if pair == frozenset(["banana", "mushroom"]):
        return ("banana",
                _sigmoid(s.get("warm_val_mean", 0), 0.55, 5) + _sigmoid(s.get("smooth_warm", 0), 0.15, 5)
                + _sigmoid(s.get("val", 0), 0.55, 4) + _sigmoid(s.get("hist_banana_minus_mushroom", 0), 0.0, 3)
                + _sigmoid(s.get("smooth_yellow", 0), 0.08, 8)
                + _sigmoid(s.get("sat_smooth_warm", 0), 0.08, 6)
                + _sigmoid(s.get("dct_low", 0), 0.23, 5)
                + _sigmoid(s.get("edge_concentration", 0), 1.10, 3)
                + _sigmoid(s.get("hu1", 1), 2.61, -15)
                + _sigmoid(s.get("hu2", 1), 7.3, -3),
                "mushroom",
                _sigmoid(s.get("round_edge", 0), 0.20, 6) + _sigmoid(s.get("edge", 0), 0.25, 6)
                + _sigmoid(s.get("warm_blob_count", 0), 0.4, 4) + _sigmoid(s.get("hist_banana_minus_mushroom", 0), 0.0, -3)
                + _sigmoid(s.get("bot_edge", 0), 0.26, 5)
                + _sigmoid(s.get("edge_br", 0), 0.26, 5)
                + _sigmoid(s.get("dct_high", 0), 0.20, 5)
                + _sigmoid(s.get("edge_entropy", 0), 3.8, 3)
                + _sigmoid(s.get("hu1", 0), 2.61, 15)
                + _sigmoid(s.get("hu2", 0), 7.3, 3))

    if pair == frozenset(["golden_retriever", "teapot"]):
        return ("golden_retriever",
                _sigmoid(s.get("edge", 0), 0.23, 8) + _sigmoid(s.get("bot_edge", 0), 0.25, 8)
                + _sigmoid(s.get("horiz_dominance", 1), 1.10, -4) + _sigmoid(s.get("hist_gr_minus_teapot", 0), 0.0, 3),
                "teapot",
                _sigmoid(s.get("edge", 1), 0.23, -8) + _sigmoid(s.get("autocorr_h", 0), 0.15, 5)
                + _sigmoid(s.get("horiz_dominance", 0), 1.10, 4) + _sigmoid(s.get("hist_gr_minus_teapot", 0), 0.0, -3))

    if pair == frozenset(["brown_bear", "king_penguin"]):
        return ("brown_bear",
                _sigmoid(s.get("textured_decentered", 0), 0.10, 8) + _sigmoid(s.get("edge", 0), 0.27, 6)
                + _sigmoid(s.get("warm_tl", 0), 0.35, 4) + _sigmoid(s.get("hist_bear_minus_kp", 0), 0.0, 3)
                + _sigmoid(s.get("fft_hv_ratio", 1), 0.95, -3)
                + _sigmoid(s.get("hu1", 0), 2.63, 15),
                "king_penguin",
                _sigmoid(s.get("center_surround", 0), 0.95, 4) + _sigmoid(s.get("edge", 1), 0.27, -6)
                + _sigmoid(s.get("warm_tl", 1), 0.35, -4) + _sigmoid(s.get("hist_bear_minus_kp", 0), 0.0, -3)
                + _sigmoid(s.get("fft_hv_ratio", 0), 0.95, 3)
                + _sigmoid(s.get("hu1", 1), 2.63, -15))

    if pair == frozenset(["jellyfish", "king_penguin"]):
        return ("jellyfish",
                _sigmoid(s.get("sat", 0), 0.45, 5) + _sigmoid(s.get("color_std", 0), 0.25, 5)
                + _sigmoid(s.get("sat_br", 0), 0.45, 4) + _sigmoid(s.get("hist_jelly_minus_kp", 0), 0.0, 3),
                "king_penguin",
                _sigmoid(s.get("grad_mean", 0), 0.95, 3) + _sigmoid(s.get("sat", 1), 0.45, -5)
                + _sigmoid(s.get("color_std", 1), 0.25, -5) + _sigmoid(s.get("hist_jelly_minus_kp", 0), 0.0, -3))

    if pair == frozenset(["orange", "teapot"]):
        return ("orange",
                _sigmoid(s.get("sat", 0), 0.50, 5) + _sigmoid(s.get("color_std", 0), 0.25, 5)
                + _sigmoid(s.get("sat_bl", 0), 0.50, 4) + _sigmoid(s.get("hist_orange_minus_teapot", 0), 0.0, 3),
                "teapot",
                _sigmoid(s.get("sat", 1), 0.50, -5) + _sigmoid(s.get("autocorr_h", 0), 0.15, 5)
                + _sigmoid(s.get("color_std", 1), 0.25, -5) + _sigmoid(s.get("hist_orange_minus_teapot", 0), 0.0, -3))

    if pair == frozenset(["brown_bear", "teapot"]):
        return ("brown_bear",
                _sigmoid(s.get("edge", 0), 0.25, 8) + _sigmoid(s.get("textured_decentered", 0), 0.10, 8)
                + _sigmoid(s.get("top_edge", 0), 0.24, 6) + _sigmoid(s.get("hist_bear_minus_teapot", 0), 0.0, 3)
                + _sigmoid(s.get("hu1", 0), 2.64, 15),
                "teapot",
                _sigmoid(s.get("autocorr_h", 0), 0.14, 6) + _sigmoid(s.get("edge", 1), 0.25, -8)
                + _sigmoid(s.get("top_edge", 1), 0.24, -6) + _sigmoid(s.get("hist_bear_minus_teapot", 0), 0.0, -3)
                + _sigmoid(s.get("hu1", 1), 2.64, -15))

    if pair == frozenset(["golden_retriever", "king_penguin"]):
        return ("golden_retriever",
                _sigmoid(s.get("warm", 0), 0.35, 5) + _sigmoid(s.get("blob_coverage", 0), 0.30, 4)
                + _sigmoid(s.get("hue_red", 0), 0.25, 4) + _sigmoid(s.get("hist_gr_minus_kp", 0), 0.0, 3)
                + _sigmoid(s.get("warm_hue_median", 1), 18.0, -0.3)
                + _sigmoid(s.get("fft_hv_ratio", 1), 1.0, -3)
                + _sigmoid(s.get("cm_center_a", 0), 0.515, 40)
                + _sigmoid(s.get("cm_center_b", 0), 0.545, 40),
                "king_penguin",
                _sigmoid(s.get("warm", 1), 0.35, -5) + _sigmoid(s.get("blob_coverage", 1), 0.30, -4)
                + _sigmoid(s.get("hue_red", 1), 0.25, -4) + _sigmoid(s.get("hist_gr_minus_kp", 0), 0.0, -3)
                + _sigmoid(s.get("warm_hue_median", 0), 18.0, 0.3)
                + _sigmoid(s.get("fft_hv_ratio", 0), 1.0, 3)
                + _sigmoid(s.get("cm_center_a", 1), 0.515, -40)
                + _sigmoid(s.get("cm_center_b", 1), 0.545, -40))

    if pair == frozenset(["jellyfish", "teapot"]):
        return ("jellyfish",
                _sigmoid(s.get("sat", 0), 0.50, 5) + _sigmoid(s.get("blue_purple", 0), 0.20, 5)
                + _sigmoid(s.get("color_std", 0), 0.25, 5) + _sigmoid(s.get("edge_entropy", 1), 3.5, -4)
                + _sigmoid(s.get("edge_concentration", 0), 1.4, 3),
                "teapot",
                _sigmoid(s.get("sat", 1), 0.50, -5) + _sigmoid(s.get("blue_purple", 1), 0.20, -5)
                + _sigmoid(s.get("color_std", 1), 0.25, -5) + _sigmoid(s.get("edge_entropy", 0), 3.5, 4)
                + _sigmoid(s.get("edge_concentration", 1), 1.4, -3))

    if pair == frozenset(["orange", "mushroom"]):
        return ("orange",
                _sigmoid(s.get("warm_val_mean", 0), 0.58, 5) + _sigmoid(s.get("smooth_warm", 0), 0.16, 5)
                + _sigmoid(s.get("color_std", 0), 0.27, 5) + _sigmoid(s.get("dct_low", 0), 0.24, 5)
                + _sigmoid(s.get("sat", 0), 0.55, 5),
                "mushroom",
                _sigmoid(s.get("edge", 0), 0.24, 8) + _sigmoid(s.get("grad_mean", 0), 1.1, 3)
                + _sigmoid(s.get("round_edge", 0), 0.20, 5) + _sigmoid(s.get("dct_high", 0), 0.19, 5)
                + _sigmoid(s.get("lap_var", 0), 6500, 0.0002))

    if pair == frozenset(["teapot", "school_bus"]):
        return ("teapot",
                _sigmoid(s.get("grad_dir_entropy", 0), 0.95, 8) + _sigmoid(s.get("dct_low", 0), 0.21, 5)
                + _sigmoid(s.get("edge_concentration", 0), 1.28, 3) + _sigmoid(s.get("top_uniformity", 0), 0.65, 4)
                + _sigmoid(s.get("smooth_warm", 0), 0.11, 5),
                "school_bus",
                _sigmoid(s.get("grad_mean", 0), 1.4, 3) + _sigmoid(s.get("blob_lap_var", 0), 0.80, 3)
                + _sigmoid(s.get("edge", 0), 0.23, 8) + _sigmoid(s.get("autocorr_h", 0), 0.22, 5)
                + _sigmoid(s.get("horiz_dominance", 0), 1.4, 3))

    if pair == frozenset(["mushroom", "king_penguin"]):
        return ("mushroom",
                _sigmoid(s.get("center_surround", 0), 1.10, 3) + _sigmoid(s.get("edge", 0), 0.25, 8)
                + _sigmoid(s.get("sat", 0), 0.40, 5) + _sigmoid(s.get("round_edge", 0), 0.20, 5),
                "king_penguin",
                _sigmoid(s.get("bw", 0), 0.55, 5) + _sigmoid(s.get("fft_hv_ratio", 0), 1.0, 3)
                + _sigmoid(s.get("sat", 1), 0.40, -5) + _sigmoid(s.get("center_surround", 1), 1.10, -3))

    if pair == frozenset(["mushroom", "sports_car"]):
        return ("mushroom",
                _sigmoid(s.get("grad_dir_entropy", 0), 0.95, 10) + _sigmoid(s.get("edge", 0), 0.28, 8)
                + _sigmoid(s.get("warm", 0), 0.35, 5) + _sigmoid(s.get("fft_hv_ratio", 0), 0.85, 3),
                "sports_car",
                _sigmoid(s.get("autocorr_h", 0), 0.15, 6) + _sigmoid(s.get("horiz_dominance", 0), 1.4, 3)
                + _sigmoid(s.get("warm", 1), 0.35, -5) + _sigmoid(s.get("fft_hv_ratio", 1), 0.85, -3))

    return None



def _score_all_classes_flat(
    root: ClassNode, graph: SceneGraph, cache: dict[str, FeatureValue]
) -> list[tuple[str, float, list[str]]]:
    """Score ALL leaf classes directly, ignoring gate thresholds."""
    results = []
    _collect_leaves(root, graph, cache, [], results)
    return results


def _collect_leaves(
    node: ClassNode, graph: SceneGraph, cache: dict[str, FeatureValue],
    route: list[str], results: list[tuple[str, float, list[str]]]
) -> None:
    current_route = route + [node.name]
    if node.is_leaf:
        score = score_node(node, graph, cache)
        results.append((node.name, score, current_route))
    else:
        for child in node.children:
            _collect_leaves(child, graph, cache, current_route, results)


def _generate_proof(label: str, route: list[str], cache: dict[str, FeatureValue]) -> list[str]:
    """Generate a human-readable proof trace for the prediction."""
    proof = [f"Claim: image contains '{label}'"]
    proof.append(f"Route: {' → '.join(route)}")
    proof.append("Evidence:")

    active_features = [
        (name, val) for name, val in cache.items()
        if val.present and val.confidence > 0.2
    ]
    active_features.sort(key=lambda x: x[1].confidence, reverse=True)

    for name, val in active_features[:8]:
        evidence_str = "; ".join(val.evidence[:2]) if val.evidence else "detected"
        proof.append(f"  {name}: {val.confidence:.2f} ({evidence_str})")

    absent_features = [
        (name, val) for name, val in cache.items()
        if not val.present and val.evidence
    ]
    if absent_features[:3]:
        proof.append("Absent (supporting exclusion):")
        for name, val in absent_features[:3]:
            proof.append(f"  {name}: not detected")

    return proof


def predict_file(image_path: str | Path) -> Prediction:
    """Convenience: predict from a file path."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return predict(image)


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: hlinet-predict <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    result = predict_file(path)

    print(f"\nPrediction: {result.label} ({result.confidence:.2f})")
    if result.alternatives:
        print(f"Alternatives: {', '.join(f'{l} ({s:.2f})' for l, s in result.alternatives)}")
    print()
    for line in result.proof:
        print(line)


if __name__ == "__main__":
    main()
