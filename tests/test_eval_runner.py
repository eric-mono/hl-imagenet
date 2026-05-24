from pathlib import Path

from hlinet.eval.runner import _infer_log_lineage


def test_default_eval_routes_to_phase2():
    assert _infer_log_lineage("repro_train", None) == "phase2"


def test_explicit_phase_tags_win():
    assert _infer_log_lineage("phase1_smoke", None) == "phase1"
    assert _infer_log_lineage("phase2_smoke", None) == "phase2"


def test_data_dir_can_identify_legacy_phase1():
    assert _infer_log_lineage("baseline", Path("data/imagenet_10")) == "phase1"
    assert _infer_log_lineage("baseline", Path("data/phase2/train")) == "phase2"
