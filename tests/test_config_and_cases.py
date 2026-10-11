"""Unit tests 1, 15 (partial): X-ABCD parsing + test-set isolation invariants."""
import pytest

from src.config import (
    ALL_CASE_IDS,
    RunConfig,
    case_label,
    format_case_id,
    parse_case_id,
)


def test_sixteen_canonical_cases():
    assert len(ALL_CASE_IDS) == 16
    assert len(set(ALL_CASE_IDS)) == 16
    assert ALL_CASE_IDS[0] == "X-0000" and ALL_CASE_IDS[-1] == "X-1111"


@pytest.mark.parametrize(
    "cid,exp",
    [
        ("X-0000", {"bbs": 0, "tlc": 0, "rope": 0, "alibi": 0}),
        ("X-1101", {"bbs": 1, "tlc": 1, "rope": 0, "alibi": 1}),
        ("x-1010", {"bbs": 1, "tlc": 0, "rope": 1, "alibi": 0}),
    ],
)
def test_parse_case_id(cid, exp):
    assert parse_case_id(cid) == exp


def test_parse_case_id_rejects_bad():
    for bad in ("X-000", "Y-0000", "X-0002", "X-abcd", "0000"):
        with pytest.raises(ValueError):
            parse_case_id(bad)


def test_format_and_label():
    assert format_case_id(1, 1, 0, 1) == "X-1101"
    assert case_label("X-0000") == "baseline"
    assert case_label("X-1101") == "BBS + TLC + ALiBi"


def test_bit_meaning_is_A_bbs_B_tlc_C_rope_D_alibi():
    # position -> factor, never reordered
    assert parse_case_id("X-1000")["bbs"] == 1
    assert parse_case_id("X-0100")["tlc"] == 1
    assert parse_case_id("X-0010")["rope"] == 1
    assert parse_case_id("X-0001")["alibi"] == 1


def test_runconfig_yaml_roundtrip(tmp_path):
    cfg = RunConfig(case_id="X-1101", seed=43)
    p = tmp_path / "config.yaml"
    cfg.save_yaml(str(p))
    back = RunConfig.from_dict(__import__("yaml").safe_load(p.read_text()))
    assert back.case_id == "X-1101" and back.seed == 43


def test_checkpoint_selection_is_validation_only():
    # brief section 28: metric_for_best_model must be a validation (eval_*) metric
    cfg = RunConfig(case_id="X-1101", seed=42)
    assert cfg.metric_for_best_model.startswith("eval_")
