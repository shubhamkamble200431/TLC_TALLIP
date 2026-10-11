"""Unit tests 9, 15: mBART-50 language-code setup + test-set isolation / manifest."""
import os

import pytest
import yaml

from src.config import CONFIGS_DIR, RunConfig
from src.models.lang_codes import infer_model_family, nllb_to_mbart50


def test_nllb_to_mbart_mapping():
    # mBART-50 has no Sanskrit code -> documented hi_IN substitute
    assert nllb_to_mbart50("san_Deva") == "hi_IN"
    assert nllb_to_mbart50("hin_Deva") == "hi_IN"
    with pytest.raises(ValueError):
        nllb_to_mbart50("xxx_Yyyy")


def test_infer_family():
    assert infer_model_family("facebook/nllb-200-distilled-600M") == "nllb"
    assert infer_model_family("facebook/mbart-large-50-many-to-many-mmt") == "mbart"


def test_mbart_model_yaml_sets_languages_explicitly():
    y = yaml.safe_load(open(os.path.join(CONFIGS_DIR, "models", "mbart50_many_to_many.yaml")))
    assert y["model_family"] == "mbart"
    assert "many-to-many" in y["base_model"]
    assert y["mbart_src_lang"] == "hi_IN"
    assert y["mbart_tgt_lang"] == "hi_IN"
    assert y["force_bos_target_lang"] is True


@pytest.mark.skipif(os.environ.get("TLC_OFFLINE") == "1", reason="needs HF tokenizer download")
def test_mbart_tokenizer_forced_bos_resolves():
    from transformers import MBart50TokenizerFast

    tok = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt", src_lang="hi_IN", tgt_lang="hi_IN"
    )
    bos = tok.convert_tokens_to_ids("hi_IN")
    assert isinstance(bos, int) and bos != tok.unk_token_id


def test_manifest_declares_test_isolation():
    from src.experiments.manifest import build

    cfg = RunConfig.resolve("X-1101", 42, "v2", "nllb", "factorial") \
        if os.path.isfile(os.path.join(CONFIGS_DIR, "datasets", "v2.yaml")) \
        else None
    if cfg is None:
        pytest.skip("dataset manifest not built yet")
    m = build(cfg)
    assert m["test_set_used_for_selection"] is False
    assert m["checkpoint_selection_split"] == "validation"
    assert m["factors"] == {"A": 1, "B": 1, "C": 0, "D": 1}


def test_dataset_test_variants_declared():
    """Every model is evaluated on sampled + unsampled of both corpora."""
    import os as _os
    from src.config import CONFIGS_DIR, dataset_test_variants
    if not _os.path.isfile(_os.path.join(CONFIGS_DIR, "datasets", "v2.yaml")):
        pytest.skip("run scripts/prepare_dataset.py first")
    tv2 = dataset_test_variants("v2")
    assert set(tv2) == {"sampled", "unsampled"}
    assert tv2["sampled"]["src"] == "test_7264_5.sa"
    assert tv2["unsampled"]["src"] == "test.sa"
    tv1 = dataset_test_variants("v1")
    assert tv1["sampled"]["src"] == "test_5414_5.sa"
    assert tv1["unsampled"]["src"] == "test.sa"


def test_eval_subdir_shape():
    from src.statistics.aggregate import _eval_subdir
    assert _eval_subdir("v2", "sampled").replace("\\", "/") == "evaluation/v2/sampled"
    assert _eval_subdir("v1", "unsampled").replace("\\", "/") == "evaluation/v1/unsampled"
