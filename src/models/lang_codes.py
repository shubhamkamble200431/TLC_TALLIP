"""
Language-code resolution for the two backbones used in TLC_TALLIP.

Sanskrit -> Hindi only.  NLLB has native codes for both; mBART-50 does not have
Sanskrit, so a documented substitute is used (see mbart50_many_to_many.yaml and
docs/ASSUMPTIONS.md).
"""
from __future__ import annotations

# NLLB-200 codes (canonical internal representation).
NLLB_SRC = "san_Deva"
NLLB_TGT = "hin_Deva"

# NLLB -> mBART-50 code.  Sanskrit has no mBART-50 code; hi_IN (Devanagari,
# large lexical overlap) is the substitute, mr_IN the alternative.
NLLB_TO_MBART50 = {
    "san_Deva": "hi_IN",   # substitute — mBART-50 lacks Sanskrit
    "hin_Deva": "hi_IN",
    "mar_Deva": "mr_IN",
    "guj_Gujr": "gu_IN",
    "nep_Deva": "ne_NP",
    "ben_Beng": "bn_IN",
}


def infer_model_family(base_model: str) -> str:
    bm = base_model.lower()
    if "mbart" in bm:
        return "mbart"
    if "mt5" in bm:
        return "mt5"
    if "indictrans" in bm:
        return "indictrans2"
    if bm.startswith("scratch-tf6"):
        return "tf6"
    if bm.startswith("scratch-tf12"):
        return "tf12"
    if bm.startswith("scratch-tf18"):
        return "tf18"
    if bm.startswith("scratch-mamba-small"):
        return "mamba_small"
    if bm.startswith("scratch-mamba-base"):
        return "mamba_base"
    if bm.startswith("scratch-mamba-large"):
        return "mamba_large"
    if "nllb" in bm or "m2m" in bm:
        return "nllb"
    return "nllb"


def nllb_to_mbart50(nllb_code: str) -> str:
    if nllb_code in NLLB_TO_MBART50:
        return NLLB_TO_MBART50[nllb_code]
    raise ValueError(f"No mBART-50 code mapping for {nllb_code!r}")
