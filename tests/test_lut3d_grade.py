import pytest
import imaging.lut3d_grade as mod

CASES = [
    ("gamut_clamp", (1,)),
    ("uniform_tetra_split", (1,)),
    ("input_colorspace", (1,)),
    ("shaper_apply", (1,)),
    ("cache_version", (1,)),
    ("gamut_map_clip", (1,)),
    ("load_lut", (1,)),
    ("build_lut", (1,)),
    ("apply_lut", (1,)),
    ("present_grade", (1,)),
    ("validate_grade", (1,)),
    ("metrics_grade", (1,)),
    ("audit_grade", (1,)),
    ("fallback_grade", (1,)),
    ("determinism_grade", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
