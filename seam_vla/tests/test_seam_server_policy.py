"""SeamServerPolicy behavior (reset flag, first-chunk baseline) with a fake SeamPolicy (no model)."""

import types

import numpy as np

from benchmark.seam_vla.serving.seam_server_policy import SeamServerPolicy


class _FakeSeamPolicy:
    """Mimics SeamPolicy.predict_chunk: first chunk baseline, subsequent use VLS; respects reset."""

    def __init__(self, *, log_corrections=False):
        self.config = types.SimpleNamespace(seam_log_corrections=log_corrections)
        self.want_diagnostics = []

    def predict_chunk(self, obs_dict, seam_state, *, want_diagnostics=True, **kw):
        self.want_diagnostics.append(want_diagnostics)
        used = not seam_state.is_first_chunk
        chunk = np.zeros((50, 14), dtype=np.float32)
        diag = types.SimpleNamespace(
            chunk_index=seam_state.chunk_index, used_vls=used,
            correction_norm=1.25 if used and want_diagnostics else 0.0,
            model_space_shape=(50, 32), physical_space_shape=(50, 14),
        )
        new_state = seam_state.with_chunk(np.zeros((50, 32)), chunk, proprio_state=None)
        return chunk, new_state, diag


def test_first_chunk_baseline_then_vls():
    sp = SeamServerPolicy(_FakeSeamPolicy())
    out1 = sp.infer({"state": np.zeros(14)})
    assert out1["seam_timing"]["used_vls"] is False
    assert out1["actions"].shape == (50, 14)
    out2 = sp.infer({"state": np.zeros(14)})
    assert out2["seam_timing"]["used_vls"] is True
    assert out2["seam_timing"]["chunk_index"] == 1
    assert out2["seam_timing"]["correction_norm_measured"] is False
    assert out2["seam_timing"]["correction_norm"] is None


def test_opt_in_measures_only_first_guided_chunk():
    fake = _FakeSeamPolicy(log_corrections=True)
    sp = SeamServerPolicy(fake)
    out1 = sp.infer({"state": np.zeros(14)})
    out2 = sp.infer({"state": np.zeros(14)})
    out3 = sp.infer({"state": np.zeros(14)})

    assert fake.want_diagnostics == [True, True, False]
    assert out1["seam_timing"]["correction_norm_measured"] is False
    assert out2["seam_timing"]["correction_norm_measured"] is True
    assert out2["seam_timing"]["correction_norm"] == 1.25
    assert out3["seam_timing"]["correction_norm_measured"] is False


def test_reset_flag_restarts_episode():
    sp = SeamServerPolicy(_FakeSeamPolicy())
    sp.infer({"state": np.zeros(14)})
    sp.infer({"state": np.zeros(14)})  # now would be VLS
    out = sp.infer({"state": np.zeros(14), "seam_reset": True})  # reset -> baseline again
    assert out["seam_timing"]["used_vls"] is False
    assert out["seam_timing"]["chunk_index"] == 0


def test_reset_method():
    sp = SeamServerPolicy(_FakeSeamPolicy())
    sp.infer({"state": np.zeros(14)})
    sp.reset()
    out = sp.infer({"state": np.zeros(14)})
    assert out["seam_timing"]["used_vls"] is False


def test_reset_key_not_leaked_to_policy():
    # The reset flag must be popped, not passed into the observation the policy sees.
    seen = {}

    class _Spy(_FakeSeamPolicy):
        def predict_chunk(self, obs_dict, seam_state, **kw):
            seen["keys"] = set(obs_dict)
            return super().predict_chunk(obs_dict, seam_state, **kw)

    sp = SeamServerPolicy(_Spy())
    sp.infer({"state": np.zeros(14), "seam_reset": True})
    assert "seam_reset" not in seen["keys"]
