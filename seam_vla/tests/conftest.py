import os
import sys

import jax
import pytest

# Ensure the workspace root is importable as `benchmark.*` regardless of pytest invocation dir.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def has_gpu() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not has_gpu(), reason="no JAX GPU device available")

# Heavy tests that load the ~2B pi0.5 checkpoint. Off by default; enable with SEAM_RUN_MODEL_TESTS=1.
run_model_tests = pytest.mark.skipif(
    os.environ.get("SEAM_RUN_MODEL_TESTS", "0") != "1",
    reason="set SEAM_RUN_MODEL_TESTS=1 to run heavy pi0.5 model tests",
)

CKPT = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi05_libero")
