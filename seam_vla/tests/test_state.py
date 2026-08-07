import numpy as np

from benchmark.seam_vla.rollout.chunk_state import SeamSession
from benchmark.seam_vla.state import SeamState


def test_initial_is_first_chunk():
    s = SeamState.initial()
    assert s.is_first_chunk
    assert s.chunk_index == 0
    assert s.previous_chunk_model_space is None
    assert s.previous_chunk_physical_space is None


def test_with_chunk_advances():
    s = SeamState.initial()
    m = np.zeros((10, 32))
    p = np.zeros((10, 7))
    s2 = s.with_chunk(m, p)
    assert not s2.is_first_chunk
    assert s2.chunk_index == 1
    assert s2.previous_chunk_model_space is m
    assert s2.previous_chunk_physical_space is p
    # original unchanged (immutable)
    assert s.is_first_chunk


def test_reset_clears_both_representations():
    s = SeamState.initial().with_chunk(np.ones((10, 32)), np.ones((10, 7)))
    r = s.reset()
    assert r.is_first_chunk
    assert r.previous_chunk_model_space is None
    assert r.previous_chunk_physical_space is None
    assert r.chunk_index == 0


def test_second_chunk_carries_first_model_space():
    s = SeamState.initial()
    m1 = np.full((10, 32), 3.0)
    s = s.with_chunk(m1, np.zeros((10, 7)))
    # Before producing chunk 2, the previous model-space tail is m1.
    assert s.previous_chunk_model_space is m1
    assert s.chunk_index == 1


def test_session_reset_and_first_chunk():
    sess = SeamSession()
    assert sess.is_first_chunk
    sess.record_chunk(np.zeros((10, 32)), np.zeros((10, 7)))
    assert not sess.is_first_chunk
    assert sess.chunk_index == 1
    sess.reset()
    assert sess.is_first_chunk
    assert sess.chunk_index == 0
