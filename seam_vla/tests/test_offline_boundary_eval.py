import numpy as np

from benchmark.seam_vla.experiments.rby1.offline_boundary_eval import _split_runs


def test_split_runs_at_prompt_change():
    chunks = [np.full((2, 1), i) for i in range(5)]
    states = [np.asarray([i]) for i in range(5)]
    prompts = ["blue", "blue", "blue", "red", "red"]

    runs = _split_runs(chunks, states, prompts)

    assert [(len(c), prompt) for c, _, prompt in runs] == [(3, "blue"), (2, "red")]
    np.testing.assert_array_equal(runs[0][0][-1], chunks[2])
    np.testing.assert_array_equal(runs[1][0][0], chunks[3])


def test_split_runs_empty():
    assert _split_runs([], [], []) == []
