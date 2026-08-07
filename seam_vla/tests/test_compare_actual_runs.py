import numpy as np

from benchmark.seam_vla.experiments.rby1.compare_actual_runs import compare_runs, evaluate_run


def _write_run(path, actions, chunks, *, condition):
    np.savez_compressed(
        path,
        executed_actions=actions,
        measured_qpos=actions * 0.8,
        predicted_chunks=chunks,
        chunk_start_steps=np.arange(len(chunks)) * 4,
        inference_states=np.zeros((len(chunks), 14)),
        inference_ms=np.full(len(chunks), 10.0),
        used_vls=np.asarray([condition == "seam"] * len(chunks)),
        chunk_indices=np.arange(len(chunks)),
        prompt=np.asarray("test task"),
        condition=np.asarray(condition),
        control_hz=np.asarray(15),
        execution_length=np.asarray(4),
    )


def test_evaluate_and_compare_actual_runs(tmp_path):
    rng = np.random.default_rng(0)
    chunks = rng.normal(size=(3, 10, 14))
    base_actions = np.concatenate([chunk[:4] for chunk in chunks])
    seam_actions = base_actions.copy()
    for boundary in (4, 8):
        seam_actions[boundary] = seam_actions[boundary - 1]

    base_path = tmp_path / "base.npz"
    seam_path = tmp_path / "seam.npz"
    _write_run(base_path, base_actions, chunks, condition="baseline")
    _write_run(seam_path, seam_actions, chunks, condition="seam")

    run = evaluate_run(base_path)
    assert run["num_steps"] == 12
    assert run["num_chunks"] == 3
    assert run["action_metrics"]["num_boundary"] == 2
    assert np.isfinite(run["overlap_residual"])

    result = compare_runs(base_path, seam_path)
    assert result["baseline"]["prompt"] == "test task"
    assert result["seam"]["vls_chunks"] == 3
