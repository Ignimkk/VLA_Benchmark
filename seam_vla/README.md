# SEAM: Smooth Execution of Action-Chunked Motion (VLS) for π0.5

Training-free, inference-time implementation of **Velocity-guided Loss Steering (VLS)** — the core of
SEAM — for the π0.5 flow-matching VLA in OpenPI, evaluated on LIBERO-10. VLS steers the flow-matching
denoising ODE toward the previous chunk's unexecuted tail with a closed-form, backprop-free correction,
reducing jerk/discontinuity at action-chunk boundaries while preserving task behavior.

## Layout
```
benchmark/seam_vla/
  config.py            SeamConfig (asserts derived H/K/N vs. the real model)
  priors/aligned_tail  build_aligned_prior (Extend tail -> length H)
  guidance/            DenoisingGuidance ABC, IdentityGuidance, VLSGuidance (pure JAX core)
  integration/         SeamSampler: jitted pi0.5 sampler + VLS guidance_fn injection
  policy/              SeamPolicy: transforms + sampler + per-session state
  state.py, rollout/   SeamState (model+physical chunks), chunk executor (K-of-H)
  refinement/          DecodedChunkRefiner ABC + IdentityChunkRefiner (reserved for TO/MPC)
  metrics/             motion (BJ/IJ/CD/AVb, paper-exact) + synchronized latency
  experiments/libero/  in-process baseline/SEAM evaluators + compare + smoke
  configs/             baseline/seam/smoke/full YAML
  docs/                code_inspection, normalization_and_shapes, implementation_report, reproduction_notes
  tests/               unit + heavy-model + GPU tests
```

## The one OpenPI change
`src/openpi/models/pi0.py` `sample_actions` gains an optional `guidance_fn(candidate, t_next)` called
after each Euler step. Default `None` ⇒ **byte-for-byte identical baseline** (verified: max diff 0.0).
No training, checkpoint, or other-model impact; no benchmark import inside OpenPI.

## Important: this checkpoint ≠ the paper profile
The local `pi05_libero` checkpoint bakes **H=10** (paper: 50) and the rollout uses **K=5** (paper: 10),
so the overlap is **L=5** and the guided window is clamped to **M≤5**. The config asserts these derived
values and fails loudly if asked for H=50/K=10 — it never silently changes the baseline. Full detail:
`docs/reproduction_notes.md`.

## Run
```bash
# unit tests (no model)
JAX_PLATFORMS=cpu src/openpi/.venv/bin/python -m pytest benchmark/seam_vla/tests/ -q

# heavy real-model parity/integration (CPU; the 2B model does not fit an 8GB GPU)
JAX_PLATFORMS=cpu SEAM_RUN_MODEL_TESTS=1 src/openpi/.venv/bin/python -m pytest \
  benchmark/seam_vla/tests/test_baseline_parity.py \
  benchmark/seam_vla/tests/test_openpi_integration.py -q

# single-task smoke (sim-free pipeline validation)
JAX_PLATFORMS=cpu src/openpi/.venv/bin/python -m benchmark.seam_vla.experiments.libero.smoke_test --sim-free

# full LIBERO-10 (requires the LIBERO sim + a >=12GB GPU; SEAM run gated)
python -m benchmark.seam_vla.experiments.libero.evaluate_baseline --config benchmark/seam_vla/configs/full_libero10.yaml --out /tmp/b.json
python -m benchmark.seam_vla.experiments.libero.evaluate_seam    --config benchmark/seam_vla/configs/full_libero10.yaml --run-full-benchmark --out /tmp/s.json
python -m benchmark.seam_vla.experiments.libero.compare_results  --baseline /tmp/b.json --seam /tmp/s.json
```

## Status
Core + integration complete and tested (79 fast tests pass; 8 heavy CPU model tests pass; GPU pure-core
passes; baseline parity exact). Task-success/full-benchmark numbers require the LIBERO container and a
larger GPU — commands are prepared but not run here. See `docs/implementation_report.md`.
