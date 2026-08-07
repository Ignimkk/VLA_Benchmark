# SEAM (VLS) Implementation Report

Reproduction of *SEAM: Smooth Execution of Action-Chunked Motion* (VLS mechanism) for π0.5 in the
JAX OpenPI codebase, evaluated on LIBERO-10. Training-free, inference-time only.

---

## 1. Repository commit / version inspected
OpenPI HEAD `15a9616` ("update output objects to support batching (#975)"). JAX 0.5.3, one CUDA device
(RTX 4060 Ti, 8 GB). Local checkpoint `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero`.

## 2. Modified and added files
**Modified (OpenPI, 1 file, +13/−1):**
- `src/openpi/src/openpi/models/pi0.py` — added optional `guidance_fn` param + post-Euler hook.

**Added (all under `benchmark/seam_vla/`):** `config.py`, `state.py`, `diagnostics.py`;
`priors/aligned_tail.py`; `guidance/{base,identity,vls}.py`; `integration/openpi_jax.py`;
`policy/seam_policy.py`; `refinement/{base,identity}.py`; `rollout/{chunk_state,chunk_executor}.py`;
`metrics/{motion,latency}.py`; `experiments/{config_io}.py` and
`experiments/libero/{rollout_common,evaluator,evaluate_baseline,evaluate_seam,compare_results,smoke_test}.py`;
`configs/*.yaml`; `docs/*.md`; `tests/*.py`; package `__init__.py` files.

## 3. Reason for every modification
Only `pi0.py` is modified because the baseline API returns the **final** chunk and never exposes the
per-step **post-Euler candidate** that VLS must correct. The change is a single optional callable
(`guidance_fn`), identity by default, generic (no benchmark import), JIT-safe. Everything else is new,
isolated benchmark code — no training/checkpoint/other-model impact.

## 4. Actual π0.5 inference call path
`Policy.infer` (`policies/policy.py:67`) → input transforms (repack → LiberoInputs → **Normalize
(quantile)** → tokenize/pad) → `model.sample_actions` (jitted, `pi0.py:216`) → raw model-space actions
`[1,10,32]` → output transforms (**Unnormalize** → `LiberoOutputs[...,:7]`) → physical actions `[10,7]`.

## 5. Exact denoising-loop location
`models/pi0.py:239-283`, a `jax.lax.while_loop` (carry `(x_t, time)`); Euler candidate computed at
`pi0.py:278` (`x_next = x_t + dt*v_t`). `dt=-0.1`, time 1→0.

## 6. Loop structure
`jax.lax.while_loop` (not Python for / scan / fori_loop). SEAM guidance must be pure/JAX-traceable.

## 7. Post-Euler guidance hook location
`models/pi0.py:278-283`: after `x_next = x_t + dt*v_t` and `time_next = time + dt`, if `guidance_fn`
is not None then `x_next = guidance_fn(x_next, time_next)`. Default None ⇒ byte-identical baseline.

## 8. Actual tensor shapes
noise/state/candidate/velocity/aligned-prior: `[B,10,32]`; raw model chunk `[10,32]`; physical chunk
`[10,7]`; executed action `[7]`. Guided block `aligned_prior[:, :M]` = `[B,≤5,32]`; masks broadcast to
`[B,10,32]`.

## 9. Effective H, K, L, M, N, D
H=10, K=5, L=5, M=min(20,5)=5, N=10, D=32 (valid 7). (Paper: 50/10/40/20/10/–,7.) See
`reproduction_notes.md`.

## 10. Valid physical-action dims and padding
Valid dims = 0..6 (6 arm-delta + 1 gripper), from `LiberoOutputs[...,:7]` and `norm_stats.json`
(`actions.q01` length 7). Dims 7..31 are zero-padding from `PadStatesAndActions`; they pass through
`Unnormalize` unchanged and are **never guided** (VLS dim mask = first 7).

## 11. Denoising coordinate system
Normalized **model space** (quantile ~[-1,1] on valid dims). The VLS aligned prior is built in this
space, matching the flow-path interpolation `x_t = (1-t)x_0 + t·ε`.

## 12. Normalization / unnormalization
Quantile (pi05). Norm: `(x-q01)/(q99-q01+1e-6)*2-1`. Unnorm: `(x+1)/2*(q99-q01+1e-6)+q01`. Round-trip
verified exact (`test_normalization.py`). Stats: `<ckpt>/assets/physical-intelligence/libero/norm_stats.json`.

## 13. Gripper representation
Action dim index 6; returned as-is (no remap). LIBERO gripper open = -1.0 (`LIBERO_DUMMY_ACTION`).

## 14. Baseline inference flow
input transforms → `sample_actions(guidance_fn=None)` → output transforms → execute first K=5 actions
via a deque (`examples/libero/main.py` semantics, replicated in `rollout/chunk_executor.py`).

## 15. SEAM inference flow
Same, except for chunks n≥1: build `a_aligned = Extend(prev_model_chunk[K:H], H)`; sample with VLS
`guidance_fn` (traced `aligned_prior`, `lam`, `enabled`); at each Euler step apply
`x_next[:M,dims] += λ(1-t_next)·(-2)(x_cand[:M,dims]-(1-t_next)a_aligned[:M,dims])`. First chunk / reset
→ baseline (no VLS).

## 16. Previous-chunk state lifecycle
`SeamState{previous_chunk_model_space[10,32], previous_chunk_physical_space[10,7], chunk_index}`
(immutable; `state.py`). Recorded **before** the executor discards the tail. One instance per episode;
`reset()` clears both representations (first chunk after reset is baseline). No process-global state.

## 17. Interface used by VLSGuidance
`DenoisingGuidance.update(candidate, t_next) -> corrected` (`guidance/base.py`), exposed to the sampler
as a plain closure via `.as_fn()`. `VLSGuidance` (`guidance/vls.py`) holds `aligned_prior`, `lam`,
`pos_mask`, `dim_mask`, `enabled`. Wired in `integration/openpi_jax.SeamSampler`.

## 18. Interface reserved for DecodedChunkRefiner
`DecodedChunkRefiner.refine(physical_chunk, context)` (`refinement/base.py`) — operates in physical
space after Unnormalize, before execution. Only `IdentityChunkRefiner` for this work; this is the
future trajectory-optimization / MPC seam, kept separate from `DenoisingGuidance`.

## 19. Unit-test commands
```
# fast suite (no model)
JAX_PLATFORMS=cpu src/openpi/.venv/bin/python -m pytest benchmark/seam_vla/tests/ -q
# heavy real-model tests (CPU; ~minutes)
JAX_PLATFORMS=cpu SEAM_RUN_MODEL_TESTS=1 src/openpi/.venv/bin/python -m pytest \
  benchmark/seam_vla/tests/test_baseline_parity.py \
  benchmark/seam_vla/tests/test_openpi_integration.py -q
# pure-core on GPU
XLA_PYTHON_CLIENT_PREALLOCATE=false src/openpi/.venv/bin/python -m pytest \
  benchmark/seam_vla/tests/test_batch_shapes.py -q
```

## 20. Unit-test results
- Fast suite: **79 passed, 7 skipped** (7 = 6 heavy model + 1 GPU, skipped without flags/GPU;
  re-run 2026-07-27 with `PYTHONPATH=src`).
- Heavy real-model suite (CPU): **8 tests, exit 0 (all passed)**.
- Pure-core on GPU (CUDA device): **7 passed**.
Coverage maps to all 35 spec items (first-chunk/reset bypass, aligned-prior shape/tail/repeat, H/K
validation, empty-tail reject, M>L clamp, M=0/λ=0/disabled→baseline, unguided positions & dims ==
candidate, explicit masks, batch + arbitrary dims, JIT, no-autodiff source check, no NaN/Inf,
dtype/device preserved, CPU/GPU, norm round-trip, second chunk gets first chunk's model tail, mask
excludes padding, BJ/IJ/CD/AVb synthetic + boundary indices, Identity guidance/refiner no-ops, OpenPI
hook parity).

## 21. CPU and GPU test status
- CPU: all pure + heavy model tests pass.
- GPU: pure-core tests pass on the CUDA device. The **full pi0.5 model does not fit in 8 GB** (OOM), so
  model-level GPU tests are intentionally not run here; the GPU-only pure test auto-skips under
  `JAX_PLATFORMS=cpu` and runs on GPU otherwise. No CUDA result is claimed that was not measured.

## 22. Baseline-parity test results
Real model, matched rng+noise, CPU: `max| sample_actions(None) − SEAM(enabled=0) | = 0.000e+00`;
`max| baseline − SEAM(λ=0) | = 0.000e+00`; `max| baseline − SEAM(enabled=1) | = 1.459e-01 (>0, finite)`.
Model-space shape `(1,10,32)`.

## 23. Single-task smoke-test command
```
JAX_PLATFORMS=cpu src/openpi/.venv/bin/python -m benchmark.seam_vla.experiments.libero.smoke_test --sim-free
# or, with the LIBERO sim installed and a >=12GB GPU:
python -m benchmark.seam_vla.experiments.libero.smoke_test --config benchmark/seam_vla/configs/smoke_libero10.yaml
```

## 24. Single-task validation results
Not executed per user decision ("code/tests only"). The sim-free harness is written and imports
cleanly; it validates shapes, effective H/K/L/M/N/D, dim mask, model-space carry, VLS effect &
correction norms, first-chunk/reset behavior, NaN/Inf, and compile-vs-steady latency. The real-env
rollout requires the LIBERO container. See `reproduction_notes.md`.

## 25. Effective correction magnitudes
On the real model with matched noise, one VLS-guided chunk changed the model-space output by up to
0.146 (max element-wise) at λ=0.1, M=5 (finite, no NaN). Per-chunk correction norms are logged by
`SeamPolicy` (`ChunkDiagnostics.correction_norm`); per-step norms are available via the sim-free smoke.

## 26. Timing methodology
`metrics/latency.time_call`: one warm-up (compile) pass reported separately, then N synchronized
steady-state repeats, each finalized with `block_until_ready()`. Compile time is never counted in
steady-state. GPU numbers require a GPU that fits the model (not this 8 GB card).

## 27. Full LIBERO-10 evaluation command
```
# 1) baseline
python -m benchmark.seam_vla.experiments.libero.evaluate_baseline \
  --config benchmark/seam_vla/configs/full_libero10.yaml --out /tmp/seam_baseline.json
# 2) SEAM (requires explicit authorization)
python -m benchmark.seam_vla.experiments.libero.evaluate_seam \
  --config benchmark/seam_vla/configs/full_libero10.yaml --run-full-benchmark --out /tmp/seam_seam.json
# 3) compare
python -m benchmark.seam_vla.experiments.libero.compare_results \
  --baseline /tmp/seam_baseline.json --seam /tmp/seam_seam.json
```
The full 10×130 run refuses to start without `--run-full-benchmark`.

## 28. Full benchmark results
Not executed (LIBERO sim not installed; 8 GB GPU). No fabricated numbers.

## 29. Baseline-vs-SEAM success/BJ/IJ/CD/AVb/latency
Pending a proper environment (see §27–28). The comparison table is produced by `compare_results.py`
(Success, BJ, IJ, CD, AVb + correction norms + wall time).

## 30. Known limitations
- H=10 checkpoint ⇒ L=5, M≤5; paper's L=40/M=20 regime not reproducible here.
- 8 GB GPU cannot host pi0.5; model runs validated on CPU only.
- LIBERO sim absent ⇒ no task-success or on-hardware latency numbers here.
- Per-step VLS correction norms require the sim-free debug path (while_loop hides intermediate states);
  per-chunk norms are always available.

## 31. Differences from the paper caused by the actual repository
See `reproduction_notes.md` §1. Summary: derived profile H=10/K=5/L=5/M≤5 vs. paper 50/10/40/20; K from
`replan_steps=5`; quantile (not z-score) normalization; state token not consumed by the net
(`discrete_state_input=False`); in-process rollout instead of the websocket client/server.
