# SEAM Reproduction Notes — differences from the paper

## 1. Horizon / overlap: paper vs. this checkpoint

| Quantity | Paper (π0.5) | This repo (`pi05_libero`) | Why it differs |
|---|---|---|---|
| H (action_horizon) | 50 | **10** | Baked into the trained checkpoint (`training/config.py:745`). Cannot change without different/retrained weights. |
| K (executed/chunk) | 10 | **5** | `replan_steps=5` in the LIBERO rollout (`examples/libero/main.py:29`). |
| L = H − K (overlap) | 40 | **5** | Derived. |
| M (guided window) | 20 | **≤5** (clamped) | `M_eff = min(M, L) = min(20, 5) = 5`. |
| N (Euler steps) | 10 | 10 | Matches. |
| D (model / valid) | – / 7 | 32 / 7 | Matches LIBERO 7-DoF (6 arm + gripper), padded to 32. |

**Consequence.** The paper's operating regime (a 40-step overlap with a 20-step guided window) is
physically unavailable on this checkpoint. SEAM's *method* is reproduced faithfully, but on a much
shorter overlap (L=5, M≤5). Absolute jerk/discontinuity numbers are therefore not directly comparable
to the paper's Table 2; the meaningful comparison here is **baseline vs. SEAM on the same H=10/K=5
checkpoint**.

Per the task spec, the config asserts the derived values (`seam_assert_baseline_shape=true`) and raises
a clear error if a config demands H=50/K=10 — SEAM never silently rewrites the baseline. Verified: a
`SeamConfig(seam_horizon=50, seam_execution_length=10)` raises `SeamConfigError`.

## 2. Environment constraints in this workspace

- **GPU**: an 8 GB RTX 4060 Ti (paper used a 24 GB RTX 3090). The ~2.3B-param pi0.5 model
  (PaliGemma-2B + Gemma-300M, bf16 ≈ 4.6 GB params + ~2.4 GB activation peak) **does not fit**; a single
  inference OOMs. All model-dependent validation was run on **CPU** (`JAX_PLATFORMS=cpu`, ~8 s/chunk).
- **LIBERO simulator not installed** (`libero`/`robosuite`/`mujoco` absent). The upstream eval runs the
  sim in a separate container and talks to the model server over websocket
  (`scripts/SETUP_CONTAINER.md`). Therefore the real-environment LIBERO rollout and the full
  10-task × 130-episode benchmark **cannot be executed in this workspace**. The in-process rollout code
  (`experiments/libero/`) is written and import-verified, and the exact run commands are prepared.

## 3. What was validated here (per user decision: code + tests only)

- Baseline parity on the real model (CPU): unmodified `sample_actions` vs. SEAM(enabled=0) differ by
  **exactly 0.0**; SEAM(enabled=1) changes the output by a finite amount (≈0.146 max) with no NaN/Inf.
- 55 fast unit tests pass; 8 heavy real-model tests pass on CPU (exit 0); GPU execution of the pure core
  passes (7 tests on the CUDA device).

## 4. What still needs a proper environment

- Task-success numbers (needs the LIBERO sim + a ≥ ~12 GB GPU for practical speed).
- Full Table-2-style baseline-vs-SEAM comparison over 10 tasks × 130 episodes.
- Steady-state denoising-loop latency on GPU (the paper's 1.01× figure) — CPU timing is not comparable.

The commands to produce all of the above are in `implementation_report.md` §27.
