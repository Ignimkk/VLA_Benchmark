# SEAM — Normalization, Coordinate Systems, and Tensor Shapes

SEAM maintains **two distinct** previous-chunk representations. They must never be mixed in one VLS
computation.

## Two coordinate systems

### Model space (`previous_chunk_model_space`)
- The raw output of `Pi0.sample_actions`, **before** `Unnormalize` and output transforms.
- Shape `[H, D] = [10, 32]` (per env; batched `[B, 10, 32]`).
- Quantile-normalized: valid dims 0..6 ≈ `[-1, 1]`; padded dims 7..31 are model outputs near 0 (never
  guided).
- **This is the space in which VLS operates.** The denoising state `x_t` and the aligned prior share
  it, so the closed-form target `r = (1-t)·a_aligned` is dimensionally consistent with the flow-path
  interpolation `x_t = (1-t)·x_0 + t·ε` used by the model (`models/pi0.py:199`).
- We retain the model-space chunk **directly** from the sampler and do **not** round-trip it through
  `Unnormalize`→`Normalize`. Quantile normalization + clipping + padding make that round trip
  non-exactly-invertible, so re-normalizing a physical chunk to rebuild the prior is prohibited.

### Physical space (`previous_chunk_physical_space`)
- After `Unnormalize` (`policy_config.py:86`) and `LiberoOutputs[..., :7]` (`libero_policy.py:100`).
- Shape `[H, 7] = [10, 7]` (per env).
- Units: 6 arm EEF-pose deltas + 1 gripper command; this is what `env.step` consumes.
- Used **only** for: environment execution, rollout logging, action-range/gripper validation, motion
  metrics (BJ/IJ/CD/AVb), and visualization. **Never** used to build the VLS prior.

## Normalization details (quantile, pi05)
- Forward (`transforms._normalize_quantile`, `transforms.py:141-145`):
  `x_norm = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1`.
- Inverse (`transforms._unnormalize_quantile`, `transforms.py:175-181`):
  `x_phys = (x_norm + 1) / 2 * (q99 - q01 + 1e-6) + q01`.
- Applied only to the first `q01.shape[-1]` dims (7 for actions); dims beyond pass through identity
  (`transforms.py:179-180`). Stats live in
  `<ckpt>/assets/physical-intelligence/libero/norm_stats.json` (`actions`: 7-dim; `state`: 8-dim).

## Shape reference

| Tensor | Shape | Space |
|---|---|---|
| Initial noise `x_1` | `[B, 10, 32]` | model (Gaussian) |
| Denoising state `x_t` / Euler candidate | `[B, 10, 32]` | model |
| Velocity `v_t` | `[B, 10, 32]` | model |
| SEAM aligned prior `a_aligned` | `[B, 10, 32]` | model |
| Guided-window slice `a_aligned[:, 0:M]` | `[B, M≤5, 32]` | model |
| Guided-dim mask | broadcastable to `[B, 10, 32]`, True on dims 0..6 | model |
| Raw sampler output `c_n` (model) | `[10, 32]` (per env) | model |
| Physical chunk (post-transform) | `[10, 7]` (per env) | physical |
| Executed action | `[7]` per env step | physical |

## VLS derived profile (this checkpoint)
`H=10, K=5, L=H−K=5, N=10, D=32, valid_dims=7, M=min(seam_guided_window, L)` with default
`seam_guided_window=5` ⇒ `M_eff=5`. Guided dims = `{0,1,2,3,4,5,6}` by default (`all_physical`).

## Tail semantics (model space)
`a_tail = c_n[K:H] = c_n[5:10]` (length L=5); `a_aligned = Extend(a_tail, H)` = the 5 tail rows plus the
last tail row repeated to reach length H=10. Only rows `0:M` of `a_aligned` receive guidance; the
repeated rows beyond L are inert padding (they only make the prior length-compatible).
