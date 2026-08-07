# SEAM — OpenPI / π0.5 / LIBERO Code Inspection

This document records the **source-verified** facts required to implement SEAM (VLS) without guessing
API names, shapes, normalization conventions, or coordinate systems. All paths are relative to
`src/openpi/src/openpi/` unless stated otherwise. Line numbers reflect the checkout inspected on
2026-07-21.

> **Headline discrepancy with the paper.** The paper reports π0.5 with `H=50`, `K=10`, `L=40`, `M=20`.
> The only local checkpoint is `pi05_libero`, which bakes **`action_horizon=10`** into the trained
> weights, and the LIBERO rollout executes **`replan_steps=5`**. Therefore the *effective* profile on
> this checkpoint is **`H=10, K=5, L=5, N=10, D=32 (7 valid), M≤5`**. `H` cannot be changed without a
> different checkpoint. Per the task spec, SEAM asserts the derived values and fails loudly if a config
> demands the paper profile; it never silently rewrites baseline behavior. See
> `reproduction_notes.md`.

## 1. π0.5 inference entry point
`policies/policy.py` → `Policy.infer(obs, *, noise=None)` (lines 67-106). Top-level call that takes a
raw observation dict and returns `{"state", "actions", "policy_timing"}`. `actions` are **physical
space** (post output-transform, sliced to 7 dims).

## 2. Complete call path (policy → action generation)
`policy_config.create_trained_policy` assembles the transform pipelines (`policies/policy_config.py:75-94`):

```
INPUT  (policy.py:71  self._input_transform):
  repack.inputs → InjectDefaultPrompt → LiberoInputs (+optional DeltaActions)
  → Normalize(quantile)                       # policy_config.py:81  -> MODEL SPACE (normalized, 32-dim)
  → ResizeImages, TokenizePrompt, PadStatesAndActions

SAMPLE (policy.py:94):
  self._sample_actions(rng, observation, **sample_kwargs)   # jitted Pi0.sample_actions
  -> raw actions, shape [1, H=10, D=32]  (MODEL SPACE)

OUTPUT (policy.py:102  self._output_transform):
  (model_transforms.outputs: empty for pi05)
  → Unnormalize(quantile)                      # policy_config.py:86  -> PHYSICAL SPACE
  → LiberoOutputs  actions[..., :7]            # libero_policy.py:100 -> 7 valid dims
  → repack.outputs
```

## 3. Function that generates an action chunk
`models/pi0.py` → `Pi0.sample_actions(self, rng, observation, *, num_steps=10, noise=None)` (lines
216-279). Returns `x_0`, shape `[b, H, D]`, model space (normalized).

## 4. Exact flow-matching Euler denoising loop
`models/pi0.py:239-278`. `dt = -1/num_steps = -0.1` (line 228). Time convention (explicit comment,
lines 226-227): **t=1 is noise, t=0 is target** (opposite of the pi0 paper). `time` starts at `1.0`
(line 278) and decreases. Velocity `v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])`
(line 269), shape `[b, H, D]`. **Euler candidate** = `x_t + dt * v_t` at **line 271**
(`return x_t + dt * v_t, time + dt`). This is the exact post-Euler hook point. Loop termination
`cond`: `time >= -dt/2` (line 276).

## 5. Loop structure
**`jax.lax.while_loop`** (line 278), carry `(x_t, time)`. Not a Python for-loop, not `scan`, not
`fori_loop`. Any guidance inserted at line 271 must be pure/traceable JAX (no Python-side per-step
state, no host transfers).

## 6. Initial denoising state
`noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))` (line 231), shape
`[b, H=10, D=32]`. `while_loop` initialized with `(noise, 1.0)`.

## 7. Velocity field I/O shapes
Input to `embed_suffix(observation, x_t, time)`: `x_t` `[b, H, D]`, `time` broadcast to `[b]`.
Output velocity `v_t`: `[b, H, D]` (`action_out_proj`, `nnx.Linear(width, action_dim)`, line 100).

## 8. Action horizon H
`action_horizon` config, default 50 (`models/pi0_config.py:26`), **overridden to 10** for `pi05_libero`
(`training/config.py:745`). **Effective H = 10** (checkpoint-baked).

## 9. Number of Euler steps N
`num_steps` argument default **10** (`models/pi0.py:222`). Not overridden by any JAX config in the repo.
**Effective N = 10**.

## 10. Model action dimension D
`action_dim` config, default **32** (`models/pi0_config.py:25`), not overridden for `pi05_libero`.
**Effective D = 32.**

## 11. Valid physical action dimensions
**7** (arm 6 delta EEF-pose + 1 gripper). Determined by `LiberoOutputs` slice `data["actions"][..., :7]`
(`policies/libero_policy.py:100`) and confirmed by `norm_stats.json` (`actions.q01` has length 7). Model
dims 7..31 are zero-padding, added by `PadStatesAndActions` (`transforms.py:327-337`, wired at
`training/config.py:136`). **SEAM guides only dims 0..6.**

## 12. Padded / unused dimensions
Yes. State is 8-dim, actions 7-dim; both padded to 32 by `PadStatesAndActions`. Additionally
`discrete_state_input=False` for `pi05_libero` (`config.py:745`) means the state token is **not
consumed** by the network (`models/pi0.py:151-157`; no `state_proj` in pi05). Padded action dims pass
through `Unnormalize` unchanged (`transforms.py:179-180` concatenates identity beyond `q01.shape[-1]`).

## 13. Coordinate system of the denoising state
The denoising state `x_t` lives in the **normalized model space** (quantile-normalized to ~[-1,1] for
valid dims, zeros for padded dims). SEAM's aligned prior is built in this same space (see
`normalization_and_shapes.md`).

## 14. Normalization (training & inference)
**Quantile** normalization for pi05 (`use_quantile_norm = model_type != PI0`, `config.py:187`).
`_normalize_quantile`: `(x - q01)/(q99 - q01 + 1e-6) * 2 - 1` (`transforms.py:141-145`). NormStats
(`shared/normalize.py:9-14`): `mean, std, q01, q99`. Stats file:
`<ckpt>/assets/physical-intelligence/libero/norm_stats.json` (asset_id resolves to
`physical-intelligence/libero`, `config.py:181`).

## 15. Unnormalization & output transform
`Unnormalize` (`transforms.py:148-181`), `_unnormalize_quantile`:
`(x + 1)/2 * (q99 - q01 + 1e-6) + q01` (lines 175-181). Invoked at `policy_config.py:86`. Then
`LiberoOutputs` slices to 7 dims (`libero_policy.py:100`). No delta/absolute output transform for
`pi05_libero` (`extra_delta_transform=False`, `config.py:749`).

## 16. Gripper representation & postprocessing
Action dim index 6 (7th) is the gripper. Returned as-is by `LiberoOutputs` (no remap). LIBERO uses
6 arm deltas + 1 gripper (absolute-ish); dummy warmup action is `[0.0]*6 + [-1.0]`
(`examples/libero/main.py:17`), i.e. gripper open = -1.0.

## 17. Existing action-chunk caching semantics
`examples/libero/main.py`: `action_plan = collections.deque()` (line 94). When empty, request a chunk;
push only the first `replan_steps` actions (`action_plan.extend(action_chunk[: replan_steps])`, line
148); `popleft()` each env step (line 150). `ActionChunkBroker`
(`packages/openpi-client/.../action_chunk_broker.py`) exists but is **not used** by LIBERO.

## 18. Rollout location that executes first K actions
`examples/libero/main.py:127-150` (deque fill + popleft). `env.step(action.tolist())` at line 153.

## 19. Actual K used by LIBERO eval
`replan_steps = 5` (`examples/libero/main.py:29`). **Effective K = 5** ⇒ **L = H − K = 5**.

## 20. RNG keys & initial Gaussian noise
`Policy._rng = rng or jax.random.key(0)` (`policy.py:65`); split each `infer` call
(`self._rng, sample_rng = jax.random.split(self._rng)`, line 75). Noise sampled in `sample_actions`
via `jax.random.normal` unless an explicit `noise` array is passed through `sample_kwargs`
(`policy.py:83-88`) — used for deterministic eval / parity tests.

## Transport note (rollout)
LIBERO eval is **websocket client/server** (`examples/libero/main.py:73`, `scripts/serve_policy.py`);
the model runs server-side. Because SEAM needs in-process access to the **model-space** chunk and
per-episode state, the benchmark provides a **new in-process rollout** (built via
`create_trained_policy` against the local checkpoint) rather than reusing the websocket path. The
existing client/server code is left untouched.

## Local checkpoint
`~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/{params, assets/physical-intelligence/libero/norm_stats.json}`.
JAX 0.5.3, one CUDA device present.
