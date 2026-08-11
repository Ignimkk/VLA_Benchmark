"""P0a — attention extraction plumbing probe for the KNOWS reverse engineering.

Validates, against the real ``pi05_libero`` checkpoint, every shape claim made in
``papers/KNOWS/02-architecture.md`` §3.4, and checks that the ``return_probs`` hook added to
``openpi/models/gemma.py`` does not perturb the baseline forward pass.

This probe does NOT test the paper's scientific claim (that layer 12 / head 3 localizes the
policy's current target) -- that is P0b, which needs real scenes with object masks. This only
establishes that we can read the right numbers out of the right tensor.

Run (CPU is fine, ~minutes):
    JAX_PLATFORMS=cpu src/openpi/.venv/bin/python -m benchmark.knows_vla.probe_p0a
"""

from __future__ import annotations

import einops
import jax
import jax.numpy as jnp
import numpy as np

# Values the analysis predicts; see papers/KNOWS/02-architecture.md §3.4.
EXPECTED = {
    "n_layers": 18,          # gemma_2b depth
    "n_heads": 8,            # gemma_2b num_heads
    "n_kv_heads": 1,         # GQA
    "tokens_per_camera": 256,  # So400m/14 at 224x224 -> (224/14)^2
    "g": 16,
    "n_cameras": 3,          # base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
    "vision_tokens": 768,
    "language_tokens": 200,  # max_token_len for pi05
    "prefix_len": 968,
    "action_horizon": 10,    # pi05_libero; the paper uses H=8
    "paper_layer": 12,
    "paper_head": 3,
    "agentview_slice": (0, 256),
}


def _load(*, attn_probs: bool):
    """Load pi05_libero, optionally with the attention-probability hook enabled.

    The flag adds no parameters, so both variants load the same checkpoint.
    """
    import dataclasses

    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    from openpi.policies import libero_policy
    from openpi.models import model as _model
    from openpi.shared import download

    cfg = _config.get_config("pi05_libero")
    if attn_probs:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, return_attn_probs=True))
    ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    policy = _pc.create_trained_policy(cfg, ckpt)

    # make_libero_example() draws RANDOM images and state, so seed it: the flag-on and flag-off
    # models must be fed byte-identical observations or the comparison below is meaningless.
    np.random.seed(0)
    example = libero_policy.make_libero_example()
    inputs = policy._input_transform(dict(example))  # noqa: SLF001
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    obs = _model.Observation.from_dict(inputs)
    return policy, obs


def _forward_with_probs(model, obs, noise, *, denoise_step: int = 0, num_steps: int = 10):
    """Replicate Pi0.sample_actions' prefix+suffix passes, returning suffix attention probs.

    Mirrors src/openpi/src/openpi/models/pi0.py:232-291. Keeps the probs of a single Euler step
    (``denoise_step``), because the paper does not say which step it reads (OPEN-Q 11) and we want
    to be able to sweep it. The model must have been built with ``return_attn_probs=True``.
    """
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    obs = _model.preprocess_observation(None, obs, train=False)
    batch = obs.state.shape[0]
    dt = -1.0 / num_steps

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    pre = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    kv_cache = pre[1]

    x_t = noise
    time = 1.0
    probs = None
    for step in range(num_steps):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            obs, x_t, jnp.broadcast_to(time, batch)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        pre_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([pre_mask, suffix_attn_mask], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        out = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=pos,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        (_, suffix_out), _, step_probs = out if len(out) == 3 else (*out, None)
        if step == denoise_step:
            probs = step_probs

        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
        x_t = x_t + dt * v_t
        time = time + dt

    return x_t, probs, int(prefix_tokens.shape[1]), int(suffix_tokens.shape[1])


def main() -> int:
    print("=" * 74)
    print("P0a — attention extraction plumbing")
    print("=" * 74)

    policy, obs = _load(attn_probs=True)
    model = policy._model  # noqa: SLF001
    noise = jnp.zeros((1, model.action_horizon, model.action_dim), dtype=jnp.float32)
    actions, probs, prefix_len, suffix_len = _forward_with_probs(model, obs, noise)

    # ---- 1. sequence layout -------------------------------------------------
    n_img = len(obs.images)
    per_cam = (prefix_len - EXPECTED["language_tokens"]) // n_img
    checks: list[tuple[str, object, object]] = [
        ("cameras in prefix", n_img, EXPECTED["n_cameras"]),
        ("tokens per camera", per_cam, EXPECTED["tokens_per_camera"]),
        ("prefix length", prefix_len, EXPECTED["prefix_len"]),
        ("suffix length (=H)", suffix_len, EXPECTED["action_horizon"]),
    ]

    # ---- 2. probs tensor ----------------------------------------------------
    L, B, K, G, T, S = probs.shape
    checks += [
        ("probs layers L", L, EXPECTED["n_layers"]),
        ("probs kv-heads K", K, EXPECTED["n_kv_heads"]),
        ("probs query-heads G", G, EXPECTED["n_heads"]),
        ("probs query len T", T, EXPECTED["action_horizon"]),
        ("probs key len S", S, prefix_len + suffix_len),
    ]

    ok = True
    print(f"\n{'check':<28} {'actual':>10} {'expected':>10}   status")
    print("-" * 74)
    for name, actual, expected in checks:
        good = actual == expected
        ok &= good
        print(f"{name:<28} {actual!s:>10} {expected!s:>10}   {'OK' if good else 'MISMATCH'}")

    # ---- 3. the agent-view slice the paper reads ----------------------------
    lo, hi = EXPECTED["agentview_slice"]
    a_raw = np.asarray(probs[EXPECTED["paper_layer"], 0, 0, EXPECTED["paper_head"], :, lo:hi], dtype=np.float32)
    g = int(round(np.sqrt(a_raw.shape[1])))
    grid = a_raw.mean(axis=0).reshape(g, g)  # mean over the H action queries -- OPEN-Q 10

    print("-" * 74)
    print(f"agent-view block  probs[L={EXPECTED['paper_layer']}, b=0, k=0, "
          f"g={EXPECTED['paper_head']}, :, {lo}:{hi}] -> {a_raw.shape}")
    print(f"grid g x g                 {g}x{g} (expected {EXPECTED['g']}x{EXPECTED['g']})   "
          f"{'OK' if g == EXPECTED['g'] else 'MISMATCH'}")
    ok &= g == EXPECTED["g"]

    # Attention over the FULL key axis must sum to 1 per query; the agent-view slice is a
    # fraction of that. probs is stored in bfloat16, so accumulate in float64 -- summing 978
    # bf16 values in bf16 loses ~2%, which is a measurement artifact, not a model property.
    full_row = np.asarray(probs[EXPECTED["paper_layer"], 0, 0, EXPECTED["paper_head"]], dtype=np.float64)
    full_sum = float(full_row.sum(axis=-1).mean())
    slice_mass = float(a_raw.astype(np.float64).sum(axis=1).mean())
    print(f"softmax row sum (full S)   {full_sum:.6f} (expected 1.0)   "
          f"{'OK' if abs(full_sum - 1.0) < 5e-3 else 'MISMATCH'}")
    ok &= abs(full_sum - 1.0) < 5e-3
    print(f"agent-view share of mass   {slice_mass:.4f}")
    print(f"grid min/max/mean          {grid.min():.3e} / {grid.max():.3e} / {grid.mean():.3e}")

    # ---- 4. does the flag perturb the model? --------------------------------
    # The claim to defend is narrow: turning return_attn_probs on must not change what the model
    # computes. Compare the SAME manual code path with the flag off vs on -- that isolates the
    # flag. Comparing against sample_actions instead would conflate it with jitted-while_loop vs
    # eager execution order, which differs in bfloat16 for reasons unrelated to this change.
    del policy, model
    base_policy, base_obs = _load(attn_probs=False)
    base_model = base_policy._model  # noqa: SLF001
    off_actions, off_probs, _, _ = _forward_with_probs(base_model, base_obs, noise)

    on, off = np.asarray(actions, dtype=np.float32), np.asarray(off_actions, dtype=np.float32)
    max_diff = float(np.abs(on - off).max())
    same = max_diff == 0.0
    print("-" * 74)
    print(f"flag off returns no probs  {off_probs is None}   {'OK' if off_probs is None else 'MISMATCH'}")
    ok &= off_probs is None
    print(f"flag on vs off, same path  max|diff| = {max_diff:.3e}   "
          f"{'OK (byte-identical)' if same else 'MISMATCH'}")
    ok &= same

    # Informational: eager manual loop vs the jitted while_loop in sample_actions.
    sampled = np.asarray(base_model.sample_actions(jax.random.key(0), base_obs, noise=noise), dtype=np.float32)
    print(f"  (info) manual vs sample_actions max|diff| = {float(np.abs(sampled - off).max()):.3e}, "
          f"action scale = {float(np.abs(sampled).max()):.3f}")

    print("=" * 74)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
