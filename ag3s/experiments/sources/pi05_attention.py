"""Extract pi0.5 cross-attention over the three RB-Y1 camera grids, for a recorded rollout.

**This runs where the checkpoint is** -- the GPU server -- not on the machine driving the MuJoCo
viewer. The policy server speaks a websocket protocol that returns actions and nothing else, so
attention cannot be pulled out of a running server; it needs a forward pass with
`Pi0Config.return_attn_probs=True`, which is inference-only, adds no parameters, and loads the same
checkpoint unchanged.

The method is `benchmark/knows_vla/probe_p0b.py` retargeted from LIBERO to RB-Y1. Two things change:

**Three cameras, not two.** `AlohaInputs` emits its image dict in the order base / left wrist /
right wrist, and `Pi0.embed_prefix` concatenates image tokens in that order, so prefix positions
0-255, 256-511 and 512-767 are the three 16x16 SigLIP grids. Those slices are read from
`policy_record.CAMERA_BINDINGS` rather than written here, so the probe and the analysis cannot
disagree about which camera is which.

**No state token in the suffix.** pi0.5 puts state in the discrete language tokens, so the suffix is
exactly `action_horizon` action tokens and query index t is the attention of predicted action step t.

Attention is read at several denoising steps because it is not constant across them: the first Euler
step sees pure noise as its action prefix and the last sees an almost-final trajectory. Which step
localizes best is an empirical question, so the probe records a few and lets the analysis decide
rather than baking in a choice.

Run (on the GPU server, from the workspace root):

    src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.sources.pi05_attention \
        --records outputs/rby1_atomic_infer/<run>/ag3s_records/run_0000 \
        --checkpoint /path/to/checkpoints/pi05_rby1_lora/<run>/<step> \
        --out benchmark/ag3s/asset/attention/run_0000.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time

import numpy as np

from benchmark.ag3s.experiments.sources.policy_record import (
    ATTENTION_GRID,
    CAMERA_BINDINGS,
    POLICY_CAMERA_NAMES,
    load_run,
)

#: Query-token aggregations. The suffix has one token per predicted action step; `mean` asks what
#: the chunk looks at overall, `first` what the immediately-next action looks at, `last` what the end
#: of the horizon looks at. They differ, and which one grounds best is measured, not assumed.
AGGREGATIONS = {
    "mean": lambda a: a.mean(axis=-2),
    "first": lambda a: a[..., 0, :],
    "last": lambda a: a[..., -1, :],
}


def load_policy(config_name: str, checkpoint: str):
    """The trained policy with attention probabilities switched on."""
    from openpi.policies import policy_config as _pc
    from openpi.shared import download
    from openpi.training import config as _config

    cfg = _config.get_config(config_name)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, return_attn_probs=True))
    path = download.maybe_download(checkpoint)
    return _pc.create_trained_policy(cfg, path)


class AttentionSampler:
    """Jitted prefix/suffix passes that hand back the prefix attention block.

    Two small executables compiled once and reused. Calling the nnx module eagerly instead compiles
    a fresh executable per call and exhausts XLA's section memory after a few dozen steps -- the
    same failure `probe_p0b` and `SeamSampler` were written around.
    """

    def __init__(self, model, *, num_steps: int = 10):
        import flax.nnx as nnx
        import jax

        self._jax = jax
        self._nnx = nnx
        self._graphdef, self._state = nnx.split(model)
        self._num_steps = int(num_steps)
        self._action_horizon = int(model.action_horizon)
        self._action_dim = int(model.action_dim)
        self._jit_prefix = jax.jit(self._prefix)
        self._jit_suffix = jax.jit(self._suffix)

    def _prefix(self, state, obs):
        import jax.numpy as jnp

        from openpi.models.pi0 import make_attn_mask

        model = self._nnx.merge(self._graphdef, state)
        tokens, mask, ar_mask = model.embed_prefix(obs)
        attn = make_attn_mask(mask, ar_mask)
        positions = jnp.cumsum(mask, axis=1) - 1
        kv = model.PaliGemma.llm([tokens, None], mask=attn, positions=positions)[1]
        return kv, mask

    def _suffix(self, state, obs, x_t, time, kv, prefix_mask):
        import einops
        import jax.numpy as jnp

        from openpi.models.pi0 import make_attn_mask

        model = self._nnx.merge(self._graphdef, state)
        batch = obs.state.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            obs, x_t, jnp.broadcast_to(time, (batch,))
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        pre_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([pre_mask, suffix_attn_mask], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (_, suffix_out), _, probs = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=pos,
            kv_cache=kv,
            adarms_cond=[None, adarms_cond],
        )
        v_t = model.action_out_proj(suffix_out[:, -self._action_horizon :])
        n_image_tokens = ATTENTION_GRID * ATTENTION_GRID * len(CAMERA_BINDINGS)
        # probs is [L, B, K, G, T, S]; gemma_2b is multi-query so K == 1 and G is the head axis.
        # -> [L, heads, action_tokens, image_tokens]
        return v_t, probs[:, 0, 0, :, :, :n_image_tokens].astype(jnp.float32)

    def attention(self, obs, denoise_steps, *, noise_seed: int = 0):
        """`{denoise_step: [L, heads, action_tokens, 768]}` for the requested Euler steps.

        The denoising loop is started from **Gaussian noise, not zeros**, because that is what
        `Pi0.sample_actions` does. It matters more than it looks: at t = 1 with x_t = 0 every action
        token is the same `action_in_proj` bias, so all fifty suffix queries are identical and the
        attention they produce is identical too. Comparing `first` against `last` pooling on that
        would be comparing a number against itself and calling the result a finding.

        `noise_seed` is fixed rather than drawn: the server's own per-request RNG state is not
        recoverable, so what this can offer is reproducibility rather than an exact replay. Pass
        several seeds to check the ranking does not depend on the draw.
        """
        import jax
        import jax.numpy as jnp

        from openpi.models import model as _model

        # preprocess_observation is not idempotent -- exactly once, before the prefix pass.
        obs = _model.preprocess_observation(None, obs, train=False)
        kv, prefix_mask = self._jit_prefix(self._state, obs)

        wanted = set(int(s) for s in denoise_steps)
        dt = -1.0 / self._num_steps
        batch = obs.state.shape[0]
        x_t = jax.random.normal(
            jax.random.key(int(noise_seed)), (batch, self._action_horizon, self._action_dim),
            jnp.float32,
        )
        # Traced, not a Python float, or jit recompiles once per value.
        time = jnp.asarray(1.0, jnp.float32)
        out = {}
        for step in range(self._num_steps):
            v_t, probs = self._jit_suffix(self._state, obs, x_t, time, kv, prefix_mask)
            if step in wanted:
                out[step] = np.asarray(probs)
            x_t = x_t + dt * v_t
            time = time + dt
        return out


def observation_from_step(policy, step, prompt):
    """The recorded step, put through the policy's own input transform."""
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model

    images = {
        key: np.transpose(step.images[key], (2, 0, 1))  # HWC -> CHW, as pi05_infer sends it
        for key in POLICY_CAMERA_NAMES
    }
    inputs = policy._input_transform(  # noqa: SLF001
        {"state": step.state, "images": images, "prompt": prompt}
    )
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    return _model.Observation.from_dict(inputs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True, help="a run_XXXX directory written by --record-ag3s")
    ap.add_argument("--checkpoint", required=True, help="fine-tuned pi0.5 checkpoint directory")
    ap.add_argument("--config", default="pi05_rby1_lora")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--num-denoise", type=int, default=10, help="Euler steps the sampler takes")
    ap.add_argument("--denoise-steps", type=int, nargs="+", default=[0, 4, 9],
                    help="which Euler steps to record attention at")
    ap.add_argument("--agg", nargs="+", default=["mean", "first", "last"], choices=sorted(AGGREGATIONS))
    ap.add_argument("--store-dtype", default="float16", choices=("float16", "float32"),
                    help="on-disk dtype. Attention values sit near 1/768, and float16 carries them "
                         "with about 0.2%% relative error once summed over a 16x16 patch grid -- "
                         "far below anything that could reorder two heads, for half the transfer")
    ap.add_argument("--limit", type=int, default=None, help="only the first N recorded steps")
    ap.add_argument("--noise-seeds", type=int, nargs="+", default=[0],
                    help="flow-matching noise seeds the denoising loop is started from; attention "
                         "is averaged over them. More than one costs proportionally more time and "
                         "answers whether the ranking survives the draw")
    ap.add_argument("--prompt", default=None, help="override the recorded prompt")
    args = ap.parse_args()

    run = load_run(args.records, limit=args.limit)
    prompt = args.prompt or run.prompt
    print(f"{len(run)} recorded steps | prompt={prompt!r}")

    policy = load_policy(args.config, args.checkpoint)
    sampler = AttentionSampler(policy._model, num_steps=args.num_denoise)  # noqa: SLF001

    cameras = list(POLICY_CAMERA_NAMES)
    denoise = sorted(set(int(s) for s in args.denoise_steps))
    aggs = list(args.agg)
    grids = None
    t0 = time.time()

    for i, step in enumerate(run):
        obs = observation_from_step(policy, step, prompt)
        per_seed = [sampler.attention(obs, denoise, noise_seed=s) for s in args.noise_seeds]
        blocks = {d: np.mean([b[d] for b in per_seed], axis=0) for d in denoise}
        if grids is None:
            n_layers, n_heads = blocks[denoise[0]].shape[:2]
            grids = np.zeros(
                (len(run), len(denoise), len(aggs), n_layers, n_heads, len(cameras),
                 ATTENTION_GRID, ATTENTION_GRID),
                np.float32,
            )
            print(f"layers={n_layers} heads={n_heads} action_tokens={blocks[denoise[0]].shape[2]}")
        for di, dstep in enumerate(denoise):
            block = blocks[dstep]  # [L, heads, action_tokens, 768]
            for ai, agg in enumerate(aggs):
                pooled = AGGREGATIONS[agg](block)  # [L, heads, 768]
                for ci, camera in enumerate(cameras):
                    lo, hi = CAMERA_BINDINGS[camera][2]
                    grids[i, di, ai, :, :, ci] = pooled[..., lo:hi].reshape(
                        pooled.shape[0], pooled.shape[1], ATTENTION_GRID, ATTENTION_GRID
                    )
        if i % 10 == 0 or i == len(run) - 1:
            print(f"  step {i + 1}/{len(run)}  ({time.time() - t0:.1f}s)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        attention=grids.astype(args.store_dtype),
        t_step=np.asarray([s.t_step for s in run], np.int64),
        denoise_steps=np.asarray(denoise, np.int64),
        aggregations=np.asarray(aggs),
        cameras=np.asarray(cameras),
        prompt=np.asarray(prompt),
        noise_seeds=np.asarray(args.noise_seeds, np.int64),
        records=np.asarray(str(pathlib.Path(args.records).resolve())),
        checkpoint=np.asarray(str(args.checkpoint)),
    )
    print(f"wrote {out}  shape={grids.shape}  dtype={args.store_dtype}  "
          f"({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
