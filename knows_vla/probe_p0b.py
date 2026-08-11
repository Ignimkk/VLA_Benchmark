"""P0b stage 2 — does a single attention head actually localize the policy's target?

This is the probe the whole KNOWS analysis hinges on. It replays the rollout recorded by
``collect_p0b.py`` through the frozen pi05_libero policy, extracts the attention grid for every
(layer, head), applies the paper's Eq. (2)-(3), and scores the argmax against the ground-truth
target read from the BDDL goal.

If the paper's (layer 12, head 3) does not beat the head-average here, there is no point building
the CBF-QP: see papers/KNOWS/06-repro-plan.md §8.1.

Run (openpi venv):
    JAX_PLATFORMS=cpu src/openpi/.venv/bin/python -m benchmark.knows_vla.probe_p0b
"""

from __future__ import annotations

import argparse
import pathlib

import einops
import jax
import jax.numpy as jnp
import numpy as np

PAPER_LAYER, PAPER_HEAD = 12, 3
AGENTVIEW_KEYS = (0, 256)
G = 16
BETA = -1.0  # Eq. (3), stated in the paper
ROBOT_PREFIXES = ("MountedPanda", "RethinkMount", "PandaGripper", "Panda")


# --------------------------------------------------------------------------------------- masks
def coverage_and_area(seg_full: np.ndarray, obj_ids: np.ndarray, resize: int = 224):
    """Per-object patch coverage c_i(r,c) and unoccluded area alpha_i, from Eq. (2)-(3).

    The rendered segmentation is 256x256 while the policy sees 224x224. resize_with_pad on a
    square input is a pure rescale (no padding), so nearest-neighbour is exact for ids.
    """
    h = seg_full.shape[0]
    idx = (np.arange(resize) * h / resize).astype(np.int64)
    seg = seg_full[np.ix_(idx, idx)]  # [224, 224] nearest-neighbour

    patch = resize // G  # 14
    cov = np.zeros((len(obj_ids), G, G), np.float32)
    area = np.zeros(len(obj_ids), np.float32)
    for k, oid in enumerate(obj_ids):
        m = (seg == oid).astype(np.float32)
        area[k] = m.sum()
        cov[k] = m.reshape(G, patch, G, patch).mean(axis=(1, 3))
    return cov, area


# ------------------------------------------------------------------------------------ forward
def _load_model():
    import dataclasses

    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    from openpi.shared import download

    cfg = _config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, return_attn_probs=True))
    ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    policy = _pc.create_trained_policy(cfg, ckpt)
    return policy


def _observation(policy, image, wrist, state, prompt):
    from openpi.models import model as _model

    inputs = policy._input_transform(  # noqa: SLF001
        {
            "observation/image": image,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": prompt,
        }
    )
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    return _model.Observation.from_dict(inputs)


class AttnSampler:
    """Jitted prefix/suffix passes that also hand back the agent-view attention block.

    Two small executables (one prefix, one Euler step) compiled once and reused, following the
    ``SeamSampler`` pattern in benchmark/seam_vla/integration/openpi_jax.py. Calling the nnx module
    eagerly instead compiles a fresh executable per call and exhausts XLA's CPU section memory
    after a few dozen steps.
    """

    def __init__(self, model, *, num_steps: int = 10):
        import flax.nnx as nnx

        self._nnx = nnx
        self._graphdef, self._state = nnx.split(model)
        self._num_steps = int(num_steps)
        self._action_horizon = int(model.action_horizon)
        self._action_dim = int(model.action_dim)
        self._jit_prefix = jax.jit(self._prefix)
        self._jit_suffix = jax.jit(self._suffix)

    def _prefix(self, state, obs):
        from openpi.models.pi0 import make_attn_mask

        model = self._nnx.merge(self._graphdef, state)
        tokens, mask, ar_mask = model.embed_prefix(obs)
        attn = make_attn_mask(mask, ar_mask)
        positions = jnp.cumsum(mask, axis=1) - 1
        kv = model.PaliGemma.llm([tokens, None], mask=attn, positions=positions)[1]
        return kv, mask

    def _suffix(self, state, obs, x_t, time, kv, prefix_mask):
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
        lo, hi = AGENTVIEW_KEYS
        # [L, B, K, G, T, S] -> [L, G, T, 256]
        return v_t, probs[:, 0, 0, :, :, lo:hi].astype(jnp.float32)

    def attention_all_steps(self, obs) -> list[np.ndarray]:
        """Agent-view attention for every Euler step: list of [L, G, T, 256] float32."""
        from openpi.models import model as _model

        obs = _model.preprocess_observation(None, obs, train=False)  # once -- it is not idempotent
        kv, prefix_mask = self._jit_prefix(self._state, obs)

        dt = -1.0 / self._num_steps
        x_t = jnp.zeros((obs.state.shape[0], self._action_horizon, self._action_dim), jnp.float32)
        time = jnp.asarray(1.0, jnp.float32)  # traced, not a Python float, or jit recompiles per value
        grids = []
        for _ in range(self._num_steps):
            v_t, grid = self._jit_suffix(self._state, obs, x_t, time, kv, prefix_mask)
            grids.append(np.asarray(grid))
            x_t = x_t + dt * v_t
            time = time + dt
        return grids


# --------------------------------------------------------------------------------------- main
AGGS = {"mean": lambda a: a.mean(axis=2), "sum": lambda a: a.sum(axis=2), "last": lambda a: a[:, :, -1]}


def run(episodes: list[pathlib.Path], *, window_k: int, denoise_steps: tuple[int, ...], agg_names: tuple[str, ...]):
    """Pool several episodes. The Eq. (3) window never straddles an episode boundary."""
    policy = _load_model()
    sampler = AttnSampler(policy._model)  # noqa: SLF001
    n_layers, n_heads = 18, 8
    per_ep = []
    for ep_path in episodes:
        per_ep.append(_one_episode(ep_path, policy, sampler, denoise_steps, agg_names, n_layers, n_heads))
    return _score(per_ep, window_k, denoise_steps, agg_names, n_layers, n_heads)


def _one_episode(episode, policy, sampler, denoise_steps, agg_names, n_layers, n_heads):
    d = np.load(episode, allow_pickle=True)
    id2name = {int(s.split(":", 1)[0]): s.split(":", 1)[1] for s in d["id2name"]}
    obj_ids = np.array([i for i in sorted(id2name) if i != 0 and not id2name[i].startswith(ROBOT_PREFIXES)])
    # Shuffle candidate order. np.argsort breaks ties by lowest index, and the GT target
    # (akita_black_bowl_1, id 1) sorts first -- every near-tie would silently score as a hit.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obj_ids))
    obj_ids = obj_ids[perm]
    names = [id2name[i] for i in obj_ids]
    target_name = str(d["manipulated"][0])
    target_k = names.index(target_name)

    print(f"episode      : {episode.name}")
    print(f"prompt       : {d['prompt']}")
    print(f"GT target    : {target_name}  (reach phase; destination = {d['destinations'][0]})")
    print(f"candidates   : {names}")
    n_steps = len(d["t"])
    print(f"steps        : {n_steps}, beta = {BETA}")

    # mass[d_idx, agg, t, layer, head, obj], alpha[t, obj]
    mass = np.zeros((len(denoise_steps), len(agg_names), n_steps, n_layers, n_heads, len(obj_ids)), np.float32)
    alpha = np.zeros((n_steps, len(obj_ids)), np.float32)
    # Controls, scored exactly like a head:
    #   uniform  -- a flat attention grid. Because Eq. (3) divides by area, this should tie every
    #               object and land at chance; if it does not, the metric is measuring object size.
    #   shuffled -- real attention values with the patch positions permuted, destroying spatial
    #               correspondence while keeping the value distribution.
    ctrl_mass = np.zeros((2, n_steps, len(obj_ids)), np.float32)
    ctrl_rng = np.random.default_rng(1)

    for t in range(n_steps):
        obs = _observation(policy, d["image"][t], d["wrist_image"][t], d["state"][t], str(d["prompt"]))
        grids = sampler.attention_all_steps(obs)
        cov, area = coverage_and_area(d["seg_full"][t], obj_ids)
        alpha[t] = area
        flat_cov = cov.reshape(len(obj_ids), -1)  # [O, 256]
        for di, ds in enumerate(denoise_steps):
            a = grids[ds]  # [L, G, T, 256]
            for ai, an in enumerate(agg_names):
                grid = AGGS[an](a)  # [L, G, 256]
                mass[di, ai, t] = np.einsum("lgp,op->lgo", grid, flat_cov)

        paper_grid = AGGS["mean"](grids[denoise_steps[0]])[PAPER_LAYER, PAPER_HEAD]  # [256]
        ctrl_mass[0, t] = flat_cov @ np.full_like(paper_grid, 1.0 / paper_grid.size)
        ctrl_mass[1, t] = flat_cov @ ctrl_rng.permutation(paper_grid)
        if (t + 1) % 20 == 0:
            print(f"  ... {t + 1}/{n_steps} steps")
    return {"mass": mass, "alpha": alpha, "ctrl_mass": ctrl_mass, "target_k": target_k,
            "names": names, "n_steps": n_steps}

def _score(per_ep, window_k, denoise_steps, agg_names, n_layers, n_heads):
    """Eq. (3): d_i = sum_K mass / sum_K alpha (beta = -1), windowed within each episode."""
    results = {}
    total = sum(e["n_steps"] for e in per_ep)
    for di, ds in enumerate(denoise_steps):
        for ai, an in enumerate(agg_names):
            hits = np.zeros((n_layers, n_heads), np.float32)
            gaps = []
            for e in per_ep:
                for t in range(e["n_steps"]):
                    s0 = max(0, t - window_k + 1)
                    num = e["mass"][di, ai, s0 : t + 1].sum(axis=0)
                    den = np.maximum(e["alpha"][s0 : t + 1].sum(axis=0), 1e-6)
                    dens = num / den
                    order = np.argsort(-dens, axis=-1)
                    hits += order[..., 0] == e["target_k"]
                    top = np.take_along_axis(dens, order[..., :2], axis=-1)
                    gaps.append(top[..., 0] - top[..., 1])
            results[(ds, an)] = (hits / total, np.stack(gaps, axis=-1))

    ctrl = {}
    for ci, cname in enumerate(("uniform", "shuffled")):
        hit = 0
        for e in per_ep:
            for t in range(e["n_steps"]):
                s0 = max(0, t - window_k + 1)
                dens = e["ctrl_mass"][ci, s0 : t + 1].sum(axis=0) / np.maximum(
                    e["alpha"][s0 : t + 1].sum(axis=0), 1e-6)
                hit += int(np.argmax(dens) == e["target_k"])
        ctrl[cname] = hit / total
    return results, (n_layers, n_heads), per_ep[0]["names"], per_ep[0]["target_k"], total, ctrl


def report(results, shape, names, target_k, n_steps, ctrl):
    n_layers, n_heads = shape
    print("\n" + "=" * 78)
    print("Target-hit rate by (denoise step, aggregation) — paper cell is layer 12 / head 3")
    print("=" * 78)
    chance = 1.0 / len(names)
    print(f"controls: uniform-attention={ctrl['uniform']:.3f}  "
          f"shuffled-attention={ctrl['shuffled']:.3f}  chance={chance:.3f}  "
          f"(target is 1 of {len(names)} candidates)")
    best_overall = None
    for (ds, an), (hits, gaps) in sorted(results.items()):
        paper = hits[PAPER_LAYER, PAPER_HEAD]
        bl, bh = np.unravel_index(np.argmax(hits), hits.shape)
        print(f"\ndenoise={ds:<2} agg={an:<5}  paper(12,3)={paper:.3f}   "
              f"best=({bl},{bh})={hits[bl, bh]:.3f}   mean={hits.mean():.3f}   chance={chance:.3f}")
        if best_overall is None or hits[bl, bh] > best_overall[0]:
            best_overall = (hits[bl, bh], ds, an, int(bl), int(bh))
    print("\n" + "=" * 78)
    print(f"best cell overall: hit={best_overall[0]:.3f} at denoise={best_overall[1]} "
          f"agg={best_overall[2]} layer={best_overall[3]} head={best_overall[4]}")

    # Full layer x head map for the paper's configuration, to compare against Figure 2.
    key = (0, "mean") if (0, "mean") in results else sorted(results)[0]
    hits, gaps = results[key]
    print("\n" + "=" * 78)
    print(f"layer x head hit rate  (denoise={key[0]}, agg={key[1]})   [paper cell marked *]")
    print("=" * 78)
    print("layer " + "".join(f"  h{h}  " for h in range(n_heads)) + "   max")
    for l in range(n_layers):
        row = "".join(
            f"{hits[l, h]:5.2f}{'*' if (l == PAPER_LAYER and h == PAPER_HEAD) else ' '}" for h in range(n_heads)
        )
        print(f"  {l:2d}  {row}  {hits[l].max():5.2f}")
    g = gaps[PAPER_LAYER, PAPER_HEAD]
    print(f"\npaper cell (12,3): hit={hits[PAPER_LAYER, PAPER_HEAD]:.3f}  "
          f"gap median={np.median(g):.3e}  gap p10={np.percentile(g, 10):.3e}  gap max={g.max():.3e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", nargs="+", default=["data/knows_p0b/libero_spatial_task0_ep0.npz"])
    p.add_argument("--window-k", type=int, default=5, help="Eq. (3) sliding window; paper value unknown")
    p.add_argument("--denoise-steps", type=int, nargs="+", default=[0, 5, 9])
    p.add_argument("--aggs", nargs="+", default=["mean", "sum", "last"])
    a = p.parse_args()
    results, shape, names, target_k, n_steps, ctrl = run(
        [pathlib.Path(x) for x in a.episodes], window_k=a.window_k, denoise_steps=tuple(a.denoise_steps), agg_names=tuple(a.aggs)
    )
    report(results, shape, names, target_k, n_steps, ctrl)


if __name__ == "__main__":
    main()
