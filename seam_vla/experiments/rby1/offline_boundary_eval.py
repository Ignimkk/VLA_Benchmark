"""Offline chunk-boundary evaluation on recorded RB-Y1 chunks (no model needed).

Loads PolicyRecorder dumps (`data/policy_records/step_*.npy`, each = one chunk query, K apart),
reproduces the baseline chunk-boundary artifact, and shows that the physical-space overlap-steering
refiner (`refinement/overlap_steer.OverlapSteerRefiner`) reduces boundary jerk and the cross-chunk
overlap residual. This is real evidence for SEAM on RB-Y1 that does not require the (server-only) model.

Usage:
    python -m benchmark.seam_vla.experiments.rby1.offline_boundary_eval \
        --records data/policy_records --K 8 --M 20 --lam 0.3
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

from benchmark.seam_vla.metrics.motion import aggregate_metrics, compute_motion_metrics
from benchmark.seam_vla.refinement.overlap_steer import OverlapSteerRefiner

ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP_DIMS = [6, 13]


def _load_records(records_dir):
    files = sorted(
        glob.glob(os.path.join(records_dir, "step_*.npy")),
        key=lambda p: int(re.search(r"step_(\d+)", p).group(1)),
    )
    chunks, states, prompts = [], [], []
    for p in files:
        d = np.load(p, allow_pickle=True).item()
        chunks.append(np.asarray(d["outputs/actions"], dtype=np.float64))
        states.append(np.asarray(d["inputs/state"], dtype=np.float64))
        prompts.append(str(d.get("inputs/prompt", "")))
    return chunks, states, prompts


def _split_runs(chunks, states, prompts):
    """Split contiguous records at prompt changes so resets are not scored as chunk boundaries."""
    if not chunks:
        return []
    runs = []
    start = 0
    for i in range(1, len(chunks)):
        if prompts[i] != prompts[i - 1]:
            runs.append((chunks[start:i], states[start:i], prompts[start]))
            start = i
    runs.append((chunks[start:], states[start:], prompts[start]))
    return runs


def _overlap_residual(chunks, K, arm_dims):
    res = []
    for a, b in zip(chunks[:-1], chunks[1:]):
        L = a.shape[0] - K
        tail = a[K : K + L]
        head = b[:L]
        res.append(np.linalg.norm((tail - head)[:, arm_dims], axis=1).mean())
    return np.asarray(res)


def _executed_stream(chunks, K, refiner=None):
    """Concatenate the first K actions of each chunk; optionally refine each chunk vs the previous."""
    executed = []
    prev = None
    for c in chunks:
        c_use = c
        if refiner is not None and prev is not None:
            c_use = np.asarray(refiner.refine(c, context={"previous_physical_chunk": prev}))
        executed.append(c_use[:K])
        prev = c  # reference = previous baseline decoded chunk (post-hoc refiner)
    return np.concatenate(executed, axis=0)


def evaluate(records_dir, K, M, lam, arm_dims=ARM_DIMS):
    chunks, states, prompts = _load_records(records_dir)
    n = len(chunks)
    if n < 3:
        raise SystemExit(f"need >=3 records, found {n} in {records_dir}")
    runs = _split_runs(chunks, states, prompts)

    # Score each contiguous prompt/run independently. Concatenating different prompts would turn the
    # reset between tasks into a fake chunk boundary and substantially inflate jerk/base-drift.
    refiner = OverlapSteerRefiner(execution_length=K, guided_window=M, lam=lam, guided_dims=arm_dims)
    base_per_run, seam_per_run = [], []
    base_res_parts, seam_res_parts, drift_parts = [], [], []
    for run_chunks, run_states, _ in runs:
        if len(run_chunks) < 2:
            continue
        base_stream = _executed_stream(run_chunks, K)[:, arm_dims]
        base_per_run.append(compute_motion_metrics(base_stream, K))
        base_res_parts.append(_overlap_residual(run_chunks, K, arm_dims))
        drift_parts.append(np.asarray([
            np.linalg.norm((s2 - s1)[arm_dims]) for s1, s2 in zip(run_states[:-1], run_states[1:])
        ]))

        seam_stream = _executed_stream(run_chunks, K, refiner)[:, arm_dims]
        seam_per_run.append(compute_motion_metrics(seam_stream, K))
        seam_chunks = []
        prev = None
        for c in run_chunks:
            seam_chunks.append(
                np.asarray(refiner.refine(c, context={"previous_physical_chunk": prev}))
                if prev is not None else c
            )
            prev = c
        seam_res_parts.append(_overlap_residual(seam_chunks, K, arm_dims))

    base = aggregate_metrics(base_per_run)
    seam = aggregate_metrics(seam_per_run)
    base_res = np.concatenate(base_res_parts)
    seam_res = np.concatenate(seam_res_parts)
    ds = np.concatenate(drift_parts)

    def pct(a, b):
        return "n/a" if not a else f"{(b - a) / abs(a) * 100:+.1f}%"

    print(f"records={n}  runs={len(runs)}  K={K}  M={M}  lam={lam}  arm_dims={arm_dims}")
    print("run lengths/prompts: " + "; ".join(f"{len(c)}×{p!r}" for c, _, p in runs))
    print(f"base drift |Δs| (arm L2 rad): mean={ds.mean():.4f} max={ds.max():.4f}")
    print("-" * 62)
    print(f"{'metric':<14}{'baseline':>12}{'SEAM(decoded)':>16}{'Δ%':>10}")
    for k in ("BJ", "IJ", "CD", "paper_avb"):
        print(f"{k:<14}{base[k]:>12.4f}{seam[k]:>16.4f}{pct(base[k], seam[k]):>10}")
    print(f"{'overlap_res':<14}{base_res.mean():>12.4f}{seam_res.mean():>16.4f}{pct(base_res.mean(), seam_res.mean()):>10}")
    print(f"{'BJ/IJ ratio':<14}{base['BJ']/max(base['IJ'],1e-9):>12.2f}{seam['BJ']/max(seam['IJ'],1e-9):>16.2f}")
    return {"baseline": base, "seam": seam,
            "base_overlap_residual": float(base_res.mean()),
            "seam_overlap_residual": float(seam_res.mean()),
            "drift_mean": float(ds.mean()), "drift_max": float(ds.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/policy_records")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.2)
    args = ap.parse_args()
    evaluate(args.records, args.K, args.M, args.lam)


if __name__ == "__main__":
    main()
