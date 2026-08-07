"""Compare actual RB-Y1 baseline and SEAM trajectories recorded by pi05_infer.py.

Paper metrics use executed absolute arm-joint targets and exclude the two near-binary gripper
dimensions. The same discrete metrics are reported separately for measured MuJoCo joint positions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.seam_vla.metrics.motion import compute_motion_metrics

ARM_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
METRIC_KEYS = ("BJ", "IJ", "CD", "paper_avb")


def _scalar(data, key):
    return np.asarray(data[key]).item()


def _overlap_residual(chunks: np.ndarray, execution_length: int) -> float:
    if len(chunks) < 2:
        return float("nan")
    overlap = chunks.shape[1] - execution_length
    values = []
    for previous, current in zip(chunks[:-1], chunks[1:]):
        delta = previous[execution_length : execution_length + overlap, ARM_DIMS]
        delta = delta - current[:overlap, ARM_DIMS]
        values.append(np.linalg.norm(delta, axis=-1))
    return float(np.mean(np.concatenate(values)))


def evaluate_run(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        actions = np.asarray(data["executed_actions"], dtype=np.float64)
        qpos = np.asarray(data["measured_qpos"], dtype=np.float64)
        chunks = np.asarray(data["predicted_chunks"], dtype=np.float64)
        execution_length = int(_scalar(data, "execution_length"))
        return {
            "path": str(path),
            "condition": str(_scalar(data, "condition")),
            "prompt": str(_scalar(data, "prompt")),
            "num_steps": int(actions.shape[0]),
            "num_chunks": int(chunks.shape[0]),
            "execution_length": execution_length,
            "action_metrics": compute_motion_metrics(actions[:, ARM_DIMS], execution_length),
            "qpos_metrics": compute_motion_metrics(qpos[:, ARM_DIMS], execution_length),
            "overlap_residual": _overlap_residual(chunks, execution_length),
            "inference_ms_median": float(np.median(np.asarray(data["inference_ms"]))),
            "vls_chunks": int(np.count_nonzero(np.asarray(data["used_vls"]))),
        }


def _change_percent(baseline: float, seam: float) -> float:
    if not np.isfinite(baseline) or not np.isfinite(seam) or baseline == 0:
        return float("nan")
    return float((seam - baseline) / abs(baseline) * 100.0)


def compare_runs(baseline_path: str | Path, seam_path: str | Path) -> dict:
    baseline = evaluate_run(baseline_path)
    seam = evaluate_run(seam_path)
    if baseline["prompt"] != seam["prompt"]:
        raise ValueError(
            f"prompt mismatch: baseline={baseline['prompt']!r}, seam={seam['prompt']!r}"
        )
    if baseline["num_steps"] != seam["num_steps"]:
        raise ValueError(
            f"step-count mismatch: baseline={baseline['num_steps']}, seam={seam['num_steps']}"
        )

    changes = {}
    for group in ("action_metrics", "qpos_metrics"):
        changes[group] = {
            key: _change_percent(baseline[group][key], seam[group][key])
            for key in METRIC_KEYS
        }
    changes["overlap_residual"] = _change_percent(
        baseline["overlap_residual"], seam["overlap_residual"]
    )
    return {"baseline": baseline, "seam": seam, "change_percent": changes}


def _print_metric_table(title: str, baseline: dict, seam: dict, changes: dict) -> None:
    print(f"\n{title}")
    print(f"{'metric':<14}{'baseline':>14}{'SEAM':>14}{'change':>12}")
    for key in METRIC_KEYS:
        label = "AVb" if key == "paper_avb" else key
        print(
            f"{label:<14}{baseline[key]:>14.6f}{seam[key]:>14.6f}"
            f"{changes[key]:>+11.1f}%"
        )
    base_ratio = baseline["BJ"] / max(baseline["IJ"], 1e-12)
    seam_ratio = seam["BJ"] / max(seam["IJ"], 1e-12)
    print(f"{'BJ/IJ':<14}{base_ratio:>14.3f}{seam_ratio:>14.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="baseline trajectory .npz")
    ap.add_argument("--seam", required=True, help="SEAM trajectory .npz")
    ap.add_argument("--out", help="optional JSON output path")
    args = ap.parse_args()

    result = compare_runs(args.baseline, args.seam)
    baseline, seam = result["baseline"], result["seam"]
    print(f"prompt={baseline['prompt']!r}")
    print(
        f"steps={baseline['num_steps']}  chunks(base/seam)="
        f"{baseline['num_chunks']}/{seam['num_chunks']}  K={baseline['execution_length']}"
    )
    _print_metric_table(
        "Executed action targets (paper metrics; arm joints only)",
        baseline["action_metrics"],
        seam["action_metrics"],
        result["change_percent"]["action_metrics"],
    )
    _print_metric_table(
        "Measured MuJoCo qpos (physical response; arm joints only)",
        baseline["qpos_metrics"],
        seam["qpos_metrics"],
        result["change_percent"]["qpos_metrics"],
    )
    overlap_change = result["change_percent"]["overlap_residual"]
    print(
        "\noverlap residual  "
        f"baseline={baseline['overlap_residual']:.6f}  SEAM={seam['overlap_residual']:.6f}  "
        f"change={overlap_change:+.1f}%"
    )
    print(
        f"median inference ms  baseline={baseline['inference_ms_median']:.1f}  "
        f"SEAM={seam['inference_ms_median']:.1f}; SEAM VLS chunks={seam['vls_chunks']}"
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
