"""Compare baseline vs SEAM result JSONs and print a paper-style table (Succ, BJ, IJ, CD, AVb)."""

from __future__ import annotations

import argparse
import json


def _fmt(x):
    return "nan" if x is None or (isinstance(x, float) and x != x) else f"{x:.4f}"


def compare(baseline: dict, seam: dict) -> str:
    bm, sm = baseline["metrics"], seam["metrics"]
    rows = [
        ("Success", baseline["overall_success"], seam["overall_success"], "up"),
        ("BJ", bm.get("BJ"), sm.get("BJ"), "down"),
        ("IJ", bm.get("IJ"), sm.get("IJ"), "down"),
        ("CD", bm.get("CD"), sm.get("CD"), "down"),
        ("AVb (paper_avb)", bm.get("paper_avb"), sm.get("paper_avb"), "down"),
    ]
    lines = [f"{'Metric':<18}{'Baseline':>12}{'SEAM':>12}{'Δ%':>10}  dir"]
    lines.append("-" * 66)
    for name, b, s, direction in rows:
        if b and s is not None and b == b and s == s and b != 0:
            delta = (s - b) / abs(b) * 100.0
            d = f"{delta:+.1f}%"
        else:
            d = "n/a"
        lines.append(f"{name:<18}{_fmt(b):>12}{_fmt(s):>12}{d:>10}  ({direction})")
    lines.append("")
    lines.append(f"SEAM correction_norm mean={seam.get('correction_norm_mean')}, "
                 f"max={seam.get('correction_norm_max')}, vls_chunks={seam.get('num_vls_chunks')}")
    lines.append(f"wall_time_s: baseline={baseline.get('wall_time_s'):.1f} seam={seam.get('wall_time_s'):.1f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--seam", required=True)
    args = ap.parse_args()
    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.seam) as f:
        seam = json.load(f)
    print(compare(baseline, seam))


if __name__ == "__main__":
    main()
