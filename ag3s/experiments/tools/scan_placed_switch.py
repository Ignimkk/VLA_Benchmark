"""SUBTASK-c §5 — after latch PLACED, does `TargetConfirm` adopt another object? (read-only scan)

Reads the server-side records (`server_constraints/run_*/chunk_*.npz`, `summary_json`) and, per
episode (a new episode starts where `t_step` does not increase), finds the first request whose
`grasp.state == "placed"`, then the first later decision with `admissibility.decision.mode ==
"switch"`. It classifies the new manipulated object by its xy distance to the registered
destination centroid (< 0.15 m: the placed object in the destination, else another object) and
counts the `exclusion` record of every chunk after that switch.

    python3 benchmark/ag3s/experiments/tools/scan_placed_switch.py \
        outputs/verify/T37/E3a_rec/server_8204/server_constraints \
        outputs/verify/T37/E3b_rec/server_8206/server_constraints \
        outputs/verify/T34/E3a_rec/server_8204/server_constraints \
        outputs/verify/T34/E3b_rec/server_8206/server_constraints
"""

from __future__ import annotations

import collections
import glob
import json
import sys

import numpy as np


def scan(roots):
    episodes, exclusion = [], collections.Counter()
    for root in roots:
        for run in sorted(glob.glob(f"{root}/run_*")):
            prev, ep, tp, first, switched = None, 0, None, None, False

            def flush():
                episodes.append(dict(run=run, ep=ep, t_placed=tp, first_switch=first))

            for f in sorted(glob.glob(f"{run}/chunk_*.npz")):
                with np.load(f, allow_pickle=False) as z:
                    s = json.loads(str(z["summary_json"]))
                t = int(s["t_step"])
                if prev is not None and t <= prev:
                    flush()
                    ep, tp, first, switched = ep + 1, None, None, False
                prev = t
                if (s.get("grasp") or {}).get("state") == "placed" and tp is None:
                    tp = t
                decision = (s.get("admissibility") or {}).get("decision") or {}
                if tp is not None and decision.get("mode") == "switch":
                    switched = True
                    if first is None:
                        m = s.get("manipulated") or {}
                        d = s.get("destination") or {}
                        c = np.asarray(m.get("centroid"), float)
                        dc = np.asarray(d.get("centroid") or [np.nan] * 3, float)
                        dxy = float(np.linalg.norm(c[:2] - dc[:2]))
                        first = dict(t=t, dt=t - tp, id=m.get("id"),
                                     centroid=np.round(c, 3).tolist(),
                                     extents_mm=(m.get("admissibility") or {}).get("extents_mm"),
                                     xy_to_destination_m=round(dxy, 3),
                                     kind="in_destination" if dxy < 0.15 else "other_object")
                if switched:
                    ex = s.get("exclusion") or {}
                    exclusion[(ex.get("source"), bool(ex.get("active")),
                               "+".join(ex.get("mechanism") or ()))] += 1
            flush()
    return episodes, exclusion


def main():
    episodes, exclusion = scan(sys.argv[1:])
    placed = [e for e in episodes if e["t_placed"] is not None]
    switched = [e for e in placed if e["first_switch"] is not None]
    kinds = collections.Counter(e["first_switch"]["kind"] for e in switched)
    print(json.dumps(dict(episodes=len(episodes), placed=len(placed), switched_after_placed=len(switched),
                          kinds=kinds, exclusion_after_switch={"|".join(map(str, k)): v
                                                               for k, v in exclusion.items()})))
    for e in placed:
        print(json.dumps(e))


if __name__ == "__main__":
    main()
