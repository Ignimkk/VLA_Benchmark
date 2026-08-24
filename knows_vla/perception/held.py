"""Detect the object the gripper is holding, and promote it from obstacle to robot part.

Eq. (4) splits the scene into one target and "everything else is an obstacle". It has no category
for the object already in the hand, so once the target moves on -- during the place phase -- the
held object is demoted to an obstacle the gripper is physically touching. The barrier is then
negative by construction: measured, h < 0 on 58.8% of steps even with a privileged, perfectly
correct target (docs/16-filter-variants.md §2). No choice of gamma_h or epsilon reaches that.

Dropping it from the obstacle set would be enough to remove the false violation, but it throws away
something worth keeping: while carrying an apple, the apple is the part of the system most likely
to hit the next fruit. Promoting it to a robot part instead protects it, which the paper's
formulation cannot express at all.

Detection uses no privileged state -- gripper command plus whether the object's centroid is
travelling with the end-effector -- so the same code runs in simulation and on hardware.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from benchmark.knows_vla.cbf.ellipsoid import Ellipsoid
from benchmark.knows_vla.cbf.filter import RobotBody


@dataclasses.dataclass(frozen=True)
class HeldParams:
    """[ASSUMPTION] throughout — the paper has no held-object stage to copy thresholds from."""

    window: int = 5  # frames of agreement required, matching the paper's K-frame smoothing
    follow_tol: float = 0.008  # m/step of centroid-vs-EEF disagreement still counted as "together"
    motion_floor: float = 0.002  # m/step of EEF travel below which the test cannot discriminate
    release_frames: int = 3  # frames of open gripper before letting go


class HeldObjectTracker:
    """Which obstacle, if any, is currently in the hand.

    Two conditions must hold together, and the second is what makes it trustworthy: a closed
    gripper alone would also fire on a failed grasp or on closing in mid-air, whereas an object
    whose centroid tracks the end-effector through actual motion is being carried. The
    `motion_floor` exists because while the arm is still, *every* object trivially "follows" it.
    """

    def __init__(self, params: HeldParams | None = None):
        self.p = params or HeldParams()
        self._prev_eef: np.ndarray | None = None
        self._prev_c: dict = {}
        self._agree: dict = {}
        self._held = None
        self._open_for = 0

    def reset(self) -> None:
        self.__init__(self.p)

    @property
    def held(self):
        return self._held

    def update(self, eef_pos: np.ndarray, gripper_closed: bool, obstacles: dict):
        """Returns the key of the held object, or None. Call once per control step."""
        eef = np.asarray(eef_pos, np.float64).reshape(3)

        if not gripper_closed:
            self._open_for += 1
            if self._open_for >= self.p.release_frames:
                self._held, self._agree = None, {}
        else:
            self._open_for = 0

        if self._prev_eef is not None and gripper_closed:
            d_eef = eef - self._prev_eef
            moving = float(np.linalg.norm(d_eef)) >= self.p.motion_floor
            for k, E in obstacles.items():
                prev = self._prev_c.get(k)
                if prev is None:
                    continue
                # Only accumulate evidence while there is motion to explain; standing still is not
                # evidence of anything.
                if moving and float(np.linalg.norm((E.c - prev) - d_eef)) <= self.p.follow_tol:
                    self._agree[k] = self._agree.get(k, 0) + 1
                elif moving:
                    self._agree[k] = 0
            ready = [k for k, n in self._agree.items() if n >= self.p.window]
            if self._held is None and ready:
                # Ties are possible when two objects touch; take the closest to the hand.
                self._held = min(ready, key=lambda k: float(np.linalg.norm(obstacles[k].c - eef)))

        self._prev_eef = eef
        self._prev_c = {k: E.c.copy() for k, E in obstacles.items()}
        if self._held is not None and self._held not in obstacles:
            self._held = None  # the track was lost; do not carry a stale attachment
        return self._held


def promote(body: RobotBody, obstacles: dict, held_key, eef_pos) -> tuple[RobotBody, dict]:
    """Move the held object out of the obstacle set and into the robot body.

    The offset is taken fresh each step rather than frozen at the grasp, so a slipping or
    re-seated object stays correctly placed; the shape ``Q`` is whatever perception last fitted.
    Returns (body with the extra part, obstacles without it).
    """
    if held_key is None or held_key not in obstacles:
        return body, obstacles
    E = obstacles[held_key]
    parts = dict(body.parts)
    parts[f"held:{held_key}"] = Ellipsoid(E.c - np.asarray(eef_pos, np.float64).reshape(3), E.Q)
    rest = {k: v for k, v in obstacles.items() if k != held_key}
    return RobotBody(body.origin, parts), rest
