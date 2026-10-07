"""T43 Q (1) — the position servo the plan is executed by, as a linear model the TO can hold.

## Why a servo model

The TO plans **commands**: `Q[:, k]` is the absolute joint target the client writes into `d.ctrl` for
control row `k` (`pi05_infer.apply_action`, 33 sim steps per row). The robot is not at `Q[:, k]` when
the row ends — a position servo lags. T41 b measured it: after a 3-chunk HOLD the plan's step 0 sat
10.8° from `q_now` (the anchor bound `v_max·dt`) and the joint moved 2.0–2.6° in that row. Every
collision row the TO had was evaluated at the commands, so the states the robot actually passed
through were never checked.

This module predicts those states. With it the TO can check **where the robot will be** (its own
response to the commands, from the current position *and velocity*), instead of where it was told to
be.

## Which model, and where its numbers come from (derived, not fitted)

Every RB-Y1 arm joint in the simulated robot (`rby1.xml`, `actuators_arm.xml`, `defaults_common.xml`)
is a MuJoCo `<position>` actuator on a hinge with

    I q̈ = kp (u − q) − kv q̇ − b q̇ − τ_bias(q, q̇)

`kp` = `actuator_gainprm[0]` (2000 `med` · 1500 default), `kv` = `−actuator_biasprm[2]` (MuJoCo's
`dampratio="1"` resolved at compile: 245–303 N·m·s/rad), `b` = `dof_damping` (30), and `I` the joint's
diagonal of the joint-space inertia **including `dof_armature` = 10 kg·m²**. The armature dominates:
measured at T40 G C1807 t=216 the left-arm diagonal is 10.00–11.01 with off-diagonal coupling ≤ 0.40,
so the joints are close to decoupled linear second-order systems (natural frequency ≈ 13–14 rad/s,
damping ratio ≈ 1.05 — about one control period of lag).

All of these are **read from the compiled model** (`ServoParams.from_mj_model`); none is fitted to the
recorded runs. The recorded runs are used only to **validate** the model (`tests/trajopt/
test_t43q_servo.py`, `outputs/impl/T43Q/`).

The discretisation mirrors MuJoCo's default `Euler` integrator step for step: joint damping implicit,
actuator `kv` explicit,

    a  = (kp (u − q) − (kv + b) v − τ) / (I + h b)
    v' = v + h a,   q' = q + h v'

composed `substeps` times per control row with the command held. `τ` — gravity and the coupling the
diagonal model leaves out — is a per-joint constant over the plan, estimated by `ServoObserver` from the
measured positions (it is what makes the arm sag 0.3–0.6° below a HOLD command, T41 c).

## What the TO gets

`ServoModel.affine_path` writes the predicted joint position at any (row, sub-step) as

    X[c] = offset[c] + Σ_i W[c, i] ⊙ Q[:, i]          (per joint, W[c, i] = 0 for i > row(c))

— exactly affine in the plan, because the model is linear. A collision row at `X[c]` therefore
linearizes into one QP row on `Q[:, 0..row(c)]` by the chain rule; nothing about the robot's dynamics
is approximated by the QP that the model itself does not already approximate.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Optional, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class ServoParams:
    """Per-joint servo constants for the optimized joints (TO order), plus the sim's time grid."""

    kp: np.ndarray        # (nq,) N·m/rad
    kv: np.ndarray        # (nq,) N·m·s/rad, actuator velocity feedback (explicit in Euler)
    damping: np.ndarray   # (nq,) N·m·s/rad, joint damping (implicit in Euler)
    inertia: np.ndarray   # (nq,) kg·m², armature + diagonal of the joint-space inertia
    timestep: float       # s, MuJoCo `opt.timestep`
    substeps: int         # sim steps per control row (`pi05_infer`: round(1 / (CTRL_HZ · timestep)))
    joint_names: tuple[str, ...] = ()
    source: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        shapes = {np.asarray(a).shape for a in (self.kp, self.kv, self.damping, self.inertia)}
        if len(shapes) != 1:
            raise ValueError(f"servo parameter vectors disagree in shape: {shapes}")
        if self.timestep <= 0.0 or int(self.substeps) < 1:
            raise ValueError(f"timestep {self.timestep} / substeps {self.substeps} must be positive")
        if np.any(np.asarray(self.inertia) <= 0.0) or np.any(np.asarray(self.kp) <= 0.0):
            raise ValueError("servo inertia and kp must be positive")

    @property
    def nq(self) -> int:
        return int(np.asarray(self.kp).shape[0])

    @property
    def row_seconds(self) -> float:
        return float(self.timestep) * int(self.substeps)

    def summary(self) -> dict[str, Any]:
        """Numbers for the start log and the records (plain lists)."""
        wn = np.sqrt(np.asarray(self.kp) / np.asarray(self.inertia))
        zeta = (np.asarray(self.kv) + np.asarray(self.damping)) / (2.0 * np.sqrt(
            np.asarray(self.kp) * np.asarray(self.inertia)))
        return {"joint_names": list(self.joint_names), "kp": _round(self.kp), "kv": _round(self.kv),
                "damping": _round(self.damping), "inertia": _round(self.inertia),
                "natural_frequency_rad_s": _round(wn), "damping_ratio": _round(zeta),
                "timestep_s": float(self.timestep), "substeps": int(self.substeps),
                "row_seconds": self.row_seconds, "source": dict(self.source)}

    def shape_coefficient(self) -> np.ndarray:
        """`κ = (2ζ² − 1) / ωn²` (s²) per joint — how far the servo's path bends off the commanded one
        per unit command acceleration (T43 TA, C).

        The joint obeys ``I q̈ + c q̇ + kp q = kp u`` (`c = kv + damping`), i.e.
        ``u = q + τ q̇ + (I/kp) q̈`` with ``τ = c / kp``. Inverting to second order,

            q(t) = u(t) − τ u̇(t) + (τ² − I/kp) ü(t) + O(d³u)
                 = u(t − τ) + (τ²/2 − I/kp) ü(t) + O(d³u)

        The first term is the command **delayed** by τ (0.16–0.18 s here, ≈ 2.5 rows — T43 TK's
        C − M ≈ 2 rows); a delay common to the joints moves along the commanded path, it does not
        leave it. The second is the **shape error**, ``κ ü`` with ``κ = τ²/2 − I/kp = (2ζ² − 1)/ωn²``
        (ωn² = kp/I, ζ = c / (2√(kp I))). It is what cuts the corners of a sharply accelerating
        command. Positive for ζ > 1/√2 (the RB-Y1 arm: ζ ≈ 1.10–1.15).
        """
        kp, inertia = np.asarray(self.kp, np.float64), np.asarray(self.inertia, np.float64)
        c = np.asarray(self.kv, np.float64) + np.asarray(self.damping, np.float64)
        return c * c / (2.0 * kp * kp) - inertia / kp

    def accel_bound(self, tolerance_rad: float, row_seconds: Optional[float] = None) -> np.ndarray:
        """`a_C` (rad/row²) per joint: the command second difference whose shape error is
        `tolerance_rad` (T43 TA, C) — ``a_C = ε Δt² / κ`` with `κ` from `shape_coefficient`,
        ``Δt`` one control row (default `row_seconds`). Raises when ζ ≤ 1/√2 (κ ≤ 0: the
        second-order term no longer bounds the deviation and this derivation does not apply)."""
        kappa = self.shape_coefficient()
        if np.any(kappa <= 0.0):
            raise ValueError(f"servo shape coefficient must be > 0 (damping ratio > 1/sqrt 2); got "
                             f"{_round(kappa, 6)}")
        dt = float(self.row_seconds if row_seconds is None else row_seconds)
        return float(tolerance_rad) * dt * dt / kappa

    @classmethod
    def from_mj_model(cls, model, joint_names: Sequence[str], *, control_hz: float,
                      qpos: Optional[np.ndarray] = None, source: Optional[dict] = None
                      ) -> "ServoParams":
        """Read the constants of `joint_names` from a compiled `MjModel`.

        `qpos` is the pose whose joint-space inertia diagonal is used (`None` = `qpos0`). Every named
        joint must be a hinge driven by exactly one `<position>`-type actuator (affine bias with
        ``biasprm[1] = −kp``); anything else is an error — a model that is not a PD servo is not
        described by this class, and pretending otherwise would hand the TO a wrong lag.
        """
        import mujoco

        data = mujoco.MjData(model)
        if qpos is not None:
            data.qpos[:] = np.asarray(qpos, np.float64)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_crb(model, data)
        kp, kv, damping, inertia = [], [], [], []
        for name in joint_names:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
            if j < 0:
                raise ValueError(f"servo model: joint {name!r} is not in the MuJoCo model")
            if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(f"servo model: joint {name!r} is not a hinge")
            acts = [a for a in range(model.nu)
                    if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
                    and int(model.actuator_trnid[a, 0]) == j]
            if len(acts) != 1:
                raise ValueError(f"servo model: joint {name!r} has {len(acts)} actuators, expected 1")
            a = acts[0]
            gain, bias = model.actuator_gainprm[a], model.actuator_biasprm[a]
            if (int(model.actuator_biastype[a]) != int(mujoco.mjtBias.mjBIAS_AFFINE)
                    or not np.isclose(float(bias[1]), -float(gain[0])) or float(gain[0]) <= 0.0):
                raise ValueError(f"servo model: actuator of {name!r} is not a position servo "
                                 f"(gain {gain[:3]}, bias {bias[:3]})")
            dof = int(model.jnt_dofadr[j])
            kp.append(float(gain[0]))
            kv.append(float(-bias[2]))
            damping.append(float(model.dof_damping[dof]))
            # diagonal of the joint-space inertia = armature + link inertia (MuJoCo adds the
            # armature in `mj_crb`). Read through `mj_mulM` — the storage of `qM` differs by version.
            unit = np.zeros(model.nv)
            unit[dof] = 1.0
            column = np.zeros(model.nv)
            mujoco.mj_mulM(model, data, column, unit)
            inertia.append(float(column[dof]))
        substeps = max(1, int(round(1.0 / (float(control_hz) * float(model.opt.timestep)))))
        if int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_EULER):
            raise ValueError("servo model mirrors MuJoCo's Euler integrator; the model uses "
                             f"integrator {int(model.opt.integrator)}")
        return cls(kp=np.asarray(kp), kv=np.asarray(kv), damping=np.asarray(damping),
                   inertia=np.asarray(inertia), timestep=float(model.opt.timestep),
                   substeps=int(substeps), joint_names=tuple(str(n) for n in joint_names),
                   source=dict(source or {}, derived_from="compiled MuJoCo model",
                               inertia_pose="qpos0" if qpos is None else "given qpos"))


def _round(values, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in np.asarray(values, np.float64).reshape(-1)]


@functools.lru_cache(maxsize=4)
def _compiled(path: str):
    import mujoco

    from benchmark.ag3s.runtime.asset_path import resolve_asset

    resolved = resolve_asset(path, what="MJCF for the servo model (limits.model_xml)")
    return mujoco.MjModel.from_xml_path(str(resolved)), str(resolved)


def servo_params_from_xml(path: str, joint_names: Sequence[str], *, control_hz: float,
                          pose: Optional[dict[str, float]] = None) -> ServoParams:
    """`ServoParams.from_mj_model` on the MJCF at `path` (compiled once per path).

    `pose` maps joint names to angles for the inertia diagonal; joints it does not name stay at
    `qpos0`.
    """
    import mujoco

    model, resolved = _compiled(str(path))
    qpos = None
    if pose:
        qpos = np.asarray(model.qpos0, np.float64).copy()
        for name, value in pose.items():
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
            if j >= 0:
                qpos[int(model.jnt_qposadr[j])] = float(value)
    return ServoParams.from_mj_model(model, joint_names, control_hz=control_hz, qpos=qpos,
                                     source={"model_xml": resolved,
                                             "pose": None if not pose else dict(pose)})


class ServoModel:
    """The discrete per-joint servo, composed over control rows and sub-steps."""

    def __init__(self, params: ServoParams):
        self.params = params
        p = params
        h = float(p.timestep)
        kp, kv, b, inertia = (np.asarray(v, np.float64) for v in (p.kp, p.kv, p.damping, p.inertia))
        ih = inertia + h * b
        n = p.nq
        # one sim step, per joint: x' = F x + g u + e τ, x = (q, v)
        F = np.zeros((n, 2, 2))
        F[:, 1, 0] = -h * kp / ih
        F[:, 1, 1] = 1.0 - h * (kv + b) / ih
        F[:, 0, 0] = 1.0 + h * F[:, 1, 0]
        F[:, 0, 1] = h * F[:, 1, 1]
        g = np.stack([h * h * kp / ih, h * kp / ih], axis=1)
        e = np.stack([-h * h / ih, -h / ih], axis=1)
        # powers: after s sim steps (s = 0..substeps) with u, τ held: x_s = Fs x + Gs u + Es τ
        S = int(p.substeps)
        self._F = np.zeros((S + 1, n, 2, 2))
        self._G = np.zeros((S + 1, n, 2))
        self._E = np.zeros((S + 1, n, 2))
        self._F[0] = np.eye(2)
        for s in range(1, S + 1):
            self._F[s] = np.einsum("nij,njk->nik", F, self._F[s - 1])
            self._G[s] = np.einsum("nij,nj->ni", F, self._G[s - 1]) + g
            self._E[s] = np.einsum("nij,nj->ni", F, self._E[s - 1]) + e
        self._cache: dict = {}

    @property
    def nq(self) -> int:
        return self.params.nq

    @property
    def substeps(self) -> int:
        return int(self.params.substeps)

    # ------------------------------------------------------------------------------------------
    def rollout(self, q0, v0, commands, tau=None, *, samples: Sequence[int] = ()) -> dict:
        """Simulate the model: positions/velocities at the end of every row, and at `samples`
        (sub-step indices `1..substeps`) inside every row. Used by the observer and the tests."""
        q = np.asarray(q0, np.float64).reshape(-1).copy()
        v = np.asarray(v0, np.float64).reshape(-1).copy()
        U = np.asarray(commands, np.float64)
        tau = np.zeros(self.nq) if tau is None else np.asarray(tau, np.float64).reshape(-1)
        S = self.substeps
        ends_q, ends_v, sub = [], [], []
        for k in range(U.shape[1]):
            x = np.stack([q, v], axis=1)
            row = []
            for s in samples:
                xs = (np.einsum("nij,nj->ni", self._F[s], x) + self._G[s] * U[:, k:k + 1]
                      + self._E[s] * tau[:, None])
                row.append(xs[:, 0])
            if samples:
                sub.append(np.stack(row, axis=1))
            xe = np.einsum("nij,nj->ni", self._F[S], x) + self._G[S] * U[:, k:k + 1] + self._E[S] * tau[:, None]
            q, v = xe[:, 0].copy(), xe[:, 1].copy()
            ends_q.append(q)
            ends_v.append(v)
        out = {"q": np.stack(ends_q, axis=1) if ends_q else np.zeros((self.nq, 0)),
               "v": np.stack(ends_v, axis=1) if ends_v else np.zeros((self.nq, 0))}
        if samples:
            out["q_samples"] = np.stack(sub, axis=1)  # (nq, H, n_samples)
        return out

    def affine_path(self, horizon: int, samples: Sequence[int]):
        """`W (nq, n_c, H)` and the maps for the offset, for configs `c = (row k, sample s)`.

        Configs are ordered row-major: for row `k`, one per entry of `samples` (sub-step indices in
        `1..substeps`; `substeps` is the end of the row). Returns `(W, Fq, Eq)` with
        `Fq (nq, n_c, 2)` mapping the initial state `(q0, v0)` and `Eq (nq, n_c)` the constant τ:

            X[:, c] = Fq[:, c] · x0 + Eq[:, c] τ + Σ_i W[:, c, i] Q[:, i]

        Cached per `(horizon, samples)` — it does not depend on the state or the plan.
        """
        key = (int(horizon), tuple(int(s) for s in samples))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        H, S, n = int(horizon), self.substeps, self.nq
        samples = [int(s) for s in samples]
        if any(s < 1 or s > S for s in samples):
            raise ValueError(f"samples must be sub-step indices in 1..{S}, got {samples}")
        n_c = H * len(samples)
        W = np.zeros((n, n_c, H))
        Fq = np.zeros((n, n_c, 2))
        Eq = np.zeros((n, n_c))
        # state at the start of row k as an affine map: x_k = A_k x0 + Σ_i B_k[i] u_i + C_k τ
        A = np.tile(np.eye(2), (n, 1, 1))
        B = np.zeros((n, H, 2))
        C = np.zeros((n, 2))
        c = 0
        for k in range(H):
            for s in samples:
                Fs, Gs, Es = self._F[s], self._G[s], self._E[s]
                Fq[:, c, :] = np.einsum("nj,njk->nk", Fs[:, 0, :], A)
                W[:, c, :k] = np.einsum("nj,nij->ni", Fs[:, 0, :], B[:, :k, :])
                W[:, c, k] = Gs[:, 0]
                Eq[:, c] = np.einsum("nj,nj->n", Fs[:, 0, :], C) + Es[:, 0]
                c += 1
            FS, GS, ES = self._F[S], self._G[S], self._E[S]
            A = np.einsum("nij,njk->nik", FS, A)
            B = np.einsum("nij,nkj->nki", FS, B)
            B[:, k, :] += GS
            C = np.einsum("nij,nj->ni", FS, C) + ES
        out = (W, Fq, Eq)
        self._cache[key] = out
        return out


@dataclasses.dataclass
class ServoState:
    """What the observer knows about the arm at a capture: position, velocity, constant torque."""

    q: np.ndarray
    v: np.ndarray
    tau: np.ndarray
    source: str
    residual_rad: Optional[np.ndarray] = None

    def record(self) -> dict[str, Any]:
        return {"source": self.source, "v_rad_s_max": float(np.abs(self.v).max(initial=0.0)),
                "v_rad_s": _round(self.v), "tau_nm": _round(self.tau, 3),
                "residual_rad_max": (None if self.residual_rad is None
                                     else float(np.abs(self.residual_rad).max(initial=0.0)))}


class ServoObserver:
    """Velocity and constant-torque estimate at each capture, from what the server already gets.

    The server sees the measured joints at every capture (`robot_state`) and, since T18, the
    commands the client actually applied between the last capture and this one
    (`exec_feedback.applied_arm`, one row per control step — HOLD rows included). Propagating the
    last estimate through those commands predicts this capture; the position residual pins the
    constant torque `τ` (one unknown per joint, one measurement per joint), and the velocity follows.
    The servo forgets its initial velocity within ~0.2 s (a chunk is 0.53 s), so an error in the
    previous velocity barely reaches this one.

    Without facts to propagate (first request, missing or mismatched feedback) the arm is taken to
    be **at rest** with the last `τ` (zero at the start) and the source says so.
    """

    def __init__(self, model: ServoModel, *, max_tau: Optional[np.ndarray] = None):
        self.model = model
        self._max_tau = None if max_tau is None else np.asarray(max_tau, np.float64)
        self.reset()

    def reset(self) -> None:
        self._last: Optional[ServoState] = None
        self._last_seq: Optional[int] = None

    def update(self, q_now, feedback: Optional[dict], seq: Optional[int]) -> ServoState:
        """Estimate at this capture and remember it for the next one."""
        q_now = np.asarray(q_now, np.float64).reshape(-1)
        state = self._estimate(q_now, feedback)
        self._last = state
        self._last_seq = None if seq is None else int(seq)
        return state

    def _estimate(self, q_now: np.ndarray, feedback: Optional[dict]) -> ServoState:
        last = self._last
        tau0 = np.zeros_like(q_now) if last is None else last.tau
        if last is None:
            return ServoState(q_now, np.zeros_like(q_now), tau0, "rest: first capture")
        fb = feedback or {}
        if not fb.get("available"):
            return ServoState(q_now, np.zeros_like(q_now), tau0,
                              f"rest: no exec_feedback ({fb.get('reason')})")
        if self._last_seq is not None and int(fb.get("seq", -1)) != self._last_seq:
            return ServoState(q_now, np.zeros_like(q_now), tau0,
                              f"rest: exec_feedback is for seq {fb.get('seq')}, last capture was "
                              f"seq {self._last_seq}")
        U = np.asarray(fb.get("applied_arm", np.zeros((0, 0))), np.float64)
        if U.ndim != 2 or U.shape[0] == 0 or U.shape[1] != q_now.shape[0]:
            return ServoState(q_now, np.zeros_like(q_now), tau0,
                              f"rest: applied_arm has shape {U.shape}, expected (n, {q_now.shape[0]})")
        commands = U.T
        free = self.model.rollout(last.q, last.v, commands, np.zeros_like(q_now))
        unit = self.model.rollout(np.zeros_like(q_now), np.zeros_like(q_now),
                                  np.zeros_like(commands), np.ones_like(q_now))
        dq, dv = unit["q"][:, -1], unit["v"][:, -1]
        residual = q_now - free["q"][:, -1]
        tau = residual / dq
        if self._max_tau is not None:
            tau = np.clip(tau, -self._max_tau, self._max_tau)
        v = free["v"][:, -1] + dv * tau
        return ServoState(q_now, v, tau, f"propagated {U.shape[0]} applied rows", residual)


__all__ = ["ServoModel", "ServoObserver", "ServoParams", "ServoState", "servo_params_from_xml"]
