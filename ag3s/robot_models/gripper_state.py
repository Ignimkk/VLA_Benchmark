"""Gripper opening (the client's normalized number) → URDF finger joint values (T29).

## 왜 이 모듈이 있나

`UrdfSphereChain` 은 `q` 에 없는 관절을 고정값으로 둔다. RB-Y1 의 손가락은 `q` 에 없고 서버는
head 관절만 넘겼으므로 **손가락은 늘 0 = 닫힘**이었다 — 열린 손으로 사과에 다가가면 모델의
손가락은 사과 한가운데 있었다 (T28 P1-3a: 모델 ↔ 실제 51 mm, live 좌우 간격 12.5 mm vs 실제
101–111 mm). T29 는 손가락 관절을 **parameter** 로 만들고 (`UrdfSphereChain.param_joint_names`),
이 모듈이 그 값을 정한다.

## 규약은 클라이언트의 것을 읽는다 — 숫자를 박지 않는다

로컬 클라이언트 (`rby1_bringup/pi05_infer.py`) 가 정의한다:

| 쪽 | 식 | 어디 |
|---|---|---|
| 관측 16D `state[7]`·`state[15]` (측정 개도) | `norm = abs(qpos[GRIPPER_L_JOINT]) / abs(RBY1_GRIPPER_OPEN)` | `build_obs` `rby1_16d` · `rby1_state()` |
| 행동 16D `action[7]`·`action[15]` (명령) | `ctrl[GRIPPER_L_ACT] = clip(action, 0, 1) * RBY1_GRIPPER_OPEN` | `apply_action` `rby1_16d` |

`GRIPPER_L_JOINT` 의 범위가 `[RBY1 open 쪽, 0]` 이므로 (`gripper_finger_l1`: `[-0.05, 0]`) 부호가
열림 쪽이다. 따라서 역함수는 둘 다 `q_driven = norm * RBY1_GRIPPER_OPEN` 이고 명령만 `clip(0, 1)` 을
거친다. `RBY1_GRIPPER_OPEN` · `GRIPPER_L_JOINT` · `GRIPPER_R_JOINT` 는 **그 파일에서 AST 로 읽는다**
(import 하지 않는다 — 클라이언트는 MuJoCo 렌더러와 rby1_manipulation 을 끌고 온다).
식 자체가 바뀌면 `tests/ag3s/test_finger_joints.py` 가 클라이언트 소스를 읽어 실패한다.

## 두 손가락

MJCF 는 `left_finger`/`right_finger` equality (`polycoef="0 -1"`) 로 둘째 손가락을 첫째의 거울로
묶는다. URDF 에는 mimic 이 없으므로 **같은 이동량**을 각 관절의 열림 방향으로 준다:
`t = (q_driven − closed_d) / (open_d − closed_d)`, `q_partner = closed_p + t (open_p − closed_p)`.
open/closed 는 `UrdfSphereChain.gripper_joint_pairs` 가 한계에서 구조적으로 정한다 (원점 간격 최대 =
열림). RB-Y1 에서 이것은 `q_l2 = −q_l1` 로 MJCF equality 와 같다.

**MJCF 왼손 이름 뒤바뀜** (`mujoco_source.MJCF_BODY_ALIASES`: URDF `ee_finger_l1` ≡ MJCF
`ee_finger_l2`) 은 여기서 문제가 되지 않는다 — 두 손가락이 대칭으로 같은 양만큼 움직이므로
"어느 이름의 관절이 측정됐나" 는 이동량 `|q|` 하나로 끝난다.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import os
import pathlib
from typing import Any, Mapping, Optional, Sequence

import numpy as np

#: 클라이언트 스크립트 위치 오버라이드.
ENV_CLIENT_SCRIPT = "AG3S_CLIENT_SCRIPT"

#: 작업 루트 기준 후보. 서버 컨테이너(`pi05_TO_hybrid/`)와 로컬 PC 작업공간(`src/`) 둘 다.
CLIENT_SCRIPT_CANDIDATES = (
    "pi05_TO_hybrid/rby1_bringup/pi05_infer.py",
    "src/rby1_bringup/pi05_infer.py",
)

#: 클라이언트 모듈 상수 → 이 모듈에서의 뜻.
_CLIENT_NAMES = {
    "open": "RBY1_GRIPPER_OPEN",
    "left": "GRIPPER_L_JOINT",
    "right": "GRIPPER_R_JOINT",
}

#: 명령 열의 클립 — `apply_action` 의 `np.clip(action[7], 0.0, 1.0)`.
COMMAND_CLIP = (0.0, 1.0)


@dataclasses.dataclass(frozen=True)
class ClientGripperConvention:
    """클라이언트가 gripper 를 정규화하는 규약 — 그 파일에서 읽은 값."""

    open_command_m: float
    driven_joints: Mapping[str, str]  # hand → MuJoCo/URDF joint name the client reads and drives
    source: str  # "path:line, ..." — 어디서 읽었나

    def driven_value(self, norm: float, *, command: bool = False) -> float:
        """정규화 개도 → 구동 관절 값 (m). 명령이면 클라이언트처럼 `[0, 1]` 로 먼저 자른다."""
        x = float(norm)
        if command:
            x = float(np.clip(x, *COMMAND_CLIP))
        return x * float(self.open_command_m)

    def normalize(self, joint_value: float) -> float:
        """구동 관절 값 → 정규화 개도 (클라이언트 `rby1_state()` 와 같은 식)."""
        return abs(float(joint_value)) / abs(float(self.open_command_m))

    def record(self) -> dict[str, Any]:
        return {"open_command_m": float(self.open_command_m),
                "driven_joints": dict(self.driven_joints), "source": self.source}


def find_client_script(path: Optional[str | os.PathLike] = None) -> pathlib.Path:
    """클라이언트 스크립트 경로. 못 찾으면 **어디를 뒤졌는지** 적어 `FileNotFoundError`."""
    tried: list[str] = []
    candidates: list[pathlib.Path] = []
    if path is not None:
        candidates.append(pathlib.Path(path))
    env = os.environ.get(ENV_CLIENT_SCRIPT)
    if env:
        candidates.append(pathlib.Path(env))
    roots = [pathlib.Path(__file__).resolve().parents[3], pathlib.Path.cwd()]
    for root in roots:
        for rel in CLIENT_SCRIPT_CANDIDATES:
            candidates.append(root / rel)
    for cand in candidates:
        tried.append(str(cand))
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        "the client script that defines the gripper normalization (RBY1_GRIPPER_OPEN, "
        "GRIPPER_L_JOINT, GRIPPER_R_JOINT) was not found. Tried:\n  " + "\n  ".join(tried)
        + f"\nSet {ENV_CLIENT_SCRIPT} to its path. The finger joints cannot be placed without it "
        "(T29: the value must come from the client's own definition, not a number typed here).")


@functools.lru_cache(maxsize=8)
def _load(path: str) -> ClientGripperConvention:
    tree = ast.parse(pathlib.Path(path).read_text(), filename=path)
    found: dict[str, tuple[Any, int]] = {}
    for node in tree.body:  # module level only — the constants the client functions read
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _CLIENT_NAMES.values():
                    found[target.id] = (ast.literal_eval(node.value), node.lineno)
    missing = [n for n in _CLIENT_NAMES.values() if n not in found]
    if missing:
        raise ValueError(f"{path}: no module-level assignment of {missing} — the client's "
                         "gripper convention changed shape; T29's inverse must be re-derived")
    open_m, _ = found[_CLIENT_NAMES["open"]]
    if not isinstance(open_m, (int, float)) or float(open_m) == 0.0:
        raise ValueError(f"{path}: RBY1_GRIPPER_OPEN = {open_m!r} is not a non-zero number")
    src = ", ".join(f"{name}@{path}:{found[name][1]}" for name in _CLIENT_NAMES.values())
    return ClientGripperConvention(
        open_command_m=float(open_m),
        driven_joints={"left": str(found[_CLIENT_NAMES["left"]][0]),
                       "right": str(found[_CLIENT_NAMES["right"]][0])},
        source=src)


def load_client_convention(path: Optional[str | os.PathLike] = None) -> ClientGripperConvention:
    """클라이언트 규약을 읽는다 (파일마다 한 번)."""
    return _load(str(find_client_script(path)))


def opening_envelope(measured: float, commanded: Sequence[float], horizon: int) -> np.ndarray:
    """TO 계획 창의 정규화 개도 `(horizon,)` — **스텝 0 = 측정, 스텝 k ≥ 1 = 측정과 명령의 윗 포락선.**

    `open[0] = measured`, `open[k] = max(measured, max(clip(cmd[0..k], 0, 1)))`.

    실제 손가락은 명령을 ~2 스텝 늦게 따르고 (T17: 명령 0.0 → 개도 1.013 → 0.904 → 0.712),
    **물체에 막히면 명령까지 닫히지 않는다** (T17 유지 구간: 명령 ≈ 0.0–0.57, 개도 0.712–0.721).
    위치 제어 손가락의 개도는 언제나 "지금 개도" 와 "지금까지의 명령" 사이에 있다. 그 구간에서
    이 창이 쓰는 값은 **더 열린 쪽** 이다:

    * 닫는 중 · 쥔 채: 측정값 — 손가락은 실제로 거기 있거나(막힘) 거기서 안쪽으로 오는 중이다.
      명령(0)을 그대로 쓰면 모델 손가락이 **사과 안**으로 들어간다 — T29 가 고치는 바로 그 결함.
    * 여는 중 (놓기): 명령 — 실제보다 1–2 스텝 먼저 열린다. 바깥 장애물(crate 벽)에 대해 보수적이다.
    * 여닫이 없음: 둘이 같다.

    길이가 모자라는 명령은 마지막 값으로 채운다 (청크가 창보다 짧은 경우).
    """
    h = int(horizon)
    if h < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    cmd = np.clip(np.asarray(commanded, np.float64).reshape(-1), *COMMAND_CLIP)
    if cmd.size == 0:
        cmd = np.full(h, float(measured))
    if cmd.size < h:
        cmd = np.concatenate([cmd, np.full(h - cmd.size, cmd[-1])])
    out = np.maximum(float(measured), np.maximum.accumulate(cmd[:h]))
    out[0] = float(measured)
    return out


class FingerJointMap:
    """손 하나의 정규화 개도 → 한 `UrdfSphereChain` 의 finger parameter 값.

    `model` 은 `param_joint_names` 와 `gripper_joint_pairs()` 를 가진 모델이다. 클라이언트가 구동하는
    관절 이름(`GRIPPER_L_JOINT` 등)이 모델 URDF 의 어느 손가락 쌍에 있는지 찾고, 부호가 맞는지
    (클라이언트의 열림 부호 = URDF 의 열림 한계 쪽) 확인한다. 안 맞으면 **생성에서 죽는다.**
    """

    def __init__(self, model: Any, convention: Optional[ClientGripperConvention] = None):
        self.convention = convention or load_client_convention()
        self.param_joint_names = tuple(getattr(model, "param_joint_names", ()) or ())
        pairs = list(model.gripper_joint_pairs())
        self.hands: dict[str, dict[str, Any]] = {}
        for hand, driven in self.convention.driven_joints.items():
            pair = next((p for p in pairs if driven in p["joints"]), None)
            if pair is None:
                raise ValueError(
                    f"client drives {driven!r} for the {hand} hand but the model has no "
                    f"gripper pair with that joint (pairs: {[p['joints'] for p in pairs]})")
            partner = next(j for j in pair["joints"] if j != driven)
            open_d, closed_d = pair["open"][driven], pair["closed"][driven]
            if np.sign(self.convention.open_command_m) != np.sign(open_d - closed_d):
                raise ValueError(
                    f"{hand}: client open value {self.convention.open_command_m:+.4f} m points "
                    f"the other way from the URDF open limit of {driven} "
                    f"({closed_d:+.3f} → {open_d:+.3f}); the inverse would close an open hand")
            self.hands[hand] = {"driven": driven, "partner": partner,
                                "parent_link": pair["parent_link"], "pair": pair}

    def joint_values(self, openings: Mapping[str, Optional[float]], *,
                     command: bool = False) -> dict[str, float]:
        """`{joint: value}` for the hands given (None/NaN = skip that hand). Clamped to URDF limits."""
        out: dict[str, float] = {}
        for hand, norm in openings.items():
            if norm is None or not np.isfinite(float(norm)) or hand not in self.hands:
                continue
            h = self.hands[hand]
            pair = h["pair"]
            d, p = h["driven"], h["partner"]
            q_d = self.convention.driven_value(float(norm), command=command)
            lo, hi = pair["limits"][d]
            q_d = float(np.clip(q_d, lo, hi))
            span_d = pair["open"][d] - pair["closed"][d]
            t = (q_d - pair["closed"][d]) / span_d
            q_p = pair["closed"][p] + t * (pair["open"][p] - pair["closed"][p])
            lo_p, hi_p = pair["limits"][p]
            out[d] = q_d
            out[p] = float(np.clip(q_p, lo_p, hi_p))
        return {k: v for k, v in out.items() if k in self.param_joint_names}

    def path(self, measured: Mapping[str, Optional[float]],
             commanded: Mapping[str, Sequence[float]], horizon: int,
             default: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """`(P[n_params, horizon], openings{hand: (horizon,)})` — `opening_envelope` per hand.

        A hand with no measured value keeps `default` (the model's current values) on every step.
        """
        h = int(horizon)
        base = (np.zeros(len(self.param_joint_names)) if default is None
                else np.asarray(default, np.float64).reshape(-1))
        P = np.tile(base.reshape(-1, 1), (1, h))
        index = {n: i for i, n in enumerate(self.param_joint_names)}
        openings: dict[str, np.ndarray] = {}
        for hand, m in measured.items():
            if m is None or not np.isfinite(float(m)) or hand not in self.hands:
                continue
            env = opening_envelope(float(m), commanded.get(hand, ()), h)
            openings[hand] = env
            for k in range(h):
                vals = self.joint_values({hand: float(env[k])})
                for name, v in vals.items():
                    P[index[name], k] = v
        return P, openings

    def record(self) -> dict[str, Any]:
        return {"convention": self.convention.record(),
                "hands": {h: {"driven": v["driven"], "partner": v["partner"],
                              "parent_link": v["parent_link"],
                              "open": dict(v["pair"]["open"]),
                              "closed": dict(v["pair"]["closed"])}
                          for h, v in self.hands.items()}}


def openings_from_mujoco(model, data, convention: Optional[ClientGripperConvention] = None
                         ) -> dict[str, float]:
    """MuJoCo `qpos` → `{hand: norm}` **the client's way** (`rby1_state()`), for offline replays."""
    import mujoco

    conv = convention or load_client_convention()
    out: dict[str, float] = {}
    for hand, joint in conv.driven_joints.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0:
            continue
        out[hand] = conv.normalize(float(data.qpos[model.jnt_qposadr[jid]]))
    return out


__all__ = [
    "CLIENT_SCRIPT_CANDIDATES",
    "COMMAND_CLIP",
    "ClientGripperConvention",
    "ENV_CLIENT_SCRIPT",
    "FingerJointMap",
    "find_client_script",
    "load_client_convention",
    "opening_envelope",
    "openings_from_mujoco",
]
