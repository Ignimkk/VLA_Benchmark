"""Joint bounds for the optimized subset, converted once into the optimizer's own units.

The URDF speaks in radians, rad/s and rad/s². The optimizer speaks in *steps*: its decision variable
is a sequence of absolute joint targets one control period apart, so a velocity limit is a bound on
consecutive differences and an acceleration limit is a bound on second differences. Doing that
conversion here, once, is what keeps a stray `dt` from appearing in one constraint and not another —
a mistake that produces a trajectory obeying limits nobody wrote down.

Everything is taken from a robot description rather than hand-typed. A hand-typed limit table for a
20-joint robot is a table with one wrong number in it, and the joint it is wrong on is the one nobody
checked.

## 어느 description 인가 (T31 G1)

| 한계 | 출처 | 왜 |
|---|---|---|
| **position** | `LimitsConfig.source` — 기본 `model_xml`: **제어 대상 MJCF** 의 `jnt_range` | 정책은 MuJoCo 로봇에서 학습됐다. URDF 범위(`arm_6` ±2.705)가 MJCF(±2.967)보다 좁아 E3 에서 TO 가 정책의 손목을 2.685 에서 잘랐다 |
| velocity | URDF `<limit velocity>` (`robot_model.velocity_limits()`) | MJCF 에는 **관절 속도 한계가 없다** — position actuator(`kp`, `dampratio`)와 `jnt_range` 뿐이고, `forcerange` 도 비어 있다 (`model_transport.xml` → `rby1.xml` · `actuators_arm.xml`) |
| acceleration | URDF `<limit acceleration>` (RB-Y1 URDF 의 확장 속성, 주석 "rad/s^2") | 같은 이유 — MJCF 에 없다 |

`model_xml` 에서는 관절마다 `jnt_range` 를 읽고, 그 관절을 움직이는 **position actuator 의
`ctrlrange`** 와 대조한다 (RB-Y1 MJCF 는 `inheritrange="1"` 이라 같아야 한다). 다르면 **둘의
교집합**을 쓰고 크게 적는다 — 명령은 `ctrlrange` 로 잘리고 자세는 `jnt_range` 에서 막히므로 로봇이
실제로 갈 수 있는 곳은 교집합이다. `limited="false"` 인 관절은 position actuator 의 `ctrlrange` 를
쓰고, 그것도 없으면 **예외** 다 (범위를 지어내지 않는다).

URDF 는 형상(구 모델 · FK)의 근거로 남는다. 서버는 시작할 때 관절마다 URDF 대 사용값 표를
찍는다 (`position_limit_table`).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Optional, Sequence

import numpy as np

from benchmark.trajopt.config import DEFAULT_LIMITS_MODEL_XML, LimitsConfig
from benchmark.trajopt.types import ChunkLayout, JointLimits

#: MJCF 의 `jnt_range` 와 actuator `ctrlrange` 가 "같다" 의 문턱 (rad). MJCF 는 소수 5–6 자리로
#: 적혀 있고 `inheritrange` 는 그 값을 그대로 복사하므로 이보다 크게 다르면 적힌 값이 다른 것이다.
_RANGE_AGREEMENT_TOL = 1e-6
#: 시작 표에서 URDF 와 사용값이 "다르다" 의 문턱 (rad). MJCF 는 소수 5 자리(`1.5708`)로 적혀 있어
#: 같은 값도 URDF(`1.570796327`)와 3.7e-6 다르다 — 그것을 "다름" 으로 외치면 진짜 차이가 묻힌다.
_DIFFERS_TOL = 1e-4


@dataclasses.dataclass(frozen=True)
class RelaxedJointLimits(JointLimits):
    """`JointLimits` + **스텝 사이 속도 행을 걸지 않는다**는 표시 (T27, `--no-limits`).

    `max_step` 은 **유한하게 남는다.** 그 값이 첫 스텝 anchor ``|Q[:, 0] - q_now| <= max_step`` 의
    근거이고 (`problem.build_problem` 의 box 블록), `sqp._limit_overshoot` 도 같은 값으로 초기 iterate
    의 적격성을 잰다 (T24). anchor 만 남기고 속도 행을 빼려면 두 곳이 같은 값을 봐야 하므로 값을
    `inf` 로 만드는 대신 이 표시를 둔다.

    `velocity_rows=False` 를 읽는 곳은 둘이다 — `build_problem` (속도 행을 만들지 않는다) 과
    `limit_report` (속도 초과를 한계 위반으로 세지 않는다: 걸지 않은 한계다). 둘 다 `getattr` 로
    읽으므로 plain `JointLimits` 는 예전 그대로다.
    """

    velocity_rows: bool = True


def velocity_rows(limits: JointLimits) -> bool:
    """이 한계가 **스텝 사이 속도 행**을 거는가. `max_step` 이 유한하고 표시가 끄지 않았을 때."""
    return bool(getattr(limits, "velocity_rows", True)) and bool(
        np.all(np.isfinite(np.asarray(limits.max_step, np.float64))))


# ======================================================================================
# position 범위의 출처 (T31 G1)
# ======================================================================================

def limits_model_xml(config: Optional[LimitsConfig] = None) -> str:
    """`source == "model_xml"` 일 때 읽을 MJCF 경로 (해석 전). 비었으면 시뮬레이션의 로봇."""
    cfg = config or LimitsConfig()
    return str(cfg.model_xml) if cfg.model_xml else DEFAULT_LIMITS_MODEL_XML


@functools.lru_cache(maxsize=8)
def mjcf_joint_ranges(path: str) -> dict[str, dict[str, Any]]:
    """MJCF 의 1-자유도 관절마다 ``{limited, range, actuators}``. 경로당 한 번만 컴파일한다.

    `actuators` 는 그 관절을 직접 움직이는 actuator 들이고 각 항목은
    ``{name, position, ctrllimited, ctrlrange}`` 이다. `position` 은 bias 가 affine 이고
    ``biasprm[1] != 0`` (``-kp·q`` 항) 인 것 — 그때만 `ctrlrange` 가 관절 각도의 단위다.
    velocity · motor actuator 의 `ctrlrange` 는 rad/s · N·m 이라 범위와 비교하지 않는다.

    **전체 씬을 컴파일한다** (메시 포함, ~2.7 s). 기본값·`inheritrange` 를 MuJoCo 가 푼 그대로
    읽어야 "시뮬레이터가 실제로 쓰는 범위" 이기 때문이다 — XML 을 직접 읽으면 `<default>` 상속을
    다시 구현하게 된다.
    """
    import mujoco

    from benchmark.ag3s.runtime.asset_path import resolve_asset

    resolved = resolve_asset(path, what="MJCF for joint position limits (limits.model_xml)")
    out = joint_ranges_from_mj_model(mujoco.MjModel.from_xml_path(str(resolved)))
    out["__path__"] = {"resolved": str(resolved)}
    return out


def joint_ranges_from_mj_model(model) -> dict[str, dict[str, Any]]:
    """`mjcf_joint_ranges` 의 본체 — 이미 컴파일된 `MjModel` 에서 (bringup 처럼 모델을 들고 있을 때)."""
    import mujoco

    one_dof = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    out: dict[str, dict[str, Any]] = {}
    for j in range(model.njnt):
        if int(model.jnt_type[j]) not in one_dof:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name:
            continue
        actuators = []
        for a in range(model.nu):
            if (int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT)
                    or int(model.actuator_trnid[a, 0]) != j):
                continue
            position = (int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
                        and float(model.actuator_biasprm[a, 1]) != 0.0)
            actuators.append({
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a),
                "position": bool(position),
                "ctrllimited": bool(model.actuator_ctrllimited[a]),
                "ctrlrange": (float(model.actuator_ctrlrange[a, 0]),
                              float(model.actuator_ctrlrange[a, 1])),
            })
        out[name] = {"limited": bool(model.jnt_limited[j]),
                     "range": (float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])),
                     "actuators": actuators}
    return out


def mismatch_with_mj_model(robot_model, layout: ChunkLayout, config: Optional[LimitsConfig],
                           mj_model) -> list[dict[str, Any]]:
    """설정된 출처의 범위가 **실제로 돌고 있는 `mj_model`** 과 다른 최적화 관절들 (bringup 검사).

    `limits.model_xml` 은 경로이고 컴파일된 `MjModel` 은 자기 경로를 모른다. 그래서 모델을 들고
    있는 쪽은 경로를 믿는 대신 **값을 대조한다** — 다른 파일을 읽었으면 여기서 드러난다.
    """
    table = joint_ranges_from_mj_model(mj_model)
    lower, upper, info = resolve_position_ranges(robot_model, config, index=layout.q_indices)
    out = []
    for i in (int(k) for k in np.asarray(layout.q_indices).reshape(-1)):
        name = info["rows"][i]["joint"]
        if name not in table:
            out.append({"joint": name, "used": [float(lower[i]), float(upper[i])],
                        "mj_model": None})
            continue
        lo, hi, _, _ = _mjcf_range(name, table[name])
        if abs(lo - lower[i]) > _DIFFERS_TOL or abs(hi - upper[i]) > _DIFFERS_TOL:
            out.append({"joint": name, "used": [float(lower[i]), float(upper[i])],
                        "mj_model": [lo, hi]})
    return out


def _mjcf_range(name: str, entry: dict[str, Any]) -> tuple[float, float, str, Optional[bool]]:
    """한 관절의 `(lower, upper, note, ctrlrange_agrees)` — 머리말의 규칙."""
    ctrl = [a for a in entry["actuators"] if a["position"] and a["ctrllimited"]]
    if entry["limited"]:
        lo, hi = entry["range"]
        if not ctrl:
            return lo, hi, "jnt_range (no position actuator ctrlrange to compare)", None
        agrees = all(abs(a["ctrlrange"][0] - lo) <= _RANGE_AGREEMENT_TOL
                     and abs(a["ctrlrange"][1] - hi) <= _RANGE_AGREEMENT_TOL for a in ctrl)
        if agrees:
            return lo, hi, "jnt_range == ctrlrange", True
        c_lo = max(a["ctrlrange"][0] for a in ctrl)
        c_hi = min(a["ctrlrange"][1] for a in ctrl)
        lo2, hi2 = max(lo, c_lo), min(hi, c_hi)
        if not lo2 < hi2:
            raise ValueError(
                f"MJCF joint {name!r}: jnt_range {entry['range']} and position actuator ctrlrange "
                f"{[a['ctrlrange'] for a in ctrl]} do not overlap — the robot cannot be commanded "
                "anywhere inside its own joint range")
        return (lo2, hi2, f"jnt_range {tuple(round(v, 6) for v in entry['range'])} != ctrlrange "
                f"{[tuple(round(v, 6) for v in a['ctrlrange']) for a in ctrl]}; using the "
                "intersection", False)
    if ctrl:
        return (max(a["ctrlrange"][0] for a in ctrl), min(a["ctrlrange"][1] for a in ctrl),
                "joint not limited; position actuator ctrlrange", None)
    raise ValueError(
        f"MJCF joint {name!r} has limited='false' and no position actuator with a ctrlrange — "
        "there is no position range to read. Set limits.source='config' with "
        "limits.position_ranges, or limits.source='urdf'; a range will not be invented")


def resolve_position_ranges(
    robot_model, config: Optional[LimitsConfig] = None, *, index: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """`(lower_all, upper_all, info)` — `robot_model.joint_names` 순, **margin 전** 의 범위.

    `index` 는 TO 가 움직이는 관절(`layout.q_indices`)이다. 그 관절은 선택된 출처에 **반드시**
    있어야 하고 (없으면 예외), 나머지(고정 관절)는 출처에 없으면 URDF 값으로 남는다 — 그 값은
    box 에 들어가지 않는다. `info = {"model_xml": 해석된 경로 | None, "rows": [...]}` 이고 `rows` 는
    관절마다 URDF 값·사용값·출처·차이다 (`position_limit_table`).
    """
    cfg = config or LimitsConfig()
    urdf_lower, urdf_upper = (np.asarray(v, np.float64) for v in robot_model.joint_limits())
    lower, upper = urdf_lower.copy(), urdf_upper.copy()
    n = lower.size
    optimized = set(range(n) if index is None else (int(i) for i in np.asarray(index).reshape(-1)))
    names = list(getattr(robot_model, "joint_names", None) or [])
    if cfg.source != "urdf" and len(names) != n:
        raise ValueError(
            f"limits.source={cfg.source!r} looks joints up by name, but the robot model exposes "
            f"{len(names)} joint name(s) for {n} joint(s). Use limits.source='urdf' for a model "
            "without joint_names")
    sources = ["urdf"] * n
    notes = [""] * n
    agrees: list[Optional[bool]] = [None] * n
    where = None
    if cfg.source == "model_xml":
        where = limits_model_xml(cfg)
        table = mjcf_joint_ranges(where)
        missing = [names[i] for i in sorted(optimized) if names[i] not in table]
        if missing:
            raise ValueError(
                f"limits.source='model_xml': joint(s) {missing} are not 1-DoF joints of {where}. "
                "The optimizer would bound them by nothing the simulator uses")
        for i, name in enumerate(names):
            if name not in table:
                notes[i] = "not in MJCF (fixed joint here) — URDF value kept"
                continue
            lo, hi, note, ok = _mjcf_range(name, table[name])
            lower[i], upper[i], sources[i], notes[i], agrees[i] = lo, hi, "model_xml", note, ok
        where = table["__path__"]["resolved"]
    elif cfg.source == "config":
        given = {str(k): tuple(float(v) for v in pair)
                 for k, pair in dict(cfg.position_ranges or {}).items()}
        unknown = sorted(set(given) - set(names))
        if unknown:
            raise ValueError(f"limits.position_ranges names joint(s) the robot does not have: "
                             f"{unknown}")
        missing = [names[i] for i in sorted(optimized) if names[i] not in given]
        if missing:
            raise ValueError(
                f"limits.source='config': limits.position_ranges must name every optimized joint; "
                f"missing {missing}")
        for i, name in enumerate(names):
            if name in given:
                lower[i], upper[i] = given[name]
                sources[i], notes[i] = "config", "limits.position_ranges"
    rows = []
    for i in range(n):
        rows.append({
            "joint": names[i] if names else str(i),
            "optimized": i in optimized,
            "urdf": [float(urdf_lower[i]), float(urdf_upper[i])],
            "used": [float(lower[i]), float(upper[i])],
            "source": sources[i],
            "delta": [float(lower[i] - urdf_lower[i]), float(upper[i] - urdf_upper[i])],
            "differs": bool(abs(lower[i] - urdf_lower[i]) > _DIFFERS_TOL
                            or abs(upper[i] - urdf_upper[i]) > _DIFFERS_TOL),
            "ctrlrange_agrees": agrees[i],
            "note": notes[i],
        })
    return lower, upper, {"model_xml": where, "rows": rows}


def position_limit_table(robot_model, layout: ChunkLayout,
                         config: Optional[LimitsConfig] = None) -> dict[str, Any]:
    """관절마다 URDF 대 **사용값** (margin 전·후). 서버 시작 로그와 기록이 쓰는 표.

    ``{"source", "model_xml", "position_margin", "rows": [...], "n_differs", "n_ctrl_mismatch"}`` —
    `rows[k]` 는 `resolve_position_ranges` 의 행에 `bound` (margin 을 뺀, TO 가 실제로 거는 값;
    움직이지 않는 관절은 `None`) 를 더한 것이다.
    """
    cfg = config or LimitsConfig()
    index = layout.q_indices
    _, _, info = resolve_position_ranges(robot_model, cfg, index=index)
    rows, where = info["rows"], info["model_xml"]
    margin = float(cfg.position_margin)
    for row in rows:
        lo, hi = row["used"]
        row["bound"] = ([lo + margin, hi - margin] if row["optimized"] and cfg.enforce_position
                        else None)
    return {
        "source": cfg.source,
        "model_xml": where,
        "position_margin": margin,
        "enforce_position": bool(cfg.enforce_position),
        "rows": rows,
        "n_differs": int(sum(1 for r in rows if r["optimized"] and r["differs"])),
        "n_ctrl_mismatch": int(sum(1 for r in rows if r["ctrlrange_agrees"] is False)),
    }


def format_position_limit_table(table: dict[str, Any]) -> str:
    """`position_limit_table` → 사람이 읽는 표. 다른 행에는 `<<` 가 붙는다."""
    lines = [
        f"joint position limits — source: {table['source']}"
        + (f" ({table['model_xml']})" if table.get("model_xml") else "")
        + f" · margin {table['position_margin']:.3f} rad"
        + ("" if table.get("enforce_position", True) else " · POSITION BOX OFF (--no-limits)"),
        f"  {'joint':14s} {'opt':3s} {'URDF lower':>10s} {'URDF upper':>10s} "
        f"{'used lower':>10s} {'used upper':>10s} {'Δlower':>8s} {'Δupper':>8s}  note",
    ]
    for r in table["rows"]:
        mark = "  << DIFFERS" if r["differs"] and r["optimized"] else (
            "  (differs, not optimized)" if r["differs"] else "")
        ctrl = "  !! ctrlrange mismatch" if r["ctrlrange_agrees"] is False else ""
        lines.append(
            f"  {r['joint']:14s} {'yes' if r['optimized'] else '-':3s} "
            f"{r['urdf'][0]:+10.4f} {r['urdf'][1]:+10.4f} {r['used'][0]:+10.4f} "
            f"{r['used'][1]:+10.4f} {r['delta'][0]:+8.4f} {r['delta'][1]:+8.4f}  "
            f"{r['source']}{mark}{ctrl}")
    return "\n".join(lines)


def build_limits(
    robot_model,
    layout: ChunkLayout,
    *,
    dt: float,
    config: Optional[LimitsConfig] = None,
) -> JointLimits:
    """Position/velocity/acceleration bounds for `layout`'s optimized joints.

    Args:
        robot_model: anything exposing `joint_limits()` and, for velocity/acceleration,
            `velocity_limits()` / `acceleration_limits()` — `UrdfSphereChain` does.
        layout: which joints are free, and in what order the decision vector holds them.
        dt: control period in seconds.

    `enforce_velocity=False` / `enforce_acceleration=False` do not remove the bound; they widen it to
    something the optimizer can never reach. Keeping the constraint row present means the ablation
    changes one number rather than the problem's structure, which is what makes the comparison mean
    anything.
    """
    cfg = config or LimitsConfig()
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    index = layout.q_indices
    # T31 G1 — position 범위는 `cfg.source` 에서 (기본: 제어 대상 MJCF). 속도·가속도는 아래에서
    # 여전히 `robot_model` (URDF) 이다.
    lower_all, upper_all, _ = resolve_position_ranges(robot_model, cfg, index=index)
    lower = np.asarray(lower_all, np.float64)[index] + cfg.position_margin
    upper = np.asarray(upper_all, np.float64)[index] - cfg.position_margin
    if np.any(lower > upper):
        bad = [layout.joint_names[index[i]] if layout.joint_names else int(index[i])
               for i in np.flatnonzero(lower > upper)]
        raise ValueError(
            f"limits.position_margin={cfg.position_margin} rad closes the range of joint(s) {bad}"
        )

    velocity = _limit_vector(robot_model, "velocity_limits", cfg.default_velocity)[index]
    acceleration = _limit_vector(
        robot_model, "acceleration_limits", cfg.default_acceleration
    )[index]

    # `v_max * dt` is how far a joint may move between two consecutive chunk entries, and
    # `a_max * dt^2` how much that displacement may itself change. Both are plain bounds on
    # differences of the decision variable, which is the whole reason for converting here.
    max_step = velocity * cfg.velocity_scale * dt
    max_step_change = acceleration * cfg.acceleration_scale * dt * dt
    # **anchor 만 남기는 판** (T27): 속도 행은 빼되 `max_step` 은 유한하게 둔다 — anchor 와
    # 초기 iterate 적격성 검사가 그 값을 본다 (`RelaxedJointLimits`).
    anchor_only = (not cfg.enforce_velocity) and cfg.keep_anchor
    if not cfg.enforce_velocity and not anchor_only:
        max_step = np.full_like(max_step, np.inf)
    if not cfg.enforce_acceleration:
        max_step_change = np.full_like(max_step_change, np.inf)
    if not cfg.enforce_position:
        # position box 를 뺀다 (T27). box 행에는 trust region 과 anchor 만 남는다.
        lower = np.full_like(lower, -np.inf)
        upper = np.full_like(upper, np.inf)

    if anchor_only:
        return RelaxedJointLimits(
            lower=lower,
            upper=upper,
            max_step=max_step,
            max_step_change=max_step_change,
            dt=float(dt),
            velocity_rows=False,
        )
    return JointLimits(
        lower=lower,
        upper=upper,
        max_step=max_step,
        max_step_change=max_step_change,
        dt=float(dt),
    )


def _limit_vector(robot_model, method: str, default: Optional[float]) -> np.ndarray:
    getter = getattr(robot_model, method, None)
    if getter is None:
        if default is None:
            raise ValueError(
                f"the robot model has no {method}() and no default was configured. Set "
                f"limits.default_{method.split('_')[0]} explicitly rather than letting a bound be "
                "invented for a real robot."
            )
        return np.full(int(robot_model.nq), float(default), np.float64)
    return np.asarray(getter(default), np.float64)


def clamp_to_limits(trajectory: np.ndarray, limits: JointLimits) -> np.ndarray:
    """Project a trajectory onto the position box. Diagnostics and fallbacks only.

    Deliberately *not* used to enforce velocity or acceleration: clamping those after the fact
    changes the trajectory's shape in ways the optimizer never saw, so the result would satisfy the
    limits while no longer satisfying the collision constraints that were solved around it.
    """
    return np.clip(np.asarray(trajectory, np.float64), limits.lower[:, None], limits.upper[:, None])


def project_to_limits(trajectory: np.ndarray, limits: JointLimits,
                      q_now_opt: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """**최소 투영** — position box 와 첫 스텝 anchor 의 교집합(원소별 box)으로 자른다 (T31 G2-ii).

    둘 다 원소별 구간이라 교집합도 원소별 구간이고, 원소별 `clip` 이 곧 유클리드 최소 투영이다:
    구간 안의 원소는 **한 비트도 바뀌지 않는다.** anchor(``|Q[:, 0] - q_now| <= max_step``)는
    QP 가 hard 로 지키는 조건이라 함께 넣는다 — `sqp._limit_overshoot` 가 초기 iterate 의 적격성을
    그것까지 포함해 재므로, 넣지 않은 투영은 적격하지 않을 수 있다. `max_step` 이 유한하지 않으면
    (속도 한계 없음, anchor 없음) position box 뿐이다 = `clamp_to_limits`.

    속도·가속도는 여기서 고치지 않는다 (`clamp_to_limits` 의 이유). 원소별 clip 은 1-Lipschitz 라
    스텝 사이 차이를 키우지 않지만 2 차 차분은 키울 수 있다 — 호출하는 쪽이 `limit_report` 로 다시
    재고 쓴다. box 와 anchor 가 겹치지 않으면(로봇이 자기 box 밖에 max_step 이상) 투영이 없으므로
    `None` 이다.
    """
    Q = np.asarray(trajectory, np.float64)
    lower = np.broadcast_to(np.asarray(limits.lower, np.float64)[:, None], Q.shape).copy()
    upper = np.broadcast_to(np.asarray(limits.upper, np.float64)[:, None], Q.shape).copy()
    step = np.asarray(limits.max_step, np.float64)
    if q_now_opt is not None and Q.shape[1] and np.all(np.isfinite(step)):
        q0 = np.asarray(q_now_opt, np.float64).reshape(-1)
        lower[:, 0] = np.maximum(lower[:, 0], q0 - step)
        upper[:, 0] = np.minimum(upper[:, 0], q0 + step)
    if np.any(lower > upper):
        return None
    return np.clip(Q, lower, upper)


def limit_report(trajectory: np.ndarray, limits: JointLimits) -> dict[str, float]:
    """How far a trajectory is outside each bound, in the bound's own units. Zero means compliant.

    Reported rather than asserted, because the interesting question during development is not
    *whether* a limit is violated but *by how much* — a micron of overshoot from the QP's tolerance
    and a joint driven past its stop are the same boolean and very different problems.
    """
    Q = np.asarray(trajectory, np.float64)
    if Q.shape[1] < 1:
        return {"position": 0.0, "velocity": 0.0, "acceleration": 0.0}
    below = float(np.max(limits.lower[:, None] - Q, initial=0.0))
    above = float(np.max(Q - limits.upper[:, None], initial=0.0))

    step = np.diff(Q, axis=1)
    # 속도 행을 걸지 않은 한계(`RelaxedJointLimits`, T27)는 속도 초과를 세지 않는다 — 걸지 않은
    # 한계를 위반으로 세면 `sqp` 가 초기 iterate 를 부적격으로 보고 reference 를 끌어당긴다.
    velocity = (
        float(np.max(np.abs(step) - limits.max_step[:, None], initial=0.0))
        if step.shape[1] and velocity_rows(limits) else 0.0
    )
    change = np.diff(Q, n=2, axis=1)
    acceleration = (
        float(np.max(np.abs(change) - limits.max_step_change[:, None], initial=0.0))
        if change.shape[1] and np.all(np.isfinite(limits.max_step_change)) else 0.0
    )
    return {
        "position": max(below, above, 0.0),
        "velocity": max(velocity, 0.0),
        "acceleration": max(acceleration, 0.0),
    }


__all__ = ["RelaxedJointLimits", "build_limits", "clamp_to_limits", "format_position_limit_table",
           "joint_ranges_from_mj_model", "limit_report", "mismatch_with_mj_model", "limits_model_xml", "mjcf_joint_ranges", "position_limit_table",
           "project_to_limits", "resolve_position_ranges", "velocity_rows"]
