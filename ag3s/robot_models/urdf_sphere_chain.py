"""A `RobotCollisionModel` built from a URDF, with FK that is literally the same code in numpy and
CasADi.

AG3S does not own robot geometry — `types.RobotCollisionModel` is a Protocol precisely so that a
Pinocchio or MuJoCo model can be dropped in instead. This module exists because pinocchio is not
installed in the openpi venv and the URDF needs nothing but `xml.etree`.

(An earlier note here claimed mujoco 3.10 could not load the RB-Y1 meshes. That was wrong: the model
loads fine from an **absolute** path, and the mesh errors came from resolving a relative one. The
URDF route is still the right default — no simulator dependency, and the capsules come ready-made —
but not for that reason.)

Two things make this worth more than a stopgap:

* `src/rby1_description/models/rby1a/urdf/model.urdf` carries **capsule collision primitives** —
  radius, length and origin per link — which is exactly the shape a sphere-chain wants. They are
  wrapped in XML comments in that file, so `parse_urdf` reads commented-out `<collision>` blocks as
  well as live ones.
* The self-filter needs numeric FK and the constraint builder needs symbolic FK. Writing them twice
  invites a silent disagreement in which the filter deletes points the constraints then treat as
  free space. Here both go through `_fk_chain` with a swapped arithmetic backend, so
  `test_robot_models.py` can assert they agree to machine precision and mean it.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import pathlib
import xml.etree.ElementTree as ET
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

#: 반지름 비교 tolerance (m) — 부동소수 잔재를 "덮지 못한다" 로 읽지 않기 위한 것. 1 μm.
_COVERAGE_TOL = 1e-6


@dataclasses.dataclass(frozen=True)
class CoverageShortfall:
    """한 capsule 의 구 사슬이 그 capsule 을 **덜 덮는다**는 기록.

    `UrdfSphereChain` 의 기본값에서는 절대 생기지 않는다 (구는 항상 capsule 을 담도록
    팽창된다). `capsule_radius_scale < 1` 이나 `max_sphere_radius` 를 준 실행에서만 찬다.

    **이것이 URDF capsule 기준이라는 점이 중요하다.** URDF 의 capsule 자체가 mesh 보다 두꺼울
    수 있으므로 (RB-Y1 `link_*_arm_5`: URDF 75 mm, MJCF mesh 실측 65.4~68.4 mm), 여기 기록이
    남는 것이 곧 "실제 팔을 못 덮는다" 는 뜻은 아니다. 이 클래스는 mesh 를 모르므로 아는 것만
    말한다 — 어느 쪽인지는 읽는 사람이 판단한다.
    """

    link: str
    n_spheres: int
    capsule_radius: float
    capsule_length: float
    sphere_radius: float  # 실제로 쓴 반지름
    required_radius: float  # capsule 을 담으려면 필요했던 반지름

    @property
    def shortfall_m(self) -> float:
        return float(self.required_radius - self.sphere_radius)

    def summary(self) -> str:
        return (f"{self.link}: 구 {self.n_spheres} 개 × {self.sphere_radius * 1000:.1f} mm, "
                f"덮으려면 {self.required_radius * 1000:.1f} mm 필요 "
                f"({self.shortfall_m * 1000:.1f} mm 부족; URDF capsule "
                f"r={self.capsule_radius * 1000:.1f} L={self.capsule_length * 1000:.1f} mm)")

@dataclasses.dataclass(frozen=True)
class CapsuleTrim:
    """한 link 의 capsule 을 **link frame z 로 잘랐다**는 기록.

    굵기(`capsule_radius_scale_by_link`)와 **다른 축**이다. 굵기는 구를 얇게 하고, 이것은 구를
    **덜 놓는다** — 원위 쪽 구를 아예 만들지 않는 것이므로 그 구간은 어떤 행에도 안 나타난다.

    RB-Y1 에서 이것이 필요한 이유는 하나다: `link_*_arm_5` 의 URDF capsule 은 r 75 mm ·
    L 250 mm 이고 축이 link frame 에서 z ∈ [−0.225, +0.025] 인데, 같은 frame 에서 손목
    (`link_*_arm_6`) 원점이 0.0, `FT_sensor_*` 가 −0.1087, 손바닥(`ee_*`)이 −0.1548,
    2 지 그리퍼 뿌리가 −0.2278 이다. **이 capsule 하나가 손목·손바닥·그리퍼를 전부 삼킨다.**
    다른 팔 capsule 은 r 40 / 35 mm 로 얇으므로 이 링크만 튀는 것이다.

    `reach_*` 는 축이 아니라 **구 표면**이 닿는 z 다 (`중심 ± 유효반지름`). 자르는 것만으로는
    표면이 그만큼 물러나지 않는다 — 반지름이 그대로면 잘린 끝의 구가 여전히 아래로 부푼다.
    그래서 굵기 손잡이와 함께 쓰는 것이 정상이고, 두 손잡이는 서로 막지 않는다.
    """

    link: str
    keep_z_min: float
    keep_z_max: float
    axis_z_min: float  # 자르기 전 축이 덮던 z 범위
    axis_z_max: float
    n_spheres_before: int
    n_spheres_after: int
    reach_z_min_before: float  # 구 **표면**이 닿던 z (중심 − 유효반지름)
    reach_z_min_after: float
    #: 이 link 아래에 달린 link 들 중, 잘라낸 구간에 원점이 있는 것. `(이름, z, 자기 구가 있나)`.
    #: **`False` 인 것이 벌거벗은 쪽이다** — 이 capsule 이 덮어 주던 것이 사라졌는데 자기 구도
    #: 없으면 그 link 은 무엇에 부딪혀도 아무도 막지 않는다.
    released_links: tuple[tuple[str, float, bool], ...] = ()

    @property
    def released_m(self) -> float:
        """구 표면이 물러난 거리. 0 이면 잘랐는데 **표면은 그대로**라는 뜻이다."""
        return float(self.reach_z_min_after - self.reach_z_min_before)

    def summary(self) -> str:
        naked = [n for n, _, covered in self.released_links if not covered]
        covered = [n for n, _, c in self.released_links if c]
        parts = [
            f"{self.link}: 축 z [{self.axis_z_min * 1000:+.1f}, {self.axis_z_max * 1000:+.1f}] "
            f"→ 유지 [{self.keep_z_min * 1000:+.1f}, {self.keep_z_max * 1000:+.1f}] mm, "
            f"구 {self.n_spheres_before} → {self.n_spheres_after} 개, "
            f"표면 도달 z {self.reach_z_min_before * 1000:+.1f} → "
            f"{self.reach_z_min_after * 1000:+.1f} mm ({self.released_m * 1000:+.1f} mm 물러남)",
        ]
        if covered:
            parts.append(f"        풀려난 하위 link (자기 구 있음): {', '.join(covered)}")
        if naked:
            parts.append(f"        **풀려났고 자기 구가 없다**: {', '.join(naked)} "
                         f"— 이 구간은 어떤 구에도 안 덮인다")
        return "\n".join(parts)


def _parse_extent(link: str, value: Any) -> tuple[float, float]:
    """`capsule_extent_by_link` 한 항목 → `(z_min, z_max)` (link frame, m).

    받는 형태는 둘이다.

    * **숫자 하나** `z` → `(z, +inf)` = "z 이상만 남긴다". RB-Y1 팔의 **원위 방향이 −z** 이므로
      (`link_*_arm_5` 축이 z ∈ [−0.225, +0.025], 손바닥이 −0.1548) 숫자 하나는 곧 원위 절단면이다.
    * **쌍** `(z_min, z_max)` → 그 구간만 남긴다. 축 방향이 다른 link 에도 뜻이 분명하다.

    형식이 아니면 **여기서 죽는다.** 조용히 넘기면 아무것도 안 잘린 채 잘랐다고 믿게 된다.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        lo, hi = float(value), math.inf
    else:
        try:
            lo, hi = (float(v) for v in value)
        except Exception:
            raise ValueError(
                f"capsule_extent_by_link[{link!r}] 는 숫자 하나(원위 절단면 z) 이거나 "
                f"(z_min, z_max) 쌍이어야 합니다: {value!r}") from None
    if not (lo < hi):
        raise ValueError(
            f"capsule_extent_by_link[{link!r}] = ({lo}, {hi}): z_min < z_max 여야 합니다. "
            f"빈 구간을 주면 그 link 의 구가 전부 사라지는데, 그것은 link 을 제약에서 빼는 것이고 "
            f"--exclude-links 가 그 일을 합니다 (그쪽은 무엇이 빠졌는지 로그에 남깁니다)")
    if not math.isfinite(lo) and not math.isfinite(hi):
        raise ValueError(
            f"capsule_extent_by_link[{link!r}] 가 양쪽 다 무한합니다 — 아무것도 자르지 않는 "
            f"설정이므로 키를 빼십시오. 잘랐다고 믿은 채 예전 모델로 도는 것이 더 나쁩니다")
    return (lo, hi)


# --------------------------------------------------------------------------- URDF data model


@dataclasses.dataclass(frozen=True)
class UrdfJoint:
    name: str
    type: str  # revolute | continuous | prismatic | fixed
    parent: str
    child: str
    origin: np.ndarray  # (4, 4) constant parent -> joint frame
    axis: np.ndarray  # (3,) unit, in the joint frame
    lower: Optional[float] = None
    upper: Optional[float] = None
    # `<limit velocity=... acceleration=...>`. Both are present on every RB-Y1 joint and neither is
    # used by AG3S itself — a collision model does not care how fast a joint may move. They are
    # parsed here because they belong to the URDF and the alternative is a second parser: a
    # trajectory optimizer needs them to bound `|dq|` and `|d2q|`, and hard-coding a guess for a
    # 20-joint robot is how a limit ends up wrong on the one joint nobody checked.
    velocity: Optional[float] = None
    acceleration: Optional[float] = None

    @property
    def is_movable(self) -> bool:
        return self.type in ("revolute", "continuous", "prismatic")


@dataclasses.dataclass(frozen=True)
class UrdfCapsule:
    """A collision capsule attached to a link.

    `origin` is the capsule frame in link coordinates; the segment runs along the capsule frame's
    +z axis from ``-length/2`` to ``+length/2``, with hemispherical caps of `radius` — the same
    convention MuJoCo and URDF extensions use.
    """

    link: str
    origin: np.ndarray  # (4, 4)
    radius: float
    length: float


@dataclasses.dataclass(frozen=True)
class UrdfModel:
    links: tuple[str, ...]
    joints: tuple[UrdfJoint, ...]
    capsules: tuple[UrdfCapsule, ...]
    root: str

    def joint_by_name(self, name: str) -> UrdfJoint:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(f"no joint named {name!r}; have {[j.name for j in self.joints]}")

    def movable_joint_names(self) -> list[str]:
        return [j.name for j in self.joints if j.is_movable]


# ------------------------------------------------------------------------------ URDF parsing


def _floats(text: str | None, default: Sequence[float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, np.float64)
    return np.asarray([float(v) for v in text.replace(",", " ").split()], np.float64)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        np.float64,
    )


def _limit_value(limit: ET.Element | None, name: str) -> Optional[float]:
    """One attribute of a `<limit>` element, or None when it is absent or unparseable.

    None rather than a default, because the two mean different things downstream: a missing velocity
    limit is "the URDF does not say", which a consumer should notice and decide about, while a
    default silently asserts a number the manufacturer never provided.
    """
    if limit is None:
        return None
    raw = limit.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _origin_matrix(elem: ET.Element | None) -> np.ndarray:
    T = np.eye(4)
    if elem is None:
        return T
    origin = elem.find("origin")
    if origin is None:
        return T
    T[:3, :3] = _rpy_to_matrix(_floats(origin.get("rpy"), (0.0, 0.0, 0.0)))
    T[:3, 3] = _floats(origin.get("xyz"), (0.0, 0.0, 0.0))
    return T


def _collision_elements(link: ET.Element) -> Iterable[ET.Element]:
    """Live `<collision>` children, plus any hiding inside an XML comment.

    RB-Y1's URDF ships its collision capsules commented out — presumably because the visual meshes
    are what the simulator consumes. They are still the manufacturer's own collision model, and
    re-deriving capsules from meshes we cannot even load would be strictly worse, so they are read
    back rather than reinvented. Comments that are not XML (the file has plenty: "270Nm, 5rad/s^2,
    ...") are skipped silently.
    """
    for child in link:
        if child.tag == "collision":
            yield child
        elif child.tag is ET.Comment and child.text and "<collision" in child.text:
            try:
                yield ET.fromstring(f"<wrap>{child.text}</wrap>").find("collision")
            except ET.ParseError:
                continue


def parse_urdf(path: str | pathlib.Path) -> UrdfModel:
    """Read links, joints and collision capsules. Meshes are ignored — nothing here needs them.

    The path goes through `ag3s.asset_path.resolve_asset`, so the workspace-relative default
    (`src/rby1_description/...`) still works when the assets actually live somewhere else —
    `pi05_TO_hybrid/rby1_description/` in this container. Resolving **here** rather than at each
    call site covers every entry point at once; three of them call `parse_urdf(RBY1_URDF)` directly.
    """
    from benchmark.ag3s.runtime.asset_path import resolve_asset

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(resolve_asset(path, what="URDF").read_text(), parser=parser)

    links: list[str] = []
    capsules: list[UrdfCapsule] = []
    for link in root.findall("link"):
        name = link.get("name")
        if name is None:
            continue
        links.append(name)
        for coll in _collision_elements(link):
            if coll is None:
                continue
            cap = coll.find("geometry/capsule")
            if cap is None:
                continue
            capsules.append(
                UrdfCapsule(
                    link=name,
                    origin=_origin_matrix(coll),
                    radius=float(cap.get("radius", 0.0)),
                    length=float(cap.get("length", 0.0)),
                )
            )

    joints: list[UrdfJoint] = []
    for j in root.findall("joint"):
        jtype = j.get("type", "fixed")
        parent = j.find("parent")
        child = j.find("child")
        if parent is None or child is None:
            continue
        axis = _floats(j.find("axis").get("xyz") if j.find("axis") is not None else None, (1.0, 0.0, 0.0))
        nrm = float(np.linalg.norm(axis))
        limit = j.find("limit")
        joints.append(
            UrdfJoint(
                name=j.get("name", ""),
                type=jtype,
                parent=parent.get("link", ""),
                child=child.get("link", ""),
                origin=_origin_matrix(j),
                axis=axis / nrm if nrm > 1e-12 else np.array([1.0, 0.0, 0.0]),
                lower=_limit_value(limit, "lower"),
                upper=_limit_value(limit, "upper"),
                velocity=_limit_value(limit, "velocity"),
                acceleration=_limit_value(limit, "acceleration"),
            )
        )

    children = {j.child for j in joints}
    roots = [ln for ln in links if ln not in children]
    if not roots:
        raise ValueError(f"{path}: no root link (every link is some joint's child)")
    return UrdfModel(tuple(links), tuple(joints), tuple(capsules), roots[0])


# ------------------------------------------------------------------------ arithmetic backends


class _NumpyBackend:
    """FK in float64."""

    @staticmethod
    def cos(x: Any) -> Any:
        return math.cos(float(x))

    @staticmethod
    def sin(x: Any) -> Any:
        return math.sin(float(x))

    @staticmethod
    def mat4(rows: list[list[Any]]) -> np.ndarray:
        return np.asarray(rows, np.float64)

    @staticmethod
    def const(mat: np.ndarray) -> np.ndarray:
        return np.asarray(mat, np.float64)

    @staticmethod
    def matmul(a: Any, b: Any) -> Any:
        return a @ b

    @staticmethod
    def index(q: Any, i: int) -> Any:
        return q[i]


class _CasadiBackend:
    """The identical FK, in CasADi SX/MX.

    Constants become `DM` before they meet a symbol: multiplying a numpy array by an SX from the
    left goes through numpy's broadcasting first and produces an object array of scalar SX, which
    silently costs a hundredfold and breaks `jacobian`.
    """

    def __init__(self) -> None:
        import casadi as ca  # local import: AG3S must import fine without casadi

        self.ca = ca

    def cos(self, x: Any) -> Any:
        return self.ca.cos(x)

    def sin(self, x: Any) -> Any:
        return self.ca.sin(x)

    def mat4(self, rows: list[list[Any]]) -> Any:
        return self.ca.vertcat(*[self.ca.horzcat(*row) for row in rows])

    def const(self, mat: np.ndarray) -> Any:
        return self.ca.DM(np.asarray(mat, np.float64))

    def matmul(self, a: Any, b: Any) -> Any:
        return self.ca.mtimes(a, b)

    def index(self, q: Any, i: int) -> Any:
        return q[i]


def _joint_transform(joint: UrdfJoint, value: Any, be: Any) -> Any:
    """Constant joint origin composed with the joint's own motion."""
    origin = be.const(joint.origin)
    if joint.type == "fixed":
        return origin
    x, y, z = (float(v) for v in joint.axis)
    if joint.type == "prismatic":
        motion = be.mat4(
            [
                [1.0, 0.0, 0.0, x * value],
                [0.0, 1.0, 0.0, y * value],
                [0.0, 0.0, 1.0, z * value],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return be.matmul(origin, motion)
    # revolute / continuous: Rodrigues about a constant unit axis.
    c = be.cos(value)
    s = be.sin(value)
    C = 1.0 - c
    motion = be.mat4(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s, 0.0],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s, 0.0],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return be.matmul(origin, motion)


# ---------------------------------------------------------------------------------- the model


DEFAULT_RBY1_JOINTS: tuple[str, ...] = (
    *(f"torso_{i}" for i in range(6)),
    *(f"right_arm_{i}" for i in range(7)),
    *(f"left_arm_{i}" for i in range(7)),
)
"""The 20 DoF that actually move the RB-Y1's collision geometry through the workspace.

Wheels and head are excluded: the wheels move the whole base (a different problem, handled by
`T_base_cam` being expressed in the base frame) and the head carries no collision capsule. Grippers
are excluded because their prismatic travel is millimetres against a 3.5 cm capsule radius.
"""


class UrdfSphereChain:
    """`RobotCollisionModel` over a URDF's capsule collision geometry.

    Each capsule is discretized into spheres along its segment, and the spheres are **inflated** to
    close the gaps between them. Spacing alone is not enough: with centres a distance `s` apart, the
    capsule-surface point midway between two of them sits `sqrt(r^2 + (s/2)^2)` from the nearest
    centre, so even at `s = r` a plain chain leaves 11.8% of the radius uncovered. Setting each
    sphere to that radius makes the union contain the capsule exactly, which is the direction that
    matters — under-covering the robot means the constraints protect less of it than they claim.

    Args:
        model: parsed URDF.
        joint_names: the ordered `q`. Movable joints not listed are pinned to `fixed_joint_values`.
        fixed_joint_values: value for each pinned movable joint; default 0.0.
        link_filter: keep only capsules on these links (e.g. arms only). None keeps all.
        extra_capsules: capsules the URDF does not provide, merged with the parsed ones. RB-Y1's
            file is a case in point — it carries collision capsules for the torso and arm links 0-5
            and nothing else, so the base, wheels, wrists, grippers and head are invisible to a
            self-filter built from it alone. On a head camera looking down at its own body that
            leaves thousands of robot points in the cloud, which then cluster into a phantom
            obstacle welded to the robot. A caller that has another source for those links (a
            simulator's meshes, a hand-measured box) passes them here rather than editing the URDF.
        sphere_spacing: maximum gap between consecutive sphere centres, in units of the capsule
            radius. 1.0 guarantees coverage; smaller is finer and slower. **Finer spacing also makes
            every sphere thinner** — the inflation term is `(spacing/2)^2` under the root — which is
            the cheap half of getting the arm down to its real thickness (T6f).
        max_spheres_per_capsule: hard cap, so a long capsule cannot blow up the constraint count.
            It binds before `sphere_spacing` does on RB-Y1's forearm (250 mm long, 75 mm radius:
            8 spheres is reached at a spacing of 0.48), so lowering the spacing without raising this
            changes nothing.
        capsule_radius_scale: multiplies every capsule radius before inflation. **1.0 is the URDF's
            own number and the default.** Below 1.0 the spheres are thinner than the URDF capsule
            and the union no longer contains it; `coverage_shortfall` records that and the
            constructor logs a warning, because a silently thinner robot is a safety layer that
            protects less than it says.
        max_sphere_radius: metres; caps the **final** (inflated) radius. Same bargain as
            `capsule_radius_scale` — reported, never silent. `None` (default) caps nothing.
        capsule_radius_scale_by_link: `{link: scale}` — **per-link** override of
            `capsule_radius_scale`, for links not named it. `None` (default) means there is no
            override and every capsule uses the global scale, so the model is letter-for-letter the
            one built before this argument existed.

            It exists because the global scale is the wrong instrument for one question. Thinning
            the gripper so the fingers can descend beside an object on a table also thins the
            forearm, the upper arm and the opposite arm by the same factor, and those are the parts
            whose thickness was the whole point (T7b). A link named here is thinned; every other
            link keeps the number it had.

            **An unknown link name is an error, not a no-op.** A typo would otherwise leave every
            capsule at the global scale while the operator believes one link was thinned — the same
            failure `--exclude-links` had to be guarded against.
    """

    def __init__(
        self,
        model: UrdfModel,
        joint_names: Sequence[str] = DEFAULT_RBY1_JOINTS,
        *,
        fixed_joint_values: dict[str, float] | None = None,
        link_filter: Sequence[str] | None = None,
        extra_capsules: Sequence[UrdfCapsule] = (),
        sphere_spacing: float = 1.0,
        max_spheres_per_capsule: int = 8,
        capsule_radius_scale: float = 1.0,
        max_sphere_radius: float | None = None,
        capsule_radius_scale_by_link: Mapping[str, float] | None = None,
        capsule_extent_by_link: Mapping[str, Any] | None = None,
    ):
        self.model = model
        self.joint_names = tuple(joint_names)
        for name in self.joint_names:
            if not model.joint_by_name(name).is_movable:
                raise ValueError(f"joint {name!r} is {model.joint_by_name(name).type}, not movable")
        self.fixed_joint_values = dict(fixed_joint_values or {})
        self.sphere_spacing = float(sphere_spacing)
        self.max_spheres_per_capsule = int(max_spheres_per_capsule)
        self.capsule_radius_scale = float(capsule_radius_scale)
        self.max_sphere_radius = None if max_sphere_radius is None else float(max_sphere_radius)
        if self.sphere_spacing <= 0.0:
            raise ValueError(f"sphere_spacing must be > 0, got {sphere_spacing}")
        if self.max_spheres_per_capsule < 1:
            raise ValueError(
                f"max_spheres_per_capsule must be >= 1, got {max_spheres_per_capsule}")
        if not 0.0 < self.capsule_radius_scale:
            raise ValueError(f"capsule_radius_scale must be > 0, got {capsule_radius_scale}")
        if self.max_sphere_radius is not None and self.max_sphere_radius <= 0.0:
            raise ValueError(f"max_sphere_radius must be > 0 or None, got {max_sphere_radius}")
        #: link 별 배율. **비어 있으면 전역 배율만 쓰인다** — 이 인자가 없던 때와 글자 그대로
        #: 같은 모델이라는 뜻이다.
        self.capsule_radius_scale_by_link: dict[str, float] = {
            str(k): float(v) for k, v in dict(capsule_radius_scale_by_link or {}).items()}
        #: link 별 **길이/범위** 손잡이. link frame z 의 유지 구간 `(z_min, z_max)` 다.
        #: **비어 있으면 아무것도 잘리지 않는다** — 이 인자가 없던 때와 글자 그대로 같은 모델.
        #:
        #: 굵기 손잡이와 **독립**이다. 한 link 에 둘을 같이 걸 수 있고, 그래야 하는 경우가
        #: RB-Y1 `link_*_arm_5` 다 (잘라도 반지름이 그대로면 잘린 끝의 구가 아래로 부푼다).
        self.capsule_extent_by_link: dict[str, tuple[float, float]] = {
            str(k): _parse_extent(k, v)
            for k, v in dict(capsule_extent_by_link or {}).items()}
        bad = {k: v for k, v in self.capsule_radius_scale_by_link.items() if not v > 0.0}
        if bad:
            raise ValueError(
                f"capsule_radius_scale_by_link values must be > 0, got {bad}")
        unknown_scaled = sorted(set(self.capsule_radius_scale_by_link) - set(model.links))
        if unknown_scaled:
            # 조용히 무시하면 **아무 link 도 얇아지지 않은 채 얇게 했다고 믿는다.**
            raise ValueError(
                f"capsule_radius_scale_by_link references links not in the URDF: "
                f"{unknown_scaled}. 조용히 넘기면 아무 link 도 얇아지지 않은 채 얇게 했다고 "
                f"믿게 되므로 여기서 멈춥니다")
        unknown_trimmed = sorted(set(self.capsule_extent_by_link) - set(model.links))
        if unknown_trimmed:
            # 같은 이유, 같은 규율 — 조용히 넘기면 **아무것도 안 잘린 채 잘랐다고 믿는다.**
            raise ValueError(
                f"capsule_extent_by_link references links not in the URDF: {unknown_trimmed}. "
                f"조용히 넘기면 아무것도 안 잘린 채 잘랐다고 믿게 되므로 여기서 멈춥니다")
        #: 이 설정이 **덮지 못한** capsule 들. `CoverageShortfall` 목록이고, 비어 있으면 구의
        #: 합집합이 URDF capsule 을 그대로 담는다 (예전 동작).
        self.coverage_shortfall: tuple[CoverageShortfall, ...] = ()
        #: 잘라낸 capsule 들. `CapsuleTrim` 목록이고, 비어 있으면 아무것도 안 잘렸다.
        self.capsule_trims: tuple[CapsuleTrim, ...] = ()

        keep = None if link_filter is None else set(link_filter)
        all_capsules = tuple(model.capsules) + tuple(extra_capsules)
        self.capsules = tuple(c for c in all_capsules if keep is None or c.link in keep)
        known_links = set(model.links)
        unknown = {c.link for c in extra_capsules} - known_links
        if unknown:
            raise ValueError(f"extra_capsules reference links not in the URDF: {sorted(unknown)}")
        # **이 모델에 capsule 이 없는 link 를 얇게 하라는 요청도 거절한다.** URDF 에는 있지만
        # `link_filter` 가 이미 빼 버린 link 를 주면 아무 일도 안 일어나는데, 요청한 사람은
        # 얇아졌다고 믿는다 — `constraint_links_minus` 가 같은 이유로 같은 것을 거절한다.
        capsule_links = {c.link for c in self.capsules}
        scaled_without_capsule = sorted(set(self.capsule_radius_scale_by_link) - capsule_links)
        if scaled_without_capsule:
            raise ValueError(
                f"capsule_radius_scale_by_link names link(s) this model has no capsule for: "
                f"{scaled_without_capsule}. 이 모델의 capsule link: {sorted(capsule_links)}. "
                f"배율이 아무 데도 안 걸리는데 걸었다고 믿게 되므로 여기서 멈춥니다")
        trimmed_without_capsule = sorted(set(self.capsule_extent_by_link) - capsule_links)
        if trimmed_without_capsule:
            raise ValueError(
                f"capsule_extent_by_link names link(s) this model has no capsule for: "
                f"{trimmed_without_capsule}. 이 모델의 capsule link: {sorted(capsule_links)}. "
                f"자를 것이 아무 데도 없는데 잘랐다고 믿게 되므로 여기서 멈춥니다")

        self._q_index = {name: i for i, name in enumerate(self.joint_names)}
        self._parent_joint = {j.child: j for j in model.joints}
        # Precompute, per capsule, the sphere centres in the *link* frame and their radii. These are
        # constant, so the per-call work is one FK chain and one 4x4 * 3 multiply per sphere.
        self._local_spheres: list[tuple[str, np.ndarray, float]] = []
        #: `(link, 원했던 구 수, 상한에 잘린 구 수)` — `max_spheres_per_capsule` 이 먼저 걸린 곳.
        self._spacing_capped: list[tuple[str, int, int]] = []
        shortfall: list[CoverageShortfall] = []
        for cap in self.capsules:
            centres, radius, required = self._capsule_sphere_centres(cap)
            for centre in centres:
                self._local_spheres.append((cap.link, centre, radius))
            if radius < required - _COVERAGE_TOL:
                shortfall.append(CoverageShortfall(
                    link=cap.link, n_spheres=len(centres), capsule_radius=float(cap.radius),
                    capsule_length=float(cap.length), sphere_radius=float(radius),
                    required_radius=float(required)))
        self.coverage_shortfall = tuple(shortfall)
        self.capsule_trims = self._record_trims(model)
        self._chains = {link: self._chain_to(link) for link in {c.link for c in self.capsules}}
        # **조용히 가늘어진 모델로 떠 있는 것이 가장 나쁘다.** 여기서 찍는 이유는 호출자가
        # 여럿이기 때문이다 — 서버·실험 스크립트·테스트가 각자 이 클래스를 짓는데, 경고를
        # 진입점에 두면 한 곳만 안 찍고 그 실행이 기록에 "예전과 같은 모델" 로 남는다.
        if self.coverage_shortfall or self.capsule_trims or self._spacing_capped:
            logging.getLogger(__name__).warning("%s", self.coverage_report())

    # --- introspection ------------------------------------------------------------------
    @property
    def nq(self) -> int:
        return len(self.joint_names)

    @property
    def n_spheres(self) -> int:
        return len(self._local_spheres)

    @property
    def radii(self) -> np.ndarray:
        return np.asarray([r for _, _, r in self._local_spheres], np.float64)

    @property
    def sphere_link_names(self) -> tuple[str, ...]:
        """The owning link of each sphere, in the order both FK paths emit.

        `_local_spheres` is built once and iterated in the same order by `sphere_centers_numeric`,
        `sphere_centers_symbolic` and here, so the three cannot drift. That ordering is what the
        clearance policy keys on — it is how "the right fingertip may touch the target" reaches a
        specific constraint row rather than a sphere index nobody can interpret.
        """
        return tuple(link for link, _, _ in self._local_spheres)

    def scale_for_link(self, link: str) -> float:
        """이 link 의 capsule 반지름 배율. 이름이 안 적혀 있으면 전역 배율.

        배율이 나오는 곳을 **한 곳**으로 두려고 메서드로 뺐다. 두 곳에서 읽으면 구를 만드는
        쪽과 로그에 찍는 쪽이 갈라지고, 갈라진 쪽이 로그면 실제보다 두꺼운 모델로 기록된다.
        """
        return float(self.capsule_radius_scale_by_link.get(
            str(link), self.capsule_radius_scale))

    def _capsule_sphere_centres(self, cap: UrdfCapsule):
        """``(centres, radius, required_radius)`` for one capsule, in the link frame.

        `required_radius` is what the chain would need to contain the URDF capsule. It equals
        `radius` unless a radius scale or cap was asked for, and the difference is what
        `coverage_shortfall` reports.
        """
        if cap.radius <= 0.0:
            return [], 0.0, 0.0
        # **link frame z 로 자른다** (`capsule_extent_by_link`). 축이 남는 구간만 구를 놓는다 —
        # 굵기를 줄이는 것과 **다른 축**이고, 한 link 에 둘을 같이 걸 수 있다.
        t_lo, t_hi = self._kept_axis_span(cap)
        if t_lo is None:
            return [], 0.0, 0.0  # 이 capsule 은 통째로 잘려 나갔다
        span = float(t_hi - t_lo)
        # **간격은 URDF 의 반지름으로 정한다, 줄인 반지름이 아니다.** 줄인 반지름으로 정하면
        # 반지름을 줄일 때마다 구가 자동으로 늘어나 "가늘게" 와 "촘촘하게" 가 한 손잡이에
        # 묶인다 — 둘은 값이 다르고(하나는 덮개를 잃고 하나는 행을 늘린다) 따로 돌려야 한다.
        n = int(math.ceil(span / max(self.sphere_spacing * cap.radius, 1e-9))) + 1
        n_wanted = n
        n = int(np.clip(n, 1, self.max_spheres_per_capsule))
        # **상한에 잘렸으면 기억한다.** `sphere_spacing` 을 내렸는데 이 상한이 먼저 걸리면
        # 구가 촘촘해지지 않고, 그러면 유효 반지름도 안 내려간다 (`_effective_radius` 의 간격
        # 항). 조용히 지나가면 얇게 만들었다고 믿는데 안 얇아진다 (A2 T8e 실측).
        if n_wanted > n:
            self._spacing_capped.append((cap.link, n_wanted, n))
        ts = np.zeros(1) + 0.5 * (t_lo + t_hi) if n == 1 else np.linspace(t_lo, t_hi, n)
        R, p = cap.origin[:3, :3], cap.origin[:3, 3]
        centres = [p + R @ np.array([0.0, 0.0, float(t)]) for t in ts]
        # 덮개 기준은 **URDF capsule 전체**가 아니라 지금 덮기로 한 구간이다. 잘라낸 구간은
        # 부족이 아니라 **의도**이고, 그 사실은 `CapsuleTrim` 이 따로 말한다 — 두 축을 한
        # 숫자에 섞으면 "얼마나 얇은가" 와 "어디까지 덮는가" 를 구분할 수 없다.
        required = self._effective_radius(cap.radius, span, ts)
        radius = self._effective_radius(
            cap.radius * self.scale_for_link(cap.link), span, ts)
        if self.max_sphere_radius is not None:
            radius = min(radius, self.max_sphere_radius)
        return centres, radius, required

    def _kept_axis_span(self, cap: UrdfCapsule):
        """이 capsule 축에서 **남기는** 파라미터 구간 `(t_lo, t_hi)`. 통째로 잘리면 `(None, None)`.

        축의 z 좌표는 `z(t) = p_z + R[2,2] · t` 다. `R[2,2] != 0` 이면 단조라서 z 구간이 t 구간으로
        정확히 옮겨진다 (`link_*_arm_5` 는 `R[2,2] = +1.000`, 즉 축이 link z 와 나란하다).

        **`R[2,2] == 0` 인 capsule 은 축이 z 평면 안에 누워 있다** — 손가락 capsule 이 그렇다
        (실측 `R[2,2] = 0.000`, 축이 x/y 방향). 그 capsule 에는 "z 로 어디를 자른다" 가 없고
        z 값이 하나뿐이므로, 그 z 가 유지 구간 안이면 **통째로 남기고** 밖이면 **통째로 뺀다.**
        근사하지 않는다 — 근사하면 어느 구가 남는지 아무도 예측할 수 없다.
        """
        window = self.capsule_extent_by_link.get(cap.link)
        half = float(cap.length) / 2.0
        if window is None:
            return -half, half
        z_min, z_max = window
        axis_z = float(cap.origin[2, 2])
        p_z = float(cap.origin[2, 3])
        if abs(axis_z) <= 1e-9:
            return (-half, half) if z_min <= p_z <= z_max else (None, None)
        a = (z_min - p_z) / axis_z
        b = (z_max - p_z) / axis_z
        lo, hi = (a, b) if a <= b else (b, a)
        t_lo, t_hi = max(-half, lo), min(half, hi)
        if not (t_hi > t_lo - 1e-12):
            return (None, None)
        return float(t_lo), float(min(t_hi, half))

    def _record_trims(self, model: UrdfModel) -> tuple[CapsuleTrim, ...]:
        """자른 link 마다 **무엇이 풀려났는지** 기록한다. 기록이 없으면 로그도 없다.

        `reach` 는 축이 아니라 구 **표면**이 닿는 z 다. 자르는 것만으로 표면이 그만큼 물러나지
        않는다는 사실이 이 표의 요점이다 — 잘린 끝의 구가 여전히 자기 반지름만큼 부푼다.
        """
        if not self.capsule_extent_by_link:
            return ()
        by_link: dict[str, list[UrdfCapsule]] = {}
        for cap in self.capsules:
            if cap.link in self.capsule_extent_by_link:
                by_link.setdefault(cap.link, []).append(cap)
        have_spheres = set(self.sphere_link_names)
        out: list[CapsuleTrim] = []
        for link, caps in sorted(by_link.items()):
            keep = self.capsule_extent_by_link[link]
            axis_lo, axis_hi = math.inf, -math.inf
            reach_before, reach_after = math.inf, math.inf
            n_before = n_after = 0
            for cap in caps:
                half = float(cap.length) / 2.0
                p_z, axis_z = float(cap.origin[2, 3]), float(cap.origin[2, 2])
                axis_lo = min(axis_lo, p_z - abs(axis_z) * half)
                axis_hi = max(axis_hi, p_z + abs(axis_z) * half)
                full = self._sphere_zs(cap, -half, half)
                n_before += len(full[0])
                reach_before = min(reach_before, min(full[0]) - full[1])
                t_lo, t_hi = self._kept_axis_span(cap)
                if t_lo is None:
                    continue
                kept = self._sphere_zs(cap, t_lo, t_hi)
                n_after += len(kept[0])
                reach_after = min(reach_after, min(kept[0]) - kept[1])
            released = tuple(
                (name, z, name in have_spheres)
                for name, z in self._descendant_origins_z(model, link)
                if not (keep[0] <= z <= keep[1]))
            out.append(CapsuleTrim(
                link=link, keep_z_min=float(keep[0]), keep_z_max=float(keep[1]),
                axis_z_min=float(axis_lo), axis_z_max=float(axis_hi),
                n_spheres_before=n_before, n_spheres_after=n_after,
                reach_z_min_before=float(reach_before),
                reach_z_min_after=float(reach_after if n_after else reach_before),
                released_links=released))
        return tuple(out)

    def _sphere_zs(self, cap: UrdfCapsule, t_lo: float, t_hi: float):
        """`(구 중심의 link-frame z 목록, 유효 반지름)` — `_capsule_sphere_centres` 와 같은 규칙."""
        span = float(t_hi - t_lo)
        n = int(math.ceil(span / max(self.sphere_spacing * cap.radius, 1e-9))) + 1
        n = int(np.clip(n, 1, self.max_spheres_per_capsule))
        ts = (np.zeros(1) + 0.5 * (t_lo + t_hi)) if n == 1 else np.linspace(t_lo, t_hi, n)
        radius = self._effective_radius(
            cap.radius * self.scale_for_link(cap.link), span, ts)
        if self.max_sphere_radius is not None:
            radius = min(radius, self.max_sphere_radius)
        zs = [float(cap.origin[2, 3] + (cap.origin[:3, :3] @ np.array([0.0, 0.0, float(t)]))[2])
              for t in ts]
        return zs, float(radius)

    @staticmethod
    def _descendant_origins_z(model: UrdfModel, link: str) -> list[tuple[str, float]]:
        """`link` 아래에 달린 link 들의 원점 z, **`link` 프레임에서** (관절값 0 기준).

        관절값 0 을 쓰는 것을 로그가 밝힌다. 팔이 접히면 z 는 달라지지만, 여기서 묻는 것은
        "이 capsule 이 자기 사슬의 어디까지를 덮고 있었나" 라는 **구조적** 질문이고 그것은
        관절값에 무관하다 — 손바닥은 어느 자세에서도 손목 아래에 있다.
        """
        children: dict[str, list[UrdfJoint]] = {}
        for j in model.joints:
            children.setdefault(j.parent, []).append(j)
        out: list[tuple[str, float]] = []
        stack = [(link, np.eye(4))]
        while stack:
            cursor, T = stack.pop()
            for joint in children.get(cursor, ()):  # 관절값 0 → origin 만
                T_child = T @ np.asarray(joint.origin, np.float64)
                out.append((joint.child, float(T_child[2, 3])))
                stack.append((joint.child, T_child))
        return out

    @staticmethod
    def _effective_radius(radius: float, length: float, ts: np.ndarray) -> float:
        """Radius that closes the gap between consecutive spheres — see the class docstring.

        A single sphere has to swallow the whole capsule, so it takes the half-length plus the
        radius; a chain only has to reach the midpoint between neighbours.
        """
        if ts.size < 2:
            return float(radius + length / 2.0)
        spacing = float(ts[1] - ts[0])
        return float(math.sqrt(float(radius) ** 2 + (spacing / 2.0) ** 2))

    # --- coverage -----------------------------------------------------------------------
    def coverage_report(self) -> str:
        """덮지 못하는 capsule 이 있으면 **크게** 말하는 한 덩어리 문자열. 없으면 한 줄.

        문장을 여기서 만드는 이유는 서버·실험·테스트가 같은 글을 봐야 하기 때문이다. 진입점마다
        따로 쓰면 하나는 경고를 빼먹고, 빼먹은 실행이 기록에 "예전과 같은 모델" 로 남는다.
        """
        settings = (f"sphere_spacing={self.sphere_spacing:g}, "
                    f"max_spheres_per_capsule={self.max_spheres_per_capsule}, "
                    f"capsule_radius_scale={self.capsule_radius_scale:g}, "
                    f"max_sphere_radius="
                    + ("none" if self.max_sphere_radius is None
                       else f"{self.max_sphere_radius * 1000:.1f} mm"))
        if self.capsule_radius_scale_by_link:
            # **link 별 배율은 반드시 함께 찍는다.** 전역 배율만 보이면 로그를 읽는 사람은
            # 손가락이 다른 값으로 얇아진 것을 알 방법이 없다.
            per_link = ", ".join(
                f"{k}={v:g}" for k, v in sorted(self.capsule_radius_scale_by_link.items()))
            settings += f", per-link capsule_radius_scale: {per_link}"
        if self.capsule_extent_by_link:
            # **길이 손잡이도 반드시 함께 찍는다.** 굵기만 보이면 로그를 읽는 사람은 어느 구간이
            # 아예 안 덮이는지 알 방법이 없다.
            per_link = ", ".join(
                f"{k}=[{v[0] * 1000:+.0f}, "
                + ("inf" if not math.isfinite(v[1]) else f"{v[1] * 1000:+.0f}") + "]"
                for k, v in sorted(self.capsule_extent_by_link.items()))
            settings += f", per-link capsule extent (link-frame z, mm): {per_link}"
        lines: list[str] = []
        if self.coverage_shortfall:
            worst = max(s.shortfall_m for s in self.coverage_shortfall)
            lines += [
                "!!! THIS SPHERE MODEL DOES NOT COVER THE ROBOT !!!",
                f"    {len(self.coverage_shortfall)} of {len(self.capsules)} capsule(s) are "
                f"thinner than the URDF says, worst by {worst * 1000:.1f} mm ({settings}).",
                "    빼는 것과 같은 성질의 flag 다 — 이 capsule 이 무엇에 부딪혀도 그만큼은 "
                "아무도 막지 않는다. 기준은 **URDF capsule** 이고, URDF 자체가 mesh 보다 두꺼울 "
                "수 있다.",
            ]
            lines += [f"    - {s.summary()}" for s in self.coverage_shortfall]
        if self.capsule_trims:
            lines += [
                "!!! CAPSULES ARE CUT SHORT (capsule_extent_by_link) !!!",
                "    잘린 구간에는 구가 **하나도 없다** — 굵기를 줄이는 것과 다른 축이다. "
                "z 는 link frame, 하위 link 원점은 관절값 0 기준이다.",
            ]
            lines += [f"    - {t.summary()}" for t in self.capsule_trims]
        if self._spacing_capped:
            worst_link, wanted, got = max(self._spacing_capped, key=lambda r: r[1] - r[2])
            lines += [
                "!!! sphere_spacing IS NOT TAKING EFFECT — max_spheres_per_capsule IS BINDING !!!",
                f"    {len(self._spacing_capped)} capsule(s) wanted more spheres than the cap "
                f"({self.max_spheres_per_capsule}); worst {worst_link} wanted {wanted}, got {got}.",
                "    구가 촘촘해지지 않으면 유효 반지름도 안 내려간다 "
                "(r_eff = sqrt((r·scale)^2 + (간격/2)^2) 의 간격 항이 안 줄기 때문이다). "
                "**얇게 만들었다고 믿는데 안 얇아진 상태다** — max_spheres_per_capsule 을 "
                "올리십시오.",
            ]
        if not lines:
            return f"sphere chain covers every capsule ({self.n_spheres} spheres; {settings})"
        lines.append(f"    지금 모델: 구 {self.n_spheres} 개 · capsule {len(self.capsules)} 개 "
                     f"({settings})")
        lines.append("    link 별 유효 반지름: " + self.radius_by_link_summary())
        return "\n".join(lines)

    def radius_by_link_summary(self) -> str:
        """`link=구수 × 반지름mm` 한 줄. **굵기·길이·간격의 결과를 한 자리에서 읽는 유일한 곳**이다.

        배율과 절단면과 간격을 따로 찍으면 읽는 사람이 `r_eff` 를 손으로 계산해야 하고, 그 계산이
        틀리는 순간 "대폭 줄였다" 는 판단이 로그와 어긋난다.
        """
        by_link: dict[str, list[float]] = {}
        for link, _, radius in self._local_spheres:
            by_link.setdefault(link, []).append(float(radius))
        return ", ".join(
            f"{link}={len(rs)}×{max(rs) * 1000:.2f} mm" for link, rs in sorted(by_link.items()))

    def _chain_to(self, link: str) -> tuple[UrdfJoint, ...]:
        """Joints from the URDF root down to `link`, root-first."""
        chain: list[UrdfJoint] = []
        cursor = link
        seen: set[str] = set()
        while cursor in self._parent_joint:
            if cursor in seen:
                raise ValueError(f"cycle in URDF kinematic tree at {cursor!r}")
            seen.add(cursor)
            joint = self._parent_joint[cursor]
            chain.append(joint)
            cursor = joint.parent
        return tuple(reversed(chain))

    def _joint_value(self, joint: UrdfJoint, q: Any, be: Any) -> Any:
        if joint.name in self._q_index:
            return be.index(q, self._q_index[joint.name])
        return self.fixed_joint_values.get(joint.name, 0.0)

    def _link_transforms(self, q: Any, be: Any) -> dict[str, Any]:
        """FK for every link that carries a capsule. Shared by both backends — that is the point."""
        out: dict[str, Any] = {}
        for link, chain in self._chains.items():
            T = be.const(np.eye(4))
            for joint in chain:
                T = be.matmul(T, _joint_transform(joint, self._joint_value(joint, q, be), be))
            out[link] = T
        return out

    # --- RobotCollisionModel ------------------------------------------------------------
    def sphere_centers_numeric(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, np.float64).reshape(-1)
        if q.shape[0] != self.nq:
            raise ValueError(f"q must have {self.nq} entries, got {q.shape[0]}")
        be = _NumpyBackend()
        T = self._link_transforms(q, be)
        centres = np.empty((len(self._local_spheres), 3), np.float64)
        radii = np.empty(len(self._local_spheres), np.float64)
        for i, (link, local, radius) in enumerate(self._local_spheres):
            M = T[link]
            centres[i] = M[:3, :3] @ local + M[:3, 3]
            radii[i] = radius
        return centres, radii

    def sphere_centers_symbolic(self, q: Any) -> list[tuple[Any, float]]:
        be = _CasadiBackend()
        ca = be.ca
        T = self._link_transforms(q, be)
        out: list[tuple[Any, float]] = []
        for link, local, radius in self._local_spheres:
            M = T[link]
            centre = ca.mtimes(M[:3, :3], ca.DM(local.reshape(3, 1))) + M[:3, 3]
            out.append((centre, float(radius)))
        return out

    # --- convenience --------------------------------------------------------------------
    def link_pose(self, q: np.ndarray, link: str) -> np.ndarray:
        """Numeric 4x4 pose of any link, for visualization and debugging."""
        if link not in self._chains:
            self._chains[link] = self._chain_to(link)
        be = _NumpyBackend()
        T = be.const(np.eye(4))
        for joint in self._chains[link]:
            T = T @ _joint_transform(joint, self._joint_value(joint, np.asarray(q, np.float64), be), be)
        return T

    def link_pose_symbolic(self, q: Any, link: str) -> Any:
        """The same chain as `link_pose`, in CasADi. `(4, 4)` SX/MX in `q`.

        Attached-object geometry needs this: the held object's pose is `FK_parent(q) @
        T_parent_object`, and if that FK were numeric the optimizer could not differentiate the
        object's motion with respect to the joints carrying it. Sharing `_joint_transform` with the
        numeric path is the same reason the sphere FK does — two hand-written chains eventually
        disagree, and here the disagreement would move the held object relative to the gripper.
        """
        if link not in self._chains:
            self._chains[link] = self._chain_to(link)
        be = _CasadiBackend()
        T = be.const(np.eye(4))
        for joint in self._chains[link]:
            T = be.matmul(T, _joint_transform(joint, self._joint_value(joint, q, be), be))
        return T

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = [], []
        for name in self.joint_names:
            j = self.model.joint_by_name(name)
            lo.append(-math.pi if j.lower is None else j.lower)
            hi.append(math.pi if j.upper is None else j.upper)
        return np.asarray(lo, np.float64), np.asarray(hi, np.float64)

    def velocity_limits(self, default: Optional[float] = None) -> np.ndarray:
        """`(nq,)` of ``|dq|`` bounds in rad/s, in `joint_names` order.

        `default` fills joints whose URDF gives no `velocity`. Passing None raises instead, which is
        the right behaviour for a trajectory optimizer: a silently-defaulted velocity limit is a
        number nobody chose being enforced on the robot.
        """
        return self._limit_vector("velocity", default)

    def acceleration_limits(self, default: Optional[float] = None) -> np.ndarray:
        """`(nq,)` of ``|d2q|`` bounds in rad/s^2. Same defaulting rule as `velocity_limits`."""
        return self._limit_vector("acceleration", default)

    def _limit_vector(self, field: str, default: Optional[float]) -> np.ndarray:
        out: list[float] = []
        missing: list[str] = []
        for name in self.joint_names:
            value = getattr(self.model.joint_by_name(name), field)
            if value is None:
                missing.append(name)
                out.append(float("nan") if default is None else float(default))
            else:
                out.append(float(value))
        if missing and default is None:
            raise ValueError(
                f"the URDF gives no {field} limit for {missing}; pass a default explicitly rather "
                "than letting one be invented"
            )
        return np.asarray(out, np.float64)


RBY1_URDF = pathlib.Path("src/rby1_description/models/rby1a/urdf/model.urdf")


def load_rby1(
    urdf_path: str | pathlib.Path | None = None,
    *,
    joint_names: Sequence[str] = DEFAULT_RBY1_JOINTS,
    **kwargs: Any,
) -> UrdfSphereChain:  # noqa: D401 - kwargs forwards extra_capsules, link_filter, ...
    """Load the RB-Y1 collision model shipped in this workspace.

    The default path is workspace-relative, matching how every other entry point here is run
    (`src/openpi/.venv/bin/python -m benchmark....` from the repo root).
    """
    # 경로 해석은 `parse_urdf` 가 한다 — 기본값이 작업공간 상대경로라도, 자산이
    # `src/` 가 아니라 `pi05_TO_hybrid/` 에 있는 환경에서 찾아낸다 (`ag3s/runtime/asset_path.py`).
    return UrdfSphereChain(parse_urdf(urdf_path if urdf_path is not None else RBY1_URDF),
                           joint_names, **kwargs)


__all__ = [
    "DEFAULT_RBY1_JOINTS",
    "RBY1_URDF",
    "UrdfCapsule",
    "UrdfJoint",
    "UrdfModel",
    "UrdfSphereChain",
    "load_rby1",
    "parse_urdf",
]
