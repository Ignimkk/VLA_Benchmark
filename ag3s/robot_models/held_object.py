"""쥔 물체를 **로봇의 일부**로 — 구 맞춤 · 손바닥 frame · self-filter 구 집합 (T32 H1·H2, S1).

cuRobo 의 attach 방식이다: 파지가 확인되면 물체를 구 몇 개로 근사해 **로봇 구 집합에 더한다.**
그러면 (1) 로봇을 depth 에서 지우는 self-filter 가 그 물체의 픽셀도 지우고 (TSDF 에 새로 적분되지
않는다), (2) 최적화기는 그 구를 환경에 대한 질의 구로 쓴다. 이 모듈은 그 재료를 만든다.

| 무엇 | 함수 |
|---|---|
| 관측 표면 → 구 (중심·반지름) | `fit_surface_sphere` — 한쪽만 보인 볼록 물체의 **중심**을 표면에서 추정 |
| 그 구 + 돌출부 → 덮는 구 몇 개 | `covering_spheres` — 주 구 1 개 + 주 구 밖 관측 점 덩어리마다 작은 구 |
| 손가락 link → 손바닥 link | `palm_link_for` — 모델의 구조(같은 parent 의 prismatic 두 개)에서 |
| self-filter 모델 + 쥔 구 | `HeldSphereFilterModel` — `sphere_centers_numeric` 에 쥔 구를 덧붙인 얇은 wrapper |

**왜 구 맞춤인가 (centroid 가 아니라).** 카메라는 과일의 한쪽만 본다. 관측 점의 centroid 는 보이는
면 쪽으로 치우치고(반지름 34 mm 사과의 반구 cap 이면 중심에서 ~17 mm), centroid + 최대거리로 만든
공은 뒷면을 놓친다 — T31b 재생에서 사과 뒷면이 공 밖으로 중앙 4.4 mm 나갔다. 표면에 구를 맞추면 cap
만으로도 중심이 나온다. 반지름은 **위에서 막는다** (`max_radius` = 손이 쥘 수 있는 최대 개도의 반,
T26): 평평한 조각에 맞추면 반지름이 무한히 커지는데, 쥘 수 있는 물체는 그보다 클 수 없다.

**왜 손바닥인가 (손가락이 아니라).** T22 의 attach 는 `ee_finger_l1` 에 붙였다. 손가락 link 는 개도에
따라 움직이므로(T29 이후 finger joint 는 실제 측정값) 물체가 손가락 하나와 함께 움직인다. 손바닥
(`ee_left`/`ee_right`)은 finger joint 의 부모라 개도와 무관하다 — 쥔 물체가 강체로 붙는 곳이다.

**구는 물체를 덮어야 한다** (self-filter inflation 0 규칙). 반지름 = 맞춘 반지름 + `pad`. pad 는
T21 guard · `_target_ball` 과 같은 한 fine voxel(5 mm)이 기본이고, 근거는 `T32a.impl.md` §2.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

import numpy as np

#: 구 맞춤을 받아들이는 최소 점 수. 이보다 적으면 centroid 규칙으로 물러난다.
MIN_FIT_POINTS = 20
#: 받아들이는 최대 잔차 (m, 잔차 절댓값의 중앙값). 표면이 구가 아니면(병합 · 바나나) 물러난다.
MAX_FIT_RESIDUAL = 0.004
#: 맞춘 반지름의 하한 (m). 이보다 작은 "구" 는 잡음이다.
MIN_FIT_RADIUS = 0.010
#: 주 구 밖 관측 점을 덮는 작은 구 — 연결 반경 (m). 사과 꼭지·잎 (T32a §2).
COVER_EPS = 0.010
#: 덮개 구 하나가 되려면 최소 이만큼의 점. 그보다 작은 덩어리도 버리지 않는다 — 주 구를 키운다.
MIN_COVER_POINTS = 1


@dataclasses.dataclass(frozen=True)
class SphereFit:
    """관측 표면에 맞춘 구. `method` 가 `"surface_fit"` 이 아니면 `centre` 는 centroid 다."""

    centre: np.ndarray
    radius: float
    #: `"surface_fit"` (받아들임) · `"centroid"` (물러남 — `reason` 에 이유)
    method: str
    n_points: int
    #: 잔차 `| |p − c| − r |` 의 중앙값 (m). 물러났으면 `None`.
    residual_median: Optional[float] = None
    #: 반지름 상한이 걸렸는가 (평평한 조각이면 걸린다).
    radius_capped: bool = False
    #: 관측 점 centroid 에서 맞춘 중심까지 (m) — 뒷면으로 얼마나 옮겼나.
    shift_from_centroid: float = 0.0
    reason: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.method == "surface_fit"

    def record(self) -> dict:
        return {
            "method": self.method,
            "centre_m": [round(float(v), 6) for v in self.centre],
            "radius_mm": round(float(self.radius) * 1000.0, 3),
            "n_points": int(self.n_points),
            "residual_median_mm": (None if self.residual_median is None
                                   else round(float(self.residual_median) * 1000.0, 3)),
            "radius_capped": bool(self.radius_capped),
            "shift_from_centroid_mm": round(float(self.shift_from_centroid) * 1000.0, 3),
            "reason": self.reason,
        }


def _algebraic_sphere(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Kåsa 대수 맞춤: `|p|² = 2 c·p + (r² − |c|²)` 의 선형 최소제곱."""
    A = np.column_stack([2.0 * pts, np.ones(len(pts))])
    b = np.einsum("ij,ij->i", pts, pts)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r2 = float(sol[3] + c @ c)
    return c, float(np.sqrt(r2)) if r2 > 0.0 else float("nan")


def fit_surface_sphere(points: np.ndarray, *, max_radius: Optional[float] = None,
                       min_radius: float = MIN_FIT_RADIUS, min_points: int = MIN_FIT_POINTS,
                       max_residual: float = MAX_FIT_RESIDUAL,
                       loss_scale: float = 0.002) -> SphereFit:
    """관측 표면 점에 구를 맞춘다 — 볼록 물체의 **중심**을 한쪽 표면에서 추정한다.

    1. Kåsa 대수 맞춤으로 시작점을 얻고,
    2. 기하 잔차 `|p − c| − r` 를 robust loss (`soft_l1`, 척도 `loss_scale`) 로 다듬는다.
       반지름은 `[min_radius, max_radius]` 로 막는다 — `max_radius` 는 쥘 수 있는 물체의 상한
       (최대 개도의 반)이다. 평평한 조각은 반지름이 무한으로 가려 하므로 상한이 걸린다.
    3. 받아들이는 조건: 점 `min_points` 이상 · 유한 · 잔차 중앙값 ≤ `max_residual` · 맞춘 중심이
       centroid 에서 반지름 안 (cap 의 centroid 는 언제나 구 안이다).

    어느 하나라도 어기면 **centroid 규칙으로 물러난다** (`method="centroid"`, radius = 최대거리) —
    T32 전 동작 그대로라 이 함수가 결과를 나쁘게 만드는 길은 없다. 이유는 `reason` 에 남는다.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    n = int(len(pts))
    if n == 0:
        raise ValueError("fit_surface_sphere: 점이 없다")
    centroid = pts.mean(axis=0)
    spread = float(np.linalg.norm(pts - centroid, axis=1).max()) if n > 1 else 0.0

    def fallback(reason: str) -> SphereFit:
        return SphereFit(centre=centroid, radius=spread, method="centroid", n_points=n,
                         reason=reason)

    if n < int(min_points):
        return fallback(f"too_few_points ({n} < {int(min_points)})")
    c0, r0 = _algebraic_sphere(pts)
    if not (np.all(np.isfinite(c0)) and np.isfinite(r0)):
        return fallback("algebraic_fit_degenerate")
    hi = float("inf") if max_radius is None else float(max_radius)
    lo = float(min_radius)
    if hi <= lo:
        return fallback(f"radius bounds empty ({lo * 1000:.1f} ≥ {hi * 1000:.1f} mm)")
    r_init = float(np.clip(r0, lo, hi))
    if np.isfinite(hi):
        r_init = min(r_init, hi - 1e-6 * max(hi, 1e-3))   # strictly inside the bounds
    r_init = max(r_init, lo + 1e-9)
    if r0 > hi:
        # 평평한 조각: 대수 해의 중심은 멀리 있다. 상한 반지름으로 **같은 쪽**에 다시 놓는다 —
        # centroid 에서 대수 중심 방향으로 r 만큼 (cap 의 중심은 그 방향에 있다).
        direction = c0 - centroid
        norm = float(np.linalg.norm(direction))
        c_init = centroid + (direction / norm) * r_init if norm > 1e-9 else centroid
    else:
        c_init = c0
    from scipy.optimize import least_squares

    def residual(x):
        return np.linalg.norm(pts - x[:3], axis=1) - x[3]

    try:
        sol = least_squares(residual, np.concatenate([c_init, [r_init]]),
                            bounds=([-np.inf] * 3 + [lo], [np.inf] * 3 + [hi if np.isfinite(hi) else np.inf]),
                            loss="soft_l1", f_scale=float(loss_scale), max_nfev=200)
        x = sol.x
    except Exception as exc:  # noqa: BLE001 — 맞춤 실패 = 물러남 (이유를 남긴다)
        return fallback(f"least_squares_failed: {type(exc).__name__}")
    centre, radius = x[:3], float(x[3])
    if not (np.all(np.isfinite(centre)) and np.isfinite(radius)):
        return fallback("fit_not_finite")
    res = np.abs(residual(x))
    med = float(np.median(res))
    shift = float(np.linalg.norm(centre - centroid))
    capped = bool(np.isfinite(hi) and radius >= hi - 1e-6)
    if med > float(max_residual):
        return fallback(f"residual {med * 1000:.1f} mm > {max_residual * 1000:.1f} mm "
                        "(the surface is not a sphere)")
    if shift > radius:
        return fallback(f"centre {shift * 1000:.1f} mm from the centroid, beyond its radius")
    return SphereFit(centre=np.asarray(centre, np.float64), radius=radius, method="surface_fit",
                     n_points=n, residual_median=med, radius_capped=capped,
                     shift_from_centroid=shift)


def enclosing_ball(points: np.ndarray, fit: SphereFit, *, pad: float) -> tuple[np.ndarray, float]:
    """`(centre, radius)` — 맞춘 구 **와** 관측 점을 모두 담는 공 (S1 `_target_ball`).

    중심 = 맞춘 중심, 반지름 = `max(맞춘 반지름, 관측 점까지 최대거리) + pad`. 관측 점을 담는 것은
    centroid 규칙과 같고, 맞춘 반지름이 뒷면(안 보인 쪽)을 담는다.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    far = float(np.linalg.norm(pts - fit.centre, axis=1).max()) if len(pts) else 0.0
    return fit.centre.copy(), max(float(fit.radius), far) + float(pad)


def covering_spheres(points: np.ndarray, fit: SphereFit, *, pad: float, max_spheres: int = 4,
                     cover_eps: float = COVER_EPS) -> tuple[list[tuple[np.ndarray, float]], dict]:
    """쥔 물체를 덮는 구 몇 개 (H2) — `([(centre, radius), ...], record)`.

    * **주 구**: 맞춘 중심, 반지름 = 맞춘 반지름 + `pad`. 물체의 몸통(보이는 면 + 뒷면).
    * **덮개 구**: 주 구 **밖**의 관측 점(사과 꼭지·잎처럼 몸통에서 튀어나온 것)을 `cover_eps` 로
      연결한 덩어리마다 하나 — 덩어리 centroid, 반지름 = 최대거리 + `pad`.
    * 덮개가 `max_spheres − 1` 개를 넘으면(= 표면이 구가 아니다) 한 개로 합친다: 주 구를 모든
      관측 점을 담도록 키운다. 덮는 것이 먼저다 — 개수는 제약 슬롯(`max_attached_primitives`)이 정한다.
    * 맞춤이 물러났으면(`fit.method == "centroid"`) 주 구 = centroid + 최대거리 + pad 하나.

    **모든 관측 점이 어떤 구 안에 든다** — 구성으로 보장된다 (테스트가 고정).
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    pad = float(pad)
    rec: dict[str, Any] = {"fit": fit.record(), "pad_mm": round(pad * 1000.0, 3),
                           "n_points": int(len(pts)), "n_cover": 0, "merged": False}
    if not fit.accepted or not len(pts):
        far = float(np.linalg.norm(pts - fit.centre, axis=1).max()) if len(pts) else fit.radius
        spheres = [(fit.centre.copy(), max(far, float(fit.radius)) + pad)]
        rec["mode"] = "single_centroid"
        return spheres, rec
    primary_r = float(fit.radius) + pad
    dist = np.linalg.norm(pts - fit.centre, axis=1)
    outside = pts[dist > primary_r]
    spheres = [(fit.centre.copy(), primary_r)]
    rec["mode"] = "fit"
    rec["n_outside_primary"] = int(len(outside))
    if len(outside):
        groups = _clusters(outside, float(cover_eps))
        if len(groups) <= max(0, int(max_spheres) - 1):
            for g in groups:
                p = outside[g]
                c = p.mean(axis=0)
                r = (float(np.linalg.norm(p - c, axis=1).max()) if len(p) > 1 else 0.0) + pad
                spheres.append((c, r))
            rec["n_cover"] = len(groups)
        else:
            spheres = [(fit.centre.copy(), float(dist.max()) + pad)]
            rec["merged"] = True
            rec["mode"] = "fit_single_enclosing"
    rec["spheres"] = [{"centre_m": [round(float(v), 6) for v in c],
                       "radius_mm": round(float(r) * 1000.0, 3)} for c, r in spheres]
    return spheres, rec


def _clusters(points: np.ndarray, eps: float) -> list[np.ndarray]:
    """`eps` 연결 성분 (index 배열 목록)."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(points)
    if n == 1:
        return [np.zeros(1, np.int64)]
    pairs = cKDTree(points).query_pairs(float(eps), output_type="ndarray")
    import scipy.sparse as sp

    g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) \
        if len(pairs) else sp.coo_matrix((n, n))
    k, labels = connected_components(g, directed=False)
    return [np.flatnonzero(labels == i) for i in range(k)]


def palm_link_for(robot_model: Any, grasp_link: str) -> Optional[str]:
    """손가락 link → 그 손가락 관절의 부모 link (손바닥). 모르면 `None`.

    구조로 찾는다 (`UrdfSphereChain.gripper_joint_pairs`: 같은 parent 의 prismatic 두 개). 이름을
    박지 않는 이유는 T29 와 같다 — 그리퍼가 바뀌어도 따라온다. 그 표면이 없는 모델(mock)은 `None`.
    이미 손바닥이면 그 자신을 돌려준다.
    """
    pairs_fn = getattr(robot_model, "gripper_joint_pairs", None)
    if not callable(pairs_fn):
        return None
    try:
        pairs = pairs_fn()
    except Exception:  # noqa: BLE001 — 구조를 못 읽으면 모른다
        return None
    for pair in pairs:
        parent = str(pair.get("parent_link"))
        if grasp_link == parent or grasp_link in tuple(str(c) for c in pair.get("children", ())):
            return parent
    return None


#: self-filter 에서 쥔 구에 **더** 붙는 미끄럼 여유 (m, T32a §2·§3). 질의 구(= 맞춘 반지름 + 한 fine voxel
#: 5 mm)는 그대로 두고 self-filter 구만 이만큼 크다 — 합쳐 맞춘 반지름 + 25 mm.
#: 근거: 기록된 E3 ep1807 5 run 에서 쥔 사과가 손바닥 frame 에서 운반 동안 **서서히 미끄러졌다** —
#: 최대 12.3 · 24.3 · 20.8 · 21.4 mm (손가락 frame 에서도 같은 값이라 frame 선택의 문제가 아니다,
#: `t32a/h2_drift.py`). 5 mm 로는 미끄러진 쪽 픽셀이 새어 TSDF 에 흔적이 다시 쌓였다 (H3: E3b ep1807 r1
#: seq 35 에서 흔적 복셀 5 mm 2,087 · 15 mm 969 · 25 mm 39). 질의 구에 붙이면 F19 (57 mm 구가 crate 여유를
#: 먹는다) 로 돌아가므로 둘을 가른다. self-filter 쪽에서 커서 드는 비용은 쥔 물체 둘레의 새 관측을
#: 버리는 것뿐이다 (가려진 광선 = 미관측, 자유가 아니다 — 이미 적분된 crate·table 은 남는다).
HELD_FILTER_SLIP_M = 0.020

#: `AG3S.attach` 가 맞춘 쥔 구의 `Primitive.semantic_role`. 이 역할의 구만 **질의 구**(linearize)와
#: **seed 제외 부피**가 된다 — T32 전 방식으로 지은 attachment(centroid + 최대거리 구, F19 의 57 mm)는
#: 예전처럼 점으로 질의된다.
HELD_ROLE = "held_object"


def held_spheres_of(attached: Any) -> Optional[list[tuple[np.ndarray, float]]]:
    """`[(centre_parent, radius)]` if `attached` carries T32 held spheres (every primitive has
    `HELD_ROLE`), else `None`."""
    prims = list(getattr(attached, "primitives", None) or ())
    if not prims or any(getattr(p, "semantic_role", None) != HELD_ROLE for p in prims):
        return None
    from benchmark.ag3s.constraints.attached import attached_spheres

    return attached_spheres(attached) or None


#: self-filter 모델의 쥔 구 이름 접두사. `sphere_link_names` 에 `held:<parent>` 로 실린다.
HELD_LINK_PREFIX = "held:"


class HeldSphereFilterModel:
    """self-filter 모델 + 쥔 물체 구 (H1). **self-filter 에만** 쓴다.

    `sphere_centers_numeric(q)` 는 로봇 구 뒤에 쥔 구를 덧붙인다 — 각 촬영 시점의 `q` 로 parent
    link 를 FK 하므로 카메라마다 그 순간의 자리다. 나머지 속성(`link_pose`, `nq`, finger
    parameter …)은 감싼 모델에 그대로 넘긴다 (손목 카메라의 `resolve_T_base_cam` 이 `link_pose`
    를 쓴다). `pose_model` 은 쥔 구의 parent link 를 FK 할 모델 — `AG3S.attach` 가 스냅샷을 뜬 모델과
    같아야 한다 (`constraint_robot_model`).
    """

    def __init__(self, base: Any, spheres_local: Sequence[tuple[np.ndarray, float]],
                 parent_link: str, *, pose_model: Any = None):
        self._base = base
        self._pose_model = pose_model if pose_model is not None else base
        self._local = np.asarray([np.asarray(c, np.float64).reshape(3) for c, _ in spheres_local],
                                 np.float64).reshape(-1, 3)
        self._radii = np.asarray([float(r) for _, r in spheres_local], np.float64)
        self.parent_link = str(parent_link)
        self.n_held = int(len(self._radii))

    def __getattr__(self, name):
        return getattr(self._base, name)

    @property
    def base_model(self):
        return self._base

    @property
    def n_spheres(self) -> int:
        n = getattr(self._base, "n_spheres", None)
        if n is None:
            n = len(np.asarray(getattr(self._base, "radii")))
        return int(n) + self.n_held

    @property
    def radii(self) -> np.ndarray:
        return np.concatenate([np.asarray(self._base.radii, np.float64).reshape(-1), self._radii])

    @property
    def sphere_link_names(self):
        names = getattr(self._base, "sphere_link_names", None)
        if names is None:
            return None
        return tuple(names) + (HELD_LINK_PREFIX + self.parent_link,) * self.n_held

    def held_spheres(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`(centres (K, 3), radii (K,))` — 쥔 구의 base 좌표 (자세 `q`)."""
        pose_fn = (getattr(self._pose_model, "link_pose", None)
                   or getattr(self._pose_model, "link_pose_numeric"))
        T = np.asarray(pose_fn(np.asarray(q, np.float64).reshape(-1), self.parent_link), np.float64)
        return self._local @ T[:3, :3].T + T[:3, 3], self._radii.copy()

    def sphere_centers_numeric(self, q: np.ndarray, *args, **kwargs):
        centres, radii = self._base.sphere_centers_numeric(q, *args, **kwargs)
        hc, hr = self.held_spheres(q)
        return (np.vstack([np.asarray(centres, np.float64).reshape(-1, 3), hc]),
                np.concatenate([np.asarray(radii, np.float64).reshape(-1), hr]))


def extend_inflation(inflation: Any, n_robot: int, n_held: int) -> Any:
    """self-filter inflation 을 쥔 구까지 늘린다 — **쥔 구는 언제나 0** (구 자체가 물체를 덮는다).

    스칼라 0 이면 그대로 0 (같은 산술), 아니면 `(n_robot + n_held,)` 배열.
    """
    if n_held <= 0:
        return inflation
    if not np.ndim(inflation) and float(inflation) == 0.0:
        return 0.0
    robot = np.broadcast_to(np.asarray(inflation, np.float64), (int(n_robot),))
    return np.concatenate([robot, np.zeros(int(n_held))])


def sphere_volume_points(spheres: Sequence[tuple[np.ndarray, float]], spacing: float,
                         z_min: Optional[float] = None) -> np.ndarray:
    """구 합집합 **안**의 격자점 (`spacing` 간격, base 좌표). `z_min` 이 있으면 그 아래는 뺀다.

    seed 제외(`CuroboFieldBuilder` 의 `attached_points`)에 넘기는 것 — 점 25 개(반지름 0)가 아니라
    쥔 구가 차지한 **부피 전체**의 복셀에서 seed 를 뺀다. `z_min`(지원면 + 한 voxel)은
    `TargetBall` 과 같은 규칙이다: 파지 순간 테이블에 닿은 구가 테이블 seed 를 지우지 않게.
    """
    out = []
    h = float(spacing)
    for c, r in spheres:
        c = np.asarray(c, np.float64).reshape(3)
        r = float(r)
        k = int(np.ceil(r / h))
        ax = np.arange(-k, k + 1) * h
        g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)
        g = g[np.einsum("ij,ij->i", g, g) <= r * r] + c
        out.append(g)
    if not out:
        return np.zeros((0, 3))
    pts = np.vstack(out)
    if z_min is not None:
        pts = pts[pts[:, 2] >= float(z_min)]
    return pts


# ---------------------------------------------------------------------------------------------
# T34 J1 — 쥔 질의 구는 받침면을 넘지 않는다
# ---------------------------------------------------------------------------------------------
#: 받침면 위 **거리장 표면 띠**(band)를 잴 수 없을 때 쓰는 값 (m) — cuRobo 필드가 있는데 측정이
#: 실패한 경우만. 근거: T33 attach 9 건의 기록 fine ESDF 에서 테이블 평면 위 높이 h 의 값이
#: `d = h − 8.6 … 9.0 mm` (중앙값, 9 건 모두; `T34.impl.md` §1.2). 평면 ±8.8 mm 가 d = 0 인 띠다
#: (fine 5 mm 복셀 · TSDF truncation 30 mm 의 표면 띠). 받침 위에 **놓인** 물체의 바닥은 그 띠
#: 안에 있으므로, 쥔 구가 그 테이블 행에서 음수를 읽지 않으려면 바닥이 띠 위에 있어야 한다.
HELD_SUPPORT_BAND_FALLBACK_M = 0.009
#: 띠 측정의 상한 (m). 이보다 크게 재지면 표면 띠가 아니라 다른 물체다 — 상한으로 자른다.
HELD_SUPPORT_BAND_MAX_M = 0.015


def support_plane_under(support_surfaces, point, *, min_normal_z: float = 0.9):
    """`point` 아래 가장 높은 **수평** 받침면 `(normal, offset, surface)`, 없으면 `None`.

    `AG3S._support_plane_z` 와 같은 규칙이다 (`|n_z| > 0.9`, 점 **아래**의 면만, 여럿이면 가장 높은
    것 — 테이블 위 물체의 받침은 바닥이 아니라 테이블이다). 법선은 위(+z)를 향하게 맞춰 돌려준다.
    """
    p = np.asarray(point, np.float64).reshape(3)
    best = None
    for surface in support_surfaces or ():
        n = np.asarray(getattr(surface, "normal", ()), np.float64).reshape(-1)
        if n.size != 3 or abs(float(n[2])) <= float(min_normal_z):
            continue
        d = float(getattr(surface, "offset", 0.0))
        if n[2] < 0.0:
            n, d = -n, -d
        z = (d - float(n[0]) * p[0] - float(n[1]) * p[1]) / float(n[2])
        if not np.isfinite(z) or z > p[2]:
            continue
        if best is None or z > best[3]:
            best = (n / float(np.linalg.norm(n)), d / float(np.linalg.norm(n)), surface, z)
    return None if best is None else best[:3]


def field_support_band(field, normal, offset, centre, *, rings=(0.06, 0.09, 0.12),
                       heights=(0.020, 0.030), n_angles: int = 24,
                       min_samples: int = 8) -> tuple[Optional[float], dict]:
    """거리장이 받침면을 **얼마나 두껍게** 보는가 — `(band, record)`, 못 재면 `(None, record)`.

    물체 둘레 고리(`rings`, 물체 중심에서 수평 거리) 위, 평면에서 높이 `h` 의 점들에서 `h − d(p)`
    를 잰다. 테이블만이 가장 가까운 표면인 점은 두 높이의 값 차가 높이 차와 같다 (eikonal, 수직
    기울기 1) — 그 점만 남겨 이웃 과일·crate 에 걸린 표본을 버린다. 남은 표본의 **90 분위수**가
    띠다 (자리마다 조금씩 다르다 — 쥔 구는 제 자리의 띠를 넘어야 한다).
    """
    n = np.asarray(normal, np.float64).reshape(3)
    c = np.asarray(centre, np.float64).reshape(3)
    # 평면 위 두 수평 축
    a = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(n, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    foot = c - (float(n @ c) - float(offset)) * n          # 중심의 평면 위 발
    ang = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    base = np.vstack([foot + r * (np.cos(t) * a + np.sin(t) * b) for r in rings for t in ang])
    h1, h2 = float(heights[0]), float(heights[1])
    rec: dict[str, Any] = {"rings_m": list(rings), "heights_m": [h1, h2], "n_samples": int(len(base))}
    try:
        d1 = np.asarray(field.distance(base + h1 * n), np.float64).reshape(-1)
        d2 = np.asarray(field.distance(base + h2 * n), np.float64).reshape(-1)
    except Exception as exc:  # noqa: BLE001 — 못 재면 못 잰 것이다 (호출자가 대체값을 쓴다)
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return None, rec
    ok = (np.isfinite(d1) & np.isfinite(d2) & (d1 > 0.0)
          & (np.abs((d2 - d1) - (h2 - h1)) <= 0.25 * (h2 - h1)))
    rec["n_table_only"] = int(ok.sum())
    if int(ok.sum()) < int(min_samples):
        rec["reason"] = f"too few table-only samples ({int(ok.sum())} < {int(min_samples)})"
        return None, rec
    q10, q50, q90 = (float(v) for v in np.percentile(h1 - d1[ok], [10, 50, 90]))
    rec["band_q10_q50_q90_mm"] = [round(v * 1000.0, 3) for v in (q10, q50, q90)]
    # q90: the band varies a little from spot to spot; the held sphere must clear it where it is.
    band = q90
    rec["band_raw_mm"] = round(band * 1000.0, 3)
    return float(np.clip(band, 0.0, HELD_SUPPORT_BAND_MAX_M)), rec


def lift_off_support(spheres: Sequence[tuple[np.ndarray, float]], normal, offset, *,
                     clearance: float, relative_poses: Sequence[np.ndarray] = (),
                     snapshot_clearance: Optional[float] = None,
                     ) -> tuple[list[tuple[np.ndarray, float]], dict]:
    """쥔 구마다 중심을 받침면 **법선 방향으로 올려** 바닥이 평면 + `clearance` 위에 있게 한다 (J1).

    `spheres` 는 스냅샷 자세의 base 좌표다. `relative_poses` 는 같은 구가 **다른 순간**(attach 요청
    자세)에 놓이는 자리로 옮기는 4×4 (`T_now · T_snap⁻¹`) — 그 순간에도 물체는 받침 위였으므로
    거기서도 넘지 않아야 한다. 올리는 방향은 스냅샷 자세의 법선 `n` 이고 (손바닥 frame 에 고정된
    벡터), 다른 자세에서는 `R n` 으로 돌아가므로 그 법선 성분으로 나눠 필요한 양을 구한다. 구마다
    필요한 양의 최댓값만큼 올린다 — 겹치지 않는 구(꼭지·잎 덮개)는 그대로다.

    반지름은 **줄이지 않는다** (`T34.impl.md` §1.1): 맞춘 반지름은 보이는 면 전체가 정한 크기이고
    (잔차 중앙 0.7–1.0 mm), 줄이면 관측 점이 구 밖으로 나간다. 겹침은 받침 쪽 한 방향의 오차다.
    """
    n = np.asarray(normal, np.float64).reshape(3)
    n = n / float(np.linalg.norm(n))
    off = float(offset)
    poses = [np.eye(4)] + [np.asarray(T, np.float64).reshape(4, 4) for T in relative_poses]
    snap_c = float(clearance) if snapshot_clearance is None else float(snapshot_clearance)
    clear = [snap_c] + [float(clearance)] * len(relative_poses)
    out: list[tuple[np.ndarray, float]] = []
    per: list[dict] = []
    for c, r in spheres:
        c = np.asarray(c, np.float64).reshape(3)
        r = float(r)
        need, bottoms = 0.0, []
        for T, cl in zip(poses, clear):
            cp = T[:3, :3] @ c + T[:3, 3]
            gap = float(n @ cp) - off - r                  # 바닥 − 평면 (음수 = 넘었다)
            bottoms.append(gap)
            gain = float(n @ (T[:3, :3] @ n))              # 올린 1 m 가 이 자세에서 법선으로 몇 m 인가
            if gain > 0.1:
                need = max(need, (cl - gap) / gain)
        lift = max(0.0, need)
        out.append((c + lift * n, r))
        per.append({"lift_mm": round(lift * 1000.0, 3),
                    "bottom_above_plane_mm_before": [round(g * 1000.0, 3) for g in bottoms],
                    "radius_mm": round(r * 1000.0, 3)})
    rec = {"method": "lift_centre_along_normal", "clearance_mm": round(float(clearance) * 1000.0, 3),
           "snapshot_clearance_mm": round(snap_c * 1000.0, 3),
           "n_poses": len(poses), "spheres": per,
           "max_lift_mm": round(max((p["lift_mm"] for p in per), default=0.0), 3)}
    return out, rec


# ---------------------------------------------------------------------------------------------
# T34 J2 — 쥔 구와 관측이 맞는가 (측정만; 판정은 `trajopt.grasp_latch` 가 한다)
# ---------------------------------------------------------------------------------------------
def held_observation_check(*, observed_centroid, held_centre, held_centre_at_attach,
                           object_centroid_at_attach, up=(0.0, 0.0, 1.0)) -> dict:
    """attach 뒤 한 프레임의 관측을 쥔 구와 비교한 **측정** (m) — J2 (i)·(iii) 의 재료.

    * `centre_distance_m` — 그 프레임에 관측된 조작 대상 centroid ↔ 쥔 주 구 중심 (맞춘 중심,
      J1 로 올리기 전 = self-filter 구의 중심, 그 프레임 촬영 자세의 손바닥으로 FK).
    * `hand_rise_m` — 그 중심이 attach 요청 자세 이후 `up` 방향으로 오른 양 (손이 든 양).
    * `object_rise_m` — 관측 centroid 가 attach 한 기하의 centroid 이후 `up` 방향으로 오른 양.

    쥔 물체라면 centroid 는 쥔 구 안(맞춘 반지름 + 미끄럼)에 있고, 손이 들리면 같이 오른다.
    """
    u = np.asarray(up, np.float64).reshape(3)
    u = u / float(np.linalg.norm(u))
    oc = np.asarray(observed_centroid, np.float64).reshape(3)
    hc = np.asarray(held_centre, np.float64).reshape(3)
    ha = np.asarray(held_centre_at_attach, np.float64).reshape(3)
    oa = np.asarray(object_centroid_at_attach, np.float64).reshape(3)
    return {"centre_distance_m": float(np.linalg.norm(oc - hc)),
            "hand_rise_m": float(u @ (hc - ha)),
            "object_rise_m": float(u @ (oc - oa))}


__all__ = [
    "HELD_LINK_PREFIX",
    "HELD_FILTER_SLIP_M",
    "HELD_ROLE",
    "HELD_SUPPORT_BAND_FALLBACK_M",
    "HELD_SUPPORT_BAND_MAX_M",
    "HeldSphereFilterModel",
    "SphereFit",
    "covering_spheres",
    "enclosing_ball",
    "extend_inflation",
    "field_support_band",
    "fit_surface_sphere",
    "held_observation_check",
    "held_spheres_of",
    "lift_off_support",
    "palm_link_for",
    "sphere_volume_points",
    "support_plane_under",
]
