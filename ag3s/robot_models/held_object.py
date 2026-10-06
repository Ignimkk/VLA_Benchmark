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


# ---------------------------------------------------------------------------------------------
# T43 R2 — 쥔 질의 구가 과일 **몸통 전체**를 덮는다 (바닥 반구 포함)
# ---------------------------------------------------------------------------------------------
#: 운반 중 쥔 과일이 손바닥 frame 에서 **중력 쪽으로** 미끄러지는 양의 허용치 (m, T43 R2).
#: 근거: T40/T41 사과-막대 접촉 5 run (고유) 에서 attach → 접촉까지 사과 중심이 손바닥 frame 에서
#: 아래로 2.1–4.5 mm 움직였다 (`T43R2.impl.md` §2). 덮개 pad 5 mm 위에 이만큼을 바닥 구로 더 덮는다.
HELD_BODY_SLIP_M = 0.005


#: 길쭉한 물체로 보는 수평 길이 비 (T43 R2). 관측 점 (+ 받침면 발자국) 의 수평 주축 길이가 그에 수직인
#: 폭의 이 배를 넘으면 구 맞춤 몸통이 아니라 주축 구간 구로 덮는다 (바나나). 사과 · 오렌지 · 배는 위에서
#: 보면 둥글다 (MuJoCo mesh 수평 66.6 × 66.6 · 61.5 × 61.5 · 60.3 × 58.7 mm).
HELD_ELONGATION_RATIO = 1.6


def body_cover_spheres(points: np.ndarray, fit: SphereFit, *, pad: float,
                       support: Optional[tuple[np.ndarray, float]] = None,
                       slip: float = HELD_BODY_SLIP_M, max_spheres: int = 4,
                       cover_eps: float = COVER_EPS, min_shift: float = 0.001,
                       max_radius: Optional[float] = None, radius_from_support: bool = True,
                       elongation_ratio: float = HELD_ELONGATION_RATIO,
                       ) -> tuple[list[tuple[np.ndarray, float]], dict]:
    """쥔 과일의 **몸통**을 덮는 구 몇 개 (T43 R2) — `([(centre, radius), ...], record)`, base 좌표.

    H2 `covering_spheres` 는 관측된 면(위 · 옆)과 맞춘 구를 덮는다. 카메라는 바닥을 못 보고, 맞춘 구가
    작게 나오면 (손에 가린 cap: T41 ep1807 반지름 29 mm, 사과 33 mm) 아래 반구가 밖으로 나간다.
    여기서는 **물체가 받침 위에 놓여 있었다**는 사실을 쓴다 — 촬영 순간 물체의 바닥은 받침면이다.

    **구형 몸통** (`mode="sphere"`, 맞춤이 받아들여졌고 길쭉하지 않다):

    * **주 구**: 맞춘 중심 `c`, 반지름 `R + pad`. 받침 위 중심 높이 `h` 가 `R` 보다 크되 그 차가 한 pad
      이하이면 (맞춘 구가 받침에 조금 못 미친다 — 가린 cap 으로 작게 나왔다) `R = min(h, max_radius)` 로
      올린다 (`radius_from_support`): 받침 위에 놓인 구형 몸통의 반지름은 중심 높이 이상이다. 더 크게
      못 미치면 몸통이 구가 아니다 (배) — 옆으로 키우지 않고 바닥 구가 받침까지 덮는다.
    * **바닥 구**: 같은 반지름, 중심을 받침 법선의 **아래로** `Δ = max(0, h − R) + slip`.
      `slip` 은 운반 중 중력 쪽 미끄럼 허용치. 같은 반지름 두 구의 허리는 `√(R² − Δ²/4)`.
      `h − R > R` 이면 그 면 위에 놓여 있지 않았던 것으로 보고 쓰지 않는다 (기록). 받침면을 모르면
      아래 = base −z, `Δ = slip`. `Δ < min_shift` 면 바닥 구를 만들지 않는다.
    * **덮개 구**: 두 구 밖의 관측 점 (꼭지 · 잎) 덩어리마다 하나 (`covering_spheres` 와 같은 규칙).
      슬롯을 넘으면 큰 구 하나로 합치지 않고 아래의 구간 덮개로 간다.

    **구간 덮개** (`mode="segments"`, 맞춤이 물러났거나 · 수평 주축이 폭의 `elongation_ratio` 배를
    넘거나 · 덮개가 슬롯을 넘을 때): 관측 점 + 그 점들의 받침면 발자국 (물체가 받침 위에 놓여 있었으므로
    바닥은 거기다; 가장 낮은 관측 점이 받침 위 `max_radius` 안일 때만) + `slip` 만큼 내린 사본을 수평
    주축을 따라 `k` (= 길이/폭 올림, 2 ≤ k ≤ `max_spheres`) 구간으로 나누고, 구간마다 그 점들을 담는
    구 (상자 중심 + 최대거리) + pad. 구는 볼록이라 구간 점들의 볼록 껍질을 담는다 — 위 면부터 받침까지.
    큰 구 하나 (H2 의 merge, 바나나에서 반지름 ~90 mm) 가 손가락 파지 여유를 삼키는 것을 피한다.

    구는 받침면 아래로 내려갈 수 있다 — 받침 가까이에서는 J1 lift 가 올린다
    (`AG3S._frame_held_attached`). 이 함수는 몸통만 말한다.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    pad = float(pad)
    slip = max(0.0, float(slip))
    c = np.asarray(fit.centre, np.float64).reshape(3)
    if fit.accepted or not len(pts):
        R = float(fit.radius)
    else:
        R = max(float(np.linalg.norm(pts - c, axis=1).max()), float(fit.radius))
    rec: dict[str, Any] = {"method": "body_cover", "fit": fit.record(), "pad_mm": round(pad * 1000.0, 3),
                           "slip_mm": round(slip * 1000.0, 3), "n_points": int(len(pts)),
                           "body_radius_mm": round(R * 1000.0, 3)}
    if support is not None:
        n = np.asarray(support[0], np.float64).reshape(3)
        n = n / float(np.linalg.norm(n))
        off = float(support[1])
    else:
        n, off = np.array([0.0, 0.0, 1.0]), None
    down = -n
    # --- 수평 주축 (관측 점 + 받침 발자국) -----------------------------------------------------
    resting = False
    cloud = pts
    if off is not None and len(pts):
        heights = pts @ n - off
        hi = float("inf") if max_radius is None else float(max_radius)
        resting = bool(heights.min() <= hi)
        if resting:
            cloud = np.vstack([pts, pts - np.clip(heights, 0.0, None)[:, None] * n[None, :]])
    rec["resting_on_support"] = resting
    a1 = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(a1) < 1e-6:
        a1 = np.cross(n, [0.0, 1.0, 0.0])
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(n, a1)
    if len(cloud) >= 2:
        hxy = np.column_stack([cloud @ a1, cloud @ a2])
        hxy = hxy - hxy.mean(axis=0)
        w, v = np.linalg.eigh(hxy.T @ hxy)
        major = v[0, 1] * a1 + v[1, 1] * a2          # 가장 큰 고윳값의 축 (eigh 는 오름차순)
        minor = np.cross(n, major)
        length = float(np.ptp(cloud @ major))
        width = float(np.ptp(cloud @ minor))
    else:
        major, minor, length, width = a1, a2, 0.0, 0.0
    rec["horizontal_extent_mm"] = [round(length * 1000.0, 3), round(width * 1000.0, 3)]
    # 한쪽에서 본 둥근 과일의 cap 은 초승달처럼 좁게 보인다 — 맞춘 구가 뒷면까지 말하므로 폭은 그 지름
    # 이상이다. 길쭉함은 그 폭에 대해 잰다.
    width_eff = max(width, 2.0 * R) if fit.accepted else width
    elongated = bool(width_eff > 0.0 and length > float(elongation_ratio) * width_eff)

    def segments(reason: str):
        body_pts = np.vstack([cloud, cloud + slip * down[None, :]]) if slip > 0.0 else cloud
        t = body_pts @ major
        k = int(np.clip(np.ceil(length / max(width, 1e-6)), 2, max(2, int(max_spheres))))
        k = min(k, max(1, int(max_spheres)))
        edges = np.linspace(t.min(), t.max(), k + 1)
        out = []
        for i in range(k):
            sel = (t >= edges[i]) & ((t <= edges[i + 1]) if i == k - 1 else (t < edges[i + 1]))
            p = body_pts[sel]
            if not len(p):
                continue
            lo, up = p.min(axis=0), p.max(axis=0)
            cc = 0.5 * (lo + up)
            out.append((cc, float(np.linalg.norm(p - cc, axis=1).max()) + pad))
        rec.update({"mode": "segments", "segments_reason": reason, "n_segments": len(out)})
        return out

    spheres = None
    if not fit.accepted or not len(pts):
        spheres = segments("fit_rejected" if len(pts) else "no_points")
    elif elongated:
        spheres = segments("elongated")
    else:
        rec["mode"] = "sphere"
        if off is not None:
            h = float(n @ c - off)                       # 맞춘 중심의 받침면 위 높이
            delta = h - R
            rec["support_gap_mm"] = round(delta * 1000.0, 3)
            if delta > R:
                rec["support_ignored"] = "fit sphere bottom more than one radius above the plane"
                delta = 0.0
            elif 0.0 < delta <= pad and radius_from_support:
                # 한 pad 안쪽으로 못 미친 맞춤 = 가린 cap 의 반지름 과소 (T41 ep1807: 3.1 mm) — 올린다.
                # 그보다 멀면 구가 아닌 몸통 (배: 위쪽 cap 에 맞춘 중심이 받침 위 43 mm, 반지름 27 mm)
                # 이라 옆으로 키우면 이웃 과일을 읽는다 — 바닥 구가 받침까지 덮는다.
                cap = float("inf") if max_radius is None else float(max_radius)
                R_new = min(h, max(R, cap))
                rec["radius_raised_mm"] = round((R_new - R) * 1000.0, 3)
                R = R_new
                rec["body_radius_mm"] = round(R * 1000.0, 3)
                delta = h - R
        else:
            delta = 0.0
            rec["support_gap_mm"] = None
        shift = max(0.0, delta) + slip
        rec["bottom_shift_mm"] = round(shift * 1000.0, 3)
        body = [(c.copy(), R + pad)]
        if shift >= float(min_shift):
            body.append((c + shift * down, R + pad))
        spheres = list(body)
        rec["n_body"] = len(body)
        rec["n_cover"] = 0
        gap = np.min(np.stack([np.linalg.norm(pts - bc, axis=1) - br for bc, br in body]), axis=0)
        outside = pts[gap > 0.0]
        rec["n_outside_body"] = int(len(outside))
        if len(outside):
            groups = _clusters(outside, float(cover_eps))
            if len(groups) <= max(0, int(max_spheres) - len(body)):
                for g in groups:
                    p = outside[g]
                    cc = p.mean(axis=0)
                    r = (float(np.linalg.norm(p - cc, axis=1).max()) if len(p) > 1 else 0.0) + pad
                    spheres.append((cc, r))
                rec["n_cover"] = len(groups)
            else:
                spheres = segments("covers_over_slots")
    rec["spheres"] = [{"centre_m": [round(float(v), 6) for v in cc],
                       "radius_mm": round(float(r) * 1000.0, 3)} for cc, r in spheres]
    return spheres, rec


# ---------------------------------------------------------------------------------------------
# T43 Z6 — 쥔 질의 구의 크기 상한 (`clustering.held_cover_cap`)
# ---------------------------------------------------------------------------------------------
#: 같은 구로 보는 거리 (m) — 중심 차 · 반지름 차가 둘 다 이 안이면 하나로 본다 (0.1 mm).
HELD_CAP_DEDUPE_TOL_M = 1e-4


def held_cap_reference(points: np.ndarray, fit: SphereFit, *, body_radius: Optional[float] = None,
                       elongation_ratio: float = HELD_ELONGATION_RATIO) -> dict[str, Any]:
    """상한의 기준 — 관측 몸통 중심과 관측 반폭 (T43 Z6). base 좌표, `points` = attach 의 관측 점.

    * 맞춤을 받아들였으면: 중심 = 맞춘 중심, 반폭 = `body_radius` (R2 가 받침으로 올린 몸통 반지름,
      없으면 맞춘 반지름). 카메라가 cap 만 본 둥근 과일의 반폭은 맞춘 반지름이다.
    * 물러났으면: 중심 = 관측 점 상자 중심, 반폭 = 수평 폭(주축에 수직)의 반.
    * 수평 주축이 폭의 `elongation_ratio` 배를 넘으면 (바나나, R2 의 `elongated` 와 같은 판정)
      `{"skip": "elongated"}` — 둥근 몸통의 상한을 길쭉한 물체에 걸면 양 끝이 덮이지 않는다.

    돌려주는 것: `{"centre": (3,), "half_width": m, "basis": ...}` 또는 `{"skip": 이유}`.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if not len(pts):
        return {"skip": "no_points"}
    length = width = 0.0
    if len(pts) >= 2:
        hxy = pts[:, :2] - pts[:, :2].mean(axis=0)
        _, v = np.linalg.eigh(hxy.T @ hxy)
        major, minor = v[:, 1], v[:, 0]
        length = float(np.ptp(hxy @ major))
        width = float(np.ptp(hxy @ minor))
    if fit.accepted:
        w = float(fit.radius if body_radius is None else body_radius)
        centre = np.asarray(fit.centre, np.float64).reshape(3).copy()
        basis = "fit"
        width_eff = max(width, 2.0 * w)
    else:
        w = 0.5 * width
        centre = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        basis = "observed_box"
        width_eff = width
    if width_eff > 0.0 and length > float(elongation_ratio) * width_eff:
        return {"skip": "elongated", "horizontal_extent_mm": [round(length * 1000.0, 3),
                                                             round(width * 1000.0, 3)]}
    if not (w > 0.0 and np.all(np.isfinite(centre))):
        return {"skip": "no_half_width"}
    return {"centre": centre, "half_width": w, "basis": basis}


def dedupe_spheres(spheres: Sequence[tuple[np.ndarray, float]], *,
                   tol: float = HELD_CAP_DEDUPE_TOL_M) -> tuple[list[tuple[np.ndarray, float]], int]:
    """같은 구 (중심 · 반지름이 `tol` 안) 와 다른 구 안에 통째로 든 구를 뺀다 — 합집합은 그대로다.

    순서는 남는 것끼리 그대로 (첫 구 = 주 구). 같은 구가 여럿이면 앞의 것이 남는다.
    `(남은 구, 뺀 수)`.
    """
    items = [(np.asarray(c, np.float64).reshape(3), float(r)) for c, r in spheres]
    n = len(items)
    drop = [False] * n
    for i in range(n):
        ci, ri = items[i]
        for j in range(n):
            if i == j or drop[j]:
                continue
            cj, rj = items[j]
            gap = float(np.linalg.norm(ci - cj))
            inside_j = gap + ri <= rj + tol            # i 가 j 안에 든다
            same = gap <= tol and abs(ri - rj) <= tol
            if inside_j and (not same or j < i):
                drop[i] = True
                break
    kept = [items[i] for i in range(n) if not drop[i]]
    return kept, n - len(kept)


def cap_held_spheres(spheres: Sequence[tuple[np.ndarray, float]], centre, half_width: float,
                     allowance: float, *, tol: float = HELD_CAP_DEDUPE_TOL_M,
                     ) -> tuple[list[tuple[np.ndarray, float]], dict]:
    """쥔 질의 구에 크기 상한을 건다 (T43 Z6, 사용자 판정 (a) 2026-10-06).

    1. 반지름 ≤ `half_width + allowance` (관측 반폭 + slip 여유).
    2. 중심이 관측 몸통 중심 `centre` 에서 `allowance` 밖이면 그 방향으로 `allowance` 까지 당긴다.
    3. 같은 구 · 다른 구 안에 든 구를 뺀다 (`dedupe_spheres`).

    그래서 모든 구는 `centre` 중심, 반지름 `half_width + 2·allowance` 의 공 안에 든다.
    T43 W V5 'segments' 4 run 은 반지름 56.7–59.1 mm, 중심이 사과 중심에서 최대 ~45 mm 였다
    (`T43Z.verify.json` numbers.Z1). `(구, 기록)`.
    """
    c0 = np.asarray(centre, np.float64).reshape(3)
    r_max = float(half_width) + float(allowance)
    d_max = float(allowance)
    capped: list[tuple[np.ndarray, float]] = []
    per = []
    for c, r in spheres:
        c = np.asarray(c, np.float64).reshape(3)
        r = float(r)
        v = c - c0
        dist = float(np.linalg.norm(v))
        moved = dist > d_max
        c_new = c0 + v * (d_max / dist) if moved else c.copy()
        r_new = min(r, r_max)
        capped.append((c_new, r_new))
        per.append({"radius_mm_before": round(r * 1000.0, 3), "radius_mm": round(r_new * 1000.0, 3),
                    "centre_offset_mm_before": round(dist * 1000.0, 3),
                    "centre_offset_mm": round(min(dist, d_max) * 1000.0, 3)})
    kept, n_removed = dedupe_spheres(capped, tol=tol)
    rec = {"half_width_mm": round(float(half_width) * 1000.0, 3),
           "allowance_mm": round(float(allowance) * 1000.0, 3),
           "radius_cap_mm": round(r_max * 1000.0, 3), "centre_limit_mm": round(d_max * 1000.0, 3),
           "centre_m": [round(float(v), 6) for v in c0],
           "n_before": len(per), "n_after": len(kept), "n_removed": int(n_removed),
           "n_radius_capped": sum(1 for p in per if p["radius_mm"] < p["radius_mm_before"]),
           "n_centre_moved": sum(1 for p in per if p["centre_offset_mm"] < p["centre_offset_mm_before"]),
           "spheres": per}
    return kept, rec


#: J1 lift 를 켜는 높이 여유 (m, T43 R2). 몸통 구의 바닥이 받침면 + 띠 + 이 값 안이면 "받침 가까이"
#: — 그때만 J1 lift 를 붙인다. 두 fine voxel: 띠에서 멈춘 내려놓기가 다음 프레임에 lift 로 넘어가고,
#: 면 · 맞춤의 voxel 단위 잡음으로 프레임마다 뒤집히지 않을 만큼.
HELD_LIFT_HYSTERESIS_M = 0.010


def support_gap(centres: np.ndarray, radii: np.ndarray, normal, offset) -> np.ndarray:
    """구마다 바닥 − 받침면 (m, 법선 방향; 음수 = 면 아래로 내려갔다)."""
    n = np.asarray(normal, np.float64).reshape(3)
    n = n / float(np.linalg.norm(n))
    c = np.asarray(centres, np.float64).reshape(-1, 3)
    return c @ n - float(offset) - np.asarray(radii, np.float64).reshape(-1)


def near_support(centres: np.ndarray, radii: np.ndarray, normal, offset, *, band: float,
                 hysteresis: float = HELD_LIFT_HYSTERESIS_M) -> tuple[bool, float]:
    """`(near, min_gap)` — 쥔 몸통이 받침면 가까이 있는가 (T43 R2 의 J1 조건).

    `near` ⇔ 몸통 구 바닥의 최솟값 ≤ `band + hysteresis`. `band` 는 거리장이 받침면을 d = 0 으로
    보는 두께 (J1 이 attach 때 잰 값) 다: 그 안에서는 lift 없는 구가 테이블 행에서 음수를 읽는다 —
    물체가 받침 위에 **놓여 있어서**이지 충돌이 아니다. 그 위에서는 몸통이 받침에서 떨어져 있으므로
    lift 를 하지 않는다 (막대처럼 받침 위 장애물이 바닥 반구로 들어오는 것을 놓치지 않게).
    """
    g = support_gap(centres, radii, normal, offset)
    m = float(g.min()) if g.size else float("inf")
    return bool(m <= float(band) + float(hysteresis)), m


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


def points_near(points: np.ndarray, distance: float, spacing: float) -> np.ndarray:
    """T43 T fix 2 — lattice points (`spacing`, base frame, a lattice anchored at the origin so
    overlapping neighbourhoods share points) within `distance` of any of `points`. `(0, 3)` when
    there are no points."""
    from scipy.spatial import cKDTree

    p = np.asarray(points, np.float64).reshape(-1, 3)
    if not len(p) or distance <= 0.0:
        return np.zeros((0, 3))
    h = float(spacing)
    lo = np.floor((p.min(axis=0) - distance) / h).astype(int)
    hi = np.ceil((p.max(axis=0) + distance) / h).astype(int)
    axes = [np.arange(a, b + 1) * h for a, b in zip(lo, hi)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    d, _ = cKDTree(p).query(g, distance_upper_bound=float(distance) + 1e-12)
    return g[np.isfinite(d)]


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
    "HELD_BODY_SLIP_M",
    "HELD_CAP_DEDUPE_TOL_M",
    "HELD_ELONGATION_RATIO",
    "HELD_LIFT_HYSTERESIS_M",
    "HELD_LINK_PREFIX",
    "HELD_FILTER_SLIP_M",
    "HELD_ROLE",
    "HELD_SUPPORT_BAND_FALLBACK_M",
    "HELD_SUPPORT_BAND_MAX_M",
    "HeldSphereFilterModel",
    "SphereFit",
    "body_cover_spheres",
    "cap_held_spheres",
    "covering_spheres",
    "dedupe_spheres",
    "enclosing_ball",
    "extend_inflation",
    "field_support_band",
    "fit_surface_sphere",
    "held_cap_reference",
    "held_observation_check",
    "held_spheres_of",
    "lift_off_support",
    "near_support",
    "palm_link_for",
    "sphere_volume_points",
    "support_gap",
    "support_plane_under",
]
