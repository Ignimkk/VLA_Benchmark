"""TSDF -> ESDF 충돌 표현 — primitive 근사를 대체하는 두 번째 backend.

primitive backend 는 후보 하나를 도형 하나로 줄인다. 조밀하고 볼록한 물체에는 맞지만 두 가지로
깨진다: 얇고 넓은 것(테이블 면)은 경계 구가 `(L/t)^2` 배로 커지고, 속이 빈 것(크레이트)은 구가
로봇이 **들어가야 하는** 공간을 채운다. 측정값은 `docs/OPEN-geometry-representation.md` 에 있다.

ESDF 는 그 축약을 하지 않는다. 관측된 표면을 복셀 격자에 그대로 적분하고, 어느 점에서든 가장
가까운 표면까지의 거리를 준다. 도형을 고르지 않으므로 도형이 틀릴 일이 없다.

    d_esdf(p) - collision_radius - safety_margin >= 0

이 파일이 책임지는 세 가지.

**투영 TSDF.** 복셀 중심을 각 카메라에 투영해 관측 깊이와의 차이를 부호 있는 거리로 쓴다.
절단(truncation)은 표면 근처에서만 값을 유지하기 위한 것이고, 그 밖은 "멀다"가 아니라
**"모른다"** 로 남는다 — 이 구분이 아래 미관측 정책의 전부다.

**세 상태 점유.** 점유 / 자유 / **미관측**. 미관측을 자유로 두면 로봇이 못 본 공간을 지나가고,
점유로 두면 테이블 뒤가 전부 막혀 아무것도 못 한다. AG3S 의 방식대로 **조용히 정하지 않고
보고한다**: `unknown_policy` 로 고르게 하되 기본값은 `free` 이고, 미관측 비율을 `EsdfField.stats`
에 실어 `ConstraintValidity` 가 읽을 수 있게 한다.

기본값이 `free` 인 것은 primitive backend 와 **같은 성질**이기 때문이다 — 관측되지 않은 기하는
후보를 만들지 않으므로 그쪽도 암묵적으로 자유다. ESDF 가 이것을 나쁘게 만들지 않고, 처음으로
**셀 수 있게** 만든다.

**국소 갱신.** 격자는 한 번 할당하고 프레임마다 같은 버퍼에 적분한다. ESDF 재계산은 TSDF 가
바뀐 복셀의 AABB 를 `max_distance` 만큼 부풀린 부분격자에서만 한다. 그 여유가 정확히
`max_distance` 이므로, 부분격자 안에서 `max_distance` 이하의 거리는 전역 재계산과 **같은 값**이
나온다. 그보다 먼 거리는 어차피 `max_distance` 로 잘린다.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

import numpy as np

_EPS = 1e-12

#: 세 상태 점유. 부호 있는 거리로 표현할 수 없는 정보라 따로 둔다.
FREE = np.int8(0)
OCCUPIED = np.int8(1)
UNKNOWN = np.int8(2)


@dataclasses.dataclass(frozen=True)
class VoxelGrid:
    """축 정렬 복셀 격자. base 프레임."""

    origin: np.ndarray  # (3,) 격자 (0,0,0) 복셀 **중심**의 좌표
    shape: tuple[int, int, int]
    voxel_size: float

    @staticmethod
    def from_bounds(lower, upper, voxel_size: float) -> "VoxelGrid":
        lower = np.asarray(lower, np.float64).reshape(3)
        upper = np.asarray(upper, np.float64).reshape(3)
        if np.any(upper <= lower):
            raise ValueError(f"격자 상한이 하한보다 커야 합니다: {lower} -> {upper}")
        if voxel_size <= 0:
            raise ValueError(f"voxel_size 는 양수여야 합니다: {voxel_size}")
        n = np.maximum(np.ceil((upper - lower) / voxel_size).astype(np.int64), 1)
        return VoxelGrid(origin=lower + 0.5 * voxel_size, shape=tuple(int(v) for v in n),
                         voxel_size=float(voxel_size))

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    @property
    def upper(self) -> np.ndarray:
        return self.origin + (np.asarray(self.shape) - 0.5) * self.voxel_size

    def centres(self) -> np.ndarray:
        """`(nx, ny, nz, 3)` 복셀 중심. 큰 격자에서는 메모리를 많이 쓰므로 필요할 때만."""
        ax = [self.origin[i] + np.arange(self.shape[i]) * self.voxel_size for i in range(3)]
        return np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)

    def to_index(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """점 -> `(정수 인덱스 (N,3), 격자 안인지 (N,))`. 반올림이 아니라 최근접 복셀."""
        p = np.asarray(points, np.float64).reshape(-1, 3)
        idx = np.rint((p - self.origin) / self.voxel_size).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < np.asarray(self.shape)), axis=1)
        return idx, inside


class TsdfVolume:
    """투영 TSDF. 카메라마다 한 번씩 `integrate` 하고 `occupancy()` 로 세 상태를 얻는다."""

    def __init__(self, grid: VoxelGrid, *, truncation: float):
        self.grid = grid
        self.truncation = float(truncation)
        if self.truncation <= 0:
            raise ValueError("truncation 은 양수여야 합니다")
        shape = grid.shape
        self.tsdf = np.full(shape, self.truncation, np.float32)
        self.weight = np.zeros(shape, np.float32)

    def _frustum_index_bounds(self, K: np.ndarray, T: np.ndarray, w: int, h: int,
                              depth_min: float, depth_max: float) -> tuple[np.ndarray, np.ndarray]:
        """카메라 절두체(근/원 평면 8 코너)의 base 프레임 AABB를 격자 인덱스 범위로 돌려준다.

        절두체는 볼록체이고 근/원 평면 각 4 코너가 그 꼭짓점 전부이므로, 8 코너의 AABB가 절두체
        전체의 정확한 AABB다 — 코너 사이 어딘가가 더 튀어나오는 경우는 없다. `pad`는 절단대역
        만큼 여유를 둔다: 표면이 원평면 바로 밖에 있어도 `sdf >= -truncation` 판정으로 갱신
        대상이 될 수 있는 복셀을 놓치지 않기 위해서다 (`_dirty_blocks`의 padding과 같은 이유).
        """
        Kinv = np.linalg.inv(K)
        corners_uv = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64)
        rays = corners_uv @ Kinv.T  # (4, 3) z=1 평면에서의 방향
        R, t = T[:3, :3], T[:3, 3]
        pts = np.concatenate(
            [(rays * d) @ R.T + t for d in (depth_min, depth_max)], axis=0
        )  # (8, 3) base 프레임
        idx, _ = self.grid.to_index(pts)
        pad = int(np.ceil(self.truncation / self.grid.voxel_size)) + 1
        shape = np.asarray(self.grid.shape)
        lo = np.clip(idx.min(axis=0) - pad, 0, shape - 1)
        hi = np.clip(idx.max(axis=0) + pad, 0, shape - 1)
        return lo, hi

    def integrate(self, depth: np.ndarray, camera_intrinsics: np.ndarray,
                  T_base_cam: np.ndarray, *, depth_min: float = 0.05,
                  depth_max: float = 3.0, max_weight: float = 64.0,
                  robot_mask: Optional[np.ndarray] = None) -> dict[str, int]:
        """한 카메라의 깊이를 적분한다.

        복셀 중심을 카메라로 투영해 그 픽셀의 관측 깊이와 비교한다. `sdf = d_meas - d_voxel`
        이므로 표면 앞은 양수, 뒤는 음수다. 절단 밖(뒤로 멀리)은 갱신하지 않는다 — 표면 뒤는
        "비어 있다"가 아니라 **가려져서 모른다**이고, 그것을 자유로 적분하면 없는 정보를
        만들어내는 셈이다.

        **전체 격자가 아니라 이 카메라의 절두체가 닿는 부분격자만 투영한다.** 이전에는 매
        카메라·매 프레임 전체 복셀(RB-Y1 기준 435만)을 투영해 그게 실측 1213 ms/frame의 주범이
        었다 — 국소 EDT 갱신과 달리 이 단계는 씬이 정적이어도 매 프레임 고정 비용이었다
        (`docs/AG3S_REVIEW_LOG.md` Step 1, E6). cuRoboV2(Sundaralingam et al. 2026)의
        block-discovery 단계와 같은 발상을 격자 전체가 아니라 절두체 AABB 단위로 적용한 것 —
        해시 테이블 없이 numpy 슬라이싱만으로 같은 효과를 낸다.
        """
        depth = np.asarray(depth, np.float64)
        if robot_mask is not None:
            # 로봇 픽셀은 깊이를 무효로 만든다. "자유"가 아니라 **미관측**이 되는 것이 옳다 —
            # 팔에 가려진 그 광선 뒤로 무엇이 있는지 이 프레임은 모른다.
            depth = np.where(np.asarray(robot_mask, bool), 0.0, depth)
        K = np.asarray(camera_intrinsics, np.float64)
        T = np.asarray(T_base_cam, np.float64)
        h, w = depth.shape

        lo, hi = self._frustum_index_bounds(K, T, w, h, depth_min, depth_max)
        sub_shape = tuple(int(hi[i] - lo[i] + 1) for i in range(3))
        n_considered = int(np.prod(sub_shape))
        ax = [self.grid.origin[i] + (lo[i] + np.arange(sub_shape[i])) * self.grid.voxel_size
              for i in range(3)]
        # float32 로 계산한다. 복셀 크기가 5 mm 이상이라 float32 의 상대오차(1e-7)는 격자
        # 해상도보다 네 자릿수 아래다.
        centres = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
        cam = ((centres - T[:3, 3]) @ T[:3, :3]).astype(np.float32)
        z = cam[:, 2]
        front = z > _EPS
        u = np.full(len(centres), -1.0)
        v = np.full(len(centres), -1.0)
        u[front] = K[0, 0] * cam[front, 0] / z[front] + K[0, 2]
        v[front] = K[1, 1] * cam[front, 1] / z[front] + K[1, 2]
        # 캐스팅 전에 자른다. z 가 아주 작은 복셀은 u, v 가 int64 범위를 넘어 캐스팅이
        # 정의되지 않는데, 그 복셀들은 어차피 `front` 로 걸러진다.
        ui = np.rint(np.clip(u, -1.0, w + 1.0)).astype(np.int64)
        vi = np.rint(np.clip(v, -1.0, h + 1.0)).astype(np.int64)
        in_image = front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if not in_image.any():
            return {"n_updated": 0, "n_in_image": 0, "n_considered": n_considered}

        sel = np.nonzero(in_image)[0]
        measured = depth[vi[sel], ui[sel]]
        valid = np.isfinite(measured) & (measured >= depth_min) & (measured <= depth_max)
        sel = sel[valid]
        if sel.size == 0:
            return {"n_updated": 0, "n_in_image": int(in_image.sum()), "n_considered": n_considered}
        measured = depth[vi[sel], ui[sel]]
        sdf = measured - z[sel]
        # 표면 뒤로 절단을 넘어선 곳은 관측이 아니라 가림이다. 건드리지 않는다.
        keep = sdf >= -self.truncation
        sel, sdf = sel[keep], sdf[keep]
        if sel.size == 0:
            return {"n_updated": 0, "n_in_image": int(in_image.sum()), "n_considered": n_considered}
        sdf = np.minimum(sdf, self.truncation)

        # `self.tsdf[lo:hi]`는 대개 비연속 뷰라 `.reshape(-1)`가 조용히 복사본을 만든다 — 그러면
        # 아래 대입이 원본에 안 먹는다. 그 대신 로컬 (i,j,k)로 되돌려 그 뷰에 팬시 인덱싱으로
        # 직접 쓴다. 팬시 인덱싱 대입은 뷰든 아니든 스트라이드를 그대로 따라가므로 원본에 반영된다.
        sl = tuple(slice(int(lo[i]), int(hi[i]) + 1) for i in range(3))
        view_t, view_w = self.tsdf[sl], self.weight[sl]
        ii, jj, kk = np.unravel_index(sel, sub_shape)
        w_old = view_w[ii, jj, kk]
        w_new = np.minimum(w_old + 1.0, max_weight)
        view_t[ii, jj, kk] = (view_t[ii, jj, kk] * w_old + sdf.astype(np.float32)) / np.maximum(w_new, _EPS)
        view_w[ii, jj, kk] = w_new
        return {"n_updated": int(sel.size), "n_in_image": int(in_image.sum()), "n_considered": n_considered}

    def occupancy(self, *, surface_band: Optional[float] = None) -> np.ndarray:
        """`FREE` / `OCCUPIED` / `UNKNOWN` 격자.

        가중치가 0인 복셀은 어느 카메라도 보지 못한 것이므로 `UNKNOWN` 이다. 관측된 복셀은
        TSDF 가 표면 띠 안이거나 음수면 `OCCUPIED`, 아니면 `FREE`.
        """
        band = self.grid.voxel_size if surface_band is None else float(surface_band)
        out = np.full(self.grid.shape, UNKNOWN, np.int8)
        seen = self.weight > 0.0
        out[seen] = FREE
        out[seen & (self.tsdf <= band)] = OCCUPIED
        return out


@dataclasses.dataclass
class EsdfField:
    """부호 있는 거리 격자와 그 조회. 이 backend 가 TO 에 넘기는 것.

    `distance` 는 삼선형 보간, `gradient` 는 격자 위 중심차분을 같은 방식으로 보간한다. 둘 다
    격자 밖 점에서는 가장자리 값을 쓰되 `outside_distance` 로 대체할 수 있다.

    **격자 밖은 격자 안의 `UNKNOWN`과 같은 것이지 별개의 상태가 아니다.** `EsdfBuilder`가
    `outside_distance`를 채울 때 `unknown_policy`를 그대로 따른다 — `free`면
    `+max_distance`, `occupied`면 `-max_distance`. 예전에는 `unknown_policy`와 무관하게
    항상 `+max_distance`(자유)였는데, 그러면 `occupied`로 안전 쪽을 택한 호출자도 격자 밖에서는
    "관측 실패"가 "0.5 m 떨어져서 안전함"으로 조용히 둔갑했다 (`docs/AG3S_REVIEW_LOG.md`
    Step 1, E4). `outside_query_count` / `outside_query_fraction`이 `distance()` 호출마다
    격자 밖 조회 비율을 세어, `unknown_fraction`처럼 셀 수 있게 만든다 — 이 파일의 설계 철학이
    "조용히 정하지 않고 보고한다"는 것이므로, 격자 밖도 예외가 아니어야 한다.

    **이산화 편향은 복셀 반 칸이고 부호가 보수적이다.** 표면은 점유 복셀의 *중심*에 표시되므로
    거리장이 참값보다 최대 `voxel_size / 2` 만큼 **작게** 나온다. 합성 구로 측정한 값이
    10 mm 복셀에서 정확히 -5.0 mm 다. 제약이 `d_esdf - r - margin >= 0` 이므로 작게 나오는 것은
    여유를 더 요구하는 쪽이고, 즉 틀리는 방향이 안전한 쪽이다. 대신 20 mm 복셀에서는 그 편향이
    10 mm 여서 50 mm 여유거리의 5분의 1을 이미 쓴다 — 해상도 선택이 곧 예산 선택이다.
    """

    grid: VoxelGrid
    distance_grid: np.ndarray  # (nx, ny, nz) float32, 미터. 음수 = 표면 안쪽
    max_distance: float
    stats: dict[str, Any] = dataclasses.field(default_factory=dict)
    outside_distance: Optional[float] = None
    #: `distance()` 호출이 누적한 것. `outside_query_fraction` 참고.
    outside_query_count: int = dataclasses.field(default=0, repr=False)
    queried_point_count: int = dataclasses.field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.distance_grid.shape != tuple(self.grid.shape):
            raise ValueError(
                f"거리 격자 {self.distance_grid.shape} 가 복셀 격자 {self.grid.shape} 와 다릅니다")

    @property
    def unknown_fraction(self) -> float:
        return float(self.stats.get("unknown_fraction", 0.0))

    @property
    def outside_query_fraction(self) -> float:
        """이 필드에 지금까지 들어온 `distance()` 조회 중 격자 밖이었던 비율.

        `unknown_fraction`은 필드를 만들 때 한 번 정해지는 정적 통계고, 이건 그 필드가
        **얼마나 많이 격자 밖에서 질의됐는지** — 즉 `default_bounds()`가 자른 상자가 이번
        롤아웃에서 실제로 얼마나 자주 걸렸는지 — 를 나중에 되짚어볼 수 있게 남기는 값이다.
        """
        if self.queried_point_count == 0:
            return 0.0
        return self.outside_query_count / self.queried_point_count

    def _lattice(self, points: np.ndarray):
        p = np.asarray(points, np.float64).reshape(-1, 3)
        f = (p - self.grid.origin) / self.grid.voxel_size
        n = np.asarray(self.grid.shape) - 1
        inside = np.all((f >= 0.0) & (f <= n), axis=1)
        fc = np.clip(f, 0.0, n)
        i0 = np.floor(fc).astype(np.int64)
        i0 = np.minimum(i0, np.maximum(n - 1, 0))
        t = fc - i0
        return i0, t, inside

    def distance(self, points: np.ndarray) -> np.ndarray:
        """`(N,)` 미터. 삼선형 보간."""
        i0, t, inside = self._lattice(points)
        g = self.distance_grid
        n = np.asarray(self.grid.shape) - 1
        out = np.zeros(len(i0), np.float64)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    idx = np.minimum(i0 + np.array([dx, dy, dz]), n)
                    wgt = ((t[:, 0] if dx else 1.0 - t[:, 0])
                           * (t[:, 1] if dy else 1.0 - t[:, 1])
                           * (t[:, 2] if dz else 1.0 - t[:, 2]))
                    out += wgt * g[idx[:, 0], idx[:, 1], idx[:, 2]]
        self.queried_point_count += len(i0)
        self.outside_query_count += int((~inside).sum())
        if self.outside_distance is not None and (~inside).any():
            out[~inside] = float(self.outside_distance)
        return out

    def gradient(self, points: np.ndarray) -> np.ndarray:
        """`(N, 3)` 무차원. 거리장의 기울기이므로 자유 공간에서 크기가 약 1이다.

        격자 위에서 중심차분한 뒤 같은 삼선형 가중치로 보간한다. 점마다 유한차분을 다시 하는
        것보다 싸고, 무엇보다 `distance` 와 **같은 격자·같은 보간**을 쓰므로 값과 기울기가
        서로 어긋나지 않는다 — 어긋나면 SQP 가 수렴하지 않는다.
        """
        if not hasattr(self, "_grad_cache") or self._grad_cache is None:
            gx, gy, gz = np.gradient(self.distance_grid.astype(np.float64),
                                     self.grid.voxel_size, edge_order=1)
            self._grad_cache = np.stack([gx, gy, gz], axis=-1).astype(np.float32)
        g = self._grad_cache
        i0, t, inside = self._lattice(points)
        n = np.asarray(self.grid.shape) - 1
        out = np.zeros((len(i0), 3), np.float64)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    idx = np.minimum(i0 + np.array([dx, dy, dz]), n)
                    wgt = ((t[:, 0] if dx else 1.0 - t[:, 0])
                           * (t[:, 1] if dy else 1.0 - t[:, 1])
                           * (t[:, 2] if dz else 1.0 - t[:, 2]))
                    out += wgt[:, None] * g[idx[:, 0], idx[:, 1], idx[:, 2]]
        if (~inside).any():
            out[~inside] = 0.0
        return out


def occupied_mask(occupancy: np.ndarray, unknown_policy: str) -> np.ndarray:
    """EDT 가 실제로 보는 이진 집합. dirty 판정도 **이것**을 비교해야 한다.

    세 상태를 그대로 비교하면 안 된다. `unknown_policy="free"` 에서 UNKNOWN -> FREE 전환은 거리장을
    전혀 바꾸지 않는데(둘 다 비점유), 손목 카메라가 움직이면 그 전환이 격자 전체에 퍼진다. 그러면
    dirty 상자가 격자 전체가 되어 국소 갱신이 이름만 남는다. (첫 판이 그랬다.)
    """
    if unknown_policy == "occupied":
        return (occupancy == OCCUPIED) | (occupancy == UNKNOWN)
    return occupancy == OCCUPIED


def esdf_from_occupancy(occupancy: np.ndarray, grid: VoxelGrid, *, max_distance: float,
                        unknown_policy: str = "free",
                        subbox: Optional[tuple[np.ndarray, np.ndarray]] = None,
                        previous: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict]:
    """점유 격자 -> 부호 있는 거리 격자. 음수는 점유 영역 안쪽.

    `subbox` 는 `(lo, hi)` 하나이거나 그 목록이다. 각 부분격자만 계산해 `previous` 에 덮어쓰며,
    호출자가 `max_distance` 만큼 부풀려 넘겨야 한다 — 그러면 그 안의 `max_distance` 이하 거리는
    전역 계산과 같은 값이 된다. 더 가까운 표면이 부분격자 밖에 있을 수 없기 때문이다.

    **목록을 받는 이유**는 하나의 AABB 가 흩어진 변화를 요약하지 못하기 때문이다. 팔이 작업
    공간의 양 끝에서 움직이면 바뀐 복셀이 800개뿐이어도 그 AABB 는 격자의 90%가 되고, 국소
    갱신이 이름만 남는다. 블록 단위로 나누면 실제로 바뀐 블록만 다시 계산한다.
    """
    from scipy import ndimage

    if unknown_policy not in ("free", "occupied"):
        raise ValueError(f"unknown_policy 는 'free' 또는 'occupied': {unknown_policy!r}")
    occ = occupied_mask(occupancy, unknown_policy)

    vs = grid.voxel_size

    def _edt(view: np.ndarray) -> np.ndarray:
        if not view.any():
            return np.full(view.shape, float(max_distance), np.float32)
        outside = ndimage.distance_transform_edt(~view, sampling=(vs, vs, vs))
        inside = ndimage.distance_transform_edt(view, sampling=(vs, vs, vs))
        return np.clip(np.where(view, -inside, outside), -float(max_distance),
                       float(max_distance)).astype(np.float32)

    if subbox is None:
        field = _edt(occ)
        n_recomputed = occ.size
    else:
        boxes = ([subbox] if isinstance(subbox, tuple) and np.ndim(subbox[0]) == 1
                 else list(subbox))
        field = previous if previous is not None else np.full(grid.shape, float(max_distance),
                                                              np.float32)
        n_recomputed = 0
        for lo, hi in boxes:
            sl = tuple(slice(int(lo[i]), int(hi[i]) + 1) for i in range(3))
            local = _edt(occ[sl])
            field[sl] = local
            n_recomputed += int(local.size)

    total = occupancy.size
    stats = {
        "n_occupied": int((occupancy == OCCUPIED).sum()),
        "n_free": int((occupancy == FREE).sum()),
        "n_unknown": int((occupancy == UNKNOWN).sum()),
        "unknown_fraction": float((occupancy == UNKNOWN).sum() / max(total, 1)),
        "unknown_policy": unknown_policy,
        "local": subbox is not None,
        "n_recomputed": int(n_recomputed),
    }
    return field, stats


def carve(occupancy: np.ndarray, grid: VoxelGrid, points: np.ndarray, *,
          dilate: int = 0) -> int:
    """`points` 가 차지한 복셀을 점유에서 빼낸다. 반환값은 빼낸 복셀 수.

    target 을 충돌 ESDF 에서 제외할 때 쓴다. **제거가 아니라 이관**이다 — target 의 기하는
    `CollisionConstraintSet` 에 여전히 후보로 남아 있고(5단계에서 확인한 성질), 여기서 빠지는
    것은 "잡아야 하는 물체를 장애물로도 세지 않는다"는 접촉 정책의 결과일 뿐이다.
    """
    idx, inside = grid.to_index(points)
    idx = idx[inside]
    if idx.size == 0:
        return 0
    mask = np.zeros(grid.shape, bool)
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if dilate > 0:
        from scipy import ndimage
        mask = ndimage.binary_dilation(mask, iterations=int(dilate))
    n = int((mask & (occupancy == OCCUPIED)).sum())
    occupancy[mask & (occupancy == OCCUPIED)] = FREE
    return n


__all__ = [
    "FREE", "OCCUPIED", "UNKNOWN",
    "EsdfField", "TsdfVolume", "VoxelGrid", "carve", "esdf_from_occupancy", "occupied_mask",
]


# ------------------------------------------------------------------------------ 빌더


@dataclasses.dataclass(frozen=True)
class CameraDepth:
    """ESDF 가 필요로 하는 카메라 관측 하나. `multiview.CameraObservation` 의 최소 부분집합."""

    name: str
    depth: np.ndarray            # (H, W) 미터
    camera_intrinsics: np.ndarray  # (3, 3)
    T_base_cam: np.ndarray       # (4, 4) 카메라 -> base
    #: (H, W) bool, True 인 픽셀은 로봇 자신이라 적분하지 않는다. `None` 이면 전부 적분한다.
    #:
    #: 로봇을 빼는 것은 두 가지 이유에서 옳다. 첫째, 팔은 자기 자신에 대한 장애물이 아니다 —
    #: 자기 충돌은 `attached.self_collision_mask` 와 로봇 모델이 따로 다룬다. 팔을 장애물
    #: 필드에 넣으면 팔이 자기 앞을 막는다. 둘째, 팔은 **움직인다.** 정적인 씬에서 매 프레임
    #: 바뀌는 복셀이 팔뿐이면 국소 갱신이 실제로 싸지는데, 팔을 남기면 바뀐 복셀이 격자 전체에
    #: 퍼져 국소 갱신이 이름만 남는다.
    robot_mask: Optional[np.ndarray] = None


def default_bounds(reach: float = 1.5, height: tuple[float, float] = (0.0, 1.60)):
    """base 프레임 작업 공간 상자. 팔이 닿을 수 없는 곳은 팔과 충돌할 수 없다.

    `reach` 는 관측으로 정한다 — RB-Y1 롤아웃에서 팔이 base 로부터 최대 1.389 m 까지 갔다
    (`docs/RUNBOOK.md`). 넉넉하게 잡는 방향이 안전한 방향이므로 1.5 m 로 둔다.

    상자를 앞쪽으로 치우치게 자르는 것은 비용 때문이다. `reach` 를 사방으로 두면 10 mm 복셀에서
    14.8 M 복셀(59 MB)이고 최초 빌드가 6.5 초다. 앞쪽 상자는 4.3 M(17 MB)에 1.8 초다. **뒤쪽을
    자르는 것이 안전한가**는 따로 답해야 하는 질문이고, 답은 "이 과제에서는 그렇다"이다 — 로봇은
    고정 베이스이고 조작 대상이 전부 앞 테이블 위에 있다. 베이스가 도는 과제로 가면
    `esdf.bounds_lower` / `bounds_upper` 로 넓혀야 하며, 그때 비용이 다시 든다.
    """
    return (np.array([-0.3, -reach * 0.6, height[0]]),
            np.array([reach * 0.8, reach * 0.6, height[1]]))


class EsdfBuilder:
    """프레임마다 TSDF 를 적분하고 ESDF 를 갱신한다. 격자는 한 번만 할당한다.

    같은 인스턴스를 프레임마다 재사용하는 것이 `incremental` 이 의미를 갖는 유일한 방법이다 —
    새로 만들면 전체를 다시 계산하게 되고, 그건 `build_once` 가 하는 일이다.
    """

    def __init__(self, config, *, bounds=None):
        self.config = config
        if bounds is None:
            if config.bounds_lower is not None:
                bounds = (np.asarray(config.bounds_lower, float),
                          np.asarray(config.bounds_upper, float))
            else:
                bounds = default_bounds()
        self.grid = VoxelGrid.from_bounds(bounds[0], bounds[1], config.voxel_size)
        self.volume = TsdfVolume(self.grid, truncation=config.truncation)
        self._field: Optional[np.ndarray] = None
        self._occupancy: Optional[np.ndarray] = None
        self._n_changed = 0
        self._frames = 0

    def _dirty_blocks(self, changed_flat: np.ndarray, cfg):
        """바뀐 복셀을 담은 블록들을 `max_distance` 만큼 부풀려 돌려준다.

        격자를 `block_voxels` 크기의 블록으로 나누고 바뀐 복셀이 있는 블록만 고른다. 각 블록은
        `max_distance / voxel_size` 만큼 부풀리므로 그 안의 거리는 전역 계산과 같다. 부풀린
        블록끼리 겹치는 것은 문제가 아니다 — 같은 값을 두 번 쓸 뿐이다.

        블록이 너무 많아 전역보다 비싸질 수 있으므로, 다시 계산할 복셀 수가 격자의 절반을 넘으면
        전역으로 되돌린다. 되돌리는 쪽이 느려지지 않고, 결과는 어느 쪽이든 같다.
        """
        shape = np.asarray(self.grid.shape)
        blk = int(cfg.block_voxels)
        pad = int(np.ceil(cfg.max_distance / cfg.voxel_size)) + 1
        ijk = np.stack(np.unravel_index(changed_flat, self.grid.shape), axis=1)
        keys = np.unique(ijk // blk, axis=0)
        boxes = []
        total = 0
        for k in keys:
            lo = np.maximum(k * blk - pad, 0)
            hi = np.minimum((k + 1) * blk - 1 + pad, shape - 1)
            boxes.append((lo, hi))
            total += int(np.prod(hi - lo + 1))
            if total > self.grid.n_voxels // 2:
                return None  # 전역이 더 싸다
        return boxes

    # --- 한 프레임 --------------------------------------------------------------------
    def update(self, cameras: Sequence[CameraDepth], *,
               target_points: Optional[np.ndarray] = None,
               exclude_target: bool = False,
               support_points: Optional[np.ndarray] = None) -> EsdfField:
        """관측을 적분하고 ESDF 를 (가능하면 국소로) 갱신해 필드를 돌려준다.

        `support_points` 는 지지면으로 이미 half-space 행이 나간 점들이다. 필드에도 남겨두면
        같은 표면을 두 backend 가 **서로 다른 여유거리로** 요구하게 된다 — 평면 행은 10 mm,
        필드는 50 mm. 파내는 것이 옳고, 잃는 것은 없다.
        """
        cfg = self.config
        per_camera = {}
        for cam in cameras:
            per_camera[cam.name] = self.volume.integrate(
                cam.depth, cam.camera_intrinsics, cam.T_base_cam,
                depth_min=cfg.depth_min, depth_max=cfg.depth_max, max_weight=cfg.max_weight,
                robot_mask=cam.robot_mask)

        occupancy = self.volume.occupancy(surface_band=cfg.surface_band)
        n_carved = 0
        n_support_carved = 0
        if support_points is not None and len(support_points):
            n_support_carved = carve(occupancy, self.grid, support_points)
        if exclude_target and target_points is not None and len(target_points):
            n_carved = carve(occupancy, self.grid, target_points,
                             dilate=cfg.target_dilate_voxels)

        # 국소 갱신의 범위는 **점유 상태가 실제로 바뀐** 복셀이 정한다. TSDF 가 닿은 복셀로
        # 잡으면 안 된다 — 카메라가 보는 부피 전체가 매 프레임 닿지만 그중 상태가 바뀌는 것은
        # 표면 근처 얇은 껍질뿐이고, 닿은 것을 기준으로 하면 dirty 상자가 격자 전체가 되어
        # 국소 갱신이 이름만 남는다. (첫 판이 그래서 1.82s -> 1.77s 로 이득이 없었다.)
        occ_now = occupied_mask(occupancy, cfg.unknown_policy)
        subbox = None
        if cfg.incremental and self._field is not None and self._occupancy is not None:
            changed = np.nonzero((occ_now != self._occupancy).reshape(-1))[0]
            if changed.size == 0:
                # 아무것도 바뀌지 않았다. 빈 목록이면 EDT 를 한 번도 돌지 않고 이전 필드가
                # 그대로 유효하다 — 정적인 씬에서 국소 갱신이 공짜가 되는 지점이다.
                self._n_changed = 0
                subbox = []
            else:
                self._n_changed = int(changed.size)
                subbox = self._dirty_blocks(changed, cfg)
        self._occupancy = occ_now

        field_grid, stats = esdf_from_occupancy(
            occupancy, self.grid, max_distance=cfg.max_distance,
            unknown_policy=cfg.unknown_policy, subbox=subbox, previous=self._field)
        self._field = field_grid
        self._frames += 1
        stats.update({
            "n_blocks": (0 if subbox is None else len(subbox)),
            "voxel_size": cfg.voxel_size,
            "grid_shape": tuple(self.grid.shape),
            "n_voxels": self.grid.n_voxels,
            "frames": self._frames,
            "n_target_voxels_carved": n_carved,
            "n_support_voxels_carved": n_support_carved,
            "n_occupancy_changed": int(getattr(self, "_n_changed", 0)),
            "per_camera": per_camera,
        })
        # 격자 밖은 격자 안의 UNKNOWN 과 같은 것이므로 같은 정책을 따른다 — `unknown_policy`
        # 와 무관하게 항상 자유였던 것이 E4 였다.
        outside = cfg.max_distance if cfg.unknown_policy == "free" else -cfg.max_distance
        return EsdfField(grid=self.grid, distance_grid=field_grid,
                         max_distance=cfg.max_distance, stats=stats,
                         outside_distance=outside)


def build_once(cameras: Sequence[CameraDepth], config, *, bounds=None,
               target_points: Optional[np.ndarray] = None,
               exclude_target: bool = False) -> EsdfField:
    """일회성 빌드. 프레임 간 재사용이 없으므로 증분 갱신도 없다."""
    return EsdfBuilder(config, bounds=bounds).update(
        cameras, target_points=target_points, exclude_target=exclude_target)


__all__ += ["CameraDepth", "EsdfBuilder", "build_once", "default_bounds"]
