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
import time
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

    def frustum_mask(self, camera_intrinsics, T_base_cam, shape, *,
                     depth_min: float = 0.1, depth_max: float = 3.0) -> np.ndarray:
        """`(nx, ny, nz)` bool — 이 카메라가 **지금 보고 있는** 복셀.

        `integrate` 의 투영을 깊이 비교 **전까지만** 쓴 것이다. 깊이가 유효한지는 묻지 않는다 —
        "보이는 자리인가" 와 "무엇이 보였나" 는 다른 질문이고, 감쇠가 묻는 것은 앞쪽이다.
        """
        K = np.asarray(camera_intrinsics, np.float64)
        T = np.asarray(T_base_cam, np.float64)
        h, w = int(shape[0]), int(shape[1])
        out = np.zeros(self.grid.shape, bool)
        lo, hi = self._frustum_index_bounds(K, T, w, h, depth_min, depth_max)
        sub = tuple(int(hi[i] - lo[i] + 1) for i in range(3))
        ax = [self.grid.origin[i] + (lo[i] + np.arange(sub[i])) * self.grid.voxel_size
              for i in range(3)]
        centres = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
        cam = ((centres - T[:3, 3]) @ T[:3, :3]).astype(np.float32)
        z = cam[:, 2]
        front = (z > _EPS) & (z >= depth_min) & (z <= depth_max)
        u = np.full(len(centres), -1.0); v = np.full(len(centres), -1.0)
        u[front] = K[0, 0] * cam[front, 0] / z[front] + K[0, 2]
        v[front] = K[1, 1] * cam[front, 1] / z[front] + K[1, 2]
        ui = np.rint(np.clip(u, -1.0, w + 1.0)).astype(np.int64)
        vi = np.rint(np.clip(v, -1.0, h + 1.0)).astype(np.int64)
        seen = front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        sl = tuple(slice(int(lo[i]), int(hi[i]) + 1) for i in range(3))
        out[sl] = seen.reshape(sub)
        return out

    def decay(self, frustums: Sequence[np.ndarray] = (), *,
              time_decay: float = 1.0, frustum_decay: float = 1.0) -> dict:
        """관측의 무게를 줄인다 — cuRoboV2 §5.3 의 frustum-aware decay.

            w <- w * a_t * a_f   (절두체 안)
            w <- w * a_t         (그 밖)

        **왜 필요한가.** 우리 가중치는 `min(w + 1, max_weight)` 로 **단조 증가**한다. 옛 관측이
        절대 흐려지지 않으므로 사라진 물체의 잔상이 오래 남는다 — 실측(`run_0004`, 머리 카메라,
        20 mm): 사과가 원래 자리에서 **302 mm 떠난 뒤에도 그 자리가 -6.7 mm 로 점유**였고,
        8 프레임 내내 한 번도 자유가 되지 않았다 (F20).

        가중치를 줄이면 새 관측이 그만큼 빨리 이긴다. TSDF 갱신이 가중평균
        ``t <- (t*w_old + sdf) / w_new`` 이므로, `w_old` 가 작을수록 이번 프레임의 `sdf` 가
        지배한다.

        **절두체 안에만 더 줄이는 이유**는 비대칭이다. 지금 보고 있는 곳은 틀렸다면 바로
        고칠 수 있지만, 안 보이는 곳은 고칠 방법이 없다. 그래서 보이는 곳은 빨리 잊고
        (`frustum_decay`), 안 보이는 곳은 천천히 잊는다 (`time_decay`).

        **끄는 것이 기본값이다** (둘 다 1.0). 감쇠는 잔상을 지우지만 **실제 장애물도 함께
        잊는다** — 시야에서 벗어난 상자가 흐려져 미관측으로 돌아가면 그것은 다시 자유로 나간다.
        우리는 카메라가 셋이고 프레임이 느려서 그 위험이 cuRoboV2 보다 크다. 켜는 값은 씬에서
        재고 정한다.
        """
        if time_decay >= 1.0 and frustum_decay >= 1.0:
            return {"decayed": False}
        before = float(self.weight.sum())
        if time_decay < 1.0:
            self.weight *= np.float32(time_decay)
        n_in = 0
        if frustum_decay < 1.0:
            for mask in frustums:
                m = np.asarray(mask, bool)
                self.weight[m] *= np.float32(frustum_decay)
                n_in += int(m.sum())
        return {"decayed": True, "weight_before": before,
                "weight_after": float(self.weight.sum()),
                "n_in_frustum": n_in}

    def occupancy(self, *, surface_band: Optional[float] = None,
                  min_weight: float = 0.0) -> np.ndarray:
        """`FREE` / `OCCUPIED` / `UNKNOWN` 격자.

        가중치가 0인 복셀은 어느 카메라도 보지 못한 것이므로 `UNKNOWN` 이다. 관측된 복셀은
        TSDF 가 표면 띠 안이거나 음수면 `OCCUPIED`, 아니면 `FREE`.
        """
        band = self.grid.voxel_size if surface_band is None else float(surface_band)
        out = np.full(self.grid.shape, UNKNOWN, np.int8)
        # `min_weight` 아래로 흐려진 복셀은 **다시 미관측**이다. 감쇠를 켰을 때 잊는다는 것이
        # 실제로 뜻하는 바가 이것이다 — 값이 남아 있어도 근거가 사라졌으면 근거 없음으로 돌린다.
        seen = self.weight > max(float(min_weight), 0.0)
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
    #: 라벨 층 `(nx, ny, nz)` int32. 각 복셀에는 **그 복셀의 거리를 만든 표면 복셀의 라벨**이
    #: 들어 있다 (`-1` = 라벨 없음). 거리장은 원래 익명이라 "가장 가까운 것이 무엇인지" 를
    #: 말하지 못하는데, 그 익명을 푸는 것이 이 층이다 — 그래야 "목적지에는 얇은 마진" 같은
    #: 정책을 걸 수 있다 (F18). `None` 이면 라벨 없이 만들어진 필드다.
    label_grid: Optional[np.ndarray] = None
    #: 라벨 id -> 이름. `label_grid` 의 값이 이 튜플의 인덱스다.
    label_names: tuple[str, ...] = ()
    #: **해석적 채널** — 아는 정적 기하 (`StaticBox` / `StaticPlane`). `distance` 가
    #: `min(복셀, 해석적)` 을 답한다. cuRoboV2 §5.1 의 `min(depth, geom)` 과 같은 발상이되,
    #: 우리는 복셀에 찍지 않고 **따로 잰다** — 중요한 기하(선반·벽)가 격자 밖에 있어서 찍기로는
    #: 닿지 않기 때문이다 (`analytic_distance` 참고). 비어 있으면 아무 일도 하지 않는다.
    static_shapes: tuple = ()
    #: `distance()` 호출이 누적한 것. `outside_query_fraction` 참고.
    outside_query_count: int = dataclasses.field(default=0, repr=False)
    queried_point_count: int = dataclasses.field(default=0, repr=False)
    #: `FieldProvenance` — 이 필드가 **언제 · 무엇으로 · 몇 번째로** 만들어졌는가.
    #: 거리값만 보면 방금 만든 것과 낡은 것을 구별할 수 없고, 그것이 2026-09-18 의 조용한
    #: 정지(14 프레임 동안 지각이 안 돌았는데 상태는 `ok`)가 안 보인 이유다.
    provenance: Any = None

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
        if self.static_shapes:
            # `min(복셀, 해석적)`. 격자 밖 질의에도 그대로 걸린다 — 거기가 이 채널이 가장
            # 필요한 자리다.
            out = np.minimum(out, analytic_distance(points, self.static_shapes))
        return out

    # --- 라벨 층 -------------------------------------------------------------------------
    @property
    def has_labels(self) -> bool:
        return self.label_grid is not None

    def label_id(self, name: str) -> int:
        """이름에 해당하는 라벨 id. 없으면 `-1` — 예외가 아니라 "라벨 없음" 과 같은 값이다.

        모르는 이름이 예외가 아닌 이유는 소비 쪽 때문이다. 정책은 "가장 가까운 것이 목적지인가"
        를 묻는데, 목적지가 없는 프레임에서 그 질문의 답은 "아니다" 이지 오류가 아니다. 그리고
        `-1` 과 비교하면 라벨 없는 복셀과도 일치하지 않으므로 **완화가 일어나지 않는 쪽**으로
        닫힌다.
        """
        try:
            return int(self.label_names.index(str(name)))
        except ValueError:
            return -1

    def label(self, points: np.ndarray) -> np.ndarray:
        """`(N,)` int32 — 각 점에서 **가장 가까운 표면이 속한 물체**의 라벨 id.

        거리와 달리 보간하지 않는다. 라벨은 범주이고 두 라벨의 가중평균에는 뜻이 없다 —
        가장 가까운 격자점의 값을 그대로 읽는다. 격자 밖과 라벨 없는 복셀은 둘 다 `-1` 이다.
        """
        if self.label_grid is None:
            return np.full(len(np.asarray(points).reshape(-1, 3)), -1, np.int32)
        i0, t, inside = self._lattice(points)
        n = np.asarray(self.grid.shape) - 1
        idx = np.minimum(i0 + np.round(t).astype(np.int64), n)
        out = self.label_grid[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int32)
        out[~inside] = -1
        return out

    def is_label(self, points: np.ndarray, name: str) -> np.ndarray:
        """`(N,)` bool — 가장 가까운 표면이 `name` 인가. 라벨이 없으면 전부 False."""
        wanted = self.label_id(name)
        if wanted < 0:
            return np.zeros(len(np.asarray(points).reshape(-1, 3)), bool)
        return self.label(points) == wanted

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
        if self.static_shapes:
            # 해석적 채널이 이긴 점은 **그 도형의** 기울기를 쓴다. 거리만 바꾸고 기울기를 두면
            # 최적화기가 "가깝다" 는 값을 받고 엉뚱한 복셀 표면 쪽으로 밀린다.
            voxel = self.distance(points)          # 이미 min 이 적용된 값
            analytic = analytic_distance(points, self.static_shapes)
            wins = analytic <= voxel + 1e-12
            if wins.any():
                out[wins] = analytic_gradient(points, self.static_shapes)[wins]
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
                        previous: Optional[np.ndarray] = None,
                        labels: Optional[np.ndarray] = None,
                        previous_labels: Optional[np.ndarray] = None,
                        ) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
    """점유 격자 -> 부호 있는 거리 격자. 음수는 점유 영역 안쪽.

    `subbox` 는 `(lo, hi)` 하나이거나 그 목록이다. 각 부분격자만 계산해 `previous` 에 덮어쓰며,
    호출자가 `max_distance` 만큼 부풀려 넘겨야 한다 — 그러면 그 안의 `max_distance` 이하 거리는
    전역 계산과 같은 값이 된다. 더 가까운 표면이 부분격자 밖에 있을 수 없기 때문이다.

    **목록을 받는 이유**는 하나의 AABB 가 흩어진 변화를 요약하지 못하기 때문이다. 팔이 작업
    공간의 양 끝에서 움직이면 바뀐 복셀이 800개뿐이어도 그 AABB 는 격자의 90%가 되고, 국소
    갱신이 이름만 남는다. 블록 단위로 나누면 실제로 바뀐 블록만 다시 계산한다.

    **`labels` 를 주면 라벨 층이 함께 나온다.** 거리장은 원래 익명이라 "가장 가까운 것이
    **무엇인지**" 를 말하지 못하고, 그래서 "목적지에만 얇은 마진" 같은 정책을 걸 수 없다
    (F18 — 전역 50 mm 가 담기 동작을 막는 문제). 해법은 volumetric mapping 쪽에 이미 있다:
    Voxblox++ 도 TSDF++ 도 Panoptic Multi-TSDFs 도 **거리 층 옆에 라벨 층**을 둔다.

    비용은 사실상 없다. **거리 변환은 최근접 표면 복셀을 이미 계산한다** — `return_indices=True`
    가 그것을 돌려주므로, 라벨은 새 알고리즘이 아니라 조회 한 번이다. (cuRobo 쪽의 PBA+ 도
    본래 최근접 site 를 구하는 알고리즘이고 거리가 거기서 파생된다. 백엔드를 바꿔도 같은 구조가
    남는다.)

    라벨 격자는 `(nx, ny, nz)` int32 이고 `-1` 은 "라벨 없음" 이다. 출력은 각 복셀에 대해
    **그 복셀의 거리를 만든 표면 복셀의 라벨**이다 — 그 복셀 자신의 라벨이 아니다.
    """
    from scipy import ndimage

    if unknown_policy not in ("free", "occupied"):
        raise ValueError(f"unknown_policy 는 'free' 또는 'occupied': {unknown_policy!r}")
    occ = occupied_mask(occupancy, unknown_policy)

    vs = grid.voxel_size

    want_labels = labels is not None
    if want_labels and labels.shape != tuple(grid.shape):
        raise ValueError(f"라벨 격자 {labels.shape} 가 복셀 격자 {tuple(grid.shape)} 와 다릅니다")

    def _edt(view: np.ndarray, lab: Optional[np.ndarray]):
        if not view.any():
            empty = None if lab is None else np.full(view.shape, -1, np.int32)
            return np.full(view.shape, float(max_distance), np.float32), empty
        if lab is None:
            outside = ndimage.distance_transform_edt(~view, sampling=(vs, vs, vs))
        else:
            # 최근접 표면 복셀의 인덱스를 함께 받는다. 거리 변환이 이미 계산하는 값이라
            # 라벨 층은 알고리즘이 아니라 조회 한 번이다.
            outside, nearest = ndimage.distance_transform_edt(
                ~view, sampling=(vs, vs, vs), return_indices=True)
        inside = ndimage.distance_transform_edt(view, sampling=(vs, vs, vs))
        field = np.clip(np.where(view, -inside, outside), -float(max_distance),
                        float(max_distance)).astype(np.float32)
        if lab is None:
            return field, None
        # 표면 안쪽 복셀의 최근접 표면은 자기 자신이다.
        out = np.where(view, lab, lab[tuple(nearest)]).astype(np.int32)
        return field, out

    if subbox is None:
        field, label_field = _edt(occ, labels)
        n_recomputed = occ.size
    else:
        boxes = ([subbox] if isinstance(subbox, tuple) and np.ndim(subbox[0]) == 1
                 else list(subbox))
        field = previous if previous is not None else np.full(grid.shape, float(max_distance),
                                                              np.float32)
        label_field = None
        if want_labels:
            label_field = (previous_labels if previous_labels is not None
                           else np.full(grid.shape, -1, np.int32))
        n_recomputed = 0
        for lo, hi in boxes:
            sl = tuple(slice(int(lo[i]), int(hi[i]) + 1) for i in range(3))
            local, local_lab = _edt(occ[sl], None if labels is None else labels[sl])
            field[sl] = local
            if want_labels:
                label_field[sl] = local_lab
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
        "labelled": bool(want_labels),
    }
    return field, label_field, stats


@dataclasses.dataclass(frozen=True)
class StaticBox:
    """축이 돌아간 상자 하나. `rotation` 의 열이 상자 축이다."""

    center: np.ndarray        # (3,)
    half_extents: np.ndarray  # (3,)
    rotation: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3))
    label: str = "static"


@dataclasses.dataclass(frozen=True)
class StaticPlane:
    """반공간. `normal` 이 **자유공간 쪽**을 가리킨다 — 그 반대편이 고체다."""

    point: np.ndarray         # (3,)
    normal: np.ndarray        # (3,)
    label: str = "static"


def analytic_distance(points: np.ndarray, shapes: Sequence[Any]) -> np.ndarray:
    """`(N,)` — 아는 기하까지의 **정확한** 부호 있는 거리. 격자와 무관하다.

    복셀에 찍지 않고 따로 재는 이유가 측정에서 나왔다. 정적 기하를 점유 격자에 찍어 봤더니
    낙관 오차가 **0.0 mm** 줄었다 — 중요한 기하가 **격자 밖**이었기 때문이다:

        ESDF 격자   x [-300, 1200]  y [-900, 900]  z [0, 1600] mm
        선반        y = -1550 ~ -1685            <- 밖
        벽          x, y = ±3 m                  <- 밖
        바닥 평면   z = 0, 격자 최하단 복셀 중심이 z = 10 mm  <- 한 복셀도 안 찍힌다

    찍힌 것은 테이블뿐이었고, 테이블은 카메라가 이미 보고 있었다. **고정 상자 구조에서는 찍기가
    E4(격자 밖·미관측은 무조건 자유)를 풀지 못한다.**

    그래서 해석적으로 잰다. 격자가 없으니 밖이라는 개념이 없고, 이산화 오차도 없다. 상자 몇
    개를 재는 비용이라 질의당 무시할 수준이다.
    """
    P = np.asarray(points, np.float64).reshape(-1, 3)
    best = np.full(P.shape[0], np.inf)
    for shape in shapes or ():
        if isinstance(shape, StaticBox):
            R = np.asarray(shape.rotation, np.float64).reshape(3, 3)
            half = np.asarray(shape.half_extents, np.float64)
            q = np.abs((P - np.asarray(shape.center, np.float64)) @ R) - half
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(q.max(axis=1), 0.0)
            best = np.minimum(best, outside + inside)
        elif isinstance(shape, StaticPlane):
            n = np.asarray(shape.normal, np.float64)
            n = n / max(float(np.linalg.norm(n)), 1e-12)
            best = np.minimum(best, (P - np.asarray(shape.point, np.float64)) @ n)
        else:
            raise TypeError(f"analytic_distance 는 StaticBox 와 StaticPlane 만 받습니다: {type(shape)}")
    return best


def analytic_gradient(points: np.ndarray, shapes: Sequence[Any]) -> np.ndarray:
    """`(N, 3)` — `analytic_distance` 의 기울기. 이기는 도형의 것을 쓴다.

    거리만 바꾸고 기울기를 안 바꾸면 **더 나쁘다**: 최적화기는 "가깝다" 는 값을 받고 엉뚱한
    복셀 표면 쪽으로 밀린다. 그래서 둘을 같은 도형에서 뽑는다.
    """
    P = np.asarray(points, np.float64).reshape(-1, 3)
    best = np.full(P.shape[0], np.inf)
    grad = np.zeros_like(P)
    for shape in shapes or ():
        if isinstance(shape, StaticBox):
            R = np.asarray(shape.rotation, np.float64).reshape(3, 3)
            half = np.asarray(shape.half_extents, np.float64)
            local = (P - np.asarray(shape.center, np.float64)) @ R
            sign = np.sign(local); sign[sign == 0] = 1.0
            q = np.abs(local) - half
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(q.max(axis=1), 0.0)
            dist = outside + inside
            g_local = np.zeros_like(local)
            out = outside > 1e-12
            g_local[out] = np.maximum(q[out], 0.0) / outside[out][:, None]
            if (~out).any():
                # 상자 안쪽: 가장 가까운 면 방향.
                axis = np.argmax(q[~out], axis=1)
                tmp = np.zeros((int((~out).sum()), 3))
                tmp[np.arange(tmp.shape[0]), axis] = 1.0
                g_local[~out] = tmp
            g = (g_local * sign) @ R.T
            take = dist < best
            best[take] = dist[take]; grad[take] = g[take]
        elif isinstance(shape, StaticPlane):
            n = np.asarray(shape.normal, np.float64)
            n = n / max(float(np.linalg.norm(n)), 1e-12)
            dist = (P - np.asarray(shape.point, np.float64)) @ n
            take = dist < best
            best[take] = dist[take]; grad[take] = n
    return grad


def stamp_static(occupancy: np.ndarray, grid: VoxelGrid,
                 shapes: Sequence[Any]) -> tuple[int, dict]:
    """아는 기하를 점유 격자에 찍는다 — cuRoboV2 의 geometry 채널에 해당하는 것.

    **왜 필요한가.** 거리장은 depth 가 본 것만 담는다. 그래서 카메라가 한 번도 보지 않은 곳은
    `unknown_policy=free` 에 따라 **자유**로 답한다 (E4 — 격자 밖·미관측은 무조건 자유). 실측
    (`run_0004`, 3 카메라, 20 mm): 격자의 **66.3 %** 가 미관측이고, 로봇 구 질의의 **44.1 %** 가
    그런 복셀에 떨어지며, 그중 **237 회는 거리가 500 mm 로 포화**된다 — "아무것도 안 보인다" 가
    "반 미터 떨어져 안전하다" 로 나가는 것이다. 그 자리의 **참** 거리는 최소 **168.7 mm** 였다.
    필드가 **331 mm 낙관적**이었다는 뜻이다.

    벽·선반·테이블 다리·바닥은 **아는 기하**다. 움직이지 않고 CAD 가 있으므로 관측을 기다릴
    이유가 없다. 여기 찍어 두면 그 331 mm 가 사라진다.

    cuRoboV2 는 복셀마다 채널을 둘 두고 질의에서 `min(depth, geom)` 을 쓰지만(§5.1), 우리
    구조는 `점유 → EDT → 거리` 라 **점유에 찍는 것이 같은 일을 한다** — EDT 는 출처를 가리지
    않고 가장 가까운 점유 복셀까지의 거리를 답하기 때문이다. 차이는 아는 기하도 복셀로
    이산화된다는 것이고, 그 비용은 다른 모든 표면과 같은 복셀 반 칸이다.

    **찍기만 하고 지우지 않는다.** 이미 점유인 복셀은 그대로 두고 자유·미관측만 점유로 바꾼다 —
    관측이 아는 기하보다 우선할 이유가 없고, 반대도 마찬가지다. 둘 다 "여기 뭔가 있다" 이므로
    합집합이 맞다.
    """
    if not shapes:
        return 0, {"n_static_voxels": 0, "n_static_shapes": 0}

    ii = np.stack(np.meshgrid(*[np.arange(n) for n in grid.shape], indexing="ij"), -1)
    centres = grid.origin + ii * grid.voxel_size
    flat = centres.reshape(-1, 3)
    inside = np.zeros(flat.shape[0], bool)
    for shape in shapes:
        if isinstance(shape, StaticBox):
            R = np.asarray(shape.rotation, np.float64).reshape(3, 3)
            local = np.abs((flat - np.asarray(shape.center, np.float64)) @ R)
            inside |= np.all(local <= np.asarray(shape.half_extents, np.float64), axis=1)
        elif isinstance(shape, StaticPlane):
            n = np.asarray(shape.normal, np.float64)
            n = n / max(np.linalg.norm(n), 1e-12)
            inside |= ((flat - np.asarray(shape.point, np.float64)) @ n) <= 0.0
        else:
            raise TypeError(f"stamp_static 은 StaticBox 와 StaticPlane 만 받습니다: {type(shape)}")

    mask = inside.reshape(grid.shape)
    changed = int((mask & (occupancy != OCCUPIED)).sum())
    occupancy[mask] = OCCUPIED
    return changed, {"n_static_voxels": int(mask.sum()),
                     "n_static_changed": changed,
                     "n_static_shapes": len(shapes)}


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
    "EsdfField", "StaticBox", "StaticPlane", "TsdfVolume", "VoxelGrid", "carve",
    "analytic_distance", "analytic_gradient", "esdf_from_occupancy", "occupied_mask", "stamp_static",
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

    #: 이 프로세스에서 이 클래스가 몇 번 만들어졌는가. **추론을 불변식으로 바꾸는 장치다.**
    #:
    #: `AG3S_TOTAL_TEST_Prompt.md` 의 T0 은 "legacy 가 조용히 돌지 않았는가" 를 즉시 실패
    #: 조건으로 둔다. 프레임마다 출처 도장(`FieldProvenance.backend`)을 보면 *쓰인* 필드가
    #: 무엇인지는 알 수 있지만, 만들어만 놓고 안 쓴 경우는 안 잡힌다. 세면 그것도 잡힌다 —
    #: cuRobo backend 로 도는 동안 이 값이 0 이 아니면 `pipeline._build_esdf` 가 죽는다.
    instances_created: int = 0

    def __init__(self, config, *, bounds=None):
        type(self).instances_created += 1
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
               support_points: Optional[np.ndarray] = None,
               attached_points: Optional[np.ndarray] = None,
               labelled_points: Optional[dict] = None,
               static_geometry: Optional[Sequence[Any]] = None,
               observed_at: Optional[float] = None,
               frame_id: str = "", frame_index: int = -1) -> EsdfField:
        """관측을 적분하고 ESDF 를 (가능하면 국소로) 갱신해 필드를 돌려준다.

        `support_points` 는 지지면으로 이미 half-space 행이 나간 점들이다. 필드에도 남겨두면
        같은 표면을 두 backend 가 **서로 다른 여유거리로** 요구하게 된다 — 평면 행은 10 mm,
        필드는 50 mm. 파내는 것이 옳고, 잃는 것은 없다.

        `attached_points` 는 **쥔 물체**가 지금 차지한 자리다 (base 좌표계). 파지가 닫히면 그
        물체는 장애물이기를 그치고 로봇 쪽 질의점이 되므로 (E3 — 쥔 물체가 optimizer 에 도달하지
        않는다), 필드에서는 빠져야 한다. 안 빼면 물체가 자기 자신에게 부딪히고 그 행은 **어떤
        해로도 못 푼다** — 손에 강체로 붙어 있어 관절로는 자기 복셀에서 못 벗어난다.

        **자기 필터가 이 일을 대신해 주지 않는다.** 실측: 쥔 동안 사과 픽셀을 100 % 지우는데도
        (손목 카메라 기준 143,942 px) 마스크를 걷어낸 대조군과 점유가 최대 4 복셀밖에 안
        다르다. 마스크가 막는 것은 **새 관측**이고, 문제를 만드는 것은 물체가 테이블에 놓여
        있을 때 남긴 **옛 관측**이기 때문이다 (`run_0004` 프레임 10~11, 자기 질의점 −72.7 mm).

        `labelled_points` 는 `{이름: (N, 3) 점}` 이다. 주면 필드가 **라벨 층**을 함께 들고 나와
        `EsdfField.is_label(p, 이름)` 으로 "이 점에서 가장 가까운 표면이 그 물체인가" 를 물을 수
        있다. 거리장은 원래 익명이라 그 질문에 답할 수 없었고, 그래서 지금까지는 필드 거리와
        해석적 거리를 한 복셀 안에서 대조하는 우회로를 썼다 (F13 의 양면 검사).

        **라벨 이름이 바뀌면 국소 갱신을 포기하고 전체를 다시 계산한다.** 라벨은 거리와 달리
        점유가 바뀌지 않은 복셀에서도 바뀔 수 있다 — 물체 하나에 이름이 새로 붙으면 그 물체에서
        먼 복셀의 '가장 가까운 것' 라벨까지 달라진다. dirty 상자는 점유 변화만 보므로 그것을
        못 잡는다.
        """
        cfg = self.config
        per_camera = {}
        for cam in cameras:
            per_camera[cam.name] = self.volume.integrate(
                cam.depth, cam.camera_intrinsics, cam.T_base_cam,
                depth_min=cfg.depth_min, depth_max=cfg.depth_max, max_weight=cfg.max_weight,
                robot_mask=cam.robot_mask)

        # 감쇠는 **적분 뒤**다. 이번 프레임 관측을 먼저 쌓고, 그다음 전체를 흐린다 — 순서가
        # 반대면 방금 본 것을 보기도 전에 지운다.
        decay_stats = {"decayed": False}
        if cfg.time_decay < 1.0 or cfg.frustum_decay < 1.0:
            masks = []
            if cfg.frustum_decay < 1.0:
                for cam in cameras:
                    masks.append(self.volume.frustum_mask(
                        cam.camera_intrinsics, cam.T_base_cam, np.asarray(cam.depth).shape,
                        depth_min=cfg.depth_min, depth_max=cfg.depth_max))
            decay_stats = self.volume.decay(masks, time_decay=cfg.time_decay,
                                            frustum_decay=cfg.frustum_decay)

        occupancy = self.volume.occupancy(surface_band=cfg.surface_band,
                                          min_weight=cfg.min_weight)
        # 아는 기하를 먼저 찍는다 — 파내기(`carve`)보다 **앞**이어야 한다. 순서가 반대면
        # 지지면을 파낸 자리를 정적 기하가 다시 메워 F15(지지면을 아무도 제약하지 않는 조합)가
        # 노리는 그 상태로 되돌린다.
        # **찍지 않는다.** 실측: 정적 기하를 점유 격자에 찍으면 낙관 오차가 0.0 mm 줄었다 —
        # 중요한 기하(선반 y=-1550, 벽 ±3 m)가 격자 밖이라 찍힐 자리가 없었고, 찍힌 것은 이미
        # 관측된 테이블뿐이었다. 게다가 프레임당 206 -> 615 ms 였다. 대신 필드에 **해석적
        # 채널**로 실어 보내 `min(복셀, 해석적)` 로 답하게 한다 (`analytic_distance`).
        # `stamp_static` 은 격자 안 기하를 EDT 에 넣고 싶은 호출자를 위해 남겨 둔다.
        static_stats = {"n_static_shapes": len(static_geometry or ())}
        n_carved = 0
        n_support_carved = 0
        if support_points is not None and len(support_points):
            n_support_carved = carve(occupancy, self.grid, support_points)
        if exclude_target and target_points is not None and len(target_points):
            n_carved = carve(occupancy, self.grid, target_points,
                             dilate=cfg.target_dilate_voxels)
        # 쥔 물체를 파낸다. **지지면 파내기 뒤, 라벨 붙이기 앞**이어야 한다 — 라벨은 점유
        # 복셀에 붙으므로, 파낸 뒤에 붙여야 사라진 복셀에 이름이 남지 않는다.
        n_attached_carved = 0
        if attached_points is not None and len(attached_points):
            n_attached_carved = carve(occupancy, self.grid, attached_points,
                                      dilate=cfg.attached_dilate_voxels)

        # 국소 갱신의 범위는 **점유 상태가 실제로 바뀐** 복셀이 정한다. TSDF 가 닿은 복셀로
        # 잡으면 안 된다 — 카메라가 보는 부피 전체가 매 프레임 닿지만 그중 상태가 바뀌는 것은
        # 표면 근처 얇은 껍질뿐이고, 닿은 것을 기준으로 하면 dirty 상자가 격자 전체가 되어
        # 국소 갱신이 이름만 남는다. (첫 판이 그래서 1.82s -> 1.77s 로 이득이 없었다.)
        # 라벨 격자. 나중에 쓴 이름이 이긴다 — 같은 복셀을 두 물체가 주장하면 마지막이 남고,
        # 그 순서는 호출자가 dict 로 정한다.
        #
        # **라벨은 관측 점의 복셀이 아니라 점유 복셀에 붙어야 한다.** 필드의 표면을 정하는 것은
        # TSDF 의 영교차이고, 그것이 원본 점이 떨어진 복셀과 꼭 같지는 않다. 처음에 점 복셀에만
        # 붙였더니 최근접 표면 복셀이 전부 라벨 없음으로 나왔다 — 실측으로 잡은 것이다. 그래서
        # 점에서 만든 라벨을 `label_snap_voxels` 안의 점유 복셀로 한 번 옮긴다.
        label_grid = None
        label_names: tuple = ()
        if labelled_points:
            from scipy import ndimage as _nd

            label_names = tuple(str(k) for k in labelled_points)
            seed = np.full(self.grid.shape, -1, np.int32)
            for lid, name in enumerate(label_names):
                pts = np.asarray(labelled_points[name], np.float64).reshape(-1, 3)
                if not len(pts):
                    continue
                idx, inside = self.grid.to_index(pts)
                idx = idx[inside]
                if idx.size:
                    seed[idx[:, 0], idx[:, 1], idx[:, 2]] = lid
            snap = float(getattr(cfg, "label_snap_voxels", 2.0)) * cfg.voxel_size
            has_seed = seed >= 0
            label_grid = np.full(self.grid.shape, -1, np.int32)
            if has_seed.any():
                vs_ = cfg.voxel_size
                near, where = _nd.distance_transform_edt(
                    ~has_seed, sampling=(vs_, vs_, vs_), return_indices=True)
                snapped = np.where(near <= snap, seed[tuple(where)], -1).astype(np.int32)
                # 라벨은 **표면**에만 붙인다. 빈 공간의 복셀이 라벨을 들고 있으면 최근접 표면의
                # 라벨을 읽는다는 약속이 깨진다.
                label_grid = np.where(occupancy == OCCUPIED, snapped, -1).astype(np.int32)

        occ_now = occupied_mask(occupancy, cfg.unknown_policy)
        subbox = None
        labels_changed = label_names != getattr(self, "_label_names", ())
        self._label_names = label_names
        if labels_changed:
            self._labels = None
        if (cfg.incremental and not labels_changed
                and self._field is not None and self._occupancy is not None):
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

        field_grid, label_field, stats = esdf_from_occupancy(
            occupancy, self.grid, max_distance=cfg.max_distance,
            unknown_policy=cfg.unknown_policy, subbox=subbox, previous=self._field,
            labels=label_grid, previous_labels=getattr(self, "_labels", None))
        self._field = field_grid
        self._labels = label_field
        self._frames += 1
        stats.update({
            "n_blocks": (0 if subbox is None else len(subbox)),
            "voxel_size": cfg.voxel_size,
            "grid_shape": tuple(self.grid.shape),
            "n_voxels": self.grid.n_voxels,
            "frames": self._frames,
            "n_target_voxels_carved": n_carved,
            "n_support_voxels_carved": n_support_carved,
            "n_attached_voxels_carved": n_attached_carved,
            "n_occupancy_changed": int(getattr(self, "_n_changed", 0)),
            **static_stats,
            "decay": decay_stats,
            "n_labels": len(label_names),
            "label_names": list(label_names),
            "per_camera": per_camera,
        })
        # 격자 밖은 격자 안의 UNKNOWN 과 같은 것이므로 같은 정책을 따른다 — `unknown_policy`
        # 와 무관하게 항상 자유였던 것이 E4 였다.
        outside = cfg.max_distance if cfg.unknown_policy == "free" else -cfg.max_distance
        out = EsdfField(grid=self.grid, distance_grid=field_grid,
                        max_distance=cfg.max_distance, stats=stats,
                        outside_distance=outside,
                        label_grid=label_field, label_names=label_names,
                        static_shapes=tuple(static_geometry or ()))
        # 출처는 두 backend 가 **같은 모양으로** 찍는다. 다르면 기록을 견줄 수 없다.
        from benchmark.ag3s.fields.provenance import FieldProvenance
        out.provenance = FieldProvenance(
            sequence=self._frames, backend="legacy",
            observed_at=observed_at, built_at=time.monotonic(),
            frame_id=str(frame_id), frame_index=int(frame_index),
            cameras=tuple(str(c.name) for c in cameras),
            tiers=({"voxel_size_m": float(cfg.voxel_size),
                    "shape": [int(v) for v in self.grid.shape],
                    "origin_m": [float(v) for v in self.grid.origin]},))
        return out


def build_once(cameras: Sequence[CameraDepth], config, *, bounds=None,
               target_points: Optional[np.ndarray] = None,
               exclude_target: bool = False) -> EsdfField:
    """일회성 빌드. 프레임 간 재사용이 없으므로 증분 갱신도 없다."""
    return EsdfBuilder(config, bounds=bounds).update(
        cameras, target_points=target_points, exclude_target=exclude_target)


__all__ += ["CameraDepth", "EsdfBuilder", "build_once", "default_bounds"]
