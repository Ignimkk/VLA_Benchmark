"""cuRobo `VoxelGrid` → trajopt 가 요구하는 거리장.

trajopt 의 소비부가 필드에 요구하는 것은 **세 가지뿐**이다 (`trajopt/linearize.py`):

    field.distance(points)   # (N,3) -> (N,)    `_esdf_clearance:350`
    field.gradient(points)   # (N,3) -> (N,3)   `linearize:533` 의 분리 방향
    field.grid.voxel_size    # E1 의 target 판정 허용오차 (`_esdf_clearance:356`)

그래서 이 모듈은 **백엔드 교체 한 겹**이다. `docs/AG3S_REVIEW_LOG.md` 의 모듈 처분표가 적은
대로 배선(`pipeline.py`)과 정책(E1 접촉 권한, G1~G4 검증 플래그)은 그대로 두고 필드 구현만
바꾼다.

**질의는 cuRobo 를 필요로 하지 않는다.** cuRobo 는 거리 격자를 *생산*할 뿐이고, 그 격자는
`(값, origin, voxel_size)` 세 개로 완전히 기술된다. 그래서 여기 들어오는 것은 numpy 배열이고,
cuRobo 가 설치되지 않은 프로세스(우리 `.venv-ag3s`)에서도 그대로 돌아간다 — 두 venv 가
numpy 버전 때문에 갈라져 있으므로 이것이 실용적으로 중요하다.

**계층 합성은 `min()` 이다.** 거친 계층은 작업공간 전체를 덮고, 미세 계층은 관심 영역만 더
정밀하게 덮는다. 같은 TSDF 에서 뽑은 두 추출이므로 둘 다 같은 표면을 재고, 차이는 이산화
편향뿐이다. `min()` 이 옳은 이유는 두 가지다.

1. **편향의 부호가 한쪽이다.** 표면은 점유 복셀의 *중심*에 표시되므로 거친 격자일수록 거리가
   과대평가될 수 있고, 과소평가되지는 않는다. 작은 쪽을 고르는 것이 곧 정확한 쪽을 고르는 것이다.
2. **미세 계층의 창 밖 표면은 시드가 없다.** 창 가장자리 근처에서 미세 계층은 "가까운 표면이
   없다"고 답한다 (`AG3S_REVIEW_PLAN.md` 의 "ROI 패딩 ≥ max_distance" 주의). 그 자리에서
   거친 계층은 진짜 거리를 알고 있고, `min()` 이 그것을 집는다. 즉 `min()` 이 그 함정을
   구조적으로 막는다.

기울기는 **거리를 내놓은 그 계층**에서 가져온다. 다른 계층의 기울기를 섞으면 값과 기울기가
어긋나고, 그러면 SQP 가 수렴하지 않는다 (`esdf.py:EsdfField.gradient` 가 같은 이유를 적어 둔다).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

import numpy as np

from benchmark.ag3s.esdf import EsdfField, VoxelGrid

__all__ = ["layer_from_curobo", "layer_from_arrays", "CuroboEsdfField", "RolloutFields"]


def layer_from_arrays(distance_grid: np.ndarray, origin: np.ndarray, voxel_size: float,
                      *, outside_distance: Optional[float] = None,
                      stats: Optional[dict] = None) -> EsdfField:
    """`(값 격자, 첫 복셀 중심, 복셀 크기)` → `EsdfField` 한 계층.

    계층 하나하나는 **기존 `EsdfField` 를 그대로 쓴다.** 삼선형 보간과 중심차분 기울기를 다시
    구현하지 않는 것이 요점이다 — 그 둘이 서로 어긋나지 않는다는 성질이 SQP 수렴의 전제이고,
    이미 검증된 구현이 하나 있는데 두 벌로 만들면 그 성질을 두 번 지켜야 한다.

    `origin` 은 `(0,0,0)` 복셀의 **중심**이다. cuRobo 의 `create_xyzr_tensor()` 가 주는 첫 점과
    같은 규약이고 (`pose[:3] - dims/2 + voxel_size/2`), 우리 `VoxelGrid.origin` 의 정의이기도
    하다. 둘이 같기 때문에 좌표 변환이 필요 없다.
    """
    d = np.asarray(distance_grid, np.float32)
    if d.ndim != 3:
        raise ValueError(f"거리 격자는 3차원이어야 합니다: {d.shape}")
    grid = VoxelGrid(origin=np.asarray(origin, np.float64).reshape(3),
                     shape=tuple(int(v) for v in d.shape), voxel_size=float(voxel_size))
    finite = d[np.isfinite(d)]
    return EsdfField(grid=grid, distance_grid=d,
                     max_distance=float(np.max(np.abs(finite))) if finite.size else 0.0,
                     stats=dict(stats or {}), outside_distance=outside_distance)


def layer_from_curobo(voxel_grid: Any, *, outside_distance: Optional[float] = None,
                      stats: Optional[dict] = None) -> EsdfField:
    """cuRobo `VoxelGrid` (torch) → `EsdfField` 한 계층. cuRobo 가 있는 프로세스에서만.

    **인덱싱을 손으로 쓰지 않는다.** `feature_tensor` 는 `(nx, ny, nz)`, x 가 가장 느리고 z 가
    가장 빠르다 (`integrator_esdf.py` 의 `esdf_grid_shape` 주석). `create_xyzr_tensor()` 의
    `reshape(-1)` 순서와 같으므로 격자를 그대로 옮기면 되고, `origin` 만 pose 에서 계산한다 —
    `docs/AG3S_REVIEW_LOG.md` 의 cuRobo 함정 목록 1·2 가 정확히 이 지점이다.

    **부호 규약은 우리와 같다** — 음수가 표면 안쪽. cuRobo 커널이
    `if tsdf_sdf < 0.0: edt_dist = -edt_dist` 로 정하고
    (`kernel/builder/builder_esdf.py`), 실측으로도 자유공간 `+0.326 m`, 테이블 상판 `+0.000 m`
    였다. 그래서 부호를 뒤집지 않는다.

    주의: `VoxelGrid.get_occupied_voxels()` 는 `feature > -0.5*voxel_size` 를 점유로 보아
    **반대 규약**을 쓴다. 이 필드에 그것을 쓰면 자유공간이 나온다 — 쓰지 말 것.
    """
    feature = voxel_grid.feature_tensor
    if feature is None:
        raise ValueError("VoxelGrid.feature_tensor 가 비어 있습니다 — compute_esdf() 를 먼저 부르세요")
    vs = float(voxel_grid.voxel_size)
    dims = np.asarray(voxel_grid.dims, np.float64).reshape(3)
    pose = np.asarray(voxel_grid.pose, np.float64).reshape(-1)[:3]
    # `esdf_origin` 은 격자 코너가 아니라 **중심**이다 (`integrator_esdf.py` 의 "Pose at center").
    origin = pose - 0.5 * dims + 0.5 * vs
    d = feature.detach().float().cpu().numpy()
    return layer_from_arrays(d, origin, vs, outside_distance=outside_distance, stats=stats)


@dataclasses.dataclass
class CuroboEsdfField:
    """해상도가 다른 계층 여럿을 `min()` 으로 합쳐 하나의 거리장처럼 보이게 한다.

    `layers` 는 **거친 것부터** 준다. 첫 계층이 작업공간 전체를 덮어야 하고, 나머지는 그 위에
    얹는 국소 계층이다. 계층이 하나뿐이면 그 계층과 동작이 같다.

    `grid` 는 **가장 미세한 계층**의 격자를 내놓는다. 이 속성을 읽는 곳은 두 군데뿐이고 둘 다
    그 선택이 맞다.

    * `_esdf_clearance:356` 의 `tol = grid.voxel_size` — "이 구의 최근접 장애물이 곧 자기 링크가
      만져도 되는 target 인가" 를 판정하는 허용오차다. 크게 잡으면 **target 이 아닌 것 옆에 있는
      구까지** 완화된 target 마진을 받으므로 안전하지 않은 쪽으로 틀린다. 미세 계층은 target 을
      중심으로 놓이므로 target 근처 구는 어차피 미세 계층이 답하고, 멀리 있는 구는 `d_target` 이
      커서 허용오차와 무관하게 판정이 뒤집히지 않는다. 즉 가장 작은 값이 안전하면서 동시에
      기능적으로도 맞다.
    * `_esdf_coverage` (G4) 의 `grid.origin` / `grid.upper` — 여기서는 **거친 계층**의 범위가
      맞다. 그래서 그 경로를 위해 `coverage_grid` 를 따로 둔다.
    """

    layers: tuple[EsdfField, ...]
    #: 격자 **전부**의 밖에서 답할 값. `None` 이면 거친 계층의 가장자리 값을 그대로 쓴다.
    #: E4 가 이 자리에서 났다 — 격자 밖을 조용히 "자유" 로 답하면 관측 실패가 안전으로 둔갑한다.
    #: 그래서 기본값을 두지 않고 호출자가 정하게 한다.
    outside_distance: Optional[float] = None
    stats: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.layers = tuple(self.layers)
        if not self.layers:
            raise ValueError("계층이 최소 하나는 있어야 합니다")
        sizes = [f.grid.voxel_size for f in self.layers]
        if sizes != sorted(sizes, reverse=True):
            raise ValueError(f"계층은 거친 것부터 주어야 합니다 (복셀 크기 내림차순): {sizes}")
        # 바깥 계층의 격자 밖 처리는 이 합성 필드가 맡는다. 국소 계층은 자기 창 밖에서
        # 답하지 않아야 하므로 `outside_distance` 를 주지 않는다.
        self.layers[0].outside_distance = self.outside_distance

    # -- trajopt 가 읽는 표면 -------------------------------------------------

    @property
    def grid(self) -> VoxelGrid:
        """가장 미세한 계층의 격자. `voxel_size` 의 선택 이유는 클래스 docstring 참고."""
        return self.layers[-1].grid

    @property
    def coverage_grid(self) -> VoxelGrid:
        """작업공간을 덮는 격자 = 거친 계층. `_esdf_coverage` (G4) 가 봐야 하는 쪽."""
        return self.layers[0].grid

    def distance(self, points: np.ndarray) -> np.ndarray:
        """`(N,)` 미터. 음수는 표면 안쪽."""
        return self._evaluate(points)[0]

    def gradient(self, points: np.ndarray) -> np.ndarray:
        """`(N, 3)`. **거리를 내놓은 계층**의 기울기 — 값과 어긋나면 SQP 가 수렴하지 않는다."""
        _, winner, pts = self._evaluate(points, want_winner=True)
        out = np.zeros((len(pts), 3), np.float64)
        for i, field in enumerate(self.layers):
            sel = winner == i
            if sel.any():
                out[sel] = field.gradient(pts[sel])
        return out

    # -- 합성 ------------------------------------------------------------------

    def _evaluate(self, points: np.ndarray, *, want_winner: bool = False):
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        d = np.asarray(self.layers[0].distance(pts), np.float64)
        winner = np.zeros(len(pts), np.int64)
        for i, field in enumerate(self.layers[1:], start=1):
            inside = self._covers(field.grid, pts)
            if not inside.any():
                continue
            local = np.asarray(field.distance(pts[inside]), np.float64)
            cur = d[inside]
            take = local < cur
            cur[take] = local[take]
            d[inside] = cur
            idx = np.flatnonzero(inside)[take]
            winner[idx] = i
        if want_winner:
            return d, winner, pts
        return d, winner

    @staticmethod
    def _covers(grid: VoxelGrid, pts: np.ndarray) -> np.ndarray:
        """`(N,)` — 그 점이 이 계층의 보간 격자 안에 있는가.

        `EsdfField._lattice` 와 **같은 경계**를 쓴다: 보간이 성립하는 범위는 첫 복셀 중심부터
        마지막 복셀 **중심**까지이지 `grid.upper`(마지막 복셀의 바깥면)까지가 아니다. 격자 밖
        점은 `_lattice` 가 가장자리로 클램프하므로, 국소 계층에 그 범위 밖을 묻지 않는 것이
        합성의 전제다.
        """
        last = grid.origin + (np.asarray(grid.shape) - 1) * grid.voxel_size
        return np.all((pts >= grid.origin) & (pts <= last), axis=1)

    # -- 진단 (G2/G4 가 보는 것) -----------------------------------------------

    @property
    def unknown_fraction(self) -> float:
        return float(self.stats.get("unknown_fraction", 0.0))

    @property
    def outside_query_fraction(self) -> float:
        """거친 계층 기준. 이 필드가 얼마나 자주 작업공간 밖에서 질의됐는가."""
        return self.layers[0].outside_query_fraction

    def layer_report(self) -> str:
        rows = []
        for i, f in enumerate(self.layers):
            g = f.grid
            rows.append(f"  [{i}] {g.voxel_size*1000:5.1f} mm  {g.shape}  "
                        f"중심 {np.round(g.origin + 0.5*(np.asarray(g.shape)-1)*g.voxel_size, 3)}  "
                        f"범위 {np.round(g.upper - g.origin, 2)} m")
        return "\n".join(rows)


@dataclasses.dataclass
class RolloutFields:
    """롤아웃 프레임마다 미리 만들어 둔 `CuroboEsdfField` 묶음.

    cuRobo 는 `.venv-curobo` 에, trajopt 는 `.venv-ag3s` 에 있다. 그래서 필드를 **미리 만들어
    npz 로 넘기고** 소비는 검증된 환경에서 한다 (`docs/AG3S_REVIEW_LOG.md` 의 D3). 이 클래스가
    그 npz 를 읽어 `esdf_rollout` 이 프레임마다 꺼내 쓸 수 있게 한다.

    생산자는 `ag3s/experiments/curobo/build_rollout_fields.py`.
    """

    fields: tuple["CuroboEsdfField", ...]

    @property
    def n_frames(self) -> int:
        return len(self.fields)

    @property
    def n_layers(self) -> int:
        return len(self.fields[0].layers) if self.fields else 0

    def field_for(self, i: int) -> "CuroboEsdfField":
        if not 0 <= i < len(self.fields):
            raise IndexError(f"프레임 {i} 의 필드가 없습니다 (총 {len(self.fields)} 개) — "
                             "덤프와 필드가 같은 --frames 로 만들어졌는지 확인하세요")
        return self.fields[i]

    @classmethod
    def load(cls, path: str, *, single_layer: bool = False,
             outside_distance: float = 0.5) -> "RolloutFields":
        """`single_layer=True` 면 거친 계층만 쓴다 — 2계층이 판정을 바꾸는지 보는 대조군."""
        blob = np.load(path)
        n = int(blob["n_frames"])
        out = []
        for i in range(n):
            layers = []
            for name in ("coarse", "fine"):
                key = f"f{i}_{name}_values"
                if key not in blob:
                    continue
                if single_layer and name != "coarse":
                    continue
                layers.append(layer_from_arrays(
                    blob[key], blob[f"f{i}_{name}_origin"],
                    float(blob[f"f{i}_{name}_voxel_size"])))
            stats = {"n_occupied": int(blob[f"f{i}_n_occupied"]),
                     "unknown_fraction": float(blob[f"f{i}_unknown_fraction"])}
            out.append(CuroboEsdfField(tuple(layers), outside_distance=outside_distance,
                                       stats=stats))
        return cls(tuple(out))
