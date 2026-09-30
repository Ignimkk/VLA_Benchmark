"""cuRobo `Mapper` 를 `EsdfBuilder` 와 **같은 서명**으로 감싼다.

    from benchmark.ag3s.fields.curobo_builder import CuroboFieldBuilder
    builder = CuroboFieldBuilder(config)          # config = AG3SConfig.esdf
    field = builder.update(depth_cameras, target_points=..., attached_points=...,
                           labelled_points=..., static_geometry=...)

같은 서명인 것이 요점이다. `pipeline._build_esdf` 는 구현 한 줄만 갈아타고, 배선과 정책은
그대로 남는다 — `AG3S_REVIEW_LOG.md` 의 모듈 처분표가 적은 대로 E1(조작 대상을 필드에서
파내면 손끝뿐 아니라 전신에게 사라진다)의 접촉 권한, G2~G4 의 검증 플래그는 어느 필드
구현을 쓰든 필요하다.

## 무엇을 cuRobo 에 넘기고 무엇을 우리가 들고 있나

| 책임 | 어디서 |
|---|---|
| depth 적분, coarse/fine ESDF | cuRobo `Mapper` (실측 적분 3.8 ms + ESDF 0.44 ms) |
| 잔상 감쇠 (F20 — 사라진 물체의 잔상이 8 프레임 뒤에도 남는다) | cuRobo native (`MapperCfg.decay_factor`/`frustum_decay_factor`). **기본 끔** |
| 라벨 층 | 우리 — `site_index` 조회 (아래) |
| 쥔 물체 (A2) | 우리 — **seed 에서 제외** (아래). 격자를 지우지 않는다 |
| 해석적 정적 기하 (N2) | 우리 — 질의 시점 `min(복셀, 해석적)`. cuRobo `update_static_obstacles` 는 **격자 안만** 찍고, 우리 실측은 중요 기하(선반 y=-1550, 벽 ±3 m)가 격자 밖이라 이득 0.0 mm 였다 |
| 지지면 파내기 | **안 한다** — live 경로는 `exclude_support_surfaces=False` 라 발동하지 않는다. 주어지면 조용히 무시하지 않고 예외를 던진다 |

## 라벨 층은 조회 한 번이다

cuRobo 의 PBA+ 는 복셀마다 **그 거리를 만든 표면 복셀**을 `site_index` 에 남긴다
(dense int32, ESDF 격자와 같은 shape, 포장 `(z<<20)|(y<<10)|x`). 실측으로 확인했다 —
유효 100 %, `|site 까지 기하 거리 - |ESDF||` 중앙 0.10 mm · 최대 0.49 mm (float16 양자화).
그래서 "이 점에서 가장 가까운 표면이 무엇인가" 는 추가 알고리즘이 아니라 조회다.

## 쥔 물체는 지우지 않고 seed 에서 뺀다

`Mapper.clear_region` 은 **파괴적이고 보수적인 AABB** 라 쓰지 않는다. 누적 TSDF 를 지우므로
침식(있는 것을 잊는 것 = 위험한 방향)을 만들고, 게다가 잔상은 물체의 *옛* 자리에 있어 현재
AABB 로는 닿지도 않는다.

대신 seed 와 propagate 사이에 끼어들고, **그 뒤에 부호를 고친다.**

    integrator._seed_esdf_impl(origin, vs)            # site_index 에 표면 복셀을 심는다
    site_index[쥔 물체 복셀] = -1                       # (1) 크기를 다음 표면까지로
    integrator._propagate_and_distance_impl(...)      # PBA 가 남은 표면에서 전파한다
    values[순수 복셀] = abs(values[순수 복셀])           # (2) 부호를 양수로

**(2) 가 없으면 (1) 은 상황을 더 나쁘게 만든다.** cuRobo 커널이 부호를 seed 가 아니라
**질의 복셀 자신의 TSDF** 에서 가져오기 때문이다 (`builder_esdf.py:455-489`):

    tsdf_sdf = lookup_combined_sdf_at_esdf_coords(tsdf, esdf_x, esdf_y, esdf_z, ...)
    if tsdf_sdf < 0.0: edt_dist = -edt_dist

seed 를 지우면 크기만 "다음 표면까지" 로 커지고 부호는 여전히 그 자리에 남아 있는 쥔 물체의
TSDF 가 정한다 — 즉 *더 먼 거리에 음수 부호*가 붙는다. 실측으로 잡혔다: 쥔 물체 질의점
184 개에서 음수가 38 → **59** 개로 늘고 최소가 −12.93 → **−33.07 mm** 가 됐다.

**순수 복셀만 고친다** (2026-09-22 판정 B). 순수 = 쥔 물체 말고는 아무것도 없는 복셀이고,
판정은 추가 데이터 없이 된다 — seed 제외 **뒤**의 거리가 복셀 반 칸보다 크면 그 자리에 다른
표면이 없다는 뜻이다. 반 칸 안이면 무언가 있으므로 부호를 **그대로 둔다**.

그래서 환경 표면과 같은 복셀을 쓰는 자리에서는 관통이 **그대로 보고된다** — 숨길 수 있는 양이
경계값이 아니라 **구조적으로 0** 이다. 대가는 그 복셀들에서 쥔 물체가 자기 자신에게 남는
잔여 행이고, 실측상 운반 구간(들어올림 100 mm 이상)에서 **0** 이며 담기 국면에서만
20 mm 계층 24 개 · 5 mm 계층 17 개까지 나온다.

파내기(legacy `carve`)와의 차이는 **TSDF 나 second-nearest 가 아니다** — `carve` 는 매 프레임
다시 계산되는 occupancy 배열을 건드리므로 그 둘을 이미 만족한다. 차이는 cuRobo 의 부호 경로가
occupancy 를 거치지 않는다는 것뿐이고, 그래서 (2) 가 필요하다.

쥔 물체 자체는 로봇 쪽으로 편입되는데 그것은 **점 기반**이다 (E3 가 `sphere_states` 에
반지름 0 질의점으로 붙였다). cuRobo 의 attached-object(구 근사)를 쓰지 않는 이유는 F19 —
관측 점군에 구 하나를 맞추면 r = 57.4 mm 로 담는 구간 여유가 중앙 13.2 mm 나빠지고 프레임
21 에서 없는 충돌을 보고한다.

**`_site_index` 와 `_seed_esdf_impl` 은 cuRobo integrator 의 private 이다.** commit `78fd485`
에 핀되어 있고, `_assert_curobo_surface()` 가 시작할 때 존재를 확인해 버전이 바뀌면 조용히
다른 동작을 하지 않고 **거기서 죽는다**.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Optional, Sequence

import numpy as np

from benchmark.ag3s.config import EsdfConfig
from benchmark.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays
from benchmark.ag3s.fields.esdf import CameraDepth, VoxelGrid, default_bounds
from benchmark.ag3s.fields.observation import probe_from_cameras
from benchmark.ag3s.fields.provenance import FieldProvenance

__all__ = ["CuroboFieldBuilder", "TargetBall", "ATTACHED_LABEL", "unpack_site_linear",
           "FineWindowRequest", "FineWindow", "place_fine_window", "WINDOW_BASES",
           "WINDOW_PLACEMENTS"]

#: 쥔 물체에 붙이는 내부 라벨. 호출자의 `labelled_points` 이름과 겹치면 안 되므로 밑줄로 감싼다.
ATTACHED_LABEL = "__attached__"


def unpack_site_linear(site_index: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """packed `site_index` -> 그 site 복셀의 **선형 인덱스**. `-1` 은 그대로 `-1`.

    포장은 `(z<<20) | (y<<10) | x` 이고 축마다 10 비트다 (`utils_quantization.pack_site_coords`).
    선형화는 `(x*ny + y)*nz + z` — numpy 의 C 순서이고 `feature_tensor` 의 `(nx, ny, nz)` 와
    같다 (X 가 가장 느리고 Z 가 가장 빠르다는 cuRobo 주석).
    """
    packed = np.asarray(site_index, np.int64)
    nx, ny, nz = (int(v) for v in shape)
    x = packed & 0x3FF
    y = (packed >> 10) & 0x3FF
    z = (packed >> 20) & 0x3FF
    lin = (x * ny + y) * nz + z
    return np.where(packed < 0, -1, lin)


@dataclasses.dataclass(frozen=True)
class TargetBall:
    """`target_free` 계층에서 **기하 범위로** 지울 공. 테이블 면 위만 지운다 (T13).

    **왜 점이 아니라 공인가.** 점으로 지우는 길(`_exclude_seed_points`, 쥔 물체가 쓰는 쪽)은
    `target_free` 에서 **한 voxel도 지우지 못했다.** 실측 (`chunk_00014.npz`, `t=104`):

    | | voxel |
    |---|---|
    | 사과 표면 (중심 45 mm 안) | **2,203** |
    | 두 계층(`esdf` · `esdf_free`)이 다른 voxel | 578 |
    | 그중 사과 45 mm 안 | **0** |

    손끝 질의점에서 두 계층이 `17.319 mm` 로 **같았다.** 원인 셋이 겹쳤다 — grounding 표본점이
    112 개뿐이고(그중 seed 72 개), 제외가 팽창하지 않고(그 결정에는 이유가 있다,
    `_exclude_seed_points` 참고), **거리장이 성긴 제거에 꿈쩍하지 않는다** (2,203 중 72 를 지워도
    지워진 자리마다 5 mm 옆에 이웃 seed 가 살아 있다).

    쥔 물체 경로는 멀쩡하다 — 그때 주는 점들이 물체 **전체**를 덮기 때문이다. 같은 함수를 두
    용도로 쓴 것이 문제였고, 그래서 이름으로 갈랐다.

    `z_min` 은 **지원면 위로만 지우기 위한 것**이다. 사과 밑면은 테이블에 닿아 있어 그 아래까지
    지우면 **테이블에 구멍**이 난다 — 팔이 테이블을 통과하고 아무도 막지 않는다. 그래서 이 값은
    추측하지 않고 AG3S 가 뽑은 평면에서 온다. 못 얻으면 호출자가 **공을 만들지 않는다.**
    """

    centre: np.ndarray
    radius: float
    #: 이 z **이상**만 지운다. 지원면(테이블 상판) + 여유.
    z_min: float
    #: 진단용 — 무엇을 재서 이 값이 나왔나. 기록과 시작 로그에 그대로 실린다.
    provenance: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre",
                           np.asarray(self.centre, np.float64).reshape(3))
        if not float(self.radius) > 0.0:
            raise ValueError(f"TargetBall.radius 는 양수여야 합니다: {self.radius}")

    def summary(self) -> dict:
        return {"centre_m": [float(v) for v in self.centre],
                "radius_mm": round(float(self.radius) * 1000.0, 3),
                "z_min_m": float(self.z_min),
                **{k: v for k, v in self.provenance.items()}}


#: `stats["window"]["basis"]` 의 값. 창이 **무엇을 근거로** 놓였나 (T21).
#:
#: | 값 | 뜻 |
#: |---|---|
#: | `hand_swept` | 권한 link 구의 swept AABB (지금 자세 + 실행 구간) ∪ manipulated AABB 가 창에 들어갔다 |
#: | `manipulated` | 손 정보가 없거나 (`q_now`·모델·권한 link 없음) 옛 규칙(`target_centroid`)이다 — 물체 중심 |
#: | `overflow` | 들어가지 않았다. 지금 손 → 실행 구간 → 물체 순으로 담고 **넘친 것을 기록**했다 |
WINDOW_BASES = ("hand_swept", "manipulated", "overflow")
#: `esdf.fine_window_placement` 의 선택지 — config 의 목록을 그대로 쓴다 (두 곳에 박지 않는다).
WINDOW_PLACEMENTS = EsdfConfig.FINE_WINDOW_PLACEMENTS


@dataclasses.dataclass(frozen=True)
class FineWindowRequest:
    """pipeline → builder: 미세 창을 **어디에** 놓을지 정하는 재료 (T21, 지침 §5.1).

    배치 자체는 builder 가 한다 — 창의 크기(격자 shape × 미세 복셀)를 아는 쪽이 builder 이기 때문이다.
    pipeline 은 "권한 있는 link 의 구가 지금 어디 있고 실행 창 동안 어디를 지나가는가" 와 "제외할
    물체가 어디 있는가" 를 넘긴다. **둘은 다른 결정이다** — 창의 배치는 손이 정하고, 제외 기하
    (`TargetBall`)는 manipulated 물체만 정한다. 창을 넓히는 것과 제외 영역을 넓히는 것을 섞지 않는다
    (지침 §5.3-3).

    | 필드 | 무엇 |
    |---|---|
    | `hand_centres` | `(T, S, 3)` — 권한 link 구 중심. **step 0 = `q_now`**, 그 뒤가 실행 구간 (우선순위 순서) |
    | `hand_radii` | `(S,)` 구 반지름 |
    | `hand_pad` | 구마다 반지름 **밖에** 더할 여유 (m) = 충돌 마진 + 보간·기울기 여유 |
    | `manipulated_points` | 제외할 물체의 점군 (T20 manipulated 기하). 없으면 `None` |
    | `manipulated_pad` | 물체 AABB 에 더할 보간 여유 (m) |
    | `placement` | `hand_swept` (기본) · `target_centroid` (T21 이전 규칙 — 물체 점 평균) |
    | `hand_source` | 기록용 — `q_now` / `q_now+path` / `none` |
    | `links` | 기록용 — 권한 link 이름 |
    """

    hand_centres: Optional[np.ndarray] = None
    hand_radii: Optional[np.ndarray] = None
    hand_pad: float = 0.0
    manipulated_points: Optional[np.ndarray] = None
    manipulated_pad: float = 0.0
    placement: str = "hand_swept"
    hand_source: str = "none"
    links: tuple = ()

    def __post_init__(self) -> None:
        if self.placement not in WINDOW_PLACEMENTS:
            raise ValueError(f"placement 는 {WINDOW_PLACEMENTS} 중 하나여야 합니다: "
                             f"{self.placement!r}")
        hc = self.hand_centres
        if hc is not None:
            hc = np.asarray(hc, np.float64)
            if hc.ndim == 2:
                hc = hc[None]
            if hc.ndim != 3 or hc.shape[-1] != 3:
                raise ValueError(f"hand_centres 는 (T, S, 3) 이어야 합니다: {hc.shape}")
            radii = np.asarray(self.hand_radii if self.hand_radii is not None
                               else np.zeros(hc.shape[1]), np.float64).reshape(-1)
            if radii.shape[0] != hc.shape[1]:
                raise ValueError(f"hand_radii {radii.shape[0]} 개 ≠ 구 {hc.shape[1]} 개")
            if hc.shape[0] == 0 or hc.shape[1] == 0:
                hc, radii = None, None
            object.__setattr__(self, "hand_centres", hc)
            object.__setattr__(self, "hand_radii", radii)
        mp = self.manipulated_points
        if mp is not None:
            mp = np.asarray(mp, np.float64).reshape(-1, 3)
            object.__setattr__(self, "manipulated_points", mp if len(mp) else None)
        if float(self.hand_pad) < 0.0 or float(self.manipulated_pad) < 0.0:
            raise ValueError("pad 는 0 이상이어야 합니다")

    @property
    def has_hand(self) -> bool:
        return self.hand_centres is not None

    def step_boxes(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """step 마다 `(lo, hi)` — 구 중심 ± (반지름 + `hand_pad`). step 0 = 지금 자세."""
        if self.hand_centres is None:
            return []
        grow = (self.hand_radii + float(self.hand_pad))[None, :, None]
        lo = (self.hand_centres - grow).min(axis=1)
        hi = (self.hand_centres + grow).max(axis=1)
        return [(lo[t], hi[t]) for t in range(lo.shape[0])]

    def manipulated_box(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self.manipulated_points is None:
            return None
        p = self.manipulated_points
        return (p.min(axis=0) - float(self.manipulated_pad),
                p.max(axis=0) + float(self.manipulated_pad))


@dataclasses.dataclass(frozen=True)
class FineWindow:
    """놓인 미세 창 한 개. `lower`·`upper` 는 **보간 격자 경계**다 — 첫/마지막 복셀 중심.

    `CuroboEsdfField._covers` 와 같은 경계를 쓴다. 두 곳이 다르면 "창 안" 의 뜻이 갈라진다.
    """

    centre: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    voxel_size: float
    shape: tuple
    basis: str
    placement: str
    #: 넘친 것 — `None` 이면 요구한 것을 다 담았다. `hand_now` · `hand_path` · `manipulated`.
    overflow: Optional[str] = None
    n_steps: int = 0
    #: 패딩한 step AABB 가 창 안에 **통째로** 든 step 수 (앞에서부터 연속일 필요는 없다).
    n_steps_covered: int = 0
    manipulated_covered: Optional[bool] = None
    required_lower: Optional[np.ndarray] = None
    required_upper: Optional[np.ndarray] = None
    hand_source: str = "none"
    links: tuple = ()

    def contains(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, np.float64).reshape(-1, 3)
        return np.all((p >= self.lower) & (p <= self.upper), axis=1)

    def contains_box(self, lo: np.ndarray, hi: np.ndarray) -> bool:
        tol = 1e-9
        return bool(np.all(lo >= self.lower - tol) and np.all(hi <= self.upper + tol))

    def summary(self) -> dict:
        def _v(x):
            return None if x is None else [round(float(v), 6) for v in np.asarray(x).reshape(-1)]
        return {
            "placement": self.placement,
            "basis": self.basis,
            "overflow": self.overflow,
            "centre_m": _v(self.centre),
            "lower_m": _v(self.lower),
            "upper_m": _v(self.upper),
            "extent_m": _v(np.asarray(self.upper) - np.asarray(self.lower)),
            "voxel_size": float(self.voxel_size),
            "shape": [int(v) for v in self.shape],
            "required_lower_m": _v(self.required_lower),
            "required_upper_m": _v(self.required_upper),
            "hand_source": self.hand_source,
            "links": [str(n) for n in self.links],
            "n_steps": int(self.n_steps),
            "n_steps_covered": int(self.n_steps_covered),
            "manipulated_covered": self.manipulated_covered,
        }


def _union(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0))


def _cover_1d(s: float, length: float, lo: float, hi: float) -> float:
    """창 `[s, s+length]` 가 구간 `[lo, hi]` 를 덮는 길이."""
    return max(0.0, min(s + length, hi) - max(s, lo))


def _slide_axis(must_lo: float, must_hi: float, length: float, wants) -> float:
    """`[must_lo, must_hi]` 를 **반드시** 담으면서 `wants` (우선순위 순 구간 목록)를 사전식으로 가장
    많이 덮는 창의 시작점. 덮는 길이는 `s` 에 대해 조각별 선형이므로 꺾이는 점만 보면 된다."""
    s_min, s_max = must_hi - length, must_lo
    mid = 0.5 * (must_lo + must_hi) - 0.5 * length
    cands = {s_min, s_max, mid}
    for lo, hi in wants:
        cands.update((lo, hi - length, 0.5 * (lo + hi) - 0.5 * length))
    best, best_key = None, None
    for s in sorted(cands):
        s = min(max(s, s_min), s_max)
        key = tuple(round(_cover_1d(s, length, lo, hi), 12) for lo, hi in wants) \
            + (-abs(s - mid),)
        if best_key is None or key > best_key:
            best, best_key = s, key
    return float(best)


def place_fine_window(request: FineWindowRequest, *, shape: Sequence[int],
                      voxel_size: float) -> Optional[FineWindow]:
    """미세 창 하나를 놓는다 (T21). **규칙은 하나다:**

        창 = 손 swept AABB ∪ manipulated AABB 가 들어가면 그것 (가운데 정렬)
             안 들어가면 **지금 손 → 실행 구간 (step 순) → 물체** 순으로 담고, 넘친 것을 기록

    손 swept AABB 는 권한 link 구마다 `중심 ± (반지름 + hand_pad)` 의 합집합이다. `hand_pad` 는
    충돌 마진 + 보간·기울기 여유(복셀 몇 칸)이고 pipeline 이 config 에서 정한다.

    창의 보간 범위는 축마다 `(n − 1) × voxel` 이다 (첫/마지막 복셀 **중심** 사이 — `_covers` 와 같은
    경계). cuRobo 의 ESDF 격자 shape 이 계층마다 같으므로 미세 창의 크기는 **고정**이다 (128³ ×
    5 mm = 0.635 m 보간 범위) — 크기를 늘리는 대신 **어디에** 놓을지를 정하는 것이 이 함수다.

    손 정보가 없으면 (`q_now`·모델·권한 link 중 하나라도 없으면) 물체 점 평균에 놓는다 —
    `basis="manipulated"`. 물체도 없으면 `None` (미세 계층을 만들지 않는다 — T21 이전과 같다).
    `placement="target_centroid"` 는 T21 이전 규칙 그대로다 (물체 점 평균, 손 무시).
    """
    shape = tuple(int(v) for v in shape)
    vs = float(voxel_size)
    length = (np.asarray(shape, np.float64) - 1.0) * vs
    steps = request.step_boxes()
    manip = request.manipulated_box()
    tol = 1e-9

    def fits(box) -> bool:
        return box is not None and bool(np.all(box[1] - box[0] <= length + tol))

    def make(centre, basis, overflow, required):
        centre = np.asarray(centre, np.float64).reshape(3)
        lower = centre - 0.5 * length
        upper = centre + 0.5 * length
        win = FineWindow(centre=centre, lower=lower, upper=upper, voxel_size=vs, shape=shape,
                         basis=basis, placement=request.placement, overflow=overflow,
                         n_steps=len(steps),
                         required_lower=None if required is None else required[0],
                         required_upper=None if required is None else required[1],
                         hand_source=request.hand_source, links=tuple(request.links))
        covered = sum(1 for lo, hi in steps if win.contains_box(lo, hi))
        m_cov = None if manip is None else win.contains_box(*manip)
        object.__setattr__(win, "n_steps_covered", int(covered))
        object.__setattr__(win, "manipulated_covered", m_cov)
        return win

    if request.placement == "target_centroid" or not steps:
        if request.manipulated_points is None:
            return None
        return make(request.manipulated_points.mean(axis=0), "manipulated", None, manip)

    hand_all = _union(steps)
    required = _union([hand_all, manip])
    if fits(required):
        return make(0.5 * (required[0] + required[1]), "hand_swept", None, required)

    # 넘친다. 지금 손 → 실행 구간을 step 순으로, 들어가는 데까지 담는다.
    k = 0
    for t in range(1, len(steps) + 1):
        if fits(_union(steps[:t])):
            k = t
        else:
            break
    if k == 0:
        # 지금 자세의 손 하나도 안 들어간다 — 손 가운데에 놓고 넘친 것을 기록한다.
        lo, hi = steps[0]
        return make(0.5 * (lo + hi), "overflow", "hand_now", required)
    must = _union(steps[:k])
    overflow = "hand_path" if k < len(steps) else "manipulated"
    wants = [hand_all] + ([manip] if manip is not None else [])
    start = np.array([
        _slide_axis(float(must[0][i]), float(must[1][i]), float(length[i]),
                    [(float(w[0][i]), float(w[1][i])) for w in wants])
        for i in range(3)])
    return make(start + 0.5 * length, "overflow", overflow, required)


@dataclasses.dataclass
class _Tier:
    """한 계층의 산출물. 전부 **복사된** 배열이다 (함정 4)."""
    name: str
    values: np.ndarray
    site_linear: np.ndarray
    origin: np.ndarray
    voxel_size: float
    extra: dict = dataclasses.field(default_factory=dict)


class CuroboFieldBuilder:
    """프레임마다 cuRobo 로 적분하고 2계층 ESDF 를 낸다. `EsdfBuilder` 와 교체 가능하다.

    같은 인스턴스를 프레임마다 재사용해야 누적 TSDF 가 의미를 갖는다 — `EsdfBuilder` 가
    격자를 한 번만 할당하는 것과 같은 이유다.
    """

    def __init__(self, config, *, bounds=None, image_hw: Optional[tuple] = None):
        self.config = config
        self._mapper = None
        #: T32 H4-fix: a second `Mapper` whose TSDF uses the fine truncation
        #: (`EsdfConfig.fine_truncation`) — the fine and target-free tiers are seeded from it, the
        #: coarse tier from `_mapper` as before. `None` when the two truncations agree (one TSDF).
        self._fine_mapper = None
        self._image_hw = image_hw
        self._frames = 0
        self._bounds = self._resolve_bounds(config, bounds)
        self._label_names: tuple[str, ...] = ()

    # -- 설정 -------------------------------------------------------------------------
    @staticmethod
    def _resolve_bounds(config, bounds):
        if bounds is not None:
            return (np.asarray(bounds[0], float), np.asarray(bounds[1], float))
        if getattr(config, "bounds_lower", None) is not None:
            return (np.asarray(config.bounds_lower, float),
                    np.asarray(config.bounds_upper, float))
        return default_bounds()

    @property
    def coarse_voxel(self) -> float:
        return float(self.config.voxel_size)

    @property
    def fine_voxel(self) -> Optional[float]:
        """미세 계층 복셀. `None`/0 이면 단일 계층으로 돈다."""
        v = getattr(self.config, "fine_voxel_size", None)
        return None if not v else float(v)

    def _assert_curobo_surface(self, integrator) -> None:
        """우리가 의지하는 private 표면이 아직 있는지 확인한다.

        없으면 여기서 죽는다. 조용히 `compute_esdf` 로 되돌아가면 쥔 물체가 필드에 남아
        **어떤 해로도 못 푸는 제약 행**을 만들고(E3), 그것은 로그가 "조용한 실패" 로 부르는
        바로 그 방식이다.
        """
        missing = [n for n in ("_site_index", "_seed_esdf_impl",
                               "_propagate_and_distance_impl", "_esdf_voxel_size",
                               "_last_esdf_origin", "get_voxel_grid")
                   if not hasattr(integrator, n)]
        if missing:
            raise RuntimeError(
                "cuRobo integrator 에서 기대한 표면이 없습니다: " + ", ".join(missing) +
                ". 이 builder 는 seed 와 propagate 사이에 끼어들어 쥔 물체를 seed 에서 빼고 "
                "(A2), site_index 로 라벨 층을 만듭니다. 검증한 commit 은 78fd485 입니다 — "
                "cuRobo 를 올렸다면 그 두 곳을 다시 맞춰야 합니다.")

    # -- Mapper 를 늦게 만든다 (첫 프레임의 depth 크기가 필요하다) ----------------------
    @property
    def coarse_truncation(self) -> float:
        """TSDF truncation (m) of the TSDF the coarse tier is built from — `EsdfConfig.truncation`."""
        return float(self.config.truncation)

    @property
    def fine_truncation(self) -> float:
        """TSDF truncation (m) the fine tiers are built from (T32 H4-fix). Equals the coarse one
        when `esdf.fine_truncation_m` is `None` or there is no fine tier."""
        ft = getattr(self.config, "fine_truncation", None)
        return self.coarse_truncation if ft is None or not self.fine_voxel else float(ft)

    @property
    def separate_fine_tsdf(self) -> bool:
        """Two TSDFs (coarse truncation / fine truncation) instead of one (T32 H4-fix).

        The coarse tier stays on the 60 mm TSDF on purpose: in `T32.h4.verify.json` a shared 30 mm
        TSDF took the coarse band of the one-side-seen crate wall to **0** (chunks 16/20/30/36 px),
        and 20 mm took both walls to 0 — a coarse 20 mm voxel needs the long truncation to keep a
        negative interior at all.
        """
        return bool(self.fine_voxel) and abs(self.fine_truncation - self.coarse_truncation) > 1e-9

    def _ensure_mapper(self, cameras: Sequence[CameraDepth]):
        if self._mapper is not None:
            return self._mapper
        self._mapper = self._make_mapper(cameras, self.coarse_truncation)
        if self.separate_fine_tsdf:
            self._fine_mapper = self._make_mapper(cameras, self.fine_truncation)
        return self._mapper

    def _make_mapper(self, cameras: Sequence[CameraDepth], truncation: float):
        import torch
        from curobo.perception import Mapper, MapperCfg

        cfg = self.config
        lower, upper = self._bounds
        centre = 0.5 * (np.asarray(lower, float) + np.asarray(upper, float))
        extent = np.asarray(upper, float) - np.asarray(lower, float)
        h, w = (self._image_hw if self._image_hw is not None
                else np.asarray(cameras[0].depth).shape[:2])
        dev = getattr(cfg, "device", None) or "cuda:0"
        mcfg = MapperCfg(
            extent_meters_xyz=tuple(float(v) for v in extent),
            voxel_size=float(getattr(cfg, "tsdf_voxel_size", 0.0) or cfg.voxel_size),
            esdf_voxel_size=self.coarse_voxel,
            grid_center=torch.tensor([float(v) for v in centre], device=dev,
                                     dtype=torch.float32),
            truncation_distance=float(truncation),
            depth_minimum_distance=float(cfg.depth_min),
            depth_maximum_distance=float(cfg.depth_max),
            image_height=int(h), image_width=int(w),
            # 감쇠는 **기본 끔**이다 — numpy 경로와 같은 조건에서 두 backend 를 대조하기 위해.
            # 켤 때 값은 F20 에서 실측해 정했다 (a_t 0.99 · a_f 0.8).
            decay_factor=float(getattr(cfg, "time_decay", 1.0)),
            frustum_decay_factor=float(getattr(cfg, "frustum_decay", 1.0)),
            device=dev)
        mapper = Mapper(mcfg)
        self._assert_curobo_surface(mapper.integrator)
        return mapper

    def reset(self) -> None:
        """누적 TSDF 와 라벨 이름을 버린다. T0 이 요구하는 reset 시점 기록의 대상이다."""
        if self._mapper is not None:
            self._mapper.reset()
        if self._fine_mapper is not None:
            self._fine_mapper.reset()
        self._frames = 0
        self._label_names = ()

    # -- 한 프레임 --------------------------------------------------------------------
    def update(self, cameras: Sequence[CameraDepth], *,
               target_points: Optional[np.ndarray] = None,
               exclude_target: bool = False,
               target_free_points: Optional[np.ndarray] = None,
               target_free_label: Optional[str] = None,
               target_free_ball: Optional["TargetBall"] = None,
               support_points: Optional[np.ndarray] = None,
               attached_points: Optional[np.ndarray] = None,
               labelled_points: Optional[dict] = None,
               static_geometry: Optional[Sequence[Any]] = None,
               observed_at: Optional[float] = None,
               frame_id: str = "", frame_index: int = -1,
               fine_window: Optional["FineWindowRequest"] = None,
               held_free_points: Optional[np.ndarray] = None) -> CuroboEsdfField:
        """`EsdfBuilder.update` 와 같은 계약. 돌려주는 것은 `CuroboEsdfField` 다.

        `exclude_target` 는 **지원하지 않는다.** E1 이 그것을 폐기했다 — 필드는 익명이라
        아직 안 쥔 target 을 파내면 손끝뿐 아니라 전신에게 사라진다. 지금 코드는 항상
        `False` 로 부르고, `True` 로 오면 조용히 무시하지 않고 예외를 던진다.

        `target_free_points` 는 그것과 **다른 것**이다. 필드에서 파내는 것이 아니라, 그 점들의
        seed 를 지운 **미세 계층 한 겹을 따로** 만들어 `field.target_free_layers` 에 실어
        보낸다. 본 계층(`layers`)은 한 복셀도 바뀌지 않으므로 권한 없는 질의점은 예전과 같은
        답을 받는다 — 익명으로 파내는 것과 이름으로 되묻는 것의 차이가 정확히 E1 이다.
        `None`(기본)이면 그 계층을 만들지 않고 비용도 0 이다.

        `fine_window` (T21) 는 미세 창(과 target 없는 창)을 **어디에** 놓을지의 재료다 —
        `place_fine_window` 가 권한 link 의 swept AABB 와 manipulated AABB 로 놓는다. `None` 이면
        T21 이전 규칙(`target_points` 평균)이다. 어느 쪽이든 결과는 `stats["window"]` 에 실린다.
        """
        # **인자 검증을 torch 앞에 둔다.** 이 계약은 CUDA 를 필요로 하지 않으므로, torch 가 없는
        # 프로세스(`.venv-ag3s`)의 테스트가 이것을 실제로 받아 볼 수 있어야 한다 (T7b 의
        # `announce_diagnostic_scope` 를 함수로 뺀 것과 같은 이유).
        self._check_target_free_request(target_free_points, target_points)

        import torch

        if exclude_target:
            raise ValueError(
                "exclude_target 은 E1(조작 대상을 필드에서 파내면 손끝뿐 아니라 전신에게 "
                "사라진다)로 폐기됐습니다. 접촉 권한은 to_adapter 의 manipulated_link_margin "
                "이 질의 쪽에서 처리합니다")
        if support_points is not None and len(support_points):
            raise NotImplementedError(
                "cuRobo backend 는 지지면 파내기를 하지 않습니다. live 경로는 "
                "exclude_support_surfaces=False 라 이 인자가 오지 않습니다. 필요해지면 "
                "점군의 uv 를 써서 depth 영상 공간에서 미리 마스킹하십시오 (로봇 self-filter "
                "와 같은 기전). 조용히 무시하면 지지면이 필드에 남아 평면 행과 다른 여유거리를 "
                "요구하게 됩니다 (F15)")
        if not cameras:
            # `_build_esdf` 가 이 경우를 먼저 걸러 노트를 남긴다. 여기까지 왔다면 호출부가
            # 바뀐 것이므로 조용히 빈 필드를 내지 않고 죽는다.
            raise ValueError("cuRobo backend 에 카메라 depth 가 하나도 오지 않았습니다")

        cfg = self.config
        mapper = self._ensure_mapper(cameras)
        itg = mapper.integrator
        dev = mapper._device if hasattr(mapper, "_device") else "cuda:0"

        per_camera = self._integrate(mapper, cameras, dev)
        # T32 H4-fix: the fine tiers read their own TSDF (fine truncation), integrated from the
        # same depth. `None` = one TSDF for all tiers (the pre-T32 build).
        itg_fine = None
        if self._fine_mapper is not None:
            self._integrate(self._fine_mapper, cameras, dev)
            itg_fine = self._fine_mapper.integrator
        # T32 H3: the held object's old traces are set **free** in the TSDF(s) — after this frame's
        # integration, before the tiers are seeded (see `_free_tsdf_points`).
        freed = {}
        if held_free_points is not None and len(held_free_points):
            fp = np.asarray(held_free_points, np.float64).reshape(-1, 3)
            freed["coarse_tsdf"] = self._free_tsdf_points(itg, fp, self.coarse_truncation)
            if itg_fine is not None:
                freed["fine_tsdf"] = self._free_tsdf_points(itg_fine, fp, self.fine_truncation)

        # 미세 계층 창의 배치 (T21, 지침 §5.1). **창의 배치는 손이 정하고, 제외할 물체는
        # manipulated 가 정한다** — 둘을 한 점(attention centroid)에 묶지 않는다. T14 seq 12–21 에서
        # 창이 target 을 따라 crate 로 가자 손가락 질의점이 창 안에 든 chunk 가 0/8 이었고, T17
        # 접근 구간의 QP 행은 전부 20 mm coarse 계층이 답했다 (T24).
        #
        # `fine_window` 가 없으면 T21 이전 규칙 — target 점 평균 — 이다 (직접 호출자, legacy 비교).
        tgt = (None if target_points is None
               else np.asarray(target_points, np.float64).reshape(-1, 3))
        request = fine_window if fine_window is not None else FineWindowRequest(
            manipulated_points=tgt, placement="target_centroid")
        window = None
        if self.fine_voxel:
            # 격자 shape 은 cuRobo 가 정한다 (`esdf_grid_shape`, 계층마다 같다) — 그래서 창의
            # **크기**는 고정이고 이 함수가 정하는 것은 위치뿐이다.
            window = place_fine_window(request, shape=tuple(int(v) for v in itg._site_index.shape),
                                       voxel_size=self.fine_voxel)
        self._fine_window = window
        self._fine_centre = None if window is None else window.centre

        # 라벨 씨앗. 쥔 물체는 내부 라벨로 함께 들고 간다 — seed 제외에도, 진단에도 쓴다.
        seeds: dict[str, np.ndarray] = {}
        if labelled_points:
            for name, pts in labelled_points.items():
                p = np.asarray(pts, np.float64).reshape(-1, 3)
                if len(p):
                    seeds[str(name)] = p
        attached = (None if attached_points is None
                    else np.asarray(attached_points, np.float64).reshape(-1, 3))
        if attached is not None and len(attached):
            seeds[ATTACHED_LABEL] = attached
        label_names = tuple(seeds)

        tiers = self._build_tiers(itg, attached, dev, torch, itg_fine=itg_fine)

        # **target 없는 미세 계층.** 본 계층을 다 만든 **뒤에** 따로 만든다 — 같은 버퍼
        # (`_site_index`, `feature_tensor`)를 재사용하므로 (함정 4) 사이에 끼면 본 계층의 값이
        # 바뀐다. 미세 계층만 만드는 이유는 비용이다 (권한 있는 link 은 손 근처에 있다).
        free_tiers: list[_Tier] = []
        free_ms = 0.0
        free_points = (None if target_free_points is None
                       else np.asarray(target_free_points, np.float64).reshape(-1, 3))
        if free_points is not None and len(free_points):
            # 위의 `_check_target_free_request` 가 미세 계층을 만들 수 있음을 이미 보장한다 —
            # 조용히 0 겹으로 지나가는 길이 없다.
            t0 = time.monotonic()
            # **공이 오면 점은 안 쓴다** (T13). 점으로는 한 voxel도 안 지워졌다 (`TargetBall`).
            # 쥔 물체 점은 공과 **함께** 지운다 — 그쪽은 점이 물체 전체를 덮으므로 맞는 길이다.
            held = (None if attached is None or not len(attached)
                    else np.asarray(attached, np.float64).reshape(-1, 3))
            remove = held if target_free_ball is not None else (
                free_points if held is None else np.vstack([held, free_points]))
            free_tiers = self._build_tiers(itg, remove, dev, torch, only="fine",
                                           tier_name="fine_no_target",
                                           ball=target_free_ball, itg_fine=itg_fine)
            free_ms = (time.monotonic() - t0) * 1e3
            self._announce_target_free_cost(free_ms, len(free_points), free_tiers,
                                            ball=target_free_ball)

        layers = []
        label_grids = []
        for tier in tiers:
            grid = VoxelGrid(origin=tier.origin,
                             shape=tuple(int(v) for v in tier.values.shape),
                             voxel_size=tier.voxel_size)
            label_grid = self._labels_from_sites(tier, grid, seeds, label_names)
            label_grids.append(label_grid)
            layer = layer_from_arrays(tier.values, tier.origin, tier.voxel_size)
            layer.label_grid = label_grid
            layer.label_names = label_names
            layers.append(layer)
        # target 없는 계층에는 **라벨을 달지 않는다.** 라벨은 "가장 가까운 표면이 무엇인가" 인데
        # 이 계층에는 그 판정의 주인공(target)이 없다. 라벨이 필요한 질문은 본 계층이 답한다.
        free_layers = tuple(layer_from_arrays(t.values, t.origin, t.voxel_size)
                            for t in free_tiers)

        # 격자 밖은 격자 안의 UNKNOWN 과 같은 것이므로 같은 정책을 따른다 (E4).
        outside = (cfg.max_distance if cfg.unknown_policy == "free"
                   else -cfg.max_distance)
        self._frames += 1
        self._label_names = label_names

        # **관측 판정은 필드와 함께 나간다** (2026-09-22 판정). block-sparse TSDF 가 답할 수
        # 없는 "이 부피를 봤는가" 를 원본 depth 로 직접 재고, 인증은 전역 비율이 아니라
        # **로봇 구와 쥔 물체가 실제로 지나가는 곳**의 관측 여부로 한다.
        probe = probe_from_cameras(cameras, depth_min=cfg.depth_min,
                                   depth_max=cfg.depth_max,
                                   truncation=float(cfg.truncation))

        stats = self._stats(tiers, label_names, per_camera,
                            n_attached=0 if attached is None else int(len(attached)),
                            n_static=len(static_geometry or ()))
        # T32 H3: TSDF voxels set free this frame (held-object traces), per TSDF.
        stats["held_free"] = {"n_points": (0 if held_free_points is None
                                           else int(len(held_free_points))),
                              "n_voxels_freed": freed}
        # **창이 어디에, 왜 놓였나** (T21 §3). 미세 계층이 없으면 `basis=None` 으로 싣는다 —
        # 키가 프레임마다 있어야 "창이 없었다" 와 "기록이 없다" 가 구별된다. 권한 link 질의점의
        # 계층별 수는 필드가 다 만들어진 뒤 pipeline 이 더한다 (`authorized_query`).
        stats["window"] = (window.summary() if window is not None
                           else {"placement": request.placement, "basis": None,
                                 "overflow": None, "hand_source": request.hand_source})
        # **target 없는 계층은 기록에 남는다.** 그 계층이 있었는지 없었는지 모르는 기록은
        # 정책이 실제로 걸린 프레임인지 알 방법이 없다 — 키를 정책이 꺼져 있을 때도 싣는다.
        stats["target_free"] = {
            "n_layers": len(free_tiers),
            "n_target_points": 0 if free_points is None else int(len(free_points)),
            "build_ms": round(float(free_ms), 3),
            "label": target_free_label,
            # **무엇으로 지웠는지 기록에 남긴다** (T13). `"points"` 로 남은 프레임은 사과가
            # 안 지워진 프레임이다 — 그 구별이 없으면 두 실행을 나란히 읽을 수 없다.
            "mode": ("ball" if target_free_ball is not None
                     else ("points" if free_points is not None and len(free_points) else "none")),
            "ball": None if target_free_ball is None else target_free_ball.summary(),
            "n_ball_seeds_excluded": sum(
                int(t.extra.get("n_ball_seeds_excluded", 0)) for t in free_tiers),
            "tiers": [{"name": t.name, "voxel_size": t.voxel_size,
                       "shape": [int(v) for v in t.values.shape],
                       "n_seeds_excluded": int(t.extra.get("n_attached_seeds_excluded", 0))}
                      for t in free_tiers],
        }
        if probe is not None:
            lower, upper = self._bounds
            # 정보용이다 — 인증에 쓰지 않는다. `stride` 를 함께 싣는 것이 추정값임을 밝히는 것.
            stats["frustum_observation"] = probe.observed_volume(
                lower, upper, voxel_size=self.coarse_voxel, stride=2)
        field = CuroboEsdfField(tuple(layers), outside_distance=outside, stats=stats,
                                target_free_layers=free_layers,
                                target_label=(target_free_label if free_layers else None))
        field.label_names = label_names
        field.static_shapes = tuple(static_geometry or ())
        field.observation = probe
        field.attached_query_points = (None if attached is None or not len(attached)
                                       else attached)
        # 출처는 **필드와 함께 나간다.** 거리값만 보면 이것이 방금 만들어진 것인지 낡은
        # 것인지 알 방법이 없다 — 2026-09-18 의 조용한 정지가 그래서 14 프레임 동안 안 보였다.
        field.provenance = FieldProvenance(
            sequence=self._frames, backend="curobo",
            observed_at=observed_at, built_at=time.monotonic(),
            frame_id=str(frame_id), frame_index=int(frame_index),
            cameras=tuple(str(c.name) for c in cameras),
            tiers=tuple({"voxel_size_m": t.voxel_size,
                         "shape": [int(v) for v in t.values.shape],
                         "origin_m": [float(v) for v in t.origin]} for t in tiers))
        return field

    # -- 조각들 ------------------------------------------------------------------------
    def _check_target_free_request(self, target_free_points, target_points) -> None:
        """target 없는 계층을 **실제로** 만들 수 있는지 미리 본다. 못 만들면 여기서 죽는다.

        조용히 0 겹으로 지나가면 사과를 뺐다고 믿은 채 예전과 같은 필드로 돌고, 그 실행은
        기록만 보면 정책이 켜진 실행과 구별되지 않는다.
        """
        if target_free_points is None:
            return
        free = np.asarray(target_free_points, np.float64).reshape(-1, 3)
        if not len(free):
            return
        if not self.fine_voxel:
            raise ValueError(
                "target_free_points 가 왔는데 미세 계층이 없습니다 "
                f"(esdf.fine_voxel_size={getattr(self.config, 'fine_voxel_size', None)!r}). "
                "target 없는 계층은 미세 계층 한 겹으로 만들어지므로, 이 설정에서는 정책이 "
                "아무 일도 하지 않습니다 — fine_voxel_size 를 주거나 "
                "constraint.target_field_policy 를 'relax' 로 두십시오")
        has_target = (target_points is not None
                      and len(np.asarray(target_points, np.float64).reshape(-1, 3)) > 0)
        if not has_target:
            raise ValueError(
                "target_free_points 가 왔는데 target_points 가 비었습니다. target 없는 계층은 "
                "manipulated 물체를 빼는 계층이므로 그 물체가 없으면 뺄 것이 없습니다 — 두 인자는 "
                "같은 프레임의 같은 manipulated 기하에서 와야 합니다")

    def _integrate(self, mapper, cameras, dev) -> dict:
        import torch
        from curobo._src.types.camera import CameraObservation
        from curobo._src.types.pose import Pose

        per_camera = {}
        for cam in cameras:
            depth = np.asarray(cam.depth, np.float32)
            if cam.robot_mask is not None:
                # 로봇 픽셀을 0 으로 둔다 = 그 광선은 자유가 아니라 **미관측**이다. raw depth 를
                # 넣으면 로봇이 자기 몸을 장애물로 본다 (C1 — 구 120 중 105 가 자기 충돌).
                depth = np.where(np.asarray(cam.robot_mask, bool), np.float32(0.0), depth)
            h, w = depth.shape
            mapper.integrate(CameraObservation(
                name=str(cam.name),
                depth_image=torch.as_tensor(depth, device=dev, dtype=torch.float32)[None],
                rgb_image=torch.zeros((1, h, w, 3), device=dev, dtype=torch.uint8),
                intrinsics=torch.as_tensor(np.asarray(cam.camera_intrinsics, np.float32),
                                           device=dev, dtype=torch.float32)[None],
                pose=Pose.from_matrix(torch.as_tensor(
                    np.asarray(cam.T_base_cam, np.float32), device=dev,
                    dtype=torch.float32)),
                depth_to_meter=1.0))
            per_camera[str(cam.name)] = self._camera_contribution(cam, depth)
        return per_camera

    def _camera_contribution(self, cam, depth: np.ndarray) -> dict:
        """G3 가 읽을 `n_updated`. **legacy 와 같은 이름, 다른 정의다.**

        legacy 는 그 카메라가 쓴 복셀 수를 세는데 cuRobo 는 그것을 내놓지 않는다
        (`get_stats()` 의 `active_blocks` 는 *새로* 할당된 블록이라, 두 번째 프레임부터 같은
        부피를 다시 봐도 0 이 되어 살아 있는 카메라를 죽은 것으로 만든다).

        대신 G3 가 **잡으려던 셋**을 직접 센다 — 죽은 드라이버(유효 픽셀 0), depth 범위 밖
        (범위 내 0), 틀린 외부 파라미터(격자 안 0). 마지막 것은 복셀 수보다 오히려 강하다.
        비용을 위해 격자 안 판정은 8 픽셀 간격으로 훑고 그 비율을 곱한다.
        """
        cfg = self.config
        h, w = depth.shape
        valid = depth > 0
        in_range = valid & (depth >= float(cfg.depth_min)) & (depth <= float(cfg.depth_max))
        n_in_range = int(in_range.sum())
        n_in_grid = 0
        if n_in_range:
            step = 8
            sub = in_range[::step, ::step]
            vv, uu = np.nonzero(sub)
            if vv.size:
                z = depth[::step, ::step][vv, uu].astype(np.float64)
                K = np.asarray(cam.camera_intrinsics, np.float64)
                x = (uu * step - K[0, 2]) / K[0, 0] * z
                y = (vv * step - K[1, 2]) / K[1, 1] * z
                cam_pts = np.stack([x, y, z], axis=1)
                T = np.asarray(cam.T_base_cam, np.float64)
                base = cam_pts @ T[:3, :3].T + T[:3, 3]
                lower, upper = self._bounds
                inside = np.all((base >= lower) & (base <= upper), axis=1)
                # 부표본의 '격자 안' 비율을 전체 범위 내 픽셀 수에 곱한다.
                n_in_grid = int(round(float(inside.mean()) * n_in_range))
        return {
            "n_valid_px": int(valid.sum()),
            "n_in_range_px": n_in_range,
            # G3 이 읽는 이름. 0 이면 이 카메라가 이 프레임에 아무것도 기여하지 않았다.
            "n_updated": n_in_grid,
            "n_updated_definition": "back-projected depth pixels inside the workspace bounds "
                                    "(8 px stride, scaled). legacy counts written voxels.",
            "n_masked_px": (0 if cam.robot_mask is None
                            else int(np.asarray(cam.robot_mask, bool).sum())),
        }

    def _tier_specs(self, target_centre: Optional[np.ndarray], *, only: Optional[str] = None,
                    tier_name: Optional[str] = None):
        """어느 계층을 만들 것인가. `only="fine"` 이면 미세 계층 **하나만** 낸다.

        `only` 를 둔 이유는 target 없는 계층의 비용이다 — 거친 계층까지 두 겹 만들면 프레임당
        비용이 두 배가 되고, 권한 있는 link 은 손 근처(=미세 창 안)에 있으므로 얻는 것이 없다.
        """
        if only is None or only == "coarse":
            yield "coarse", None, self.coarse_voxel
        if only == "coarse":
            return
        fine = self.fine_voxel
        if fine and target_centre is not None:
            yield (tier_name or "fine"), np.asarray(target_centre, np.float64), fine

    def _build_tiers(self, itg_coarse, attached, dev, torch, *, only: Optional[str] = None,
                     tier_name: Optional[str] = None,
                     ball: Optional["TargetBall"] = None,
                     itg_fine=None) -> list[_Tier]:
        """계층마다 seed -> (제외) -> propagate -> 복사.

        `Mapper.compute_esdf` 를 부르지 않는다 — 그쪽은 CUDA graph 로 seed·propagate·distance
        를 한 덩어리로 실행해서 사이에 끼어들 자리가 없다. 대신 `compute_esdf` 가 하는 부수
        효과(`_esdf_voxel_size` 와 `_last_esdf_origin` 갱신)를 여기서 같이 한다.

        **제외 경로가 둘이고, 용도가 다르다** (T13 에서 갈랐다).

        | 인자 | 어떻게 지우나 | 누구 |
        |---|---|---|
        | `attached` | 점이 차지한 복셀 (`_exclude_seed_points`) | 쥔 물체 — 점이 물체 전체를 덮는다 |
        | `ball` | 공 안, **테이블 위만** (`_exclude_ball_seeds`) | target — 표본점 112 개로는 못 지웠다 |

        둘을 같은 함수로 쓰다가 `target_free` 가 **한 voxel도 안 지워졌다** (`TargetBall` 의 표).
        `ball` 은 `target_free` 계층에서만 오므로 **주 계층은 한 복셀도 안 바뀐다.**

        부호 교정은 **지운 복셀 집합 전체**에 걸린다 — 두 경로의 인덱스를 합쳐 넘긴다.
        """
        centre = getattr(self, "_fine_centre", None)  # update() 가 정한다
        tiers: list[_Tier] = []
        for name, origin_override, vs in self._tier_specs(centre, only=only,
                                                          tier_name=tier_name):
            # T32 H4-fix: fine tiers from the fine-truncation TSDF when there is one.
            itg = itg_coarse if (name == "coarse" or itg_fine is None) else itg_fine
            itg._esdf_voxel_size.copy_(
                torch.tensor([float(vs)], device=dev, dtype=torch.float32))
            origin = (itg._origin if origin_override is None
                      else torch.as_tensor(origin_override, device=dev, dtype=torch.float32))
            itg._last_esdf_origin.copy_(origin)

            itg._seed_esdf_impl(origin, itg._esdf_voxel_size)

            shape = tuple(int(v) for v in itg._site_index.shape)
            org = self._origin_of(itg, shape, vs)
            n_excluded = 0
            att_idx = None
            if attached is not None and len(attached):
                n_excluded, att_idx = self._exclude_seed_points(
                    itg, attached, org, vs, shape)
            n_ball = 0
            if ball is not None:
                n_ball, ball_idx = self._exclude_ball_seeds(itg, ball, org, vs, shape)
                # 부호 교정은 **지운 복셀 전체**에 걸려야 한다. 한쪽만 넘기면 나머지 복셀의
                # 음수 부호가 그대로 남고, 그것이 `_force_sign_on_pure_voxels` 가 막는 실패다.
                if ball_idx is not None:
                    att_idx = (ball_idx if att_idx is None
                               else np.unique(np.vstack([att_idx, ball_idx]), axis=0))

            itg._propagate_and_distance_impl(origin, itg._esdf_voxel_size)
            vg = itg.get_voxel_grid()
            # 함정 4 — `feature_tensor` 와 `_site_index` 는 **둘 다** 재사용 버퍼다
            # (실측: 두 계층의 site_index 가 다르다). 다음 계층 전에 복사한다.
            values = vg.feature_tensor.detach().float().cpu().numpy().copy()
            site = itg._site_index.detach().cpu().numpy().copy()
            sign = self._force_sign_on_pure_voxels(
                values, att_idx, vs,
                threshold_voxels=float(getattr(
                    self.config, "attached_sign_threshold_voxels", 1.5)))
            tier = _Tier(name=name, values=values,
                         site_linear=unpack_site_linear(site, values.shape),
                         origin=org, voxel_size=float(vs))
            tier.extra["tsdf_truncation_m"] = (self.coarse_truncation if itg is itg_coarse
                                               else self.fine_truncation)
            tier.extra["n_attached_seeds_excluded"] = n_excluded
            tier.extra["n_ball_seeds_excluded"] = n_ball
            if ball is not None:
                tier.extra["ball"] = ball.summary()
            tier.extra.update(sign)
            tiers.append(tier)
        return tiers

    def _announce_target_free_cost(self, ms: float, n_points: int, tiers, *,
                                   ball: Optional["TargetBall"] = None) -> None:
        """**첫 프레임에 비용을 크게 찍는다.** 그 뒤로는 stats 에만 남는다.

        시작 로그에 ms 가 없으면 "한 겹 더 만든다" 가 프레임 예산에 무엇을 했는지 아무도
        모른다 — 그리고 이 계층은 조용히 켜져 있으면 안 되는 종류의 것이다 (안전 계층을 한
        겹 비껴가는 길이므로).
        """
        if getattr(self, "_announced_target_free", False):
            return
        self._announced_target_free = True
        logging.getLogger(__name__).warning(
            "!!! TARGET-FREE ESDF LAYER IS ON (constraint.target_field_policy) !!!\n"
            "    미세 계층 %d 겹을 target 점 %d 개의 seed 를 지우고 한 번 더 만든다 — "
            "첫 프레임 %.2f ms%s.\n"
            "    본 계층(coarse+fine)은 한 복셀도 바뀌지 않는다. 이 계층을 읽을 권한이 있는 "
            "질의점만 사과를 통과할 수 있고, table·crate 는 이 계층에도 그대로 있다.",
            len(tiers), int(n_points), float(ms),
            "" if not tiers else f" ({tiers[0].voxel_size * 1000:.0f} mm 복셀, "
                                 f"seed {tiers[0].extra.get('n_attached_seeds_excluded', 0)} 개 제외)")
        # **무엇으로 지웠는지, 어떤 수를 재서 그렇게 정했는지 크게 적는다** (T13). 점으로 지운
        # 판은 사과를 한 voxel도 못 지웠으므로, 이 한 줄이 두 판을 가르는 유일한 표지다.
        log = logging.getLogger(__name__)
        if ball is None:
            log.warning(
                "    제외 방식 = **점** (표본점 %d 개). T13 실측에서 이 방식은 사과를 한 voxel도 "
                "지우지 못했다 (표면 2,203 중 0). 공(중심+반지름)을 주려면 pipeline 이 "
                "target_free_ball 을 실어야 한다 — 지원면 z 를 못 얻으면 여기로 물러난다.",
                int(n_points))
        else:
            log.warning("    제외 방식 = **공** %s · seed %d 개 제외 (테이블 면 z 위만).",
                        ball.summary(),
                        sum(int(t.extra.get("n_ball_seeds_excluded", 0)) for t in tiers))

    @staticmethod
    def _free_tsdf_points(itg, points: np.ndarray, truncation: float) -> int:
        """Set the TSDF voxels containing `points` to **free** (`+truncation`, weight kept). T32 H3.

        Why: once the held object's pixels are masked (H1) nothing new of it is integrated, but what
        was integrated **before** attach stays — its place on the table and the first lifted frames
        (measured: E3b ep1807 r1, the table spot keeps 1,400–1,540 of 1,852 fine voxels occupied
        26 requests after the lift; a masked ray is unobserved, so nothing ever clears it). The
        pipeline hands the voxels only the held object can occupy (`AG3S._held_free_points`).

        Per voxel, not `Mapper.clear_region` (whole 8³ blocks inside an AABB — erases the table
        under the object and everything else in those blocks). The weight is kept, so the voxel is
        "observed free" and later observations average against it; unallocated voxels are skipped
        (nothing to clear). Indexing = cuRobo `storage.py` `_feature_node_indices_at_centers`:
        centred block keys + `ceil(grid/BS)//2`, voxel `(p − origin)/vs + grid/2`, `z·BS² + y·BS + x`.
        Returns the number of voxels freed.
        """
        import torch

        ts = itg.tsdf
        data = ts.data if hasattr(ts, "data") else ts
        bd = data.block_data
        n_alloc = int(data.num_allocated.detach().cpu().reshape(-1)[0])
        if n_alloc <= 0 or bd.dim() != 3 or bd.shape[1] <= 1:
            return 0
        bs = int(round(bd.shape[1] ** (1.0 / 3.0)))
        D, H, W = (int(v) for v in itg.grid_shape)
        vs = float(itg.voxel_size)
        origin = itg.origin.detach().cpu().numpy().reshape(-1)[:3].astype(np.float64)
        coords = data.block_coords.detach().cpu().numpy().reshape(-1, 3)[:n_alloc].astype(np.int64)
        slots = data.block_to_hash_slot.detach().cpu().numpy()[:n_alloc]
        live = np.flatnonzero(slots >= 0)
        if not live.size:
            return 0
        grid = np.array([W, H, D], np.int64)
        offsets = ((grid + bs - 1) // bs) // 2
        g = np.floor((np.asarray(points, np.float64) - origin) / vs + 0.5 * grid).astype(np.int64)
        g = np.unique(g, axis=0)
        key = g // bs - offsets
        loc = g % bs

        def pack(k):
            k = k + 4096                            # hash keys are 13-bit signed per axis
            return (k[:, 0] << 26) | (k[:, 1] << 13) | k[:, 2]

        live_keys = pack(coords[live])
        order = np.argsort(live_keys)
        sorted_keys = live_keys[order]
        want = pack(key)
        pos = np.searchsorted(sorted_keys, want)
        pos = np.clip(pos, 0, len(sorted_keys) - 1)
        hit = sorted_keys[pos] == want
        if not hit.any():
            return 0
        pool = live[order[pos[hit]]]
        li = loc[hit, 2] * bs * bs + loc[hit, 1] * bs + loc[hit, 0]
        dev = bd.device
        t_pool = torch.as_tensor(pool, device=dev, dtype=torch.long)
        t_li = torch.as_tensor(li, device=dev, dtype=torch.long)
        w = bd[t_pool, t_li, 1].float()
        observed = w > 0
        if not bool(observed.any()):
            return 0
        bd[t_pool[observed], t_li[observed], 0] = (w[observed] * float(truncation)).to(bd.dtype)
        return int(observed.sum().item())

    @staticmethod
    def _origin_of(itg, shape, voxel_size: float) -> np.ndarray:
        """첫 복셀 **중심**. `_last_esdf_origin` 은 격자 **중심**이다 (함정 1)."""
        centre = itg._last_esdf_origin.detach().cpu().numpy().reshape(-1)[:3].astype(float)
        dims = np.asarray(shape, float) * float(voxel_size)
        return centre - 0.5 * dims + 0.5 * float(voxel_size)

    @staticmethod
    def _force_sign_on_pure_voxels(values: np.ndarray, att_idx, voxel_size: float,
                                   *, threshold_voxels: float = 1.5) -> dict:
        """쥔 물체 **단독** 복셀의 부호를 양수로. 환경 표면도 든 복셀은 그대로 둔다.

        순수 판정에 추가 데이터가 필요 없다 — seed 제외 **뒤**의 `|d|` 가 복셀 반 칸보다 크면
        그 자리에 다른 표면이 없다는 뜻이다 (있으면 PBA 가 **보존된 이웃 seed** 에서 전파해
        거리가 반 칸 안으로 나온다). 반 칸 안이면 무언가 있으므로 **손대지 않는다.**

        **이 판정이 성립하려면 제외가 팽창하지 않아야 한다** — 이웃 환경 seed 를 지우면
        판정할 근거가 사라진다 (`_exclude_attached_seeds` 의 주석 참고).

        한 복셀 안에 쥔 물체와 환경 표면이 **함께** 있으면 익명 필드로는 가를 수 없다 (F13 의
        교훈). 그때는 이 판정이 "공유" 로 보고 부호를 그대로 두므로 **보수적 쪽**으로 닫힌다.

        임계가 보수적인 쪽으로 기운다: 무언가 가까우면 언제나 "그대로 둔다" 를 고른다. 그래서
        숨길 수 있는 관통이 경계값이 아니라 **구조적으로 0** 이다 (2026-09-22 판정 B).
        """
        if att_idx is None or not len(att_idx):
            return {"n_attached_voxels": 0, "n_sign_forced": 0,
                    "n_shared_left_alone": 0,
                    "threshold_voxels": float(threshold_voxels)}
        i, j, k = att_idx[:, 0], att_idx[:, 1], att_idx[:, 2]
        d = values[i, j, k]
        thr = float(threshold_voxels) * float(voxel_size)
        pure = np.abs(d) > thr
        values[i[pure], j[pure], k[pure]] = np.abs(d[pure])
        return {"n_attached_voxels": int(len(att_idx)),
                "n_sign_forced": int(pure.sum()),
                "n_shared_left_alone": int((~pure).sum()),
                "threshold_voxels": float(threshold_voxels),
                "threshold_m": thr}

    def _exclude_seed_points(self, itg, points: np.ndarray, origin: np.ndarray,
                             voxel_size: float, shape):
        """**점이 실제로 차지한** 복셀의 seed 를 지운다. 쥔 물체(attached)가 쓰는 길이다.

        T13 에서 이름을 바꿨다 (`_exclude_attached_seeds` → 이것). 이름이 용도를 말해야 하는
        이유가 이 STEP 의 요지다 — 같은 함수를 `target_free` 에도 썼다가 **한 voxel도 지우지
        못했다** (`TargetBall` docstring 의 표). 점이 물체 **전체**를 덮는 쥔 물체에는 이 길이
        맞고, 표본점 112 개뿐인 target 에는 맞지 않는다. target 은 `_exclude_ball_seeds` 를 쓴다.

        지우는 것은 `site_index` 의 seed 뿐이고 TSDF 는 건드리지 않는다. 그래서 다음 프레임에
        물체를 놓으면 그 표면이 그대로 살아 있고, 쥔 물체 **뒤에** 있는 다른 장애물의 거리도
        유지된다 — `clear_region` 으로 지웠을 때 잃는 것이 그 둘이다.
        """
        import torch

        # **팽창하지 않는다.** legacy `carve` 의 `attached_dilate_voxels` 는 쥔 물체를
        # 확실히 없애려는 것이었는데, 여기서 팽창하면 **이웃 환경 표면의 seed 까지 지운다.**
        # 그러면 순수 판정(아래 `_force_sign_on_pure_voxels`)이 무력해진다 — 그 자리에 다른
        # 표면이 있는지를 "seed 제외 뒤의 거리" 로 보는데, 그 표면의 seed 를 내가 지웠으니
        # 언제나 "순수" 로 나온다. 실측으로 잡혔다: 팽창 1 에서 공유 복셀이 **0 개**로
        # 보고돼 판정 B 가 A 로 퇴화했다.
        #
        # 환경 증거를 지우는 것은 **위험한 방향**이므로 물체가 실제로 차지한 복셀만 건드린다.
        idx = np.floor((points - origin) / float(voxel_size) + 0.5).astype(np.int64)
        cand = np.unique(idx, axis=0)
        return self._clear_seeds(itg, cand, shape)

    def _exclude_ball_seeds(self, itg, ball: "TargetBall", origin: np.ndarray,
                            voxel_size: float, shape):
        """**공 안의** 복셀 seed 를 지운다 — `z >= ball.z_min` 인 것만. target 이 쓰는 길이다.

        점이 아니라 범위로 지우는 이유는 `TargetBall` 의 표에 있다. 여기서 중요한 것은 두 가지다.

        **(1) 테이블에 구멍을 내지 않는다.** `z_min` 아래 복셀은 손대지 않으므로 지원면 상판의
        seed 가 그대로 남는다. 사과 밑면이 테이블에 닿아 있어 공을 그냥 지우면 그 접촉면의
        table seed 까지 사라지고, 그러면 팔이 테이블을 통과해도 아무도 막지 않는다.

        **(2) 주 계층은 한 복셀도 안 바뀐다.** 이 함수는 `target_free` 계층을 만들 때만 불린다
        (`_build_tiers(..., ball=...)`). 그래서 권한 없는 link 의 보호는 그대로다 — 그것이
        E1 과 다른 점이고, 이 변경이 `_exclude_seed_points` 의 경고를 키우면서도 안전한 이유다.

        범위를 **복셀 중심**으로 판정한다 (`floor(...+0.5)` 와 같은 규약). 중심이 공 안이면 그
        복셀의 seed 를 지운다 — 표면 복셀이 공 껍질에 걸쳐 있을 때 반 칸의 차이로 남는 것을 줄인다.
        """
        nx, ny, nz = (int(v) for v in shape)
        r = float(ball.radius)
        lo = np.floor((ball.centre - r - origin) / float(voxel_size) + 0.5).astype(np.int64)
        hi = np.ceil((ball.centre + r - origin) / float(voxel_size) + 0.5).astype(np.int64)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi + 1, np.array([nx, ny, nz]))
        if np.any(hi <= lo):
            return 0, None
        # AABB 안만 본다 — 격자 전체를 훑으면 2 백만 복셀이고 프레임 예산에 들어오지 않는다.
        ax = [np.arange(int(lo[i]), int(hi[i]), dtype=np.int64) for i in range(3)]
        grid = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
        centres = origin[None, :] + grid.astype(np.float64) * float(voxel_size)
        inside = np.linalg.norm(centres - ball.centre[None, :], axis=1) <= r
        inside &= centres[:, 2] >= float(ball.z_min)   # **테이블 위만**
        cand = grid[inside]
        if not len(cand):
            return 0, None
        return self._clear_seeds(itg, cand, shape)

    def _clear_seeds(self, itg, cand: np.ndarray, shape):
        """`site_index` 의 seed 를 지우고 `(지운 수, 복셀 인덱스)` 를 돌려준다.

        두 경로(점·공)가 이 한 곳을 쓴다. 부호 교정이 **같은 복셀 집합**에 걸려야 하므로
        인덱스를 함께 돌려주는 것이 계약이다.
        """
        import torch

        nx, ny, nz = (int(v) for v in shape)
        inside = np.all((cand >= 0) & (cand < np.array([nx, ny, nz])), axis=1)
        cand = cand[inside]
        if not len(cand):
            return 0, None
        t = torch.as_tensor(cand, device=itg._site_index.device, dtype=torch.long)
        before = int((itg._site_index[t[:, 0], t[:, 1], t[:, 2]] >= 0).sum().item())
        itg._site_index[t[:, 0], t[:, 1], t[:, 2]] = -1
        return before, cand

    @staticmethod
    def _labels_from_sites(tier: _Tier, grid: VoxelGrid, seeds: dict,
                           label_names: tuple) -> Optional[np.ndarray]:
        """`site_index` 조회로 라벨 격자를 만든다.

        절차는 셋이다.
        1. 실제로 site 로 쓰인 복셀만 모은다 (표면 복셀이고, 보통 격자의 수 % 다).
        2. 그 복셀 중심에서 **가장 가까운 씨앗 점**을 찾아 스냅 반경 안이면 그 라벨을 준다.
           `label_snap_voxels` 가 그 반경이다 — 씨앗 점이 떨어진 복셀과 TSDF 영교차가 정한
           표면 복셀이 같지 않기 때문에 스냅이 필요하다 (라벨 스냅).
        3. 복셀마다 `site_label[site_index[복셀]]` 을 읽는다.
        """
        if not seeds:
            return None
        from scipy.spatial import cKDTree

        shape = tuple(int(v) for v in tier.values.shape)
        n_vox = int(np.prod(shape))
        site_lin = tier.site_linear.reshape(-1)
        used = np.unique(site_lin[site_lin >= 0])
        if not used.size:
            return None

        nx, ny, nz = shape
        sx = used // (ny * nz)
        sy = (used // nz) % ny
        sz = used % nz
        centres = (tier.origin
                   + np.stack([sx, sy, sz], axis=1).astype(float) * tier.voxel_size)

        pts = np.concatenate([seeds[n] for n in label_names], axis=0)
        owner = np.concatenate([np.full(len(seeds[n]), i, np.int32)
                                for i, n in enumerate(label_names)])
        snap = 2.0 * tier.voxel_size
        dist, near = cKDTree(pts).query(centres, k=1)
        site_label = np.full(n_vox, -1, np.int32)
        hit = dist <= snap
        site_label[used[hit]] = owner[near[hit]]

        out = np.full(n_vox, -1, np.int32)
        valid = site_lin >= 0
        out[valid] = site_label[site_lin[valid]]
        return out.reshape(shape)

    def _observation_coverage(self, coarse: _Tier) -> dict:
        """"이 부피가 관측됐는가" 에 대한 답. **0.0 으로 보고하지 않는다.**

        legacy 의 `unknown_fraction` 은 dense TSDF 에서 **가중치가 0 인 복셀의 비율**이다.
        광선이 지나간 자유 공간은 가중치를 받으므로 `known` 이고, 아무도 안 본 곳만 `unknown`
        이다. 그 구분이 G2(미관측 100 % 필드가 `valid` · `certified True` 로 보고됐다)의 축이다.

        **cuRobo 의 block-sparse TSDF 는 그 구분을 할 수 없다.** 표면 근처 블록만 할당하므로
        "보고 지나간 자유 공간" 과 "한 번도 안 본 곳" 이 **둘 다 미할당**이다. 따라서
        할당 블록 부피로 비율을 만들면 언제나 "거의 다 미관측" 이 나오고, 그것은 legacy 의
        숫자와 **같은 것을 재지 않는다** — 정의를 안 적고 두 숫자를 나란히 놓는 것이
        C2("거친 33 mm 대 미세 2 mm 는 측정법 산물")를 만든 방식이다.

        그래서 **지어내지 않는다.** `unknown_fraction` 을 `None` 으로 두고 그 사실을 싣는다.
        G2 가 실제로 필요한 것(아무것도 안 봤는가)은 `n_active_blocks == 0` 이 그대로 답한다.
        이 자리는 F16(블록-스파스가 만드는 새 안전 질문 — 할당 안 한 곳을 물으면 뭐라
        답하는가)이 실제로 도착한 지점이다.
        """
        stats = {}
        try:
            st = self._mapper.get_stats()
            active = int(st.get("active_blocks", 0))
        except Exception:  # noqa: BLE001 — 통계가 없어도 필드는 유효하다
            active = -1
        tsdf_vs = float(getattr(self.config, "tsdf_voxel_size", 0.0)
                        or self.config.voxel_size)
        stats["n_active_blocks"] = active
        stats["observed_block_volume_m3"] = max(active, 0) * (8.0 * tsdf_vs) ** 3
        stats["grid_volume_m3"] = float(np.prod(
            np.asarray(coarse.values.shape, float) * coarse.voxel_size))
        # 아무것도 안 봤으면 그것은 확실히 100 % 미관측이다 — 그 경우만 수치로 답한다.
        stats["unknown_fraction"] = 1.0 if active == 0 else None
        stats["unknown_fraction_available"] = active == 0
        stats["unknown_fraction_definition"] = (
            "block-sparse TSDF 는 '보고 지나간 자유 공간' 과 '한 번도 안 본 곳' 을 구분하지 "
            "못한다 (둘 다 미할당). 그래서 legacy 의 per-voxel UNKNOWN 비율에 대응하는 값이 "
            "없다. active_blocks == 0 일 때만 1.0 으로 확정한다.")
        return stats

    def _stats(self, tiers, label_names, per_camera, *, n_attached: int,
               n_static: int) -> dict:
        coarse = tiers[0]
        occupied = int((coarse.values < 0).sum())
        return {
            "backend": "curobo",
            "voxel_size": coarse.voxel_size,
            "grid_shape": tuple(int(v) for v in coarse.values.shape),
            "n_voxels": int(np.prod(coarse.values.shape)),
            "frames": self._frames,
            "n_occupied": occupied,
            **self._observation_coverage(coarse),
            "n_labels": len(label_names),
            "label_names": list(label_names),
            "n_attached_points": n_attached,
            "n_attached_seeds_excluded": sum(
                t.extra.get("n_attached_seeds_excluded", 0) for t in tiers),
            "attached_sign_correction": [
                {"tier": t.name, **{k: v for k, v in t.extra.items()
                                    if k != "n_attached_seeds_excluded"}} for t in tiers],
            "n_static_shapes": n_static,
            # T32 H4-fix: which TSDF truncation each tier was built from.
            "tsdf_truncation": {"coarse_m": self.coarse_truncation,
                                "fine_m": self.fine_truncation,
                                "fine_requested_m": getattr(self.config, "fine_truncation_m", None),
                                "separate_fine_tsdf": self.separate_fine_tsdf},
            "decay": {"decayed": bool(
                float(getattr(self.config, "time_decay", 1.0)) < 1.0
                or float(getattr(self.config, "frustum_decay", 1.0)) < 1.0)},
            "tiers": [{"name": t.name, "voxel_size": t.voxel_size,
                       "shape": [int(v) for v in t.values.shape],
                       "origin": [float(v) for v in t.origin]} for t in tiers],
            "per_camera": per_camera,
        }
