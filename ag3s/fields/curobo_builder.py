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
import time
from typing import Any, Optional, Sequence

import numpy as np

from benchmark.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays
from benchmark.ag3s.fields.esdf import CameraDepth, VoxelGrid, default_bounds
from benchmark.ag3s.fields.observation import probe_from_cameras
from benchmark.ag3s.fields.provenance import FieldProvenance

__all__ = ["CuroboFieldBuilder", "ATTACHED_LABEL", "unpack_site_linear"]

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
    def _ensure_mapper(self, cameras: Sequence[CameraDepth]):
        if self._mapper is not None:
            return self._mapper
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
            truncation_distance=float(cfg.truncation),
            depth_minimum_distance=float(cfg.depth_min),
            depth_maximum_distance=float(cfg.depth_max),
            image_height=int(h), image_width=int(w),
            # 감쇠는 **기본 끔**이다 — numpy 경로와 같은 조건에서 두 backend 를 대조하기 위해.
            # 켤 때 값은 F20 에서 실측해 정했다 (a_t 0.99 · a_f 0.8).
            decay_factor=float(getattr(cfg, "time_decay", 1.0)),
            frustum_decay_factor=float(getattr(cfg, "frustum_decay", 1.0)),
            device=dev)
        self._mapper = Mapper(mcfg)
        self._assert_curobo_surface(self._mapper.integrator)
        return self._mapper

    def reset(self) -> None:
        """누적 TSDF 와 라벨 이름을 버린다. T0 이 요구하는 reset 시점 기록의 대상이다."""
        if self._mapper is not None:
            self._mapper.reset()
        self._frames = 0
        self._label_names = ()

    # -- 한 프레임 --------------------------------------------------------------------
    def update(self, cameras: Sequence[CameraDepth], *,
               target_points: Optional[np.ndarray] = None,
               exclude_target: bool = False,
               target_free_points: Optional[np.ndarray] = None,
               target_free_label: Optional[str] = None,
               support_points: Optional[np.ndarray] = None,
               attached_points: Optional[np.ndarray] = None,
               labelled_points: Optional[dict] = None,
               static_geometry: Optional[Sequence[Any]] = None,
               observed_at: Optional[float] = None,
               frame_id: str = "", frame_index: int = -1) -> CuroboEsdfField:
        """`EsdfBuilder.update` 와 같은 계약. 돌려주는 것은 `CuroboEsdfField` 다.

        `exclude_target` 는 **지원하지 않는다.** E1 이 그것을 폐기했다 — 필드는 익명이라
        아직 안 쥔 target 을 파내면 손끝뿐 아니라 전신에게 사라진다. 지금 코드는 항상
        `False` 로 부르고, `True` 로 오면 조용히 무시하지 않고 예외를 던진다.

        `target_free_points` 는 그것과 **다른 것**이다. 필드에서 파내는 것이 아니라, 그 점들의
        seed 를 지운 **미세 계층 한 겹을 따로** 만들어 `field.target_free_layers` 에 실어
        보낸다. 본 계층(`layers`)은 한 복셀도 바뀌지 않으므로 권한 없는 질의점은 예전과 같은
        답을 받는다 — 익명으로 파내는 것과 이름으로 되묻는 것의 차이가 정확히 E1 이다.
        `None`(기본)이면 그 계층을 만들지 않고 비용도 0 이다.
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

        # 미세 계층 창의 중심. 지금은 **grounding 의 target 무게중심**이다 — 기존
        # `build_rollout_fields.py:110` 과 같은 규칙이라 두 backend 를 같은 조건에서 견준다.
        # 계획된 방향은 action chunk 의 swept volume 이고, 짝 비교는 T3 에서 한다
        # (F11 — 파지 순간 attention 이 목적지로 넘어가 창이 따라가는 것 — 과 같은 자리).
        tgt = (None if target_points is None
               else np.asarray(target_points, np.float64).reshape(-1, 3))
        self._fine_centre = (None if tgt is None or not len(tgt)
                             else tgt.mean(axis=0))

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

        tiers = self._build_tiers(itg, attached, dev, torch)

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
            remove = free_points if attached is None or not len(attached) else np.vstack(
                [np.asarray(attached, np.float64).reshape(-1, 3), free_points])
            free_tiers = self._build_tiers(itg, remove, dev, torch, only="fine",
                                           tier_name="fine_no_target")
            free_ms = (time.monotonic() - t0) * 1e3
            self._announce_target_free_cost(free_ms, len(free_points), free_tiers)

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
        # **target 없는 계층은 기록에 남는다.** 그 계층이 있었는지 없었는지 모르는 기록은
        # 정책이 실제로 걸린 프레임인지 알 방법이 없다 — 키를 정책이 꺼져 있을 때도 싣는다.
        stats["target_free"] = {
            "n_layers": len(free_tiers),
            "n_target_points": 0 if free_points is None else int(len(free_points)),
            "build_ms": round(float(free_ms), 3),
            "label": target_free_label,
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
                "target_free_points 가 왔는데 target_points 가 비었습니다. 미세 창의 중심이 "
                "target 무게중심이므로 계층을 놓을 자리가 없습니다 — 두 인자는 같은 프레임의 "
                "같은 target 에서 와야 합니다")

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

    def _build_tiers(self, itg, attached, dev, torch, *, only: Optional[str] = None,
                     tier_name: Optional[str] = None) -> list[_Tier]:
        """계층마다 seed -> (쥔 물체 제외) -> propagate -> 복사.

        `Mapper.compute_esdf` 를 부르지 않는다 — 그쪽은 CUDA graph 로 seed·propagate·distance
        를 한 덩어리로 실행해서 사이에 끼어들 자리가 없다. 대신 `compute_esdf` 가 하는 부수
        효과(`_esdf_voxel_size` 와 `_last_esdf_origin` 갱신)를 여기서 같이 한다.

        `attached` 로 오는 것이 **반드시 쥔 물체일 필요는 없다** — 여기서 그것은 "seed 에서 뺄
        점" 이고, target 없는 계층은 같은 길로 target 점을 뺀다 (T8b). 부호 교정도 같은 집합에
        걸려야 하므로 두 경우가 한 함수인 것이 맞다.
        """
        centre = getattr(self, "_fine_centre", None)  # update() 가 정한다
        tiers: list[_Tier] = []
        for name, origin_override, vs in self._tier_specs(centre, only=only,
                                                          tier_name=tier_name):
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
                n_excluded, att_idx = self._exclude_attached_seeds(
                    itg, attached, org, vs, shape)

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
            tier.extra["n_attached_seeds_excluded"] = n_excluded
            tier.extra.update(sign)
            tiers.append(tier)
        return tiers

    def _announce_target_free_cost(self, ms: float, n_points: int, tiers) -> None:
        """**첫 프레임에 비용을 크게 찍는다.** 그 뒤로는 stats 에만 남는다.

        시작 로그에 ms 가 없으면 "한 겹 더 만든다" 가 프레임 예산에 무엇을 했는지 아무도
        모른다 — 그리고 이 계층은 조용히 켜져 있으면 안 되는 종류의 것이다 (안전 계층을 한
        겹 비껴가는 길이므로).
        """
        import logging

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

    def _exclude_attached_seeds(self, itg, attached: np.ndarray, origin: np.ndarray,
                                voxel_size: float, shape):
        """쥔 물체가 차지한 복셀의 seed 를 지운다. **propagate 앞**이어야 한다.

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
        idx = np.floor((attached - origin) / float(voxel_size) + 0.5).astype(np.int64)
        nx, ny, nz = (int(v) for v in shape)
        cand = np.unique(idx, axis=0)
        inside = np.all((cand >= 0) & (cand < np.array([nx, ny, nz])), axis=1)
        cand = cand[inside]
        if not len(cand):
            return 0, None
        t = torch.as_tensor(cand, device=itg._site_index.device, dtype=torch.long)
        before = int((itg._site_index[t[:, 0], t[:, 1], t[:, 2]] >= 0).sum().item())
        itg._site_index[t[:, 0], t[:, 1], t[:, 2]] = -1
        # 마스크를 함께 돌려준다 — 부호 교정이 **같은 복셀 집합**에 걸려야 한다.
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
            "decay": {"decayed": bool(
                float(getattr(self.config, "time_decay", 1.0)) < 1.0
                or float(getattr(self.config, "frustum_decay", 1.0)) < 1.0)},
            "tiers": [{"name": t.name, "voxel_size": t.voxel_size,
                       "shape": [int(v) for v in t.values.shape],
                       "origin": [float(v) for v in t.origin]} for t in tiers],
            "per_camera": per_camera,
        }
