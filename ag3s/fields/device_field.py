"""GPU 에 머무는 ESDF 계층 — 질의점에서만 답한다 (T38 B3).

`EsdfField` 는 격자를 host numpy 로 들고 있고, `gradient` 는 첫 질의에서 **격자 전체**에
`np.gradient` 를 돌려 캐시한다. cuRobo backend 에서 그것은 계층마다 128³ D2H (값 8.4 MB) 와
128³ × 3 기울기 계산이고, 그런데 TO 가 실제로 묻는 점은 구 × step 몇 천 개뿐이다 (T38 Phase A:
`_build_tiers` D2H 37–50 MB/청크, linearize 97 ms 의 대부분이 `np.gradient`).

`DeviceEsdfField` 는 값·라벨 격자를 **GPU tensor 로 둔 채** 같은 질문에 답한다.

    distance(points)  gradient(points)  label(points)  is_label(points, name)

**정의는 `EsdfField` 와 하나도 다르지 않다.** 다른 것은 "격자 값을 어디서 읽나" 뿐이다.

| 단계 | 어디서 | 산술 |
|---|---|---|
| `_lattice` (i0 · t · inside) | CPU — `EsdfField._lattice` 그대로 | 같은 코드 |
| 8 코너 인덱스 · gather | GPU | 정수 — 정확 |
| 코너 기울기 | float64 | `np.gradient(edge_order=1)` 의 **그 복셀 값**: 내부 `(f[i+1] − f[i−1]) / (2.·h)`, 경계 `(f[1] − f[0]) / h` · `(f[n] − f[n−1]) / h`, 그리고 캐시와 같이 **float32 로 반올림** |
| 삼선형 가중치 · 누적 | float64 | `w = (wx·wy)·wz`, `out += w·v` 를 코너 순서 `(dx, dy, dz)` 그대로 8 번 |
| 격자 밖 · 해석적 채널 · 계수 | CPU — `EsdfField` 와 같은 꼬리 | 같은 코드 |

산술(위 표의 float64 두 줄)을 **어디서** 하는지는 질의점 수로 고른다
(`config.ESDF_QUERY_HOST_ARITH_MAX_POINTS`): 적으면 GPU 는 질의점이 쓰는 칸만 `index_select` 로
꺼내 주고 host numpy 가 산술한다 (kernel 몇 개 · 왕복 한 번 — 이 서버의 호출 대부분), 많으면 GPU
float64 로 한다. **어느 쪽이든 격자 전체는 host 로 오지 않고, 값은 `EsdfField` 와 비트 단위로 같다.**

float64 의 `+ − × ÷` 는 IEEE 정확 반올림이므로 **연산 순서가 같으면 비트가 같다.** GPU 산술에서
지켜야 할 것은 둘이다: (1) 한 kernel 안에서 곱·덧셈을 묶지 않는다 (torch 원소별 연산은 연산마다
kernel 이라 FMA 로 합쳐지지 않는다), (2) **스칼라로 나누지 않는다** — torch CUDA 는 CPU 스칼라
나눗셈을 `a · (1/b)` 로 바꾼다 (실측 100 만 개 중 133,303 개가 1 ulp 다름). 그래서 분모는 CUDA
tensor 다.

**전체 격자는 요청할 때만 host 로 온다.** `distance_grid` · `label_grid` 는 lazy property 이고
(읽기 전용 사본을 캐시), 질의 경로는 그것을 건드리지 않는다. 기록기(`--record-constraints-esdf
full`)는 `host_distance_grid(cache=False)` 로 **공유 상태를 바꾸지 않고** worker 스레드에서
가져간다.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from benchmark.ag3s.fields.esdf import EsdfField, VoxelGrid, analytic_distance, analytic_gradient

__all__ = ["DeviceEsdfField", "layer_from_device", "is_device_layer"]

#: 코너 순서 — `EsdfField.distance` 의 `for dx: for dy: for dz:` 와 같다. 누적 순서가 곧 비트다.
_CORNERS = tuple((dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1))
_CORNER_OFFSETS = np.asarray(_CORNERS, np.int64)


def _host_arith_max_points() -> int:
    """호출 시점에 읽는다 (테스트가 두 경로를 강제할 수 있게)."""
    from benchmark.ag3s import config

    return int(getattr(config, "ESDF_QUERY_HOST_ARITH_MAX_POINTS", 0))


def _trilinear_weight(t: np.ndarray, dx: int, dy: int, dz: int) -> np.ndarray:
    """`EsdfField.distance` 의 가중치 식 그대로 — `(wx · wy) · wz`."""
    return ((t[:, 0] if dx else 1.0 - t[:, 0])
            * (t[:, 1] if dy else 1.0 - t[:, 1])
            * (t[:, 2] if dz else 1.0 - t[:, 2]))


def is_device_layer(layer: Any) -> bool:
    return isinstance(layer, DeviceEsdfField)


def _max_abs_finite(values) -> float:
    """`float(np.max(np.abs(d[np.isfinite(d)])))` (없으면 0.0) — `layer_from_arrays` 와 같은 값.

    비유한 값을 0 으로 바꿔 넣어도 최대는 같다 (`|d| ≥ 0`). 유한 값이 하나도 없으면 0.0 이고
    그것도 `layer_from_arrays` 와 같다. float32 의 max 는 정확하다.
    """
    import torch

    if values.numel() == 0:
        return 0.0
    a = torch.where(torch.isfinite(values), values.abs(), torch.zeros_like(values))
    return float(a.max().item())


class DeviceEsdfField(EsdfField):
    """`EsdfField` 와 같은 질의 · 같은 정의, 격자는 GPU.

    `distance_grid` / `label_grid` 는 **lazy** 다 — 처음 읽을 때 한 번 D2H 해서 읽기 전용으로
    캐시한다. 질의(`distance` · `gradient` · `label`)는 그것을 쓰지 않는다. 캐시를 고쳐도 필드는
    바뀌지 않으므로 (참값은 GPU tensor) 고칠 수 없게 막는다.

    dataclass 생성자를 부르지 않는다 — `distance_grid` · `label_grid` 가 여기서는 property 다.
    나머지 필드(`grid` · `max_distance` · `stats` · `outside_distance` · `label_names` ·
    `static_shapes` · 계수 · `provenance`)는 `EsdfField` 와 같은 이름 · 같은 뜻의 보통 속성이다.
    """

    def __init__(self, grid: VoxelGrid, values, *, labels=None,
                 label_names: tuple = (), stats: Optional[dict] = None,
                 outside_distance: Optional[float] = None,
                 max_distance: Optional[float] = None):
        import torch

        if not isinstance(values, torch.Tensor):
            raise TypeError(f"values 는 torch.Tensor 여야 합니다: {type(values).__name__}")
        if values.dtype != torch.float32:
            raise TypeError(f"values 는 float32 여야 합니다 (EsdfField.distance_grid 와 같다): "
                            f"{values.dtype}")
        if tuple(values.shape) != tuple(grid.shape):
            raise ValueError(
                f"거리 격자 {tuple(values.shape)} 가 복셀 격자 {grid.shape} 와 다릅니다")
        self.grid = grid
        self._values = values.contiguous()
        self._labels = None
        self.max_distance = (_max_abs_finite(self._values) if max_distance is None
                             else float(max_distance))
        self.stats = dict(stats or {})
        self.outside_distance = outside_distance
        self.label_names = tuple(label_names)
        self.static_shapes = ()
        self.outside_query_count = 0
        self.queried_point_count = 0
        self.provenance = None
        self._host_values: Optional[np.ndarray] = None
        self._host_labels: Optional[np.ndarray] = None
        self.n_host_materialisations = 0
        #: T43 TP — per-field host copy of the value / label grids (`mirror_to_host`), or None.
        self._mirror_values: Optional[np.ndarray] = None
        self._mirror_labels: Optional[np.ndarray] = None
        # 코너 상수는 한 번만 올린다 (질의마다 작은 H2D 를 만들지 않는다).
        self._corner_offsets = torch.as_tensor(np.asarray(_CORNERS, np.int64), device=values.device)
        self._corner_select = torch.as_tensor(np.asarray(_CORNERS, bool), device=values.device)
        self._n_last = torch.as_tensor(np.asarray(grid.shape, np.int64) - 1, device=values.device)
        if labels is not None:
            self.label_grid = labels

    # -- device 상태 ----------------------------------------------------------------------
    @property
    def device(self):
        return self._values.device

    @property
    def values_device(self):
        """GPU 값 격자 `(nx, ny, nz)` float32. **읽기 전용으로 다룬다** — 기록기 worker 도 읽는다."""
        return self._values

    @property
    def labels_device(self):
        return self._labels

    # -- lazy host 사본 -------------------------------------------------------------------
    def host_distance_grid(self, *, cache: bool = True) -> np.ndarray:
        """값 격자의 host 사본 (float32). `cache=False` 면 이 객체를 **바꾸지 않는다** (기록 worker)."""
        if self._host_values is not None:
            return self._host_values
        arr = self._values.detach().cpu().numpy()
        if not cache:
            return arr
        arr.flags.writeable = False
        self._host_values = arr
        self.n_host_materialisations += 1
        return arr

    def host_label_grid(self, *, cache: bool = True) -> Optional[np.ndarray]:
        if self._labels is None:
            return None
        if self._host_labels is not None:
            return self._host_labels
        arr = self._labels.detach().cpu().numpy()
        if not cache:
            return arr
        arr.flags.writeable = False
        self._host_labels = arr
        self.n_host_materialisations += 1
        return arr

    @property
    def distance_grid(self) -> np.ndarray:  # type: ignore[override]
        return self.host_distance_grid()

    @property
    def label_grid(self) -> Optional[np.ndarray]:  # type: ignore[override]
        return self.host_label_grid()

    @label_grid.setter
    def label_grid(self, value) -> None:
        """`None` · numpy · torch 를 받는다. host 배열이면 GPU 로 올린다 (참값은 GPU 쪽 하나)."""
        import torch

        self._host_labels = None
        if getattr(self, "_mirror_labels", None) is not None:
            self._mirror_labels = None                  # T43 TP: the mirror follows the grid
        if value is None:
            self._labels = None
            return
        if isinstance(value, torch.Tensor):
            t = value.to(device=self._values.device, dtype=torch.int32)
        else:
            t = torch.as_tensor(np.asarray(value, np.int32), device=self._values.device)
        if tuple(t.shape) != tuple(self.grid.shape):
            raise ValueError(f"라벨 격자 {tuple(t.shape)} 가 복셀 격자 {self.grid.shape} 와 다릅니다")
        self._labels = t.contiguous()

    @property
    def has_labels(self) -> bool:  # type: ignore[override]
        return self._labels is not None

    # -- T43 TP: host mirror -------------------------------------------------------------
    def mirror_to_host(self) -> None:
        """Copy the value (and label) grid to host **once**, so later queries read it there.

        Every query on this field is a GPU round trip (a gather and a sync per call — measured
        0.6–1.3 ms at 1–4 k points, `outputs/impl/T43TP/prof/esdf_bench.json`); the TO makes ~20 per
        solve. One copy of both 128³ layers costs ~1.3 ms. After it, `_gather` / `label` read the
        mirror and the arithmetic is always the host one — the same code and order the host path
        runs for small queries, which this class already guarantees equal to `EsdfField` (and to
        its own GPU path) bit for bit. Not a whole-grid gradient: corners are still differentiated
        on demand. `drop_mirror` undoes it.
        """
        if getattr(self, "_mirror_values", None) is None:
            self._mirror_values = self._values.detach().cpu().numpy().reshape(-1)
        if self._labels is not None and getattr(self, "_mirror_labels", None) is None:
            self._mirror_labels = self._labels.detach().cpu().numpy().reshape(-1)

    def drop_mirror(self) -> None:
        self._mirror_values = None
        self._mirror_labels = None

    @property
    def mirrored(self) -> bool:
        return getattr(self, "_mirror_values", None) is not None

    # -- 질의 ------------------------------------------------------------------------------
    def _device_lattice(self, i0: np.ndarray):
        """`(코너 (8, N, 3) int64 GPU, n (3,) host)` — `np.minimum(i0 + [dx,dy,dz], n)`."""
        import torch

        dev = self._values.device
        n = np.asarray(self.grid.shape, np.int64) - 1
        i0_d = torch.as_tensor(np.ascontiguousarray(i0, np.int64), device=dev)
        idx = torch.minimum(i0_d[None, :, :] + self._corner_offsets[:, None, :], self._n_last)
        return idx, n

    def _linear(self, idx):
        _, ny, nz = (int(v) for v in self.grid.shape)
        return (idx[..., 0] * ny + idx[..., 1]) * nz + idx[..., 2]

    def _weights(self, t: np.ndarray):
        """`(8, N)` float64 GPU — `(wx · wy) · wz`, 코너 순서 `_CORNERS`."""
        import torch

        dev = self._values.device
        t_d = torch.as_tensor(np.ascontiguousarray(t, np.float64), device=dev)
        lo = 1.0 - t_d                                  # numpy 의 `1.0 - t[:, a]` 와 같은 연산
        sel = self._corner_select                       # (8, 3)
        per_axis = torch.where(sel[:, None, :], t_d[None, :, :], lo[None, :, :])  # (8, N, 3)
        return (per_axis[..., 0] * per_axis[..., 1]) * per_axis[..., 2]

    # -- 적은 질의점: GPU 는 칸만 꺼내 주고 산술은 host (EsdfField 와 같은 코드) ---------------
    def _linear_host(self, idx: np.ndarray) -> np.ndarray:
        _, ny, nz = (int(v) for v in self.grid.shape)
        return (idx[..., 0] * ny + idx[..., 1]) * nz + idx[..., 2]

    def _gather(self, lin: np.ndarray) -> np.ndarray:
        """`lin` (임의 모양, int64) 위치의 float32 값 — 이 경로에서 GPU 가 하는 일의 전부."""
        mirror = getattr(self, "_mirror_values", None)
        if mirror is not None:
            # T43 TP: the host mirror — the same float32 values the GPU gather returns
            return mirror[np.ascontiguousarray(lin, np.int64).reshape(-1)].reshape(lin.shape)
        import torch

        idx = torch.as_tensor(np.ascontiguousarray(lin, np.int64).reshape(-1),
                              device=self._values.device)
        vals = torch.index_select(self._values.reshape(-1), 0, idx)
        return vals.cpu().numpy().reshape(lin.shape)

    def _host_distance(self, i0: np.ndarray, t: np.ndarray) -> np.ndarray:
        n = np.asarray(self.grid.shape) - 1
        idx = np.minimum(i0[None, :, :] + _CORNER_OFFSETS[:, None, :], n)   # (8, N, 3)
        v = self._gather(self._linear_host(idx))                            # (8, N) float32
        out = np.zeros(len(i0), np.float64)
        for c, (dx, dy, dz) in enumerate(_CORNERS):
            out += _trilinear_weight(t, dx, dy, dz) * v[c]
        return out

    def _host_corner_gradients(self, idx: np.ndarray) -> np.ndarray:
        """`(8, N, 3)` float32 — 코너 복셀의 `np.gradient(edge_order=1)` 값 (`_grad_cache` 와 같은 값).

        분모는 `np.gradient` 와 같은 수다: 내부 `2. * h`, 경계 `h` (둘 다 Python float 를 float64 로).
        """
        shape = np.asarray(self.grid.shape, np.int64)
        if np.any(shape < 2):
            raise ValueError("Shape of array too small to calculate a numerical gradient, "
                             "at least (edge_order + 1) elements are required.")
        n = shape - 1
        h = float(self.grid.voxel_size)
        lins = np.empty((2, 3) + idx.shape[:2], np.int64)
        edge = np.empty((3,) + idx.shape[:2], bool)
        for a in range(3):
            up = idx.copy()
            dn = idx.copy()
            up[..., a] = np.minimum(idx[..., a] + 1, n[a])
            dn[..., a] = np.maximum(idx[..., a] - 1, 0)
            lins[0, a] = self._linear_host(up)
            lins[1, a] = self._linear_host(dn)
            edge[a] = (idx[..., a] == 0) | (idx[..., a] == n[a])
        f = self._gather(lins).astype(np.float64)                          # (2, 3, 8, N)
        den = np.where(edge, h, 2.0 * h)
        g = ((f[0] - f[1]) / den).astype(np.float32)                        # (3, 8, N)
        return np.moveaxis(g, 0, -1)                                        # (8, N, 3)

    def _host_gradient(self, i0: np.ndarray, t: np.ndarray) -> np.ndarray:
        n = np.asarray(self.grid.shape) - 1
        idx = np.minimum(i0[None, :, :] + _CORNER_OFFSETS[:, None, :], n)
        g = self._host_corner_gradients(idx)
        out = np.zeros((len(i0), 3), np.float64)
        for c, (dx, dy, dz) in enumerate(_CORNERS):
            out += _trilinear_weight(t, dx, dy, dz)[:, None] * g[c]
        return out

    # -- 많은 질의점: 같은 산술을 GPU float64 로 ---------------------------------------------
    def _device_distance(self, i0: np.ndarray, t: np.ndarray) -> np.ndarray:
        import torch

        idx, _ = self._device_lattice(i0)
        v = self._values.reshape(-1)[self._linear(idx)].to(torch.float64)   # (8, N) 정확
        terms = self._weights(t) * v
        out = torch.zeros(terms.shape[1], dtype=torch.float64, device=terms.device)
        for c in range(len(_CORNERS)):                   # numpy 의 `out += wgt * g[...]` 순서
            out += terms[c]
        return out.cpu().numpy()

    def _corner_gradients(self, idx):
        """`(8, N, 3)` float64 GPU — 각 코너 복셀의 `np.gradient(edge_order=1)` 값을 float32 로
        반올림한 뒤 float64 로 (`EsdfField._grad_cache` 와 같은 값)."""
        import torch

        dev = self._values.device
        shape = np.asarray(self.grid.shape, np.int64)
        if np.any(shape < 2):
            # np.gradient 와 같은 실패 — 조용히 다른 정의로 답하지 않는다.
            raise ValueError("Shape of array too small to calculate a numerical gradient, "
                             "at least (edge_order + 1) elements are required.")
        n = shape - 1
        h = float(self.grid.voxel_size)
        flat = self._values.reshape(-1)
        one_h = torch.tensor(h, dtype=torch.float64, device=dev)            # 경계: `/ dx_0`
        two_h = torch.tensor(2.0 * h, dtype=torch.float64, device=dev)      # 내부: `/ (2. * ax_dx)`
        out = []
        for a in range(3):
            ia = idx[..., a]
            edge = (ia == 0) | (ia == int(n[a]))
            up = idx.clone()
            dn = idx.clone()
            up[..., a] = torch.clamp(ia + 1, max=int(n[a]))
            dn[..., a] = torch.clamp(ia - 1, min=0)
            fp = flat[self._linear(up)].to(torch.float64)
            fm = flat[self._linear(dn)].to(torch.float64)
            den = torch.where(edge, one_h, two_h)        # CUDA tensor 로 나눈다 (스칼라 아님)
            out.append(((fp - fm) / den).to(torch.float32).to(torch.float64))
        return torch.stack(out, dim=-1)

    def _device_gradient(self, i0: np.ndarray, t: np.ndarray) -> np.ndarray:
        import torch

        idx, _ = self._device_lattice(i0)
        g = self._corner_gradients(idx)                  # (8, N, 3)
        terms = self._weights(t)[..., None] * g
        out = torch.zeros(terms.shape[1:], dtype=torch.float64, device=terms.device)
        for c in range(len(_CORNERS)):
            out += terms[c]
        return out.cpu().numpy()

    def distance(self, points: np.ndarray) -> np.ndarray:
        """`EsdfField.distance` 와 같은 정의 (삼선형). 격자 값은 GPU 에서 질의점에서만 읽는다."""
        i0, t, inside = self._lattice(points)
        if not len(i0):
            out = np.zeros(0, np.float64)
        elif len(i0) < _host_arith_max_points() or self.mirrored:
            out = self._host_distance(i0, t)
        else:
            out = self._device_distance(i0, t)
        self.queried_point_count += len(i0)
        self.outside_query_count += int((~inside).sum())
        if self.outside_distance is not None and (~inside).any():
            out[~inside] = float(self.outside_distance)
        if self.static_shapes:
            out = np.minimum(out, analytic_distance(points, self.static_shapes))
        return out

    def gradient(self, points: np.ndarray) -> np.ndarray:
        """`EsdfField.gradient` 와 같은 정의 — 격자 중심차분(float32 캐시 값)의 삼선형 보간.

        격자 전체를 미분하지 않는다. 질의점이 쓰는 8 코너 × 3 축의 stencil 만 GPU 에서 계산한다.
        """
        i0, t, inside = self._lattice(points)
        if not len(i0):
            out = np.zeros((0, 3), np.float64)
        elif len(i0) < _host_arith_max_points() or self.mirrored:
            out = self._host_gradient(i0, t)
        else:
            out = self._device_gradient(i0, t)
        if (~inside).any():
            out[~inside] = 0.0
        if self.static_shapes:
            voxel = self.distance(points)
            analytic = analytic_distance(points, self.static_shapes)
            wins = analytic <= voxel + 1e-12
            if wins.any():
                out[wins] = analytic_gradient(points, self.static_shapes)[wins]
        return out

    def label(self, points: np.ndarray) -> np.ndarray:
        """`EsdfField.label` 와 같은 정의 — 가장 가까운 격자점의 라벨, 격자 밖은 `-1`."""
        if self._labels is None:
            return np.full(len(np.asarray(points).reshape(-1, 3)), -1, np.int32)
        import torch

        i0, t, inside = self._lattice(points)
        if not len(i0):
            return np.zeros(0, np.int32)
        n = np.asarray(self.grid.shape) - 1
        idx = np.minimum(i0 + np.round(t).astype(np.int64), n)
        mirror = getattr(self, "_mirror_labels", None)
        if mirror is not None:
            out = mirror[self._linear_host(idx)].astype(np.int32)      # T43 TP host mirror
        else:
            lin_d = torch.as_tensor(self._linear_host(idx), device=self._values.device)
            out = torch.index_select(self._labels.reshape(-1), 0,
                                     lin_d).cpu().numpy().astype(np.int32)
        out[~inside] = -1
        return out

    def __repr__(self) -> str:  # dataclass repr 은 격자를 읽는다 — 그러지 않는다
        return (f"DeviceEsdfField(grid={self.grid!r}, device={self.device}, "
                f"max_distance={self.max_distance!r}, labels={self.has_labels})")

    __eq__ = object.__eq__
    __hash__ = object.__hash__


def layer_from_device(values, origin: np.ndarray, voxel_size: float, *,
                      labels=None, label_names: tuple = (),
                      outside_distance: Optional[float] = None,
                      stats: Optional[dict] = None) -> DeviceEsdfField:
    """`layer_from_arrays` 의 GPU 판. `(값 격자 tensor, 첫 복셀 중심, 복셀 크기)` → 한 계층.

    `max_distance` 는 `layer_from_arrays` 와 같은 정의 (유한 값의 `|d|` 최대) 를 GPU 에서 잰다.
    """
    import torch

    if not isinstance(values, torch.Tensor):
        raise TypeError(f"values 는 torch.Tensor 여야 합니다: {type(values).__name__}")
    if values.dim() != 3:
        raise ValueError(f"거리 격자는 3차원이어야 합니다: {tuple(values.shape)}")
    grid = VoxelGrid(origin=np.asarray(origin, np.float64).reshape(3),
                     shape=tuple(int(v) for v in values.shape), voxel_size=float(voxel_size))
    return DeviceEsdfField(grid, values.to(torch.float32), labels=labels,
                           label_names=label_names, stats=stats,
                           outside_distance=outside_distance)
