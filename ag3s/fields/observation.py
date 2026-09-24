""""이 점이 관측됐는가" — 카메라 depth 로 직접 답한다.

`block-sparse` TSDF 는 "보고 지나간 자유 공간" 과 "한 번도 안 본 곳" 을 구분하지 못한다
(둘 다 미할당). 그래서 legacy dense 필드의 `unknown_fraction`(가중치 0 인 복셀의 비율)에
대응하는 값이 없고, 그것이 G2(미관측 100 % 필드가 `valid`·`certified True` 로 보고됐다)의
축이었다. F16(블록-스파스로 가면 "할당 안 한 곳을 물으면 뭐라 답하는가" 가 안전 문제가 된다)이
예고한 자리다.

**대신 질의점에서 직접 답한다** (2026-09-22 판정). 필드를 거치지 않고 원본 depth 로 재므로
backend 와 무관하고, 재는 대상이 *격자 전체*가 아니라 **로봇이 실제로 지나가는 곳**이다 —
그쪽이 안전 판정이 필요한 유일한 곳이고, 실측으로도 격자의 7.29 % 만 질의된다 (Step 8 의
격자 활용률).

## 판정식

한 카메라가 점 `p` 를 관측했다는 것은 네 가지가 모두 참인 것이다.

    1. `p` 를 그 카메라로 투영한 픽셀이 영상 안에 있다
    2. 카메라 좌표의 깊이 `z` 가 `[depth_min, depth_max]` 안에 있다
    3. 그 픽셀의 depth 가 **유효**하다 (`> 0`). 0 은 센서가 아무것도 못 준 것이고, 그 광선은
       아무것도 알려주지 않는다. **로봇 마스크는 여기서 적용하지 않는다** — 아래 클래스
       머리말의 이유
    4. `z <= depth[u, v] + truncation` — 즉 `p` 가 관측된 표면 **앞**이거나 절단대역 안이다.
       표면 뒤는 가려진 것이고, 가려진 것은 자유가 아니라 **미관측**이다

여러 카메라 중 **하나라도** 관측했으면 관측된 것으로 본다 (`OR`). 이것이 dense TSDF 의
가중치가 쌓이는 방식과 같다.

## 부호가 안전한 쪽인가

이 판정은 틀릴 때 **미관측 쪽으로** 틀린다. 3 번이 센서가 못 준 픽셀을 미관측으로 세고,
4 번이 절단대역 밖의 표면 뒤를 미관측으로 센다. 미관측을 위험으로 취급하는 것이 fail-closed
이므로 보수적 방향이다.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["ObservationProbe", "probe_from_cameras"]


@dataclasses.dataclass
class ObservationProbe:
    """카메라 depth 를 들고 `observed(points)` 에 답한다. 순수 numpy.

    depth 는 **마스킹하지 않은 것**을 받는다. 이유가 하나인데 처음에 반대로 만들어 실측에서
    잡았으므로 적어 둔다.

    로봇 마스크의 목적은 로봇을 **필드에 넣지 않는 것**이다. 가림 판정의 목적은 그 광선에서
    **첫 표면이 어디였나**를 아는 것이고, 둘은 다른 질문이다. 마스킹한 depth 를 넘기면 로봇
    위의 점이 전부 0 픽셀에 투영되어 "미관측" 이 되는데, 그것은 동어반복이다 — 로봇이
    자기를 지웠으니 자기가 안 보인다고 답하는 것이다. 실측: 마스킹한 depth 로는 구 120 개 중
    **113 개**가 미관측으로 나왔다.

    raw depth 를 쓰면 판정이 맞는다. 로봇이 첫 표면인 광선에서 로봇 표면 위의 점은
    `z ~= depth` 를 만족해 관측으로, 그 **뒤**의 점은 `z > depth + truncation` 이라 미관측으로
    나온다. 후자가 이 검사가 실제로 잡아야 하는 것이다 — 팔이 가린 부피를 자유로 읽는 것.
    """

    #: `(이름, depth(H,W) m, K(3,3), T_base_cam(4,4))`
    cameras: tuple
    depth_min: float
    depth_max: float
    #: 표면 뒤 얼마까지를 아직 관측으로 볼 것인가. TSDF 절단대역과 같은 값을 쓴다.
    truncation: float

    def per_camera(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """`{카메라 이름: (N,) bool}` — 그 카메라가 각 점을 관측했는가."""
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        out: dict[str, np.ndarray] = {}
        for name, depth, K, T in self.cameras:
            out[str(name)] = self._one(pts, np.asarray(depth, np.float64),
                                       np.asarray(K, np.float64),
                                       np.asarray(T, np.float64))
        return out

    def observed(self, points: np.ndarray) -> np.ndarray:
        """`(N,)` bool — 어느 카메라든 하나라도 관측했는가."""
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        if not len(pts) or not self.cameras:
            return np.zeros(len(pts), bool)
        hit = np.zeros(len(pts), bool)
        for flags in self.per_camera(pts).values():
            hit |= flags
        return hit

    def _one(self, pts: np.ndarray, depth: np.ndarray, K: np.ndarray,
             T: np.ndarray) -> np.ndarray:
        h, w = depth.shape
        # base -> camera. `T_base_cam` 은 camera -> base 이므로 뒤집는다.
        R = T[:3, :3]
        t = T[:3, 3]
        cam = (pts - t) @ R          # R^T (pts - t) 를 행벡터로 쓴 것
        z = cam[:, 2]
        ok = (z >= self.depth_min) & (z <= self.depth_max)
        out = np.zeros(len(pts), bool)
        if not ok.any():
            return out
        zz = z[ok]
        u = np.rint(cam[ok, 0] / zz * K[0, 0] + K[0, 2]).astype(np.int64)
        v = np.rint(cam[ok, 1] / zz * K[1, 1] + K[1, 2]).astype(np.int64)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not inside.any():
            return out
        idx = np.flatnonzero(ok)[inside]
        d = depth[v[inside], u[inside]]
        # 3: depth 가 유효한가 (0 = 무효 또는 로봇 마스크). 4: 표면 앞이거나 절단대역 안인가.
        good = (d > 0.0) & (zz[inside] <= d + self.truncation)
        out[idx] = good
        return out

    # -- 정보용: 절두체 기반 관측 부피 -------------------------------------------------
    def observed_volume(self, lower, upper, *, voxel_size: float,
                        stride: int = 2) -> dict:
        """작업공간 상자에서 관측된 **부피**. 판정에 쓰지 않고 기록에만 쓴다 (2026-09-22 판정).

        `stride` 로 솎아 센다 — 전 복셀을 세 카메라에 투영하는 것이 E6(TSDF 적분이 전 복셀
        중심을 매 카메라·매 프레임 투영한다)에서 비싸다고 판정된 바로 그 연산이라, 인증에
        쓰지 않는 값에 그 비용을 쓸 이유가 없다. 추정값임을 `stride` 와 함께 싣는다.
        """
        lower = np.asarray(lower, np.float64).reshape(3)
        upper = np.asarray(upper, np.float64).reshape(3)
        step = float(voxel_size) * max(int(stride), 1)
        axes = [np.arange(lower[i] + 0.5 * step, upper[i], step) for i in range(3)]
        if any(len(a) == 0 for a in axes):
            return {"n_probes": 0, "observed_fraction": None, "stride": int(stride)}
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        probes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        seen = self.observed(probes)
        cell = step ** 3
        return {
            "n_probes": int(len(probes)),
            "stride": int(stride),
            "probe_spacing_m": step,
            "observed_fraction": float(seen.mean()),
            "observed_volume_m3": float(seen.sum()) * cell,
            "box_volume_m3": float(np.prod(np.maximum(upper - lower, 0.0))),
            "definition": ("절두체 판정을 격자에서 solid 로 솎아 센 추정값. 인증 기준이 "
                           "아니다 — 인증은 swept volume 의 질의점으로 한다"),
        }


def probe_from_cameras(cameras: Sequence[Any], *, depth_min: float, depth_max: float,
                       truncation: float) -> Optional[ObservationProbe]:
    """`CameraDepth` 목록 -> `ObservationProbe`.

    **로봇 마스크를 적용하지 않는다.** 클래스 머리말의 이유 — 마스킹한 depth 를 쓰면 로봇
    위의 점이 전부 미관측으로 나온다 (실측 113/120).
    """
    packed = []
    for cam in cameras or ():
        depth = np.asarray(cam.depth, np.float64)
        packed.append((str(cam.name), depth,
                       np.asarray(cam.camera_intrinsics, np.float64),
                       np.asarray(cam.T_base_cam, np.float64)))
    if not packed:
        return None
    return ObservationProbe(tuple(packed), depth_min=float(depth_min),
                            depth_max=float(depth_max), truncation=float(truncation))
