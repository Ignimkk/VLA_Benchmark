"""3D 결과를 **정책이 실제로 본 이미지** 위에 되돌려 그리기.

위에서 내려다본 그림은 정확하지만 사람이 보는 공간이 아니다. 클러스터가 사과 위에 앉았다는
것을, 도형이 물체 뒤를 놓쳤다는 것을, 후보 원이 어디를 덮는지를 — **입력 이미지 위에서**
확인할 수 있어야 한다. 이 모듈이 그 되돌리기를 한 곳에서 담당한다.

두 가지 규약이 여기 모여 있다.

**정규화 좌표로 옮긴다.** 깊이·내부 파라미터는 480x640이고 정책 이미지는 224x224다. 둘 다
4:3이고, `render_cam(match_rby1_dataset=True)`의 299x224 → 224x224 변환은 crop 도 pad 도 아닌
순수 가로 압축이므로 **정규화 좌표가 보존된다** (2단계에서 확인). 그래서 `K`로 얻은 픽셀을
`u/W_depth`, `v/H_depth`로 정규화한 뒤 정책 이미지 크기를 곱하면 된다. 오프셋이 없다.

**구의 실루엣은 원으로 근사한다.** 핀홀 카메라에서 구의 실루엣은 엄밀히는 타원이고, 광축에서
멀수록 커진다. 여기서는 중심을 투영한 자리에 반지름 `f*r/z` 인 원을 그린다 — 광축 위에서는
정확하고, 이 씬의 물체는 화면 중앙 근처라 오차가 작다. **판정에 쓰는 숫자는 언제나 3D에서
계산하고**, 이 근사는 그림에만 쓴다.
"""

from __future__ import annotations

import numpy as np


def project(points: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
            depth_hw: tuple[int, int], image_hw: tuple[int, int]):
    """base 프레임 점 -> 정책 이미지 픽셀 `(u, v)` 와 카메라 z. 카메라 뒤의 점은 마스크로 제외."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    T = np.asarray(T_base_cam, np.float64)
    cam = (pts - T[:3, 3]) @ T[:3, :3]
    z = cam[:, 2]
    ok = z > 1e-6
    u = np.full(len(pts), np.nan)
    v = np.full(len(pts), np.nan)
    K = np.asarray(K, np.float64)
    u[ok] = K[0, 0] * cam[ok, 0] / z[ok] + K[0, 2]
    v[ok] = K[1, 1] * cam[ok, 1] / z[ok] + K[1, 2]
    # 정규화 좌표를 거쳐 정책 이미지 크기로. 오프셋 없음 — 위 docstring 참조.
    u = u / depth_hw[1] * image_hw[1]
    v = v / depth_hw[0] * image_hw[0]
    return u, v, z, ok


def sphere_circle(centre: np.ndarray, radius: float, K: np.ndarray, T_base_cam: np.ndarray,
                  depth_hw: tuple[int, int], image_hw: tuple[int, int]):
    """구를 이미지 위의 `(u, v, r_px)` 원으로. 카메라 뒤면 None."""
    u, v, z, ok = project(np.asarray(centre).reshape(1, 3), K, T_base_cam, depth_hw, image_hw)
    if not ok[0]:
        return None
    scale = image_hw[1] / depth_hw[1]
    r_px = float(np.asarray(K, np.float64)[0, 0] * float(radius) / float(z[0]) * scale)
    return float(u[0]), float(v[0]), r_px


def show(ax, image: np.ndarray, title: str = "", *, ink="#0b0b0b", grid_ink="#d8d7d2"):
    """관측 이미지를 축에 깔고 눈금을 지운다."""
    ax.imshow(image)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(grid_ink)
    if title:
        ax.set_title(title, fontsize=9.5, color=ink)
    return ax


__all__ = ["project", "show", "sphere_circle"]
