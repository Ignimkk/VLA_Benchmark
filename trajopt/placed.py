"""성공 판정 — 조작 대상이 목적지에 놓였는가 (A3).

`GraspLatch` 의 **풀기**는 두 갈래다. `placed`(외부가 주는 성공 판정)와, 그리퍼가 몇 프레임
연속 열려 있는 **폴백**. `SafePolicy(placed_fn=...)` 가 앞엣것을 받도록 열려 있는데 넘기는
호출자가 없어 폴백만 돌고 있었다 (A3).

**AG3S 도 `SafePolicy` 도 스스로 판단하지 않는다.** 바구니의 내경이 얼마이고 테두리가 어디인지
는 과제를 아는 쪽만 안다 — `phase` · 목적지 · `attach`/`detach` 시점 · 정적 기하(A1)와 같은
**주입 계약**이다. 이 모듈은 그 계약을 채우는 **참조 구현**이고, 실기에서는 같은 자리에 자기
판정식을 끼우면 된다.

### 무엇으로 판정하는가 — 관측만 쓴다

`constraint_set` 안에 이미 있는 것만 본다. MuJoCo 참값을 쓰지 않으므로 실기에서 그대로 돈다:

* **쥔 물체가 지금 어디인가** — `attached` 의 점을 현재 자세로 옮긴 것 (A2 의
  `attached_points_in_base`, 최적화기가 질의하는 것과 **같은 점 집합**).
* **목적지가 어디인가** — 거리장의 **라벨 층**. 목적지로 라벨된 복셀들의 자리가 곧 목적지의
  범위다. 거리장은 원래 익명이라 이 질문에 답할 수 없었고, 라벨 층이 그 익명을 푼 것이다.

두 조건을 본다 — 기록의 실측 판정식에서 **기하 부분**이다:

    (쥔 물체가 목적지의 수평 범위 안) AND (쥔 물체가 테두리보다 아래)

### 세 번째 조건은 왜 여기 없는가

실측 판정식의 세 번째 조건은 **"손에서 이탈"**(손–사과 > 80 mm)이었다. 그것을 여기서 잴 수
없다: `attached` 는 파지 순간의 **스냅샷**이고 손에 강체로 붙어 있으므로, 손과의 거리가
정의상 변하지 않는다. 관측으로 그것을 보려면 물체를 다시 찾아내야 하는데, 그러면 F17(조작
대상 식별에 에피소드 상태가 없다)이 잠금을 만든 이유로 되돌아간다.

**그래서 이탈은 그리퍼가 답한다.** `GraspLatch` 가 이미 그리퍼를 보고 있으므로, 이 판정이
기하만 답하고 잠금이 둘을 **AND** 로 묶는다 — 기하만으로는 **그리퍼가 닫혀 있는 동안 절대
풀리지 않는다.** 그것이 없으면 물체가 바구니 위를 지나는 순간 아직 쥔 채로 detach 된다.

실측(`run_0004`, `experiments/a3_release_signal.py`): 성공 판정 프레임 19, 그리퍼 폴백
프레임 20 — **1 프레임 차이**이고 판정은 25 프레임 동안 한 번도 안 깜빡인다. 값어치는 정확도가
아니라 **보험**이다: 그리퍼가 놓지 않았는데 열리거나 놓았는데 안 열리면, A2 의 파내기가
손을 따라다니는 유령 구멍을 만든다 (실측 팔 구 낙관 **최대 +19.7 mm**).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

__all__ = ["destination_placement", "destination_extent"]


def destination_extent(field, label: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """목적지로 라벨된 복셀들의 `(하한, 상한)` 축정렬 상자. 라벨이 없으면 `None`.

    거리장은 원래 익명이라 "여기서 가장 가까운 것이 무엇인가" 를 못 말한다. 라벨 층이 그것을
    푼 덕분에 **목적지의 범위를 필드에서 직접 읽을 수 있다** — 따로 기하를 주입받지 않아도 된다.
    """
    grid = getattr(field, "label_grid", None)
    if grid is None:
        return None
    wanted = field.label_id(label)
    if wanted < 0:
        return None
    idx = np.argwhere(np.asarray(grid) == wanted)
    if idx.size == 0:
        return None
    # 복셀 인덱스 -> 복셀 **중심**. `esdf_origin` 이 코너가 아니라 중심이라는 cuRobo 함정과
    # 같은 자리라 여기서 반 칸을 더하고 빼지 않는다 — `VoxelGrid` 의 규약을 그대로 따른다.
    origin = np.asarray(field.grid.origin, np.float64)
    size = float(field.grid.voxel_size)
    centres = origin + idx.astype(np.float64) * size
    return centres.min(axis=0), centres.max(axis=0)


def destination_placement(
    *,
    robot_model,
    radial_tolerance: float = 0.0,
    below_rim: float = 0.060,
    min_points_inside: float = 0.5,
) -> Callable[[dict, Any], bool]:
    """`placed_fn(scene, constraint_set) -> bool` 을 만든다.

    **기하만 답한다.** 손에서 이탈했는지는 `GraspLatch` 의 그리퍼가 답하고, 잠금이 둘을 AND 로
    묶는다 (모듈 머리말 참고).

    Args:
        robot_model: `attach()` 가 쓴 것과 **같은** 제약 로봇 모델. 다른 모델로 되돌리면 쥔
            물체가 조용히 엉뚱한 자리에 놓인 것으로 판정된다.
        radial_tolerance: 목적지 수평 범위를 이만큼 **넓혀** 준다 (m). 0 이면 라벨된 복셀의
            수평 범위 그대로. 관측이 목적지의 일부만 볼 때 범위가 좁게 잡히므로 여유를 준다.
        below_rim: 테두리(목적지 라벨의 최고점)보다 이만큼 아래여야 한다 (m). 기본 60 mm 는
            실측값이고, 그 프레임의 실제 깊이는 82 mm 라 여유가 있다.
        min_points_inside: 쥔 물체의 질의점 중 이 비율 이상이 범위 안이어야 한다. 점 하나로
            판정하면 잡티 한 점이 판정을 뒤집는다.
    """
    from benchmark.ag3s.attached import attached_points_in_base

    def placed_fn(_scene: dict, constraint_set: Any) -> bool:
        attached = getattr(constraint_set, "attached", None)
        field = getattr(constraint_set, "esdf", None)
        label = getattr(constraint_set, "destination_label", None)
        if attached is None or field is None or not label:
            # 아직 쥐지 않았거나, 필드가 없거나, 목적지가 안 잡혔다. **판정하지 않는다** —
            # 모르는 것을 "놓았다" 로 읽으면 쥔 채로 detach 된다 (fail-closed).
            return False

        held = attached_points_in_base(
            attached, robot_model=robot_model,
            robot_state=np.asarray(constraint_set.robot_state, np.float64))
        if held is None or not len(held):
            return False

        extent = destination_extent(field, label)
        if extent is None:
            return False
        lo, hi = extent

        inside_xy = np.all(
            (held[:, :2] >= lo[:2] - radial_tolerance)
            & (held[:, :2] <= hi[:2] + radial_tolerance), axis=1)
        below = held[:, 2] <= float(hi[2]) - float(below_rim)
        return bool(float(np.mean(inside_xy & below)) >= float(min_points_inside))

    return placed_fn
