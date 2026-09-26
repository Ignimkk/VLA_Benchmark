"""MuJoCo **참값** — 기록에 남길 로봇 자세와 물체 자세.

`T5`/`T6` 가 세 번 연속 같은 자리에서 막혔다. 기록에 로봇 관절도 물체 자세도 없어서
*"과제가 실제로 완결됐나"* 와 *"영구히 멈춘 자리가 어디인가"* 를 **측정할 방법이 없었다.**
그 둘은 지각이 추정할 값이 아니라 시뮬레이터가 이미 알고 있는 값이다 — 그래서 여기서 그냥
읽어 온다.

## 왜 별 모듈인가

기록기(`frame_record.py`)는 MuJoCo 를 모른다. 알게 만들면 기록기가 **씬을 읽는 쪽**이 되고,
그러면 기록 형식을 검사하는 테스트가 MuJoCo 모델을 필요로 한다. 반대로 `pi05_infer.py` 에
직접 박으면 그 파일 말고는 아무도 테스트할 수 없다 (1700 줄이고 정책 서버가 있어야 돈다).
그래서 **씬에서 값을 뽑는 일만** 하는 이 모듈이 가운데 있다.

## 기록하지 않을 때는 아무 비용도 들지 않는다

`ObjectPoseProbe` 는 **생성될 때 한 번** body id 를 찾고, 그 뒤로는 `data.xpos` 를 인덱싱할
뿐이다. 그리고 그 생성 자체가 `--record-frames` 안쪽에서만 일어난다 — 기록하지 않는 실행은
이 모듈을 import 조차 하지 않는다. `tests/ag3s/test_record_cost_guard.py` 가 호출부에서
그것을 고정한다.

**id 를 매 프레임 다시 찾지 않는 이유**는 비용보다 정직함이다. `mj_name2id` 는 없는 이름에
`-1` 을 주는데, 매 프레임 그것을 물으면 *"이 프레임에는 사과가 없었다"* 와 *"이 모델에는
사과가 처음부터 없다"* 가 구별되지 않는다. 한 번 찾아 `missing` 으로 들고 있으면 그것이
manifest 에 한 줄로 남는다.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["DEFAULT_OBJECT_BODIES", "ObjectPoseProbe", "joint_positions"]

#: 과일 넷 + crate. 이름은 `rby1_manipulation.simulation.transport_scene` 의
#: `OBJECT_BODIES` · `CRATE_BODY` 와 같고, **그 모듈을 import 하지 않는다** — 그쪽은
#: `pi05_TO_hybrid/rby1_manipulation/src` 에 있어 benchmark 패키지에서는 sys.path 가
#: 맞을 때만 보인다. 이름 다섯 개를 복제하는 대가로 이 모듈이 어디서나 돈다.
#:
#: 이 목록에 없는 body 를 재려면 `bodies=` 로 넘긴다. 목록을 늘리는 것이 아니라 **넘기는**
#: 것인 이유는, 기본값이 조용히 자라면 옛 기록과 새 기록의 키 집합이 갈라지기 때문이다.
DEFAULT_OBJECT_BODIES: tuple[str, ...] = ("apple", "banana", "orange", "pear", "crate")


def joint_positions(data: Any) -> list[float]:
    """`data.qpos` 를 **평범한 float 리스트**로.

    `ndarray` 를 그대로 기록기에 넘기면 `json.dumps(..., default=str)` 가 그것을 **문자열
    repr** 로 적는다 — `"[0.1 0.2 ... 0.9]"` 처럼 줄임표가 박힌, 되읽을 수 없는 값이다.
    예외가 나지 않고 통과하므로 기록이 망가진 것을 아무도 모른다. 그래서 변환을 호출부에
    맡기지 않고 여기서 한다.

    **자유물체까지 포함한 `qpos` 전체**다. 팔 관절만 남기지 않는 것은, 물체가 어디 있었는지를
    `object_poses` 와 두 경로로 확인할 수 있어야 하기 때문이다 (하나는 body 자세, 하나는
    joint 좌표 — 둘이 어긋나면 그것 자체가 신호다).
    """
    return [float(v) for v in data.qpos]


class ObjectPoseProbe:
    """과일 넷 + crate 의 MuJoCo body 자세를 프레임마다 읽는다.

    Args:
        model: `mujoco.MjModel`.
        bodies: 잴 body 이름. 기본은 `DEFAULT_OBJECT_BODIES`.

    없는 body 는 **예외가 아니다.** 씬 XML 은 실행마다 다르고(장애물 profile 에 따라 셋이
    있다), 없는 물체 하나 때문에 기록이 통째로 죽는 것은 기록기가 할 일이 아니다. 대신
    `missing` 에 남고 `describe()` 가 그것을 manifest 로 내보낸다 — *"안 쟀다"* 와
    *"재려고 했는데 없었다"* 가 구별되어야 한다.
    """

    def __init__(self, model: Any, bodies: Sequence[str] = DEFAULT_OBJECT_BODIES):
        import mujoco

        self._body_ids: dict[str, int] = {}
        missing: list[str] = []
        for name in bodies:
            bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name)))
            if bid < 0:
                missing.append(str(name))
            else:
                self._body_ids[str(name)] = bid
        #: 이 모델에 없던 이름들. 빈 목록이 정상이다.
        self.missing: tuple[str, ...] = tuple(missing)

    @property
    def present(self) -> tuple[str, ...]:
        """실제로 재는 body 이름. 기록의 `object_poses` 키 집합이 이것이다."""
        return tuple(self._body_ids)

    def describe(self) -> dict[str, Any]:
        """manifest 에 실을 한 덩어리 — **무엇을 쟀고 무엇이 없었나.**"""
        return {
            "bodies": list(self._body_ids),
            "body_ids": dict(self._body_ids),
            "missing": list(self.missing),
            "frame": "MuJoCo world frame (data.xpos / data.xquat), metres and unit quaternion",
        }

    def poses(self, data: Any) -> dict[str, dict[str, list[float]]]:
        """`{body: {"pos": [x, y, z], "quat": [w, x, y, z]}}` — 이 프레임의 참값.

        `data.xpos`/`data.xquat` 를 읽는다. `mj_step` 이 끝나면 정방향 기구학이 이미 돌아
        있으므로 **여기서 `mj_forward` 를 부르지 않는다** — 부르면 기록이 시뮬레이션 상태를
        건드리게 되고, 기록 때와 기록 안 할 때의 물리가 갈라진다.

        자세까지 싣는 이유: 위치만으로는 과일이 crate 에 **들어갔는지** 와 옆에 **넘어져
        있는지** 가 구별되지 않는다.
        """
        out: dict[str, dict[str, list[float]]] = {}
        for name, bid in self._body_ids.items():
            out[name] = {
                "pos": [float(v) for v in data.xpos[bid]],
                "quat": [float(v) for v in data.xquat[bid]],
            }
        return out
