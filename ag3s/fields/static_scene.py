"""정적 기하를 **누가 주는가** — 해석적 채널에 실을 도형 목록을 만드는 곳 (A1).

거리장은 카메라가 **본 것**만 안다. 안 본 곳과 격자 밖은 낙관적으로 "멀다" 고 답하고
(발견 E4), `EsdfField.static_shapes` 가 `min(복셀, 해석적)` 으로 그 낙관을 막는다 (N2).
그 채널에 실을 도형이 어디서 오는가가 이 모듈이다.

**AG3S 도 `SafePolicy` 도 무엇이 고정 기하인지 스스로 알 수 없다.** `phase` · 목적지와 같은
주입 계약이다. 그래서 출처를 둘 둔다:

* **`from_mujoco`** — 시뮬레이션. MJCF 가 정본이므로 거기서 뽑는다.
* **`load` / `dump`** — 실기. 배포 기계에 MuJoCo 가 없어도 돌아야 하므로 JSON 이 계약이다.
  `dump(from_mujoco(...))` 로 시뮬에서 뽑은 것을 그대로 실기 파일로 굳힐 수 있다.

이 모듈 자체는 **순수 numpy** 다 — `mujoco` 임포트는 `from_mujoco` 안에서만 일어난다.

### 무엇을 뽑고 무엇을 버리는가 (실측으로 정한 규칙, `run_0004`)

| 갈래 | 판정 | 왜 |
|---|---|---|
| free joint 가 달린 body | **버린다** | 움직인다. 사과·바나나·귤·배·상자(목적지) 53 geom. 관측이 답할 몫이다 |
| 로봇 body | **버린다** | 자기 필터가 따로 본다. 1,238 geom |
| `contype == 0 and conaffinity == 0` | **버린다** | 시각 전용 29 geom. `vis` 복제본뿐 아니라 **`left_ee_target`/`right_ee_target` 마커가 로봇 작업공간 한가운데 (0.5, 0, 0.5) 에 있다** — 안 거르면 없는 장애물을 만들어 넣는다 |
| 나머지 중 box · plane | **뽑는다** | 16 개 — 바닥 1, 테이블 5, 선반 6, 벽 4 |
| 나머지 중 구 · 메시 · 캡슐 | **버리고 센다** | 해석적 채널이 상자와 반공간만 안다. 말없이 빠지면 안 되므로 `SurveyStats.unsupported` 에 남긴다 |

### 좌표계

도형은 **로봇 base 좌표계**로 나와야 한다 — 거리장과 로봇 구가 거기 있기 때문이다.
`run_0004` 에서 base 는 44 프레임 동안 월드 원점에서 **0.16 mm** 움직였으므로 월드와 같다고
보아도 된다. 베이스가 실제로 움직이는 순간 그 가정이 깨지므로 `T_world_base` 를 열어 둔다
(`build_constraint_robot_model` 이 바퀴·베이스 구를 빼면서 둔 가정과 **같은 가정**이다).

### 이 도형들이 실제로 제약을 만드는가 (`run_0004` 15 프레임, 양팔 120 구, margin 50 mm)

```
table_top   -62.4 mm   <- 유일하게 음수. 사과를 집으러 내려가는 프레임 7~11
table_leg_*  +38 ~ +359 mm
floor       +758 mm
shelf_*     +963 ~ +1,232 mm
office_wall_*  +2,612 ~ +3,256 mm
```

**16 개 중 15 개는 이번 롤아웃에서 제약 행을 하나도 안 만든다 — 보험이지 비용이 아니다.**
단, 이것은 제약 모델이 **양팔**일 때다. 전신 194 구로 재면 바닥이 `base` −342 mm ·
`wheel_l/r` −108 mm 를 15 프레임 내내 상수로 깔고, 그 행은 어떤 해도 못 푼다.
그 위험은 `build_constraint_robot_model(link_filter=ARM_LINKS)` 가 이미 막고 있고,
**`--links all` 로 도는 순간 되살아난다.**
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, Optional, Sequence

import numpy as np

from benchmark.ag3s.fields.esdf import StaticBox, StaticPlane

__all__ = [
    "SurveyStats", "from_mujoco", "to_jsonable", "from_jsonable", "load", "dump",
]


@dataclasses.dataclass(frozen=True)
class SurveyStats:
    """무엇을 뽑았고 **무엇을 왜 버렸는가.** 말없이 빠지는 것이 없게 하려고 센다."""

    n_shapes: int = 0
    #: free joint 가 달린 body 의 geom 수 — 움직이므로 관측이 답할 몫이다.
    n_free: int = 0
    #: 로봇 body 의 geom 수 — 자기 필터가 본다.
    n_robot: int = 0
    #: `contype == conaffinity == 0` 인 geom 수 — 시각 전용 · 마커.
    n_visual: int = 0
    #: 상자·반공간이 아니라 못 싣는 것. `(이름, mjtGeom)` — 비어 있지 않으면 **읽어야 한다**.
    unsupported: tuple = ()

    def summary(self) -> str:
        line = (f"static_scene: {self.n_shapes} 도형 (free {self.n_free} · robot {self.n_robot}"
                f" · visual {self.n_visual} 제외)")
        if self.unsupported:
            line += f" · 미지원 {len(self.unsupported)}: " + ", ".join(
                f"{n}({t})" for n, t in self.unsupported[:6])
        return line


def _is_robot_body(name: str, prefixes: Sequence[str]) -> bool:
    return bool(name) and name.startswith(tuple(prefixes))


#: 로봇 body 이름의 머리. `experiments.mujoco_source.ROBOT_BODY_PREFIXES` 와 같은 값이되,
#: 이 모듈이 `experiments` 에 의존하지 않도록 여기 둔다 — 라이브러리가 실험 코드를 임포트하면
#: 실기 배포에서 mujoco·matplotlib 까지 딸려온다.
DEFAULT_ROBOT_PREFIXES = (
    "base", "link_", "wheel", "ee_", "EE_", "FT_", "d435i", "wrist_bracket", "zed",
)


def from_mujoco(model, data=None, *, T_world_base: Optional[np.ndarray] = None,
                robot_prefixes: Sequence[str] = DEFAULT_ROBOT_PREFIXES,
                require_collidable: bool = True) -> tuple[list, SurveyStats]:
    """MJCF 에서 정적 도형을 뽑는다. `(shapes, stats)`.

    Args:
        model: `mujoco.MjModel`.
        data: `mujoco.MjData`. 도형의 **월드 자세**는 `data.geom_xpos`/`geom_xmat` 에 있으므로
            `mj_forward` 가 한 번은 돌아 있어야 한다. None 이면 새로 만들어 `mj_forward` 를 부른다.
        T_world_base: `(4, 4)` 월드 -> base. None 이면 월드 = base 로 본다 (모듈 머리말 참고).
        robot_prefixes: 로봇으로 볼 body 이름의 머리.
        require_collidable: `contype`/`conaffinity` 가 둘 다 0 인 geom 을 버릴지. **끄지 말 것** —
            끄면 `left_ee_target` 같은 시각 마커가 작업공간 한가운데 장애물로 들어온다.
    """
    import mujoco

    if data is None:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

    if T_world_base is None:
        R_bw = np.eye(3)
        t_bw = np.zeros(3)
    else:
        T = np.asarray(T_world_base, float).reshape(4, 4)
        R_wb, t_wb = T[:3, :3], T[:3, 3]
        R_bw = R_wb.T                 # base <- world
        t_bw = -R_bw @ t_wb

    free_bodies = {int(model.jnt_bodyid[i]) for i in range(model.njnt)
                   if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE}

    shapes: list = []
    n_free = n_robot = n_visual = 0
    unsupported: list[tuple[str, int]] = []

    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"

        if _is_robot_body(body_name, robot_prefixes):
            n_robot += 1
            continue
        if body in free_bodies:
            n_free += 1
            continue
        if require_collidable and not (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])):
            n_visual += 1
            continue

        pos = R_bw @ np.asarray(data.geom_xpos[g], float) + t_bw
        rot = R_bw @ np.asarray(data.geom_xmat[g], float).reshape(3, 3)
        label = f"{body_name}/{geom_name}" if body_name else geom_name
        kind = model.geom_type[g]

        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            shapes.append(StaticBox(center=pos,
                                    half_extents=np.asarray(model.geom_size[g], float).copy(),
                                    rotation=rot, label=label))
        elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
            # MuJoCo 평면의 법선은 geom 프레임의 +z 이고 자유공간 쪽을 가리킨다 —
            # `StaticPlane.normal` 의 규약과 같다.
            shapes.append(StaticPlane(point=pos, normal=rot[:, 2].copy(), label=label))
        else:
            unsupported.append((label, int(kind)))

    stats = SurveyStats(n_shapes=len(shapes), n_free=n_free, n_robot=n_robot,
                        n_visual=n_visual, unsupported=tuple(unsupported))
    return shapes, stats


# ------------------------------------------------------------------ JSON 계약 (실기)


def to_jsonable(shapes: Sequence[Any]) -> dict:
    """도형 목록 -> JSON 으로 쓸 수 있는 dict. `from_jsonable` 의 역이다."""
    out = []
    for s in shapes:
        if isinstance(s, StaticBox):
            out.append({"type": "box", "label": s.label,
                        "center": np.asarray(s.center, float).ravel().tolist(),
                        "half_extents": np.asarray(s.half_extents, float).ravel().tolist(),
                        "rotation": np.asarray(s.rotation, float).reshape(3, 3).tolist()})
        elif isinstance(s, StaticPlane):
            out.append({"type": "plane", "label": s.label,
                        "point": np.asarray(s.point, float).ravel().tolist(),
                        "normal": np.asarray(s.normal, float).ravel().tolist()})
        else:
            raise TypeError(f"to_jsonable 은 StaticBox 와 StaticPlane 만 받습니다: {type(s)}")
    return {"frame": "robot_base", "units": "m", "shapes": out}


def from_jsonable(payload: dict) -> list:
    """`to_jsonable` 이 만든 dict -> 도형 목록.

    `frame` 이 `robot_base` 가 아니면 거절한다. 좌표계가 틀린 도형은 **조용히** 엉뚱한 자리를
    막고, 그 결과는 "최적화기가 왜 여기서 멈추지" 로만 보인다.
    """
    frame = payload.get("frame", "robot_base")
    if frame != "robot_base":
        raise ValueError(
            f"정적 기하는 로봇 base 좌표계여야 합니다 (frame={frame!r}). 월드 좌표로 받았다면 "
            "from_mujoco(T_world_base=...) 로 변환해서 다시 dump 하십시오")
    shapes: list = []
    for i, s in enumerate(payload.get("shapes", ())):
        kind = s.get("type")
        label = str(s.get("label", f"static_{i}"))
        if kind == "box":
            rot = s.get("rotation")
            shapes.append(StaticBox(
                center=np.asarray(s["center"], float),
                half_extents=np.asarray(s["half_extents"], float),
                rotation=np.eye(3) if rot is None else np.asarray(rot, float).reshape(3, 3),
                label=label))
        elif kind == "plane":
            shapes.append(StaticPlane(point=np.asarray(s["point"], float),
                                      normal=np.asarray(s["normal"], float), label=label))
        else:
            raise ValueError(f"알 수 없는 정적 도형 type={kind!r} (box | plane)")
    return shapes


def dump(path: str | pathlib.Path, shapes: Sequence[Any], *, note: str = "") -> pathlib.Path:
    """도형 목록을 JSON 으로 굳힌다. 시뮬에서 뽑은 것을 실기로 옮기는 길이다."""
    payload = to_jsonable(shapes)
    if note:
        payload["note"] = note
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return p


def load(path: str | pathlib.Path) -> list:
    """JSON 에서 도형 목록을 읽는다. 실기 경로의 기본 출처 — MuJoCo 를 요구하지 않는다."""
    return from_jsonable(json.loads(pathlib.Path(path).read_text()))
