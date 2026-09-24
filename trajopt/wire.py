"""클라이언트와 서버가 주고받는 것의 **유일한 정의**.

포장과 해체를 같은 파일에 둔 이유는 하나다. 둘이 갈라지면 증상이 조용하다 — 키 이름 하나가
달라도 서버는 "카메라가 안 왔다" 며 기하 없이 계속 돌고, 로봇은 제약 없는 궤적을 안전하다고
믿으며 실행한다. 예외가 아니라 **더 위험한 정상 동작**으로 실패하는 종류의 버그다. 그래서
`pack_*` 와 `unpack_*` 은 짝을 이뤄 여기 있고, 테스트가 왕복을 검사한다.

## 요청

정책이 이미 쓰는 키(`state`, `images`, `prompt`)에 `ag3s/` 접두 키를 **더한다**. 접두사를
붙이는 것은 정책 입력 변환이 모르는 키를 만나 깨지지 않게 하려는 것이고, 서버에서 벗겨낸다.

| 키 | 내용 |
|---|---|
| `ag3s/cameras` | 카메라 이름 목록. 이 순서가 나머지 배열의 순서다 |
| `ag3s/depth/<cam>` | (H, W) uint16 밀리미터 — 실제 depth 카메라가 주는 것 |
| `ag3s/K/<cam>` | (3, 3) intrinsics |
| `ag3s/T_base_cam/<cam>` | (4, 4) extrinsics, **촬영 시점** |
| `ag3s/robot_state/<cam>` | (nq,) **촬영 시점** 관절. 카메라마다 다르다 |
| `ag3s/stamp/<cam>` | 촬영 시각 (초, 단조시계) |
| `ag3s/phase` | 조작 단계. AG3S 는 절대 추론하지 않고 주입받는다 |
| `ag3s/active_manipulators` | 접촉이 허용된 매니퓰레이터 |
| `ag3s/reset` | True 면 SEAM·AG3S·TO 의 내부 상태와 warm-start 를 모두 버린다 |
| `ag3s/seq` | 요청 일련번호. 응답에 그대로 돌아오고, 오래된 응답을 버리는 근거가 된다 |

카메라마다 `robot_state` 를 따로 싣는 것이 이 형식의 핵심이다. 손목 카메라는 팔과 함께
움직이므로 80 ms 전 프레임은 80 ms 전 자세에 놓여야 한다. 하나의 `q_now` 로 세 대를 변환하면
손목 클라우드가 번지고, 더 나쁘게는 자기 필터가 어긋나 로봇 점이 씬에 남아 그리퍼에 용접된
유령 장애물로 뭉친다.

## 응답

`actions` 외에 안전 판정을 싣는다. 로컬은 이 판정이 유효할 때만 실행한다.

| 키 | 내용 |
|---|---|
| `actions` | `[H, ACTION_WIDTH]` (16D 기본). 안전하지 않아도 실린다 — 왜 멈췄는지 보려면 무엇이 제안됐는지 알아야 한다 |
| `seq` | 요청의 일련번호를 그대로 돌려준다. 오래된 응답을 버리는 근거 |
| `timing_ms` | 단계별 시간 (서버 시계) |
| `ag3s_status` · `geometry_certified` · `trajopt_status` · `max_violation_m` · `safe` · `notes` | 안전 판정 |
| `field` | **거리장의 출처** — `sequence` · `backend` · `observed_at` · `state` · 계층. 아래 |

### `field` — 거리장이 언제 무엇으로 만들어졌는가

`actions` 만 보면 이 청크의 기하가 방금 관측된 것인지 낡은 것인지 알 방법이 없다. 2026-09-18
의 통합에서 `attach()` 뒤 **14 프레임 동안 지각이 한 번도 안 돌았는데 상태는 `ok`** 로
나갔고, 그것이 안 보인 이유가 이 블록이 없었기 때문이다.

`state` 는 `new` | `carried` | `stale` | `unavailable` 이다. **서버는 `new` 나 `unavailable`
만 찍는다** — planning frame 마다 AG3S 를 돌리므로. `carried` 와 `stale` 은 **control frame**
에서 생기고 (청크 하나가 8 스텝 = 533 ms 를 덮는다) 클라이언트가
`FieldProvenance.applied_by_client()` 로 채운다.

**age 의 기준은 `observed_at`(클라이언트 시계의 촬영 시각)이다.** 서버의 `built_at` 은
monotonic 원점이 달라 클라이언트가 자기 시계와 견줄 수 없으므로 단계 시간에만 쓴다.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "PREFIX", "DEFAULT_CAMERAS", "GRIPPER_COLUMNS", "gripper_columns",
    "ARM_JOINT_DIM", "ACTION_WIDTH",
    "pack_request", "strip_request", "unpack_camera_observations",
    "pack_response", "unpack_field", "SafetyVerdict",
]

PREFIX = "ag3s/"

#: 머리 하나 + 손목 둘. `CameraID` 에 HEAD 가 하나뿐이라 `zed_right` 는 `zed_left` 와 겹친다.
DEFAULT_CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")

#: 팔당 관절 수. 청크 레이아웃 전체가 이 값에서 나온다 — openpi 가 delta mask 를
#: ``make_bool_mask(N, -1, N, -1)`` 로 만들므로 (`training/config.py:278-281`) 레이아웃이
#: ``[왼팔 N, 왼 그리퍼, 오른팔 N, 오른 그리퍼]`` 이고 총 차원이 ``2*(N+1)`` 이다.
#:
#: **7 이 기본이다** — 2026-09-24 에 16D(`pi05_rby1_randomized_pick_place_16d_lora`)로 전환하고
#: 14D 를 버리기로 판정했다. 16D 는 `arm_6` 손목을 정책이 지령한다 (14D 에서는 고정이었다).
ARM_JOINT_DIM = 7

#: 액션의 총 차원. `2*(ARM_JOINT_DIM+1)`.
ACTION_WIDTH = 2 * (ARM_JOINT_DIM + 1)


def gripper_columns(arm_joint_dim: int = ARM_JOINT_DIM) -> tuple[int, int]:
    """그리퍼가 앉은 두 열. TO 는 이 열을 **절대 건드리지 않는다** — 결정 변수가 아니고,
    손을 여닫는 것은 충돌 회피가 판단할 일이 아니다.

    상수로 박지 않는 이유: 14D 는 `(6, 13)`, 16D 는 `(7, 15)` 다. 박아 두면 차원을 바꿀 때
    **엉뚱한 열을 그리퍼로 보호하고** 그것이 조용히 지나간다 — 손목 관절이 보호되고 그리퍼가
    최적화되는, 증상이 안 보이는 종류의 실패다.
    """
    n = int(arm_joint_dim)
    return (n, 2 * n + 1)


#: 기본 차원의 그리퍼 열. 편의용이고, 차원이 다른 호출자는 `gripper_columns(N)` 을 쓴다.
GRIPPER_COLUMNS = gripper_columns()

#: depth 를 uint16 밀리미터로 보내는 배율. 실제 depth 카메라가 주는 형식이고
#: `PointCloudConfig.depth_scale = 0.001` 이 그것을 되돌린다. float64 로 보내면 payload 가
#: 네 배가 되면서 모든 실제 센서가 갖는 1 mm 양자화를 조용히 숨긴다.
DEPTH_SCALE_MM = 1000.0


def pack_request(obs: dict[str, Any], *, cameras: Sequence[str],
                 depth: dict[str, np.ndarray], intrinsics: dict[str, np.ndarray],
                 extrinsics: dict[str, np.ndarray], robot_state: dict[str, np.ndarray],
                 stamps: dict[str, float], phase: str,
                 active_manipulators: Sequence[str] = (),
                 reset: bool = False, seq: int = 0) -> dict[str, Any]:
    """정책 관측에 AG3S 가 필요한 것을 더한다. `obs` 는 제자리에서 바뀌지 않는다."""
    out = dict(obs)
    out[PREFIX + "cameras"] = list(cameras)
    out[PREFIX + "phase"] = str(phase)
    out[PREFIX + "active_manipulators"] = list(active_manipulators)
    out[PREFIX + "reset"] = bool(reset)
    out[PREFIX + "seq"] = int(seq)
    for cam in cameras:
        d = np.asarray(depth[cam], np.float64)
        out[f"{PREFIX}depth/{cam}"] = np.clip(
            np.rint(d * DEPTH_SCALE_MM), 0, 65535).astype(np.uint16)
        out[f"{PREFIX}K/{cam}"] = np.asarray(intrinsics[cam], np.float64)
        out[f"{PREFIX}T_base_cam/{cam}"] = np.asarray(extrinsics[cam], np.float64)
        out[f"{PREFIX}robot_state/{cam}"] = np.asarray(robot_state[cam], np.float64)
        out[f"{PREFIX}stamp/{cam}"] = float(stamps[cam])
    return out


def strip_request(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(정책이 볼 관측, AG3S 가 볼 필드)` 로 가른다.

    정책 입력 변환은 자기가 모르는 키를 만나면 깨진다. 그래서 서버가 가장 먼저 하는 일이 이
    분리이고, 정책은 접두 키를 **한 번도 보지 않는다**.
    """
    policy_obs = {k: v for k, v in request.items() if not k.startswith(PREFIX)}
    scene = {k[len(PREFIX):]: v for k, v in request.items() if k.startswith(PREFIX)}
    return policy_obs, scene


def unpack_camera_observations(scene: dict[str, Any], *, robot_model,
                               attention: Optional[dict[str, np.ndarray]] = None) -> list:
    """`CameraObservation` 목록. 카메라가 하나도 없으면 빈 목록 — 예외가 아니다.

    카메라 누락은 정상적으로 일어난다(드라이버 한 프레임 드롭). 여기서 예외를 내면 지각 결함이
    정책을 통째로 죽인다. 빈 목록이면 AG3S 가 기하 없음을 **상태로** 보고하고, 그 상태를 보고
    실행할지 멈출지는 상위가 정한다 — AG3S 가 스스로 정하지 않는 것과 같은 이유다.
    """
    from benchmark.ag3s.types import CameraID, CameraObservation
    from benchmark.ag3s.experiments.sources.mujoco_source import CAMERA_MOUNTS

    attention = attention or {}
    out = []
    for cam in scene.get("cameras", ()) or ():
        depth_mm = scene.get(f"depth/{cam}")
        if depth_mm is None:
            continue
        mount_link, camera_id = CAMERA_MOUNTS.get(cam, (None, cam))
        out.append(CameraObservation(
            camera_id=CameraID.parse(camera_id),
            depth=np.asarray(depth_mm, np.float64) / DEPTH_SCALE_MM,
            camera_intrinsics=np.asarray(scene[f"K/{cam}"], np.float64),
            # 외부 파라미터가 왔으면 그대로 쓴다. 클라이언트가 촬영 시점에 이미 풀어서 보냈고,
            # 서버가 FK 로 다시 푸는 것보다 정확하다 — 서버는 그 순간의 자세를 모른다.
            T_base_cam=np.asarray(scene[f"T_base_cam/{cam}"], np.float64),
            robot_state=np.asarray(scene[f"robot_state/{cam}"], np.float64),
            timestamp=float(scene.get(f"stamp/{cam}", 0.0)),
            attention_map=attention.get(cam),
        ))
    return out


class SafetyVerdict:
    """서버가 내리는 판정. 로컬이 실행 여부를 결정할 때 읽는 전부.

    **판정이지 명령이 아니다.** 서버는 "이 청크를 안전하다고 인증할 수 있는가" 까지만 말하고,
    멈출지 실행할지는 로컬이 정한다. AG3S 가 기하 불확실성을 보고만 하고 결정하지 않는 것과
    같은 계약이다.
    """

    def __init__(self, *, ag3s_status: str, geometry_certified: bool, trajopt_status: str,
                 max_violation_m: float, safe: bool, notes: Sequence[str] = ()):
        self.ag3s_status = ag3s_status
        self.geometry_certified = bool(geometry_certified)
        self.trajopt_status = trajopt_status
        self.max_violation_m = float(max_violation_m)
        self.safe = bool(safe)
        self.notes = list(notes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ag3s_status": self.ag3s_status,
            "geometry_certified": self.geometry_certified,
            "trajopt_status": self.trajopt_status,
            "max_violation_m": self.max_violation_m,
            "safe": self.safe,
            "notes": self.notes,
        }


def pack_response(actions: np.ndarray, verdict: SafetyVerdict, *, seq: int,
                  timing_ms: dict[str, float], field: Optional[Any] = None,
                  extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """응답. `actions` 는 `[H, ACTION_WIDTH]` 이고, 안전하지 않아도 실린다.

    안전하지 않을 때 청크를 **빼지 않는** 이유는, 로컬이 "왜 멈췄는지" 를 보려면 무엇이
    제안됐는지 알아야 하기 때문이다. 실행 여부는 `safe` 하나로 결정된다.

    `field` 는 `FieldProvenance` 다. **`None` 이면 `unavailable` 로 싣는다** — 키를 빼면
    읽는 쪽이 "필드가 없었다" 와 "서버가 옛 버전이라 안 보냈다" 를 구별할 수 없고, 전자는
    hold 해야 하고 후자는 배선 결함이라 대응이 다르다.
    """
    from benchmark.ag3s.fields.provenance import FieldProvenance

    prov = field if field is not None else FieldProvenance.unavailable(
        "the server produced no collision field for this chunk")
    out = {
        "actions": np.asarray(actions, np.float32),
        "seq": int(seq),
        "timing_ms": {k: float(v) for k, v in timing_ms.items()},
        "field": prov.to_dict(),
        **verdict.to_dict(),
    }
    if extra:
        out.update(extra)
    return out


def unpack_field(response: dict[str, Any]):
    """응답의 `field` 블록 -> `FieldProvenance`. 키가 없으면 그것도 상태로 답한다.

    키 없음은 **서버가 이 블록을 모르는 버전**이라는 뜻이고, 필드가 없는 것과 다르다.
    조용히 같게 취급하면 배선 결함이 안전 판정처럼 보인다.
    """
    from benchmark.ag3s.fields.provenance import FieldProvenance

    blob = response.get("field")
    if blob is None:
        return FieldProvenance.unavailable(
            "the response carried no `field` block — the server predates field provenance; "
            "staleness cannot be judged for this chunk")
    return FieldProvenance.from_dict(blob)
