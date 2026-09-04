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
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "PREFIX", "DEFAULT_CAMERAS", "GRIPPER_COLUMNS",
    "pack_request", "strip_request", "unpack_camera_observations",
    "pack_response", "SafetyVerdict",
]

PREFIX = "ag3s/"

#: 머리 하나 + 손목 둘. `CameraID` 에 HEAD 가 하나뿐이라 `zed_right` 는 `zed_left` 와 겹친다.
DEFAULT_CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")

#: 14-D action 에서 그리퍼가 앉은 열. TO 는 이 두 열을 **절대 건드리지 않는다** — 결정 변수가
#: 아니고, 손을 여닫는 것은 충돌 회피가 판단할 일이 아니다.
GRIPPER_COLUMNS = (6, 13)

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
    from benchmark.ag3s.experiments.mujoco_source import CAMERA_MOUNTS

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
                  timing_ms: dict[str, float], extra: Optional[dict[str, Any]] = None
                  ) -> dict[str, Any]:
    """응답. `actions` 는 언제나 [H, 14] 이고, 안전하지 않아도 실린다.

    안전하지 않을 때 청크를 **빼지 않는** 이유는, 로컬이 "왜 멈췄는지" 를 보려면 무엇이
    제안됐는지 알아야 하기 때문이다. 실행 여부는 `safe` 하나로 결정된다.
    """
    out = {
        "actions": np.asarray(actions, np.float32),
        "seq": int(seq),
        "timing_ms": {k: float(v) for k, v in timing_ms.items()},
        **verdict.to_dict(),
    }
    if extra:
        out.update(extra)
    return out
