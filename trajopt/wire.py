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
| `actions_reference` | 정책의 **원본** 청크. **2026-09-26 부터 closed loop 에서도 실린다.** 아래 |
| `shadow` | **모드**. `actions_reference` 가 아니라 **이 키**가 shadow 서버의 신호다. 아래 |
| `seq` | 요청의 일련번호를 그대로 돌려준다. 오래된 응답을 버리는 근거 |
| `timing_ms` | 단계별 시간 (서버 시계) |
| `ag3s_status` · `geometry_certified` · `trajopt_status` · `max_violation_m` · `safe` · `notes` | 안전 판정 |
| `field` | **거리장의 출처** — `sequence` · `backend` · `observed_at` · `state` · 계층. 아래 |
| `ag3s` | **선택 키.** `ag3s_status` 가 `ok` 가 아닐 때만 실린다 — **왜** 인증이 안 됐나. 아래 |
| `max_violation_pair` | **선택 키.** `max_violation_m` 을 만든 **행의 신원**. 아래 |

### `max_violation_pair` — `violated` 가 **어느 제약**인가

`T5f` 가 여기서 막혔다: 판정에는 `max_violation_m` 한 숫자만 있고, 그 숫자를 만든 행이 어느
link 대 어느 obstacle 인지는 **최적화기 안에만** 있었다. 숫자만 보고는 고칠 수 없다 — 팔꿈치가
탁자를 스친 것과 손끝이 사과를 파고든 것은 전혀 다른 일이고, 대응도 다르다.

`CollisionLinearizer.worst_row` 가 그 신원을 만든다: `block`(candidate · plane · esdf) ·
`step` · `link` · `slot` · `candidate_id` · `obstacle` · `point_m`. 값은 `clearance_m` 이고
`max_violation_m` 과 부호만 반대다 (`violation = max(0, -clearance)`).

**활성 제약이 하나도 없던 프레임에는 키가 실리지 않는다.** 없는 신원을 `null` 로 싣는 것보다
키를 빼는 것이 낫다 — 그래야 *"신원을 낼 수 있는 서버"* 와 옛 서버가 구별된다.
`actions_reference` · `ag3s` 와 같은 규약이다.

### `shadow` — **모드는 이제 명시된다** (2026-09-26, T9)

`actions_reference` 의 **있음/없음**이 shadow 서버의 신호였다. 그 신호는 값이 하나뿐인 자리에
두 가지 뜻(무엇을 실행할지 고르는 데 필요한 청크 · 서버가 어느 모드인지)을 실었고, 둘째 뜻
때문에 **closed loop 에서는 원본 청크를 실을 수 없었다.**

그 결핍에 다섯 번 막혔다. 마지막이 결정적이다 — 충돌 제약이 하나도 활성이 아닌 판에서도 TO 가
청크를 고친다 (실행되는 8 step 안에서 중앙값 2.6°, 최대 8.8°, 손이 다가갈수록 커진다). 그러면
남은 변형은 전부 **목적함수**가 만든 것인데, closed loop 의 원본 청크가 기록에 없으면 그 크기를
잴 수 없다. shadow 는 성공하고(245.2 mm 들어 올린다) closed loop 은 헛잡는 차이가 거기 있다.

그래서 두 뜻을 갈랐다.

| 키 | 뜻 | 언제 |
|---|---|---|
| `actions_reference` | 정책 원본 청크 — **데이터** | 언제나 (closed loop 포함) |
| `shadow` | 서버가 reference 를 **실행하라고 내보내는 모드인가** — **모드** | 언제나 (`True`/`False`) |

**`shadow` 는 `False` 여도 실린다.** 이 키만이 짝 검사의 근거이므로, 없음을 "closed loop" 으로
읽으면 **옛 서버**(키를 모르는 서버)와 구별할 수 없다. 옛 서버에는 `actions_reference` 의 있음/
없음으로 물러나 추론한다 (`client._check_shadow_pairing`).

### `actions_reference` — 서버가 계산에 쓴 **입력** 청크

**shadow 실행(T5)은 전부 계산하되 수정된 청크를 로봇에 보내지 않는다.** 수정이 여유거리를 나쁘게
만드는지를 로봇을 움직이기 전에 보려는 것이다. 그러려면 로컬이 **정책의 원본 청크**를 알아야 하는데,
지금까지 응답에는 `actions`(= refined) 하나뿐이라 로컬이 그것을 볼 길이 없었다.

**`actions` 는 shadow 에서도 여전히 refined 다.** 서버는 자기가 계산한 것을 그대로 말하고
(거짓말하지 않는다), 무엇을 실행할지는 로컬이 고른다 — `SafetyVerdict` 가 판정이지 명령이
아닌 것과 같은 계약이다.

**키는 이제 언제나 실린다** (T9). *"TO 가 청크를 얼마나 바꿨나"* 는 closed loop 에서 가장
알아야 하는 값이고, 그것을 재려면 `actions`(refined) 옆에 원본이 있어야 한다.

**대가는 응답 크기다.** 청크 하나가 `[H, 16] float32` 이므로 planning 기록 한 줄이 실측
19,837 B → shadow 수준(36,704 B)으로 커진다. 그 값을 아는 채로 고른 것이다 — 15 Hz 에서
한 프레임에 17 KB 가 더 흐르는 것보다, 같은 결핍에 여섯 번째로 막히는 것이 비싸다.

**무엇을 실행할지는 여전히 `shadow` 키가 정한다.** 로컬이 그 짝을 검사하고
(`client.SafeRemoteClient`), 한쪽만 켜져 있으면 즉시 실패한다 — 조용히 refined 를 실행하면
shadow 가 아닌데 shadow 라고 기록된다. **그 거절은 T9 에서 한 줄도 느슨해지지 않았다**: 근거가
"reference 가 왔나" 에서 "서버가 shadow 라고 말했나" 로 옮겨간 것뿐이다.

### `ag3s` — `degraded` 의 **사유**

`ag3s_status` 는 *"인증했나"* 만 말하고 *"왜 못 했나"* 는 말하지 않았다. 2026-09-25 첫 live
smoke 에서 그것이 바로 막혔다: 2 청크 중 1 개가 `max_violation_m = 0.0` (궤적은 모든 제약을
통과)인데 `ag3s_status = degraded` · `geometry_certified = False` 로 HOLD 됐고, **왜 degraded
인지가 와이어에도 서버 로그에도 없었다.**

사유는 서버 안에 이미 있었다 — `CollisionConstraintSet.notes` 다. 나가는 길이 없었을 뿐이다
(응답의 `notes` 는 **최적화기**쪽 `TrajOptResult.notes` 이고 지각 쪽이 아니다).

| 안쪽 키 | 내용 |
|---|---|
| `status` | `CollisionConstraintSet.status` (= `ag3s_status` 와 같은 값. 블록만 보고도 짝을 확인할 수 있게 둔다) |
| `validity` | `ConstraintValidity` — `status` 가 그것에서 파생된다 |
| `grounding_status` | target 을 왜 못 잡았나. **T5b 까지 로컬은 이 값을 `unavailable-on-client` 로 적고 있었다** |
| `reasons` | `[{"code", "detail"}]` — **기계가 읽는 사유.** 코드는 `ag3s/runtime/degradation.py:CODES` 에 등록된 것뿐이고, 코드 하나가 소스의 한 분기에 대응한다 |
| `notes` | 지각 쪽 노트 전부 (코드 없는 산문 노트까지). 사람이 읽는 쪽 |

**`status` 가 `ok` 일 때는 키가 아예 없다.** 회귀 기준선과 T0 기록이 정상 프레임의 응답에
달려 있으므로 그쪽은 한 바이트도 건드리지 않는다. `actions_reference` 와 같은 규약이고, 이유도
같다 — 키의 있음/없음 자체가 신호다.

**`reasons` 가 비는 경우는 없다.** `degradation.ensure_reason` 이 마지막 관문에서 `degraded`
인데 코드가 하나도 없으면 `degraded_without_reason` 을 달아 보낸다. 그 코드가 보이면 씬의 성질이
아니라 **AG3S 의 배선 결함**이다.

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
    "ARM_JOINT_DIM", "ACTION_WIDTH", "ACTIONS_REFERENCE", "SHADOW",
    "pack_request", "strip_request", "unpack_camera_observations",
    "pack_response", "unpack_field", "unpack_actions_reference", "unpack_ag3s",
    "unpack_violation_pair", "unpack_shadow",
    "SafetyVerdict", "AG3S_BLOCK", "VIOLATION_PAIR",
]

PREFIX = "ag3s/"

#: 응답 키 — 정책의 원본 청크. **T9 부터 closed loop 에서도 실린다** (위 머리말).
#: 이름을 상수로 두는 이유는 서버·클라이언트·테스트 세 곳이 같은 문자열을 써야 하고, 오타가
#: 나면 로컬이 "reference 가 안 왔다" 로 읽어 즉시 실패하기 때문이다 — 조용히는 안 지나가지만
#: 원인을 찾는 데 시간이 든다.
ACTIONS_REFERENCE = "actions_reference"

#: 응답 키 — **모드.** `True` 면 서버는 로컬이 reference 를 실행하기를 기대한다 (shadow).
#: `False` 여도 실린다: 이 키가 짝 검사의 유일한 근거이므로, 없음은 "closed loop" 이 아니라
#: **"이 키를 모르는 옛 서버"** 를 뜻해야 한다. 그 구분이 없으면 옛 서버에 shadow 로컬을
#: 붙였을 때 조용히 refined 가 실행된다.
SHADOW = "shadow"

#: 응답의 **선택 키** — `ag3s_status` 가 `ok` 가 아닐 때의 사유 블록 (위 머리말).
#: `ag3s/` 접두(요청 쪽)와 글자가 겹치지만 충돌하지 않는다: `strip_request` 는 **요청**만
#: 가르고 `"ag3s/"`(슬래시 포함)로 시작하는 키만 본다. 응답은 애초에 stripping 을 안 지난다.
AG3S_BLOCK = "ag3s"

#: 응답의 **선택 키** — `max_violation_m` 을 만든 행의 신원 (`SafetyVerdict.max_violation_pair`).
#: `T5f` 가 *"`violated` 가 어느 제약인가"* 에서 막힌 것을 여는 키다. 활성 제약이 하나도 없던
#: 프레임에는 실리지 않는다 — 없는 신원을 `null` 로 싣는 것과 키를 빼는 것 중, 뒤쪽이
#: *"이 서버는 신원을 낼 수 있다"* 를 잃지 않는다 (옛 서버와 구별된다).
VIOLATION_PAIR = "max_violation_pair"

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
                 max_violation_m: float, safe: bool, notes: Sequence[str] = (),
                 max_violation_pair: Optional[dict[str, Any]] = None):
        self.ag3s_status = ag3s_status
        self.geometry_certified = bool(geometry_certified)
        self.trajopt_status = trajopt_status
        self.max_violation_m = float(max_violation_m)
        self.safe = bool(safe)
        self.notes = list(notes)
        #: `max_violation_m` 을 만든 **행의 신원** (`CollisionLinearizer.worst_row`), 또는
        #: `None` — 활성 제약이 없었거나 서버가 이 키를 모르는 버전이다. 위 머리말 참조.
        self.max_violation_pair = (None if max_violation_pair is None
                                   else dict(max_violation_pair))

    def to_dict(self) -> dict[str, Any]:
        out = {
            "ag3s_status": self.ag3s_status,
            "geometry_certified": self.geometry_certified,
            "trajopt_status": self.trajopt_status,
            "max_violation_m": self.max_violation_m,
            "safe": self.safe,
            "notes": self.notes,
        }
        # **없으면 키를 싣지 않는다** — `ag3s` 블록·`actions_reference` 와 같은 규약이다.
        # 활성 제약이 하나도 없던 프레임의 응답이 T0 때와 한 바이트도 달라지지 않게 두려는 것이고,
        # 있음/없음이 *"서버가 이 신원을 낼 수 있는 버전인가"* 를 그대로 말해 준다.
        if self.max_violation_pair is not None:
            out[VIOLATION_PAIR] = dict(self.max_violation_pair)
        return out


def pack_response(actions: np.ndarray, verdict: SafetyVerdict, *, seq: int,
                  timing_ms: dict[str, float], field: Optional[Any] = None,
                  actions_reference: Optional[np.ndarray] = None,
                  shadow: bool = False,
                  ag3s: Optional[dict[str, Any]] = None,
                  extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """응답. `actions` 는 `[H, ACTION_WIDTH]` 이고, 안전하지 않아도 실린다.

    안전하지 않을 때 청크를 **빼지 않는** 이유는, 로컬이 "왜 멈췄는지" 를 보려면 무엇이
    제안됐는지 알아야 하기 때문이다. 실행 여부는 `safe` 하나로 결정된다.

    `field` 는 `FieldProvenance` 다. **`None` 이면 `unavailable` 로 싣는다** — 키를 빼면
    읽는 쪽이 "필드가 없었다" 와 "서버가 옛 버전이라 안 보냈다" 를 구별할 수 없고, 전자는
    hold 해야 하고 후자는 배선 결함이라 대응이 다르다.

    `actions_reference` 는 **`None` 이면 키를 싣지 않는다** — 그러나 그것이 더 이상 "shadow 가
    아니다" 를 뜻하지 않는다 (T9). 모드는 `shadow` 키가 말하고, 이 키는 **데이터의 있음/없음**
    만 말한다: 원본 청크를 낼 수 없었던 프레임(정책이 청크를 못 낸 hold 등)이 그 경우다.

    `shadow` 는 **`False` 여도 싣는다.** 짝 검사의 근거가 이 키 하나이므로, 없음이 "closed
    loop" 이 아니라 "이 키를 모르는 옛 서버" 를 뜻해야 한다.

    `ag3s` 도 **`None` 이면 키를 싣지 않는다** — 정상 프레임(`ag3s_status == "ok"`)의 응답을
    T0 때와 같게 두기 위해서다. 호출부(`SafePolicy._ag3s_block`)가 `ok` 일 때 `None` 을 준다.

    Raises:
        ValueError: `actions_reference` 의 모양이 `actions` 와 다를 때. 로컬이 둘 중 하나를
            골라 실행하므로 모양이 어긋난 채 나가면 **한 칸씩 밀린 청크가 실행된다** — 형태는
            맞고 뜻은 틀린, 가장 위험한 실패다. 여기서 크게 죽는 편이 낫다.
    """
    from benchmark.ag3s.fields.provenance import FieldProvenance

    prov = field if field is not None else FieldProvenance.unavailable(
        "the server produced no collision field for this chunk")
    packed = np.asarray(actions, np.float32)
    out = {
        "actions": packed,
        "seq": int(seq),
        "timing_ms": {k: float(v) for k, v in timing_ms.items()},
        "field": prov.to_dict(),
        # **모드는 명시한다** (T9). `actions_reference` 의 있음/없음으로 추론하던 자리다.
        SHADOW: bool(shadow),
        **verdict.to_dict(),
    }
    if actions_reference is not None:
        reference = np.asarray(actions_reference, np.float32)
        if reference.shape != packed.shape:
            raise ValueError(
                f"actions_reference has shape {reference.shape} but actions has "
                f"{packed.shape}. The client picks one of the two to execute, so a mismatch "
                "would put a shifted chunk on the robot")
        out[ACTIONS_REFERENCE] = reference
    if ag3s:
        out[AG3S_BLOCK] = dict(ag3s)
    if extra:
        out.update(extra)
    return out


def unpack_actions_reference(response: dict[str, Any]) -> Optional[np.ndarray]:
    """응답의 정책 원본 청크, 또는 **`None`** — 그 프레임에 원본이 없었다는 뜻이다.

    **T9 부터 "없음" 은 모드가 아니라 데이터의 부재다.** 모드는 `unpack_shadow` 가 답한다.
    `unpack_field` 와 달리 없음을 상태 객체로 감싸지 않는 것은 그대로다 — 이 값을 어떻게 다룰지
    (실행할지, 기록만 할지)는 `SafeRemoteClient` 한 곳에서 정한다.
    """
    blob = response.get(ACTIONS_REFERENCE)
    return None if blob is None else np.asarray(blob)


def unpack_shadow(response: dict[str, Any]) -> Optional[bool]:
    """서버가 shadow 모드인가. **`None` 은 "말하지 않았다"** = 이 키를 모르는 옛 서버.

    세 값을 가르는 것이 요점이다. `True`/`False` 는 서버의 선언이고 `None` 은 선언이 없는
    것이다. 없음을 `False` 로 접으면 옛 서버에 shadow 로컬을 붙였을 때 조용히 refined 가
    실행된다 — 그래서 호출부는 `None` 일 때 `actions_reference` 의 있음/없음으로 물러나
    추론한다 (`client._check_shadow_pairing`).
    """
    if SHADOW not in response:
        return None
    return bool(response[SHADOW])


def unpack_ag3s(response: dict[str, Any]) -> dict[str, Any]:
    """응답의 `ag3s` 블록, 없으면 **빈 딕셔너리**.

    키 없음은 두 가지를 뜻할 수 있다 — 기하가 인증됐다(`ok`) 거나, 서버가 이 블록을 모르는
    버전이다. `unpack_field` 처럼 그 둘을 구분해 주지 않는 이유는 **`ag3s_status` 가 이미 같은
    응답에 있기** 때문이다: `ag3s_status != "ok"` 인데 이 블록이 비어 있으면 그것이 곧 옛 서버다.
    읽는 쪽이 그 조합을 보고 판단할 수 있으므로 여기서 상태 객체를 만들지 않는다.
    """
    blob = response.get(AG3S_BLOCK)
    return dict(blob) if isinstance(blob, dict) else {}


def unpack_violation_pair(response: dict[str, Any]) -> Optional[dict[str, Any]]:
    """응답의 `max_violation_pair`, 없으면 `None`.

    `unpack_field` 처럼 "없음" 을 상태 객체로 감싸지 않는 이유는 `unpack_ag3s` 와 같다 —
    같은 응답의 `max_violation_m` 이 이미 숫자를 말하고 있으므로, 이 키의 없음은 판정을
    바꾸지 않는다. 신원을 못 얻었다는 사실 자체가 기록에 `null` 로 남으면 그것으로 충분하다.
    """
    blob = response.get(VIOLATION_PAIR)
    return dict(blob) if isinstance(blob, dict) else None


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
