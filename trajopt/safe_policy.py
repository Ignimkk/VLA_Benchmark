"""π0.5 → (SEAM) → AG3S → TO 를 서버 한 프로세스로 묶는 `BasePolicy` 데코레이터.

`PolicyRecorder` 가 이미 만들어 둔 자리를 쓴다 — 정책을 감싸는 정책. 그래서 `serve_policy.py`
나 openpi 의 어떤 파일도 고치지 않는다.

**왜 서버인가.** 로컬 배선(`trajopt/bringup.py`)을 실측해 보니 depth 렌더 167 ms + AG3S
393~628 ms 로 청크 예산 533 ms 를 1.4~2.7배 넘었다. 그 대부분은 CPU 소프트웨어 래스터라이저와
지각이고, 둘 다 GPU 서버에서 훨씬 싸다. 더 중요한 것은 attention 이다 — 정책 내부에서 나오는
값이라 서버 밖으로 꺼내려면 매 프레임 전송해야 하는데, 같은 프로세스 안이면 그냥 변수다.

**실행 여부는 여기서 정하지 않는다.** 이 클래스는 `safe` 를 계산해 실어 보내고 끝이다. 멈출지
실행할지는 로컬이 정한다. AG3S 가 기하 불확실성을 보고만 하고 결정하지 않는 것과 같은 계약이고,
이유도 같다 — 무엇이 허용 가능한 위험인지는 지각도 최적화도 알 수 없다.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Callable, Optional, Sequence

import numpy as np

from benchmark.ag3s.types import subtask_no_target
from benchmark.trajopt import wire
from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.grasp_latch import (
    CentroidIdentity,
    GraspLatch,
    GraspPhase,
    LatchConfig,
    grasp_signal_from_feedback,
)
from benchmark.trajopt.linearize import CollisionLinearizer, scene_from_constraint_set
from benchmark.trajopt.policy_seed import apply_request_seed
from benchmark.trajopt.refiner import TrajOptChunkRefiner
from benchmark.trajopt.types import ChunkLayout

__all__ = ["SafePolicy", "classify_violations", "VERDICT_POLICIES",
           "GRASP_CONTINUITY_OFF_PHASES", "grasp_continuity_off_reason"]


#: 어느 손이 어느 링크로 쥐는가. 로봇마다 다르므로 **주입**이지만, 기본값이 두 곳에서 필요하다 —
#: `SafePolicy` 가 `attach` 를 부를 때와, **그보다 먼저** AG3S 가 attached 슬롯을 예약할 때다.
DEFAULT_GRASP_LINKS: dict[str, tuple] = {
    "left": ("ee_finger_l1", ("ee_finger_l1", "ee_finger_l2")),
    "right": ("ee_finger_r1", ("ee_finger_r1", "ee_finger_r2")),
}


#: 쥔 물체가 **붙는** link — 손바닥 (T32 H2). 손가락 link(`DEFAULT_GRASP_LINKS` 의 첫 값)는 개도에 따라
#: 움직이므로 물체가 손가락 하나와 함께 움직였다 (T22 · T29). reach 는 여전히 손가락에서 잰다.
DEFAULT_HOLD_LINKS: dict[str, str] = {"left": "ee_left", "right": "ee_right"}


def hold_links_for(grasp_links: Optional[dict] = None,
                   hold_links: Optional[dict] = None) -> dict[str, str]:
    """손마다 쥔 물체가 붙는 link. `hold_links` 가 주어지면 그것, 기본 `grasp_links` 면 손바닥
    (`DEFAULT_HOLD_LINKS`), 직접 준 `grasp_links` 면 그 grasp link 자신 (예전 계약)."""
    links = grasp_links or DEFAULT_GRASP_LINKS
    if hold_links is not None:
        hold = dict(hold_links)
    elif grasp_links is None:
        hold = dict(DEFAULT_HOLD_LINKS)
    else:
        hold = {}
    return {str(h): str(hold.get(h) or v[0]) for h, v in links.items()}


def grasp_parent_links(grasp_links: Optional[dict] = None,
                       hold_links: Optional[dict] = None) -> tuple[str, ...]:
    """AG3S 를 지을 때 `attached_parent_links=` 로 넘겨야 하는 링크들.

    **슬롯은 생성 시점에 예약된다.** `ConstraintBuilder` 의 심볼 그래프와 희소성이 거기서
    고정되므로 `attach()` 가 나중에 늘릴 수 없다. 안 넘기면 파지가 닫히는 **다음 프레임**에
    `_scene_fn` 이 `ValueError` 를 던지고, `refiner` 가 그것을 삼켜 **지각과 최적화가 통째로
    멈춘 채 응답만 계속 나간다** — 실측으로 잡은 결함이다 (2026-09-18, `run_0004` 프레임 10
    이후 14 프레임 동안 AG3S 가 한 번도 안 돌았다).
    """
    # T32 H2: 슬롯은 쥔 물체가 **붙는** link (기본 = 손바닥) 에 예약한다 — 손가락이 아니다.
    return tuple(hold_links_for(grasp_links, hold_links).values())


class SafePolicy:
    """정책 하나를 감싸 안전 판정이 붙은 청크를 돌려준다.

    Args:
        policy: 감쌀 것. π0.5 그 자체이거나 SEAM 이 이미 감싼 것. **여기서는 구분하지 않는다** —
            둘 다 `infer(obs) -> {"actions": [H, 14], ...}` 를 지킨다.
        ag3s: 구성된 `AG3S`.
        constraint_robot_model: AG3S 가 제약을 쓴 것과 **같은** 모델. 두 모델이 다르면 최적화기는
            한 곳의 기하를 피하면서 제약은 다른 곳을 설명하게 되고, 아무도 그 불일치를 보고하지
            않는다. 그래서 인자로 받지 않고 `ag3s.constraint_robot_model` 에서 꺼낸다.
        attention_fn: `(policy_obs, policy_result) -> {카메라: 맵}`. 정책이 attention 을 어떻게
            내놓는지는 체크포인트마다 다르므로 주입받는다. None 이면 attention 없이 돌고, 그러면
            AG3S 가 target 을 못 잡아 거리장이 target 을 파내지 않는다 — 제약이 더 보수적이
            되는, 안전한 방향의 실패다.
        shadow: **shadow 실행(T5)이면 True.** 서버가 하는 일은 하나도 줄지 않는다 — AG3S 지각·
            ESDF·SQP·판정이 그대로 돌고 `actions` 도 그대로 refined 다. 달라지는 것은 둘뿐이다:
            응답에 정책 원본 청크를 `actions_reference` 로 함께 싣고(그래야 로컬이 그것을
            실행할 수 있다), 연속성 기준을 **로봇이 실제로 실행한 쪽**으로 잡는다. 기본값
            False 에서는 응답도 내부 상태도 예전과 한 바이트도 다르지 않다.

        기하 인증을 요구할지는 **`to_config.safety.require_certified_geometry` 하나가 정한다.**
        여기에 같은 뜻의 두 번째 스위치를 두었다가 반쪽만 작동하는 것을 실측으로 확인했다 —
        SafePolicy 쪽만 끄면 sqp 가 여전히 상태를 VIOLATED 로 내려서, 위반이 0 mm 인데도
        모든 프레임이 hold 로 갔다. 스위치가 둘이면 언젠가 갈라지고, 갈라진 쪽이 안전 판정이면
        그 결과는 조용하다.
    """

    def __init__(self, policy, *, ag3s, to_config: Optional[TrajOptConfig] = None,
                 attention_fn: Optional[Callable[[dict, dict], dict]] = None,
                 default_phase: str = "approach", recorder=None,
                 latch: Optional[LatchConfig] = None,
                 grasp_links: Optional[dict] = None,
                 hold_links: Optional[dict] = None,
                 placed_fn: Optional[Callable[[dict, Any], bool]] = None,
                 static_geometry: Optional[Sequence[Any]] = None,
                 shadow: bool = False,
                 verdict_policy: str = "reasons",
                 allowed_contact_pad_m: float = 0.01,
                 allow_unresolved_contact_tier: bool = False):
        self._policy = policy
        #: **`safe` 를 무엇으로 정하나** (T23). `"reasons"`(기본) 는 `verdict_reasons` 에 HOLD 처리인
        #: 사유가 없을 때 `safe` 다 — 권한 link ↔ manipulated 접촉(`allowed_contact`) · 가려진
        #: target(`occluded_target`) · 예산 종료(`budget_only`) 는 실행이다. `"legacy"` 는 T23 전
        #: 그대로 `TrajOptStatus.safe` 다 (사유는 똑같이 계산해 싣는다 — 기록은 같다).
        if verdict_policy not in VERDICT_POLICIES:
            raise ValueError(f"verdict_policy must be one of {VERDICT_POLICIES}, "
                             f"got {verdict_policy!r}")
        self.verdict_policy = verdict_policy
        #: `allowed_contact` 판정의 **마지막 fallback** (manipulated 점군·구·라벨이 다 없을 때만)
        #: 에서 centroid 공에 더하는 여유 (m). 주 판정은 `_esdf_clearance` 와 같은 거리 일치 검사다.
        self.allowed_contact_pad_m = float(allowed_contact_pad_m)
        #: True 면 미세 창 밖(`coarse`·`window_edge`·`outside`) 계층이 답한 행도 `allowed_contact`
        #: 의 근거로 받는다. 기본 False — T21 의 규칙 (`curobo_field.RESOLVED_TIERS`: 창 밖 답은
        #: 접촉 허용의 근거가 못 된다). 그런 행은 `unverified` (HOLD) 로 간다.
        self.allow_unresolved_contact_tier = bool(allow_unresolved_contact_tier)
        self.ag3s = ag3s
        self.to_config = to_config or TrajOptConfig.from_dict({
            "collision": {"backend": "esdf", "esdf_margin": 0.05, "use_support_planes": False},
        })
        self.attention_fn = attention_fn
        self.default_phase = default_phase
        #: `ConstraintRecordWriter` 또는 None. `--safe-remote` 로 도는 청크는 서버 밖으로
        #: attention·grounding·거리장을 내보내지 않으므로, 진단하려면 여기서 남겨야 한다 —
        #: `benchmark/trajopt/bringup.py`의 `LivePipeline._record`와 같은 계약.
        self.recorder = recorder
        #: 조작 대상을 에피소드 상태로 붙드는 것 (F17 — grounding 은 무상태라 파지 순간
        #: attention 이 목적지로 넘어가면 권한이 엉뚱한 물체에 붙는다). `attach`/`detach` 를
        #: **언제** 불러야 하는지만 답하고, 부르는 것은 이 클래스다 — AG3S 는 그 결정을 하지
        #: 않는다는 계약(`attached.py` 머리말)이 그대로 남는다.
        #:
        #: **T22 — attach 는 파지 확인 뒤에.** 기본(`legacy_gripper_attach=False`)은 evidence 모드다:
        #: 실행 피드백의 적용 명령 · 측정 개도 · 조작 대상(T20) · reach 가 모두 맞을 때만 attach 하고,
        #: 그 판단은 **제약을 짓기 전에** 한다 (`_advance_grasp`). 계획값으로 attach 하는 경로는
        #: `legacy_gripper_attach=True` 로만 켜진다 (옛 기록 재생용).
        latch_cfg = latch or LatchConfig()
        #: T26: 목적지는 **AG3S 가 등록한 destination** 이다 (`ag3s.destination`). grounding 이름으로
        #: 배우던 길(파지 뒤의 target = 목적지)은 T26 뒤 target 이 사과로 남으므로 닫는다.
        self._latch = GraspLatch(latch_cfg, evidence=not latch_cfg.legacy_gripper_attach,
                                 external_destination=True)
        #: attach/detach 가 일어날 때마다 +1 (T22). 제약이 어느 버전으로 지어졌는지와 비교해
        #: 어긋남을 검출한다 (`summary_json.grasp.state_version` / `constraint_version`).
        self._grasp_state_version = 0
        self._grasp_constraint_version: Optional[int] = None
        #: 이번 청크의 파지 판단 기록 (`summary_json.grasp`).
        self._last_grasp_record: dict[str, Any] = {}
        #: grounding 은 이름표를 주지 않으므로 무게중심으로 같은 물체인지 본다.
        self._identity = CentroidIdentity()
        #: 어느 손이 어느 링크로 쥐는가. 주입이다 — 로봇마다 다르고 추측할 수 없다.
        self.grasp_links = dict(grasp_links or DEFAULT_GRASP_LINKS)
        #: T32 H2: 손마다 쥔 물체가 붙는 link (기본 = 손바닥). reach 는 `grasp_links` 에서 잰다.
        self.hold_links = hold_links_for(grasp_links, hold_links)
        #: `(scene, constraint_set) -> bool`. 조작 대상이 목적지 안에 들어가 손에서 떨어졌는가.
        #: 기하 판정이라 외부가 준다 — 이 클래스도 잠금도 씬을 보지 않는다.
        self.placed_fn = placed_fn
        #: 아는 정적 기하 (`esdf.StaticBox` / `StaticPlane`) — 벽 · 선반 · 테이블 · 바닥.
        #: 거리장이 `min(복셀, 해석적)` 으로 답하게 해 미관측·격자 밖의 낙관을 없앤다 (N2 · E4:
        #: 실측 최대 낙관 +219.5 → +0.0 mm). **주입이다** — AG3S 도 이 클래스도 무엇이 고정
        #: 기하인지 알 수 없고, `phase` · 목적지와 같은 계약이다. `None` 이면 예전과 같이 돈다.
        self.static_geometry = tuple(static_geometry or ())
        #: shadow 실행인가. **판정은 하되 수정을 로봇에 보내지 않는** 실행이고, 이 클래스가
        #: 그것 때문에 바꾸는 것은 응답에 `actions_reference` 를 더하는 것과 연속성 기준을
        #: 실행된 쪽으로 잡는 것뿐이다. 계산은 하나도 건너뛰지 않는다 — shadow 의 목적이
        #: **전체 파이프라인을 돌린 결과**를 로봇을 움직이기 전에 보는 것이기 때문이다.
        self.shadow = bool(shadow)
        #: 이번 프레임에 AG3S 가 쓴 목적지 점 (**기록용 거울**, T26). 목적지를 정하는 것은 AG3S 다 —
        #: 인지 등록(`DestinationRegistry`) 또는 `ag3s.reset(destination_points=...)` 주입.
        self._destination_points = None
        #: 이번 청크가 AG3S 에 넘긴 실행 구간 (T21b/T26 §6) — 기록용 요약.
        self._last_execution_path: dict[str, Any] = {}
        #: 잠긴 이름이 마지막으로 target 으로 나왔을 때의 그 target. `attach` 가 이것을 쓴다 —
        #: 잠금이 걸린 뒤의 target 은 이미 목적지일 수 있으므로 지금 프레임 것을 쓰면 안 된다.
        self._latched_target = None

        # **파지가 닫히기 전에 여기서 죽는다.** attached 슬롯은 AG3S 생성 시점에 예약되고
        # `attach()` 가 나중에 늘릴 수 없다. 안 예약된 채로 두면 파지 다음 프레임에
        # `_scene_fn` 이 던지고 `refiner` 가 그것을 삼켜 **지각과 최적화가 통째로 멈춘 채
        # 응답만 계속 나간다** — 실측으로 14 프레임 동안 조용했다 (2026-09-18).
        # 시작할 때 큰 소리로 죽는 편이 낫다.
        builder = getattr(ag3s, "builder", None)
        reserved = tuple(getattr(builder, "attached_parent_links", ()) or ())
        missing = [p for p in self.hold_links.values() if p not in reserved]
        if missing:
            raise ValueError(
                f"AG3S reserved attached slots for {list(reserved)} but this SafePolicy grasps "
                f"with {missing}. Build AG3S with "
                f"attached_parent_links={grasp_parent_links(self.grasp_links)!r} — the slots are "
                "fixed at construction and attach() cannot add them later, so without this the "
                "pipeline dies silently on the frame after the grasp closes."
            )

        model = ag3s.constraint_robot_model
        if model is None:
            raise ValueError(
                "AG3S was built without a constraint robot model, so there is nothing to write "
                "collision constraints against")
        self.constraint_robot_model = model
        self.layout = ChunkLayout.rby1(model.joint_names)
        #: 그리퍼가 앉은 두 열, **레이아웃에서 유도한다**. 14D 는 `(6, 13)`, 16D 는 `(7, 15)` 다.
        #: 박아 두면 차원을 바꿀 때 엉뚱한 열을 그리퍼로 읽는다 — 이 프로젝트가 이미 밟았다:
        #: `_run_latch` 가 16D 에서 열 6(= `left_arm_6` 손목)을 그리퍼 개도로 읽고 있었고
        #: (T6f 에서 고쳤다), 손목 각도가 우연히 문턱을 넘으면 잠금이 파지로 오인한다. 증상이
        #: 궤적 오차가 아니라 **권한이 엉뚱한 물체에 붙는 것**으로 나오므로 조용하다.
        self.gripper_columns = wire.gripper_columns(self.layout.nq_opt // 2)
        self.linearizer = CollisionLinearizer(model, self.layout, self.to_config.horizon.planned)
        self.refiner = TrajOptChunkRefiner(model, self.layout, self._scene_fn, self.to_config)
        #: T29 — 손가락 관절을 실제 개도로. 모델에 finger parameter 가 있으면 (RB-Y1
        #: `UrdfSphereChain`) 클라이언트의 gripper 규약(`pi05_infer.py` 의 `RBY1_GRIPPER_OPEN` ·
        #: `GRIPPER_L/R_JOINT`)을 **여기서 읽는다** — 못 읽으면 시작할 때 죽는다 (손가락이 조용히
        #: 닫힌 채 떠 있는 것이 T29 가 고치는 결함이다). 없는 모델(mock)은 예전 그대로.
        self._finger_map = self._build_finger_map(model, getattr(ag3s, "robot_model", None))
        #: 이번 청크의 손가락 상태 기록 (`summary_json.finger_joints`).
        self._last_finger: dict[str, Any] = {}

        self._pending: dict[str, Any] = {}
        #: 연속성 기준의 **기본값** — 피드백이 없을 때 쓰는 "로봇이 실행했으리라 가정하는" 청크
        #: (closed loop 은 refined, shadow 는 reference). 피드백이 있으면 `_previous_plan` 과
        #: 피드백으로 다시 고른다 (`_continuity_input`).
        self._previous_chunk: Optional[np.ndarray] = None
        #: **계획 캐시** (T18) — 마지막으로 **보낸** 계획: `{"seq", "refined", "reference"}`.
        #: 실행 이력과 분리한다 (지침 §6.4). 무엇이 실행됐는지는 다음 요청의 피드백이 말한다.
        self._previous_plan: Optional[dict[str, Any]] = None
        #: 이번 청크가 받은 실행 피드백과 그것으로 한 일 — 기록(`summary_json`)과 응답 `notes` 용.
        self._last_feedback: dict[str, Any] = wire.no_exec_feedback("no request yet")
        self._last_continuity: dict[str, Any] = {}
        #: T41 b — 이 청크를 정지 출발 범위로 풀었나 (`limits.rest_start` 가 켜졌을 때만 채운다).
        self._last_rest_start: dict[str, Any] = {}
        #: T37 — 지난 청크에서 grasp latch 가 continuity 항을 껐나 (켜짐/꺼짐이 바뀔 때만 로그 한 줄).
        self._continuity_gate_on = False
        self._last_latch_signal: dict[str, Any] = {}
        self._last_constraint_set = None
        #: 이번 청크의 `SceneSnapshot` — 최적화기가 받은 것 그대로 (T23: 위반 행 분류용).
        self._last_snapshot = None
        #: 이번 청크의 판정 사유와 행 분류 (`summary_json.verdict`).
        self._last_reasons: Optional[list[dict[str, Any]]] = None
        self._last_classification: dict[str, Any] = {}
        self._last_debug: dict[str, Any] = {}
        self._last_attention_by_camera: dict[str, Any] = {}
        self._last_q_now: Optional[np.ndarray] = None
        self._chunk_index = -1
        #: 기록 실패 횟수. **첫 실패만 크게 외치고** 그 뒤는 한 줄씩 — 같은 문장을 75 번
        #: 읽게 하면 읽는 사람이 로그를 건너뛰기 시작한다.
        self._record_failures = 0
        #: 정책 RNG seed (T39). `_seed_applied` 는 **이 요청**에 적용한 값, `_episode_seed` 는 이
        #: 에피소드(마지막 reset 뒤)에 적용한 값. 둘 다 없으면 `None` — RNG 가 서버 순서대로 이어진다.
        self._seed_applied: Optional[int] = None
        self._episode_seed: Optional[int] = None

    # --- BasePolicy 인터페이스 -----------------------------------------------------------
    @property
    def metadata(self) -> dict[str, Any]:
        meta = dict(getattr(self._policy, "metadata", {}) or {})
        meta.update({
            "safe_policy": True,
            "constraint_spheres": int(self.linearizer.n_spheres),
            "collision_backend": self.to_config.collision.backend,
            "esdf_margin_m": float(self.to_config.collision.esdf_margin),
            "planned_horizon": int(self.to_config.horizon.planned),
        })
        # **shadow 일 때만 키를 더한다.** 로컬이 접속하자마자 짝이 맞는지 보는 근거이고
        # (`SafeRemoteClient` 가 생성자에서 검사한다), 없을 때 `False` 를 넣지 않는 것은
        # 기본 메타데이터를 T0 때와 같게 두기 위해서다.
        if self.shadow:
            meta["shadow"] = True
        # T41 a — 같은 규약: 켰을 때만 키가 생긴다 (client 기록에서 어느 설정으로 돌았는지 본다).
        obstacle_margin = float(getattr(self.to_config.collision, "obstacle_margin", 0.0) or 0.0)
        if obstacle_margin > 0.0:
            meta["obstacle_margin_m"] = obstacle_margin
            meta["obstacle_margin_support"] = bool(self.to_config.collision.obstacle_margin_support)
        return meta

    def reset(self) -> None:
        """에피소드 경계. SEAM·AG3S·TO 의 내부 상태와 warm-start 를 **모두** 버린다.

        하나라도 남기면 다음 에피소드의 첫 청크가 지난 에피소드의 씬을 warm-start 로 받는다.
        보통은 조금 나쁜 초기값에 그치지만, 물체가 옮겨졌다면 사라진 장애물을 피하려 애쓰는
        궤적이 나온다.
        """
        for obj in (self._policy, self.ag3s, self.refiner):
            reset = getattr(obj, "reset", None)
            if callable(reset):
                reset()
        self._previous_chunk = None
        self._previous_plan = None
        self._last_constraint_set = None
        self._last_snapshot = None
        self._last_reasons = None
        self._last_classification = {}
        # 잠금도 에피소드 상태다. 남기면 다음 과제가 지난 과제의 조작 대상을 물려받는다.
        self._latch.reset()
        self._identity.reset()
        self._destination_points = None
        self._last_execution_path = {}
        self._latched_target = None
        self._last_debug = {}
        self._last_attention_by_camera = {}
        self._last_q_now = None
        self._pending = {}
        self._last_continuity = {}
        self._last_rest_start = {}
        self._continuity_gate_on = False
        self._episode_seed = None
        self._last_latch_signal = {}
        self._grasp_state_version = 0
        self._grasp_constraint_version = None
        self._last_grasp_record = {}
        self._last_finger = {}
        for lin in self._finger_linearizers():
            lin.set_joint_parameter_path(None)

    # ------------------------------------------------------------------------------------
    def infer(self, obs: dict[str, Any], **kwargs) -> dict[str, Any]:
        started = time.monotonic()
        timing: dict[str, float] = {}
        policy_obs, scene = wire.strip_request(obs)
        seq = int(scene.get("seq", 0))

        if scene.get("reset"):
            self.reset()

        # T39 — 정책 RNG seed. **이 요청의 `policy.infer` 바로 앞**에서 다시 놓는다 (reset 뒤). 없으면
        # 아무것도 안 한다 — RNG 는 서버 순서대로 이어진다 (T39 전 동작).
        self._seed_applied = apply_request_seed(self._policy, scene, seq=seq)
        if self._seed_applied is not None:
            self._episode_seed = self._seed_applied
            print(f"[safe_policy] seq={seq} policy_seed={self._seed_applied} "
                  "(policy RNG re-keyed before this request's infer)")

        t = time.monotonic()
        result = self._policy.infer(policy_obs, **kwargs)
        chunk = np.asarray(result["actions"], np.float64)
        timing["infer"] = (time.monotonic() - t) * 1000.0

        attention = {}
        if self.attention_fn is not None:
            try:
                attention = self.attention_fn(policy_obs, result) or {}
            except Exception:  # noqa: BLE001 — attention 실패가 정책을 죽이면 안 된다
                traceback.print_exc()

        # **직전 청크의 실행 사실** (T18). 없으면 `available=False` 와 이유 — 키가 언제나 있다.
        feedback = wire.unpack_exec_feedback(scene)
        self._last_feedback = feedback
        # latch 가 안 돈 프레임(씬 없음)에 지난 프레임의 신호가 남아 기록되지 않게 먼저 지운다.
        self._last_latch_signal = {}
        self._last_grasp_record = {}
        self._grasp_constraint_version = None
        # 씬을 못 얻은 청크가 지난 청크의 스냅샷으로 분류되지 않게 (T23).
        self._last_snapshot = None

        # `_scene_fn` 이 읽을 것들. refiner 가 콜백을 부를 때 인자로 넘길 수 없는 값이라
        # 여기 둔다 — 콜백 계약(`context -> (scene, q_now, certified)`)을 바꾸지 않으려는 것이다.
        self._last_finger = {}
        self._pending = {"scene": scene, "attention": attention, "timing": timing,
                         "chunk": chunk, "exec_feedback": feedback,
                         # T29: 정책 관측의 16D state — gripper 열이 촬영 시점 측정 개도다.
                         "state": policy_obs.get("state"),
                         # SUBTASK-c: `AttentionPolicy` 가 실은 subtask label (없으면 None).
                         "subtask": result.get("subtask") if isinstance(result, dict) else None}

        previous, executed_steps = self._continuity_input(feedback)
        context: dict[str, Any] = {"t_step": seq, "previous_physical_chunk": previous}
        if executed_steps is not None:
            context["previous_executed_steps"] = executed_steps
        # T41 b — 직전 청크가 통째로 HOLD 였으면 로봇은 q_now 에 정지해 있다. 꺼져 있으면 `{}` 이고
        # context 에 키가 생기지 않는다 (refiner 호출이 예전 그대로).
        self._last_rest_start = self._rest_start(feedback)
        if self._last_rest_start.get("applied"):
            context["start_at_rest"] = True
        # T43 Q — the servo observer propagates the applied rows of the previous chunk to this
        # capture (`servo.ServoObserver`). Only with `limits.servo_model`: otherwise no key.
        if getattr(self.to_config.limits, "servo_model", False):
            context["servo_feedback"] = {"feedback": feedback, "seq": seq}
        t = time.monotonic()
        refined = self.refiner.refine(chunk, context)
        timing["trajopt"] = (time.monotonic() - t) * 1000.0 - timing.get("ag3s", 0.0)
        refined = np.asarray(refined, np.float64)
        self._last_continuity.update(
            {k: v for k, v in (getattr(self.refiner, "last_continuity", None) or {}).items()})
        # 연속성 기준은 **로봇이 실제로 실행한 청크**여야 한다 — `_continuity_reference` 가
        # "앞 execution_length 스텝은 이미 실행됐다" 를 전제로 꼬리를 잘라 쓰기 때문이다
        # (`refiner.py:179-195`). shadow 에서 로봇이 실행하는 것은 reference 이므로 그것을
        # 물려준다. refined 를 물려주면 SQP 가 **날아간 적 없는 궤적**에서 이어지는 것으로
        # 계획하고, 그 오차는 어디에도 안 찍힌다.
        #
        # **T18 부터 이것은 피드백이 없을 때의 기본값이다.** 피드백이 오면 다음 요청에서
        # `_previous_plan` 과 그 피드백으로 다시 고른다 (`_continuity_input`) — 여기서 저장하는
        # 것은 *"보낸 계획"* 이지 *"실행된 것"* 이 아니다 (지침 §6.4).
        self._previous_chunk = chunk if self.shadow else refined
        self._previous_plan = {"seq": seq, "refined": refined, "reference": chunk}

        refined = self._preserve_grippers(chunk, refined)
        verdict = self._verdict(chunk)
        # 응답 `notes` 에 **피드백을 받았는가** 한 줄 (T18). 로컬 기록만 보고도 서버가 이 청크를
        # 계획할 때 무엇을 알았는지 읽을 수 있어야 한다.
        verdict.notes.extend(self._feedback_notes())
        timing["total"] = (time.monotonic() - started) * 1000.0

        if self.recorder is not None:
            self._record(seq, chunk, refined)

        ag3s_block = self._ag3s_block(verdict)
        if ag3s_block is not None:
            # **서버 로그에도 사유를 남긴다.** 첫 live smoke 에서 응답에도 서버 로그에도 없어서
            # 왜 절반이 HOLD 인지 알 수 없었다 (2026-09-25). 와이어에만 싣고 여기 안 찍으면
            # 로컬 기록을 못 얻는 상황(수동 실행·정책만 띄운 세션)에서 같은 일이 되풀이된다.
            from benchmark.ag3s.runtime import degradation

            print(f"[safe_policy] seq={seq} ag3s={ag3s_block['status']} "
                  f"(validity={ag3s_block['validity']}, "
                  f"grounding={ag3s_block['grounding_status']}) — "
                  + (degradation.explain(ag3s_block["notes"])
                     or "; ".join(ag3s_block["notes"][:2]) or "no reason recorded"))

        # **판정 사유도 서버 로그에** (T23). `budget_only` 만 있는 청크는 조용히 둔다 — 거의 매
        # 청크가 예산으로 끝나므로 그 줄은 다른 것을 밀어낸다.
        if any(r.get("kind") != "budget_only" for r in (self._last_reasons or ())):
            print(f"[safe_policy] seq={seq} verdict={'execute' if verdict.safe else 'HOLD'} — "
                  + "; ".join(f"{r['kind']}: {r['detail']}" for r in self._last_reasons))

        extra = {k: v for k, v in result.items() if k != "actions"}
        if self._seed_applied is not None:
            # T39 — 적용했다는 회신. 적용한 요청에만 — 없는 요청의 응답은 T39 전과 같다.
            extra[wire.POLICY_SEED] = self._seed_applied
        # **`actions` 는 shadow 에서도 refined 다.** 서버는 자기가 계산한 것을 그대로 말하고,
        # 무엇을 실행할지는 로컬이 고른다 — `SafetyVerdict` 가 판정이지 명령이 아닌 것과 같은
        # 계약이다.
        #
        # **원본 청크는 이제 closed loop 에서도 싣는다** (T9). 그 전까지는 shadow 에서만 실었고,
        # 그 이유는 키의 있음/없음이 **모드의 신호**였기 때문이다. 그 겸직을 끊었다 — 모드는
        # `shadow=` 인자가 명시적으로 말하고, `actions_reference` 는 데이터일 뿐이다.
        #
        # 왜 지금 필요한가: 충돌 제약이 하나도 활성이 아닌 판에서도 TO 가 청크를 고치므로
        # (실행 창 안 중앙값 2.6°, 최대 8.8°), 남은 변형은 전부 목적함수가 만든 것이다. 그 크기를
        # closed loop 에서 재려면 `actions`(refined) 옆에 원본이 있어야 한다. 대가는 응답 크기다
        # (planning 기록 한 줄 19,837 B → shadow 수준 36,704 B).
        return wire.pack_response(refined, verdict, seq=seq, timing_ms=timing,
                                  field=self._field_provenance(),
                                  actions_reference=chunk,
                                  shadow=self.shadow,
                                  ag3s=ag3s_block,
                                  to=self._to_block(),
                                  extra=extra)

    def _continuity_input(self, feedback: dict[str, Any]
                          ) -> tuple[Optional[np.ndarray], Optional[int]]:
        """연속성 항에 줄 `(이전 청크, 실행된 스텝 수)`. 실행 수가 `None` 이면 refiner 가 K 를 가정한다.

        **계획 캐시와 실행 이력을 여기서 잇는다** (지침 §6.4). 캐시(`_previous_plan`)는 서버가
        **보낸** 것이고, 피드백은 로컬이 **실행한** 것이다:

        | 피드백 | 이전 청크 | 실행 수 | 결과 |
        |---|---|---|---|
        | 없음 (`available=False`) | `_previous_chunk` (예전 기본값) | `None` | K 스텝 실행을 **가정** (예전 그대로) |
        | seq 가 캐시와 다름 | `None` | — | 어느 계획이 실행됐는지 모른다 → 연속성 항 없음 |
        | `n_exec == 0` (HOLD) | 캐시 | `0` | refiner 가 `None` 을 낸다 — 날지 않은 궤적의 꼬리를 쓰지 않는다 |
        | 앞에서부터 `k` 스텝 실행 | 로컬이 **고른** 쪽 (refined/reference) | `k` | 인덱스 `k` 부터 정렬 |
        | 실행이 앞에서부터가 아니다 | `None` | — | 정렬 기준이 없다 |
        """
        plan = self._previous_plan
        info: dict[str, Any] = {
            "plan_seq": None if plan is None else int(plan["seq"]),
            "feedback_seq": feedback.get("seq"),
            "previous_chunk": None,
            "n_exec": feedback.get("n_exec"),
        }
        self._last_continuity = info
        if not feedback.get("available"):
            info["mode"] = "assumed (no feedback)"
            if self._previous_chunk is not None:
                info["previous_chunk"] = "reference" if self.shadow else "refined"
            return self._previous_chunk, None
        if plan is None:
            info["mode"] = "no cached plan (first chunk after reset or server restart)"
            return None, None
        if int(feedback["seq"]) != int(plan["seq"]):
            info["mode"] = (f"feedback is about seq {feedback['seq']} but the cached plan is "
                            f"seq {plan['seq']} — which plan executed is unknown")
            return None, None
        flags = list(feedback.get("executed") or ())
        n_exec = int(feedback["n_exec"])
        prefix = next((i for i, f in enumerate(flags) if not f), len(flags))
        if n_exec != prefix:
            info["mode"] = (f"execution was not a prefix ({flags}); no alignment exists")
            return None, None
        which = feedback.get("executed_chunk")
        if which not in ("refined", "reference"):
            which = "reference" if self.shadow else "refined"
        info["mode"] = "feedback"
        info["previous_chunk"] = which if n_exec > 0 else None
        return np.asarray(plan[which], np.float64), n_exec

    def _rest_start(self, feedback: dict[str, Any]) -> dict[str, Any]:
        """T41 b — 이 청크의 계획을 **정지 상태에서 출발**시키나. `limits.rest_start` 가 꺼져 있으면 `{}`.

        정지라고 아는 경우는 하나다: 직전 청크의 **모든 스텝이 HOLD** 였다 (`exec_feedback.n_exec == 0`).
        HOLD 는 한 자세를 8 스텝 명령하므로 로봇은 그 안에 멈춘다 (T41 c: `measured` · `fixed` 모두 8 스텝
        값과 64 스텝 값의 차이 < 0.1°). 일부만 HOLD 였거나 피드백이 없으면 속도를 모른다 — 적용하지 않는다.
        """
        if not getattr(self.to_config.limits, "rest_start", False):
            return {}
        out: dict[str, Any] = {"enabled": True, "applied": False,
                               "tolerance_rad": self.to_config.limits.rest_start_tolerance}
        if not feedback.get("available"):
            out["reason"] = f"no exec_feedback ({feedback.get('reason')}); velocity unknown"
            return out
        n_exec, n_steps = int(feedback.get("n_exec") or 0), int(feedback.get("n_steps") or 0)
        if n_steps > 0 and n_exec == 0:
            out.update(applied=True,
                       reason=(f"previous chunk seq {feedback.get('seq')} was held for all "
                               f"{n_steps} steps (hold_kind={feedback.get('hold_kind')}); "
                               "the robot is at rest at q_now"))
        else:
            out["reason"] = (f"previous chunk seq {feedback.get('seq')} executed {n_exec}/{n_steps} "
                             "steps; the robot is not known to be at rest")
        return out

    def _feedback_notes(self) -> list[str]:
        """응답 `notes` 에 더할 줄. 피드백 한 줄 + (있으면) 연속성·latch 신호 한 줄씩."""
        fb = self._last_feedback
        if fb.get("available"):
            line = f"피드백 수신: seq {fb['seq']}, exec {fb['n_exec']}/{fb['n_steps']}"
            if fb.get("n_hold"):
                line += f" (HOLD {fb.get('hold_kind')}: {fb['n_hold']} steps)"
        else:
            line = f"피드백 없음: {fb.get('reason', '')}"
        notes = [line]
        cont = self._last_continuity or {}
        if fb.get("available") and cont.get("mode") and cont.get("mode") != "feedback":
            notes.append(f"continuity reference dropped: {cont['mode']}")
        elif fb.get("available") and not cont.get("used") and cont.get("aligned_from_step") == 0:
            notes.append("continuity reference dropped: previous chunk did not execute (n_exec=0)")
        latch = self._last_latch_signal or {}
        if latch.get("source") == "planned":
            notes.append(latch["note"])
        # T41 b — 정지 출발 범위로 푼 청크만 한 줄 (꺼져 있으면 notes 가 예전 그대로).
        rest = self._last_rest_start or {}
        if rest.get("applied"):
            record = ((getattr(self.refiner.last_result, "metrics", None) or {}).get("rest_start")
                      if self.refiner.last_result is not None else None) or {}
            line = "rest start: " + str(rest.get("reason"))
            if record:
                line += (f"; plan step 0 is {np.degrees(record['step0_offset_rad']):.2f} deg from "
                         f"q_now (policy {np.degrees(record['reference_step0_offset_rad']):.2f} deg)")
            notes.append(line)
        # T22 — 파지 상태가 **바뀐** 청크(attach · detach · 거절 · 상태 전이)만 한 줄.
        grasp = self._last_grasp_record or {}
        if grasp.get("transition") and grasp.get("note"):
            notes.append(f"grasp: {grasp['note']}")
        return notes

    def _to_block(self) -> Optional[dict[str, Any]]:
        """**최적화기가 몇 번 돌았고 무엇에 걸렸나** (T15). 없으면 `None` — 키가 안 실린다.

        `notes` 의 문장으로는 셀 수 없었다. *"청크가 바뀌었다면 몇 번 돌았나"* 가 언제나 다음
        질문이고, 그때 기록이 없으면 실행을 한 번 더 해야 한다 — 같은 결핍에 여섯 번 막혔다.

        `collision_enabled` 를 여기 함께 싣는다. 이 설정으로 돈 기록이 *"위반 0"* 으로 읽히면 안
        되고, `notes` 는 사람이 읽는 쪽이라 기계가 세려면 값이 있어야 한다.
        """
        result = self.refiner.last_result
        metrics = dict(getattr(result, "metrics", None) or {}) if result is not None else {}
        if result is None:
            return None
        return {
            "status": getattr(getattr(result, "status", None), "value", None),
            "iterations": int(getattr(result, "iterations", 0) or 0),
            "solve_ms": round(float(getattr(result, "solve_time_ms", 0.0) or 0.0), 3),
            "sqp_iterations": int(metrics.get("sqp_iterations", 0) or 0),
            "qp_iterations": int(metrics.get("qp_iterations", 0) or 0),
            "max_iterations": int(metrics.get("max_iterations", 0) or 0),
            "time_budget_ms": float(metrics.get("time_budget_ms", 0.0) or 0.0),
            "time_budget_hit": bool(metrics.get("time_budget_hit", False)),
            "max_iterations_hit": bool(metrics.get("max_iterations_hit", False)),
            "collision_enabled": bool(metrics.get("collision_enabled", True)),
            # T37 — 이 청크의 목적함수에 continuity 항이 있었나 · 없으면 왜 (로컬 기록에서도 센다).
            "continuity_active": bool(metrics.get("continuity_active", False)),
            "continuity_reason": metrics.get("continuity_reason"),
            # T41 b — 정지 출발 범위로 푼 청크에만 키가 생긴다.
            **({"rest_start": _jsonable(metrics["rest_start"])} if "rest_start" in metrics else {}),
            # T43 Q — servo / sweep rows: only when on.
            **({"path_check": _jsonable(metrics["path_check"])} if "path_check" in metrics else {}),
            "reference_deviation": round(
                float(getattr(result, "reference_deviation", 0.0) or 0.0), 6),
        }

    def _ag3s_block(self, verdict: wire.SafetyVerdict) -> Optional[dict[str, Any]]:
        """`ag3s_status` 가 `ok` 가 아닐 때의 **사유**. `ok` 면 `None` — 키가 안 실린다.

        **이 블록이 없어서 첫 live smoke 가 막혔다** (2026-09-25). 2 청크 중 1 개가
        `max_violation_m = 0.0` 인데 `degraded` 로 HOLD 됐고, 궤적이 아니라 기하 인증이 실패한
        것인데 **왜인지가 응답에도 서버 로그에도 없었다.** 사유는 줄곧 여기 있었다 —
        `CollisionConstraintSet.notes` 다. 나가는 길이 없었을 뿐이다: 응답의 `notes` 는
        `TrajOptResult.notes`(최적화기 쪽)이고 지각 쪽이 아니다.

        **`ok` 일 때 `None` 인 것이 요구사항이다.** 정상 프레임의 응답이 T0 때와 한 바이트도
        달라지면 회귀 기준선이 재현되지 않는다.

        `reasons` 가 비는 일은 없다 — `degradation.ensure_reason` 이 마지막 관문에서 채운다.
        비어 있다면 그것은 **옛 서버**라는 뜻이고, 그 조합(`status != ok` + 빈 블록)을 로컬이
        구분할 수 있게 두는 것이 `unpack_ag3s` 가 상태 객체를 만들지 않는 이유다.
        """
        from benchmark.ag3s.runtime import degradation

        if verdict.ag3s_status == "ok":
            return None
        cs = self._last_constraint_set
        notes = [str(n) for n in (getattr(cs, "notes", ()) or ())]
        block: dict[str, Any] = {
            "status": verdict.ag3s_status,
            "validity": getattr(getattr(cs, "validity", None), "value", "unknown"),
            "grounding_status": getattr(
                getattr(cs, "grounding_status", None), "value", "unknown"),
            "reasons": degradation.reasons(notes),
            "notes": notes,
        }
        if cs is None:
            # 제약 집합이 아예 없다 = 씬을 못 얻었다. 그 이유는 refiner 가 붙들고 있다
            # (`_field_provenance` 가 쓰는 것과 같은 값) — 여기서도 싣는다. 그러지 않으면
            # `status: no_geometry` 만 나가고 왜 지각이 안 돌았는지는 다시 서버 안에 남는다.
            why = getattr(self.refiner, "last_failure", None)
            block["notes"] = [
                "AG3S produced no constraint set for this chunk"
                + (f": {why}" if why else "")]
            block["reasons"] = []
        return block

    def _field_provenance(self):
        """이번 청크의 거리장 출처. 없으면 `unavailable` 로 **이유를 달아** 돌려준다.

        서버는 `new` 나 `unavailable` 만 찍는다 — planning frame 마다 AG3S 를 돌리므로
        여기서 필드를 이어 쓰는 경로가 없다. `carried` 와 `stale` 은 청크 하나가 덮는
        8 개 control frame 에서 생기고, 클라이언트가
        `FieldProvenance.applied_by_client()` 로 채운다.
        """
        from benchmark.ag3s.fields.provenance import FieldProvenance

        cs = self._last_constraint_set
        if cs is None:
            failure = getattr(self.refiner, "last_failure", None)
            return FieldProvenance.unavailable(
                "AG3S did not produce a constraint set for this chunk"
                + (f": {failure}" if failure else ""))
        field = getattr(cs, "esdf", None)
        if field is None:
            return FieldProvenance.unavailable(
                f"the constraint set carried no ESDF field (status={cs.status.value}, "
                f"backend={self.ag3s.config.esdf.backend})")
        prov = getattr(field, "provenance", None)
        if prov is None:
            # 필드는 있는데 도장이 없다 = builder 가 안 찍은 것이다. 조용히 `new` 로
            # 만들어 주면 배선 결함이 정상으로 보이므로 그렇게 하지 않는다.
            return FieldProvenance.unavailable(
                f"the {type(field).__name__} carried no provenance stamp — the builder did "
                "not set it, so this chunk's geometry cannot be dated")
        return prov

    # ------------------------------------------------------------------------------------
    def _scene_fn(self, context: Optional[dict]):
        """refiner 계약. 예외를 삼키지 않는다 — refiner 가 감싸고 노트를 낸다."""
        scene = self._pending.get("scene", {})
        observations = wire.unpack_camera_observations(
            scene, robot_model=self.ag3s.robot_model,
            attention=self._pending.get("attention"),
        )
        if not observations:
            # 카메라가 하나도 없다. 예외 대신 None 을 돌려주면 refiner 가 정책 청크를 그대로
            # 통과시키고 노트를 남긴다. 그 프레임은 `safe=False` 로 나가므로 로컬이 hold 한다.
            self._last_constraint_set = None
            raise ValueError("no camera observations in the request")

        manipulators = list(scene.get("active_manipulators", ()) or ())
        # **T29: 손가락 관절을 먼저 놓는다** — 이번 프레임의 모든 FK (self-filter · depth mask ·
        # 미세 창 · reach · attach · TO) 가 촬영 시점 개도를 보게. 파지 판단(reach)보다 앞이다.
        self._apply_finger_state()
        # **파지 상태를 먼저 갱신한다** (T22, 지침 §6.3: 피드백 → 파지 상태 → 기하 → TO).
        # attach/detach 가 이번 프레임의 거리장(쥔 물체 파내기 · target-free 계층)과 제약(attached
        # 행 · 접촉 허용)에 바로 들어가야 한다. 뒤에서 하면 한 프레임 늦고, 그 프레임을 다시 지으려면
        # `process_multi_debug` 를 두 번 불러야 해서 grounding 의 프레임 수·ESDF 증분이 두 번 센다.
        if self._latch.evidence:
            self._advance_grasp(observations, manipulators)
        self._grasp_constraint_version = self._grasp_state_version
        # T26 §4: 닫힘 시도부터 놓을 때까지 조작 대상이 바뀌면 안 된다 — 바뀌면 attach 전 프레임에
        # 사과가 손가락의 거리장으로 돌아온다. AG3S 는 이 동안 도전자를 세지 않는다.
        set_grasp = getattr(self.ag3s, "set_grasp_active", None)
        if callable(set_grasp):
            set_grasp(self._latch.phase in (GraspPhase.CLOSING, GraspPhase.HELD))
        # SUBTASK-c: 이번 요청의 subtask label 과 latch 가 PLACED 인지 (B3). label 이 없으면 None —
        # AG3S 는 그것을 "label 없음" 으로 센다. **PLACED 만** 넘긴다: T34 의 거짓 attach 회수도
        # detach 지만 놓은 것이 아니다 (회수 뒤 phase 는 LATCHED). gate (`clustering.subtask_gate`)
        # 가 꺼져 있으면 AG3S 는 기록만 한다.
        set_subtask = getattr(self.ag3s, "set_subtask", None)
        if callable(set_subtask):
            set_subtask(_subtask_probabilities(self._pending.get("subtask")))
        set_placed = getattr(self.ag3s, "set_placed", None)
        if callable(set_placed):
            set_placed(self._latch.phase is GraspPhase.PLACED)

        newest = max(observations, key=lambda o: o.timestamp)
        execution_path = self._execution_path(np.asarray(newest.robot_state, np.float64))

        t = time.monotonic()
        # `process_multi_debug` 는 `process_multi` 와 **같은 계산**이고 중간 결과를 버리지 않고
        # 돌려줄 뿐이다 (`AG3S.process_multi`는 이 호출의 [0]이다) — 기록기가 없어도 비용이 없다.
        constraint_set, debug = self.ag3s.process_multi_debug(
            observations,
            phase=scene.get("phase") or self.default_phase,
            active_manipulators=manipulators,
            # 목적지는 **AG3S 가 가진 것**이다 (T26 — 인지 등록 또는 reset 주입). 여기서 넘기지
            # 않는다: 넘기면 AG3S 는 그것을 주입으로 받아 등록보다 앞세운다.
            static_geometry=self.static_geometry or None,
            # T21b/T26 §6: 정책 reference 청크의 **실행 창** (K = execution_length) 행 — 미세 창이
            # 손이 지금 있는 곳뿐 아니라 이번 청크 동안 지나갈 곳을 덮는다.
            **({} if execution_path is None else {"execution_path": execution_path}),
        )
        self._pending.setdefault("timing", {})["ag3s"] = (time.monotonic() - t) * 1000.0
        self._last_constraint_set = constraint_set
        self._last_debug = debug
        self._run_latch(constraint_set, observations, manipulators)
        self._finish_grasp_record(constraint_set)
        # T37 (K1): 이번 청크의 **최종** latch 상태로 continuity 항을 정한다 (`_run_latch` 뒤 — 기록의
        # `grasp.state` 와 같은 값). refiner 는 `scene_fn` 다음에 같은 `context` 로 기준을 고른다.
        off = self._grasp_continuity_gate()
        if context is not None:
            if off:
                context["continuity_off"] = off
            else:
                context.pop("continuity_off", None)
        self._last_attention_by_camera = {
            o.camera_id: np.asarray(o.attention_map)
            for o in observations if getattr(o, "attention_map", None) is not None
        }

        snapshot = scene_from_constraint_set(
            constraint_set, self.linearizer.robot_radii, self.to_config)
        # 판정이 위반 행을 **최적화기가 본 것과 같은 씬으로** 다시 분류한다 (T23, `_verdict`).
        self._last_snapshot = snapshot
        # 촬영 시점 중 **가장 최신** 자세가 TO 가 계획을 시작하는 곳이다. AG3S 도 같은 규칙을
        # 쓰므로 (`process_multi` 의 robot_state 기본값) 둘이 어긋나지 않는다.
        q_now = np.asarray(
            max(observations, key=lambda o: o.timestamp).robot_state, np.float64)
        self._last_q_now = q_now
        # SUBTASK-e: 판정(`_verdict`)과 **같은 규칙**. 여기서 다르면 sqp 가 `violated` 로 내린 청크를
        # 판정이 인증됐다고 읽는다 (또는 그 반대) — HOLD 를 만드는 두 곳이 같은 술어를 쓴다.
        certified = geometry_certified(constraint_set)
        return snapshot, q_now, certified

    # --- T29: finger joints ---------------------------------------------------------------
    @staticmethod
    def _build_finger_map(constraint_model, filter_model):
        """`FingerJointMap` over the model that has finger parameters, or None (mock models)."""
        for model in (constraint_model, filter_model):
            if model is not None and getattr(model, "param_joint_names", ()):
                from benchmark.ag3s.robot_models.gripper_state import FingerJointMap

                return FingerJointMap(model)
        return None

    def _finger_linearizers(self) -> list:
        out = [self.linearizer]
        optimizer = getattr(self.refiner, "optimizer", None)
        lin = getattr(optimizer, "linearizer", None)
        if lin is not None and lin is not self.linearizer:
            out.append(lin)
        return [x for x in out if getattr(x, "param_names", ())]

    def _measured_openings(self) -> tuple[dict[str, float], str]:
        """`({hand: norm}, source)` — 촬영 시점 측정 개도 (클라이언트 규약, 1 = 열림).

        1순위는 정책 관측 16D `state` 의 gripper 열 (`self.gripper_columns`, 16D = 7 · 15) 이다 —
        클라이언트 `build_obs` 가 `|qpos| / |RBY1_GRIPPER_OPEN|` 로 쓴 값이고, 같은 요청의 카메라와
        같은 순간에 찍힌다. 없으면 실행 피드백의 `measured_gripper` (같은 식, T18). 둘 다 없으면 빈 dict.
        """
        state = self._pending.get("state")
        left_col, right_col = self.gripper_columns
        if state is not None:
            arr = np.asarray(state, np.float64).reshape(-1)
            if arr.shape[0] > right_col and np.all(np.isfinite(arr[[left_col, right_col]])):
                return ({"left": float(arr[left_col]), "right": float(arr[right_col])},
                        f"state[{left_col}],state[{right_col}]")
        fb = self._pending.get("exec_feedback") or {}
        if fb.get("available") and fb.get("measured_gripper") is not None:
            mg = np.asarray(fb["measured_gripper"], np.float64).reshape(-1)
            if mg.shape[0] >= 2 and np.all(np.isfinite(mg[:2])):
                return ({"left": float(mg[0]), "right": float(mg[1])},
                        f"exec_feedback.measured_gripper (seq {fb.get('seq')})")
        return {}, "none"

    def _apply_finger_state(self) -> None:
        """손가락 관절 — AG3S 두 모델(촬영 시점 측정값)과 TO 창(스텝별 포락선)에 (T29).

        | 어디 | 값 |
        |---|---|
        | self-filter · 제약 모델의 numeric FK (이번 프레임) | 측정 개도 |
        | TO 창 스텝 0 | 측정 개도 |
        | TO 창 스텝 k ≥ 1 | `max(측정, max(clip(명령[0..k], 0, 1)))` (`gripper_state.opening_envelope`) |

        손가락은 결정 변수가 아니다 — linearizer 의 parameter 이고 Jacobian 은 팔 관절만이다.
        """
        fmap = self._finger_map
        if fmap is None:
            return
        measured, source = self._measured_openings()
        set_fingers = getattr(self.ag3s, "set_finger_joints", None)
        if not measured:
            if callable(set_fingers):
                set_fingers(None, record={"source": "model_default",
                                          "note": "no gripper opening in the request"})
            for lin in self._finger_linearizers():
                lin.set_joint_parameter_path(None)
            self._last_finger = {"source": "model_default",
                                 "note": "no gripper opening in the request (state/feedback)"}
            return
        values = fmap.joint_values(measured)
        record = {"source": source, "opening_norm": dict(measured), "joints": dict(values)}
        if callable(set_fingers):
            set_fingers(values, record=record)
        else:  # AG3S 가 이 메서드를 모르면 모델에 직접
            for model in (getattr(self.ag3s, "robot_model", None), self.constraint_robot_model):
                setter = getattr(model, "set_joint_parameters", None)
                if callable(setter):
                    names = set(getattr(model, "param_joint_names", ()))
                    setter({k: v for k, v in values.items() if k in names})
        chunk = self._pending.get("chunk")
        left_col, right_col = self.gripper_columns
        commanded: dict[str, np.ndarray] = {}
        if chunk is not None and len(chunk) and chunk.ndim == 2 and chunk.shape[1] > right_col:
            commanded = {"left": np.asarray(chunk[:, left_col], np.float64),
                         "right": np.asarray(chunk[:, right_col], np.float64)}
        path_record: dict[str, Any] = {}
        for lin in self._finger_linearizers():
            default = self.constraint_robot_model.param_vector()
            P, openings = fmap.path(measured, commanded, lin.horizon, default=default)
            # 모델의 parameter 순서와 map 의 순서는 같은 모델에서 왔다 — 다르면 여기서 죽는다.
            if tuple(lin.param_names) != tuple(fmap.param_joint_names):
                raise ValueError(f"finger parameter order differs: linearizer {lin.param_names} "
                                 f"vs map {fmap.param_joint_names}")
            lin.set_joint_parameter_path(P)
            path_record = {"horizon": int(lin.horizon),
                           "opening_norm": {h: [round(float(x), 4) for x in v]
                                            for h, v in openings.items()},
                           "commanded_available": bool(commanded)}
        self._last_finger = {**record, "model": "envelope(measured, running max of clip(cmd,0,1))",
                             "plan": path_record,
                             "convention": fmap.convention.record()}

    def _execution_path(self, q_now: np.ndarray) -> Optional[np.ndarray]:
        """`(K, nq)` — 정책 reference 청크의 실행 창을 전체 로봇 상태 벡터로 (T21b, T26 §6).

        refiner 는 `scene_fn` 을 reference 보다 먼저 부르지만, 청크 자체는 `infer` 가 이미 받아
        `_pending["chunk"]` 에 두었다. 자유 관절은 청크 열에서(`ChunkLayout.chunk_to_trajectory`),
        나머지(토르소 등)는 `q_now` 에서 채운다 — linearizer 가 FK 에 넣는 것과 같은 `full_q` 다.
        청크나 자세가 모양이 맞지 않으면 넘기지 않고(AG3S 는 `q_now` 창으로 물러난다) 이유를 남긴다.
        """
        chunk = self._pending.get("chunk")
        K = int(self.to_config.horizon.execution_length)
        self._last_execution_path = {"rows": 0, "source": None, "note": None}
        if chunk is None or not len(chunk):
            self._last_execution_path["note"] = "no reference chunk"
            return None
        if q_now.shape[0] != self.layout.nq_model:
            self._last_execution_path["note"] = (
                f"q_now has {q_now.shape[0]} entries, layout expects {self.layout.nq_model}")
            return None
        try:
            traj = self.layout.chunk_to_trajectory(np.asarray(chunk, np.float64))[:, :K]
            path = self.layout.full_q(traj, q_now).T
        except ValueError as exc:
            self._last_execution_path["note"] = f"{type(exc).__name__}: {exc}"
            return None
        self._last_execution_path = {"rows": int(path.shape[0]), "source": "reference",
                                     "execution_length": K, "note": None}
        return path

    def _run_latch(self, constraint_set, observations, manipulators) -> None:
        """잠금을 한 프레임 돌리고, 그것이 말할 때만 `attach`/`detach` 를 부른다.

        **판단과 호출이 여기서 만난다.** 잠금은 씬을 보지 않고 이름·점수·그리퍼만 보며,
        AG3S 는 파지 성공을 판정하지 않는다. 그 둘 사이를 잇는 것이 이 메서드다.

        예외를 삼키지 않는다 — 잠금이 조용히 죽으면 쥔 물체가 optimizer 에서 사라지고, 그것이
        E3(쥔 물체가 optimizer 에 도달하지 않는다)가 만들던 바로 그 구멍이다.
        """
        target = getattr(constraint_set, "target", None)
        label = self._identity.label(getattr(target, "centroid", None))
        if target is not None and label is not None and label == self._latch.manipulated:
            # 잠긴 이름이 지금도 target 으로 나온다 — attach 가 쓸 기하를 갱신해 둔다.
            self._latched_target = target

        hand = (manipulators or ["left"])[0]
        parent_link, allowed = self.grasp_links.get(
            str(hand), self.grasp_links.get("left", ("ee_finger_l1", ())))
        gripper = self._latch_gripper_signal(str(hand))

        placed = False
        # evidence 모드에서는 성공 판정도 `_advance_grasp` 가 (측정 개도와 함께) 본다 — 여기서
        # 다시 부르지 않는다.
        if self.placed_fn is not None and self._latch.holding and not self._latch.evidence:
            placed = bool(self.placed_fn(self._pending.get("scene", {}), constraint_set))

        # T26: 목적지는 AG3S 가 등록(또는 주입받은) destination 이다. 잠금에는 그 이름만 준다.
        destination = getattr(self.ag3s, "destination", None)
        self._latch.set_destination(None if destination is None else destination.label)
        self._destination_points = (None if destination is None
                                    else np.asarray(destination.points, np.float64))

        metrics = getattr(target, "metrics", {}) or {}
        event = self._latch.update(
            label=label,
            score=float(getattr(target, "confidence", 0.0) or 0.0),
            runner_up=float(metrics.get("runner_up_score", 0.0) or 0.0),
            gripper=gripper,
            placed=placed,
        )

        # **evidence 모드(T22 기본)에서는 `update` 가 attach/detach 를 내지 않는다** — 그 판단은
        # 제약을 짓기 전에 `_advance_grasp` 가 했다. 아래는 `legacy_gripper_attach=True` 의 옛 경로다.
        if event.attach and self._latched_target is not None:
            from benchmark.ag3s.constraints.attached import AttachRejected

            q = np.asarray(
                max(observations, key=lambda o: o.timestamp).robot_state, np.float64)
            try:
                self.ag3s.attach(self._latched_target, robot_state=q, parent_link=parent_link,
                                 allowed_contact_links=allowed, label="manipulated",
                                 reach=self._latch.config.reach,
                                 hold_link=self.hold_links.get(str(hand), parent_link))
                self._grasp_state_version += 1
                self._last_grasp_record.update({
                    "mode": "legacy_gripper", "transition": True,
                    "note": f"legacy: {event.note} (제약은 이 프레임 뒤에 붙는다)"})
            except AttachRejected as exc:
                ev = self._latch.revert_attach(exc.reason)
                self._last_grasp_record.update({
                    "mode": "legacy_gripper", "transition": True, "note": ev.note,
                    "rejected": exc.record()})
        elif event.detach:
            self.ag3s.detach()
            self._grasp_state_version += 1
            # T26: 목적지는 AG3S 의 것이고 에피소드 동안 남는다 (여기서 지우지 않는다).
            if not self._latch.evidence:
                self._last_grasp_record.update({
                    "mode": "legacy_gripper", "transition": True,
                    "note": f"legacy: {event.note}"})

        # (T26 전: "잠금이 목적지를 확정했고 지금 target 이 그것이면 그 점구름" — T26 뒤 target 은
        # 파지 뒤에도 사과이므로 그 길은 목적지를 영영 못 잡거나 사과를 목적지로 잡는다. 위에서
        # AG3S 의 destination 을 읽는 것으로 바꿨다.)

    def _grasp_hand(self, manipulators) -> tuple[str, str, Any]:
        hand = str((manipulators or ["left"])[0])
        parent_link, allowed = self.grasp_links.get(
            hand, self.grasp_links.get("left", ("ee_finger_l1", ())))
        return hand, parent_link, allowed

    def _advance_grasp(self, observations, manipulators) -> None:
        """**제약을 짓기 전에** 파지 상태를 갱신하고, 잠금이 말할 때만 `attach`/`detach` 를 부른다 (T22).

        입력은 셋이고 **계획 청크는 없다**:

        * 직전 청크의 실행 피드백 (T18) — 실제로 적용된 gripper 명령, 촬영 시점 측정 개도.
          피드백이 없으면(옛 클라이언트) 상태를 바꾸지 않고 이유를 남긴다.
        * 조작 대상 (T20 `AG3S.manipulated`) — id · 상태 · 마지막 관측 기하. **이번 프레임의
          grounding 은 아직 안 돌았으므로 직전 프레임까지의 것이다.** 파지 순간 물체는 테이블
          위에 있거나 손가락 사이라 한 청크 동안 거의 안 움직인다 — 그 가정을 `alignment` 에 남긴다.
        * 지금 자세 (촬영 시점 중 최신) — parent link 원점 ↔ 조작 대상 centroid 거리 = reach.

        `AG3S.attach` 가 reach 로 거절하면 잠금을 닫힘 시도로 되돌린다 (`revert_attach`).
        """
        if self._pending.get("grasp_observed"):
            return                      # 같은 요청에서 두 번 세지 않는다 (`would_violate` 등)
        self._pending["grasp_observed"] = True
        from benchmark.ag3s.constraints.attached import (
            AttachRejected,
            _link_pose_numeric,
            parent_reach,
        )

        hand, parent_link, allowed = self._grasp_hand(manipulators)
        cfg = self._latch.config
        signal = grasp_signal_from_feedback(
            self._pending.get("exec_feedback"), 0 if hand == "left" else 1,
            cfg.gripper_closed_below)
        newest = max(observations, key=lambda o: o.timestamp)
        q = np.asarray(newest.robot_state, np.float64)
        manip = getattr(self.ag3s, "manipulated", None)
        usable = bool(manip is not None and getattr(manip, "usable", False))

        model = self.constraint_robot_model
        reach = None
        alignment: dict[str, Any] = {
            "q_stamp": float(newest.timestamp),
            "pipeline_frame": int(getattr(self.ag3s, "frame_index", -1)),
            "geometry_frame": None if manip is None else int(manip.last_seen_frame),
            "geometry_time": None if manip is None else manip.last_seen_time,
            "geometry_age_frames": None if manip is None else int(manip.age_frames),
            "hand_shift_mm": None,
        }
        if manip is not None:
            try:
                reach = parent_reach(model, q, parent_link, manip.centroid)
                if self._last_q_now is not None and len(self._last_q_now) == len(q):
                    # 직전 요청 이후 parent link 원점이 움직인 거리 — 조작 대상 기하(직전 프레임
                    # 관측)와 지금 자세 사이의 시점 차이가 얼마나 비싼가.
                    origin_now = _link_pose_numeric(model, q, parent_link)[:3, 3]
                    alignment["hand_shift_mm"] = round(parent_reach(
                        model, self._last_q_now, parent_link, origin_now) * 1000.0, 2)
            except Exception as exc:  # noqa: BLE001 — FK 실패는 "reach 모름" 이다, 파지 확인 불가
                alignment["reach_error"] = f"{type(exc).__name__}: {exc}"
                reach = None

        placed = False
        if (self.placed_fn is not None and self._latch.holding
                and self._last_constraint_set is not None):
            # 직전 프레임의 제약 집합으로 판정한다 — 이번 것은 아직 없다. 측정 개도가 열렸을 때만
            # 쓰인다 (`observe_grasp`).
            placed = bool(self.placed_fn(self._pending.get("scene", {}),
                                         self._last_constraint_set))

        phase_before = self._latch.phase
        # T34 J2: 쥔 동안 직전 프레임에 조작 대상이 관측됐으면 쥔 구와 비교한 측정 (없으면 None).
        held_observation = None
        if self._latch.holding:
            held_fn = getattr(self.ag3s, "held_observation", None)
            held_observation = held_fn() if callable(held_fn) else None
        event = self._latch.observe_grasp(
            signal, reach_m=reach,
            manipulated_id=None if manip is None else int(manip.id),
            manipulated_state=None if manip is None else str(manip.state),
            placed=placed, held_observation=held_observation)
        rejected = None
        revoked = None
        if event.attach:
            try:
                if not usable:
                    raise AttachRejected("manipulated object has no usable geometry",
                                         parent_link=parent_link)
                self.ag3s.attach(manip.geometry, robot_state=q, parent_link=parent_link,
                                 allowed_contact_links=allowed, label="manipulated",
                                 reach=cfg.reach,
                                 hold_link=self.hold_links.get(hand, parent_link))
                self._grasp_state_version += 1
            except AttachRejected as exc:
                rejected = exc.record()
                event = self._latch.revert_attach(exc.reason)
        elif event.detach:
            self.ag3s.detach()
            self._grasp_state_version += 1
            # T26: 목적지는 AG3S 의 것이고 에피소드 동안 남는다 (여기서 지우지 않는다).
            if event.revoked:
                # T34 J2: 거짓 attach 회수. 조작 대상은 쥔 구가 아니라 다시 **자기 anchor**(마지막으로
                # 관측된 기하)로 대표된다 — detach 로 쥔 구가 빠지고, 잠금이 LATCHED 라 아래
                # `set_grasp_active(False)` 가 동결을 푼다. 기록에는 그 anchor 를 남긴다.
                revoked = {"reasons": list(((event.evidence or {}).get("revoke") or {})
                                           .get("reasons") or ()),
                           "held_observation": held_observation,
                           "anchor": None if manip is None else getattr(manip, "anchor", None)}
                print(f"[safe_policy] attach_revoked: {event.note}")

        self._last_grasp_record = {
            "mode": "evidence",
            "hand": hand,
            "parent_link": parent_link,
            "hold_link": self.hold_links.get(hand, parent_link),
            "state": self._latch.phase.value,
            "state_before": phase_before.value,
            "transition": bool(event.attach or event.detach or rejected is not None
                               or self._latch.phase is not phase_before),
            "attach": bool(event.attach),
            "detach": bool(event.detach),
            "rejected": rejected,
            # T34 J2: `{reasons, held_observation, anchor}` when this detach revoked a false attach.
            "attach_revoked": revoked,
            "note": event.note,
            "evidence": dict(event.evidence or {}),
            "alignment": alignment,
        }
        if self._last_grasp_record["transition"]:
            print(f"[safe_policy] grasp {phase_before.value} -> {self._latch.phase.value}: "
                  f"{event.note}")

    def _grasp_continuity_gate(self) -> Optional[str]:
        """이 청크에서 continuity 항을 끌 사유, 또는 `None` (T37 K1).

        `cost.grasp_continuity_off` (기본 False — O2 · T37 K1 뒤 기본 off, 켜려면
        `{"cost": {"grasp_continuity_off": true}}`) 가 True 이고 latch 가 `closing` · `held` 이면 끈다.
        continuity 기준은 직전 청크의 꼬리(정책이 **그때** 낸 손 자리)다. 닫는 동안과 쥔 동안 정책은
        매 청크 손 자리를 파지에 맞춰 고쳐 내는데, 기준이 손을 옛 자리로 끌어 손끝이 파지 자리에서
        벗어난다 — T35.diag: E3b ep1808 r1 t=120 · r2 t=112 (`closing`) 의 손끝 편차 19.6 / 34.7 mm
        가 continuity 만 끄면 0.0 mm. 나머지 항 · 제약(충돌 · limit · smoothness)은 그대로다.
        상태가 바뀌어 켜짐/꺼짐이 달라질 때 서버 로그에 한 줄 남긴다.
        """
        phase = self._latch.phase
        off = grasp_continuity_off_reason(
            phase, bool(getattr(self.to_config.cost, "grasp_continuity_off", False)))
        was = self._continuity_gate_on
        self._continuity_gate_on = off is not None
        if self._continuity_gate_on != was:
            print(f"[safe_policy] continuity term "
                  f"{'OFF — ' + off if off else 'back ON (grasp latch ' + phase.value + ')'}")
        return off

    def _finish_grasp_record(self, constraint_set) -> None:
        """제약을 지은 뒤 — 이번 제약이 어느 파지 상태로 지어졌는지 기록에 싣는다 (T22).

        `constraint_version` 은 `process_multi_debug` 를 부르기 **직전**의 상태 버전이고,
        `state_version` 은 지금 것이다. 둘이 다르면 이 청크의 제약은 낡은 파지 상태로 지어졌다
        (legacy 경로의 attach 가 그렇다).
        """
        rec = self._last_grasp_record
        if not rec:
            rec = {"mode": "evidence" if self._latch.evidence else "legacy_gripper",
                   "state": self._latch.phase.value, "transition": False}
            self._last_grasp_record = rec
        rec["state"] = self._latch.phase.value
        rec["state_version"] = int(self._grasp_state_version)
        rec["constraint_version"] = (None if self._grasp_constraint_version is None
                                     else int(self._grasp_constraint_version))
        attached = getattr(self.ag3s, "attached", None)
        in_cs = getattr(constraint_set, "attached", None)
        rec["attached_in_constraints"] = in_cs is not None
        rec["constraints_match_state"] = (in_cs is attached
                                          and rec["constraint_version"] == rec["state_version"])
        rec["attached"] = None if attached is None else {
            "label": getattr(attached, "label", None),
            "parent_link": getattr(attached, "parent_link", None),
            "source_candidate_id": getattr(attached, "source_candidate_id", None),
            "attached_at": getattr(attached, "attached_at", None),
            "n_points": (0 if getattr(attached, "points", None) is None
                         else int(len(attached.points))),
            # T32 H2: the held spheres (parent frame) and how they were fitted.
            "n_spheres": len(getattr(attached, "primitives", ()) or ()),
            "held": getattr(self.ag3s, "held_record", None),
        }

    def _latch_gripper_signal(self, hand: str) -> Optional[float]:
        """latch 가 볼 gripper 값. **측정 개도(피드백)가 먼저**, 없으면 계획 첫 행 (T18).

        예전에는 언제나 계획 청크의 첫 행이었다 — TO 판정과 클라이언트 실행 **전**의 값이다.
        T16 에서 닫힘이 제안됐지만 HOLD 로 실행되지 않았을 때 latch 는 그 제안을 닫힘으로 읽을 수
        있었다 (지침 §6.1). 측정 개도는 **이번 요청의 촬영 시점**에 잰 값이므로 직전 청크가 실제로
        손을 닫았는지를 말한다.

        **latch 의 판단 규칙은 바꾸지 않는다** (문턱 `gripper_closed_below` 그대로, T22 의 일).
        바뀌는 것은 입력의 출처이고, 그 출처를 `_last_latch_signal` 에 남긴다.
        """
        chunk = self._pending.get("chunk")
        planned = None
        if chunk is not None and len(chunk):
            left_col, right_col = self.gripper_columns
            column = left_col if hand == "left" else right_col
            if chunk.shape[1] > column:
                planned = float(chunk[0, column])
        fb = self._pending.get("exec_feedback") or {}
        if fb.get("available"):
            measured = np.asarray(fb["measured_gripper"], np.float64)
            value = float(measured[0 if hand == "left" else 1])
            self._last_latch_signal = {
                "hand": hand, "source": "measured", "value": value, "planned_first_row": planned,
                "note": f"gripper signal = measured (feedback seq {fb['seq']}, {hand} {value:.3f})"}
            return value
        self._last_latch_signal = {
            "hand": hand, "source": "planned", "value": planned, "planned_first_row": planned,
            "note": "gripper signal = planned (no feedback)"}
        return planned

    def _record(self, seq: int, reference: np.ndarray, refined: np.ndarray) -> None:
        """진단 기록. `bringup.LivePipeline._record`와 같은 계약 — 예외가 정책을 죽이면 안 된다.

        `--safe-remote` 로 도는 청크는 attention·grounding·거리장이 서버 밖으로 나가지 않으므로
        (응답은 안전 판정과 카메라별 attention 셀 하나뿐), 이것이 로컬 `--record-constraints`와
        동급으로 서버 쪽 파이프라인을 진단할 수 있는 유일한 자리다.
        """
        self._chunk_index += 1
        try:
            cs = self._last_constraint_set
            clearance = None
            centres = radii = None
            result = self.refiner.last_result
            if cs is not None and getattr(cs, "esdf", None) is not None and result is not None:
                all_centres = self.linearizer.sphere_states(
                    result.trajectory, self._last_q_now)[0]
                radii = self.linearizer.robot_radii
                # 계획 지평 전체의 여유거리 — 첫 스텝만 재면 TO 의 판정(지평 전체를 본다)과
                # 기록이 어긋나 "TO 는 violated 인데 기록은 위반 0" 이 나온다.
                clearance = (np.asarray(cs.esdf.distance(all_centres.reshape(-1, 3)), np.float64)
                             .reshape(all_centres.shape[:2])
                             - radii[None, :] - float(self.to_config.collision.esdf_margin))
                centres = all_centres[0]
            self.recorder.record(
                t_step=seq, chunk_index=self._chunk_index,
                constraint_set=cs, debug=self._last_debug,
                reference_chunk=reference, refined_chunk=refined,
                sphere_centres=centres, sphere_radii=radii,
                sphere_link_names=getattr(self.linearizer.robot_model, "sphere_link_names", None),
                clearance=clearance, to_result=result,
                attention_maps=self._last_attention_by_camera,
                occupancy=self._occupancy(),
                summary_extra=self._exec_summary(),
            )
        except Exception as exc:  # noqa: BLE001
            # **조용히 넘기지 않는다.** 2026-09-28 에 이 자리가 `print` 한 줄이라 75 chunk 가
            # 전부 실패하고도 실행이 끝까지 갔고, 끝나고 나온 것은 **빈 디렉토리**였다
            # (`'CuroboEsdfField' object has no attribute 'max_distance'`). 기록을 남기려고
            # 켠 flag 가 아무것도 남기지 않는 것은 실행을 한 번 더 돌려야 한다는 뜻이다.
            self._record_failures += 1
            log = logging.getLogger(__name__)
            if self._record_failures == 1:
                log.warning(
                    "!!! CONSTRAINT RECORDING IS FAILING — THIS RUN WILL HAVE NO RECORD !!!\n"
                    "    seq=%d: %s: %s\n"
                    "    디렉토리는 만들어져 있고 파일은 0 개입니다. 실행을 끝까지 돌려도 볼 "
                    "것이 없으므로 **지금 멈추고 고치는 편이 낫습니다.**\n%s",
                    seq, type(exc).__name__, exc, traceback.format_exc())
            else:
                log.warning("[safe_policy] constraint record failed at seq=%d (%d 번째): %s: %s",
                            seq, self._record_failures, type(exc).__name__, exc)

    def _exec_summary(self) -> dict[str, Any]:
        """`summary_json` 에 실을 `exec_feedback` 계열 세 키 (T18). **셋 다 언제나 실린다.**

        | 키 | 무엇 |
        |---|---|
        | `exec_feedback` | 이 요청이 실어 온 직전 청크의 실행 사실. 없으면 `{"available": false, "reason"}` |
        | `exec_continuity` | 연속성 항이 무엇을 근거로 정렬했나 (`previous_chunk` 가 `null` 이면 항 없음) |
        | `exec_latch_signal` | latch 가 본 gripper 값과 그 출처 (`measured` / `planned`) |
        """
        return {
            "exec_feedback": wire.exec_feedback_jsonable(self._last_feedback),
            "exec_continuity": dict(self._last_continuity or {}),
            "exec_latch_signal": dict(self._last_latch_signal or {}),
            # T22 — 파지 상태 · 근거 · 상태/제약 버전. 씬이 없어 latch 가 안 돈 청크는 `{}`.
            "grasp": _jsonable(self._last_grasp_record or {}),
            # T26 §6 — AG3S 에 넘긴 실행 구간 (행 수 · 출처, 못 넘겼으면 이유).
            "execution_path": _jsonable(self._last_execution_path or {}),
            # T29 — 손가락 관절: 측정 개도 · 출처 · 관절값 · TO 창의 스텝별 개도.
            "finger_joints": _jsonable(self._last_finger or {}),
            # T39 — 정책 RNG seed. `applied` 는 이 요청에 적용한 값, `episode` 는 이 에피소드의 값.
            # 둘 다 `null` 이면 seed 없이 서버 순서대로 이어진 noise 다.
            "policy_seed": {"applied": self._seed_applied, "episode": self._episode_seed},
            # T41 b — `limits.rest_start` 가 켜졌을 때만 (적용 여부 · 이유). 꺼져 있으면 키가 없다.
            **({"rest_start": _jsonable(self._last_rest_start)} if self._last_rest_start else {}),
            # T23 — 판정 사유와 위반 행 분류. `reasons` 가 `None` 이면 판정 전(기록 순서상 없음).
            "verdict": _jsonable({
                "policy": self.verdict_policy,
                "reasons": self._last_reasons,
                "classification": self._last_classification or {},
            }),
        }

    def _occupancy(self):
        """세 상태 점유 배열. `EsdfField` 가 아니라 그것을 만든 builder 안에 있다."""
        builder = getattr(self.ag3s, "_esdf_builder", None)
        return None if builder is None else getattr(builder, "_occupancy", None)

    def _preserve_grippers(self, original: np.ndarray, refined: np.ndarray) -> np.ndarray:
        """그리퍼 열은 정책이 낸 값 그대로. TO 는 그 열의 변수를 갖지도 않는다.

        `ChunkLayout` 이 이미 그리퍼를 결정 변수에서 빼고 `template` 로 되돌려 놓지만, 여기서
        한 번 더 강제한다. 이 두 열이 조용히 바뀌면 손이 엉뚱한 순간에 열리고, 그것은 궤적
        오차와 달리 눈에 띄지 않는다.

        열은 `self.gripper_columns` — **레이아웃에서 유도한 것**이고 모듈 상수가 아니다.
        `wire.GRIPPER_COLUMNS` 는 기본 차원(16D)의 편의값이라 14D 레이아웃에서는 손목 열을
        지키고 그리퍼를 최적화에 내준다.
        """
        out = np.array(refined, np.float64, copy=True)
        for col in self.gripper_columns:
            if col < out.shape[1]:
                out[:, col] = original[:, col]
        return out

    def _verdict(self, policy_chunk: np.ndarray) -> wire.SafetyVerdict:
        result = self.refiner.last_result
        cs = self._last_constraint_set
        ag3s_status = getattr(getattr(cs, "status", None), "value", "no_geometry")
        certified = geometry_certified(cs)
        notes = list(getattr(result, "notes", ()) or ())
        self._last_reasons = None
        self._last_classification = {}

        if result is None:
            # 씬을 못 얻었다. 정책 청크가 그대로 나가고, 그것은 검증된 적이 없다.
            #
            # **왜 못 얻었는지를 응답에 싣는다.** 예전에는 "no scene was available" 한 줄뿐이라,
            # 파지 다음 프레임부터 파이프라인이 통째로 멈춘 것을 14 프레임 동안 아무도 몰랐다
            # (2026-09-18, 원인은 `attached_parent_links` 미예약). 원인 문자열이 여기 있으면
            # 응답만 보고도 알 수 있다.
            #
            # **`verdict_reasons` 를 싣지 않는다** (T23). 최적화기가 돌지 않았으니 나눌 사유가
            # 없고, 이 응답의 모양은 T23 전과 같다 — 로컬은 `safe=False` · `geometry_certified=False`
            # 에서 `uncertified` 를 유도한다 (`wire.derive_verdict_reasons`). 기록에는 남긴다.
            why = getattr(self.refiner, "last_failure", None)
            self._last_reasons = [wire.make_reason(
                "uncertified", "no scene: the optimizer did not run"
                + (f" — {why}" if why else ""), carried_on_wire=False)]
            return wire.SafetyVerdict(
                ag3s_status=ag3s_status, geometry_certified=False,
                trajopt_status="no_solution", max_violation_m=float("inf"), safe=False,
                notes=notes + ["no scene was available; the policy chunk is unverified"
                               + (f" — {why}" if why else "")])

        status = getattr(result.status, "value", "unknown")
        violation = float(result.max_violation)
        # **그 위반을 만든 행이 무엇이었나.** 숫자는 줄곧 나갔지만 신원은 최적화기 안에만
        # 있었고, `T5f` 가 그 때문에 막혔다 (`sqp._finish` → `metrics`).
        pair = (getattr(result, "metrics", None) or {}).get("max_violation_pair")
        # **사유를 나눈다** (T23, 지침 §8.3 · §9). `TrajOptStatus.safe` 는 `violated` 하나에 파지
        # 접촉 · 가려진 target · 진짜 충돌을 섞는다 — T16 의 25/34 HOLD 가 그것이었다.
        reasons = self._reasons(result, cs, certified, violation, pair)
        action, holding = wire.gate_decision(reasons)
        if self.verdict_policy == "legacy":
            # T23 전 그대로. 기하 미인증은 이미 그 안에 들어가 있다 — sqp 가
            # `safety.require_certified_geometry` 를 보고 상태를 내린다.
            safe = bool(getattr(result.status, "safe", False))
        else:
            safe = action == "execute"
        self._last_reasons = reasons
        if reasons:
            notes.append(_reasons_note(reasons, safe=safe, policy=self.verdict_policy))
        return wire.SafetyVerdict(
            ag3s_status=ag3s_status, geometry_certified=certified, trajopt_status=status,
            max_violation_m=violation, safe=safe, notes=notes,
            max_violation_pair=pair, verdict_reasons=reasons)

    # --- 판정 사유 (T23) ------------------------------------------------------------------
    def _reasons(self, result, cs, certified: bool, violation: float,
                 pair: Optional[dict]) -> list[dict[str, Any]]:
        """`verdict_reasons` — 기하 인증 · 최적화기 종료 · 위반 행 · 예산을 각각 본다.

        순서는 기록을 읽는 순서다 (지각 → 최적화기 → 궤적). 판정은 `wire.gate_decision` 이
        사유 **전부**를 보고 내린다 — 하나라도 HOLD 면 HOLD 다.
        """
        reasons: list[dict[str, Any]] = []
        status = getattr(result.status, "value", "unknown")
        metrics = dict(getattr(result, "metrics", None) or {})
        tolerance = float(self.to_config.safety.violation_tolerance)

        if not certified:
            reasons.append(self._certification_reason(cs))
        elif subtask_no_target(cs):
            # SUBTASK-e: 인증된 기하지만 target 이 없다 — 실행되는 청크에도 "왜" 가 남게.
            reasons.append(self._subtask_no_target_reason(cs))
        if status in ("solver_failed", "unconstrained"):
            reasons.append(wire.make_reason(
                "unverified", f"trajopt_status={status}: the returned chunk was not checked "
                "against the scene", trajopt_status=status))
        elif violation > tolerance:
            reasons.extend(self._violation_reasons(result, cs, pair, tolerance))
        budget = bool(metrics.get("time_budget_hit") or metrics.get("max_iterations_hit")
                      or status == "feasible")
        if budget and status not in ("solver_failed", "unconstrained") and not any(
                r["kind"] in ("collision", "unverified") for r in reasons):
            reasons.append(wire.make_reason(
                "budget_only", f"the solve stopped on its "
                f"{'time budget' if metrics.get('time_budget_hit') else 'iteration cap'} "
                f"({metrics.get('sqp_iterations')} iteration(s)); the final check passed",
                time_budget_hit=bool(metrics.get("time_budget_hit")),
                max_iterations_hit=bool(metrics.get("max_iterations_hit")),
                sqp_iterations=metrics.get("sqp_iterations")))
        return reasons

    def _manipulated_record(self, cs) -> dict[str, Any]:
        """이 프레임의 조작 대상 기록 (T20 `ManipulatedIdentity.record()`), 없으면 `{}`.

        제약 집합의 `metrics["manipulated"]` 가 먼저다 — **이 프레임의** 스냅샷이다. 없으면
        `ag3s.manipulated` (같은 객체의 지금 값).
        """
        record = (getattr(cs, "metrics", None) or {}).get("manipulated")
        if isinstance(record, dict):
            return dict(record)
        live = getattr(self.ag3s, "manipulated", None)
        rec = getattr(live, "record", None)
        try:
            return dict(rec()) if callable(rec) else {}
        except Exception:  # noqa: BLE001 — 기록을 못 읽는 것으로 판정이 죽지 않는다
            return {}

    def _certification_reason(self, cs) -> dict[str, Any]:
        """기하 미인증을 `occluded_target` · `uncertified_waived` · `uncertified` 로 가른다.

        **`occluded` 인 것만으로는 미인증이 아니다** (T20, T14 seq 22): 상태가 `no_target` 이고
        (= 나머지 기하는 인증됐다, `validity` 가 VALID) manipulated 가 `occluded` 로 살아 있으며
        그 접촉 권한(`cs.manipulated`)이 이번 제약에 실려 있으면 `occluded_target` 이다.
        `lost` · 대상 없음 · degraded · incomplete 는 `uncertified` 다.
        """
        if cs is None:
            return wire.make_reason("uncertified", "AG3S produced no constraint set")
        status = getattr(getattr(cs, "status", None), "value", "unknown")
        grounding = getattr(getattr(cs, "grounding_status", None), "value", "unknown")
        validity = getattr(getattr(cs, "validity", None), "value", "unknown")
        man = self._manipulated_record(cs)
        evidence = {"ag3s_status": status, "grounding_status": grounding, "validity": validity,
                    "manipulated_id": man.get("id"), "manipulated_state": man.get("state"),
                    "manipulated_age_frames": man.get("age_frames")}
        geometry_ok = bool(getattr(cs, "geometry_certified", False))
        if (status == "no_target" and geometry_ok and man.get("state") == "occluded"
                and getattr(cs, "manipulated", None) is not None):
            return wire.make_reason(
                "occluded_target",
                f"no target this frame ({grounding}) but manipulated id={man.get('id')} is "
                f"occluded (unseen {man.get('age_frames')} frame(s)); its last geometry keeps the "
                "contact permission and the rest of the geometry is certified", **evidence)
        if not bool(self.to_config.safety.require_certified_geometry):
            return wire.make_reason(
                "uncertified_waived", f"geometry not certified (status={status}) but the server "
                "runs with safety.require_certified_geometry=False", **evidence)
        return wire.make_reason(
            "uncertified", f"AG3S could not certify the geometry (status={status}, "
            f"grounding={grounding}, validity={validity}"
            + (f", manipulated {man.get('state')}" if man.get("state") else "") + ")", **evidence)

    def _subtask_no_target_reason(self, cs) -> dict[str, Any]:
        """`subtask_no_target` — target 이 없는데 실행하는 이유를 기록에 남긴다 (SUBTASK-e, -f).

        `grounding_status` 가 둘을 가른다: `subtask_gated` (gate 가 비웠다, SUBTASK-e) ·
        `no_admissible` (놓은 뒤 home 에서 잡을 것이 없다, SUBTASK-f).

        증거는 `_certification_reason` 과 같은 키에 subtask label (`metrics["subtask"]`) 을 더한다 —
        "gate 가 무엇을 보고 비웠나" (label · released · placed_seen) 가 같은 줄에 있게.
        """
        status = getattr(getattr(cs, "status", None), "value", "unknown")
        grounding = getattr(getattr(cs, "grounding_status", None), "value", "unknown")
        validity = getattr(getattr(cs, "validity", None), "value", "unknown")
        metrics = getattr(cs, "metrics", None) or {}
        sub = metrics.get("subtask") if isinstance(metrics.get("subtask"), dict) else {}
        exclusion = metrics.get("exclusion") if isinstance(metrics.get("exclusion"), dict) else {}
        if grounding == "no_admissible":
            # SUBTASK-f: 놓은 뒤 home — 보이는 것이 놓을 곳 (crate) 뿐이라 잡을 것이 없는 것이 정상.
            why = (f"no target this frame: nothing admissible is in view after the object was "
                   f"placed (placed_seen={sub.get('placed_seen')}, label={sub.get('label')}, "
                   f"gate={sub.get('gate')}, released={sub.get('released')})")
        else:
            why = (f"no target this frame because the subtask gate withheld it (label="
                   f"{sub.get('label')}, blocked={sub.get('blocked')}, "
                   f"released={sub.get('released')})")
        return wire.make_reason(
            "subtask_no_target",
            why + "; nothing is carved, so the field holds every object at full clearance and "
            "the geometry is certified",
            ag3s_status=status, grounding_status=grounding, validity=validity,
            subtask_label=sub.get("label"), subtask_blocked=sub.get("blocked"),
            subtask_released=sub.get("released"), subtask_released_id=sub.get("released_id"),
            placed_seen=sub.get("placed_seen"), exclusion_source=exclusion.get("source"),
            exclusion_active=exclusion.get("active"))

    def _authorized_links(self, cs) -> frozenset:
        """접촉 권한이 있는 link — 제약을 지을 때 쓴 **같은** 집합 (`ClearancePolicy.authorized_links`)."""
        ctx = getattr(cs, "contact_context", None)
        policy = getattr(getattr(self.ag3s, "builder", None), "clearance_policy", None)
        if ctx is None or policy is None:
            return frozenset()
        return frozenset(str(n) for n in policy.authorized_links(ctx))

    def _violation_reasons(self, result, cs, pair: Optional[dict],
                           tolerance: float) -> list[dict[str, Any]]:
        """위반 행 **전부**를 분류해 `allowed_contact` / `unverified` / `collision` 사유로.

        최악 행 하나만 보지 않는다: 손가락이 사과를 −10 mm 파고들고(허용) 팔꿈치가 테이블을
        −3 mm 스치면(비허용), 최악 행만 보면 팔꿈치가 가려진다. 분류는 `classify_violations`.
        """
        scene = self._last_snapshot
        q_now = self._last_q_now
        trajectory = getattr(result, "trajectory", None)
        optimizer = getattr(self.refiner, "optimizer", None)
        linearizer = getattr(optimizer, "linearizer", None)
        if scene is None or q_now is None or trajectory is None or linearizer is None:
            return [wire.make_reason(
                "collision", f"{float(result.max_violation) * 1000:.1f} mm of penetration and the "
                "rows could not be classified (no scene snapshot on the server)",
                max_violation_pair=_jsonable(pair))]
        cls = classify_violations(
            linearizer, np.asarray(trajectory, np.float64), q_now, scene,
            authorized_links=self._authorized_links(cs), tolerance=tolerance,
            manipulated=self._manipulated_record(cs),
            candidate_sources={int(getattr(c, "id", -1)): getattr(
                getattr(c, "source_type", None), "value", str(getattr(c, "source_type", "")))
                for c in (getattr(cs, "candidates", None) or ())},
            pad_m=self.allowed_contact_pad_m,
            allow_unresolved_tier=self.allow_unresolved_contact_tier)
        self._last_classification = cls
        out: list[dict[str, Any]] = []
        summary = {k: cls[k] for k in ("n_violating", "n_allowed", "n_unresolved", "n_collision",
                                       "allowed_by_tier", "allowed_by_evidence",
                                       "residual_violation_m")}
        if cls["n_allowed"]:
            row = cls["allowed_worst"]
            out.append(wire.make_reason(
                "allowed_contact",
                f"{cls['n_allowed']} row(s): authorized {row['link']} ↔ manipulated "
                f"id={cls['manipulated_id']}, worst {row['clearance_m'] * 1000:.1f} mm "
                f"(tier {row['tier']}, by {row['evidence']})",
                row=row, manipulated_id=cls["manipulated_id"], **summary))
        if cls["n_unresolved"]:
            row = cls["unresolved_worst"]
            out.append(wire.make_reason(
                "unverified",
                f"{cls['n_unresolved']} authorized-link row(s) on the manipulated object were "
                f"answered by an unresolved tier ({row['tier']}), worst "
                f"{row['clearance_m'] * 1000:.1f} mm at {row['link']} — neither free space nor a "
                "verified contact (T21)", row=row, **summary))
        if cls["n_collision"]:
            row = cls["collision_worst"]
            out.append(wire.make_reason(
                "collision",
                f"{cls['n_collision']} row(s): {row['link']} ↔ "
                f"{row.get('label') or row['block']} {row['clearance_m'] * 1000:.1f} mm "
                f"({row['why']})", row=row, **summary))
        if not out:
            # `max_violation` 은 넘었는데 행 분류에서 위반이 안 나왔다 = 두 계산이 어긋났다.
            # 실행 쪽으로 읽지 않는다.
            out.append(wire.make_reason(
                "collision", f"{float(result.max_violation) * 1000:.1f} mm reported but no row "
                "re-measured above tolerance — the classification disagrees with the optimizer",
                max_violation_pair=_jsonable(pair), **summary))
        return out


#: `SafePolicy(verdict_policy=)` 의 선택지. 첫 항목이 기본이다.
VERDICT_POLICIES = ("reasons", "legacy")

#: 행 분류가 `allowed_contact` 로 받는 계층 — T21 `curobo_field.RESOLVED_TIERS` 중 조작 대상이
#: **들어 있는** 것 (`fine`). `target_free` 는 대상을 뺀 답이고 `static` 은 해석적 정적 기하라,
#: 그 답이 만든 위반은 정의상 대상과의 접촉이 아니다. `None` 은 계층을 말하지 못하는 필드
#: (단일 격자 — 계층이 하나뿐이다).
_CONTACT_TIERS = ("fine", None)
_NOT_TARGET_TIERS = ("target_free", "static")


#: T37 (K1) — continuity 항을 끄는 grasp latch 상태. 닫힘 시도(`closing`)와 파지 확인(`held`).
def geometry_certified(constraint_set: Any) -> bool:
    """이 제약 집합의 기하를 TO 가 **인증된 것으로** 읽어도 되나 — `_scene_fn` (→ sqp) 과
    `_verdict` (→ 사유) 가 같이 쓰는 술어 하나.

    `status == ok` 이거나, SUBTASK-e: target 이 없는 이유가 subtask gate 이고 carve 중인 것이 없다
    (`ag3s.types.subtask_no_target`). SUBTASK-f: gate on ∧ 확정 label home ∧ PLACED 도달 뒤의
    `no_admissible` 도 같은 조건으로 인증된다. 그 밖의 `no_target` (`no_seed` · `low_score` · `lost` ·
    놓기 전 `no_admissible` …) · degraded · incomplete 는 지금처럼 미인증이다. gate off 에서는 두 갈래
    모두 닫히므로 이 값은 `status == ok` 와 같다.
    """
    if constraint_set is None:
        return False
    status = getattr(getattr(constraint_set, "status", None), "value", None)
    return bool(status == "ok" or subtask_no_target(constraint_set))


def _subtask_probabilities(subtask: Any) -> Optional[dict[str, float]]:
    """`result["subtask"]` (`{"p": {...}, "argmax", "probe"}`) → `{"pick", "place", "home"}` 확률.

    없거나 모양이 틀리면 None (= 이 요청은 label 없음). 예외를 내지 않는다 — label 이 정책 호출이나
    지각을 죽이면 안 된다 (attention 과 같은 규칙).
    """
    if not isinstance(subtask, dict):
        return None
    p = subtask.get("p")
    if not isinstance(p, dict) or not p:
        return None
    try:
        return {str(k): float(v) for k, v in p.items()}
    except (TypeError, ValueError):
        return None


GRASP_CONTINUITY_OFF_PHASES = (GraspPhase.CLOSING, GraspPhase.HELD)


def grasp_continuity_off_reason(phase, enabled: bool = True) -> Optional[str]:
    """latch 상태 → continuity 항을 끌 사유 문자열, 또는 `None` (끄지 않음). T37 K1.

    `phase` 는 `GraspPhase` 또는 그 값(`"closing"` 등 — 기록의 `summary.grasp.state`). 재생 도구가
    기록된 상태로 같은 판단을 하도록 순수 함수로 둔다. 사유는 `context["continuity_off"]` 로
    refiner 에 가고 `metrics.continuity_reason` / `continuity_off_by` 에 남는다.
    """
    if not enabled or phase is None:
        return None
    try:
        phase = GraspPhase(phase)
    except ValueError:
        return None
    if phase not in GRASP_CONTINUITY_OFF_PHASES:
        return None
    return f"grasp latch {phase.value} (cost.grasp_continuity_off)"


def classify_violations(linearizer, trajectory: np.ndarray, q_now: np.ndarray, scene, *,
                        authorized_links, tolerance: float,
                        manipulated: Optional[dict] = None,
                        candidate_sources: Optional[dict] = None,
                        pad_m: float = 0.01,
                        allow_unresolved_tier: bool = False,
                        keep_rows: bool = False) -> dict[str, Any]:
    """위반 행(`clearance < −tolerance`) 을 하나씩 **허용 접촉 · 미해결 · 충돌** 로 가른다 (T23).

    값은 다시 계산하지 않는다 — `linearizer.clearances` / `esdf_clearance` 가 최적화기가 본 것과
    같은 행을 낸다 (`sqp._finish` 의 `worst_row` 와 같은 씬 · 같은 attached 상태).

    ESDF 행 하나가 `allowed_contact` 인 조건 (전부):

    1. 질의점이 로봇 구이고 그 link 가 `authorized_links` 에 있다 (쥔 물체의 점은 권한 대상이 아니다).
    2. 답한 계층이 조작 대상을 뺀 것이 아니다 (`target_free` · `static` 이면 대상이 아닌 무엇과의 위반).
    3. **그 행의 최근접 장애물이 조작 대상이다.** 판정 근거는 있는 것 중 첫째:
       `distance` — `_esdf_clearance` 와 같은 검사 (`|d_field − d_object| ≤ voxel`, 대상 점군/구로),
       `label` — 라벨 층이 `TARGET_LABEL`, `centroid` — T20 기록의 centroid 공
       (반지름 + 구 반지름 + 그 link 의 대상 마진 + `pad_m`).
    4. 계층이 풀린 것(`fine`)이다. 미세 창 밖(`coarse` · `window_edge` · `outside`) 이면 1–3 이
       맞아도 **미해결** (`unverified`) 로 따로 센다 — `allow_unresolved_tier` 면 허용으로.

    후보(primitive) 행은 1 과 "그 slot 의 후보가 `target` 이다" 로, 평면 행은 언제나 충돌이다.

    `keep_rows` 면 분류한 행 전부를 `rows` 로 싣는다 (분석·테스트용 — 기록에는 싣지 않는다,
    위반 행이 수백 개일 수 있다).
    """
    from benchmark.ag3s.types import TARGET_LABEL

    tol = float(tolerance)
    authorized = frozenset(str(a) for a in (authorized_links or ()))
    manipulated = dict(manipulated or {})
    sources = dict(candidate_sources or {})
    n_spheres = int(linearizer.n_spheres)
    states = linearizer.sphere_states(trajectory, q_now)
    centres = states[0]
    candidate, plane, _ = linearizer.clearances(trajectory, q_now, scene, states)
    esdf = linearizer.esdf_clearance(trajectory, q_now, scene, states)
    radii = np.asarray(linearizer.query_radii, np.float64).reshape(-1)

    def name(query: int) -> str:
        return str(linearizer._query_name(int(query)))

    rows: list[dict[str, Any]] = []

    # --- primitive 후보 행 ------------------------------------------------------------------
    if candidate.size:
        ids = np.asarray(scene.candidate_ids, np.int64).reshape(-1)
        for s, q, m in np.argwhere(candidate < -tol):
            cid = int(ids[m]) if m < ids.shape[0] else -1
            link = name(q)
            ok_link = q < n_spheres and link in authorized
            is_target = sources.get(cid) == "target"
            rows.append({"block": "candidate", "step": int(s), "query": int(q), "link": link,
                         "clearance_m": float(candidate[s, q, m]),
                         "point_m": [float(v) for v in centres[s, q]], "tier": None,
                         "label": sources.get(cid), "candidate_id": cid,
                         "evidence": "candidate_source" if ok_link else None,
                         "class": "allowed" if (ok_link and is_target) else "collision",
                         "why": ("authorized link on the target candidate" if ok_link and is_target
                                 else "unauthorized_link" if not ok_link
                                 else "not_target_candidate")})
    # --- 평면 행 — 테이블 · 바닥은 누구에게도 허용 접촉이 아니다 ---------------------------------
    if plane.size:
        for s, q, k in np.argwhere(plane < -tol):
            rows.append({"block": "plane", "step": int(s), "query": int(q), "link": name(q),
                         "clearance_m": float(plane[s, q, k]),
                         "point_m": [float(v) for v in centres[s, q]], "tier": None,
                         "label": f"plane[{int(k)}]", "evidence": None, "class": "collision",
                         "why": "plane"})
    # --- ESDF 행 ---------------------------------------------------------------------------
    if esdf.size:
        values = esdf[..., 0]
        hits = np.argwhere(values < -tol)
        if len(hits):
            rows.extend(_classify_esdf_rows(
                hits, values, centres, radii, scene, name, n_spheres, authorized, manipulated,
                pad_m=float(pad_m), target_label=TARGET_LABEL,
                allow_unresolved_tier=bool(allow_unresolved_tier)))

    def worst(cls: str) -> Optional[dict]:
        pick = [r for r in rows if r["class"] == cls]
        return min(pick, key=lambda r: r["clearance_m"]) if pick else None

    allowed = [r for r in rows if r["class"] == "allowed"]
    unresolved = [r for r in rows if r["class"] == "unresolved"]
    collision = [r for r in rows if r["class"] == "collision"]
    counts = lambda key, pick: {str(k): sum(1 for r in pick if r[key] == k)  # noqa: E731
                                for k in sorted({r[key] for r in pick}, key=str)}
    residual = [r["clearance_m"] for r in unresolved + collision]
    extra = {"rows": rows} if keep_rows else {}
    return {
        **extra,
        "tolerance_m": tol,
        "n_violating": len(rows),
        "n_allowed": len(allowed),
        "n_unresolved": len(unresolved),
        "n_collision": len(collision),
        "allowed_worst": worst("allowed"),
        "unresolved_worst": worst("unresolved"),
        "collision_worst": worst("collision"),
        "allowed_by_tier": counts("tier", allowed),
        "allowed_by_evidence": counts("evidence", allowed),
        "collision_by_why": counts("why", collision),
        # 허용 접촉을 빼고 남은 최악 위반 (m, 양수). 0 이면 남은 것이 없다.
        "residual_violation_m": float(max(0.0, -min(residual))) if residual else 0.0,
        "manipulated_id": manipulated.get("id"),
        "manipulated_state": manipulated.get("state"),
        "authorized_links": sorted(authorized),
    }


def _classify_esdf_rows(hits, values, centres, radii, scene, name, n_spheres, authorized,
                        manipulated, *, pad_m, target_label, allow_unresolved_tier) -> list[dict]:
    """`classify_violations` 의 ESDF 부분 — 위반 행만 모아 한 번에 묻는다."""
    field = scene.esdf
    steps, queries = hits[:, 0], hits[:, 1]
    pts = np.asarray(centres[steps, queries], np.float64).reshape(-1, 3)
    n = len(pts)
    mask = None
    if scene.target_free_mask is not None:
        full = np.asarray(scene.target_free_mask, bool).reshape(-1)
        mask = np.zeros(max(int(queries.max()) + 1, full.shape[0]), bool)
        mask[:full.shape[0]] = full
        mask = mask[queries]
    free_flags = np.zeros(n, bool) if mask is None else mask

    # 어느 계층이 답했나 (T21). 못 말하는 필드면 None.
    tiers: list[Optional[str]] = [None] * n
    tier_of = getattr(field, "answer_tier", None)
    if tier_of is not None:
        try:
            tiers = [str(t) for t in np.asarray(tier_of(pts, target_free=free_flags)).reshape(-1)]
        except Exception:  # noqa: BLE001 — 계층을 못 읽으면 None 으로 (기록만 비는 것)
            tiers = [None] * n

    # 그 행이 쓴 거리 — `_esdf_clearance` 와 같은 합성 (권한 행은 target 없는 계층에 되묻는다).
    d_row = None
    try:
        d_row = np.asarray(field.distance(pts), np.float64).reshape(-1)
        if free_flags.any():
            free = getattr(field, "target_free_distance", None)
            if free is not None:
                d_free = np.asarray(free(pts[free_flags]), np.float64).reshape(-1)
                d_row[free_flags] = d_free
    except Exception:  # noqa: BLE001
        d_row = None

    # 조작 대상까지의 거리 — `_esdf_clearance` 와 같은 두 형식.
    d_obj = None
    if scene.manipulated_spheres is not None and len(scene.manipulated_spheres):
        delta = pts[:, None, :] - np.asarray(scene.manipulated_spheres)[None, :, :]
        d_obj = (np.linalg.norm(delta, axis=2)
                 - np.asarray(scene.manipulated_sphere_radii)[None, :]).min(axis=1)
    elif scene.manipulated_points is not None and len(scene.manipulated_points):
        from scipy.spatial import cKDTree

        d_obj = cKDTree(np.asarray(scene.manipulated_points, np.float64)).query(pts)[0]
    voxel = float(getattr(getattr(field, "grid", None), "voxel_size", 0.0) or 0.0)

    labels: list[Optional[str]] = [None] * n
    if getattr(field, "has_labels", False):
        try:
            idx = np.asarray(field.label(pts)).reshape(-1)
            names = tuple(getattr(field, "label_names", ()) or ())
            labels = [str(names[int(i)]) if 0 <= int(i) < len(names) else None for i in idx]
        except Exception:  # noqa: BLE001
            labels = [None] * n

    per_link = (None if scene.manipulated_link_margin is None
                else np.asarray(scene.manipulated_link_margin, np.float64).reshape(-1))
    centroid = manipulated.get("centroid")
    centroid = None if centroid is None else np.asarray(centroid, np.float64).reshape(3)
    ball = float(manipulated.get("radius") or 0.0)
    # **이번 제약이 권한을 싣고 있어야 한다** — `manipulated_link_margin` 이 곧 그 권한이다
    # (T20: visible/occluded 면 있고 lost 면 없다). 기록에 centroid 가 남아 있어도 권한이 없는
    # 프레임에서는 아무것도 허용하지 않는다.
    has_object = per_link is not None

    out = []
    for i in range(n):
        s, q = int(steps[i]), int(queries[i])
        link = name(q)
        tier = tiers[i]
        row = {"block": "esdf", "step": s, "query": q, "link": link,
               "clearance_m": float(values[s, q]),
               "point_m": [float(v) for v in pts[i]], "tier": tier, "label": labels[i],
               "object_distance_m": None if d_obj is None else float(d_obj[i]),
               "evidence": None}
        if q >= n_spheres:
            row.update({"class": "collision", "why": "attached_point"})
        elif link not in authorized:
            row.update({"class": "collision", "why": "unauthorized_link"})
        elif not has_object:
            row.update({"class": "collision", "why": "no_manipulated"})
        elif tier in _NOT_TARGET_TIERS:
            row.update({"class": "collision",
                        "why": ("answer_excludes_target" if tier == "target_free"
                                else "static_geometry")})
        else:
            if d_obj is not None and d_row is not None and voxel > 0.0:
                on_object = bool(abs(d_row[i] - d_obj[i]) <= voxel)
                row["evidence"] = "distance"
            elif labels[i] is not None:
                on_object = labels[i] == target_label
                row["evidence"] = "label"
            elif centroid is not None:
                margin = float(per_link[q]) if per_link is not None and q < per_link.shape[0] else 0.0
                reach = ball + float(radii[q]) + margin + float(pad_m)
                on_object = bool(np.linalg.norm(pts[i] - centroid) <= reach)
                row["evidence"] = "centroid"
            else:
                on_object = False
            if not on_object:
                row.update({"class": "collision", "why": "nearest_is_not_manipulated"})
            elif tier in _CONTACT_TIERS or allow_unresolved_tier:
                row.update({"class": "allowed", "why": "authorized link on the manipulated object"})
            else:
                row.update({"class": "unresolved", "why": f"unresolved_tier:{tier}"})
        out.append(row)
    return out


def _reasons_note(reasons: Sequence[dict], *, safe: bool, policy: str) -> str:
    """응답 `notes` 의 한 줄 — `verdict: allowed_contact, budget_only → execute`."""
    kinds = ", ".join(str(r.get("kind")) for r in reasons)
    tail = "" if policy == "reasons" else f" (verdict_policy={policy})"
    return f"verdict: {kinds} → {'execute' if safe else 'HOLD'}{tail}"


def _jsonable(value):
    """기록용 — numpy 스칼라·배열을 파이썬 값으로 (T22 `grasp` 블록)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
