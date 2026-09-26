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

import time
import traceback
from typing import Any, Callable, Optional, Sequence

import numpy as np

from benchmark.trajopt import wire
from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.grasp_latch import CentroidIdentity, GraspLatch, LatchConfig
from benchmark.trajopt.linearize import CollisionLinearizer, scene_from_constraint_set
from benchmark.trajopt.refiner import TrajOptChunkRefiner
from benchmark.trajopt.types import ChunkLayout

__all__ = ["SafePolicy"]


#: 어느 손이 어느 링크로 쥐는가. 로봇마다 다르므로 **주입**이지만, 기본값이 두 곳에서 필요하다 —
#: `SafePolicy` 가 `attach` 를 부를 때와, **그보다 먼저** AG3S 가 attached 슬롯을 예약할 때다.
DEFAULT_GRASP_LINKS: dict[str, tuple] = {
    "left": ("ee_finger_l1", ("ee_finger_l1", "ee_finger_l2")),
    "right": ("ee_finger_r1", ("ee_finger_r1", "ee_finger_r2")),
}


def grasp_parent_links(grasp_links: Optional[dict] = None) -> tuple[str, ...]:
    """AG3S 를 지을 때 `attached_parent_links=` 로 넘겨야 하는 링크들.

    **슬롯은 생성 시점에 예약된다.** `ConstraintBuilder` 의 심볼 그래프와 희소성이 거기서
    고정되므로 `attach()` 가 나중에 늘릴 수 없다. 안 넘기면 파지가 닫히는 **다음 프레임**에
    `_scene_fn` 이 `ValueError` 를 던지고, `refiner` 가 그것을 삼켜 **지각과 최적화가 통째로
    멈춘 채 응답만 계속 나간다** — 실측으로 잡은 결함이다 (2026-09-18, `run_0004` 프레임 10
    이후 14 프레임 동안 AG3S 가 한 번도 안 돌았다).
    """
    return tuple(v[0] for v in (grasp_links or DEFAULT_GRASP_LINKS).values())


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
                 placed_fn: Optional[Callable[[dict, Any], bool]] = None,
                 static_geometry: Optional[Sequence[Any]] = None,
                 shadow: bool = False):
        self._policy = policy
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
        self._latch = GraspLatch(latch)
        #: grounding 은 이름표를 주지 않으므로 무게중심으로 같은 물체인지 본다.
        self._identity = CentroidIdentity()
        #: 어느 손이 어느 링크로 쥐는가. 주입이다 — 로봇마다 다르고 추측할 수 없다.
        self.grasp_links = dict(grasp_links or DEFAULT_GRASP_LINKS)
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
        #: 이번 프레임에 목적지로 넘길 점. 잠금이 목적지를 확정한 **다음** 프레임부터 찬다.
        self._destination_points = None
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
        missing = [p for p in grasp_parent_links(self.grasp_links) if p not in reserved]
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
        self.linearizer = CollisionLinearizer(model, self.layout, self.to_config.horizon.planned)
        self.refiner = TrajOptChunkRefiner(model, self.layout, self._scene_fn, self.to_config)

        self._pending: dict[str, Any] = {}
        self._previous_chunk: Optional[np.ndarray] = None
        self._last_constraint_set = None
        self._last_debug: dict[str, Any] = {}
        self._last_attention_by_camera: dict[str, Any] = {}
        self._last_q_now: Optional[np.ndarray] = None
        self._chunk_index = -1

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
        self._last_constraint_set = None
        # 잠금도 에피소드 상태다. 남기면 다음 과제가 지난 과제의 조작 대상을 물려받는다.
        self._latch.reset()
        self._identity.reset()
        self._destination_points = None
        self._latched_target = None
        self._last_debug = {}
        self._last_attention_by_camera = {}
        self._last_q_now = None
        self._pending = {}

    # ------------------------------------------------------------------------------------
    def infer(self, obs: dict[str, Any], **kwargs) -> dict[str, Any]:
        started = time.monotonic()
        timing: dict[str, float] = {}
        policy_obs, scene = wire.strip_request(obs)
        seq = int(scene.get("seq", 0))

        if scene.get("reset"):
            self.reset()

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

        # `_scene_fn` 이 읽을 것들. refiner 가 콜백을 부를 때 인자로 넘길 수 없는 값이라
        # 여기 둔다 — 콜백 계약(`context -> (scene, q_now, certified)`)을 바꾸지 않으려는 것이다.
        self._pending = {"scene": scene, "attention": attention, "timing": timing,
                         "chunk": chunk}

        t = time.monotonic()
        refined = self.refiner.refine(chunk, {"t_step": seq,
                                              "previous_physical_chunk": self._previous_chunk})
        timing["trajopt"] = (time.monotonic() - t) * 1000.0 - timing.get("ag3s", 0.0)
        refined = np.asarray(refined, np.float64)
        # 연속성 기준은 **로봇이 실제로 실행한 청크**여야 한다 — `_continuity_reference` 가
        # "앞 execution_length 스텝은 이미 실행됐다" 를 전제로 꼬리를 잘라 쓰기 때문이다
        # (`refiner.py:179-195`). shadow 에서 로봇이 실행하는 것은 reference 이므로 그것을
        # 물려준다. refined 를 물려주면 SQP 가 **날아간 적 없는 궤적**에서 이어지는 것으로
        # 계획하고, 그 오차는 어디에도 안 찍힌다.
        self._previous_chunk = chunk if self.shadow else refined

        refined = self._preserve_grippers(chunk, refined)
        verdict = self._verdict(chunk)
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

        extra = {k: v for k, v in result.items() if k != "actions"}
        # **`actions` 는 shadow 에서도 refined 다.** 서버는 자기가 계산한 것을 그대로 말하고,
        # 무엇을 실행할지는 로컬이 고른다 — `SafetyVerdict` 가 판정이지 명령이 아닌 것과 같은
        # 계약이다. shadow 가 아니면 키가 아예 실리지 않아 응답이 예전과 같다.
        return wire.pack_response(refined, verdict, seq=seq, timing_ms=timing,
                                  field=self._field_provenance(),
                                  actions_reference=(chunk if self.shadow else None),
                                  ag3s=ag3s_block,
                                  extra=extra)

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

        t = time.monotonic()
        # `process_multi_debug` 는 `process_multi` 와 **같은 계산**이고 중간 결과를 버리지 않고
        # 돌려줄 뿐이다 (`AG3S.process_multi`는 이 호출의 [0]이다) — 기록기가 없어도 비용이 없다.
        manipulators = list(scene.get("active_manipulators", ()) or ())
        constraint_set, debug = self.ag3s.process_multi_debug(
            observations,
            phase=scene.get("phase") or self.default_phase,
            active_manipulators=manipulators,
            # 목적지는 **지난 프레임**에 잠금이 확정한 것이다. 한 프레임 늦는 것은 의도다 —
            # 이번 프레임의 grounding 결과를 쓰려면 필드를 두 번 지어야 한다.
            destination_points=self._destination_points,
            static_geometry=self.static_geometry or None,
        )
        self._pending.setdefault("timing", {})["ag3s"] = (time.monotonic() - t) * 1000.0
        self._last_constraint_set = constraint_set
        self._last_debug = debug
        self._run_latch(constraint_set, observations, manipulators)
        self._last_attention_by_camera = {
            o.camera_id: np.asarray(o.attention_map)
            for o in observations if getattr(o, "attention_map", None) is not None
        }

        snapshot = scene_from_constraint_set(
            constraint_set, self.linearizer.robot_radii, self.to_config)
        # 촬영 시점 중 **가장 최신** 자세가 TO 가 계획을 시작하는 곳이다. AG3S 도 같은 규칙을
        # 쓰므로 (`process_multi` 의 robot_state 기본값) 둘이 어긋나지 않는다.
        q_now = np.asarray(
            max(observations, key=lambda o: o.timestamp).robot_state, np.float64)
        self._last_q_now = q_now
        certified = constraint_set.status.value == "ok"
        return snapshot, q_now, certified

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
        chunk = self._pending.get("chunk")
        gripper = None
        if chunk is not None and len(chunk):
            column = 6 if str(hand) == "left" else 13
            if chunk.shape[1] > column:
                gripper = float(chunk[0, column])

        placed = False
        if self.placed_fn is not None and self._latch.holding:
            placed = bool(self.placed_fn(self._pending.get("scene", {}), constraint_set))

        metrics = getattr(target, "metrics", {}) or {}
        event = self._latch.update(
            label=label,
            score=float(getattr(target, "confidence", 0.0) or 0.0),
            runner_up=float(metrics.get("runner_up_score", 0.0) or 0.0),
            gripper=gripper,
            placed=placed,
        )

        if event.attach and self._latched_target is not None:
            q = np.asarray(
                max(observations, key=lambda o: o.timestamp).robot_state, np.float64)
            self.ag3s.attach(self._latched_target, robot_state=q, parent_link=parent_link,
                             allowed_contact_links=allowed, label="manipulated")
        elif event.detach:
            self.ag3s.detach()
            # 놓았으니 목적지도 더는 목적지가 아니다. 다음 과제는 새로 잠근다.
            self._destination_points = None

        # 목적지 점 — 잠금이 목적지를 확정했고 지금 target 이 그것이면 그 점구름을 쓴다.
        if event.destination is not None and label == event.destination and target is not None:
            self._destination_points = np.asarray(target.points, np.float64)

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
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[safe_policy] constraint record failed at seq={seq}: {exc}")

    def _occupancy(self):
        """세 상태 점유 배열. `EsdfField` 가 아니라 그것을 만든 builder 안에 있다."""
        builder = getattr(self.ag3s, "_esdf_builder", None)
        return None if builder is None else getattr(builder, "_occupancy", None)

    def _preserve_grippers(self, original: np.ndarray, refined: np.ndarray) -> np.ndarray:
        """그리퍼 열은 정책이 낸 값 그대로. TO 는 그 열의 변수를 갖지도 않는다.

        `ChunkLayout` 이 이미 그리퍼를 결정 변수에서 빼고 `template` 로 되돌려 놓지만, 여기서
        한 번 더 강제한다. 이 두 열이 조용히 바뀌면 손이 엉뚱한 순간에 열리고, 그것은 궤적
        오차와 달리 눈에 띄지 않는다.
        """
        out = np.array(refined, np.float64, copy=True)
        for col in wire.GRIPPER_COLUMNS:
            if col < out.shape[1]:
                out[:, col] = original[:, col]
        return out

    def _verdict(self, policy_chunk: np.ndarray) -> wire.SafetyVerdict:
        result = self.refiner.last_result
        cs = self._last_constraint_set
        ag3s_status = getattr(getattr(cs, "status", None), "value", "no_geometry")
        certified = bool(cs is not None and ag3s_status == "ok")
        notes = list(getattr(result, "notes", ()) or ())

        if result is None:
            # 씬을 못 얻었다. 정책 청크가 그대로 나가고, 그것은 검증된 적이 없다.
            #
            # **왜 못 얻었는지를 응답에 싣는다.** 예전에는 "no scene was available" 한 줄뿐이라,
            # 파지 다음 프레임부터 파이프라인이 통째로 멈춘 것을 14 프레임 동안 아무도 몰랐다
            # (2026-09-18, 원인은 `attached_parent_links` 미예약). 원인 문자열이 여기 있으면
            # 응답만 보고도 알 수 있다.
            why = getattr(self.refiner, "last_failure", None)
            return wire.SafetyVerdict(
                ag3s_status=ag3s_status, geometry_certified=False,
                trajopt_status="no_solution", max_violation_m=float("inf"), safe=False,
                notes=notes + ["no scene was available; the policy chunk is unverified"
                               + (f" — {why}" if why else "")])

        status = getattr(result.status, "value", "unknown")
        violation = float(result.max_violation)
        # `TrajOptStatus.safe` 하나가 근거다. 기하 미인증은 이미 그 안에 들어가 있다 —
        # sqp 가 `safety.require_certified_geometry` 를 보고 상태를 내린다.
        safe = bool(getattr(result.status, "safe", False))
        return wire.SafetyVerdict(
            ag3s_status=ag3s_status, geometry_certified=certified, trajopt_status=status,
            max_violation_m=violation, safe=safe, notes=notes)
