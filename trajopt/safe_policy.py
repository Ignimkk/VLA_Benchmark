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
from benchmark.trajopt.linearize import CollisionLinearizer, scene_from_constraint_set
from benchmark.trajopt.refiner import TrajOptChunkRefiner
from benchmark.trajopt.types import ChunkLayout

__all__ = ["SafePolicy"]


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
        기하 인증을 요구할지는 **`to_config.safety.require_certified_geometry` 하나가 정한다.**
        여기에 같은 뜻의 두 번째 스위치를 두었다가 반쪽만 작동하는 것을 실측으로 확인했다 —
        SafePolicy 쪽만 끄면 sqp 가 여전히 상태를 VIOLATED 로 내려서, 위반이 0 mm 인데도
        모든 프레임이 hold 로 갔다. 스위치가 둘이면 언젠가 갈라지고, 갈라진 쪽이 안전 판정이면
        그 결과는 조용하다.
    """

    def __init__(self, policy, *, ag3s, to_config: Optional[TrajOptConfig] = None,
                 attention_fn: Optional[Callable[[dict, dict], dict]] = None,
                 default_phase: str = "approach", recorder=None):
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
        self._pending = {"scene": scene, "attention": attention, "timing": timing}

        t = time.monotonic()
        refined = self.refiner.refine(chunk, {"t_step": seq,
                                              "previous_physical_chunk": self._previous_chunk})
        timing["trajopt"] = (time.monotonic() - t) * 1000.0 - timing.get("ag3s", 0.0)
        refined = np.asarray(refined, np.float64)
        self._previous_chunk = refined

        refined = self._preserve_grippers(chunk, refined)
        verdict = self._verdict(chunk)
        timing["total"] = (time.monotonic() - started) * 1000.0

        if self.recorder is not None:
            self._record(seq, chunk, refined)

        extra = {k: v for k, v in result.items() if k != "actions"}
        return wire.pack_response(refined, verdict, seq=seq, timing_ms=timing, extra=extra)

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
        constraint_set, debug = self.ag3s.process_multi_debug(
            observations,
            phase=scene.get("phase") or self.default_phase,
            active_manipulators=list(scene.get("active_manipulators", ()) or ()),
        )
        self._pending.setdefault("timing", {})["ag3s"] = (time.monotonic() - t) * 1000.0
        self._last_constraint_set = constraint_set
        self._last_debug = debug
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
            return wire.SafetyVerdict(
                ag3s_status=ag3s_status, geometry_certified=False,
                trajopt_status="no_solution", max_violation_m=float("inf"), safe=False,
                notes=notes + ["no scene was available; the policy chunk is unverified"])

        status = getattr(result.status, "value", "unknown")
        violation = float(result.max_violation)
        # `TrajOptStatus.safe` 하나가 근거다. 기하 미인증은 이미 그 안에 들어가 있다 —
        # sqp 가 `safety.require_certified_geometry` 를 보고 상태를 내린다.
        safe = bool(getattr(result.status, "safe", False))
        return wire.SafetyVerdict(
            ag3s_status=ag3s_status, geometry_certified=certified, trajopt_status=status,
            max_violation_m=violation, safe=safe, notes=notes)
