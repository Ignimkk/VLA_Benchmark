"""live 제어 루프에 AG3S + TO 를 붙이는 팩토리.

`pi05_infer.py` 는 bringup 코드다. 여기에 지각·최적화의 배선을 늘어놓으면 두 관심사가 한 파일에
섞이고, 플래그를 끈 상태의 루프가 정말 예전과 같은지 읽어서 확인할 수 없게 된다. 그래서 배선을
전부 이 모듈로 옮기고 `pi05_infer` 쪽에는 호출 네 줄만 남긴다.

`TrajOptChunkRefiner` 가 시뮬레이터를 import 하지 않는 것도 같은 이유다 — MuJoCo 를 아는 코드는
이 파일 하나로 끝난다. refiner 는 `scene_fn` 이라는 콜백만 받고, 그 콜백을 만드는 것이 여기다.

**depth 는 청크마다 한 번만 캡처한다.** 제어 스텝마다(15 Hz) 세 카메라를 렌더하면 osmesa 에서
167 ms x 15 = 2.5 초/초로, 실시간의 두 배 반이다. TO 는 어차피 청크당 1회 도므로 그 시점에만
찍는다. 대신 그 사이 8 스텝 동안 씬이 변하면 제약은 최대 533 ms 묵은 것이 된다 — 정적인
테이블 위 물체에서는 받아들일 만하지만, 움직이는 장애물에서는 아니다. 기록에 캡처 시각을 남기는
이유가 이것이다.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.pipeline import AG3S
from benchmark.ag3s.types import CameraObservation
from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.linearize import CollisionLinearizer, scene_from_constraint_set
from benchmark.trajopt.refiner import TrajOptChunkRefiner
from benchmark.trajopt.types import ChunkLayout

__all__ = ["LivePipeline", "build_live_pipeline"]


class LivePipeline:
    """AG3S + TO 를 한 덩어리로 들고, 청크마다 한 번 불린다.

    Attributes:
        refiner: `pi05_infer` 가 청크에 대고 부르는 것. `refine(chunk, context)` 하나면 된다.
        last_debug: 마지막 프레임의 `process_multi_debug` 두 번째 값. 기록기가 읽는다.
    """

    def __init__(self, *, ag3s: AG3S, refiner: TrajOptChunkRefiner,
                 linearizer: CollisionLinearizer, capture_fn: Callable[[], list[CameraObservation]],
                 state_fn: Callable[[], np.ndarray], to_config: TrajOptConfig,
                 trace=None, recorder=None, phase_fn: Optional[Callable[[int], str]] = None):
        self.ag3s = ag3s
        self.refiner = refiner
        self.linearizer = linearizer
        self._capture = capture_fn
        self._state = state_fn
        self.to_config = to_config
        self.trace = trace
        self.recorder = recorder
        self._phase_fn = phase_fn or (lambda _t: "approach")
        self.last_debug: dict[str, Any] = {}
        self.last_constraint_set = None
        self.last_attention: dict[str, np.ndarray] = {}
        self.chunk_index = -1

    # ----------------------------------------------------------------------------------
    def scene_fn(self, context: Optional[dict]):
        """refiner 계약: `context -> (SceneSnapshot | None, q_now, geometry_certified)`.

        예외를 삼키지 않는다. refiner 가 이미 감싸고 있고, 거기서 "씬을 못 얻었으니 정책 청크를
        그대로 실행한다" 는 판단과 노트를 낸다. 여기서 한 번 더 삼키면 그 노트가 사라져 실패가
        조용해진다.
        """
        context = context or {}
        t_step = int(context.get("t_step", 0))
        with _maybe_span(self.trace, "depth_capture"):
            observations = self._capture()
        q_now = np.asarray(self._state(), np.float64)

        with _maybe_span(self.trace, "ag3s") as span:
            constraint_set, debug = self.ag3s.process_multi_debug(
                observations, phase=self._phase_fn(t_step)
            )
            span["status"] = getattr(constraint_set.status, "value", None)
            span["has_target"] = bool(constraint_set.has_target)
            span["n_candidates"] = len(constraint_set.candidates or ())

        self.last_constraint_set = constraint_set
        self.last_debug = debug
        self.last_attention = {
            o.camera_id: np.asarray(o.attention_map)
            for o in observations if getattr(o, "attention_map", None) is not None
        }

        scene = scene_from_constraint_set(
            constraint_set, self.linearizer.robot_radii, self.to_config
        )
        certified = constraint_set.status.value == "ok"
        return scene, q_now, certified

    # ----------------------------------------------------------------------------------
    def refine(self, chunk: np.ndarray, *, t_step: int,
               previous_chunk: Optional[np.ndarray] = None) -> np.ndarray:
        """정책 청크 하나를 안전하게. 실패해도 예외를 내지 않고 원본을 돌려준다(refiner 계약)."""
        self.chunk_index += 1
        if self.trace is not None:
            self.trace.begin_chunk(self.chunk_index)
        context = {"t_step": t_step, "previous_physical_chunk": previous_chunk}
        # 이 스팬은 **청크 하나의 전체**다. `refine` 안에서 `scene_fn` 이 불리므로
        # `depth_capture` 와 `ag3s` 가 이 안에 중첩돼 있다. 이름을 `to` 로 두면 QP 가 800 ms
        # 걸린 것처럼 읽힌다 — 실제로는 그 대부분이 지각이다. 순수 최적화 시간은
        # `chunk_total - depth_capture - ag3s` 이고, `solve_ms` 로도 따로 실린다.
        with _maybe_span(self.trace, "chunk_total") as span:
            refined = self.refiner.refine(chunk, context)
            result = self.refiner.last_result
            if result is not None:
                span["status"] = getattr(result.status, "value", None)
                span["iterations"] = int(result.iterations)
                span["solve_ms"] = float(getattr(result, "solve_ms", 0.0) or 0.0)
        if self.recorder is not None:
            self._record(t_step, chunk, refined)
        return refined

    def _record(self, t_step: int, reference: np.ndarray, refined: np.ndarray) -> None:
        """기록은 진단이지 제어가 아니다 — 여기서 나는 예외가 루프를 죽이면 안 된다."""
        try:
            cs = self.last_constraint_set
            clearance = None
            centres = radii = None
            result = self.refiner.last_result
            if cs is not None and getattr(cs, "esdf", None) is not None and result is not None:
                all_centres = self.linearizer.sphere_states(
                    result.trajectory, np.asarray(self._state(), np.float64))[0]
                radii = self.linearizer.robot_radii
                # **계획 지평 전체**의 여유거리를 남긴다. 첫 스텝만 재면 TO 의 판정(지평 전체를
                # 본다)과 기록이 어긋나 "TO 는 violated 인데 기록은 위반 0" 이 나온다.
                clearance = (np.asarray(cs.esdf.distance(all_centres.reshape(-1, 3)), np.float64)
                             .reshape(all_centres.shape[:2])
                             - radii[None, :] - float(self.to_config.collision.esdf_margin))
                centres = all_centres[0]  # 시각화가 겹쳐 그릴 자세는 첫 스텝
            self.recorder.record(
                t_step=t_step, chunk_index=self.chunk_index,
                constraint_set=cs, debug=self.last_debug,
                reference_chunk=reference, refined_chunk=refined,
                sphere_centres=centres, sphere_radii=radii,
                sphere_link_names=getattr(self.linearizer.robot_model, "sphere_link_names", None),
                clearance=clearance, to_result=result,
                attention_maps=self.last_attention,
                occupancy=self._occupancy(),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ag3s] constraint record failed at t={t_step}: {exc}")

    def _occupancy(self):
        """세 상태 점유 배열. `EsdfField` 가 아니라 그것을 만든 builder 안에 있다.

        비공개 이름을 읽는 것이 마음에 들지는 않지만, 대안은 `EsdfField` 에 진단용 필드를
        하나 더 달아 TO 가 읽는 자료구조를 진단 때문에 부풀리는 것이다. 기록기 하나가 여기서
        한 줄 참는 편이 낫다.
        """
        builder = getattr(self.ag3s, "_esdf_builder", None)
        return None if builder is None else getattr(builder, "_occupancy", None)

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
        if self.trace is not None:
            self.trace.close()


# ------------------------------------------------------------------------------------ 팩토리


def build_live_pipeline(
    *,
    mj_model,
    mj_data,
    capture_fn: Callable[[], list[CameraObservation]],
    state_fn: Callable[[], np.ndarray],
    ag3s_config: Optional[AG3SConfig] = None,
    to_config: Optional[TrajOptConfig] = None,
    constraint_links: Optional[Sequence[str]] = "arms",
    trace=None,
    recorder=None,
    phase_fn: Optional[Callable[[int], str]] = None,
) -> LivePipeline:
    """live 루프용 AG3S + TO 한 벌.

    Args:
        capture_fn: 부를 때마다 **지금** 세 카메라를 찍어 `CameraObservation` 리스트로. 각
            관측이 자기 `timestamp` 와 자기 `robot_state` 를 들고 있어야 한다 — 손목 카메라는
            팔과 함께 움직이므로 하나의 `q_now` 로 세 대를 변환하면 클라우드가 번진다.
        state_fn: 제약이 기준으로 삼을 현재 `q`.
        constraint_links: `"arms"` 면 양팔 링크와 손끝만 제약에 건다. `None` 이면 전신.
            바퀴·베이스는 결정 변수가 아니라 어떤 해에서도 같은 값을 내고, 고칠 수 없는 위반을
            상수로 깔아 실제 신호를 묻는다. **자기 필터 모델은 어느 쪽이든 전신이다** — 머리
            카메라가 자기 몸을 내려다보므로 바퀴를 빼면 그 점이 로봇에 용접된 유령 장애물이 된다.
    """
    from benchmark.ag3s.experiments.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)

    ag_cfg = ag3s_config or AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": 0.020, "max_distance": 0.4,
                 "exclude_support_surfaces": False},
    })
    to_cfg = to_config or TrajOptConfig.from_dict({
        "collision": {"backend": "esdf", "esdf_margin": 0.05, "use_support_planes": False},
    })

    scene_like = _SceneAdapter(mj_model, mj_data)
    filter_robot = build_robot_model(scene_like)
    links = ARM_LINKS if constraint_links == "arms" else (
        None if constraint_links is None else tuple(constraint_links))
    constraint_robot = build_constraint_robot_model(scene_like, link_filter=links)

    ag3s = AG3S(ag_cfg, robot_model=filter_robot, constraint_robot_model=constraint_robot)
    layout = ChunkLayout.rby1(constraint_robot.joint_names)
    planned = to_cfg.horizon.planned
    linearizer = CollisionLinearizer(constraint_robot, layout, planned)

    pipeline = LivePipeline(
        ag3s=ag3s, refiner=None, linearizer=linearizer, capture_fn=capture_fn,
        state_fn=state_fn, to_config=to_cfg, trace=trace, recorder=recorder, phase_fn=phase_fn,
    )
    pipeline.refiner = TrajOptChunkRefiner(
        constraint_robot, layout, pipeline.scene_fn, to_cfg
    )
    return pipeline


class _SceneAdapter:
    """`build_robot_model` 이 기대하는 최소한의 씬 인터페이스.

    그 함수는 `TransportScene` 을 받도록 쓰였지만 실제로 읽는 것은 `.model`, `.data.qpos`,
    `._qadr` 셋뿐이다. 어댑터 하나가 그 함수를 live 루프에서도 그대로 쓰게 해 준다 — 같은 모델
    구성을 두 벌 유지하는 것보다 낫다. 두 벌이면 언젠가 하나만 고쳐진다.
    """

    def __init__(self, model, data):
        import mujoco

        self.model = model
        self.data = data
        self._qadr = {}
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name:
                self._qadr[name] = int(model.jnt_qposadr[j])


class _NullSpan:
    def __enter__(self):
        return {}

    def __exit__(self, *exc):
        return False


def _maybe_span(trace, name: str):
    return trace.span(name) if trace is not None else _NullSpan()


def observation_time() -> float:
    """캡처 시각. `CameraObservation.timestamp` 와 trace 가 같은 시계를 쓰게 한다."""
    return time.monotonic()
