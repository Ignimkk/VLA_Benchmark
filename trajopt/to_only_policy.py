"""π0.5 → **TO 만** — 지각 없는 서버 경로 (T27, `serve_safe --no-perception`).

사용자 사다리 실험의 E1·E2 가 이것이다 (2026-09-28):

| 판 | 서버 | TO 가 보는 것 |
|---|---|---|
| E1 "TO only" | `--no-perception --no-limits --w-smooth 0 --w-continuity 0` | 추적 항 하나. 충돌 행 0 · joint limit 행 0 |
| E2 | `--no-perception` | 기본 목적함수(추적 · 평활 · 연속성) + joint limit |
| E3 | (`SafePolicy`) 거리장 추가 | 위 + ESDF 충돌 행 (`--links gripper`) |

**`SafePolicy` 와 무엇이 다른가.** AG3S 를 **만들지 않는다** — depth 를 읽지 않고, 점군·ESDF·
grounding·attention(두 번째 체크포인트 사본)·grasp latch 가 없다. 그래서 `--no-collision` 과도 다르다:
그쪽은 행만 끄고 지각·거리장·판정이 전부 돈다 (T16 이 refined ≡ reference 0.00° 인데 HOLD 25/34 로
실패한 것이 그 판정 때문이었다).

**응답은 같은 와이어다** (`wire.pack_response`). 기존 `--safe-remote` 클라이언트가 그대로 받는다:

| 키 | 값 |
|---|---|
| `actions` | refined 청크 (TO 결과, gripper 열은 정책 값 그대로) |
| `actions_reference` | 정책 원본 청크 |
| `to` | `SafePolicy._to_block` 과 같은 키 + `perception: False` · `limits_enforced` |
| `safe` | `True` (TO 가 돈 청크). 서버는 **아무것도 검사하지 않았고**, 막을 근거도 없다 |
| `verdict_reasons` | `[{kind: "no_perception"}]` — `wire.GATE_DEFAULT` 로는 **HOLD** 다 (검사되지 않은 청크). 로컬이 `--safe-gate off` 로 **명시해야** 실행된다 |
| `ag3s_status` · `geometry_certified` | `"no_perception"` · `False` |
| `field` | `unavailable` (이유: 이 서버는 거리장을 만들지 않는다) |

**최적화기는 빈 세계를 본다.** `sqp.TrajectoryOptimizer.solve` 는 씬이 비면 SQP 를 돌리지 않고
reference 를 그대로 내보낸다 (`UNCONSTRAINED`) — 그것은 "TO 만" 이 아니라 "TO 없음" 이다. 그래서
`EmptyWorldField` (어디서나 `+inf` 를 답하는 거리장) 하나를 실은 씬을 준다. 충돌 행은
`collision.enabled=False` 로 비고 (T15), 여유거리 측정은 `+inf` 라 위반 0 · 최악 행 신원 `None`
이다 — 없는 신원을 지어내지 않는다. 결과적으로 SQP 는 **목적함수 + (켜져 있다면) joint limit** 만 본다.
`sqp.py` 는 한 줄도 바꾸지 않았다.

`--record-constraints DIR` 이면 청크마다 reference · refined · 관절별 편차 · limit overshoot · SQP
metrics 를 남긴다 (`ToOnlyRecorder`).
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
import traceback
from typing import Any, Optional

import numpy as np

from benchmark.trajopt import wire
from benchmark.trajopt.config import LimitsConfig, TrajOptConfig
from benchmark.trajopt.limits import build_limits, limit_report
from benchmark.trajopt.linearize import SceneSnapshot
from benchmark.trajopt.refiner import TrajOptChunkRefiner
from benchmark.trajopt.types import ChunkLayout

__all__ = ["ToOnlyPolicy", "ToOnlyRecorder", "EmptyWorldField", "empty_world_scene",
           "NO_PERCEPTION_STATUS", "q_now_from_request"]

#: `ag3s_status` 자리에 싣는 값. AG3S 의 상태값(`ok` · `degraded` …)과 겹치지 않는다.
NO_PERCEPTION_STATUS = "no_perception"


class EmptyWorldField:
    """**어디서나 `+inf` 를 답하는 거리장.** 장애물이 하나도 없는 세계다.

    `SceneSnapshot.is_empty` 가 `esdf is not None` 이면 비지 않은 것으로 보므로, 이것을 실으면
    SQP 가 돈다. 답이 `+inf` 이므로 ESDF 행의 여유거리도 `+inf` 이고, `worst_row` 는 그것을 "행이
    없다" 로 읽어 신원을 `None` 으로 둔다 (`linearize.worst_row`).

    `has_labels=False` · `grid=None` — 라벨·목적지·조작 대상 경로는 씬 쪽 값(`destination_label` ·
    `manipulated_link_margin`)이 `None` 이라 이 필드에 묻지 않는다.
    """

    has_labels = False
    grid = None
    provenance = None

    def distance(self, points) -> np.ndarray:
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        return np.full(pts.shape[0], np.inf)


def empty_world_scene(robot_radii) -> SceneSnapshot:
    """`EmptyWorldField` 하나만 실은 씬. candidate slot 하나 · 평면 하나는 **비활성**이다
    (`tests/trajopt/fixtures.empty_scene` 과 같은 모양 — 배열이 비면 reshape 규약이 갈린다)."""
    radii = np.asarray(robot_radii, np.float64).reshape(-1)
    return SceneSnapshot(
        candidate_pos=np.zeros((1, 3)),
        candidate_radius=np.zeros(1),
        candidate_active=np.zeros(1, bool),
        d_safe=np.zeros((radii.shape[0], 1)),
        plane_normal=np.array([[0.0, 0.0, 1.0]]),
        plane_offset=np.zeros(1),
        plane_active=np.zeros(1, bool),
        robot_radii=radii,
        candidate_ids=np.full(1, -1, np.int64),
        esdf=EmptyWorldField(),
        esdf_margin=0.0,
    )


def q_now_from_request(scene: dict[str, Any]) -> np.ndarray:
    """요청의 카메라별 `robot_state` 중 **가장 최근 촬영 시점**의 관절 (모델 q).

    `SafePolicy._scene_fn` 과 같은 규칙이다 (촬영 시점 중 가장 최신 자세에서 계획을 시작한다).
    depth 는 읽지 않는다. 카메라가 하나도 없으면 **죽는다** — refiner 가 삼키고 reference 를 그대로
    내보내며 노트를 남긴다 (`refiner.refine`).
    """
    best = None
    for cam in scene.get("cameras", ()) or ():
        state = scene.get(f"robot_state/{cam}")
        if state is None:
            continue
        stamp = float(scene.get(f"stamp/{cam}", 0.0))
        if best is None or stamp > best[0]:
            best = (stamp, np.asarray(state, np.float64).reshape(-1))
    if best is None:
        raise ValueError("the request carried no robot_state — q_now is unknown, TO cannot anchor")
    return best[1]


class ToOnlyPolicy:
    """정책 하나를 감싸 **TO 만** 돌린 청크를 돌려준다 (AG3S 없음).

    Args:
        policy: π0.5 (`infer(obs) -> {"actions": [H, D], ...}`).
        robot_model: TO 의 FK · joint limit 의 근거 (`UrdfSphereChain`). 충돌 행이 없으므로 구
            범위(`--links`)는 FK 비용만 바꾼다.
        to_config: 서버가 만든 config. **여기서 두 값을 강제한다** — `collision.enabled=False`
            (충돌 행 0, T15) 와 `safety.require_certified_geometry=False` (인증할 기하가 애초에 없다;
            켜 두면 모든 청크가 `violated` 로 내려간다).
        recorder: `ToOnlyRecorder` 또는 `None`.
    """

    def __init__(self, policy, *, robot_model, to_config: Optional[TrajOptConfig] = None,
                 recorder: Optional["ToOnlyRecorder"] = None):
        self._policy = policy
        base = to_config or TrajOptConfig()
        self.to_config = base.with_overrides({
            "collision": {"enabled": False},
            "safety": {"require_certified_geometry": False},
        })
        self.robot_model = robot_model
        self.layout = ChunkLayout.rby1(robot_model.joint_names)
        self.gripper_columns = wire.gripper_columns(self.layout.nq_opt // 2)
        self.refiner = TrajOptChunkRefiner(robot_model, self.layout, self._scene_fn,
                                           self.to_config)
        #: **로봇 자신의 한계** — `--no-limits` 여도 이것으로 잰다. 기록의 `limit_overshoot` 가
        #: "최적화기가 무엇을 걸었나" 가 아니라 "이 청크가 실제 로봇 한계를 얼마나 넘나" 를 말하게.
        self.robot_limits = build_limits(robot_model, self.layout,
                                         dt=self.to_config.horizon.dt, config=LimitsConfig())
        self.limits_enforced = {
            "position": bool(self.to_config.limits.enforce_position),
            "velocity": bool(self.to_config.limits.enforce_velocity),
            "acceleration": bool(self.to_config.limits.enforce_acceleration),
            "anchor": bool(self.to_config.limits.enforce_velocity
                           or self.to_config.limits.keep_anchor),
        }
        self.recorder = recorder
        self._scene = empty_world_scene(self.refiner.optimizer.linearizer.robot_radii)
        self._pending: dict[str, Any] = {}
        self._previous_chunk: Optional[np.ndarray] = None
        self._previous_plan: Optional[dict[str, Any]] = None
        self._last_feedback: dict[str, Any] = wire.no_exec_feedback("no request yet")
        self._last_continuity: dict[str, Any] = {}
        self._last_q_now: Optional[np.ndarray] = None
        self._chunk_index = -1
        self._record_failures = 0

    # --- BasePolicy 인터페이스 -----------------------------------------------------------
    @property
    def metadata(self) -> dict[str, Any]:
        meta = dict(getattr(self._policy, "metadata", {}) or {})
        meta.update({
            "safe_policy": False,
            "to_only": True,
            "perception": False,
            "planned_horizon": int(self.to_config.horizon.planned),
            "limits_enforced": dict(self.limits_enforced),
        })
        return meta

    def reset(self) -> None:
        for obj in (self._policy, self.refiner):
            reset = getattr(obj, "reset", None)
            if callable(reset):
                reset()
        self._pending = {}
        self._previous_chunk = None
        self._previous_plan = None
        self._last_continuity = {}
        self._last_q_now = None

    def infer(self, obs: dict[str, Any], **kwargs) -> dict[str, Any]:
        started = time.monotonic()
        timing: dict[str, float] = {}
        policy_obs, scene = wire.strip_request(obs)
        seq = int(scene.get("seq", 0))
        if scene.get("reset"):
            self.reset()

        # T39 — 정책 RNG seed (`SafePolicy.infer` 와 같은 자리 · 같은 계약).
        from benchmark.trajopt.policy_seed import apply_request_seed

        seed = apply_request_seed(self._policy, scene, seq=seq)

        t = time.monotonic()
        result = self._policy.infer(policy_obs, **kwargs)
        chunk = np.asarray(result["actions"], np.float64)
        timing["infer"] = (time.monotonic() - t) * 1000.0

        feedback = wire.unpack_exec_feedback(scene)
        self._last_feedback = feedback
        self._pending = {"scene": scene}

        previous, executed_steps = self._continuity_input(feedback)
        context: dict[str, Any] = {"t_step": seq, "previous_physical_chunk": previous}
        if executed_steps is not None:
            context["previous_executed_steps"] = executed_steps
        t = time.monotonic()
        refined = np.asarray(self.refiner.refine(chunk, context), np.float64)
        timing["trajopt"] = (time.monotonic() - t) * 1000.0
        self._last_continuity.update(dict(getattr(self.refiner, "last_continuity", None) or {}))
        self._previous_chunk = refined
        self._previous_plan = {"seq": seq, "refined": refined, "reference": chunk}

        refined = self._preserve_grippers(chunk, refined)
        verdict = self._verdict()
        verdict.notes.extend(self._feedback_notes())
        timing["total"] = (time.monotonic() - started) * 1000.0

        if self.recorder is not None:
            self._record(seq, chunk, refined)

        from benchmark.ag3s.fields.provenance import FieldProvenance

        extra = {k: v for k, v in result.items() if k != "actions"}
        if seed is not None:
            extra[wire.POLICY_SEED] = seed
        return wire.pack_response(
            refined, verdict, seq=seq, timing_ms=timing,
            field=FieldProvenance.unavailable(
                "--no-perception: this server builds no distance field (no depth, no ESDF)"),
            actions_reference=chunk, shadow=False,
            ag3s=self._ag3s_block(),
            to=self._to_block(),
            extra=extra)

    # --- 판정 -----------------------------------------------------------------------------
    def _verdict(self) -> wire.SafetyVerdict:
        result = self.refiner.last_result
        if result is None:
            why = getattr(self.refiner, "last_failure", None) or "unknown"
            return wire.SafetyVerdict(
                ag3s_status=NO_PERCEPTION_STATUS, geometry_certified=False,
                trajopt_status="solver_failed", max_violation_m=0.0, safe=False,
                notes=[f"TO did not run ({why}); the policy chunk passed through unchanged"],
                verdict_reasons=[wire.make_reason(
                    "unverified", f"TO did not run: {why}", stage="to_only")])
        notes = list(result.notes) + [
            "NO PERCEPTION (--no-perception): no depth, no ESDF, no attention, no grasp latch. "
            "The optimizer saw an empty world — nothing was checked for collision."]
        return wire.SafetyVerdict(
            ag3s_status=NO_PERCEPTION_STATUS, geometry_certified=False,
            trajopt_status=result.status.value,
            max_violation_m=float(result.max_violation),
            safe=True, notes=notes,
            verdict_reasons=[wire.make_reason(
                "no_perception",
                "server runs π0.5 → TO only (--no-perception): no collision geometry was built "
                "or checked",
                limits_enforced=dict(self.limits_enforced))])

    def _ag3s_block(self) -> dict[str, Any]:
        """`ag3s` 블록 — `ag3s_status` 가 `ok` 가 아니면 싣는 규약 (`wire` 머리말). 사유 코드는 없다
        (`degradation` 의 코드는 AG3S 안의 분기에 대응하고, 여기는 AG3S 가 없다)."""
        return {"status": NO_PERCEPTION_STATUS, "validity": None,
                "grounding_status": "not_run", "reasons": [],
                "notes": ["AG3S is not running on this server (--no-perception): no depth, no "
                          "ESDF, no attention, no grasp latch"]}

    def _to_block(self) -> Optional[dict[str, Any]]:
        """`SafePolicy._to_block` 과 **같은 키** + `perception` · `limits_enforced` · `returned`."""
        result = self.refiner.last_result
        if result is None:
            return None
        metrics = dict(getattr(result, "metrics", None) or {})
        return {
            "status": result.status.value,
            "iterations": int(result.iterations or 0),
            "solve_ms": round(float(result.solve_time_ms or 0.0), 3),
            "sqp_iterations": int(metrics.get("sqp_iterations", 0) or 0),
            "qp_iterations": int(metrics.get("qp_iterations", 0) or 0),
            "max_iterations": int(metrics.get("max_iterations", 0) or 0),
            "time_budget_ms": float(metrics.get("time_budget_ms", 0.0) or 0.0),
            "time_budget_hit": bool(metrics.get("time_budget_hit", False)),
            "max_iterations_hit": bool(metrics.get("max_iterations_hit", False)),
            "collision_enabled": bool(metrics.get("collision_enabled", False)),
            "reference_deviation": round(float(result.reference_deviation or 0.0), 6),
            "perception": False,
            "limits_enforced": dict(self.limits_enforced),
            "returned": metrics.get("returned"),
        }

    # --- 연속성 (SafePolicy._continuity_input 과 같은 표, T18) --------------------------------
    def _continuity_input(self, feedback: dict[str, Any]
                          ) -> tuple[Optional[np.ndarray], Optional[int]]:
        """연속성 항에 줄 `(이전 청크, 실행된 스텝 수)`.

        **`SafePolicy._continuity_input` 과 같은 규칙**이다 (지침 §6.4 — 계획 캐시와 실행 이력을
        잇는다). 그 메서드는 T26 이 고치는 중인 파일 안의 private 이라 여기서 부르지 않고 같은 표를
        옮겨 적었다 — shadow 갈래는 없다 (이 서버는 shadow 가 아니다). 두 곳이 갈라지면
        `tests/trajopt/test_to_only_policy.py` 의 연속성 절이 이쪽을 박는다.
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
                info["previous_chunk"] = "refined"
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
            info["mode"] = f"execution was not a prefix ({flags}); no alignment exists"
            return None, None
        which = feedback.get("executed_chunk")
        if which not in ("refined", "reference"):
            which = "refined"
        info["mode"] = "feedback"
        info["previous_chunk"] = which if n_exec > 0 else None
        return np.asarray(plan[which], np.float64), n_exec

    def _feedback_notes(self) -> list[str]:
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
        return notes

    # --- refiner 계약 ---------------------------------------------------------------------
    def _scene_fn(self, context: Optional[dict]):
        q_now = q_now_from_request(self._pending.get("scene", {}))
        self._last_q_now = q_now
        # certified=False — 인증할 기하가 없다. `require_certified_geometry` 는 생성자에서 껐다.
        return self._scene, q_now, False

    def _preserve_grippers(self, original: np.ndarray, refined: np.ndarray) -> np.ndarray:
        out = np.array(refined, np.float64, copy=True)
        for col in self.gripper_columns:
            if col < out.shape[1]:
                out[:, col] = original[:, col]
        return out

    # --- 기록 -----------------------------------------------------------------------------
    def chunk_diagnostics(self, reference: np.ndarray, refined: np.ndarray) -> dict[str, Any]:
        """청크 하나의 진단 — **계획한 창**에서 reference 대 refined.

        | 키 | 무엇 |
        |---|---|
        | `planned_steps` | TO 가 다듬은 스텝 수 (그 뒤는 정책 청크 그대로) |
        | `deviation_rad` | `[nq_opt, planned]` refined − reference |
        | `max_abs_deviation_deg` · `per_joint_max_abs_deviation_deg` | 위의 최대 절댓값 (°) |
        | `limit_overshoot_reference` · `limit_overshoot_refined` | **로봇 자신의 한계**(`LimitsConfig()`) 대비 초과 — `--no-limits` 여도 같은 기준 |
        | `q_now` | TO 가 anchor 로 쓴 관절 (모델 q) |
        """
        planned = min(int(self.to_config.horizon.planned), int(np.asarray(reference).shape[0]))
        ref_q = self.layout.chunk_to_trajectory(np.asarray(reference, np.float64))[:, :planned]
        out_q = self.layout.chunk_to_trajectory(np.asarray(refined, np.float64))[:, :planned]
        dev = out_q - ref_q
        per_joint = np.degrees(np.abs(dev).max(axis=1)) if dev.size else np.zeros(dev.shape[0])
        names = [self.layout.joint_names[q] for _, q in self.layout.action_to_q] \
            if self.layout.joint_names else [str(i) for i in range(dev.shape[0])]
        return {
            "planned_steps": planned,
            "deviation_rad": dev,
            "max_abs_deviation_deg": float(per_joint.max()) if per_joint.size else 0.0,
            "per_joint_max_abs_deviation_deg": {n: float(v) for n, v in zip(names, per_joint)},
            "limit_overshoot_reference": limit_report(ref_q, self.robot_limits),
            "limit_overshoot_refined": limit_report(out_q, self.robot_limits),
            "q_now": None if self._last_q_now is None else self._last_q_now.copy(),
        }

    def _record(self, seq: int, reference: np.ndarray, refined: np.ndarray) -> None:
        self._chunk_index += 1
        try:
            result = self.refiner.last_result
            diag = self.chunk_diagnostics(reference, refined)
            metrics = dict(getattr(result, "metrics", None) or {}) if result is not None else {}
            self.recorder.record(
                seq=seq, chunk_index=self._chunk_index,
                reference=reference, refined=refined, diagnostics=diag,
                to_block=self._to_block(),
                sqp={k: metrics.get(k) for k in (
                    "returned", "returned_index", "initial_merit", "initial_eligible",
                    "initial_limit_overshoot", "n_candidates", "n_accepted", "qp_failures",
                    "candidates", "limit_overshoot", "timing_ms", "sqp_iterations",
                    "qp_iterations", "time_budget_hit", "max_iterations_hit")},
                notes=list(getattr(result, "notes", []) or []) if result is not None
                else [f"TO did not run: {self.refiner.last_failure}"],
                exec_feedback=wire.exec_feedback_jsonable(self._last_feedback),
                continuity=dict(self._last_continuity or {}))
        except Exception as exc:  # noqa: BLE001 — 기록 실패가 정책을 죽이면 안 된다
            self._record_failures += 1
            log = logging.getLogger(__name__)
            if self._record_failures == 1:
                log.warning("!!! TO-ONLY RECORDING IS FAILING — THIS RUN WILL HAVE NO RECORD !!!\n"
                            "    seq=%d: %s: %s\n%s", seq, type(exc).__name__, exc,
                            traceback.format_exc())
            else:
                log.warning("[to_only] record failed at seq=%d (%d 번째): %s: %s",
                            seq, self._record_failures, type(exc).__name__, exc)


class ToOnlyRecorder:
    """`--no-perception --record-constraints DIR` 의 기록기.

    `DIR/run_XXXX/` 아래에:

    * `meta.json` — 실행 조건 (서버가 준 것).
    * `chunk_XXXX.npz` — `reference` · `refined` (`[H, D]`), `deviation_rad` (`[nq_opt, planned]`),
      `q_now`.
    * `chunks.jsonl` — 청크마다 한 줄: `seq` · `max_abs_deviation_deg` · 관절별 최대 편차 ·
      limit overshoot (reference · refined, 로봇 한계 기준) · `to` 블록 · SQP 후보 선택(T24) ·
      notes · exec_feedback · continuity.

    `AG3S` 기록기(`ConstraintRecordWriter`)를 쓰지 않는 이유: 그것의 단위는 제약 집합이고 여기에는
    제약 집합이 없다.
    """

    def __init__(self, output_dir, *, meta: Optional[dict[str, Any]] = None):
        root = pathlib.Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        i = 0
        while True:
            run_dir = root / f"run_{i:04d}"
            try:
                run_dir.mkdir(exist_ok=False)
                break
            except FileExistsError:
                i += 1
        self.run_dir = run_dir
        self.meta = {"kind": "to_only", **dict(meta or {})}
        (run_dir / "meta.json").write_text(json.dumps(_jsonable(self.meta), indent=2,
                                                      ensure_ascii=False))
        self.n_records = 0

    def record(self, *, seq: int, chunk_index: int, reference, refined, diagnostics: dict,
               to_block: Optional[dict], sqp: dict, notes: list, exec_feedback: dict,
               continuity: dict) -> None:
        np.savez_compressed(
            self.run_dir / f"chunk_{chunk_index:04d}.npz",
            reference=np.asarray(reference, np.float32),
            refined=np.asarray(refined, np.float32),
            deviation_rad=np.asarray(diagnostics["deviation_rad"], np.float64),
            q_now=np.asarray(diagnostics["q_now"] if diagnostics["q_now"] is not None
                             else np.zeros(0), np.float64))
        line = {
            "seq": int(seq), "chunk_index": int(chunk_index),
            "planned_steps": diagnostics["planned_steps"],
            "max_abs_deviation_deg": diagnostics["max_abs_deviation_deg"],
            "per_joint_max_abs_deviation_deg": diagnostics["per_joint_max_abs_deviation_deg"],
            "limit_overshoot_reference": diagnostics["limit_overshoot_reference"],
            "limit_overshoot_refined": diagnostics["limit_overshoot_refined"],
            "to": to_block, "sqp": sqp, "notes": notes,
            "exec_feedback": exec_feedback, "continuity": continuity,
        }
        with (self.run_dir / "chunks.jsonl").open("a") as fh:
            fh.write(json.dumps(_jsonable(line), ensure_ascii=False) + "\n")
        self.n_records += 1


def _jsonable(value):
    """JSON 으로 옮길 수 있게. 비유한 float 는 `None` (`Infinity` 는 표준 JSON 이 아니다)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _jsonable(value.value)
    return value
