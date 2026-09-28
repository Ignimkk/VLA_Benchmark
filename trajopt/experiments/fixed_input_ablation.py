"""T24 — 기록된 한 청크의 **입력을 고정**하고 TO 의 항을 하나씩 넣어 본다 (지침 §7.2).

closed loop 에서는 관측과 정책 출력이 매 청크 달라지므로 "충돌 항이 손끝을 얼마나 옮겼나" 를 실행끼리
비교해서는 가를 수 없다. 여기서는 서버의 constraint record 한 chunk 에서 **같은 입력**을 복원하고,
그 위에서 변형만 바꿔 다시 푼다.

    MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python \\
        -m benchmark.trajopt.experiments.fixed_input_ablation \\
        --constraints outputs/live_test/20260928_t14/server_constraints/run_0000 \\
        --frames outputs/live_test/20260928_t14/execute_ep1807/frames.jsonl \\
        --server-log outputs/live_test/20260928_t14/serve_safe.log \\
        --out outputs/t24/t14_fixed_input_ablation.json

## 무엇을 복원하나

| 입력 | 어디서 |
|---|---|
| reference | `chunk_k.npz` 의 `reference_chunk` (정책 원본) |
| **robot model** | 그 실행의 sphere option 으로 **다시 짓는다** (`--links`, `--sphere-*`; 없으면 `--server-log` 의 시작 로그에서 읽는다). 기록의 `sphere_centres` 는 결과이지 입력이 아니다 |
| `q_now` | 로컬 `frames.jsonl` 의 planning 행 `seq = k+1` 의 `qpos` → `DEFAULT_RBY1_JOINTS` (T14.audit §1) |
| ESDF | `esdf_coarse_*` · `esdf_*`(fine) · `esdf_free_*` → `layer_from_arrays` → `CuroboEsdfField` |
| 접촉 정책 | `ClearancePolicy`(AG3S 기본 contact 설정) × `(phase, manipulators)` (로컬 manifest) → `manipulated_link_margin` · `target_field_exclude`; 조작 대상 점 = `target_points` |
| 목적함수·예산 | 시작 로그의 `TO objective` · `esdf_margin` · plan window |

## 재현하지 **못하는** 것 (기록에 없다)

* **라벨 층.** 필드의 라벨 격자는 기록되지 않는다 → 최악 행의 `obstacle` 이름은 `None` 이고, 목적지
  마진(쥔 물체 점에만 붙는다)은 재현하지 못한다.
* **쥔 물체의 질의점** (`attached`). 기록에 없다 → `esdf_stats.n_attached_points > 0` 인 chunk 는
  변형을 돌리지 않고 `skipped_reason` 을 적는다.
* **QP 의 청크 간 warm start** (OSQP 인수분해·dual). 청크마다 새 solver 로 푼다 — `replay` 가 그 차이를
  잰다.
* **벽시계 예산.** `time_budget_ms` 는 이 머신의 부하에 따라 반복 수를 바꾼다. 실제 반복 수를 함께
  싣는다. `--budget-clock off` 면 예산은 `max_iterations` 만 묶는다.

## 먼저 하는 것 — 재구성 충실도

변형을 돌리기 전에 chunk 마다 둘을 잰다.

1. **기록 clearance 재현** — 기록의 `refined_chunk` 를 재구성 필드와 재구성 모델로 다시 재서
   (`CollisionLinearizer.esdf_clearance`, 기록기와 같은 정의: 본 계층 `distance − r − esdf_margin`)
   기록의 `clearance (8, S)` 와 비교한다.
2. **TO 가 본 최악 행 재현** — 같은 궤적에 접촉 정책까지 실은 씬으로 `worst_row` 를 돌려 기록의
   `summary.to.metrics.clearance_m` · `max_violation_pair` 와 비교한다.

둘 중 하나라도 `--fidelity-tol-mm`(기본 1 mm)를 넘으면 **도구가 무효**다. 기본(`--strict`)은 그때
어떤 변형도 돌리지 않고 결과 JSON 에 `"valid": false` 와 이유를 적고 종료 코드 2 로 끝난다.

## 변형

| 이름 | 충돌 | w_smooth | w_continuity | 예산 |
|---|---|---|---|---|
| `replay` | 기록대로 | 기록대로 | 기록대로 | 반복 수 = 기록, 시계 끔 — **기록의 refined 를 재현하는가** |
| `track_only` | OFF | 0 | 0 | 기록 예산 |
| `smooth_only` | OFF | `--smooth-weight` (기본 `CostConfig` 기본값) | 0 | 기록 예산 |
| `continuity_only` | OFF | 0 | `--continuity-weight` (이전 청크 = 기록의 직전 chunk) | 기록 예산 |
| `collision` | ON (기록의 접촉 정책) | 0 | 0 | 기록 예산 |
| `collision_budget_x2` · `_x4` | ON | 0 | 0 | `time_budget_ms`·`max_iterations` ×2, ×4 |

`w_track` 은 언제나 기록값(기본 1)이다 — 추적까지 없애면 원본 복원 검사가 아니다 (지침 §7.2).

## 출력 (JSON 하나, verifier 가 figure 를 만든다)

chunk 별·변형별: 실행 창의 **스텝별** 관절 최대 편차(°)·관절별 최대 편차·손끝(finger sphere 평균) 위치
변위(mm)·palm 방향 변위(°)·손끝 경로(m), 활성 행 목록(link · 계층 · target 제외 여부 · 마진 · clearance),
reference 의 한계 초과량(anchor 포함), SQP/QP 반복·예산 도달·후보 `(merit, 위반, 수락)`·반환 판정
(`returned`, 그리고 **수정 전 규칙**이었다면 무엇이 나갔나 = `legacy`)·최종 위반.
"전체 원소 중앙값 하나" 로 축약하지 않는다 (지침 §11) — 요약 통계는 싣지 않는다.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import pathlib
import re
import sys
import time
from typing import Any, Optional

import numpy as np

TOOL = "benchmark.trajopt.experiments.fixed_input_ablation"

#: 손끝 = 그 손의 finger sphere 전부의 평균. 방향 = palm link 의 회전.
FINGER_LINKS = {"left": ("ee_finger_l1", "ee_finger_l2"), "right": ("ee_finger_r1", "ee_finger_r2")}
PALM_LINK = {"left": "ee_left", "right": "ee_right"}

VARIANTS = ("replay", "track_only", "smooth_only", "continuity_only", "collision",
            "collision_budget_x2", "collision_budget_x4")

#: `--budget-clock off` 일 때의 시간 예산. 사실상 무한 — `max_iterations` 만 묶는다.
NO_CLOCK_MS = 1e9


# ======================================================================================
# 설정 — 시작 로그에서 읽고, CLI 가 덮는다
# ======================================================================================

def parse_server_log(path: Optional[str]) -> dict[str, Any]:
    """`serve_safe.log` 의 시작 줄에서 그 실행의 제약 모델·목적함수·정책을 읽는다.

    **찾지 못한 키는 넣지 않는다.** 기본값을 여기서 지어내면 로그가 말하지 않은 설정이 로그에서
    온 것처럼 기록된다. 빠진 것은 `resolve_settings` 가 CLI 나 코드 기본값으로 채우고 그 출처를 적는다.
    """
    if not path:
        return {}
    text = pathlib.Path(path).read_text(errors="replace")
    out: dict[str, Any] = {}
    m = re.search(r"sphere_spacing=([0-9.]+), max_spheres_per_capsule=(\d+), "
                  r"capsule_radius_scale=([0-9.]+), max_sphere_radius=([0-9.]+|none)", text)
    if m:
        opts: dict[str, Any] = {"sphere_spacing": float(m.group(1)),
                                "max_spheres_per_capsule": int(m.group(2)),
                                "capsule_radius_scale": float(m.group(3))}
        if m.group(4) != "none":
            opts["max_sphere_radius"] = float(m.group(4))
        out["sphere_options"] = opts
    m = re.search(r"AG3S: self-filter \d+ spheres, constraints (\d+) spheres \(([^)]+)\)", text)
    if m:
        out["n_constraint_spheres"] = int(m.group(1))
        out["links"] = m.group(2).strip()
    m = re.search(r"TO esdf_margin: ([0-9.]+) mm", text)
    if m:
        out["esdf_margin"] = float(m.group(1)) / 1000.0
    m = re.search(r"TO objective: w_track=(\S+) w_smooth=(\S+) w_continuity=(\S+) w_slack=(\S+)",
                  text)
    if m:
        out["cost"] = {"w_track": float(m.group(1)), "w_smooth": float(m.group(2)),
                       "w_continuity": float(m.group(3)), "w_slack": float(m.group(4))}
    m = re.search(r"--target-field-policy ([a-z-]+)\)", text)
    out["target_field_policy"] = m.group(1).replace("-", "_") if m else None
    if "COLLISION CONSTRAINTS ARE OFF" in text:
        out["collision_enabled"] = False
    elif "TO objective:" in text:
        out["collision_enabled"] = True
    if "shadow=True" in text:
        out["shadow"] = True
    elif "shadow=False" in text:
        out["shadow"] = False
    m = re.search(r"TO plan window: (\d+) of (\d+) chunk steps \(execution_length=(\d+)\)", text)
    if m:
        out["plan_window"] = {"planned": int(m.group(1)), "horizon": int(m.group(2)),
                              "execution_length": int(m.group(3))}
    m = re.search(r"row budget: .*?rows_per_step (\d+)", text)
    if m:
        out["rows_per_step"] = int(m.group(1))
    return out


@dataclasses.dataclass
class RunSettings:
    """그 실행의 설정. `sources` 가 각 값이 어디서 왔는지 적는다 (`log` · `cli` · `default`)."""

    links: str
    sphere_options: dict[str, Any]
    esdf_margin: float
    cost: dict[str, float]
    collision_enabled: bool
    target_field_policy: str
    shadow: bool
    plan_horizon: Any
    rows_per_step: Optional[int]
    require_certified: bool
    phase: str
    manipulators: list[str]
    model_xml: str
    hand: str
    sources: dict[str, str]
    n_constraint_spheres: Optional[int] = None

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def resolve_settings(args, log: dict[str, Any], manifest: dict[str, Any]) -> RunSettings:
    from benchmark.trajopt.config import PLAN_EXECUTION_WINDOW, CostConfig

    sources: dict[str, str] = {}

    def pick(name, cli_value, log_value, default, *, required=False):
        if cli_value is not None:
            sources[name] = "cli"
            return cli_value
        if log_value is not None:
            sources[name] = "log"
            return log_value
        if required:
            raise SystemExit(f"설정 {name!r} 를 알 수 없습니다 — --server-log 에서 못 찾았고 CLI 로도 "
                             "안 줬습니다. 추측하지 않습니다")
        sources[name] = "default"
        return default

    links = pick("links", args.links, log.get("links"), None, required=True)
    if "minus" in str(links):
        raise SystemExit(f"--exclude-links 로 돈 실행입니다 ({links!r}) — 이 도구는 그 모델을 다시 "
                         "짓지 않습니다. --links 와 모델을 확인하세요")
    cli_opts = {k: v for k, v in {
        "sphere_spacing": args.sphere_spacing,
        "max_spheres_per_capsule": args.max_spheres_per_capsule,
        "capsule_radius_scale": args.capsule_radius_scale,
        "max_sphere_radius": args.max_sphere_radius}.items() if v is not None}
    if cli_opts:
        sphere_options = cli_opts
        sources["sphere_options"] = "cli"
    elif "sphere_options" in log:
        sphere_options = log["sphere_options"]
        sources["sphere_options"] = "log"
    else:
        sphere_options = {}
        sources["sphere_options"] = "default"
    esdf_margin = pick("esdf_margin", args.esdf_margin, log.get("esdf_margin"), None,
                       required=True)
    defaults = CostConfig()
    cost = dict(log.get("cost") or {})
    sources["cost"] = "log" if cost else "default"
    cost = {k: float(cost.get(k, getattr(defaults, k)))
            for k in ("w_track", "w_smooth", "w_continuity", "w_slack")}
    collision = pick("collision_enabled", None, log.get("collision_enabled"), True)
    policy = pick("target_field_policy",
                  args.target_field_policy.replace("-", "_") if args.target_field_policy else None,
                  log.get("target_field_policy"), "relax")
    shadow = pick("shadow", None, log.get("shadow"), False)
    window = log.get("plan_window")
    if window:
        plan_horizon = (PLAN_EXECUTION_WINDOW if window["planned"] == window["execution_length"]
                        else int(window["planned"]))
        sources["plan_horizon"] = "log"
    else:
        plan_horizon = PLAN_EXECUTION_WINDOW
        sources["plan_horizon"] = "default"
    policy_block = manifest.get("policy", {}) or {}
    phase = pick("phase", args.phase, policy_block.get("safe_phase"), None, required=True)
    manipulators = pick("manipulators", args.manipulators, policy_block.get("safe_manipulators"),
                        None, required=True)
    model_xml = pick("model_xml", args.model_xml, (manifest.get("scene") or {}).get("model_xml"),
                     None, required=True)
    hand = args.hand or (list(manipulators)[0] if manipulators else "left")
    sources["hand"] = "cli" if args.hand else "manipulators"
    sources["require_certified"] = "cli" if args.allow_uncertified else "default"
    return RunSettings(
        links=str(links), sphere_options=dict(sphere_options), esdf_margin=float(esdf_margin),
        cost=cost, collision_enabled=bool(collision), target_field_policy=str(policy),
        shadow=bool(shadow), plan_horizon=plan_horizon,
        rows_per_step=args.rows_per_step or log.get("rows_per_step"),
        require_certified=not args.allow_uncertified, phase=str(phase),
        manipulators=[str(m) for m in manipulators], model_xml=str(model_xml), hand=str(hand),
        sources=sources, n_constraint_spheres=log.get("n_constraint_spheres"))


# ======================================================================================
# 입력 복원
# ======================================================================================

def build_robot(settings: RunSettings):
    """`serve_safe.build_ag3s` 의 **제약 모델** 경로를 그대로 밟는다 (자기 필터 모델은 필요 없다).

    `serve_safe` 를 import 하지 않고 같은 두 함수를 부른다 — 그 모듈은 서버 진입점이라 이 도구가
    그 모듈의 수정 중인 상태에 묶이지 않게 한다. 링크 집합은 `constraint_link_filter` 와 같은 표다.
    """
    import mujoco

    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, GRIPPER_LINKS, build_constraint_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene
    from benchmark.ag3s.runtime.asset_path import resolve_asset

    link_filter = {"all": None, "arms": ARM_LINKS, "gripper": GRIPPER_LINKS}.get(settings.links)
    if settings.links not in ("all", "arms", "gripper"):
        raise SystemExit(f"알 수 없는 --links {settings.links!r}")
    path = resolve_asset(settings.model_xml, what="transport model")
    model = mujoco.MjModel.from_xml_path(str(path))
    scene = TransportScene.attach(model, mujoco.MjData(model))
    robot = build_constraint_robot_model(scene, link_filter=link_filter,
                                         sphere_options=dict(settings.sphere_options))
    qadr = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): int(model.jnt_qposadr[i])
            for i in range(model.njnt)}
    return robot, qadr, str(path)


def q_now_from_qpos(qpos, qadr: dict[str, int]) -> np.ndarray:
    """MuJoCo `qpos` → 서버의 `robot_state` (20, `DEFAULT_RBY1_JOINTS` 순) — 클라이언트와 같은 사상."""
    from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS

    qpos = np.asarray(qpos, np.float64)
    return np.asarray([qpos[qadr[j]] for j in DEFAULT_RBY1_JOINTS], np.float64)


def build_field(z, outside_distance: float):
    """기록의 float16 격자 → `CuroboEsdfField` (coarse → fine, 그리고 target 없는 계층)."""
    from benchmark.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays

    def layer(prefix):
        return layer_from_arrays(z[f"{prefix}_distance"], z[f"{prefix}_origin"],
                                 float(z[f"{prefix}_voxel_size"]))

    layers = []
    if "esdf_coarse_distance" in z.files:
        layers.append(layer("esdf_coarse"))
    if "esdf_distance" not in z.files:
        raise ValueError("기록에 거리장이 없습니다 (--record-constraints-esdf full 로 돈 실행이 아님)")
    layers.append(layer("esdf"))
    free = (layer("esdf_free"),) if "esdf_free_distance" in z.files else ()
    return CuroboEsdfField(tuple(layers), outside_distance=outside_distance,
                           target_free_layers=free)


def contact_policy(settings: RunSettings, link_names, *, has_target: bool, has_free: bool,
                   attached: bool):
    """`to_adapter.build_constraint_set` 의 두 배열을 **같은 함수로** 다시 만든다.

    `manipulated_link_margin` = `ClearancePolicy.margin_matrix(..., [TARGET], target_grounded=True)`,
    `target_field_exclude` = 권한 link 마스크 (`exclude_authorized`) 또는 전부 (`exclude_all`).
    조건(정책 ≠ relax · 쥔 것 없음 · target 있음 · 계층 있음)도 그 함수와 같다.
    """
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.constraints.clearance import ClearancePolicy
    from benchmark.ag3s.types import ContactPolicyContext, SourceType

    config = AG3SConfig.from_dict({"collision_backend": "esdf"})
    policy = ClearancePolicy.from_config(config.contact, config.geometry, config.support_surface)
    ctx = ContactPolicyContext.make(settings.phase, settings.manipulators)
    names = [str(n) for n in link_names]
    margin = None
    if has_target and not attached:
        margin = policy.margin_matrix(names, [SourceType.TARGET], context=ctx,
                                      target_grounded=True)[:, 0]
    authorized = sorted(policy.authorized_links(ctx))
    mask = None
    if (settings.target_field_policy != "relax" and not attached and has_target and has_free):
        if settings.target_field_policy == "exclude_all":
            mask = np.ones(len(names), bool)
        else:
            mask = np.asarray([n in authorized for n in names], bool)
        mask = mask if mask.any() else None
    return margin, mask, authorized, float(config.esdf.max_distance), config.esdf.unknown_policy


def make_scene(field, radii, esdf_margin, *, target_points=None, margin=None, mask=None):
    """`scene_from_constraint_set` 의 field-only 경로 (`spec is None`) 와 같은 `SceneSnapshot`."""
    from benchmark.trajopt.linearize import SceneSnapshot

    radii = np.asarray(radii, np.float64)
    return SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((radii.size, 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=radii, candidate_ids=np.zeros(0, np.int64),
        esdf=field, esdf_margin=float(esdf_margin),
        manipulated_points=(None if target_points is None or margin is None
                            else np.asarray(target_points, np.float64)),
        manipulated_link_margin=margin, target_free_mask=mask)


def to_config(settings: RunSettings, *, collision: bool, cost: dict[str, float],
              sqp: dict[str, Any]):
    """`serve_safe.main` 이 짓는 `TrajOptConfig` 와 같은 dict — 변형이 바꾸는 키만 다르다."""
    from benchmark.trajopt.config import TrajOptConfig

    data: dict[str, Any] = {
        "collision": {"backend": "esdf", "esdf_margin": settings.esdf_margin,
                      "use_support_planes": False, "enabled": bool(collision)},
        "cost": dict(cost),
        "safety": {"require_certified_geometry": settings.require_certified},
        "horizon": {"plan_horizon": settings.plan_horizon},
        "sqp": dict(sqp),
    }
    if settings.rows_per_step:
        data["reduction"] = {"rows_per_step": int(settings.rows_per_step)}
    return TrajOptConfig.from_dict(data)


# ======================================================================================
# 측정
# ======================================================================================

class Instrumented:
    """`TrajOptChunkRefiner` 하나 + **수정 전 규칙**이 반환했을 궤적을 되짚는 장치.

    `sqp.solve` 는 궤적마다 `linearizer.sphere_states` 를 정확히 한 번 부른다: reference
    (`full_violation`), 초기 iterate, 그리고 QP 가 풀린 후보마다 하나. 그 호출을 가로채 궤적을
    모으면 후보 `i` 의 궤적은 `captured[2 + i]` 다. 수정 전 규칙(`best_merit = inf` 에서 시작하는
    엄격한 `<`)은 **후보 merit 의 첫 최소**를 반환하므로, 새 코드를 한 번 돌리는 것으로 두 규칙의
    반환을 모두 얻는다 — 수락/신뢰영역 동역학은 두 규칙에서 같다. 호출 수가 `2 + n_candidates` 와
    다르면 되짚기를 포기하고 `None` 을 싣는다.
    """

    def __init__(self, robot, layout, config, scene_holder):
        from benchmark.trajopt.refiner import TrajOptChunkRefiner

        self.holder = scene_holder
        self.refiner = TrajOptChunkRefiner(robot, layout, lambda _ctx: self.holder["value"], config)
        lin = self.refiner.optimizer.linearizer
        original = lin.sphere_states
        self.captured: list[np.ndarray] = []

        def capture(trajectory, q_now):
            self.captured.append(np.array(trajectory, np.float64, copy=True))
            return original(trajectory, q_now)

        lin.sphere_states = capture  # 인스턴스 속성 — `full_violation` 의 `self.sphere_states` 도 탄다

    def run(self, chunk: np.ndarray, context: dict[str, Any]):
        self.refiner.reset()  # 청크마다 새로: warm start·연속성·QP 인수분해를 넘기지 않는다
        self.captured = []
        self.refiner.refine(chunk, context)
        result = self.refiner.last_result
        if result is None:
            raise RuntimeError(f"refiner 가 결과를 내지 않았습니다: {self.refiner.last_failure}")
        return result, list(self.captured)


def legacy_selection(result, captured) -> dict[str, Any]:
    """수정 전 규칙이었다면 무엇이 나갔나 — 후보 기록에서 계산한다."""
    m = result.metrics
    cands = m.get("candidates") or []
    if not cands:
        idx = -1
    else:
        merits = np.asarray([c["merit"] for c in cands], np.float64)
        idx = int(np.argmin(merits))  # 첫 최소 = `merit < best_merit` 의 엄격한 비교
    label = "initial" if idx < 0 else ("accepted" if cands[idx]["accepted"] else "best_unaccepted")
    traj = None
    if len(captured) == 2 + len(cands):
        traj = captured[1] if idx < 0 else captured[2 + idx]
    return {"returned": label, "returned_index": idx, "trajectory": traj,
            "capture_ok": traj is not None}


class Meter:
    """궤적 → 실행 창 지표. FK 는 한 벌의 linearizer 로 한다 (변형의 것과 섞지 않는다)."""

    def __init__(self, robot, layout, horizon: int, hand: str, limits):
        from benchmark.trajopt.linearize import CollisionLinearizer

        self.robot = robot
        self.layout = layout
        self.limits = limits
        self.lin = CollisionLinearizer(robot, layout, horizon)
        names = [str(n) for n in robot.sphere_link_names]
        fingers = FINGER_LINKS[hand]
        self.finger_idx = np.asarray([i for i, n in enumerate(names) if n in fingers], np.int64)
        if not self.finger_idx.size:
            raise SystemExit(f"제약 모델에 {fingers} 구가 없습니다 — --links 를 확인하세요")
        self.palm = PALM_LINK[hand]
        self.joint_names = [str(robot.joint_names[i]) for i in layout.q_indices]

    def tip_and_rot(self, traj: np.ndarray, q_now: np.ndarray):
        centres = self.lin.sphere_states(traj, q_now)[0]
        tip = centres[:, self.finger_idx].mean(axis=1)
        full = self.layout.full_q(traj, q_now)
        rot = np.stack([np.asarray(self.robot.link_pose(full[:, k], self.palm))[:3, :3]
                        for k in range(traj.shape[1])])
        return tip, rot

    def compare(self, traj, reference, q_now, ref_tip=None, ref_rot=None) -> dict[str, Any]:
        if ref_tip is None:
            ref_tip, ref_rot = self.tip_and_rot(reference, q_now)
        tip, rot = self.tip_and_rot(traj, q_now)
        dev = np.degrees(np.abs(traj - reference))
        disp = np.linalg.norm(tip - ref_tip, axis=1) * 1000.0
        cosang = (np.einsum("kij,kij->k", ref_rot, rot) - 1.0) / 2.0
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        return {
            "steps": int(traj.shape[1]),
            "joint_dev_max_deg": float(dev.max()),
            "joint_dev_per_step_deg": dev.max(axis=0).tolist(),
            "joint_dev_per_joint_deg": {n: float(v) for n, v in zip(self.joint_names,
                                                                    dev.max(axis=1))},
            "fingertip_disp_mm_per_step": disp.tolist(),
            "fingertip_disp_mm_max": float(disp.max()),
            "palm_rot_deg_per_step": ang.tolist(),
            "palm_rot_deg_max": float(ang.max()),
            "fingertip_path_m": tip.tolist(),
        }

    def overshoot(self, traj, q_now) -> dict[str, float]:
        from benchmark.trajopt.limits import limit_report

        report = {k: float(v) for k, v in limit_report(traj, self.limits).items()}
        q0 = np.asarray(q_now, np.float64)[self.layout.q_indices]
        step = np.asarray(self.limits.max_step, np.float64)
        report["anchor"] = (float(max(0.0, np.max(np.abs(traj[:, 0] - q0) - step)))
                            if np.all(np.isfinite(step)) else 0.0)
        return report


def row_details(meter: Meter, traj, q_now, scene, field) -> dict[str, np.ndarray]:
    """(H, S) 배열들: TO 가 본 clearance 와 그 값을 만든 계층·마스크·마진."""
    centres = meter.lin.sphere_states(traj, q_now)[0]
    flat = centres.reshape(-1, 3)
    shape = centres.shape[:2]
    clearance = meter.lin.esdf_clearance(traj, q_now, scene)[..., 0]
    d, winner = field._evaluate(flat)[:2]
    d = np.asarray(d, np.float64).reshape(shape)
    winner = np.asarray(winner).reshape(shape)
    d_free = (np.asarray(field.target_free_distance(flat), np.float64).reshape(shape)
              if getattr(field, "has_target_free", False) else d)
    mask = (np.zeros(shape[1], bool) if scene.target_free_mask is None
            else np.asarray(scene.target_free_mask, bool))
    d_eff = np.where(mask[None, :], d_free, d)
    margin = d_eff - meter.lin.robot_radii[None, :] - clearance
    return {"clearance": clearance, "d": d, "d_free": d_free, "winner": winner, "mask": mask,
            "margin": margin}


def active_rows(meter: Meter, reference, q_now, scene, field, config, link_names) -> dict[str, Any]:
    """**QP 가 첫 반복에서 받은 행** (`linearize` 가 고른 것) 과 reference 의 위반 행."""
    rows = meter.lin.linearize(reference, q_now, scene, config)
    det = row_details(meter, reference, q_now, scene, field)
    tier_names = ["coarse" if i == 0 and len(field.layers) > 1 else "fine"
                  for i in range(len(field.layers))]
    selected = []
    used = np.asarray(rows.used)
    for k, j in zip(*np.nonzero(used)):
        q = int(rows.sphere[k, j])
        selected.append({
            "step": int(k), "query": q, "link": str(link_names[q]),
            "clearance_mm": float(rows.value[k, j] * 1000.0),
            "tier": tier_names[int(det["winner"][k, q])],
            "target_free": bool(det["mask"][q]),
            "target_free_gain_mm": float((det["d_free"][k, q] - det["d"][k, q]) * 1000.0),
            "margin_mm": float(det["margin"][k, q] * 1000.0),
            "label": None,  # 라벨 층은 기록되지 않는다
        })
    selected.sort(key=lambda r: r["clearance_mm"])
    neg = det["clearance"] < 0.0

    def by(key, items):
        out: dict[str, int] = {}
        for it in items:
            out[str(it[key])] = out.get(str(it[key]), 0) + 1
        return out

    return {
        "definition": "QP 가 첫 SQP 반복에서 받는 행 = `linearize(reference)` 의 used 행 "
                      "(activation band 안, rows_per_step 예산으로 고른 것). tier = 본 계층 중 "
                      "거리를 낸 쪽, target_free = 그 질의점이 target 없는 계층에 묻는가, "
                      "margin_mm = 그 행이 요구한 여유 (esdf_margin 또는 조작 대상 완화)",
        "activation_band_m": float(config.reduction.activation_band),
        "rows_per_step": int(config.reduction.rows_per_step),
        "budget_bound_steps": list(rows.budget_bound_steps),
        "n_selected": len(selected),
        "selected": selected,
        "selected_by_link": by("link", selected),
        "selected_by_tier": by("tier", selected),
        "n_negative_at_reference": int(neg.sum()),
        "negative_by_link": {str(n): int(c) for n, c in zip(
            *np.unique(np.asarray(link_names)[np.nonzero(neg)[1]], return_counts=True))},
        "min_clearance_per_step_mm": (det["clearance"].min(axis=1) * 1000.0).tolist(),
    }


# ======================================================================================
# 한 chunk
# ======================================================================================

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None if np.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


def _parse_pair(value) -> Optional[dict]:
    """기록기는 metrics 의 dict 를 `str()` 로 적는다 (`constraint_record._scalar`)."""
    if value is None or isinstance(value, dict):
        return value
    import ast

    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None


def fidelity(meter: Meter, z, summary, refined_traj, q_now, field, scene_to, radii,
             esdf_margin) -> dict[str, Any]:
    """재구성 충실도 두 가지. 모듈 docstring 의 "먼저 하는 것" 참고."""
    out: dict[str, Any] = {}
    planned = refined_traj.shape[1]
    plain = make_scene(field, radii, esdf_margin)
    if "clearance" in z.files:
        recorded = np.asarray(z["clearance"], np.float64)
        mine = meter.lin.esdf_clearance(refined_traj, q_now, plain)[..., 0]
        h = min(recorded.shape[0], planned)
        diff = np.abs(mine[:h] - recorded[:h])
        out["record_clearance"] = {
            "definition": "기록기와 같은 정의: 본 계층 distance − r − esdf_margin (target 제외·"
                          "조작 대상 마진 없음), 기록의 refined 궤적 위",
            "shape": list(recorded.shape),
            "max_abs_diff_mm": float(diff.max() * 1000.0),
            "argmax_step_query": [int(v) for v in np.unravel_index(int(diff.argmax()),
                                                                    diff.shape)],
        }
    if "sphere_centres" in z.files:
        centres = meter.lin.sphere_states(refined_traj, q_now)[0][0]
        out["sphere_centres_step0_max_diff_mm"] = float(
            np.abs(centres - np.asarray(z["sphere_centres"], np.float64)).max() * 1000.0)
    to = summary.get("to") or {}
    rec_metrics = to.get("metrics") or {}
    if "clearance_m" in rec_metrics:
        mine_c, mine_pair = meter.lin.worst_row(refined_traj, q_now, scene_to)
        rec_pair = _parse_pair(rec_metrics.get("max_violation_pair"))
        same = (rec_pair is not None and mine_pair is not None
                and int(rec_pair.get("step", -1)) == int(mine_pair["step"])
                and int(rec_pair.get("query", -1)) == int(mine_pair["query"]))
        out["to_view"] = {
            "definition": "TO 가 본 씬(접촉 정책 포함)으로 worst_row — 기록의 "
                          "summary.to.metrics.clearance_m / max_violation_pair 와 비교",
            "recorded_clearance_m": float(rec_metrics["clearance_m"]),
            "reconstructed_clearance_m": float(mine_c),
            "abs_diff_mm": float(abs(mine_c - float(rec_metrics["clearance_m"])) * 1000.0),
            "recorded_pair": rec_pair,
            "reconstructed_pair": mine_pair,
            "same_row": bool(same),
        }
    return out


@dataclasses.dataclass
class Shared:
    """chunk 마다 같은 것 — 설정·모델·측정기·변형 config."""

    args: Any
    settings: RunSettings
    robot: Any
    qadr: dict[str, int]
    layout: Any
    meter: Meter
    plan_rows: dict[int, dict]
    variant_configs: dict[str, Any]
    active_rows_config: Any
    outside_distance: float
    recorded_iters: dict[int, int]
    runners: dict[Any, Instrumented] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ChunkInputs:
    """한 chunk 의 복원된 입력. 변형은 전부 이것 위에서 돈다."""

    k: int
    seq: int
    z: Any
    summary: dict
    q_now: np.ndarray
    field: Any
    scene: Any
    reference_chunk: np.ndarray
    reference: np.ndarray
    refined_traj: np.ndarray
    certified: bool
    ref_tip: np.ndarray
    ref_rot: np.ndarray


def prepare_chunk(sh: Shared, path: pathlib.Path) -> tuple[dict[str, Any], Optional[ChunkInputs]]:
    """입력 복원 + 재구성 충실도. 변형을 돌릴 수 없으면 `(기록, None)` 과 `skipped_reason`."""
    settings, meter, layout = sh.settings, sh.meter, sh.layout
    z = np.load(path, allow_pickle=False)
    summary = json.loads(str(z["summary_json"]))
    k = int(z["chunk_index"])
    seq = k + 1  # T14.audit §1: chunk k ↔ 로컬 seq k+1
    rec: dict[str, Any] = {"file": str(path), "chunk_index": k, "seq": seq,
                           "npz_t_step": int(z["t_step"]), "status": summary.get("status"),
                           "validity": summary.get("validity"),
                           "grounding_status": summary.get("grounding_status"),
                           "has_target": bool(summary.get("has_target")),
                           "target_n_points": summary.get("target_n_points"),
                           "variants": None, "skipped_reason": None}
    row = sh.plan_rows.get(seq)
    if row is None:
        rec["skipped_reason"] = f"로컬 기록에 seq={seq} planning 행이 없습니다 (q_now 를 모름)"
        return rec, None
    rec["t_step"] = int(row.get("t_step", 8 * k))
    rec["object_poses"] = row.get("object_poses")
    reference_chunk = np.asarray(z["reference_chunk"], np.float64)
    refined_chunk = np.asarray(z["refined_chunk"], np.float64)
    ref_pair = (float(np.abs(np.asarray(row["actions_reference"]) - reference_chunk).max())
                if "actions_reference" in row else None)
    rec["pairing"] = {
        "definition": "T14.audit §1 — 로컬 planning 행과 서버 기록이 같은 청크인가",
        "refined_vs_local_actions_max": float(
            np.abs(np.asarray(row["actions"]) - refined_chunk).max()),
        "reference_vs_local_actions_reference_max": ref_pair,
    }
    if not (rec["pairing"]["refined_vs_local_actions_max"] <= 1e-5
            and (ref_pair is None or ref_pair <= 1e-5)):
        rec["skipped_reason"] = "서버 기록과 로컬 행이 짝이 아닙니다 (actions 가 다름)"
        return rec, None
    q_now = q_now_from_qpos(row["qpos"], sh.qadr)
    rec["q_now"] = q_now.tolist()

    stats = summary.get("esdf_stats") or {}
    attached = int(stats.get("n_attached_points", 0) or 0) > 0
    rec["attached"] = attached
    field = build_field(z, outside_distance=sh.outside_distance)
    radii = meter.lin.robot_radii
    target_points = z["target_points"] if "target_points" in z.files else None
    has_target = bool(summary.get("has_target")) and target_points is not None
    margin, mask, authorized, _, _ = contact_policy(
        settings, sh.robot.sphere_link_names, has_target=has_target,
        has_free=bool(field.target_free_layers), attached=attached)
    scene = make_scene(field, radii, settings.esdf_margin, target_points=target_points,
                       margin=margin, mask=mask)
    names = [str(n) for n in sh.robot.sphere_link_names]
    rec["contact_policy"] = {
        "phase": settings.phase, "manipulators": settings.manipulators,
        "target_field_policy": settings.target_field_policy,
        "authorized_links": authorized,
        "target_free_mask_links": (sorted({names[i] for i in np.flatnonzero(mask)})
                                   if mask is not None else []),
        "manipulated_source": "target" if margin is not None else None,
        "manipulated_margin_by_link_mm": (
            {n: float(margin[names.index(n)] * 1000.0) for n in sorted(set(names))}
            if margin is not None else None),
        "labels_recorded": False,
    }

    planned = meter.lin.horizon
    reference = layout.chunk_to_trajectory(reference_chunk)[:, :planned]
    refined_traj = layout.chunk_to_trajectory(refined_chunk)[:, :planned]
    fid = fidelity(meter, z, summary, refined_traj, q_now, field, scene, radii,
                   settings.esdf_margin)
    tol = sh.args.fidelity_tol_mm
    checks = []
    if "record_clearance" in fid:
        checks.append(fid["record_clearance"]["max_abs_diff_mm"] <= tol)
    if "to_view" in fid and not attached:
        checks.append(fid["to_view"]["abs_diff_mm"] <= tol)
    fid["tolerance_mm"] = tol
    fid["within_tolerance"] = bool(checks) and all(checks)
    rec["fidelity"] = fid

    ref_tip, ref_rot = meter.tip_and_rot(reference, q_now)
    rec["reference"] = {
        "limit_overshoot": meter.overshoot(reference, q_now),
        "fingertip_path_m": ref_tip.tolist(),
        "to_view_min_clearance_mm": float(
            meter.lin.esdf_clearance(reference, q_now, scene).min() * 1000.0),
    }
    rec["recorded_refined_vs_reference"] = meter.compare(refined_traj, reference, q_now,
                                                         ref_tip, ref_rot)
    to = summary.get("to") or {}
    rec["recorded_to"] = {k2: to.get(k2) for k2 in ("status", "iterations", "sqp_iterations",
                                                    "qp_iterations", "time_budget_hit",
                                                    "collision_enabled")}
    if attached:
        rec["skipped_reason"] = ("쥔 물체의 질의점(attached)이 기록되지 않아 TO 입력을 복원할 수 "
                                 "없습니다 — 변형을 돌리지 않습니다")
        return rec, None
    if not fid["within_tolerance"]:
        rec["skipped_reason"] = "재구성 충실도가 허용 오차를 넘었습니다 — 이 chunk 의 도구 결과는 무효"
        return rec, None
    certified = summary.get("status") == "ok"
    rec["geometry_certified"] = certified
    return rec, ChunkInputs(k=k, seq=seq, z=z, summary=summary, q_now=q_now, field=field,
                            scene=scene, reference_chunk=reference_chunk, reference=reference,
                            refined_traj=refined_traj, certified=certified, ref_tip=ref_tip,
                            ref_rot=ref_rot)


def _runner(sh: Shared, name: str, cfg) -> Instrumented:
    key = (name, _cfg_key(cfg))
    runner = sh.runners.get(key)
    if runner is None:
        runner = Instrumented(sh.robot, sh.layout, cfg, {"value": None})
        sh.runners[key] = runner
    return runner


def run_variants(sh: Shared, rec: dict[str, Any], inp: ChunkInputs, prev_z,
                 legacy_counts: dict[str, dict[str, int]]) -> None:
    """변형 전부. 청크마다 새 solver 로 푼다 (`Instrumented.run` 의 `reset`)."""
    args, settings, meter = sh.args, sh.settings, sh.meter
    names = [str(n) for n in sh.robot.sphere_link_names]
    rec["active_rows_at_reference"] = active_rows(meter, inp.reference, inp.q_now, inp.scene,
                                                  inp.field, sh.active_rows_config, names)
    selected = rec["active_rows_at_reference"]["selected"]

    prev = replay_prev = None
    prev_index = None
    if prev_z is not None:
        key = "refined_chunk" if args.continuity_source == "refined" else "reference_chunk"
        prev = np.asarray(prev_z[key], np.float64)
        # replay 는 **서버가 실제로 넘긴 것**을 쓴다: shadow 면 정책 청크, closed loop 면 refined
        # (`SafePolicy.infer` 의 `_previous_chunk`).
        replay_prev = np.asarray(
            prev_z["reference_chunk" if settings.shadow else "refined_chunk"], np.float64)
        prev_index = int(prev_z["chunk_index"])

    variants: dict[str, Any] = {}
    for name in args.variants:
        cfg = sh.variant_configs[name](inp.k)
        runner = _runner(sh, name, cfg)
        runner.holder["value"] = (inp.scene, inp.q_now, inp.certified)
        if name == "continuity_only":
            use_prev, source = prev, args.continuity_source
        elif name == "replay" and settings.cost["w_continuity"] > 0:
            use_prev, source = replay_prev, ("reference" if settings.shadow else "refined")
        else:
            use_prev, source = None, None
        started = time.perf_counter()
        result, captured = runner.run(inp.reference_chunk,
                                      {"t_step": inp.seq, "previous_physical_chunk": use_prev})
        wall = (time.perf_counter() - started) * 1000.0
        v = variant_record(sh, inp, result, captured, cfg, wall,
                           prev_index if use_prev is not None else None, source, selected)
        if name == "replay":
            v["vs_recorded_refined"] = meter.compare(result.trajectory, inp.refined_traj, inp.q_now)
            legacy_traj = v.pop("_legacy_trajectory", None)
            if legacy_traj is not None:
                v["legacy_vs_recorded_refined"] = meter.compare(legacy_traj, inp.refined_traj,
                                                                inp.q_now)
        v.pop("_legacy_trajectory", None)
        counts = legacy_counts.setdefault(name, {"initial": 0, "accepted": 0,
                                                 "best_unaccepted": 0})
        counts[v["legacy"]["returned"]] += 1
        variants[name] = v
    rec["variants"] = variants


def variant_record(sh: Shared, inp: ChunkInputs, result, captured, cfg, wall_ms, prev_index,
                   source, selected) -> dict[str, Any]:
    meter = sh.meter
    m = result.metrics
    traj = result.trajectory
    legacy = legacy_selection(result, captured)
    v: dict[str, Any] = {
        "config": {
            "collision_enabled": bool(cfg.collision.enabled),
            "w_track": cfg.cost.w_track, "w_smooth": cfg.cost.w_smooth,
            "w_continuity": cfg.cost.w_continuity, "w_slack": cfg.cost.w_slack,
            "time_budget_ms": cfg.sqp.time_budget_ms, "max_iterations": cfg.sqp.max_iterations,
            "previous_chunk_index": prev_index, "continuity_source": source,
        },
        "status": result.status.value,
        "max_violation_m": float(result.max_violation),
        "reference_violation_m": float(result.reference_violation),
        "cost": float(result.cost),
        "solve_ms": float(result.solve_time_ms), "wall_ms": float(wall_ms),
        "sqp_iterations": m.get("sqp_iterations"), "qp_iterations": m.get("qp_iterations"),
        "time_budget_hit": m.get("time_budget_hit"),
        "max_iterations_hit": m.get("max_iterations_hit"),
        "timing_ms": m.get("timing_ms"),
        **{key: m.get(key) for key in ("returned", "returned_index", "initial_merit",
                                      "initial_violation_m", "initial_eligible",
                                      "initial_limit_overshoot", "candidates",
                                      "n_candidates", "n_accepted", "qp_failures")},
        "legacy": {"returned": legacy["returned"], "returned_index": legacy["returned_index"],
                   "capture_ok": legacy["capture_ok"]},
        "worst_row": m.get("max_violation_pair"),
        "limit_overshoot": m.get("limit_overshoot"),
        "exec_window": meter.compare(traj, inp.reference, inp.q_now, inp.ref_tip, inp.ref_rot),
        "notes": list(result.notes),
    }
    if legacy["trajectory"] is not None:
        diff = np.degrees(np.abs(legacy["trajectory"] - traj))
        v["legacy"]["differs_from_new"] = bool(diff.max() > 0.0)
        v["legacy"]["max_joint_dev_vs_new_deg"] = float(diff.max())
        if diff.max() > 0.0:
            v["legacy"]["exec_window"] = meter.compare(legacy["trajectory"], inp.reference,
                                                       inp.q_now, inp.ref_tip, inp.ref_rot)
        v["_legacy_trajectory"] = legacy["trajectory"]
    det = row_details(meter, traj, inp.q_now, inp.scene, inp.field)
    v["clearance_at_returned"] = {
        "min_mm": float(det["clearance"].min() * 1000.0),
        "min_per_step_mm": (det["clearance"].min(axis=1) * 1000.0).tolist(),
        "n_negative": int((det["clearance"] < 0).sum()),
        "selected_rows_mm": [float(det["clearance"][r["step"], r["query"]] * 1000.0)
                             for r in selected],
    }
    return v


def chained_replay(sh: Shared, files: list[pathlib.Path]) -> dict[int, dict[str, Any]]:
    """**서버처럼 청크를 차례로, solver 를 이어서** 푼다 — OSQP warm start 까지 재현하려는 것.

    서버의 `QpSolver` 는 청크 사이에 OSQP 객체와 직전 해 `(x, y)` 를 들고 간다 (`qp.py`). QP 를
    `eps = 1e-3` 에서 멈추므로 그 해는 warm start 에 따라 달라지고, 청크마다 새 solver 로 푸는
    `replay` 변형은 기록의 refined 를 정확히 재현하지 못한다. 여기서는 chunk 0 부터 한 refiner 로
    reset 없이 풀어 그 사슬을 따라간다. **사슬이 끊기는 chunk**(로컬 행 없음 · 쥔 물체 · 충실도
    실패)를 만나면 그 뒤는 `chain_intact = false` 로 적는다 — 그 청크의 QP 를 재현할 수 없으므로
    solver 상태가 서버와 갈라진다.
    """
    out: dict[int, dict[str, Any]] = {}
    runner: Optional[Instrumented] = None
    intact = True
    broken_at = None
    for f in files:
        rec, inp = prepare_chunk(sh, f)
        k = rec["chunk_index"]
        if inp is None:
            if intact:
                broken_at = k
            intact = False
            out[k] = {"ran": False, "chain_intact": False, "chain_broken_at": broken_at,
                      "reason": rec.get("skipped_reason")}
            continue
        cfg = sh.variant_configs["replay"](k)
        if runner is None:
            runner = Instrumented(sh.robot, sh.layout, cfg, {"value": None})
            runner.refiner.reset()
        # 반복 수만 chunk 마다 다르다 — 같은 optimizer(같은 solver)에 config 만 갈아 끼운다.
        runner.refiner.optimizer.config = runner.refiner.optimizer.config.with_overrides(
            {"sqp": {"max_iterations": cfg.sqp.max_iterations,
                     "time_budget_ms": cfg.sqp.time_budget_ms}})
        runner.holder["value"] = (inp.scene, inp.q_now, inp.certified)
        runner.captured = []
        prev = None
        if sh.settings.cost["w_continuity"] > 0:
            prev_path = f.with_name(f"chunk_{k - 1:05d}.npz")
            if prev_path.exists():
                with np.load(prev_path) as pz:
                    prev = np.asarray(pz["reference_chunk" if sh.settings.shadow
                                         else "refined_chunk"], np.float64)
        runner.refiner.refine(inp.reference_chunk,
                              {"t_step": inp.seq, "previous_physical_chunk": prev})
        result = runner.refiner.last_result
        legacy = legacy_selection(result, list(runner.captured))
        entry = {"ran": True, "chain_intact": intact, "chain_broken_at": broken_at,
                 "returned": result.metrics.get("returned"),
                 "legacy_returned": legacy["returned"],
                 "qp_iterations": result.metrics.get("qp_iterations"),
                 "recorded_qp_iterations": (inp.summary.get("to") or {}).get(
                     "metrics", {}).get("qp_iterations"),
                 "new_vs_recorded_refined": sh.meter.compare(result.trajectory, inp.refined_traj,
                                                             inp.q_now)}
        if legacy["trajectory"] is not None:
            entry["legacy_vs_recorded_refined"] = sh.meter.compare(
                legacy["trajectory"], inp.refined_traj, inp.q_now)
        out[k] = entry
        inp.z.close()
    return out


def _cfg_key(cfg) -> str:
    return json.dumps(cfg.to_dict(), sort_keys=True, default=str)


# ======================================================================================
# main
# ======================================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--constraints", required=True,
                    help="서버 `--record-constraints` 의 run 디렉터리 (chunk_*.npz)")
    ap.add_argument("--frames", required=True, help="같은 실행의 로컬 frames.jsonl")
    ap.add_argument("--manifest", default=None,
                    help="로컬 manifest.json (기본: frames.jsonl 옆). phase·manipulators·model_xml")
    ap.add_argument("--server-log", default=None, help="serve_safe.log — 제약 모델·목적함수 설정")
    ap.add_argument("--out", required=True, help="결과 JSON")
    ap.add_argument("--chunks", default="all",
                    help="`all` 또는 `3,5,9-14` 처럼 chunk_index 목록")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--no-chain", dest="chain", action="store_false",
                    help="청크를 이어 푸는 replay(`chained_replay`)를 건너뛴다")
    # 제약 모델 — 주면 로그를 덮는다
    ap.add_argument("--links", choices=("arms", "gripper", "all"), default=None)
    ap.add_argument("--sphere-spacing", type=float, default=None)
    ap.add_argument("--max-spheres-per-capsule", type=int, default=None)
    ap.add_argument("--capsule-radius-scale", type=float, default=None)
    ap.add_argument("--max-sphere-radius", type=float, default=None)
    ap.add_argument("--model-xml", default=None)
    # 정책·목적함수
    ap.add_argument("--esdf-margin", type=float, default=None, help="m")
    ap.add_argument("--target-field-policy", default=None,
                    choices=("relax", "exclude-authorized", "exclude-all"))
    ap.add_argument("--phase", default=None)
    ap.add_argument("--manipulators", nargs="+", default=None)
    ap.add_argument("--hand", choices=("left", "right"), default=None,
                    help="손끝 지표를 잴 손 (기본: 첫 manipulator)")
    ap.add_argument("--rows-per-step", type=int, default=None)
    ap.add_argument("--allow-uncertified", action="store_true")
    # 변형
    ap.add_argument("--smooth-weight", type=float, default=None,
                    help="smooth_only 의 w_smooth (기본: CostConfig 기본값)")
    ap.add_argument("--continuity-weight", type=float, default=None,
                    help="continuity_only 의 w_continuity (기본: CostConfig 기본값)")
    ap.add_argument("--continuity-source", choices=("refined", "reference"), default="refined",
                    help="continuity_only 의 이전 청크. refined(기본) = 기록의 직전 refined_chunk. "
                         "shadow 실행에서 로봇이 실제로 실행한 것은 reference 다")
    ap.add_argument("--budget-clock", choices=("wall", "off"), default="wall",
                    help="wall(기본) = time_budget_ms 를 벽시계로 (서버와 같은 의미, 머신 부하에 "
                         "따라 반복 수가 바뀐다). off = 시간 예산을 끄고 max_iterations 만")
    ap.add_argument("--fidelity-tol-mm", type=float, default=1.0)
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="충실도를 넘은 chunk 만 건너뛰고 나머지는 돌린다 (기본은 하나라도 넘으면 "
                         "변형을 하나도 안 돌린다)")
    return ap


def _parse_chunks(spec: str) -> Optional[set[int]]:
    if spec == "all":
        return None
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    return out


def _chunk_index(path: pathlib.Path) -> int:
    with np.load(path) as z:
        return int(z["chunk_index"])


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from benchmark.trajopt.config import CostConfig, SqpConfig
    from benchmark.trajopt.limits import build_limits
    from benchmark.trajopt.types import ChunkLayout

    started = time.time()
    frames = pathlib.Path(args.frames)
    manifest_path = (pathlib.Path(args.manifest) if args.manifest
                     else frames.with_name("manifest.json"))
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    log = parse_server_log(args.server_log)
    settings = resolve_settings(args, log, manifest)

    robot, qadr, xml_path = build_robot(settings)
    if (settings.n_constraint_spheres is not None
            and robot.n_spheres != settings.n_constraint_spheres):
        raise SystemExit(f"다시 지은 제약 모델의 구 {robot.n_spheres} 개 ≠ 로그의 "
                         f"{settings.n_constraint_spheres} 개 — sphere option 이 다릅니다")
    layout = ChunkLayout.rby1(robot.joint_names)

    # AG3S 의 격자 밖 정책 (`curobo_builder`: unknown_policy free → +max_distance)
    _, _, _, max_distance, unknown_policy = contact_policy(
        settings, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
    outside = max_distance if unknown_policy == "free" else -max_distance

    rows = [json.loads(line) for line in frames.read_text().splitlines() if line.strip()]
    plan_rows = {int(r["seq"]): r for r in rows if r.get("kind") == "planning"}
    files = sorted(pathlib.Path(args.constraints).glob("chunk_*.npz"), key=_chunk_index)
    wanted = _parse_chunks(args.chunks)
    selected_files = [f for f in files if wanted is None or _chunk_index(f) in wanted]

    recorded_iters: dict[int, int] = {}
    for f in files:
        with np.load(f) as z:
            to = json.loads(str(z["summary_json"])).get("to") or {}
            recorded_iters[int(z["chunk_index"])] = int(
                to.get("sqp_iterations") or to.get("iterations") or 1)

    # --- 변형 설정 ------------------------------------------------------------------
    recorded_cost = dict(settings.cost)
    sqp_default = SqpConfig()
    clock = args.budget_clock == "wall"
    cost_defaults = CostConfig()
    smooth_w = args.smooth_weight if args.smooth_weight is not None else cost_defaults.w_smooth
    cont_w = (args.continuity_weight if args.continuity_weight is not None
              else cost_defaults.w_continuity)
    track = {"w_track": recorded_cost["w_track"], "w_slack": recorded_cost["w_slack"]}
    off_terms = {"w_smooth": 0.0, "w_continuity": 0.0}

    def budget(scale: int) -> dict[str, Any]:
        return {"time_budget_ms": (float(sqp_default.time_budget_ms) * scale if clock
                                   else NO_CLOCK_MS),
                "max_iterations": int(sqp_default.max_iterations) * scale}

    def cfg(collision, cost, sqp):
        return to_config(settings, collision=collision, cost=cost, sqp=sqp)

    variant_configs = {
        "replay": lambda k: cfg(settings.collision_enabled, recorded_cost,
                                {"time_budget_ms": NO_CLOCK_MS,
                                 "max_iterations": max(1, recorded_iters.get(k, 1))}),
        "track_only": lambda k: cfg(False, {**track, **off_terms}, budget(1)),
        "smooth_only": lambda k: cfg(False, {**track, **off_terms, "w_smooth": smooth_w},
                                     budget(1)),
        "continuity_only": lambda k: cfg(False, {**track, **off_terms, "w_continuity": cont_w},
                                         budget(1)),
        "collision": lambda k: cfg(True, {**track, **off_terms}, budget(1)),
        "collision_budget_x2": lambda k: cfg(True, {**track, **off_terms}, budget(2)),
        "collision_budget_x4": lambda k: cfg(True, {**track, **off_terms}, budget(4)),
    }
    planning = variant_configs["collision"](0)
    limits = build_limits(robot, layout, dt=planning.horizon.dt, config=planning.limits)
    meter = Meter(robot, layout, planning.horizon.planned, settings.hand, limits)
    # `active_rows` 가 읽을 계획 config — `collision` 변형을 refiner 가 계획 창으로 줄인 것
    active_cfg = Instrumented(robot, layout, planning, {"value": None}).refiner.optimizer.config
    sh = Shared(args=args, settings=settings, robot=robot, qadr=qadr, layout=layout, meter=meter,
                plan_rows=plan_rows, variant_configs=variant_configs,
                active_rows_config=active_cfg, outside_distance=outside,
                recorded_iters=recorded_iters)

    # 1) 충실도 먼저. strict 면 하나라도 넘을 때 변형을 하나도 돌리지 않는다.
    prepared = []
    for f in selected_files:
        rec, inp = prepare_chunk(sh, f)
        prepared.append((f, rec, inp))
    failing = [rec for _, rec, _ in prepared
               if rec.get("fidelity") and not rec["fidelity"]["within_tolerance"]]
    if failing and args.strict:
        print("!!! RECONSTRUCTION FIDELITY FAILED — THE TOOL IS NOT VALID FOR THIS RECORD !!!\n"
              f"    {len(failing)} chunk(s) exceed {args.fidelity_tol_mm} mm: "
              + ", ".join(str(r["chunk_index"]) for r in failing), file=sys.stderr)
        _write(args, settings, xml_path, [rec for _, rec, _ in prepared], {}, {}, started,
               valid=False, reason=f"{len(failing)} chunk(s) exceed the fidelity tolerance")
        return 2

    # 2) 변형
    by_index = {_chunk_index(f): f for f in files}
    legacy_counts: dict[str, dict[str, int]] = {}
    records = []
    for f, rec, inp in prepared:
        if inp is not None and args.variants:
            prev_path = by_index.get(inp.k - 1)
            prev_z = np.load(prev_path) if prev_path is not None else None
            try:
                run_variants(sh, rec, inp, prev_z, legacy_counts)
            finally:
                if prev_z is not None:
                    prev_z.close()
        if inp is not None:
            inp.z.close()
        records.append(rec)
        print(f"[t24] chunk {rec['chunk_index']:3d} seq {rec['seq']:3d} "
              + ("skipped: " + rec["skipped_reason"] if rec.get("skipped_reason")
                 else "fidelity %.5f mm" % rec["fidelity"].get("record_clearance", {}).get(
                     "max_abs_diff_mm", float("nan"))), flush=True)

    # 3) 이어 푸는 replay — 전 chunk 를 차례로 (선택과 무관하게 사슬은 chunk 0 부터다)
    chain = chained_replay(sh, files) if args.chain and "replay" in args.variants else {}
    for rec in records:
        if rec["chunk_index"] in chain:
            rec["replay_chained"] = chain[rec["chunk_index"]]
    chain_counts: dict[str, int] = {}
    for entry in chain.values():
        if entry.get("ran"):
            chain_counts[entry["legacy_returned"]] = chain_counts.get(entry["legacy_returned"], 0) + 1
    _write(args, settings, xml_path, records, legacy_counts, chain_counts, started, valid=True,
           reason=None)
    return 0


def _write(args, settings, xml_path, records, legacy_counts, chain_counts, started, *, valid,
           reason):
    measured = [r for r in records if r.get("fidelity")]
    rc = [r["fidelity"]["record_clearance"]["max_abs_diff_mm"] for r in measured
          if "record_clearance" in r["fidelity"]]
    tv = [r["fidelity"]["to_view"]["abs_diff_mm"] for r in measured
          if "to_view" in r["fidelity"] and not r.get("attached")]
    out = {
        "tool": TOOL,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "valid": bool(valid),
        "invalid_reason": reason,
        "inputs": {"constraints": args.constraints, "frames": args.frames,
                   "server_log": args.server_log, "model_xml_resolved": xml_path},
        "settings": settings.to_json(),
        "variant_budget": {"clock": args.budget_clock, "continuity_source": args.continuity_source,
                           "smooth_weight": args.smooth_weight,
                           "continuity_weight": args.continuity_weight},
        "not_reproduced": [
            "label layer (obstacle names are None; destination margin on held points)",
            "attached query points (chunks with n_attached_points > 0 are skipped)",
            "cross-chunk QP warm start in the per-chunk variants (each chunk is solved by a fresh "
            "solver) — `replay_chained` follows the server's solver chain instead",
            "wall-clock load of the original server (budget variants depend on this machine)",
        ],
        "fidelity_summary": {
            "tolerance_mm": args.fidelity_tol_mm,
            "n_chunks_measured": len(measured),
            "record_clearance_max_abs_diff_mm": max(rc) if rc else None,
            "to_view_max_abs_diff_mm": max(tv) if tv else None,
            "n_within_tolerance": int(sum(1 for r in measured
                                          if r["fidelity"]["within_tolerance"])),
            "n_skipped": int(sum(1 for r in records if r.get("skipped_reason"))),
        },
        "legacy_selection_counts": legacy_counts,
        "legacy_selection_counts_chained_replay": chain_counts,
        "elapsed_s": round(time.time() - started, 1),
        "chunks": records,
    }
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(out), ensure_ascii=False, indent=1))
    print(f"[t24] wrote {path} ({len(records)} chunk records, valid={valid})")


if __name__ == "__main__":
    raise SystemExit(main())
