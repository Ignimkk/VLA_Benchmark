"""GPU 서버 진입점 — π0.5 (+SEAM) + AG3S + TO 를 한 프로세스로 서빙한다.

`openpi/scripts/serve_policy.py` 를 고치지 않는다. 그 스크립트가 하는 일(체크포인트로 정책을
만들고 websocket 으로 띄우기)을 그대로 부르고, 정책을 `SafePolicy` 로 한 겹 감쌀 뿐이다.
openpi 는 vendored 서브모듈이라 손대면 다음 sync 에서 충돌하고, 그 충돌을 푸는 사람은 이 감싸기가
왜 거기 있었는지 모른다.

    python -m benchmark.trajopt.serve_safe \\
        --config pi05_rby1_atomic_lora \\
        --checkpoint /mnt/dev/work/.../29999 \\
        --model-xml <RB-Y1 씬 XML> --port 8000

`--no-safe` 로 띄우면 감싸지 않는다 — 기존 서빙과 같은 동작이라, 문제가 안전 계층에 있는지
아닌지를 플래그 하나로 가를 수 있다.

`--shadow` 는 **계산을 줄이지 않는다.** AG3S·ESDF·SQP·판정이 전부 돌고 `actions` 도 그대로
refined 다. 응답에 정책 **원본** 청크를 `actions_reference` 로 함께 실어, 로컬이 그것을
실행할 수 있게 하는 것뿐이다 (T5 — 수정이 여유거리를 나쁘게 만드는지를 로봇을 움직이기 전에
본다). 로컬도 `pi05_infer.py --safe-shadow` 로 켜야 하고, 짝이 안 맞으면 클라이언트가 즉시
죽는다 — 조용히 refined 를 실행하면 shadow 가 아닌데 shadow 라고 기록된다.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from collections.abc import Sequence


def constraint_links_minus(present: Sequence[str], exclude: Sequence[str],
                           *, known: Sequence[str] = ()) -> tuple[str, ...]:
    """`present` 에서 `exclude` 를 뺀 link 이름. **없는 이름을 주면 여기서 죽는다.**

    조용히 넘어가면 오타 하나로 **아무것도 안 빠진 채 "뺐다" 고 믿게 된다** — 이 프로젝트가
    이미 밟은 함정이다 (`EE_BODY_L` 대 `ee_left`, gripper 열 6 대 7). `link_filter` 는
    집합 교집합이라 모르는 이름을 그냥 버리므로 (`urdf_sphere_chain.py:389`
    `keep = set(link_filter)`), 검사는 필터를 만드는 이 자리에서 해야 한다.

    Args:
        present: 지금 제약 모델이 구를 갖고 있는 link 이름 (`sphere_link_names`).
        exclude: 빼 달라고 받은 이름.
        known: URDF 의 전체 link 이름. 주면 **"URDF 에 없는 이름"** 과 **"URDF 에는 있지만
            제약 모델에 구가 없는 link"** 를 갈라 말한다 — 후자는 빼도 아무 일이 안 일어나므로
            뺐다고 믿는 것이 그대로 위험이다.

    Returns:
        `link_filter` 에 그대로 넘길 수 있는, 순서를 지킨 link 이름 tuple.
    """
    present_order = tuple(dict.fromkeys(present))
    present_set = set(present_order)
    asked = tuple(dict.fromkeys(exclude))
    known_set = set(known)

    missing = [n for n in asked if n not in present_set]
    if missing:
        unknown = [n for n in missing if known_set and n not in known_set]
        sphereless = [n for n in missing if n not in unknown]
        detail = []
        if unknown:
            detail.append(f"URDF 에 없는 이름: {unknown}")
        if sphereless:
            detail.append(f"URDF 에는 있지만 제약 모델에 구가 없는 link: {sphereless}")
        raise ValueError(
            "--exclude-links 가 제약 모델에 없는 이름을 받았습니다 — "
            + "; ".join(detail or [f"제약 모델에 없는 이름: {missing}"])
            + f". 조용히 넘기면 아무것도 안 빠진 채 뺐다고 믿게 되므로 여기서 멈춥니다. "
              f"지금 제약 모델의 link: {sorted(present_set)}")

    kept = tuple(n for n in present_order if n not in set(asked))
    if not kept:
        raise ValueError(
            f"--exclude-links {list(asked)} 가 제약 모델의 link 를 전부 뺐습니다. 충돌 제약 "
            "행이 하나도 없는 실행은 안전 계층이 꺼진 것과 같으므로(그런데 켜진 것처럼 "
            "보입니다) 여기서 멈춥니다 — 그럴 의도면 --no-safe 를 쓰십시오")
    return kept


def sphere_options(args) -> dict:
    """`--sphere-*` flag 중 **실제로 준 것만** 담은 dict (T6f).

    안 준 flag 는 키가 아예 없다 — 기본값을 여기 적으면 `UrdfSphereChain` 의 기본값과 두 곳이
    되고, 갈라지는 날 제약 모델의 굵기가 조용히 달라진다. 빈 dict 면 호출이 예전과 글자 그대로
    같다는 것이 이 함수의 계약이고, 테스트가 그것을 지킨다.
    """
    given = {
        "sphere_spacing": args.sphere_spacing,
        "max_spheres_per_capsule": args.max_spheres_per_capsule,
        "capsule_radius_scale": args.capsule_radius_scale,
        "max_sphere_radius": args.max_sphere_radius,
    }
    return {k: v for k, v in given.items() if v is not None}


def resolve_plan_horizon(value: str):
    """`--plan-horizon` 문자열 → `HorizonConfig.plan_horizon` 값.

    `execution`(기본) = 실행되는 창만 계획한다 · `full` = 청크 전체 · 숫자 = 그 스텝 수.
    `full`/숫자가 T6f 이전 동작으로 되돌리는 길이다 (그때의 기본값은 32 였다).
    """
    from benchmark.trajopt.config import PLAN_EXECUTION_WINDOW

    text = str(value).strip().lower()
    if text in (PLAN_EXECUTION_WINDOW, "exec", "window"):
        return PLAN_EXECUTION_WINDOW
    if text in ("full", "chunk", "all", "none"):
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(
            f"--plan-horizon 은 {PLAN_EXECUTION_WINDOW!r} · 'full' · 정수 중 하나여야 합니다: "
            f"{value!r}") from None


def build_ag3s(model_xml: str, *, voxel: float, range_max: float, links: str,
               backend: str = "legacy", fine_voxel: float | None = None,
               tsdf_voxel: float | None = None,
               attached_sign_threshold: float = 1.5,
               max_field_age_sec: float | None = None,
               exclude_links: Sequence[str] = (),
               constraint_sphere_options: dict | None = None):
    """서버가 쓸 AG3S. 자기 필터는 전신, 제약은 양팔.

    **두 모델은 일부러 다르다.** 자기 필터는 바퀴·베이스까지 있어야 한다 — 머리 카메라가 자기
    몸을 내려다보므로 빠뜨리면 그 점이 클라우드에 남아 로봇에 용접된 유령 장애물로 뭉친다.
    제약은 최적화기가 움직일 수 있는 구에만 걸려야 의미가 있다.

    XML 은 **로봇 모델을 만드는 데만** 쓴다. 서버는 시뮬레이션을 돌리지 않고 로컬이 보낸
    관측만 본다 — 서버가 자기 씬을 돌리면 로봇이 있는 곳과 제약이 설명하는 곳이 갈라진다.

    `exclude_links` 는 **제약 모델에서만** 뺀다 (`--exclude-links`, T6e 진단). 자기 필터는
    그대로 전신이다 — 거기서 빼면 그 link 의 점이 필터를 통과해 장애물로 샌다. 없는 이름은
    `constraint_links_minus` 가 거절한다.

    `constraint_sphere_options` 도 **제약 모델에서만** 쓴다 (T6f — 팔을 실제 굵기로). 같은
    이유다: 자기 필터가 가늘어지면 그 link 의 점이 장애물로 새고, 그것은 제약을 푸는 것과 반대
    방향의 사고다. 덮지 못하는 capsule 이 생기면 `UrdfSphereChain` 이 경고를 찍고 여기서
    `coverage_report()` 를 한 번 더 찍는다 — 조용히 가늘어진 모델로 떠 있는 것이 가장 나쁘다.
    """
    import mujoco

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene
    from benchmark.ag3s.runtime.pipeline import AG3S

    mj_model = mujoco.MjModel.from_xml_path(str(pathlib.Path(model_xml).resolve()))
    scene = TransportScene.attach(mj_model, mujoco.MjData(mj_model))
    filter_robot = build_robot_model(scene)
    spheres = dict(constraint_sphere_options or {})
    constraint_robot = build_constraint_robot_model(
        scene, link_filter=None if links == "all" else ARM_LINKS, sphere_options=spheres)
    # **제약 모델에서만 뺀다.** 자기 필터(`filter_robot`)는 손대지 않는다 — 거기서 빼면 그
    # link 의 점이 필터를 통과해 장애물로 새고, 그것은 제약을 푸는 것과 반대 방향의 사고다.
    links_label = links
    excluded = tuple(dict.fromkeys(exclude_links or ()))
    if excluded:
        before = constraint_robot.n_spheres
        keep = constraint_links_minus(constraint_robot.sphere_link_names, excluded,
                                      known=constraint_robot.model.links)
        constraint_robot = build_constraint_robot_model(scene, link_filter=keep,
                                                       sphere_options=spheres)
        links_label = f"{links} minus {','.join(excluded)}"
        # 조용히 다른 제약으로 떠 있는 것이 가장 나쁘다 — legacy backend 경고와 같은 이유다.
        logging.warning(
            "constraint model: EXCLUDING %s — 구 %d → %d (%d 개 빠짐). 자기 필터 모델은 "
            "그대로 전신이다 (%d 구, 이 link 들도 거기 남아 있다). **빼는 것이 위험을 지우지 "
            "않는다** — 이 link 가 무엇에 부딪혀도 이제 아무도 막지 않는다. 진단용 flag 다",
            ", ".join(excluded), before, constraint_robot.n_spheres,
            before - constraint_robot.n_spheres, filter_robot.n_spheres)
    esdf = {"voxel_size": voxel, "max_distance": 0.4,
            "exclude_support_surfaces": False,
            "backend": backend,
            "attached_sign_threshold_voxels": attached_sign_threshold}
    if backend == "curobo":
        esdf["fine_voxel_size"] = fine_voxel or None
        esdf["tsdf_voxel_size"] = tsdf_voxel or None
    timing = {} if max_field_age_sec is None else {
        "max_field_age_sec": max_field_age_sec}
    config = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": range_max},
        "esdf": esdf,
        **({"timing": timing} if timing else {}),
    })
    logging.info("AG3S: self-filter %d spheres, constraints %d spheres (%s)",
                 filter_robot.n_spheres, constraint_robot.n_spheres, links_label)
    # **구 굵기는 시작할 때 크게 말한다.** 기본값이면 한 줄이고, 덮지 못하는 capsule 이 있으면
    # 여러 줄짜리 경고다 — `--exclude-links` 와 같은 성질의 flag 이므로 같은 크기로 말한다.
    report = constraint_robot.coverage_report()
    if constraint_robot.coverage_shortfall:
        logging.warning("constraint sphere model:\n%s", report)
        logging.warning("self-filter model is untouched by --sphere-* (%d spheres) — 거기서 "
                        "가늘어지면 그 link 의 점이 필터를 통과해 장애물로 샌다",
                        filter_robot.n_spheres)
    else:
        logging.info("constraint sphere model: %s", report)
    # **어느 필드 구현이 도는지 시작할 때 크게 말한다.** T0 의 즉시 실패 조건이
    # "legacy backend 호출" 이므로, 서버가 조용히 legacy 로 떠 있는 것이 가장 나쁜 결과다.
    if backend == "curobo":
        logging.info("AG3S ESDF backend: cuRobo — coarse %.0f mm%s, TSDF %.0f mm, "
                     "attached sign threshold %.1f voxels",
                     voxel * 1000,
                     (f" + fine {fine_voxel * 1000:.0f} mm" if fine_voxel
                      else " (단일 계층)"),
                     (tsdf_voxel or voxel) * 1000, attached_sign_threshold)
    else:
        logging.warning(
            "AG3S ESDF backend: legacy (numpy EsdfBuilder). "
            "AG3S_TOTAL_TEST_Prompt.md 의 T0 은 이것을 즉시 실패 조건으로 둔다 — "
            "통합 테스트를 돌리려면 --esdf-backend curobo 를 주십시오")
    logging.info("AG3S field age limit: %s",
                 "미정 (stale 판정을 하지 않고 staleness_checked=false 를 싣는다)"
                 if max_field_age_sec is None else f"{max_field_age_sec:.3f} s")
    # **attached 슬롯을 여기서 예약한다.** 슬롯은 생성 시점에 고정되고 `attach()` 가 나중에
    # 늘릴 수 없다 — 심볼 그래프와 희소성이 거기서 결정되기 때문이다. 안 넘기면 파지 다음
    # 프레임부터 지각도 최적화도 멈추고 **어디에도 안 찍힌다** (2026-09-18 실측: 14 프레임
    # 동안 상태가 9 프레임 전의 낡은 `ok` 로 나갔다).
    #
    # `safe_replay` 는 그때 고쳤는데 **이 production 서버는 안 고쳤다.** 2026-09-24 에
    # `SafePolicy` 생성자의 가드가 시작할 때 죽여 잡았다 — 그것이 그 가드를 심은 이유다
    # ("파지 뒤에 죽으면 조용하지만 시작할 때 죽으면 안 조용하다").
    from benchmark.trajopt.safe_policy import grasp_parent_links

    parents = grasp_parent_links()
    logging.info("AG3S attached slots: %s", ", ".join(parents))
    return AG3S(config, robot_model=filter_robot, constraint_robot_model=constraint_robot,
                attached_parent_links=parents)


def load_static_geometry(spec: str, model_xml: str, *, links: str):
    """`--static-geometry` 를 도형 목록으로. `None` 이면 해석적 채널을 안 쓴다.

    **기본은 `none` 이다.** 켜는 것이 안전 방향(`min` 이라 답이 커질 수 없다)이지만, 어떤
    도형이 들어왔는지 모르는 채 제약이 조용히 늘어나는 것보다 명시적으로 켜는 편이 낫다 —
    감쇠(F20)를 기본 끔으로 둔 것과 같은 규약이다.

    `auto` 는 `--model-xml` 에서 뽑는다. 이 서버는 시뮬레이션을 돌리지 않고 로컬이 보낸 관측만
    보지만(`build_ag3s` 머리말), **정적 기하는 정의상 안 움직이므로** 시작 때 한 번 읽는 것이
    그 경고에 걸리지 않는다. 움직이는 것(자유물체·로봇)은 `from_mujoco` 가 버린다.
    """
    from benchmark.ag3s.fields import static_scene

    if not spec or spec == "none":
        logging.info("static geometry: 없음 — 격자 밖·미관측은 예전처럼 낙관적으로 답한다 (E4)")
        return None

    if spec == "auto":
        import mujoco

        mj_model = mujoco.MjModel.from_xml_path(str(pathlib.Path(model_xml).resolve()))
        shapes, stats = static_scene.from_mujoco(mj_model)
        logging.info("static geometry: auto from %s — %s", model_xml, stats.summary())
    else:
        shapes = static_scene.load(spec)
        logging.info("static geometry: %s — %d 도형", spec, len(shapes))

    if not shapes:
        raise ValueError(
            f"--static-geometry {spec} 가 도형을 하나도 내놓지 않았습니다. 켰다고 믿은 채 "
            "아무 효과가 없는 것이 가장 나쁜 결과이므로 여기서 멈춥니다")
    if links == "all":
        logging.warning(
            "--links all 과 정적 기하를 함께 켰습니다. 바닥 평면이 바퀴·베이스 구에 어떤 해로도 "
            "못 푸는 위반을 상수로 깝니다 (실측 base -342 mm, wheel -108 mm). 의도한 것이 "
            "아니면 --links arms 를 쓰십시오")
    for s in shapes:
        logging.debug("  static %s", getattr(s, "label", s))
    return shapes


def attention_extractor():
    """정책 응답에서 카메라별 attention 맵을 꺼내는 함수.

    **합성 attention 을 몰래 끼워넣지 않는다.** 정책이 `attention` 키를 실어 보내지 않으면 빈
    딕셔너리를 돌려주고, AG3S 는 target 없이 돈다 — 거리장이 target 복셀을 파내지 않아 제약이
    더 보수적이 되는, 안전한 방향의 실패다. 가짜 attention 을 채우면 이 서버가 실측
    파이프라인인 척하게 되고, 결과를 읽는 사람이 그것을 알 방법이 없다.
    """
    from benchmark.trajopt import wire

    #: 정책 입력 이름 → 로컬이 쓰는 MuJoCo 카메라 이름.
    ALIAS = {"cam_high": "zed_left",
             "cam_left_wrist": "wrist_cam_l",
             "cam_right_wrist": "wrist_cam_r"}

    def extract(_policy_obs, result):
        raw = result.get("attention")
        if not raw:
            return {}
        return {ALIAS.get(k, k): v for k, v in raw.items()
                if k in ALIAS or k in wire.DEFAULT_CAMERAS}

    return extract


def wire_reference_key() -> str:
    """로그에 찍을 선택 키 이름. `wire` 에서 가져온다 — 문자열을 두 곳에 박으면 갈라진다."""
    from benchmark.trajopt import wire

    return wire.ACTIONS_REFERENCE


def build_parser() -> argparse.ArgumentParser:
    """CLI. `main()` 과 테스트가 **같은 parser** 를 본다.

    따로 만들면 "flag 를 안 주면 지금과 같다" 를 테스트가 확인할 수 없다 — 기본값이 갈리는
    순간 서버는 조용히 다른 설정으로 뜬다.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="openpi train config 이름")
    ap.add_argument("--checkpoint", required=True, help="체크포인트 디렉터리")
    ap.add_argument("--model-xml", required=True,
                    help="RB-Y1 씬 XML. 로봇 모델(자기 필터·제약)을 만드는 데만 쓴다")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--default-prompt", default=None)
    ap.add_argument("--voxel", type=float, default=0.020, help="ESDF 복셀 크기 (m)")
    ap.add_argument("--range-max", type=float, default=2.0, help="점군 최대 거리 (m)")
    ap.add_argument("--links", choices=("arms", "all"), default="arms",
                    help="제약을 걸 링크. arms 는 양팔과 손끝만 — 바퀴·베이스는 결정 변수가 "
                         "아니라 고칠 수 없는 위반을 상수로 깔아 실제 신호를 묻는다")
    ap.add_argument("--exclude-links", nargs="+", default=(), metavar="LINK",
                    help="제약 모델에서 **뺄** link 이름. 예: --exclude-links "
                         "link_left_arm_5 link_right_arm_5. --links 가 고른 집합에서 뺀다. "
                         "자기 필터 모델은 손대지 않는다 (거기서 빼면 그 link 의 점이 장애물로 "
                         "샌다). **진단용이다** — 뺀 link 가 무엇에 부딪혀도 아무도 막지 "
                         "않는다. 제약 모델에 없는 이름을 주면 서버가 시작하지 않는다")
    ap.add_argument("--esdf-margin", type=float, default=0.05,
                    help="구 표면이 장애물에서 떨어져 있어야 하는 거리 (m). 제약은 "
                         "`d_esdf(p) - 구반지름 - 이 값 >= 0` 이므로, 구 반지름 81 mm 와 합치면 "
                         "중심에서 131 mm 자유공간을 요구한다 (T6d). **기본값은 안 바꿨다** — "
                         "이 flag 로 조절한다")
    ap.add_argument("--plan-horizon", default="execution",
                    metavar="execution|full|N",
                    help="최적화기가 **다듬는** 스텝 수. 기본 `execution` = 실행되는 창만 "
                         "(`horizon.execution_length`, 로컬의 OPEN_LOOP_HORIZON 과 같은 수) — "
                         "T6f 판정. 그 앞에는 32 였고, 그때 TO 는 회피를 실행되는 앞부분에 "
                         "몰고 접근을 버려지는 뒷부분으로 미뤘다 (실행 창 +89.86 mm, 50 스텝 "
                         "−17.61 mm, 사과 이동 0.0 mm). `full` 이나 숫자를 주면 그 동작으로 "
                         "되돌아간다 — 계획 지평이 길면 예지력이 생기지만 **미룰 자리도 생긴다**")
    ap.add_argument("--sphere-spacing", type=float, default=None, metavar="S",
                    help="제약 모델의 구 간격 (capsule 반지름 단위, 기본 1.0). 낮추면 구가 "
                         "촘촘해지고 팽창이 줄어 반지름이 **내려간다** — 덮개를 잃지 않는 쪽의 "
                         "손잡이다. `--max-spheres-per-capsule` 이 먼저 걸리므로 (RB-Y1 팔뚝은 "
                         "0.48 에서 8 개에 닿는다) 둘을 같이 준다. 자기 필터 모델은 안 바뀐다")
    ap.add_argument("--max-spheres-per-capsule", type=int, default=None, metavar="N",
                    help="capsule 하나가 가질 수 있는 구 수의 상한 (기본 8). 올리면 "
                         "`--sphere-spacing` 이 실제로 듣는다. 제약 행 수가 함께 늘어난다")
    ap.add_argument("--capsule-radius-scale", type=float, default=None, metavar="F",
                    help="제약 모델 capsule 반지름의 배율 (기본 1.0 = URDF 그대로). 1 보다 "
                         "작으면 구가 URDF capsule 을 **덮지 못하고**, 그 사실을 시작 로그에 "
                         "크게 찍는다. RB-Y1 `link_*_arm_5` 는 URDF 75 mm 인데 MJCF mesh "
                         "실측은 65.4~68.4 mm 다 — 그 차이만큼은 근사에서 온 살이다")
    ap.add_argument("--max-sphere-radius", type=float, default=None, metavar="R",
                    help="제약 모델 구 반지름의 **절대 상한** (m). 팽창까지 끝난 값에 걸린다 "
                         "(팔뚝 기본값 81.2 mm). `--capsule-radius-scale` 과 같은 성질이고, "
                         "덮지 못하면 시작 로그에 크게 찍는다")
    ap.add_argument("--no-safe", action="store_true",
                    help="AG3S+TO 를 감싸지 않는다. 기존 서빙과 동일")
    ap.add_argument("--shadow", action="store_true",
                    help="shadow 실행(T5). **서버가 하는 일은 하나도 줄지 않는다** — AG3S 지각·"
                         "ESDF·SQP·판정이 그대로 돌고 응답의 `actions` 도 그대로 refined 다. "
                         "달라지는 것은 응답에 정책 **원본** 청크를 `actions_reference` 로 함께 "
                         "싣는 것뿐이고, 그것을 실행할지는 로컬이 고른다 "
                         "(`pi05_infer.py --safe-shadow`). 로컬도 켜야 한다 — 짝이 안 맞으면 "
                         "클라이언트가 즉시 죽는다")
    ap.add_argument("--allow-uncertified", action="store_true",
                    help="AG3S 가 기하를 인증하지 못한 프레임도 safe 로 볼지. 기본은 보지 않는다")
    ap.add_argument("--no-attention", action="store_true",
                    help="attention 추출을 끈다 (체크포인트를 한 번 더 로드하지 않는다). AG3S 는 "
                         "target 없이 돌아 제약이 더 보수적이 된다")
    ap.add_argument("--record-constraints", default=None, metavar="DIR",
                    help="청크마다 AG3S 중간 산출물(attention·grounding·거리장·제약 여유)을 "
                         "npz 로 남긴다. `--safe-remote` 청크는 이것들을 응답에 싣지 않으므로, "
                         "서버 쪽 파이프라인을 진단하는 유일한 자리다 — 로컬 "
                         "`--record-constraints`(`benchmark/ag3s/experiments/sources/constraint_record.py`)"
                         "와 같은 형식")
    ap.add_argument("--record-constraints-esdf", choices=("none", "occupancy", "full"),
                    default="full")
    ap.add_argument("--esdf-backend", choices=("legacy", "curobo"), default="legacy",
                    help="어느 필드 구현이 돌 것인가. `curobo` 는 cuRobo Mapper 가 필요하므로 "
                         "`.venv-openpi-live` 에서 띄워야 한다. T0 의 즉시 실패 조건이 "
                         "legacy backend 호출이므로 통합 테스트에서는 `curobo` 여야 한다")
    ap.add_argument("--fine-voxel", type=float, default=0.005,
                    help="`--esdf-backend curobo` 의 미세 계층 복셀 (m). 0 이면 단일 계층")
    ap.add_argument("--tsdf-voxel", type=float, default=0.005,
                    help="`--esdf-backend curobo` 의 TSDF 복셀 (m)")
    ap.add_argument("--attached-sign-threshold", type=float, default=1.5,
                    help="쥔 물체 복셀에서 부호를 양수로 강제할지 가르는 문턱 (복셀 단위). "
                         "실제 파지 sweep 으로 정한 값이 1.5 다")
    ap.add_argument("--max-field-age-sec", type=float, default=None,
                    help="거리장이 몇 초까지 쓸 만한가. **기본값은 미정(None)** 이고 그때는 "
                         "stale 판정을 하지 않고 응답에 staleness_checked=false 를 싣는다. "
                         "추측으로 정하지 않는다 (F14)")
    ap.add_argument("--static-geometry", default="none", metavar="none|auto|PATH",
                    help="아는 고정 기하(벽·선반·테이블·바닥)를 해석적 채널에 싣는다 — 거리장이 "
                         "min(복셀, 해석적) 을 답해 미관측·격자 밖의 낙관을 없앤다 (E4·N2). "
                         "none(기본) = 지금과 같다. auto = --model-xml 에서 뽑는다(시뮬). "
                         "PATH = benchmark.ag3s.fields.static_scene 이 쓴 JSON(실기 — MuJoCo 불필요). "
                         "주의: --links all 과 함께 쓰면 바닥이 바퀴·베이스에 못 푸는 행을 "
                         "상수로 깐다 (실측 base -342 mm)")
    return ap


def reject_bad_flag_combinations(ap: argparse.ArgumentParser, args) -> None:
    """안전 계층을 끈 채 그 계층의 flag 를 준 조합을 시작할 때 막는다.

    아무 일도 안 하는 flag 를 받아 놓고 뜨면, 로그를 읽는 사람은 그것이 효과가 있었다고 믿는다.
    """
    if args.shadow and args.no_safe:
        ap.error("--shadow 는 안전 계층이 돌아야 뜻이 있습니다 (--no-safe 는 그것을 끕니다). "
                 "shadow 는 '전부 계산하되 수정을 로봇에 보내지 않는' 실행이므로, 계산이 없으면 "
                 "그림자로 둘 것도 없습니다")
    if args.exclude_links and args.no_safe:
        ap.error("--exclude-links 는 제약 모델을 고치는 flag 입니다 (--no-safe 는 그 모델을 "
                 "아예 안 만듭니다). 그대로 띄우면 flag 가 아무 일도 안 하는데 link 를 뺐다고 "
                 "믿게 됩니다")
    if args.no_safe and sphere_options(args):
        ap.error("--sphere-* 는 제약 모델의 구를 고치는 flag 입니다 (--no-safe 는 그 모델을 "
                 "아예 안 만듭니다). 그대로 띄우면 팔을 가늘게 모델링했다고 믿은 채 안전 계층이 "
                 "꺼진 서버가 뜹니다")
    # `--plan-horizon` 은 여기서 한 번 읽어 본다. 잘못된 값이 서버를 띄운 뒤에 죽으면 그때는
    # 체크포인트 두 벌을 이미 GPU 에 올린 뒤다.
    try:
        resolve_plan_horizon(args.plan_horizon)
    except ValueError as exc:
        ap.error(str(exc))


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    reject_bad_flag_combinations(ap, args)

    logging.basicConfig(level=logging.INFO, force=True)
    # openpi 는 서브모듈이라 sys.path 에 올려야 한다 (`serve_policy.py` 와 같은 규칙).
    repo = pathlib.Path(__file__).resolve().parents[2]
    for path in (repo, repo / "src" / "openpi" / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config), args.checkpoint, default_prompt=args.default_prompt)

    if args.no_safe:
        logging.info("safety layer OFF — serving the bare policy")
        served = policy
    else:
        from benchmark.trajopt.config import TrajOptConfig
        from benchmark.trajopt.safe_policy import SafePolicy

        served_policy = policy
        if not args.no_attention:
            from benchmark.trajopt.attention_policy import AttentionPolicy, load_attention_model

            logging.info("loading a second copy of the checkpoint for attention extraction")
            attn_model = load_attention_model(args.config, args.checkpoint)
            served_policy = AttentionPolicy(policy, attn_model)

        recorder = None
        if args.record_constraints:
            from benchmark.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

            recorder = ConstraintRecordWriter(
                args.record_constraints, esdf_mode=args.record_constraints_esdf,
                meta={"config": args.config, "checkpoint": args.checkpoint,
                      "model_xml": args.model_xml, "links": args.links,
                      # shadow·exclude 일 때만 더한다 — 기본 기록을 T0 때와 같은 키 집합으로
                      # 둔다. 준 경우에는 반드시 남긴다: 어느 link 를 뺀 실행인지 모르는 기록은
                      # 다른 실행과 비교할 수 없다.
                      **({"shadow": True} if args.shadow else {}),
                      **({"exclude_links": list(args.exclude_links)}
                         if args.exclude_links else {}),
                      # **다듬는 창은 항상 남긴다.** T6f 에서 기본값이 32 → 실행 창으로
                      # 바뀌었으므로, 안 남기면 T6d 기록과 이 기록이 meta 로 구별되지 않는다 —
                      # 그리고 둘은 서로 다른 것을 재고 있다.
                      "plan_horizon": args.plan_horizon,
                      **({"sphere_options": sphere_options(args)}
                         if sphere_options(args) else {})})
            logging.info("recording AG3S constraint diagnostics to %s", recorder.run_dir)

        to_config = TrajOptConfig.from_dict({
            "collision": {"backend": "esdf", "esdf_margin": args.esdf_margin,
                          "use_support_planes": False},
            # 기하 인증 요구는 여기 **한 곳**에서만 켜고 끈다.
            "safety": {"require_certified_geometry": not args.allow_uncertified},
            # 다듬는 창. `execution` 이면 `horizon.execution_length` 를 따라가므로 실행 길이가
            # 두 번 적히지 않는다 (`config.PLAN_EXECUTION_WINDOW` 머리말).
            "horizon": {"plan_horizon": resolve_plan_horizon(args.plan_horizon)},
        })
        horizon = to_config.horizon
        logging.info(
            "TO plan window: %d of %d chunk steps (execution_length=%d) — %s",
            horizon.planned, horizon.horizon, horizon.execution_length,
            "실행되는 창만 다듬는다: 회피를 미룰 뒷부분이 없다"
            if horizon.plans_execution_window_only else
            f"실행 창 뒤로 {horizon.planned - horizon.execution_length} 스텝을 더 계획한다 — "
            "예지력이 생기지만 **회피를 미룰 자리도 생긴다** (T6d)")
        logging.info("TO esdf_margin: %.1f mm (구 반지름과 합쳐야 중심 기준 요구 자유공간이다)",
                     to_config.collision.esdf_margin * 1000)

        served = SafePolicy(
            served_policy,
            ag3s=build_ag3s(args.model_xml, voxel=args.voxel,
                            range_max=args.range_max, links=args.links,
                            backend=args.esdf_backend,
                            fine_voxel=args.fine_voxel,
                            tsdf_voxel=args.tsdf_voxel,
                            attached_sign_threshold=args.attached_sign_threshold,
                            max_field_age_sec=args.max_field_age_sec,
                            exclude_links=args.exclude_links,
                            constraint_sphere_options=sphere_options(args)),
            static_geometry=load_static_geometry(args.static_geometry, args.model_xml,
                                                 links=args.links),
            to_config=to_config,
            attention_fn=attention_extractor(),
            recorder=recorder,
            shadow=args.shadow,
        )
        # **어느 모드로 떠 있는지 시작할 때 크게 말한다.** legacy backend 경고와 같은 이유다 —
        # 조용히 shadow 로 떠 있으면(또는 shadow 가 아닌 채로) 로그를 읽는 사람이 그 실행이
        # 로봇을 움직였는지 아닌지 알 방법이 없다.
        if args.shadow:
            logging.info(
                "SHADOW mode: everything runs (AG3S · ESDF · SQP · verdict) and the response "
                "carries the policy chunk as %r next to the refined `actions`. The local side "
                "must run with --safe-shadow; a mismatched pair fails immediately on the "
                "client.", wire_reference_key())
        else:
            logging.info("closed-loop mode: the response carries the refined chunk only "
                         "(no %s key). Local --safe-shadow will refuse to run against this "
                         "server.", wire_reference_key())

    logging.info("serving on port %d", args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=served, host="0.0.0.0", port=args.port,
        metadata=getattr(served, "metadata", {}),
    ).serve_forever()


if __name__ == "__main__":
    main()
