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
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys


def build_ag3s(model_xml: str, *, voxel: float, range_max: float, links: str):
    """서버가 쓸 AG3S. 자기 필터는 전신, 제약은 양팔.

    **두 모델은 일부러 다르다.** 자기 필터는 바퀴·베이스까지 있어야 한다 — 머리 카메라가 자기
    몸을 내려다보므로 빠뜨리면 그 점이 클라우드에 남아 로봇에 용접된 유령 장애물로 뭉친다.
    제약은 최적화기가 움직일 수 있는 구에만 걸려야 의미가 있다.

    XML 은 **로봇 모델을 만드는 데만** 쓴다. 서버는 시뮬레이션을 돌리지 않고 로컬이 보낸
    관측만 본다 — 서버가 자기 씬을 돌리면 로봇이 있는 곳과 제약이 설명하는 곳이 갈라진다.
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
    constraint_robot = build_constraint_robot_model(
        scene, link_filter=None if links == "all" else ARM_LINKS)
    config = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": range_max},
        "esdf": {"voxel_size": voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False},
    })
    logging.info("AG3S: self-filter %d spheres, constraints %d spheres (%s)",
                 filter_robot.n_spheres, constraint_robot.n_spheres, links)
    return AG3S(config, robot_model=filter_robot, constraint_robot_model=constraint_robot)


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


def main() -> None:
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
    ap.add_argument("--esdf-margin", type=float, default=0.05)
    ap.add_argument("--no-safe", action="store_true",
                    help="AG3S+TO 를 감싸지 않는다. 기존 서빙과 동일")
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
    ap.add_argument("--static-geometry", default="none", metavar="none|auto|PATH",
                    help="아는 고정 기하(벽·선반·테이블·바닥)를 해석적 채널에 싣는다 — 거리장이 "
                         "min(복셀, 해석적) 을 답해 미관측·격자 밖의 낙관을 없앤다 (E4·N2). "
                         "none(기본) = 지금과 같다. auto = --model-xml 에서 뽑는다(시뮬). "
                         "PATH = benchmark.ag3s.fields.static_scene 이 쓴 JSON(실기 — MuJoCo 불필요). "
                         "주의: --links all 과 함께 쓰면 바닥이 바퀴·베이스에 못 푸는 행을 "
                         "상수로 깐다 (실측 base -342 mm)")
    args = ap.parse_args()

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
                      "model_xml": args.model_xml, "links": args.links})
            logging.info("recording AG3S constraint diagnostics to %s", recorder.run_dir)

        served = SafePolicy(
            served_policy,
            ag3s=build_ag3s(args.model_xml, voxel=args.voxel,
                            range_max=args.range_max, links=args.links),
            static_geometry=load_static_geometry(args.static_geometry, args.model_xml,
                                                 links=args.links),
            to_config=TrajOptConfig.from_dict({
                "collision": {"backend": "esdf", "esdf_margin": args.esdf_margin,
                              "use_support_planes": False},
                # 기하 인증 요구는 여기 **한 곳**에서만 켜고 끈다.
                "safety": {"require_certified_geometry": not args.allow_uncertified},
            }),
            attention_fn=attention_extractor(),
            recorder=recorder,
        )

    logging.info("serving on port %d", args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=served, host="0.0.0.0", port=args.port,
        metadata=getattr(served, "metadata", {}),
    ).serve_forever()


if __name__ == "__main__":
    main()
