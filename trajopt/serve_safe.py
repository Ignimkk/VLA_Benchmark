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


#: `--links` 선택지. **기본은 `arms` 이고 그것이 지금까지의 서버다.**
LINK_GROUPS: tuple[str, ...] = ("arms", "gripper", "all")


def target_field_policy_choices() -> tuple[str, ...]:
    """`--target-field-policy` 선택지 (CLI 는 하이픈, config 는 밑줄).

    목록의 정본은 `ConstraintConfig` 쪽 하나다 (`config.TARGET_FIELD_POLICIES`) — 선택지를
    여기 다시 적으면 갈라지고, 갈라지는 날 parser 가 받은 이름이 config 에 없는 이름이 된다.
    """
    from benchmark.ag3s.config import TARGET_FIELD_POLICIES

    return tuple(name.replace("_", "-") for name in TARGET_FIELD_POLICIES)


def resolve_target_field_policy(value: str) -> str:
    """`"exclude-authorized"` → `"exclude_authorized"`. 모르는 이름은 여기서 죽는다."""
    from benchmark.ag3s.config import TARGET_FIELD_POLICIES

    text = str(value).strip().lower().replace("-", "_")
    if text not in TARGET_FIELD_POLICIES:
        raise ValueError(
            f"--target-field-policy 는 {list(target_field_policy_choices())} 중 하나여야 "
            f"합니다: {value!r}")
    return text


def constraint_link_filter(links: str):
    """`--links` 값 → `build_constraint_robot_model(link_filter=...)` 에 줄 것.

    `arms`(기본) = 양팔 14 link + 손가락 4 · `gripper` = **손가락 4 만** · `all` = 전신(`None`).

    이름 목록을 `grounding_report` 에서 가져오는 것이 요점이다 (T7b). 같은 일을
    `--exclude-links` 로 하려면 남길 것 넷을 빼고 열넷을 손으로 적어야 하고, `link_filter` 는
    집합 교집합이라 오타 하나가 조용히 사라진다 — 그래서 집합 이름을 선택지로 둔다.
    """
    from benchmark.ag3s.experiments.reports.grounding_report import ARM_LINKS, GRIPPER_LINKS

    if links == "all":
        return None
    if links == "arms":
        return ARM_LINKS
    if links == "gripper":
        return GRIPPER_LINKS
    raise ValueError(f"--links 는 {LINK_GROUPS} 중 하나여야 합니다: {links!r}")


def parse_link_scales(pairs: Sequence[str]) -> dict[str, float]:
    """`["gripper=0.35"]` → `{link: scale}`. 빈 입력이면 빈 dict (예전과 같은 모델).

    키는 **link 이름이거나 `--links` 의 집합 이름** (`arms` · `gripper`)이다. 집합 이름을
    받는 것이 이 flag 의 요점이다 — 손가락 넷을 손으로 적는 것이 오타 위험이기 때문이다.
    `all` 은 받지 않는다: 전신에 거는 배율은 `--capsule-radius-scale` 그 자체다.

    형식이 아니거나 배율이 0 이하면 **여기서 죽는다.** 조용히 버리면 얇게 했다고 믿은 채
    예전 굵기로 뜨고, 그것이 `--exclude-links` 가 이미 밟은 함정이다. URDF 에 없는 이름은
    `UrdfSphereChain` 이 거절한다 — 검사는 이름을 아는 곳에서 한다.
    """
    out: dict[str, float] = {}
    for item in pairs or ():
        text = str(item)
        if "=" not in text:
            raise ValueError(
                f"--capsule-radius-scale-link 는 LINK=배율 형식입니다: {text!r} "
                f"(예: gripper=0.35, link_left_arm_5=0.8)")
        name, _, value = text.partition("=")
        name = name.strip()
        try:
            scale = float(value)
        except ValueError:
            raise ValueError(
                f"--capsule-radius-scale-link {text!r} 의 배율이 숫자가 아닙니다") from None
        if not scale > 0.0:
            raise ValueError(
                f"--capsule-radius-scale-link {text!r}: 배율은 0 보다 커야 합니다")
        if name == "all":
            raise ValueError(
                "--capsule-radius-scale-link all=... 은 받지 않습니다. 전신에 거는 배율은 "
                "--capsule-radius-scale 그 자체입니다")
        if name in LINK_GROUPS:
            group = constraint_link_filter(name)
            for link in group or ():
                out[str(link)] = scale
        elif not name:
            raise ValueError(f"--capsule-radius-scale-link {text!r}: link 이름이 비었습니다")
        else:
            out[name] = scale
    return out


def parse_link_extents(pairs: Sequence[str]) -> dict[str, tuple[float, float]]:
    """`["link_left_arm_5=-0.10"]` → `{link: (z_min, z_max)}`. 빈 입력이면 빈 dict (예전 모델).

    **단위는 미터다** (`--esdf-margin` 과 같은 규약). `LINK=Z` 는 "z 이상만 남긴다" = 원위
    절단면이고 (RB-Y1 팔의 원위 방향이 −z 다), `LINK=Zmin:Zmax` 는 그 구간만 남긴다.

    `|z| > 3 m` 는 **거절한다.** mm 로 적어 `-100` 을 주면 유지 구간이 `[-100 m, inf]` 가 되어
    **아무것도 안 잘리는데 잘랐다고 믿는다** — 이 flag 에서 가장 있을 법한 실수이고, 조용히
    지나가면 안 되는 종류다 (`--capsule-radius-scale-link` 의 오타 거절과 같은 규율).

    집합 이름(`arms`·`gripper`·`all`)은 받지 않는다. 절단면은 **link frame 의 z** 이고 그 값은
    link 마다 다른 것을 뜻하므로, 집합에 한 숫자를 거는 것은 뜻이 없다 — 양팔을 자르려면 두
    이름을 다 적는다 (`link_left_arm_5=-0.10 link_right_arm_5=-0.10`).
    """
    out: dict[str, tuple[float, float]] = {}
    for item in pairs or ():
        text = str(item)
        if "=" not in text:
            raise ValueError(
                f"--capsule-extent-link 는 LINK=Z 또는 LINK=Zmin:Zmax 형식입니다 (미터): "
                f"{text!r} (예: link_left_arm_5=-0.10)")
        name, _, value = text.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"--capsule-extent-link {text!r}: link 이름이 비었습니다")
        if name in LINK_GROUPS:
            raise ValueError(
                f"--capsule-extent-link {text!r}: 집합 이름({LINK_GROUPS})은 받지 않습니다. "
                "절단면은 link frame 의 z 이므로 link 마다 뜻이 다릅니다 — 이름을 각각 적으십시오")
        try:
            parts = [float(v) for v in value.split(":")] if ":" in value else [float(value)]
        except ValueError:
            raise ValueError(
                f"--capsule-extent-link {text!r} 의 z 가 숫자가 아닙니다 (미터)") from None
        if any(abs(v) > 3.0 for v in parts):
            raise ValueError(
                f"--capsule-extent-link {text!r}: **단위는 미터입니다.** |z| > 3 m 는 받지 "
                f"않습니다 — mm 로 적으면 아무것도 안 잘리는데 잘랐다고 믿게 됩니다 "
                f"(-100 mm 는 -0.10 입니다)")
        if len(parts) == 1:
            out[name] = (parts[0], float("inf"))
        elif len(parts) == 2:
            if not parts[0] < parts[1]:
                raise ValueError(
                    f"--capsule-extent-link {text!r}: Zmin < Zmax 여야 합니다")
            out[name] = (parts[0], parts[1])
        else:
            raise ValueError(
                f"--capsule-extent-link {text!r}: 값은 Z 하나이거나 Zmin:Zmax 둘입니다")
    return out


#: `--w-*` flag 이름 → `CostConfig` 필드 이름. 목록이 한 곳이어야 flag 를 더할 때 announce 와
#: config 가 갈라지지 않는다.
COST_WEIGHTS: tuple[str, ...] = ("w_track", "w_smooth", "w_continuity", "w_slack")


def cost_overrides(args) -> dict[str, float]:
    """`--w-*` 중 **실제로 준 것만** 담은 dict. 안 준 flag 는 키가 아예 없다.

    `sphere_options` 와 같은 계약이다 — 기본값을 여기 적으면 `CostConfig` 와 두 곳이 되고,
    갈라지는 날 목적함수가 조용히 달라진다. 빈 dict 면 `TrajOptConfig` 호출이 예전과 글자
    그대로 같다.
    """
    given = {name: getattr(args, name, None) for name in COST_WEIGHTS}
    return {k: float(v) for k, v in given.items() if v is not None}


def announce_cost_weights(overrides: dict) -> None:
    """**기본이 아닌 목적함수로 떠 있으면 크게 말한다.**

    충돌 제약과 달리 이 넷은 *"무엇을 금지하나"* 가 아니라 *"무엇을 더 좋다고 보나"* 를 바꾼다.
    그래서 조용히 달라져도 위반 수치에는 안 나타나고, **청크가 얼마나 바뀌었나**에만 나타난다 —
    그 값은 지금까지 closed loop 기록에 없었다 (T9 가 그것을 실었다). 로그가 이 조합을 적어야
    두 실행의 궤적 차이를 목적함수 차이로 되짚을 수 있다.
    """
    if not overrides:
        return
    from benchmark.trajopt.config import CostConfig

    base = CostConfig()
    logging.getLogger(__name__).warning(
        "!!! THE OBJECTIVE IS NOT THE DEFAULT ONE !!!\n    %s\n"
        "    제약이 아니라 **목적함수**를 바꾼 것입니다 — 위반 수치는 그대로여도 청크가 달라집니다."
        " 응답의 actions_reference 와 actions 를 견주면 그 크기가 보입니다.",
        ", ".join(f"{k}: {getattr(base, k):g} → {v:g}" for k, v in sorted(overrides.items())))


def announce_row_budget(*, n_constraint_spheres: int, n_filter_spheres: int,
                        planned: int | None = None, rows_per_step: int | None = None) -> None:
    """구 개수와 **행 개수**를 시작 로그에 찍는다. 실시간으로 도는지가 여기 달렸다.

    `--sphere-spacing` 을 내려 구를 촘촘하게 만들면 구 개수가 는다. 그런데 **QP 행은 늘지
    않는다** — `horizon × reduction.rows_per_step` 이고 구 개수와 무관하다
    (`linearize.py` 의 활성 띠). 늘어나는 것은 ESDF 질의점 수와 FK 평가 비용이다 (실측: 구
    120 → 218 에서 QP 행 192 그대로, FK 0.509 → 0.690 ms). 그 구분이 로그에 있어야 "구를 늘렸다"
    가 "실시간을 잃었다" 로 읽히지 않는다.
    """
    log = logging.getLogger(__name__)
    if planned is None or rows_per_step is None:
        log.info("AG3S spheres: constraint %d, self-filter %d",
                 n_constraint_spheres, n_filter_spheres)
        return
    log.info(
        "row budget: 제약 구 %d 개 → ESDF 질의 행 %d/청크 (%d 스텝 × %d 구) · "
        "QP 행 %d/청크 (%d × rows_per_step %d, **구 개수와 무관**) · 자기 필터 구 %d 개",
        n_constraint_spheres, planned * n_constraint_spheres, planned, n_constraint_spheres,
        planned * rows_per_step, planned, rows_per_step, n_filter_spheres)


def announce_diagnostic_scope(*, links: str, self_collision: bool,
                              link_scales: dict, constraint_links: Sequence[str] = (),
                              filter_spheres: int | None = None,
                              target_field_policy: str = "relax",
                              authorized_links: Sequence[str] = (),
                              link_extents: dict | None = None,
                              radius_by_link: str = "") -> None:
    """**무엇을 끄고 떴는지 시작 로그에 크게 찍는다.** 조용한 것이 이 프로젝트의 함정이다.

    `--exclude-links` 와 `--sphere-*` 가 같은 성질의 flag 이고 같은 크기로 말한다. 이 함수를
    따로 뺀 이유는 하나다 — 메시지가 실제로 나가는지 테스트가 MuJoCo 없이 확인할 수 있어야
    한다. `build_ag3s` 안에 묻어 두면 그 검사가 소스 문자열 비교로 내려간다.
    """
    log = logging.getLogger(__name__)
    if links != "arms":
        log.warning(
            "!!! CONSTRAINT SCOPE IS %s, NOT THE DEFAULT 'arms' !!!\n"
            "    제약이 걸리는 link: %s\n"
            "    **여기 없는 것은 무엇에 부딪혀도 아무도 막지 않습니다** — 팔뚝·몸통·반대팔이"
            " 그 안에 있습니다. 진단용입니다 (T7b).",
            links.upper(), ", ".join(str(n) for n in constraint_links) or "(전신)")
    if not self_collision:
        log.warning(
            "!!! SELF-COLLISION IS OFF (--no-self-collision) !!!\n"
            "    쥔 물체와 로봇 자신의 쌍이 전부 면제됩니다 — 로봇이 자기 자신과 부딪혀도 "
            "아무도 모릅니다.\n"
            "    이 repo 에 로봇-로봇 쌍 검사는 원래 없습니다 (N1). 이 flag 가 끄는 것은 "
            "**쥔 물체 대 로봇 구** 한 블록뿐이고, 그것이 여기 있는 자기 충돌 전부입니다.\n"
            "    행은 남고 mask 만 0 이므로 다시 켜는 것은 flag 를 빼는 것뿐입니다.")
    if target_field_policy == "exclude_authorized":
        log.warning(
            "!!! TARGET IS EXCLUDED FROM THE DISTANCE FIELD FOR AUTHORIZED LINKS "
            "(--target-field-policy exclude-authorized) !!!\n"
            "    권한 있는 link (%s) 의 질의점은 **target 이 빠진 계층**에 거리를 묻습니다 — "
            "그 link 은 아직 쥐지 않은 target 을 통과할 수 있습니다.\n"
            "    권한 없는 link 은 그대로이고, table·crate 는 그 계층에도 남아 있으므로 "
            "보호가 유지됩니다. 쥔 뒤에는 이 정책이 돌지 않습니다 (그때 target 은 crate 이고 "
            "crate 는 빼면 안 됩니다).",
            ", ".join(str(n) for n in authorized_links) or "(없음 — 아무 일도 일어나지 않습니다)")
    if target_field_policy == "exclude_all":
        log.warning(
            "!!! TARGET IS EXCLUDED FROM THE DISTANCE FIELD FOR **EVERY** LINK "
            "(--target-field-policy exclude-all) !!!\n"
            "    **E1 이 되살아납니다.** 조작 대상을 필드에서 파내면 손끝뿐 아니라 몸통·전완·"
            "반대팔에게도 사라집니다 — 사과 위로 팔꿈치가 지나가도 아무도 막지 않습니다.\n"
            "    사용자가 명시한 fallback 입니다. 이름으로 빼는 쪽은 exclude-authorized "
            "입니다. table·crate 는 어느 쪽에서도 그대로 막습니다.")
    extents = dict(link_extents or {})
    if link_scales or extents:
        # **굵기와 절단을 한 warning 에 같이 찍는다.** 둘은 다른 축이지만 같은 link 에 함께
        # 걸리는 것이 정상이고 (자르기만 하면 잘린 끝의 구가 여전히 자기 반지름만큼 부푼다),
        # 따로 찍으면 읽는 사람이 유효 반지름을 손으로 계산해야 한다. 그래서 마지막 줄에
        # **결과**(link 별 유효 반지름)를 함께 싣는다.
        bits = []
        if link_scales:
            bits.append("    PER-LINK CAPSULE RADIUS SCALE: " + ", ".join(
                f"{k}={v:g}" for k, v in sorted(link_scales.items())))
        if extents:
            bits.append("    CAPSULES CUT SHORT (link frame z, mm): " + ", ".join(
                f"{k}=[{lo * 1000:+.0f}, "
                + ("inf" if hi == float("inf") else f"{hi * 1000:+.0f}") + "]"
                for k, (lo, hi) in sorted(extents.items())))
        if radius_by_link:
            bits.append(f"    link 별 유효 반지름: {radius_by_link}")
        log.warning(
            "!!! THE CONSTRAINT SPHERE MODEL IS NOT THE DEFAULT ONE !!!\n%s\n"
            "    URDF capsule 보다 얇거나 짧은 구는 그만큼 로봇을 덮지 못합니다 — 아래 coverage "
            "report 가 그 수치와 **어느 하위 link 이 풀려났는지**를 적습니다. 자기 필터 모델은 "
            "손대지 않습니다%s.",
            "\n".join(bits),
            "" if filter_spheres is None else f" ({filter_spheres} 구 그대로)")


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
        # 안 준 flag 는 빈 dict 가 되고, 빈 dict 는 여기서 떨어진다 — `None` 검사와 같은 규약.
        "capsule_radius_scale_by_link":
            parse_link_scales(getattr(args, "capsule_radius_scale_link", ())) or None,
        # 길이 손잡이도 같은 통로로 간다 — 안 준 flag 는 키가 아예 없다.
        "capsule_extent_by_link":
            parse_link_extents(getattr(args, "capsule_extent_link", ())) or None,
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
               constraint_sphere_options: dict | None = None,
               self_collision: bool = True,
               target_field_policy: str = "relax",
               plan_horizon_steps: int | None = None,
               rows_per_step: int | None = None):
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

    `links="gripper"` 는 **손가락 넷만** 제약에 남긴다 (T7b 진단). `self_collision=False` 는 쥔
    물체 대 로봇 구 블록을 끈다 — 이 repo 의 자기 충돌은 그 블록 하나뿐이다. 둘 다 기본값이
    예전 그대로이고, 기본이 아닐 때는 `announce_diagnostic_scope` 가 시작 로그에 크게 찍는다.
    """
    import mujoco

    from benchmark.ag3s.config import DEFAULT_CONTACT_LINKS, AG3SConfig
    from benchmark.ag3s.experiments.reports.grounding_report import (
        build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene
    from benchmark.ag3s.runtime.pipeline import AG3S

    mj_model = mujoco.MjModel.from_xml_path(str(pathlib.Path(model_xml).resolve()))
    scene = TransportScene.attach(mj_model, mujoco.MjData(mj_model))
    filter_robot = build_robot_model(scene)
    spheres = dict(constraint_sphere_options or {})
    constraint_robot = build_constraint_robot_model(
        scene, link_filter=constraint_link_filter(links), sphere_options=spheres)
    # **기본이 아닌 범위·자기충돌·굵기로 떠 있으면 여기서 크게 말한다.** 제약 모델을 만든
    # 직후에 찍는 이유는, 이 아래의 `--exclude-links` 경고와 coverage report 가 그 뒤에 오면서
    # 로그가 "무엇을 뺐는가" 순서로 읽히게 하기 위해서다.
    policy = resolve_target_field_policy(target_field_policy)
    announce_diagnostic_scope(
        links=links, self_collision=self_collision,
        link_scales=dict(spheres.get("capsule_radius_scale_by_link") or {}),
        constraint_links=sorted(set(constraint_robot.sphere_link_names)),
        filter_spheres=filter_robot.n_spheres,
        target_field_policy=policy,
        link_extents=dict(spheres.get("capsule_extent_by_link") or {}),
        radius_by_link=constraint_robot.radius_by_link_summary(),
        # 권한 집합의 정본은 `contact.contact_links` 다. 여기서는 **읽기만** 한다 — 서버가
        # 자기 목록을 따로 들고 있으면 config 를 고친 날 로그와 실제가 갈라진다. 제약 모델에
        # 없는 link 은 애초에 질의점이 없으므로 교집합만 찍는다.
        authorized_links=sorted(
            {n for names in DEFAULT_CONTACT_LINKS.values() for n in names}
            & set(constraint_robot.sphere_link_names)))
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
    constraint_section: dict = {}
    if not self_collision:
        constraint_section["self_collision"] = False
    if policy != "relax":
        constraint_section["target_field_policy"] = policy
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
        # **기본인 경우에는 키를 아예 넣지 않는다.** 기본값을 여기 다시 적으면
        # `ConstraintConfig` 와 두 곳이 되고, 갈라지는 날 갈라진 쪽이 안전 판정이다
        # (`sphere_options` 와 같은 규약). 두 flag 가 같은 section 을 쓰므로 한 dict 로 모은다.
        **({"constraint": constraint_section} if constraint_section else {}),
    })
    logging.info("AG3S: self-filter %d spheres, constraints %d spheres (%s)",
                 filter_robot.n_spheres, constraint_robot.n_spheres, links_label)
    # **구 개수와 행 개수를 갈라 찍는다.** 구를 촘촘하게 만들면 구는 늘지만 QP 행은 안 늘고,
    # 그 구분이 없으면 "구를 늘렸다" 가 "실시간을 잃었다" 로 읽힌다.
    announce_row_budget(n_constraint_spheres=constraint_robot.n_spheres,
                        n_filter_spheres=filter_robot.n_spheres,
                        planned=plan_horizon_steps, rows_per_step=rows_per_step)
    # **구 굵기는 시작할 때 크게 말한다.** 기본값이면 한 줄이고, 덮지 못하는 capsule 이 있으면
    # 여러 줄짜리 경고다 — `--exclude-links` 와 같은 성질의 flag 이므로 같은 크기로 말한다.
    report = constraint_robot.coverage_report()
    if (constraint_robot.coverage_shortfall or constraint_robot.capsule_trims
            or getattr(constraint_robot, "_spacing_capped", ())):
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


def wire_shadow_key() -> str:
    """로그에 찍을 **모드** 키 이름. `wire` 에서 가져온다 — 문자열을 두 곳에 박으면 갈라진다.

    `wire_reference_key()` 와 따로 있는 이유가 T9 의 요점이다: 예전에는 그 한 키가 데이터와
    모드를 겸했고, 그 겸직 때문에 closed loop 이 원본 청크를 실을 수 없었다.
    """
    from benchmark.trajopt import wire

    return wire.SHADOW


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
    ap.add_argument("--links", choices=LINK_GROUPS, default="arms",
                    help="제약을 걸 링크. arms(기본) 는 양팔과 손끝만 — 바퀴·베이스는 결정 "
                         "변수가 아니라 고칠 수 없는 위반을 상수로 깔아 실제 신호를 묻는다. "
                         "gripper 는 **손가락 넷만** (ee_finger_l1/l2/r1/r2, T7b 진단): 팔뚝·"
                         "손목·몸통·반대팔이 전부 제약에서 빠지므로 무엇이 미는지 한 번에 "
                         "갈리지만, 빠진 것은 무엇에 부딪혀도 아무도 막지 않는다. all 은 전신")
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
    ap.add_argument("--capsule-radius-scale-link", nargs="+", default=(), metavar="LINK=F",
                    help="**link 별** capsule 반지름 배율 (T7b). --capsule-radius-scale 는 팔 "
                         "전체에 걸리므로 손가락을 얇게 하면 팔뚝·반대팔도 같이 얇아진다 — 그 "
                         "굵기가 필요한 부분이 바로 거기다. 예: --capsule-radius-scale-link "
                         "gripper=0.35. 키는 link 이름이거나 --links 의 집합 이름(arms·gripper)"
                         "이고, 집합 이름을 허용하는 것이 손가락 넷을 손으로 적는 오타 위험을 "
                         "없애기 위한 것이다. 없는 이름이면 서버가 시작하지 않는다. 이름을 안 "
                         "적은 link 는 --capsule-radius-scale 값을 그대로 쓴다")
    ap.add_argument("--capsule-extent-link", nargs="+", default=(), metavar="LINK=Z",
                    help="**link 별 capsule 절단** (T8b). 단위는 미터이고, LINK=Z 는 link "
                         "frame 에서 z 이상만 남긴다 (RB-Y1 팔의 원위 방향이 −z 이므로 곧 원위 "
                         "절단면이다). LINK=Zmin:Zmax 로 구간을 줄 수도 있다. 예: "
                         "--capsule-extent-link link_left_arm_5=-0.10. 그 capsule 은 URDF 에서 "
                         "r 75 mm · L 250 mm 이고 축이 z ∈ [−0.225, +0.025] 라, 손목(0.0)·"
                         "FT sensor(−0.1087)·손바닥(−0.1548)·그리퍼 뿌리(−0.2278)를 **하나가 "
                         "통째로 삼킨다**. 자르면 그 구간에 구가 없어지므로 "
                         "`--links` 가 손바닥(ee_left/ee_right)을 포함하는지 확인하라 — "
                         "기본 집합은 포함한다. 굵기 배율과 **함께** 걸 수 있고 그것이 정상이다 "
                         "(자르기만 하면 잘린 끝의 구가 여전히 자기 반지름만큼 부푼다). 없는 "
                         "이름·capsule 없는 link·mm 로 적은 값은 서버가 시작하지 않는다")
    ap.add_argument("--no-self-collision", action="store_true",
                    help="쥔 물체 대 로봇 구 제약을 끈다 (T7b 진단). **기본은 켠 상태이고 그것이 "
                         "지금까지의 서버다.** 이 repo 에 로봇-로봇 쌍 검사는 원래 없으므로 "
                         "(N1), 이 블록이 여기 있는 자기 충돌 전부다 — 끄면 쥔 물체가 팔뚝·"
                         "토르소·반대팔을 통과해도 아무도 막지 않는다. 행은 남고 mask 만 0 이 "
                         "되므로 다시 켜는 것은 이 flag 를 빼는 것뿐이다 (행 수·희소성 불변)")
    ap.add_argument("--target-field-policy", choices=target_field_policy_choices(),
                    default="relax",
                    help="아직 **쥐지 않은** target 을 거리장에서 어떻게 다룰지 (T8b). "
                         "relax(기본) = 지금까지의 서버: target 은 필드에 그대로 있고 권한 있는 "
                         "link 의 마진만 완화한다. exclude-authorized = 권한 있는 link "
                         "(contact.contact_links) 의 질의점만 **target 이 빠진 계층**에 거리를 "
                         "묻는다 — 마진은 0 보다 작아질 수 없는데 성공한 grasp 는 손끝이 사과 "
                         "표면을 17.96 mm 관통하므로 마진으로는 닿지 않는다 (T7a). "
                         "exclude-all = 모든 제약 구가 그 계층에 묻는다 — **E1 이 되살아난다** "
                         "(몸통·전완·반대팔에게도 사과가 사라진다). 어느 쪽이든 table·crate 는 "
                         "그 계층에도 있으므로 보호가 유지되고, 쥔 뒤에는 정책이 돌지 않는다 "
                         "(그때 target 은 crate 이고 crate 는 빼면 안 된다)")
    ap.add_argument("--max-sphere-radius", type=float, default=None, metavar="R",
                    help="제약 모델 구 반지름의 **절대 상한** (m). 팽창까지 끝난 값에 걸린다 "
                         "(팔뚝 기본값 81.2 mm). `--capsule-radius-scale` 과 같은 성질이고, "
                         "덮지 못하면 시작 로그에 크게 찍는다")
    # --- 목적함수 가중치 (T9) ------------------------------------------------------------
    # **제약이 하나도 활성이 아닌 판에서도 TO 가 청크를 고친다** (실행되는 8 step 안에서 중앙값
    # 2.6°, 최대 8.8°). 그러면 남은 변형은 전부 이 넷이 만든 것이므로, 훑을 수 있어야 한다.
    # 지금까지는 CLI 가 없어 코드를 고쳐야만 실험할 수 있었다. **기본값은 여기 적지 않는다** —
    # `None` 이면 키를 안 넣고 `CostConfig` 의 값이 그대로 쓰인다 (`sphere_options` 와 같은 규약).
    ap.add_argument("--w-track", type=float, default=None, metavar="W",
                    help="정책 청크를 따라가는 항의 가중치 (기본 1.0). **내리면 TO 가 청크를 더 "
                         "자유롭게 바꾼다** — 잡기 동작이 빗나가는 원인을 이 축에서 찾는다")
    ap.add_argument("--w-smooth", type=float, default=None, metavar="W",
                    help="2 차 차분(jerk) 항의 가중치 (기본 0.05). 올리면 궤적이 매끄러워지는 "
                         "대신 청크의 급한 변화(파지 순간의 손목 꺾임)를 깎는다")
    ap.add_argument("--w-continuity", type=float, default=None, metavar="W",
                    help="앞 청크의 겹치는 꼬리와의 연속성 항 (기본 0.5). **0 이면 그 항이 "
                         "아예 없어진다** (`refiner` 가 그때 이전 청크를 안 쓴다)")
    ap.add_argument("--w-slack", type=float, default=None, metavar="W",
                    help="제약 위반 slack 의 선형 벌점 (기본 1e3). `w_track` 보다 커야 하고 "
                         "(안 그러면 최적화기가 위반을 사는 편이 싸다) config 가 그것을 검사한다")
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
    weights = cost_overrides(args)
    if weights and args.no_safe:
        ap.error("--w-* 는 최적화기의 목적함수를 고치는 flag 입니다 (--no-safe 는 최적화기를 "
                 "아예 안 돌립니다). 그대로 띄우면 목적함수를 바꿨다고 믿게 되는데, 실은 "
                 "정책 청크가 그대로 나가는 서버가 뜹니다")
    if args.target_field_policy != "relax" and args.no_safe:
        ap.error("--target-field-policy 는 안전 계층의 거리장 질의를 고치는 flag 입니다 "
                 "(--no-safe 는 그 계층을 아예 안 만듭니다). 그대로 띄우면 target 을 필드에서 "
                 "뺐다고 믿게 되는데, 실은 어떤 충돌 제약도 없는 서버가 뜹니다")
    if args.target_field_policy != "relax":
        # **서버가 뜨기 전에 막는다.** 이 두 조합은 첫 프레임에서 죽는데, 그때는 체크포인트
        # 두 벌이 이미 GPU 에 올라간 뒤이고 로컬은 이미 붙어 있다.
        if args.esdf_backend != "curobo":
            ap.error("--target-field-policy 는 target 없는 계층을 요구하고, 그 계층은 cuRobo "
                     "backend 만 만듭니다 (--esdf-backend curobo). legacy 로는 정책을 켰다고 "
                     "믿은 채 아무 일도 일어나지 않습니다")
        if not args.fine_voxel:
            ap.error("--target-field-policy 는 미세 계층 한 겹으로 만들어집니다. "
                     "--fine-voxel 0 은 단일 계층이므로 정책이 아무 일도 하지 않습니다")
    if args.no_self_collision and args.no_safe:
        ap.error("--no-self-collision 은 제약 모델을 고치는 flag 입니다 (--no-safe 는 그 모델을 "
                 "아예 안 만듭니다). 그대로 띄우면 자기 충돌을 껐다고 믿게 되는데, 실은 어떤 "
                 "충돌 제약도 없는 서버가 뜹니다")
    # **`--capsule-radius-scale-link` 의 형식 오류는 여기서 죽는다.** `sphere_options` 가
    # 파싱하므로 아래 호출이 그것을 겸하는데, 그 예외는 argparse 가 아니라 traceback 으로
    # 나가고 그때는 체크포인트가 이미 올라간 뒤다.
    try:
        options = sphere_options(args)
    except ValueError as exc:
        ap.error(str(exc))
    if args.no_safe and options:
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
                      # **끈 경우에만 남긴다.** 켠 실행의 기록을 T0 때와 같은 키 집합으로
                      # 두면서, 끈 실행은 반드시 기록으로 구별된다 — 자기 충돌이 꺼진 기록을
                      # 켜진 것과 나란히 읽는 것이 이 flag 의 가장 나쁜 실패다.
                      **({"self_collision": False} if args.no_self_collision else {}),
                      # **기본이 아닐 때만 남긴다.** 같은 이유다 — target 이 필드에서 빠진
                      # 기록을 빠지지 않은 것과 나란히 읽는 것이 이 flag 의 가장 나쁜 실패다.
                      **({"target_field_policy": args.target_field_policy}
                         if args.target_field_policy != "relax" else {}),
                      # **다듬는 창은 항상 남긴다.** T6f 에서 기본값이 32 → 실행 창으로
                      # 바뀌었으므로, 안 남기면 T6d 기록과 이 기록이 meta 로 구별되지 않는다 —
                      # 그리고 둘은 서로 다른 것을 재고 있다.
                      "plan_horizon": args.plan_horizon,
                      **({"sphere_options": sphere_options(args)}
                         if sphere_options(args) else {})})
            logging.info("recording AG3S constraint diagnostics to %s", recorder.run_dir)

        # **준 것만 넣는다.** 빈 dict 면 키가 없고, 그러면 `CostConfig` 기본값이 그대로다 —
        # 호출이 예전과 글자 그대로 같다는 뜻이다 (`sphere_options` 와 같은 규약).
        weights = cost_overrides(args)
        to_config = TrajOptConfig.from_dict({
            "collision": {"backend": "esdf", "esdf_margin": args.esdf_margin,
                          "use_support_planes": False},
            **({"cost": weights} if weights else {}),
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
        announce_cost_weights(weights)
        logging.info(
            "TO objective: w_track=%g w_smooth=%g w_continuity=%g w_slack=%g%s",
            to_config.cost.w_track, to_config.cost.w_smooth,
            to_config.cost.w_continuity, to_config.cost.w_slack,
            "" if weights else " (전부 기본값)")
        # **응답이 원본 청크를 함께 싣는다** (T9). shadow 모드와 **다른 것**이다 — 이것은
        # 기록에 싣는 것이고, shadow 는 로컬이 그것을 **실행하는** 모드다. 둘을 헷갈리면
        # "shadow 라고 적힌 closed loop" 이 남는다.
        logging.info(
            "response carries %r on every chunk (T9): the policy's ORIGINAL chunk rides next to "
            "the refined one so 'how much did TO change it' is measurable in closed loop too. "
            "This is NOT shadow mode — the robot still executes the refined chunk. Mode is "
            "declared separately as %r=%s. Cost: a planning record row grows from ~19.8 KB to "
            "~36.7 KB.", wire_reference_key(), wire_shadow_key(), bool(args.shadow))

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
                            constraint_sphere_options=sphere_options(args),
                            self_collision=not args.no_self_collision,
                            target_field_policy=args.target_field_policy,
                            plan_horizon_steps=horizon.planned,
                            rows_per_step=to_config.reduction.rows_per_step),
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
