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
아닌지를 플래그 하나로 가를 수 있다. (T39 부터 `SeededPolicy` 한 겹만 — 요청에 `policy_seed` 가 없으면
요청을 그대로 넘기는 감싸개다. `trajopt/policy_seed.py`.)

**정책 RNG seed (T39).** 요청에 seed (`ag3s/policy_seed`, `--no-safe` 는 맨 키 `policy_seed`) 가 있으면
그 요청의 `policy.infer` 직전에 정책 RNG 를 `jax.random.key(seed)` 로 다시 놓는다. 없으면 RNG 는 서버
시작 때의 `jax.random.key(0)` 에서 요청마다 이어 split 된다 (T39 전 동작). `wire.py` 머리말.

`--no-perception` (T27) 은 그 사이다: π0.5 → **TO 만** 돌리고 AG3S·depth·ESDF·attention 두 번째
사본·grasp latch 를 하나도 만들지 않는다 (`trajopt/to_only_policy.py`). `--no-limits` 는 TO 에서 joint
position box · 속도 · 가속도 행을 뺀다 (trust region 과 첫 스텝 anchor 는 남는다). 사용자 사다리 실험:

    E1 "TO only"  --no-perception --no-limits --w-smooth 0 --w-continuity 0
    E2            --no-perception
    E3            (flag 없음 = AG3S + 거리장, 충돌 제약은 gripper 에만 — `--links` 기본)

세 판 모두 로컬은 `pi05_infer.py --safe-remote --safe-gate off` 다 (판정은 기록만, HOLD 없음).

`--shadow` 는 **계산을 줄이지 않는다.** AG3S·ESDF·SQP·판정이 전부 돌고 `actions` 도 그대로
refined 다. 응답에 정책 **원본** 청크를 `actions_reference` 로 함께 실어, 로컬이 그것을
실행할 수 있게 하는 것뿐이다 (T5 — 수정이 여유거리를 나쁘게 만드는지를 로봇을 움직이기 전에
본다). 로컬도 `pi05_infer.py --safe-shadow` 로 켜야 하고, 짝이 안 맞으면 클라이언트가 즉시
죽는다 — 조용히 refined 를 실행하면 shadow 가 아닌데 shadow 라고 기록된다.
"""

from __future__ import annotations

import argparse
import logging
import math
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


#: `--links` 선택지. **첫 항목이 기본이다 — T27 부터 `gripper`** (사용자 판정 2026-09-28: "거리장 충돌제약은
#: gripper 에만, 나머지 link 은 전부 제외"). 손바닥 둘 + 손가락 넷(`GRIPPER_LINKS`)에만 제약이 걸린다.
#: `arms`(T27 전 기본) · `all` 은 옵션으로 남고, 고르면 시작 로그가 크게 말한다.
LINK_GROUPS: tuple[str, ...] = ("gripper", "arms", "all")

#: `--links` 의 기본값. `LINK_GROUPS[0]` 과 같다 — 두 곳에 적지 않는다.
DEFAULT_LINKS: str = LINK_GROUPS[0]


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

    `gripper`(기본, T27) = **손바닥 2 + 손가락 4** · `arms` = 양팔 14 link + 손가락 4 · `all` = 전신(`None`).

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


def announce_collision_switch(enabled: bool) -> None:
    """**충돌 제약이 꺼져 있으면 크게 말한다.** `--no-self-collision` 과 같은 수위다.

    그쪽은 쥔 물체 대 로봇 한 블록을 껐고, 이쪽은 **충돌 제약 전부**다. 조용히 이 설정으로 떠
    있는 것은 안전 계층이 꺼진 서버를 켜진 것처럼 보이게 하는 것이다.
    """
    if enabled:
        return
    logging.getLogger(__name__).warning(
        "!!! COLLISION CONSTRAINTS ARE OFF (--no-collision) !!!\n"
        "    최적화기는 충돌 행을 **하나도** 보지 않습니다 — 로봇이 무엇에 부딪혀도 아무도 막지 "
        "않습니다. 테이블·crate·사과·자기 자신 전부입니다.\n"
        "    TO 는 그대로 돕니다 (추적·jerk·연속성·limit·SQP). `--no-safe` 와 다릅니다.\n"
        "    **여유거리 측정은 계속 돕니다** — max_violation_m 은 참값이고, 궤적이 실제로 무엇을 "
        "했는지 말합니다. 다만 그것을 **강제하지 않았습니다.** 그 사실이 응답 notes 와 "
        "metrics['collision_enabled'] 에 실립니다.\n"
        "    **진단용입니다.** 되돌리는 방법은 이 flag 를 빼는 것뿐입니다.")


def announce_limits_switch(limits) -> None:
    """**joint limit 행이 꺼져 있으면 크게 말한다** (T27, `--no-limits`). `limits` 는 `LimitsConfig`.

    무엇이 남는지를 **같은 줄에** 말한다 — trust region 과 첫 스텝 anchor 는 SQP 장치라 남는다.
    그것을 안 적으면 *"limit 을 다 뺐는데 첫 스텝이 움직였다"* 를 읽을 근거가 없다.
    """
    off = [name for name in ("position", "velocity", "acceleration")
           if not getattr(limits, f"enforce_{name}")]
    if not off:
        return
    anchor = bool(getattr(limits, "enforce_velocity", True) or getattr(limits, "keep_anchor", False))
    logging.getLogger(__name__).warning(
        "!!! JOINT LIMIT ROWS ARE OFF (--no-limits): %s !!!\n"
        "    최적화기는 이 한계를 **보지 않습니다** — 청크가 로봇 한계를 넘어도 그대로 나갑니다.\n"
        "    남는 것: SQP trust region (sqp.trust_radius) · 첫 스텝 anchor %s.\n"
        "    기록의 limit_overshoot 는 로봇 자신의 한계로 계속 잽니다 (강제하지 않은 값).",
        ", ".join(off),
        "(|Q0 − q_now| ≤ v_max·dt — 남음)" if anchor else "(**없음**)")


def announce_no_perception() -> None:
    """**지각이 없는 서버로 떠 있으면 크게 말한다** (T27, `--no-perception`)."""
    logging.getLogger(__name__).warning(
        "!!! NO PERCEPTION (--no-perception): π0.5 → TO ONLY !!!\n"
        "    AG3S · depth · 점군 · ESDF · attention(두 번째 체크포인트 사본) · grasp latch 가 "
        "**하나도 없습니다.** 최적화기는 빈 세계를 봅니다 — 충돌 행 0, 충돌 판정 없음.\n"
        "    `--no-collision` 과 다릅니다 (그쪽은 행만 끄고 지각·거리장·판정이 전부 돕니다).\n"
        "    응답: actions = refined · actions_reference · to · safe=True · "
        "verdict_reasons=[no_perception].\n"
        "    **로컬은 `--safe-gate off` 여야 합니다** — 사유 게이트는 no_perception 을 HOLD 로 "
        "읽습니다 (검사되지 않은 청크).")


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


def announce_reference_payload() -> None:
    """**데이터 한 가지 사실만 말한다** — 응답이 정책 원본 청크를 함께 싣는다 (T9).

    **모드를 주장하지 않는다.** 처음 쓴 문구는 여기서 *"the robot still executes the refined
    chunk"* 라고 단정했는데, `--shadow` 로 띄운 서버에서는 그 문장이 거짓이다. 데이터를 설명하는
    줄이 모드를 단정하면 `announce_execution_mode` 와 반대를 말할 수 있고, T9 이 끊으려던 겸직이
    로그에서 되살아난다. 모드는 그 함수만 말한다 — 여기서는 그쪽을 가리키기만 한다.
    """
    logging.getLogger(__name__).info(
        "response carries %r on every chunk (T9): the policy's ORIGINAL chunk rides next to the "
        "refined one, in BOTH modes, so 'how much did TO change it' is measurable in closed loop "
        "too. This key is data, not a mode — which chunk the robot executes is the next line. "
        "Cost: a planning record row grows from ~19.8 KB to ~36.7 KB.", wire_reference_key())


def announce_execution_mode(shadow: bool) -> None:
    """**로봇이 무엇을 실행하는가.** 모드는 `shadow` 로만 말한다.

    T9 이 이 자리를 한 번 거짓으로 만들었다. 예전 문구는 closed loop 을
    *"the response carries the refined chunk only (no 'actions_reference' key)"* 라고 적었는데,
    T9 부터 원본 청크는 **두 모드 모두** 실리므로 앞 절반이 거짓이 됐다 — 같은 시작 로그 안에서
    두 줄이 반대를 말했다.

    원인은 문구가 아니라 **키 이름을 모드의 설명에 쓴 것**이다. 그러면 데이터 키가 다시 모드를
    겸하기 시작하고, T9 가 끊은 겸직이 로그에서 되살아난다. 그래서 여기서는 `actions_reference`
    를 **입에 올리지 않는다** — 그 키를 설명하는 줄은 따로 있다 (위의 "response carries ...").

    셋은 **각각 다른 사실**이고, 문장도 셋으로 갈라 적는다.

    1. 로봇이 무엇을 실행하는가 (refined / 정책 원본).
    2. 원본 청크가 기록에 실리는가 — **두 모드 모두 실린다.** 모드의 특징이 아니다.
    3. 로컬의 `--safe-shadow` 짝이 맞아야 하는가 — 안 맞으면 클라이언트가 즉시 죽는다.
    """
    log = logging.getLogger(__name__)
    if shadow:
        log.info(
            "SHADOW mode (%s=True): everything runs (AG3S · ESDF · SQP · verdict) and **the "
            "robot executes the POLICY chunk, not the refined one** — the local side makes that "
            "choice. The local side must run with --safe-shadow; a mismatched pair fails "
            "immediately on the client. Carrying the original chunk is NOT what makes this "
            "shadow: every response carries it in either mode (see the line above).",
            wire_shadow_key())
        return
    log.info(
        "closed-loop mode (%s=False): **the robot executes the REFINED chunk.** Local "
        "--safe-shadow will refuse to run against this server (the pair must match, and the "
        "client fails immediately). The policy's original chunk is still carried in every "
        "response for the record — that is data, not a mode (see the line above).",
        wire_shadow_key())


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
    if links != DEFAULT_LINKS:
        # **T27 에서 방향이 뒤집혔다.** 기본이 `gripper` 이므로 이제 크게 말하는 쪽은 `arms`·`all`
        # 이다 — 팔뚝·손목·몸통까지 제약에 걸리면 파지 접근이 그 행에 밀린다 (T6d·T7b 가 그것을
        # 쟀다). 사용자 판정(2026-09-28)이 gripper 이므로 그 밖은 비기본이다.
        log.warning(
            "!!! CONSTRAINT SCOPE IS %s, NOT THE DEFAULT %r !!!\n"
            "    제약이 걸리는 link: %s\n"
            "    **손 밖의 link(팔뚝·손목·%s)까지 거리장 제약이 걸립니다** — 파지 접근이 그 행에 "
            "밀려 청크가 바뀔 수 있습니다 (T6d·T7b). 사용자 판정(2026-09-28)의 기본은 gripper "
            "(손바닥 2 + 손가락 4) 입니다.",
            links.upper(), DEFAULT_LINKS, ", ".join(str(n) for n in constraint_links) or "(전신)",
            "몸통·바퀴·베이스" if links == "all" else "반대팔")
    else:
        log.info(
            "constraint scope: %s (기본, T27) — %s. **여기 없는 link(팔뚝·몸통·반대팔)는 무엇에 "
            "부딪혀도 아무도 막지 않습니다**; 자기 필터 모델은 그대로 전신입니다.",
            links, ", ".join(str(n) for n in constraint_links) or "(모델에서 읽지 못함)")
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


def announce_gripper_cover(report: dict | None) -> None:
    """T43 R — 덮개를 켰을 때만 (WARNING): 구 수 · link 별 수 · 못 덮은 표본. 꺼져 있으면 조용하다."""
    if not report:
        return
    per = ", ".join(f"{link}+{r['n_spheres']} (r {r['radius_mm'][0]:.1f}–{r['radius_mm'][1]:.1f} mm, "
                    f"미달 표본 {r['n_uncovered']}/{r['n_targets']})" if r.get("radius_mm")
                    else f"{link}+0" for link, r in report["links"].items())
    logging.warning(
        "constraint gripper cover (T43 R): 구 %d → %d (+%d 내접 덮개, 최대 틈 %.1f mm, 손 %s) — %s. "
        "ESDF 질의 · FK 비용이 구 수에 비례한다",
        report["n_base_spheres"], report["n_base_spheres"] + report["n_cover_spheres"],
        report["n_cover_spheres"], 1e3 * float(report["options"].get("max_gap", 0.0025)),
        report["options"].get("hands", "both"), per)


def gripper_cover_options(args) -> dict | None:
    """`--gripper-cover [HANDS]` · `--gripper-cover-max-gap MM` → `build_constraint_robot_model` 의
    `gripper_cover` (T43 R). 안 줬으면 `None` — 제약 모델이 예전과 글자 그대로 같다."""
    hands = getattr(args, "gripper_cover", None)
    gap_mm = getattr(args, "gripper_cover_max_gap", None)
    if hands is None:
        if gap_mm is not None:
            raise SystemExit("--gripper-cover-max-gap 는 --gripper-cover 와 함께 줘야 합니다 — "
                             "혼자서는 아무 일도 하지 않습니다")
        return None
    out: dict = {"hands": str(hands)}
    if gap_mm is not None:
        if not (math.isfinite(float(gap_mm)) and float(gap_mm) > 0):
            raise SystemExit(f"--gripper-cover-max-gap 는 0 보다 커야 합니다 (mm): {gap_mm}")
        out["max_gap"] = float(gap_mm) * 1e-3
    return out


def parse_self_filter_inflation(pairs: Sequence[str]) -> dict[str, float]:
    """`["gripper=0.01"]` → `{"gripper": 0.01}` (T19). 빈 입력이면 빈 dict (예전과 같은 필터).

    키는 **link 이름이거나 집합 이름** (`arms` · `gripper`) 이고 **펼치지 않고 그대로** config 로
    넘긴다 — 집합을 펼치고 우선순위(`arms` < `gripper` < link)를 매기고 모르는 이름을 거절하는
    것은 `robot_filter.resolve_self_filter_inflation` 한 곳이다. 여기서 한 번 더 펼치면 두 곳이
    되고, 갈라지는 날 로그가 말하는 필터와 실제 필터가 달라진다.

    `all` 은 받지 않는다: 전신 값은 `pointcloud.self_filter_inflation` 그 자체다
    (`--capsule-radius-scale-link` 와 같은 규약). **단위는 미터다.** 형식이 아니거나 음수면 여기서
    죽는다 — 조용히 버리면 손가락 여유를 줄였다고 믿은 채 50 mm 로 뜬다.
    """
    out: dict[str, float] = {}
    for item in pairs or ():
        text = str(item)
        if "=" not in text:
            raise ValueError(
                f"--self-filter-inflation-link 는 LINK_OR_GROUP=미터 형식입니다: {text!r} "
                f"(예: gripper=0.01, link_left_arm_6=0.03)")
        name, _, value = text.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"--self-filter-inflation-link {text!r}: link 이름이 비었습니다")
        if name == "all":
            raise ValueError(
                "--self-filter-inflation-link all=... 은 받지 않습니다. 전신 값은 "
                "pointcloud.self_filter_inflation 그 자체입니다")
        try:
            margin = float(value)
        except ValueError:
            raise ValueError(
                f"--self-filter-inflation-link {text!r} 의 값이 숫자가 아닙니다 (미터)") from None
        if not margin >= 0.0:
            raise ValueError(
                f"--self-filter-inflation-link {text!r}: inflation 은 0 이상이어야 합니다 (미터)")
        out[name] = margin
    return out


def self_filter_options(args) -> dict:
    """self-filter flag 중 **실제로 준 것만** 담은 `pointcloud` 조각 (T19).

    `sphere_options` 와 같은 계약이다 — 안 준 flag 는 키가 아예 없고, 빈 dict 면
    `AG3SConfig.from_dict` 호출이 예전과 글자 그대로 같다. 기본값(0.05 · 빈 표 · guard 0)을 여기
    다시 적으면 `PointCloudConfig` 와 두 곳이 된다.
    """
    given = {
        "self_filter_inflation_by_link":
            parse_self_filter_inflation(getattr(args, "self_filter_inflation_link", ())) or None,
        "self_filter_target_guard_radius": getattr(args, "self_filter_target_guard", None),
    }
    out = {k: v for k, v in given.items() if v is not None}
    guard = out.get("self_filter_target_guard_radius")
    if guard is not None and not float(guard) >= 0.0:
        raise ValueError(
            f"--self-filter-target-guard 는 0 이상이어야 합니다 (미터, 0 = 끔): {guard!r}")
    return out


def announce_self_filter(settings: dict, *, cli: dict | None = None) -> None:
    """**자기 필터가 실제로 쓰는 값을 시작 로그에 찍는다** (T19, 지침 §3.2).

    `settings` 는 `AG3S.self_filter_settings()` — 집합을 펼치고 우선순위를 매긴 **뒤의** link 별
    표다. 지침 §3.2 가 짚은 것은 `rby1_three_camera.yaml` 의 0.02 가 이 서버에 **닿지 않는다**는
    사실이 어디에도 안 보였다는 것이므로, 그 문장을 여기서 말한다.

    기본(재정의 없음 · guard 끔)이면 info 한 줄, 아니면 warning 여러 줄이다 — `--sphere-*` 와 같은
    수위다. MuJoCo 없이 테스트할 수 있게 `build_ag3s` 밖에 둔다.
    """
    log = logging.getLogger(__name__)
    cli = dict(cli or {})
    table = dict(settings.get("inflation_effective") or {})
    base = float(settings.get("inflation_default", 0.0))
    guard = float(settings.get("target_guard_radius", 0.0))
    by_value: dict[float, list[str]] = {}
    for link, value in table.items():
        by_value.setdefault(float(value), []).append(str(link))
    rows = "\n".join(
        f"      {value * 1000:6.1f} mm : {len(links):2d} link — {', '.join(links)}"
        for value, links in sorted(by_value.items()))
    source = ("CLI " + ", ".join(f"{k}={v}" for k, v in sorted(cli.items()))
              if cli else "코드 기본값 (PointCloudConfig)")
    head = (f"self-filter: {settings.get('n_spheres')} 구 · 기본 inflation "
            f"{base * 1000:.1f} mm · target guard "
            + (f"{guard * 1000:.1f} mm (확정 target centroid 근방은 inflation 0 — 로봇 구 "
               "내부는 여전히 지운다)" if guard > 0.0 else "끔")
            + f" · 출처: {source}. configs/*.yaml 의 pointcloud 값은 이 서버에 닿지 않는다 "
              "(build_ag3s 가 config 를 직접 만든다)")
    if not settings.get("inflation_by_link") and guard <= 0.0:
        log.info("%s", head)
        return
    log.warning("!!! SELF-FILTER IS NOT THE SINGLE-VALUE DEFAULT !!!\n    %s\n"
                "    실제 유효 inflation (link 별, 집합 펼친 뒤):\n%s", head, rows)


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


def subtask_config_section(subtask_gate: bool | None,
                           subtask_probe_path: str | None) -> dict:
    """`build_ag3s` 가 config 에 얹는 subtask 조각 (SUBTASK-c · SUBTASK-g).

    `subtask_gate` 가 bool 이면 **켜든 끄든** `clustering.subtask_gate` 로 싣고 (기본이 바뀐 뒤에도
    서버 동작이 config 기본값에 기대지 않게), None 이면 싣지 않는다 (= config 기본). probe 경로는
    줄 때만 싣는다. 둘 다 없으면 `{}` — config 는 글자 그대로다.
    """
    clustering = {
        **({"subtask_gate": bool(subtask_gate)} if subtask_gate is not None else {}),
        **({"subtask_probe_path": str(subtask_probe_path)} if subtask_probe_path else {})}
    return {"clustering": clustering} if clustering else {}


def t43t_identity_section(args) -> dict:
    """T43 T fix 3 — `clustering` keys for the given flags only (none given = `{}`, config as is)."""
    out = {}
    if getattr(args, "closing_keep_geometry", False):
        out["closing_keep_geometry"] = True
    if getattr(args, "latched_identity_hold", False):
        out["latched_identity_hold"] = True
    if getattr(args, "refused_identity_release", None) is not None:
        out["refused_identity_release_frames"] = int(args.refused_identity_release)
    if getattr(args, "supported_max_bottom_mm", None) is not None:
        out["supported_max_bottom"] = float(args.supported_max_bottom_mm) / 1000.0
    return out


def build_ag3s(model_xml: str, *, voxel: float, range_max: float, links: str,
               backend: str = "legacy", fine_voxel: float | None = None,
               tsdf_voxel: float | None = None,
               attached_sign_threshold: float = 1.5,
               max_field_age_sec: float | None = None,
               exclude_links: Sequence[str] = (),
               constraint_sphere_options: dict | None = None,
               gripper_cover: dict | None = None,
               self_collision: bool = True,
               target_field_policy: str = "relax",
               plan_horizon_steps: int | None = None,
               rows_per_step: int | None = None,
               self_filter: dict | None = None,
               subtask_gate: bool | None = None,
               subtask_probe_path: str | None = None,
               held_body_cover: bool = False,
               identity: dict | None = None,
               held_free_observed_pad: float | None = None):
    """서버가 쓸 AG3S. 자기 필터는 전신, 제약은 `links` (T27 부터 기본 `gripper` — 손바닥 + 손가락).

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

    `links="gripper"` 는 **손바닥 둘 + 손가락 넷만** 제약에 남긴다 (T7b 진단 → T27 기본). `arms` ·
    `all` 이면 `announce_diagnostic_scope` 가 크게 찍는다. `self_collision=False` 는 쥔 물체 대 로봇
    구 블록을 끈다 — 이 repo 의 자기 충돌은 그 블록 하나뿐이다.

    `self_filter` 는 `self_filter_options(args)` — **준 것만** `pointcloud` 에 얹는다 (T19). 비면
    `pointcloud` 는 예전과 글자 그대로 `{"range_max": ...}` 이다. 실제 유효 inflation 표는
    `announce_self_filter` 가 찍는다.

    `subtask_gate` (SUBTASK-g): bool 을 주면 **켜든 끄든 늘** `clustering.subtask_gate` 로 싣는다 —
    기본이 off → on 으로 바뀌었으므로, 서버가 띄운 gate 가 config 기본값에 기대면 기본이 다시
    바뀌는 날 서버 동작이 말없이 바뀐다. `main` 은 늘 bool (`resolve_subtask_gate`) 을 넘긴다.
    None (함수 기본) 이면 키를 넣지 않는다 = config 기본 (on). `subtask_probe_path` 는 SUBTASK-c
    그대로 **줄 때만** 얹는다 (안 주면 동봉 asset).
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
    # T43 R — 덮개는 준 것만 (`None` 이면 인자를 안 넘긴다 = 예전 호출 그대로).
    cover = {"gripper_cover": gripper_cover} if gripper_cover else {}
    constraint_robot = build_constraint_robot_model(
        scene, link_filter=constraint_link_filter(links), sphere_options=spheres, **cover)
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
                                                       sphere_options=spheres, **cover)
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
        # self-filter 재정의는 **준 것만** 얹는다 (T19) — 안 주면 이 dict 는 예전 그대로다.
        "pointcloud": {"range_max": range_max, **dict(self_filter or {})},
        "esdf": esdf,
        **({"timing": timing} if timing else {}),
        # **기본인 경우에는 키를 아예 넣지 않는다.** 기본값을 여기 다시 적으면
        # `ConstraintConfig` 와 두 곳이 되고, 갈라지는 날 갈라진 쪽이 안전 판정이다
        # (`sphere_options` 와 같은 규약). 두 flag 가 같은 section 을 쓰므로 한 dict 로 모은다.
        **({"constraint": constraint_section} if constraint_section else {}),
        # SUBTASK-g — gate 는 bool 이면 켜든 끄든 명시 (위 docstring). probe 경로는 SUBTASK-c
        # 규약 그대로: 줄 때만.
        **subtask_config_section(subtask_gate, subtask_probe_path),
    })
    if identity:
        # T43 T — the identity / closing-geometry keys, only when a flag gave them.
        config = config.with_overrides({"clustering": dict(identity)})
    logging.info("AG3S: self-filter %d spheres, constraints %d spheres (%s)",
                 filter_robot.n_spheres, constraint_robot.n_spheres, links_label)
    announce_gripper_cover(getattr(constraint_robot, "gripper_cover_report", None))
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
    ag3s = AG3S(config, robot_model=filter_robot, constraint_robot_model=constraint_robot,
                attached_parent_links=parents)
    # 생성 **뒤에** 찍는다 — 모르는 link 이름은 생성자가 거절하므로, 여기 도달한 표는 실제로
    # `robot_sphere_mask` 에 들어가는 값이다.
    announce_self_filter(ag3s.self_filter_settings(), cli=dict(self_filter or {}))
    if held_body_cover:
        # T43 R2 — 쥔 과일 질의 구 = 몸통 덮개 (바닥 반구 포함), J1 lift 는 받침 가까이에서만.
        ag3s.held_body_cover = True
        logging.warning("AG3S held query spheres: body cover (T43 R2) — slip %.1f mm, J1 lift "
                        "only within band + %.1f mm of the support",
                        ag3s.held_body_slip_m * 1000.0, ag3s.held_lift_hysteresis_m * 1000.0)
    if identity:
        logging.warning("AG3S identity (T43 T fix 3): %s", ", ".join(
            f"{k}={v}" for k, v in sorted(identity.items())))
    if held_free_observed_pad is not None:
        # T43 T fix 2 — the pre-grasp observation, freed at its place on every held frame.
        ag3s.held_free_observed_pad = float(held_free_observed_pad)
        logging.warning("AG3S held_free: + the manipulated object's last pre-grasp observation "
                        "(%.1f mm pad, at its place) on every held frame (T43 T fix 2)",
                        ag3s.held_free_observed_pad * 1000.0)
    return ag3s


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
            "아니면 --links gripper(기본) 나 arms 를 쓰십시오")
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
    """로그에 찍을 **데이터** 키 이름. `wire` 에서 가져온다 — 문자열을 두 곳에 박으면 갈라진다.

    T9 부터 이 키는 선택 키가 아니다 (두 모드 모두 실린다). 그리고 **모드를 설명하는 문장에는
    쓰지 않는다** — 그 자리에 키 이름이 나오면 데이터가 다시 모드를 겸하기 시작한다.
    """
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
    ap.add_argument("--links", choices=LINK_GROUPS, default=DEFAULT_LINKS,
                    help="거리장 충돌 제약을 걸 링크. **gripper(기본, T27 — 사용자 판정 2026-09-28)** "
                         "= 손바닥 둘 + 손가락 넷 (ee_left/ee_right, ee_finger_l1/l2/r1/r2): 팔뚝·"
                         "손목·몸통·반대팔은 제약에서 빠지고, 빠진 것은 무엇에 부딪혀도 아무도 "
                         "막지 않는다. arms = 양팔과 손끝 (T27 전 기본) · all = 전신 — 둘 다 시작 "
                         "로그가 크게 말한다. 자기 필터 모델(전신)은 이 값과 무관하다")
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
    ap.add_argument("--obstacle-margin", type=float, default=0.0, metavar="M",
                    help="T41 a — **target 이 아닌 장애물에만** 붙는 여유거리 (m). 기본 0 = 꺼짐 "
                         "(동작이 예전과 같다). 켜면 손 구 행 중 최근접 표면이 조작 대상(사과)도 "
                         "지지면(테이블·바닥)도 아닌 행만 `max(--esdf-margin, M)` 을 받는다. 쥔 "
                         "물체 행·target 행은 0 그대로. 질의 구는 키우지 않는다. 시험값 0.010")
    ap.add_argument("--target-volume-exempt", action="store_true",
                    help="T43 T step 2 — 최근접 표면점이 조작 대상의 carve-out 공 (target-free 계층이 "
                         "물체로 보는 부피) 안인 손 구 행은 target 행이다: --obstacle-margin 을 안 받고 "
                         "sweep 경로 행에서도 빠진다. 기본 off")
    ap.add_argument("--obstacle-margin-support", action="store_true",
                    help="T41 a — 지지면(테이블·바닥)에도 --obstacle-margin 을 건다. 기본은 "
                         "제외 (파지하려면 손가락이 테이블 가까이 가야 한다). ablation 용")
    ap.add_argument("--rest-start", action="store_true",
                    help="T41 b — 직전 청크가 통째로 HOLD 였으면 (exec_feedback n_exec=0, 로봇이 "
                         "q_now 에 정지) 그 청크의 TO 계획을 **정지에서 출발**시킨다: 모든 스텝을 "
                         "|Q_k − q_now| ≤ Σ_{i≤k} min(e0 + i·a_max·dt², v_max·dt) 로 묶는다. 기본 "
                         "꺼짐 = 예전 anchor (첫 스텝 |Q0 − q_now| ≤ v_max·dt = 10.8°) 그대로. "
                         "HOLD 가 아닌 청크는 켜도 바뀌지 않는다")
    ap.add_argument("--rest-start-tolerance", type=float, default=None, metavar="RAD",
                    help="T41 b — --rest-start 의 첫 스텝 문턱 e0 (rad, 관절 공통). 기본 = 관절마다 "
                         "a_max·dt² (0.040 rad = 2.29°, 정지에서 한 주기에 닿는 거리). 0 이면 "
                         "Q0 = q_now. v_max·dt 보다 크면 그 값에서 잘린다")
    ap.add_argument("--servo-model", action="store_true",
                    help="T43 Q (1) — TO 가 위치 servo 의 응답 (제어 대상 MJCF 의 kp · kv · damping · "
                         "armature 에서 유도, 속도는 exec_feedback 으로 추정) 으로 로봇이 **실제로 "
                         "지날 자세** 를 예측하고, 각 제어 행 끝의 예측 자세에 충돌 행을 더한다. "
                         "기본 꺼짐 = 예전 그대로 (명령 waypoint 만 검사)")
    ap.add_argument("--sweep-check", action="store_true",
                    help="T43 Q (2) — waypoint 사이 연속 충돌 검사. --servo-model 이 없으면 관절 공간 "
                         "보간, 있으면 servo 예측 경로의 sub-step. 질의 구가 검사 사이에 "
                         "--sweep-max-mm 이상 움직이지 않게 행마다 자른다. 기본 꺼짐")
    ap.add_argument("--sweep-max-mm", type=float, default=None, metavar="MM",
                    help="T43 Q — --sweep-check 의 검사 자세 사이 최대 이동 (mm). 기본 12 "
                         "(24 mm 막대의 절반)")
    ap.add_argument("--sweep-max-samples", type=int, default=None, metavar="N",
                    help="T43 Q — --sweep-check 의 제어 행당 조각 수 상한. 기본 16")
    ap.add_argument("--sweep-rows-per-step", type=int, default=None, metavar="N",
                    help="T43 Q — 중간 자세 행의 QP 예산 (제어 행마다). 기본 24. --servo-model 또는 "
                         "--sweep-check 와 함께만")
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
    ap.add_argument("--no-collision", action="store_true",
                    help="**충돌 제약 행만 끈다** (사용자 ablation, T15). TO 는 그대로 돈다 — "
                         "추적·jerk(w_smooth)·연속성·limit·SQP·trust region 전부. `--no-safe` 와 "
                         "**다르다**: 그쪽은 TO 를 아예 안 돌려 정책 청크가 그대로 나간다. "
                         "판정하려는 것은 '충돌을 다 빼도 못 잡으면 궤적최적화의 문제인가' 이고, "
                         "`--w-smooth 0` 과 함께 주면 최적해가 **reference 그 자체**여야 한다 — "
                         "아니면 optimizer 자체(SQP 수렴·limit)의 문제다. "
                         "**이 설정에서는 로봇이 무엇에 부딪혀도 아무도 막지 않는다.** "
                         "여유거리 측정은 계속 돌아가므로 max_violation_m 은 참값이고, 충돌이 "
                         "꺼졌다는 사실이 notes 와 기록에 실린다")
    ap.add_argument("--no-limits", action="store_true",
                    help="**TO 에서 joint limit 행을 뺀다** (T27 진단). joint position box · 스텝 사이 "
                         "속도 · 가속도 행이 없어진다. **남는 것**: SQP trust region 과 첫 스텝 "
                         "anchor (|Q0 − q_now| ≤ v_max·dt — 로봇이 지금 있는 곳에서 계획을 시작하게 "
                         "하는 장치). 추적 항만 두면 (`--w-smooth 0 --w-continuity 0`) 최적해가 "
                         "reference 그 자체다 — reference 가 anchor 를 넘지 않는 한. 기록의 "
                         "limit_overshoot 는 로봇 자신의 한계로 계속 잰다")
    # --- joint limit 출처 · SQP 예산 (T31) -----------------------------------------------
    ap.add_argument("--limits-source", choices=("model_xml", "urdf", "config"),
                    default="model_xml",
                    help="TO 의 joint **position** 범위를 어디서 읽나 (T31 G1). model_xml(기본) = "
                         "--model-xml 의 jnt_range (시뮬레이터가 쓰는 범위, actuator ctrlrange 와 "
                         "대조) — 정책이 학습된 로봇이다. urdf = T31 전 동작 (URDF <limit>; arm_6 "
                         "±2.705 로 MJCF ±2.967 보다 좁다). config = --joint-ranges 파일의 표 (실기). "
                         "속도·가속도 한계는 셋 모두 URDF 다 (MJCF 에 없다). 시작 로그에 관절별 "
                         "URDF 대 사용값 표가 찍힌다")
    ap.add_argument("--joint-ranges", default=None, metavar="FILE",
                    help="--limits-source config 의 표: YAML/JSON {joint_name: [lower, upper]} (rad). "
                         "TO 가 움직이는 관절은 전부 적어야 한다")
    ap.add_argument("--sqp-min-iterations", type=int, default=None, metavar="N",
                    help="벽시계 예산이 걸리기 전에 **반드시** 도는 SQP 반복 수 (T31 G2; 기본 "
                         "`SqpConfig.min_iterations` = 3 = max_iterations). 1 = T31 전 동작 (첫 반복 뒤 곧바로 예산 "
                         "검사). 실기처럼 제어 루프가 TO 를 기다리지 않는 곳에서는 1 로 둔다")
    ap.add_argument("--sqp-max-iterations", type=int, default=None, metavar="N",
                    help="SQP 반복 상한 (기본 `SqpConfig.max_iterations`)")
    ap.add_argument("--sqp-time-budget-ms", type=float, default=None, metavar="MS",
                    help="min 반복 뒤의 벽시계 예산 (기본 `SqpConfig.time_budget_ms`)")
    ap.add_argument("--no-limit-projection", action="store_true",
                    help="T31 G2-ii 를 끈다: limit 만 어긴 reference 에 받아들여진 후보가 없을 때 "
                         "최소 투영 대신 T24 규칙대로 거절된 QP 후보(best_unaccepted)를 반환한다 "
                         "(ablation)")
    ap.add_argument("--no-perception", action="store_true",
                    help="**π0.5 → TO 만** (T27, 사용자 사다리 실험 E1·E2). AG3S·depth·ESDF·"
                         "attention 두 번째 사본·grasp latch 를 **하나도** 만들지 않는다 — "
                         "`--no-collision`(행만 끄고 지각·판정은 돈다) 과 다르다. 응답은 같은 "
                         "와이어: actions=refined · actions_reference · to · safe=True · "
                         "verdict_reasons=[no_perception]. 로컬 사유 게이트는 no_perception 을 "
                         "HOLD 로 읽으므로 **로컬은 `--safe-gate off` 여야 한다**. "
                         "`--record-constraints DIR` 이면 청크마다 reference·refined·관절별 편차·"
                         "limit overshoot·SQP metrics 를 남긴다")
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
    ap.add_argument("--gripper-cover", nargs="?", const="both", default=None,
                    choices=("both", "left", "right"), metavar="HANDS",
                    help="T43 R — 제약 모델의 손바닥 · 손가락에 **내접 덮개 구**를 더한다 "
                         "(collision mesh 바깥 면을 --gripper-cover-max-gap 안으로, 반지름은 그 link "
                         "의 지금 구 최대 이하, 파지 면 쪽으로는 지금 구보다 안 나온다). 값 없이 주면 "
                         "both. 기본 꺼짐 = 예전 모델 그대로. 구 수가 크게 늘어 TO 시간이 는다 (T43R.impl)")
    ap.add_argument("--gripper-cover-max-gap", type=float, default=None, metavar="MM",
                    help="T43 R — 덮개 목표: 바깥 면에서 구 합집합까지 최대 거리 (mm, 기본 2.5). "
                         "--gripper-cover 와 함께만")
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
    # SUBTASK-g — gate 는 **기본 on** 이다. 세 값: None(안 줌 = 기본 on) · True(`--subtask-gate`
    # 명시) · False(`--no-subtask-gate`). 명시와 기본을 구별해야 `--no-attention` 등과의 조합에서
    # "명시 → 오류, 기본 → gate 끄고 한 줄" 을 가를 수 있다 (`reject_bad_flag_combinations`,
    # `resolve_subtask_gate`). 둘을 같이 주면 argparse 가 거절한다.
    gate = ap.add_mutually_exclusive_group()
    gate.add_argument("--subtask-gate", dest="subtask_gate", action="store_const", const=True,
                      default=None,
                      help="SUBTASK-c: subtask label (kv_L4 probe, attention 서버가 싣는다) 이 확정 "
                           "place/home 이면 AG3S 가 새 조작 대상을 채택하지도 바꾸지도 않는다 "
                           "(attention target 을 carve 하지 않는다). PLACED 뒤 home 이면 놓인 대상을 "
                           "푼다. SUBTASK-g 부터 **기본 on** — 이 flag 는 호환용이고, 명시하면 "
                           "label 이 없는 조합 (--no-safe · --no-attention · --no-perception) 을 "
                           "오류로 거절한다")
    gate.add_argument("--no-subtask-gate", dest="subtask_gate", action="store_const", const=False,
                      help="subtask gate 를 끈다 (SUBTASK-g 전 기본) — label 은 기록만 하고 결정은 "
                           "label 과 무관하다")
    ap.add_argument("--subtask-probe", default=None, metavar="NPZ",
                    help="subtask probe asset (`clustering.subtask_probe_path`). 안 주면 동봉 "
                         "benchmark/ag3s/asset/subtask_probe/kv_L4_v1.npz")
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
    ap.add_argument("--self-filter-inflation-link", nargs="+", default=(),
                    metavar="LINK_OR_GROUP=M",
                    help="**자기 필터** 의 link 별 inflation (미터, T19). 키는 link 이름 또는 "
                         "arms · gripper (우선순위 arms < gripper < link). 적지 않은 link 는 "
                         "pointcloud.self_filter_inflation (코드 기본 0.05) 을 쓴다. 안 주면 "
                         "지금과 같다. 제약 모델의 --sphere-* 와 다른 모델이다. 예: "
                         "--self-filter-inflation-link gripper=0.01")
    ap.add_argument("--self-filter-target-guard", type=float, default=None, metavar="R",
                    help="직전 프레임에 확정한 target centroid 반경 R (미터) 안에서는 자기 필터가 "
                         "inflation 0 (구 실제 반지름) 으로만 지운다 — 예외가 아니다, 로봇 구 "
                         "내부 점은 여전히 지운다 (T19). 안 주면 코드 기본값 (0 = 끔)")
    ap.add_argument("--held-body-cover", action="store_true",
                    help="T43 R2 — 쥔 과일의 질의 구가 몸통 전체 (바닥 반구 포함) 를 덮는다: 맞춘 구 + "
                         "받침면까지 내린 바닥 구 + 미끄럼 5 mm. J1 lift 는 몸통이 받침면 + 띠 + "
                         "10 mm 안일 때만 (파지 · 들어 올림 · 내려놓기). 기본 off = T43 전과 같다")
    # --- T43 T ---------------------------------------------------------------------------
    ap.add_argument("--closing-keep-geometry", action="store_true",
                    help="T43 T step 2 — 닫는 동안 (closing, attach 전) 조작 대상 기하를 그 프레임 관측으로 "
                         "바꾸지 않는다: 닫기 전 기하 (물체 전체) 가 carve-out 과 attach snapshot 에 남는다. "
                         "기본 off (T30: 동결 중 기하가 관측을 따라감)")
    ap.add_argument("--latched-identity-hold", action="store_true",
                    help="T43 T 3(a) — grasp latch 가 pick 대상을 잠근 동안 (LATCHED · CLOSING) 조작 대상 "
                         "정체는 '안 보임' 만으로 바뀌지 않는다: 물체가 사라졌다는 증거 (제자리 cluster 가 "
                         "다르고 손이 비켜 있음) 나 손이 도전자로 간 증거가 있을 때만 switch 를 센다. "
                         "기본 off")
    ap.add_argument("--refused-identity-release", type=int, default=None, metavar="N",
                    help="T43 T 3(b) — exclusion 관문이 조작 대상을 목적지 겹침으로 N 프레임 연속 "
                         "거절하면 그 정체를 놓고 first 로 다시 고른다. 기본 off")
    ap.add_argument("--supported-max-bottom-mm", type=float, default=None, metavar="MM",
                    help="T43 T 3(c) — 후보 cluster 의 가장 낮은 점이 받침면 위 MM 보다 높고 손에 "
                         "있지도 않으면 (손 구 hand_occlusion_reach 밖) admissible 이 아니다 (허들 막대 "
                         "윗부분 · crate 테두리). 받침면 아래에만 있는 것도. 기본 off. 근거값 60")
    ap.add_argument("--held-free-observed-mm", type=float, default=None, metavar="MM",
                    help="T43 T fix 2 — 쥔 동안 매 프레임, 파지 직전 관측한 조작 대상 점들의 **그 자리** "
                         "(base frame, 손을 따라가지 않음) 에서 MM 안의 TSDF voxel 을 비운다 (받침면 + 1 "
                         "voxel 위만, 목적지 근처 제외). cuRobo backend 만. 기본 off. 근거값 15")
    ap.add_argument("--static-geometry", default="none", metavar="none|auto|PATH",
                    help="아는 고정 기하(벽·선반·테이블·바닥)를 해석적 채널에 싣는다 — 거리장이 "
                         "min(복셀, 해석적) 을 답해 미관측·격자 밖의 낙관을 없앤다 (E4·N2). "
                         "none(기본) = 지금과 같다. auto = --model-xml 에서 뽑는다(시뮬). "
                         "PATH = benchmark.ag3s.fields.static_scene 이 쓴 JSON(실기 — MuJoCo 불필요). "
                         "주의: --links all 과 함께 쓰면 바닥이 바퀴·베이스에 못 푸는 행을 "
                         "상수로 깐다 (실측 base -342 mm)")
    return ap


def _label_unavailable(args) -> bool:
    """이 조합에서는 subtask label 이 AG3S 에 오지 않는다 (안전 계층 · attention 서버 · 지각 중 하나가 꺼짐)."""
    return bool(args.no_safe or args.no_attention or getattr(args, "no_perception", False))


def resolve_subtask_gate(args) -> str | None:
    """SUBTASK-g — `args.subtask_gate` 를 bool 로 확정한다 (`reject_bad_flag_combinations` **뒤에**).

    None(안 줌) = 기본 on. 단 label 이 오지 않는 조합 (`_label_unavailable`) 이면 끈다 — label 이
    없으면 gate 는 어차피 아무것도 하지 않고, 켜 둔 채 뜨면 기록 (`subtask.gate`) 이 켠 실행처럼
    읽힌다. 그때 기동 로그에 남길 한 줄을 돌려준다 (나머지는 None). 명시한 True 와 그 조합은
    `reject_bad_flag_combinations` 가 이미 거절했다. 두 번 불러도 같다 (bool 은 그대로 둔다).
    """
    if args.subtask_gate is not None:
        args.subtask_gate = bool(args.subtask_gate)
        return None
    if _label_unavailable(args):
        args.subtask_gate = False
        off = [f for f, on in (("--no-safe", args.no_safe), ("--no-attention", args.no_attention),
                               ("--no-perception", getattr(args, "no_perception", False))) if on]
        return (f"subtask gate: off ({' · '.join(off)} — subtask label 이 오지 않으므로 기본 on 인 "
                f"gate 를 끈다. 명시한 --subtask-gate 였다면 오류)")
    args.subtask_gate = True
    return None


def reject_bad_flag_combinations(ap: argparse.ArgumentParser, args) -> None:
    """안전 계층을 끈 채 그 계층의 flag 를 준 조합을 시작할 때 막는다.

    아무 일도 안 하는 flag 를 받아 놓고 뜨면, 로그를 읽는 사람은 그것이 효과가 있었다고 믿는다.
    """
    # T31 — 표를 줬는데 다른 출처가 쓰이면 그 실행은 적힌 표로 돈 것처럼 읽힌다.
    if getattr(args, "joint_ranges", None) and getattr(args, "limits_source", None) != "config":
        ap.error("--joint-ranges 는 --limits-source config 일 때만 읽힙니다 "
                 f"(지금 {getattr(args, 'limits_source', None)!r}). 둘을 같이 주십시오")
    if getattr(args, "limits_source", None) == "config" and not getattr(args, "joint_ranges", None):
        ap.error("--limits-source config 에는 --joint-ranges FILE 이 필요합니다")
    if args.shadow and args.no_safe:
        ap.error("--shadow 는 안전 계층이 돌아야 뜻이 있습니다 (--no-safe 는 그것을 끕니다). "
                 "shadow 는 '전부 계산하되 수정을 로봇에 보내지 않는' 실행이므로, 계산이 없으면 "
                 "그림자로 둘 것도 없습니다")
    if args.exclude_links and args.no_safe:
        ap.error("--exclude-links 는 제약 모델을 고치는 flag 입니다 (--no-safe 는 그 모델을 "
                 "아예 안 만듭니다). 그대로 띄우면 flag 가 아무 일도 안 하는데 link 를 뺐다고 "
                 "믿게 됩니다")
    if args.no_collision and args.no_safe:
        ap.error("--no-collision 은 **최적화기의 충돌 행만** 끄는 flag 입니다 (--no-safe 는 "
                 "최적화기를 아예 안 돌립니다). 둘을 같이 주면 뜻이 없습니다 — TO 가 안 도는데 "
                 "TO 의 제약을 끄는 것이므로, 그대로 띄우면 ablation 을 했다고 믿게 되는데 실은 "
                 "정책 청크가 그대로 나가는 서버가 뜹니다")
    weights = cost_overrides(args)
    if weights and args.no_safe:
        ap.error("--w-* 는 최적화기의 목적함수를 고치는 flag 입니다 (--no-safe 는 최적화기를 "
                 "아예 안 돌립니다). 그대로 띄우면 목적함수를 바꿨다고 믿게 되는데, 실은 "
                 "정책 청크가 그대로 나가는 서버가 뜹니다")
    # SUBTASK-g — **명시한** `--subtask-gate` (True) 만 거절한다. 안 준 것(None = 기본 on)은
    # 거절하지 않고 `resolve_subtask_gate` 가 gate 를 끄며 한 줄 남긴다.
    if getattr(args, "subtask_gate", None) is True and _label_unavailable(args):
        ap.error("--subtask-gate 는 attention 서버가 싣는 subtask label 을 AG3S 가 읽어야 뜻이 "
                 "있습니다 (--no-safe · --no-attention · --no-perception 은 그 중 하나를 끕니다). "
                 "그대로 띄우면 label 이 늘 없어 gate 가 아무것도 막지 않는데 켰다고 믿게 됩니다")
    if getattr(args, "subtask_probe", None) and (args.no_safe or args.no_attention
                                                 or getattr(args, "no_perception", False)):
        ap.error("--subtask-probe 는 attention 서버의 probe 를 고르는 flag 입니다 (--no-safe · "
                 "--no-attention · --no-perception 은 attention 서버를 안 만듭니다)")
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
    t43t_ag3s = [name for name, on in (
        ("--closing-keep-geometry", getattr(args, "closing_keep_geometry", False)),
        ("--target-volume-exempt", getattr(args, "target_volume_exempt", False)),
        ("--latched-identity-hold", getattr(args, "latched_identity_hold", False)),
        ("--refused-identity-release", getattr(args, "refused_identity_release", None) is not None),
        ("--supported-max-bottom-mm", getattr(args, "supported_max_bottom_mm", None) is not None),
        ("--held-free-observed-mm", getattr(args, "held_free_observed_mm", None) is not None))
        if on]
    if t43t_ag3s and (args.no_safe or getattr(args, "no_perception", False)):
        ap.error(f"{', '.join(t43t_ag3s)} 는 AG3S / SafePolicy 의 flag 입니다 (--no-safe · "
                 "--no-perception 은 그것을 안 만듭니다) — 빼고 띄우십시오")
    if (getattr(args, "held_free_observed_mm", None) is not None
            and getattr(args, "esdf_backend", None) != "curobo"):
        ap.error("--held-free-observed-mm 는 cuRobo TSDF 의 voxel 을 비웁니다 (--esdf-backend curobo)")
    if getattr(args, "held_body_cover", False) and (args.no_safe
                                                    or getattr(args, "no_perception", False)):
        ap.error("--held-body-cover 는 AG3S 의 쥔 물체 질의 구를 고치는 flag 입니다 "
                 "(--no-safe · --no-perception 은 AG3S 를 안 만듭니다). 그대로 띄우면 효과가 "
                 "있었다고 믿게 됩니다 — 빼고 띄우십시오")
    # --- T27 ------------------------------------------------------------------------------
    if args.no_limits and args.no_safe:
        ap.error("--no-limits 는 최적화기의 joint limit 행을 끄는 flag 입니다 (--no-safe 는 "
                 "최적화기를 아예 안 돌립니다). 그대로 띄우면 limit 을 뺐다고 믿게 되는데, 실은 "
                 "정책 청크가 그대로 나가는 서버가 뜹니다")
    if args.no_perception:
        if args.no_safe:
            ap.error("--no-perception 은 π0.5 → TO 를 돌리는 서버입니다 (--no-safe 는 TO 를 안 "
                     "돌립니다). 둘을 같이 주면 어느 서버인지 알 수 없습니다 — 정책만 서빙하려면 "
                     "--no-safe 하나만 주십시오")
        if args.shadow:
            ap.error("--no-perception 과 --shadow 는 같이 쓸 수 없습니다. shadow 는 '전부 "
                     "계산하되 로봇은 정책 원본을 실행' 이고, 지각이 없으면 그것은 --no-safe 와 "
                     "같습니다. E1/E2 는 refined 를 실행해야 합니다 (로컬 --safe-gate off)")
        # **AG3S 를 고치는 flag 는 이 서버에서 아무 일도 하지 않는다.** 받아 놓고 뜨면 그것이
        # 효과가 있었다고 믿게 된다 (이 함수의 머리말).
        inert = [flag for flag, given in (
            ("--exclude-links", bool(args.exclude_links)),
            ("--no-self-collision", bool(args.no_self_collision)),
            ("--target-field-policy", args.target_field_policy != "relax"),
            ("--static-geometry", bool(args.static_geometry) and args.static_geometry != "none"),
            ("--esdf-backend curobo", args.esdf_backend == "curobo"),
            ("--self-filter-*", bool(getattr(args, "self_filter_inflation_link", ()))
             or getattr(args, "self_filter_target_guard", None) is not None),
            ("--max-field-age-sec", args.max_field_age_sec is not None),
        ) if given]
        if inert:
            ap.error(f"--no-perception 서버에는 AG3S 가 없으므로 {', '.join(inert)} 가 아무 일도 "
                     "하지 않습니다. 그대로 띄우면 효과가 있었다고 믿게 됩니다 — 빼고 띄우십시오")
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
    if args.no_perception and options:
        ap.error("--sphere-* 는 충돌 제약 모델의 구를 고치는 flag 입니다 (--no-perception 서버에는 "
                 "충돌 행이 없습니다). 그대로 띄우면 효과가 있었다고 믿게 됩니다")
    # T43 R — 같은 규율: 형식 오류는 체크포인트를 올리기 전에, 제약 모델이 없는 서버에서는 거절.
    cover = gripper_cover_options(args)
    if cover and (args.no_safe or args.no_perception):
        ap.error("--gripper-cover 는 제약 모델의 그리퍼 구를 더하는 flag 입니다 (--no-safe · "
                 "--no-perception 서버에는 그 모델의 충돌 행이 없습니다). 그대로 띄우면 효과가 "
                 "있었다고 믿게 됩니다")
    # **self-filter flag 의 형식 오류도 여기서 죽는다** (T19). 모르는 link 이름은 모델이 있어야
    # 가를 수 있으므로 `AG3S` 생성자가 거절한다 — 그것도 체크포인트를 올리기 전이다.
    try:
        filter_options = self_filter_options(args)
    except ValueError as exc:
        ap.error(str(exc))
    if args.no_safe and filter_options:
        ap.error("--self-filter-* 는 AG3S 의 자기 필터를 고치는 flag 입니다 (--no-safe 는 AG3S 를 "
                 "아예 안 만듭니다). 그대로 띄우면 사과를 보존했다고 믿은 채 필터 자체가 없는 "
                 "서버가 뜹니다")
    # T31 — TO 가 없으면 limit 출처·SQP 예산 flag 는 아무 일도 안 한다.
    to_flags = [name for name, given in (
        ("--limits-source", getattr(args, "limits_source", "model_xml") != "model_xml"),
        ("--sqp-*", bool(sqp_overrides(args))),
    ) if given]
    if args.no_safe and to_flags:
        ap.error(f"{', '.join(to_flags)} 는 TO 를 고치는 flag 입니다 (--no-safe 는 TO 를 아예 안 "
                 "돌립니다). 그대로 띄우면 효과가 있었다고 믿게 됩니다")
    for flag, value in (("--sqp-min-iterations", getattr(args, "sqp_min_iterations", None)),
                        ("--sqp-max-iterations", getattr(args, "sqp_max_iterations", None))):
        if value is not None and value < 1:
            ap.error(f"{flag} 는 1 이상이어야 합니다 (got {value})")
    # `--plan-horizon` 은 여기서 한 번 읽어 본다. 잘못된 값이 서버를 띄운 뒤에 죽으면 그때는
    # 체크포인트 두 벌을 이미 GPU 에 올린 뒤다.
    try:
        resolve_plan_horizon(args.plan_horizon)
    except ValueError as exc:
        ap.error(str(exc))


def limits_overrides(args) -> dict:
    """`--limits-source` · `--model-xml` · `--joint-ranges` → `limits` 절 (T31 G1).

    `model_xml` 출처면 **서버가 로드한 바로 그 MJCF** 를 적는다 — `LimitsConfig.model_xml` 이 비면
    작업공간 기본 모델을 읽는데, `--model-xml` 이 다른 파일이면 한계와 시뮬레이터가 갈라진다.
    """
    source = getattr(args, "limits_source", None) or "model_xml"
    out: dict = {"source": source}
    if source == "model_xml":
        model_xml = getattr(args, "model_xml", None)
        if model_xml:
            out["model_xml"] = str(model_xml)
    elif source == "config":
        out["position_ranges"] = load_joint_ranges(getattr(args, "joint_ranges", None))
    return out


def load_joint_ranges(path) -> dict:
    """`--joint-ranges FILE` → ``{joint: [lower, upper]}``. YAML 이 JSON 을 포함하므로 YAML 로 읽는다."""
    if not path:
        raise SystemExit("--limits-source config 에는 --joint-ranges FILE 이 필요합니다 — 표 없는 "
                         "config 출처는 아무도 적지 않은 범위로 돕니다")
    import yaml

    data = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"--joint-ranges {path}: {{joint: [lower, upper]}} 매핑이어야 합니다")
    return {str(k): [float(v) for v in pair] for k, pair in data.items()}


def sqp_overrides(args) -> dict:
    """`--sqp-*` · `--no-limit-projection` 중 **준 것만** (T31 G2). 빈 dict 면 `SqpConfig` 기본값."""
    out: dict = {}
    for flag, key, cast in (("sqp_min_iterations", "min_iterations", int),
                            ("sqp_max_iterations", "max_iterations", int),
                            ("sqp_time_budget_ms", "time_budget_ms", float)):
        value = getattr(args, flag, None)
        if value is not None:
            out[key] = cast(value)
    if getattr(args, "no_limit_projection", False):
        out["limit_projection_fallback"] = False
    return out


def announce_joint_limits(robot_model, layout, limits_config) -> dict:
    """**관절마다 URDF 대 사용값 표를 시작 로그에 찍는다** (T31 G1). 다르면 크게.

    표를 dict 로도 돌려준다 — 기록기 meta 에 실어 "이 실행이 어느 범위로 돌았나" 를 기록에서
    되짚을 수 있게 한다. T31 전 기록에는 이 키가 없고 그 실행들은 URDF 범위였다.
    """
    from benchmark.trajopt.limits import format_position_limit_table, position_limit_table

    log = logging.getLogger(__name__)
    table = position_limit_table(robot_model, layout, limits_config)
    text = format_position_limit_table(table)
    if table["n_differs"] or table["n_ctrl_mismatch"]:
        log.warning(
            "!!! TO JOINT POSITION LIMITS DIFFER FROM THE URDF ON %d OPTIMIZED JOINT(S) "
            "(source: %s) !!!\n%s\n"
            "    URDF 는 형상(구 모델·FK)에만 쓰인다. 속도·가속도 한계는 URDF 그대로다%s",
            table["n_differs"], table["source"], text,
            (f"\n    !!! {table['n_ctrl_mismatch']} joint(s): MJCF jnt_range ≠ actuator ctrlrange "
             "— 교집합을 쓴다 !!!") if table["n_ctrl_mismatch"] else "")
    else:
        log.info("%s", text)
    return table


def joint_limits_meta(limits_config) -> dict:
    """기록 meta 의 `joint_limits` — 출처와 (MJCF 면) 파일. 표 전체는 시작 로그에 있다."""
    from benchmark.trajopt.limits import limits_model_xml

    out = {"source": limits_config.source,
           "position_margin": float(limits_config.position_margin),
           "velocity_acceleration": "urdf"}
    if limits_config.source == "model_xml":
        out["model_xml"] = limits_model_xml(limits_config)
    elif limits_config.source == "config":
        out["position_ranges"] = {k: list(v) for k, v in
                                  dict(limits_config.position_ranges or {}).items()}
    if getattr(limits_config, "rest_start", False):
        # T41 b — 켰을 때만 키가 생긴다 (꺼진 기록의 meta 는 예전 그대로).
        out["rest_start"] = {"tolerance_rad": limits_config.rest_start_tolerance}
    if getattr(limits_config, "servo_model", False):
        # T43 Q — 같은 규약.
        out["servo_model"] = True
    return out


def sqp_meta(sqp_config) -> dict:
    """기록 meta 의 `sqp` — 반복 수 우선 예산과 투영 대비 (T31 G2)."""
    return {"min_iterations": int(sqp_config.min_iterations),
            "max_iterations": int(sqp_config.max_iterations),
            "time_budget_ms": float(sqp_config.time_budget_ms),
            "limit_projection_fallback": bool(sqp_config.limit_projection_fallback)}


def announce_path_check(to_config) -> None:
    """T43 Q — 켰을 때만 한 줄 (WARNING). 꺼져 있으면 아무것도 찍지 않는다."""
    servo = bool(getattr(to_config.limits, "servo_model", False))
    red = to_config.reduction
    sweep = bool(getattr(red, "sweep_check", False))
    if not (servo or sweep):
        return
    logging.warning(
        "TO path check (T43 Q): %s%s — 중간 자세 행 %d 개/제어 행, floor = min(0, q_now 의 여유)",
        "servo model (MJCF kp·kv·damping·armature, exec_feedback 속도 추정) · 각 행 끝 예측 자세"
        if servo else "관절 공간 보간",
        (f" · sweep ≤ {red.sweep_max_m * 1000:.1f} mm (행당 ≤ {red.sweep_max_samples} 조각)"
         if sweep else ""), red.sweep_rows_per_step)


def announce_sqp_budget(sqp_config) -> None:
    """SQP 예산을 한 줄로 — **반복 수 우선**이면 그렇다고, 투영 대비가 꺼져 있으면 크게 (T31 G2)."""
    log = logging.getLogger(__name__)
    log.info("TO SQP budget: at least %d iteration(s) before the %.0f ms wall-clock budget applies, "
             "at most %d (T31: iterations first — the simulation loop waits for the TO)",
             min(sqp_config.min_iterations, sqp_config.max_iterations),
             sqp_config.time_budget_ms, sqp_config.max_iterations)
    if not sqp_config.limit_projection_fallback:
        log.warning("!!! LIMIT PROJECTION FALLBACK IS OFF (--no-limit-projection) !!!\n"
                    "    limit 만 어긴 reference 에 받아들여진 후보가 없으면 거절된 QP 후보"
                    "(best_unaccepted)가 나간다 — T24 규칙")


def obstacle_overrides(args) -> dict:
    """`--obstacle-margin` · `--obstacle-margin-support` → `collision` 의 키 (T41 a). 안 줬으면 `{}`."""
    out: dict = {}
    margin = float(getattr(args, "obstacle_margin", 0.0) or 0.0)
    if not margin >= 0.0:
        raise SystemExit(f"--obstacle-margin 은 0 이상이어야 합니다: {margin}")
    if margin > 0.0:
        out["obstacle_margin"] = margin
    if getattr(args, "obstacle_margin_support", False):
        if margin <= 0.0:
            raise SystemExit("--obstacle-margin-support 는 --obstacle-margin M (> 0) 과 함께 줘야 "
                             "합니다 — 혼자서는 아무 일도 하지 않습니다")
        out["obstacle_margin_support"] = True
    if getattr(args, "target_volume_exempt", False):
        # T43 T step 2 — rows nearest the target's carve-out volume are target rows.
        out["target_volume_exempt"] = True
    return out


def rest_start_overrides(args) -> dict:
    """`--rest-start` · `--rest-start-tolerance` → `limits` 의 키 (T41 b). 안 줬으면 `{}`."""
    out: dict = {}
    tol = getattr(args, "rest_start_tolerance", None)
    if getattr(args, "rest_start", False):
        out["rest_start"] = True
        if tol is not None:
            tol = float(tol)
            if not (math.isfinite(tol) and tol >= 0.0):
                raise SystemExit(f"--rest-start-tolerance 는 0 이상의 rad 이어야 합니다: {tol}")
            out["rest_start_tolerance"] = tol
    elif tol is not None:
        raise SystemExit("--rest-start-tolerance 는 --rest-start 와 함께 줘야 합니다 — 혼자서는 "
                         "아무 일도 하지 않습니다")
    return out


def path_overrides(args) -> tuple[dict, dict]:
    """`--servo-model` · `--sweep-*` → (`limits` 의 키, `reduction` 의 키) (T43 Q). 안 줬으면 둘 다 `{}`."""
    limits: dict = {}
    reduction: dict = {}
    servo = bool(getattr(args, "servo_model", False))
    sweep = bool(getattr(args, "sweep_check", False))
    if servo:
        limits["servo_model"] = True
    if sweep:
        reduction["sweep_check"] = True
    for flag, key, scale, needs in (("sweep_max_mm", "sweep_max_m", 1e-3, sweep),
                                    ("sweep_max_samples", "sweep_max_samples", None, sweep),
                                    ("sweep_rows_per_step", "sweep_rows_per_step", None,
                                     sweep or servo)):
        value = getattr(args, flag, None)
        if value is None:
            continue
        if not needs:
            raise SystemExit(f"--{flag.replace('_', '-')} 는 "
                             + ("--sweep-check" if flag != "sweep_rows_per_step"
                                else "--servo-model 또는 --sweep-check")
                             + " 와 함께 줘야 합니다 — 혼자서는 아무 일도 하지 않습니다")
        if not (math.isfinite(float(value)) and float(value) > 0):
            raise SystemExit(f"--{flag.replace('_', '-')} 는 0 보다 커야 합니다: {value}")
        reduction[key] = float(value) * scale if scale is not None else int(value)
    return limits, reduction


def trajopt_config_from_args(args):
    """CLI → `TrajOptConfig`. `SafePolicy` 경로와 `--no-perception` 경로가 **같은 함수**를 쓴다.

    **준 것만 넣는다.** 빈 dict 면 키가 없고, 그러면 `CostConfig` · `LimitsConfig` 기본값이 그대로다
    — 호출이 예전과 글자 그대로 같다는 뜻이다 (`sphere_options` 와 같은 규약).
    """
    from benchmark.trajopt.config import NO_LIMITS, TrajOptConfig

    weights = cost_overrides(args)
    no_collision = bool(getattr(args, "no_collision", False)) or bool(
        getattr(args, "no_perception", False))
    limits = limits_overrides(args)
    sqp = sqp_overrides(args)
    path_limits, path_reduction = path_overrides(args)
    return TrajOptConfig.from_dict({
        "collision": {"backend": "esdf", "esdf_margin": args.esdf_margin,
                      "use_support_planes": False,
                      # T41 a — 준 것만 넣는다 (기본 off 면 키가 없다).
                      **obstacle_overrides(args),
                      # **기본값을 여기 다시 적지 않는다.** 켠 경우에는 키가 아예 없다.
                      **({"enabled": False} if no_collision else {})},
        **({"cost": weights} if weights else {}),
        # T27 — `--no-limits`. 무엇이 빠지고 무엇이 남는지는 `config.NO_LIMITS` 한 곳에.
        # T31 — position 범위의 출처 (`limits_overrides`): 기본은 `--model-xml` 의 MJCF.
        "limits": {**limits,
                   **(dict(NO_LIMITS) if getattr(args, "no_limits", False) else {}),
                   # T41 b — 준 것만 (기본 off 면 키가 없다).
                   **rest_start_overrides(args),
                   # T43 Q — 준 것만.
                   **path_limits},
        **({"reduction": path_reduction} if path_reduction else {}),
        **({"sqp": sqp} if sqp else {}),
        # 기하 인증 요구는 여기 **한 곳**에서만 켜고 끈다 (`--no-perception` 은 `ToOnlyPolicy` 가
        # 끈다 — 인증할 기하가 애초에 없다).
        "safety": {"require_certified_geometry": not args.allow_uncertified},
        # 다듬는 창. `execution` 이면 `horizon.execution_length` 를 따라가므로 실행 길이가
        # 두 번 적히지 않는다 (`config.PLAN_EXECUTION_WINDOW` 머리말).
        "horizon": {"plan_horizon": resolve_plan_horizon(args.plan_horizon)},
    })


def build_to_only_robot_model(model_xml: str, *, links: str):
    """`--no-perception` 의 TO 가 쓸 로봇 모델 — FK 와 joint limit 의 근거. AG3S 를 만들지 않는다.

    `SafePolicy` 경로와 **같은 builder** (`build_constraint_robot_model`) 와 같은 `--links` 필터다.
    충돌 행이 없으므로 구 범위는 FK 비용만 바꾼다 (gripper 가 가장 싸다).
    """
    import mujoco

    from benchmark.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene

    mj_model = mujoco.MjModel.from_xml_path(str(pathlib.Path(model_xml).resolve()))
    scene = TransportScene.attach(mj_model, mujoco.MjData(mj_model))
    return build_constraint_robot_model(scene, link_filter=constraint_link_filter(links))


def build_to_only_policy(policy, args):
    """`--no-perception` 서버의 정책. attention 사본·AG3S·거리장 기록기를 **만들지 않는다**."""
    from benchmark.trajopt.to_only_policy import ToOnlyPolicy, ToOnlyRecorder

    to_config = trajopt_config_from_args(args)
    horizon = to_config.horizon
    announce_no_perception()
    announce_limits_switch(to_config.limits)
    weights = cost_overrides(args)
    announce_cost_weights(weights)
    logging.info(
        "TO objective: w_track=%g w_smooth=%g w_continuity=%g w_slack=%g%s",
        to_config.cost.w_track, to_config.cost.w_smooth,
        to_config.cost.w_continuity, to_config.cost.w_slack,
        "" if weights else " (전부 기본값)")
    logging.info("TO plan window: %d of %d chunk steps (execution_length=%d)",
                 horizon.planned, horizon.horizon, horizon.execution_length)
    recorder = None
    if args.record_constraints:
        recorder = ToOnlyRecorder(args.record_constraints, meta={
            "config": args.config, "checkpoint": args.checkpoint,
            "model_xml": args.model_xml, "links": args.links,
            "perception": False, "plan_horizon": args.plan_horizon,
            **({"limits": "off"} if args.no_limits else {}),
            # T31 — position 범위의 출처. **항상 남긴다**: 기본값이 바뀌었으므로(URDF → MJCF)
            # 이 키가 없는 기록은 T31 전 = URDF 범위다.
            "joint_limits": joint_limits_meta(to_config.limits),
            "sqp": sqp_meta(to_config.sqp),
            **({"cost": weights} if weights else {}),
            "trajopt_config": to_config.to_dict()})
        logging.info("recording TO-only diagnostics to %s (reference · refined · 관절별 편차 · "
                     "limit overshoot · SQP metrics)", recorder.run_dir)
    served = ToOnlyPolicy(policy, robot_model=build_to_only_robot_model(
        args.model_xml, links=args.links), to_config=to_config, recorder=recorder)
    # T31 G1 — 관절별 URDF 대 사용값. 다르면 크게.
    announce_joint_limits(served.robot_model, served.layout, to_config.limits)
    announce_sqp_budget(to_config.sqp)
    announce_reference_payload()
    announce_execution_mode(False)
    return served


def exit_cleanly_on_sigterm() -> None:
    """SIGTERM 을 `SystemExit` 으로 바꾼다 (T38 B5) — 그래야 `atexit` 이 돈다.

    기록기가 background 로 쓰므로(`ConstraintRecordWriter(background=True)`), 종료 순간 큐에
    남은 청크는 `atexit` flush 가 써야 디스크에 남는다. 기본 SIGTERM 은 `atexit` 없이 프로세스를
    끝내고, 실험 스크립트는 서버를 `kill $PID` (SIGTERM) 로 멈춘다 (`outputs/verify/T34/p/
    run_chain_e3.sh`). 누가 이미 handler 를 달았으면 건드리지 않는다.
    """
    import signal

    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    reject_bad_flag_combinations(ap, args)

    logging.basicConfig(level=logging.INFO, force=True)
    # SUBTASK-g — 기본 on 인 gate 를 bool 로 확정한다. 이 아래는 전부 bool 만 본다.
    gate_note = resolve_subtask_gate(args)
    if gate_note:
        logging.info(gate_note)
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

    recorder = None  # SafePolicy 의 `ConstraintRecordWriter` (아래 else 분기에서만 생긴다)
    if args.no_safe:
        from benchmark.trajopt.policy_seed import SeededPolicy

        logging.info("safety layer OFF — serving the bare policy")
        # T39 — 요청의 맨 키 `policy_seed` 를 벗겨 소비하는 감싸개. 키가 없으면 obs 를 그대로 넘긴다.
        served = SeededPolicy(policy)
    elif args.no_perception:
        # T27 — π0.5 → TO 만. attention 두 번째 사본을 **올리지 않는다** (필요 없다).
        served = build_to_only_policy(policy, args)
    else:
        from benchmark.trajopt.safe_policy import SafePolicy

        served_policy = policy
        if not args.no_attention:
            from benchmark.trajopt.attention_policy import AttentionPolicy, load_attention_model

            logging.info("loading a second copy of the checkpoint for attention extraction")
            attn_model = load_attention_model(args.config, args.checkpoint)
            # SUBTASK-c — 같은 prefix pass 에서 subtask label 을 싣는다 (`clustering.subtask_probe_path`).
            served_policy = AttentionPolicy(policy, attn_model,
                                            subtask_probe=args.subtask_probe or "default")
            probe = served_policy.subtask_probe
            logging.info("subtask label: %s · gate %s",
                         "OFF (probe 를 못 읽었다 — label 없음)" if probe is None
                         else f"probe {probe.name} ({args.subtask_probe or 'shipped asset'})",
                         "ON (place/home → 새 조작 대상 없음 · PLACED 뒤 home → 해제)"
                         if args.subtask_gate else "off (기록만)")

        if args.record_constraints:
            from benchmark.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

            meta_to_config = trajopt_config_from_args(args)

            # T38 B5 — npz 쓰기(≈ 245 ms/청크, 대부분 zipfile 압축)를 응답 경로 밖 스레드로.
            # 기록 내용 · 순서는 같다; 종료 시 flush (`exit_cleanly_on_sigterm` 참고).
            recorder = ConstraintRecordWriter(
                args.record_constraints, esdf_mode=args.record_constraints_esdf,
                background=True,
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
                      # **끈 경우에만 남긴다.** 충돌이 꺼진 기록을 켜진 것과 나란히 읽는 것이
                      # 이 flag 의 가장 나쁜 실패다 (`self_collision` 과 같은 규약).
                      **({"collision": "off"} if args.no_collision else {}),
                      # T27 — 같은 규약. limit 이 꺼진 기록을 켜진 것과 나란히 읽으면 안 된다.
                      **({"limits": "off"} if args.no_limits else {}),
                      # T31 — position 범위의 출처와 SQP 예산. **항상 남긴다**: 기본값이 바뀌었으므로
                      # (URDF → MJCF, 1 → 2 반복) 이 키가 없는 기록은 T31 전이다.
                      "joint_limits": joint_limits_meta(meta_to_config.limits),
                      "sqp": sqp_meta(meta_to_config.sqp),
                      # **다듬는 창은 항상 남긴다.** T6f 에서 기본값이 32 → 실행 창으로
                      # 바뀌었으므로, 안 남기면 T6d 기록과 이 기록이 meta 로 구별되지 않는다 —
                      # 그리고 둘은 서로 다른 것을 재고 있다.
                      "plan_horizon": args.plan_horizon,
                      **({"sphere_options": sphere_options(args)}
                         if sphere_options(args) else {}),
                      # T43 R — 켰을 때만 키가 생긴다.
                      **({"gripper_cover": gripper_cover_options(args)}
                         if gripper_cover_options(args) else {}),
                      **({"self_filter": self_filter_options(args)}
                         if self_filter_options(args) else {}),
                      # SUBTASK-c — **항상 남긴다**: 이 키가 없는 기록은 SUBTASK-c 전이다.
                      "subtask": {"gate": bool(args.subtask_gate),
                                  "probe": args.subtask_probe or "shipped:kv_L4_v1"}})
            # **시작할 때 거절한다.** 기록기가 이 backend 의 거리장을 실을 수 없으면 청크마다
            # 예외가 나고, 예전에는 그것이 `print` 한 줄이라 75 chunk 를 돌리고도 빈 디렉토리가
            # 남았다 (2026-09-28). 여기서 죽으면 체크포인트를 올리기 전이다.
            from benchmark.ag3s.experiments.sources.constraint_record import (
                SUPPORTED_ESDF_BACKENDS)

            if args.esdf_backend not in SUPPORTED_ESDF_BACKENDS:
                ap_error = f"--record-constraints 는 esdf backend {args.esdf_backend!r} 의 " \
                           f"거리장을 실을 수 없습니다 (지원: {list(SUPPORTED_ESDF_BACKENDS)}). " \
                           f"그대로 띄우면 청크마다 기록이 실패하고 끝나고 **빈 디렉토리**가 " \
                           f"남습니다 — 그러면 실행을 한 번 더 돌려야 합니다"
                raise SystemExit(ap_error)
            logging.info("recording AG3S constraint diagnostics to %s (esdf=%s, backend=%s)",
                         recorder.run_dir, args.record_constraints_esdf, args.esdf_backend)
            if args.record_constraints_esdf == "occupancy" and args.esdf_backend == "curobo":
                # 거절하지 않는다 — 격자 메타와 나머지 진단은 그대로 남는다. 다만 **점유 배열이
                # 안 남는다는 사실**을 시작할 때 말한다. 끝나고 `esdf_occupancy` 를 찾다가 없는
                # 것을 "점유가 비어 있었다" 로 읽는 것이 이 모드의 조용한 실패다.
                logging.warning(
                    "!!! --record-constraints-esdf occupancy 는 cuRobo backend 에서 점유 배열을 "
                    "남기지 않습니다 !!!\n"
                    "    block-sparse TSDF 에는 legacy 의 3 상태(FREE/OCCUPIED/UNKNOWN)에 "
                    "대응하는 배열이 없습니다. 격자 메타·계층 정보는 남고, **거리장 자체를 "
                    "보려면 `full` 을 쓰십시오** (계층마다 float16 격자를 싣습니다: 주 계층 · "
                    "coarse · target 없는 계층).")

        weights = cost_overrides(args)
        to_config = trajopt_config_from_args(args)
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
        if to_config.collision.obstacle_margin > 0.0:
            logging.warning(
                "TO obstacle_margin: %.1f mm — target 이 아닌 장애물 행만 max(esdf_margin, 이 값). "
                "지지면 %s · target · 쥔 물체 행은 제외 (T41 a)",
                to_config.collision.obstacle_margin * 1000,
                "포함" if to_config.collision.obstacle_margin_support else "제외")
        if getattr(to_config.limits, "rest_start", False):
            tol = to_config.limits.rest_start_tolerance
            logging.warning(
                "TO rest_start: 직전 청크가 통째로 HOLD 인 청크는 정지 출발 범위로 푼다 — 첫 스텝 "
                "|Q0 − q_now| ≤ %s, 이후 a_max·dt² 씩 늘어난다 (T41 b)",
                "a_max·dt² (관절마다)" if tol is None else f"{tol:.4f} rad ({math.degrees(tol):.2f}°)")
        announce_path_check(to_config)
        announce_collision_switch(to_config.collision.enabled)
        announce_limits_switch(to_config.limits)
        announce_cost_weights(weights)
        logging.info(
            "TO objective: w_track=%g w_smooth=%g w_continuity=%g w_slack=%g%s",
            to_config.cost.w_track, to_config.cost.w_smooth,
            to_config.cost.w_continuity, to_config.cost.w_slack,
            "" if weights else " (전부 기본값)")
        announce_reference_payload()

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
                            gripper_cover=gripper_cover_options(args),
                            self_collision=not args.no_self_collision,
                            target_field_policy=args.target_field_policy,
                            plan_horizon_steps=horizon.planned,
                            rows_per_step=to_config.reduction.rows_per_step,
                            self_filter=self_filter_options(args),
                            subtask_gate=args.subtask_gate,
                            subtask_probe_path=args.subtask_probe,
                            held_body_cover=args.held_body_cover,
                            identity=t43t_identity_section(args) or None,
                            held_free_observed_pad=(
                                None if args.held_free_observed_mm is None
                                else float(args.held_free_observed_mm) / 1000.0)),
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
        announce_execution_mode(args.shadow)
        # T31 G1 — TO 가 실제로 쓰는 로봇 모델로 표를 찍는다. 다르면 크게.
        announce_joint_limits(served.constraint_robot_model, served.layout, to_config.limits)
        announce_sqp_budget(to_config.sqp)

    if recorder is not None:
        exit_cleanly_on_sigterm()
    # T39 — seed 입구가 정책 RNG 에 닿는가. 못 닿아도 서버는 뜬다 (seed 를 안 보내는 실행은 그대로) —
    # seed 를 실은 요청이 오면 그때 크게 실패한다.
    import os

    from benchmark.trajopt.policy_seed import describe_rng_owner, describe_xla_determinism

    logging.info("%s", describe_rng_owner(served))
    logging.info("%s", describe_xla_determinism(os.environ.get("XLA_FLAGS")))
    logging.info("serving on port %d", args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=served, host="0.0.0.0", port=args.port,
        metadata=getattr(served, "metadata", {}),
    ).serve_forever()


if __name__ == "__main__":
    main()
