"""`ConstraintValidity` 를 내리는 자리마다 **왜** 를 남기는 한 가지 방법.

## 왜 이 모듈이 있나

2026-09-25 의 첫 live smoke 에서 2 청크 중 1 개가 HOLD 로 갔다. `max_violation_m` 은 0.0 —
궤적은 모든 제약을 통과했고, **AG3S 가 기하를 인증하지 못한 것**이다 (`ag3s_status: degraded`,
`geometry_certified: False`). 그런데 **왜 degraded 인지가 와이어에도 서버 로그에도 없었다.**

사유가 아예 없던 것은 아니다. `DEGRADED` 를 내리는 자리는 거의 다 그 옆에서 `notes` 에 사람이
읽을 문장을 하나 넣고 있었다. 빠진 것은 둘이다:

1. **한 자리에는 정말로 노트가 없었다** — `multiview.py` 의 "어떤 카메라가 `max_points` cap 에
   걸렸다" 분기. `validity` 만 내리고 조용히 지나갔다.
2. **노트가 와이어에 실리지 않았다.** 응답의 `notes` 는 `TrajOptResult.notes`(최적화기 쪽)이고,
   `CollisionConstraintSet.notes`(지각 쪽)는 서버 프로세스 밖으로 나가는 길이 없었다.

## 왜 문자열 앞에 코드를 붙이나 — 새 자료구조를 만들지 않는 이유

`notes: list[str]` 는 `CollisionConstraintSet` · `FusionResult` · 기록 npz · 그림 스크립트까지
이미 관통하고 있다. 여기에 병렬 자료구조를 하나 더 만들면 **두 개를 다 갱신해야 하고, 갱신을
빠뜨린 쪽이 사유가 빈 경로가 된다** — 이 모듈이 막으려는 바로 그 실패다. 그래서 매체는 그대로
`notes` 를 쓰고, **문장 앞에 `<code>: ` 를 붙인다.**

    "esdf_unknown_fraction: 31.2% of the ESDF volume was never observed ..."

얻는 것은 셋이다.

* **자리를 특정할 수 있다.** 코드는 `CODES` 에 등록된 것만 쓸 수 있고 (`reason()` 이 모르는
  코드를 거절한다), 등록된 코드 하나가 소스의 한 분기에 대응한다 — `grep` 이 답을 준다.
* **기계가 읽을 수 있다.** `reasons(notes)` 가 코드와 문장을 갈라 돌려주고, 와이어와 프레임
  기록이 그것을 싣는다. 판정을 세려는 쪽이 산문을 파싱하지 않는다.
* **옛 소비자가 안 깨진다.** 문장은 그대로 남으므로 노트를 사람이 읽는 곳(서버 로그·기록·
  `"capped at max_points" in note` 로 검사하는 테스트)은 손댈 필요가 없다.

## 이 모듈이 강제하지 않는 것

**무엇을 할지는 정하지 않는다.** `ConstraintValidity` 가 그러지 않는 것과 같은 이유다 (그
docstring: *"AG3S does not decide what to do about it"*). 코드는 *"이 프레임의 기하를 인증할 수
없는 이유가 이것이다"* 까지만 말한다.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

__all__ = ["CODES", "reason", "reasons", "codes", "explain", "ensure_reason", "SEPARATOR"]

#: 코드와 문장을 가르는 것. 노트 문장 안에 `": "` 가 또 나와도 **첫 번째**에서만 자른다.
SEPARATOR = ": "

#: 등록된 사유 코드 → 그 코드가 사는 자리와 조건. **여기 없는 코드는 쓸 수 없다**
#: (`reason()` 이 `KeyError` 를 던진다) — 오타가 조용히 새 코드를 만들면 "사유가 비지 않았다"
#: 는 검사가 통과하면서도 아무도 그 코드를 모르는 상태가 된다.
#:
#: 값은 **사람을 위한 한 줄**이고 기계는 키만 본다. 새 `DEGRADED` 분기를 만들 때는 여기에
#: 한 줄 더하는 것이 그 분기를 등록하는 유일한 방법이다.
CODES: dict[str, str] = {
    # --- 점군 (단일 카메라 경로) -------------------------------------------------------
    "pointcloud_capped":
        "pipeline: 점군이 pointcloud.max_points cap 에 걸려 voxel 을 키웠다. 거칠어졌을 뿐 "
        "빠진 것은 없다 (모든 점유 셀에 대표가 남는다)",
    # --- 점군 (다중 카메라 융합 경로) --------------------------------------------------
    "fused_pointcloud_capped":
        "multiview: 카메라 중 적어도 하나가 자기 점군에서 max_points cap 에 걸렸다. "
        "**2026-09-25 까지 이 분기에는 노트가 없었다** — degraded 인데 사유가 비는 유일한 자리",
    # --- 카메라 신선도 ------------------------------------------------------------------
    "camera_state_stale":
        "multiview: 어떤 카메라의 robot_state 가 그 이미지의 촬영 시각에서 "
        "timing.max_state_age_sec 이상 떨어져 있다. 클라우드가 엉뚱한 자세에 놓인다",
    "camera_transform_stale":
        "multiview: 어떤 카메라의 이미지가 프레임 기준 시각보다 "
        "timing.max_transform_age_sec 이상 뒤처졌다",
    "camera_skew":
        "multiview: 카메라 사이 촬영 시각 차이가 timing.max_camera_skew_sec 를 넘었다. "
        "두 시점이 움직인 씬에 대해 서로 다른 말을 할 수 있다",
    "camera_missing":
        "multiview: timing.expected_cameras 중 보고하지 않은 카메라가 있다. 씬이 부분 관측이다",
    # --- 후보/제약 용량 -----------------------------------------------------------------
    "candidate_overflow":
        "pipeline: 클러스터가 collision_candidate cap 을 넘어 보수적 aggregate 로 접혔다. "
        "점은 하나도 버려지지 않았다 (버리는 경로는 INCOMPLETE 다)",
    "constraint_sphere_overflow":
        "to_adapter: 제약 구가 builder.max_candidates 를 넘어 overflow 슬롯으로 접혔다. "
        "전부 여전히 표현된다",
    # --- 거리장 -------------------------------------------------------------------------
    "esdf_unknown_fraction":
        "pipeline: ESDF 부피 중 미관측 비율이 esdf.unknown_report_threshold 를 넘었다 "
        "(dense 계층에만 있는 판정 — block-sparse 에는 대응하는 값이 없다)",
    "esdf_dead_camera":
        "pipeline(_esdf_coverage, G3): 어떤 카메라가 이 프레임에 voxel 을 하나도 안 썼다. "
        "정상 프레임에서는 카메라마다 13k~150k 개를 쓰므로 조용한 프레임이 아니라 깨진 것이다",
    "spheres_outside_grid":
        "pipeline(_esdf_coverage, G4): 제약 구 일부가 coverage 격자 밖이다. 거리장은 그 구에 "
        "무엇이 있든 outside_distance 를 답하므로 그 구들은 제약을 받지 않는다",
    # --- 배선 결함 그 자체 ---------------------------------------------------------------
    "degraded_without_reason":
        "**이 코드가 보이면 AG3S 의 배선 결함이다.** validity 가 DEGRADED 로 내려갔는데 어느 "
        "자리도 코드 달린 사유를 남기지 않았다. 새 DEGRADED 분기를 만들면서 reason() 을 "
        "빠뜨린 것이다 — 조용히 지나가면 로컬이 '왜 degraded 인지' 를 다시 알 수 없게 된다",
    # --- 로봇 모델 ----------------------------------------------------------------------
    "no_robot_model":
        "pipeline(_constraints): 로봇 모델이 주입되지 않아 제약을 쓸 대상이 없다. 기하는 있는데 "
        "제약이 없는 것을 VALID 로 두면 소비자가 '제약 없음' 을 '깨끗함' 으로 읽는다",
}


def reason(code: str, detail: str) -> str:
    """`notes` 에 넣을 한 줄. `"<code>: <detail>"`.

    Raises:
        KeyError: `code` 가 `CODES` 에 없을 때. 오타를 조용히 새 코드로 만들지 않는다 —
            그러면 "사유가 비지 않았다" 는 검사는 통과하는데 그 코드를 아무도 모른다.
    """
    if code not in CODES:
        raise KeyError(
            f"unknown degradation code {code!r}. Register it in "
            f"benchmark/ag3s/runtime/degradation.py:CODES with one line saying which branch it "
            f"names — that registration is what makes the reason findable from the wire. "
            f"Known: {sorted(CODES)}")
    text = str(detail).strip()
    return f"{code}{SEPARATOR}{text}" if text else code


def reasons(notes: Iterable[str]) -> list[dict[str, Any]]:
    """등록된 코드를 달고 있는 노트만 `[{"code", "detail"}]` 로. 순서는 그대로.

    코드가 없는 노트는 **버리지 않고 그냥 여기 안 담긴다** — 그쪽은 `notes` 원본이 그대로
    싣는다. 이 함수는 *"기하를 인증할 수 없는 이유"* 만 모으는 자리이고, 산문 노트를 코드로
    추측해 채우면 그 순간 기록이 거짓이 된다.
    """
    out: list[dict[str, Any]] = []
    for note in notes or ():
        head, sep, tail = str(note).partition(SEPARATOR)
        if sep and head in CODES:
            out.append({"code": head, "detail": tail})
        elif not sep and str(note) in CODES:
            out.append({"code": str(note), "detail": ""})
    return out


def codes(notes: Iterable[str]) -> list[str]:
    """등록된 코드만, 나온 순서대로 (중복 제거)."""
    seen: list[str] = []
    for item in reasons(notes):
        if item["code"] not in seen:
            seen.append(item["code"])
    return seen


def explain(notes: Iterable[str], *, limit: int = 3,
            detail_chars: Optional[int] = None) -> str:
    """사람이 읽을 한 줄. 코드가 하나도 없으면 빈 문자열.

    `limit` 과 `detail_chars` 는 **자르는 곳이 어디냐에 따라 다르기 때문에** 인자다. 서버 로그는
    한 줄이 길어도 되지만(기본값: 자르지 않는다), 제어 루프의 HOLD 한 줄은 프레임마다 찍히므로
    길면 다른 것을 다 밀어낸다 — 그쪽은 `detail_chars` 를 준다.

    **자른 것은 잘랐다고 말한다.** 빠뜨린 사유 개수를 뒤에 붙이고 잘린 문장에 `…` 를 남긴다.
    조용히 자르면 읽는 사람이 그것이 전부라고 믿고, 그때 남은 사유가 진짜 원인일 수 있다.
    전문은 언제나 `ag3s` 블록의 `notes` 와 프레임 기록에 있다.
    """
    items = reasons(notes)
    if not items:
        return ""
    shown = items[:limit]
    parts = []
    for item in shown:
        detail = item["detail"]
        if detail_chars is not None and len(detail) > detail_chars:
            detail = detail[:detail_chars].rstrip() + "…"
        parts.append(f"{item['code']}({detail})" if detail else item["code"])
    extra = len(items) - len(shown)
    return "; ".join(parts) + (f"; +{extra} more" if extra > 0 else "")


def ensure_reason(notes: Iterable[str], *, degraded: bool) -> list[str]:
    """`notes` 를 그대로, 단 **degraded 인데 코드가 하나도 없으면 그 사실을 사유로 더해서.**

    이 함수가 이 모듈의 계약을 *구조로* 만든다: *"degraded 인데 사유가 빈 경로는 없다."*
    새 `DEGRADED` 분기를 만들면서 `reason()` 을 빠뜨려도 와이어에 빈 사유가 나가지 않는다.

    **예외를 던지지 않는다.** 진단이 빠진 것으로 지각을 죽이면 그 프레임의 기하가 통째로
    사라지고, 그것은 사유를 모르는 것보다 나쁘다 (`unpack_camera_observations` 가 카메라 누락에
    예외를 내지 않는 것과 같은 판단). 대신 **코드를 달아** 남기므로 `degraded_without_reason`
    을 grep 하면 배선 결함이 바로 나온다 — 단위 테스트는 이 코드가 **나오지 않는 것**을 검사한다.
    """
    out = list(notes or ())
    if degraded and not codes(out):
        out.append(reason(
            "degraded_without_reason",
            "AG3S lowered this frame's validity to DEGRADED but no stage declared a coded "
            "reason. This is a wiring bug in AG3S, not a property of the scene: some branch "
            "calls ConstraintValidity.worst(..., DEGRADED) without the matching "
            "degradation.reason() note"))
    return out
