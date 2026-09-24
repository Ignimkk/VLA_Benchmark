"""거리장이 **언제 · 무엇으로 · 몇 번째로** 만들어졌는가.

`AG3S_TOTAL_TEST_Prompt.md` 의 T0 이 프레임마다 요구하는 것이고, 공통 원칙 4 가
*"갱신되지 않은 프레임의 값을 새로 계산한 것처럼 기록하지 않는다"* 로 못 박은 것이다.

이 검토에서 가장 비싼 실패가 정확히 이 정보가 없어서 안 보였다 — 2026-09-18 의 통합에서
`attach()` 뒤 **14 프레임 동안 지각도 최적화도 한 번 안 돌았는데 상태는 `ok`(9 프레임 전의
낡은 값)로 보고**됐다. 거리값만 보면 그것을 알 방법이 없었다.

## 네 가지 상태

| 상태 | 뜻 |
|---|---|
| `new` | 이 프레임에서 관측을 적분해 새로 만들었다 |
| `carried` | 갱신은 없지만 한도 안이라 아직 유효하다 |
| `stale` | 한도를 넘었다. 안전 게이트가 막아야 한다 |
| `unavailable` | 필드가 없다 (카메라 없음, 예외) |

**서버에는 `carried` 가 없다.** `SafePolicy` 는 planning frame 마다 AG3S 를 돌리므로 필드가
`new` 이거나 `unavailable` 이다. `carried` 는 **control frame** 에서 생긴다 — 청크 하나가
8 스텝(533 ms)을 덮으므로 그 스텝들은 하나의 필드를 나눠 쓴다. 그래서 상태 판정이 두 곳으로
갈리고, `applied_by_client()` 가 클라이언트 쪽 절반이다.

## age 의 기준은 **관측 시각**이다 — 서버 시계가 아니다

`time.monotonic()` 은 프로세스마다 원점이 다르므로 서버가 찍은 시각을 클라이언트가 자기
시계와 견줄 수 없다. 그래서 age 의 기준을 **이 필드가 적분한 관측 중 가장 최신의 촬영
시각**(`observed_at`, 클라이언트가 요청에 실어 보낸 `ag3s/stamp/<cam>`)으로 둔다.
클라이언트는 `now - observed_at` 으로 age 를 얻고, 그것은 서버 시계를 몰라도 성립한다.

그리고 그 값은 애초에 우리가 알고 싶은 것이다 — "이 거리장이 **얼마나 오래된 세상**을
말하고 있는가". F14(상태 지연 한계 100 ms 가 피해 시작점보다 6 배 이상 느슨했다)가 재던 것과
같은 양이다.

`built_at` 은 **서버 시계**이고 단계 시간 계산에만 쓴다. 클라이언트와 견주지 말 것.

## 한도는 아직 정하지 않았다

`age_limit_sec` 이 `None` 이면 **`stale` 판정을 할 수 없다**. 그때 `state` 는 `new`/`carried`
로 남고 `age_limit_sec: null` 이 응답에 실려, 읽는 쪽이 "한도 검사를 안 했다" 를 안다.
0 으로 두거나 아무 값이나 넣지 않는 이유는 F14 다 — 그때 `timing.max_state_age_sec` 의
기본값 100 ms 가 측정 없이 정해진 값이었고, 실측하니 피해가 16 ms 에서 이미 시작했다.
한도는 T3(전 프레임 TSDF/ESDF)에서 프레임 간 필드 변화량을 재고 정한다.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

__all__ = ["FieldProvenance", "NEW", "CARRIED", "STALE", "UNAVAILABLE", "STATES"]

NEW = "new"
CARRIED = "carried"
STALE = "stale"
UNAVAILABLE = "unavailable"
STATES = (NEW, CARRIED, STALE, UNAVAILABLE)


@dataclasses.dataclass(frozen=True)
class FieldProvenance:
    """한 거리장의 출처. 필드에 `field.provenance` 로 붙고 응답에 실린다."""

    #: 이 builder 가 만든 몇 번째 필드인가. 1 부터. 되돌아가거나 건너뛰면 배선 결함이다.
    sequence: int
    #: 어느 구현이 만들었나. `legacy` | `curobo`. T0 의 즉시 실패 조건이 이 값을 본다.
    backend: str
    #: 적분한 관측 중 **가장 최신의 촬영 시각** (클라이언트 시계). age 의 기준.
    observed_at: Optional[float] = None
    #: 필드를 만든 서버 시각 (서버 monotonic). **클라이언트와 견주지 말 것.**
    built_at: Optional[float] = None
    #: 어느 observation frame 에서 나왔나.
    frame_id: str = ""
    frame_index: int = -1
    #: 적분에 실제로 들어간 카메라 이름.
    cameras: tuple[str, ...] = ()
    #: 계층마다 `{voxel_size_m, shape, origin_m}`.
    tiers: tuple[dict, ...] = ()
    #: 소비 시점에 채워지는 것들.
    state: str = NEW
    applied_at: Optional[float] = None
    age_ms: Optional[float] = None
    #: `None` 이면 `stale` 판정을 **하지 않았다**. 위 머리말 참고.
    age_limit_sec: Optional[float] = None
    #: `unavailable` 일 때 왜인지.
    reason: str = ""

    # -- 만들기 --------------------------------------------------------------------
    @staticmethod
    def unavailable(reason: str, *, sequence: int = -1, backend: str = "") -> "FieldProvenance":
        """필드가 없다. **0 이나 빈 값으로 대신하지 않는다** — 없는 것은 없다고 말한다."""
        return FieldProvenance(sequence=sequence, backend=backend,
                               state=UNAVAILABLE, reason=str(reason))

    # -- 소비 ----------------------------------------------------------------------
    def applied_by_client(self, now: float, *,
                          age_limit_sec: Optional[float] = None,
                          carried: bool = False) -> "FieldProvenance":
        """control frame 이 이 필드를 쓰는 순간의 상태를 채워 돌려준다.

        `now` 는 **클라이언트 시계**이고 `observed_at` 과 같은 원점이어야 한다.
        `carried=True` 는 "이 프레임에서 필드를 새로 만들지 않았다" 는 뜻이다 — 청크 하나가
        8 스텝을 덮으므로 두 번째 스텝부터가 그렇다.

        `age_limit_sec` 이 `None` 이면 `stale` 로 올리지 않는다. 한도를 모르는 채 판정하면
        그 판정에 근거가 없고, 근거 없는 판정을 기록에 남기지 않는 것이 이 검토의 규칙이다.
        """
        if self.state == UNAVAILABLE:
            return dataclasses.replace(self, applied_at=float(now),
                                       age_limit_sec=age_limit_sec)
        age_ms = (None if self.observed_at is None
                  else max(0.0, (float(now) - float(self.observed_at)) * 1000.0))
        state = CARRIED if carried else NEW
        if (age_limit_sec is not None and age_ms is not None
                and age_ms > float(age_limit_sec) * 1000.0):
            state = STALE
        return dataclasses.replace(self, state=state, applied_at=float(now),
                                   age_ms=age_ms, age_limit_sec=age_limit_sec)

    @property
    def staleness_checked(self) -> bool:
        """한도가 있어 `stale` 판정이 성립했는가. `False` 면 그 검사를 **안 한 것**이다."""
        return self.age_limit_sec is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "backend": str(self.backend),
            "observed_at": self.observed_at,
            "built_at": self.built_at,
            "frame_id": str(self.frame_id),
            "frame_index": int(self.frame_index),
            "cameras": list(self.cameras),
            "tiers": [dict(t) for t in self.tiers],
            "state": str(self.state),
            "applied_at": self.applied_at,
            "age_ms": self.age_ms,
            "age_limit_sec": self.age_limit_sec,
            "staleness_checked": self.staleness_checked,
            "reason": str(self.reason),
        }

    @staticmethod
    def from_dict(blob: dict) -> "FieldProvenance":
        return FieldProvenance(
            sequence=int(blob.get("sequence", -1)),
            backend=str(blob.get("backend", "")),
            observed_at=blob.get("observed_at"),
            built_at=blob.get("built_at"),
            frame_id=str(blob.get("frame_id", "")),
            frame_index=int(blob.get("frame_index", -1)),
            cameras=tuple(blob.get("cameras", ()) or ()),
            tiers=tuple(blob.get("tiers", ()) or ()),
            state=str(blob.get("state", NEW)),
            applied_at=blob.get("applied_at"),
            age_ms=blob.get("age_ms"),
            age_limit_sec=blob.get("age_limit_sec"),
            reason=str(blob.get("reason", "")),
        )
