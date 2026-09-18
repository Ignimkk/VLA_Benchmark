"""조작 대상을 **에피소드 상태로 붙드는 것** — 걸기 · 유지 · 풀기 (F17).

## 왜 필요한가

`target_grounding` 은 무상태다. 프레임마다 "지금 attention 이 가리키는 덩어리" 를 새로 고르고,
프레임 사이에 아무것도 기억하지 않는다. 그런데 실측에서 **정책의 attention 은 파지에 착수하는
순간 목적지로 옮겨간다** (F11): `run_0004` 에서 손이 사과를 쥔 내내 grounding 이 낸 target 은
바구니였고, 접촉 권한을 거기 걸면 쥔 사과가 완전 여유거리를 요구하는 장애물이 되어 최악
-138.7 mm 로 관통했다.

그래서 식별을 **잠근다.** 두 기록이 잠금을 정당화한다 — `run_0004` 는 앞 7 프레임, `run_0002`
는 앞 8 프레임이 전부 `apple` 이고 purity/iou 가 1.00, 1위와 2위의 점수 격차가 1.7~47배다.
그리고 `run_0002` 에서는 **apple 이 앞 8 프레임에만** 나온다 — 초기를 안 믿으면 그 롤아웃에는
사과를 옳게 식별할 기회가 아예 없다.

## 세 조각

    걸기   연속 `confirm_frames` 프레임 같은 물체 + 1·2위 점수 격차 문턱
    유지   attention 이 어디로 가든 무시
    풀기   그리퍼가 열리거나 성공 판정이 서면

**걸기에 문턱이 필요한 이유**: 두 기록 모두 프레임 2 에서 purity 가 0.62 로 떨어진다. 하필
거기서 잠그면 62 % 짜리 뭉치를 에피소드 내내 붙들게 된다. 지금 구조는 틀려도 다음 프레임에
회복되지만 **잠금에는 회복 지점이 없다** — 실패의 방향이 뒤집힌다.

## 목적지는 F11 을 뒤집어 얻는다

잠금이 걸린 **뒤에** grounding 이 내는 이름은 더 이상 조작 대상이 아니다. 실측에서 그것은
**목적지**다 — `run_0004` 의 파지~놓기 구간(프레임 7~20)에서 grounding 이 `crate` 를 낸 것이
**13/14 프레임**, 최장 연속 11 이고, 프롬프트가 `put the apple in the basket` 이므로 `crate` 가
바로 목적지다. F11 이 결함으로 본 거동이 여기서는 신호가 된다.

목적지에도 같은 잠금을 건다. 한 프레임(18)이 `orange` 로 튀기 때문이다.

## 이 모듈은 부르지 않는다

`attach` / `detach` 를 **여기서 부르지 않는다.** 언제 불러야 하는지만 답하고, 부르는 것은
과제를 소유한 쪽이다 — `attached.py` 머리말이 못박은 계약("AG3S does not decide when a grasp
succeeded")과 같은 이유다. 이 모듈도 로봇도 카메라도 보지 않는다: 이름과 점수, 그리퍼 값,
그리고 외부가 준 성공 판정만 본다.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Optional


class GraspPhase(str, enum.Enum):
    """잠금 상태 기계. 이름은 `Phase`(조작 단계)와 다르다 — 그쪽은 주입 입력이고 이쪽은 우리 상태다."""

    SEARCHING = "searching"   # 아직 아무것도 잠기지 않았다
    LATCHED = "latched"       # 조작 대상이 정해졌다. 아직 쥐지 않았다
    HELD = "held"             # 쥐었다. `attach` 가 불렸어야 한다
    PLACED = "placed"         # 놓았다. `detach` 가 불렸어야 한다


@dataclasses.dataclass(frozen=True)
class LatchConfig:
    """문턱들. 전부 실측에서 나왔다."""

    #: 같은 이름이 몇 프레임 연속이어야 잠그나. 두 기록 다 선두 7~8 프레임이 같은 물체였으므로
    #: 3 은 넉넉하고, 프레임 2 의 purity 0.62 한 번으로는 잠기지 않는다.
    confirm_frames: int = 3
    #: 1위 점수가 2위의 몇 배여야 하나. 실측 격차가 1.7~47배라 1.3 은 통과하고, 애매한 프레임은
    #: 통과하지 못한다.
    score_ratio: float = 1.3
    #: 그리퍼 값이 이보다 작으면 닫힌 것으로 본다. `run_0004` 실측: 열림 ≈ 1.00, 닫힘 ≈ 0.72,
    #: 왼손이 프레임 10 에 닫히고 19 에 열린다 — 손-사과 거리로 잰 파지 구간과 정확히 맞는다.
    gripper_closed_below: float = 0.85
    #: 해제도 연속으로 확인한다. 한 프레임짜리 그리퍼 튐으로 쥔 물체를 잃으면 안 된다.
    release_frames: int = 2


@dataclasses.dataclass(frozen=True)
class LatchEvent:
    """이번 프레임에 무엇을 해야 하는가. 부르는 것은 호출자다."""

    phase: GraspPhase
    #: 잠긴 조작 대상의 이름. `None` 이면 아직 없다.
    manipulated: Optional[str] = None
    #: 잠긴 목적지의 이름.
    destination: Optional[str] = None
    #: 이번 프레임에 `attach()` 를 불러야 한다.
    attach: bool = False
    #: 이번 프레임에 `detach()` 를 불러야 한다.
    detach: bool = False
    #: 왜 그렇게 판단했는지. 로그와 기록에 그대로 남긴다.
    note: str = ""


class CentroidIdentity:
    """무게중심으로 "같은 물체인가" 를 판정한다 — grounding 은 이름표를 주지 않는다.

    `TargetGeometry.id` 는 그 프레임의 클러스터 번호라 프레임 간에 뜻이 없다. 대신 무게중심을
    쓴다: 실측에서 target 이 **다른 물체로 바뀔 때 무게중심이 307~597 mm 뛴다** (F2·F9), 같은
    물체를 계속 보는 동안의 흔들림과 자릿수가 다르다. 그래서 `tolerance`(기본 60 mm) 하나로
    갈린다.

    돌려주는 이름은 사람이 읽으라고 있는 것이 아니라 **잠금이 비교할 수 있으면 되는 것**이다.
    """

    def __init__(self, tolerance: float = 0.06):
        self.tolerance = float(tolerance)
        self._anchors: list = []

    def label(self, centroid) -> Optional[str]:
        if centroid is None:
            return None
        import numpy as np

        c = np.asarray(centroid, float).reshape(3)
        for i, a in enumerate(self._anchors):
            if float(np.linalg.norm(c - a)) <= self.tolerance:
                # 앵커를 갱신하지 않는다 — 물체가 천천히 움직이면 앵커가 따라가며 다른 물체까지
                # 삼킬 수 있다. 처음 본 자리를 기준으로 둔다.
                return f"obj{i}"
        self._anchors.append(c)
        return f"obj{len(self._anchors) - 1}"

    def reset(self) -> None:
        self._anchors.clear()


class _Confirm:
    """같은 이름이 연속 N 번 + 격차 문턱을 넘으면 잠근다. 한 번 잠기면 `release()` 전까지 유지."""

    def __init__(self, config: LatchConfig):
        self._cfg = config
        self._candidate: Optional[str] = None
        self._streak = 0
        self.locked: Optional[str] = None

    def update(self, label: Optional[str], score: float = 1.0,
               runner_up: float = 0.0) -> Optional[str]:
        if self.locked is not None:
            return self.locked           # 유지 — 무엇이 들어오든 무시한다
        if label is None:
            self._candidate, self._streak = None, 0
            return None
        confident = score >= self._cfg.score_ratio * max(runner_up, 1e-9)
        if not confident:
            # 애매한 프레임은 연속을 **끊지는 않되** 세지도 않는다. 끊으면 한 번의 잡음으로
            # 잠금이 영영 안 걸리고, 세면 문턱이 뜻을 잃는다.
            return None
        if label == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = label, 1
        if self._streak >= self._cfg.confirm_frames:
            self.locked = label
        return self.locked

    def release(self) -> None:
        self._candidate, self._streak, self.locked = None, 0, None


class GraspLatch:
    """조작 대상과 목적지를 각각 잠그고, `attach`/`detach` 시점을 답한다.

    한 에피소드에 하나. 호출자는 프레임마다 `update` 를 부르고, 돌아온 `LatchEvent` 의
    `attach`/`detach` 가 참일 때만 AG3S 의 해당 메서드를 부른다.
    """

    def __init__(self, config: Optional[LatchConfig] = None):
        self.config = config or LatchConfig()
        self._manipulated = _Confirm(self.config)
        self._destination = _Confirm(self.config)
        self.phase = GraspPhase.SEARCHING
        self._open_streak = 0

    # --- 조회 --------------------------------------------------------------------------
    @property
    def manipulated(self) -> Optional[str]:
        return self._manipulated.locked

    @property
    def destination(self) -> Optional[str]:
        return self._destination.locked

    @property
    def holding(self) -> bool:
        return self.phase is GraspPhase.HELD

    # --- 한 프레임 ---------------------------------------------------------------------
    def update(
        self,
        *,
        label: Optional[str],
        score: float = 1.0,
        runner_up: float = 0.0,
        gripper: Optional[float] = None,
        placed: bool = False,
    ) -> LatchEvent:
        """`label` 은 이번 프레임 grounding 이 낸 이름, `gripper` 는 그 손의 그리퍼 값.

        `placed` 는 **외부가 주는 성공 판정**이다 — 조작 대상이 목적지 안에 놓였는가. 이 모듈은
        기하를 보지 않으므로 스스로 판단하지 않는다 (`trajopt/placed.py` 가 참조 구현).

        **`placed` 는 그리퍼가 닫혀 있는 동안 풀지 않는다.** 주입된 판정은 기하만 답할 수
        있고(스냅샷은 손에 강체로 붙어 있어 "이탈" 을 모른다), 이탈은 그리퍼가 답한다.
        그래서 둘을 AND 로 묶는다 — 그것이 없으면 물체가 목적지 위를 지나는 순간 아직 쥔 채로
        detach 된다.
        """
        cfg = self.config
        closed = gripper is not None and float(gripper) < cfg.gripper_closed_below
        attach = detach = False
        note = ""

        if self.phase is GraspPhase.SEARCHING:
            if self._manipulated.update(label, score, runner_up) is not None:
                self.phase = GraspPhase.LATCHED
                note = f"조작 대상 잠금: {self._manipulated.locked}"

        if self.phase is GraspPhase.LATCHED:
            if closed:
                self.phase = GraspPhase.HELD
                attach = True
                self._open_streak = 0
                note = f"그리퍼가 닫혔다 — attach({self._manipulated.locked})"

        elif self.phase is GraspPhase.HELD:
            # 잠금이 걸린 뒤 grounding 이 내는 이름은 조작 대상이 아니라 **목적지**다 (F11 의
            # 뒤집기). 여기서만 목적지 잠금에 먹인다.
            self._destination.update(label, score, runner_up)
            self._open_streak = self._open_streak + 1 if (gripper is not None and not closed) else 0
            # **성공 판정은 그리퍼가 닫혀 있는 동안 풀지 않는다.** 주입된 판정은 기하만
            # 답한다 — 쥔 물체가 목적지 안에 있고 테두리 아래인가 (`trajopt/placed.py`).
            # 손에서 이탈했는지는 그것으로 알 수 없다: `attached` 는 파지 순간의 스냅샷이고
            # 손에 강체로 붙어 있어 손과의 거리가 정의상 안 변한다. 그래서 이탈은 그리퍼가
            # 답하고 여기서 **AND** 로 묶는다. 안 묶으면 물체가 바구니 위를 지나는 순간
            # 아직 쥔 채로 detach 되고, 그러면 쥔 물체가 장애물로 되돌아온다 (E3 의 구멍).
            placed_now = bool(placed) and not closed
            if placed_now or self._open_streak >= cfg.release_frames:
                self.phase = GraspPhase.PLACED
                detach = True
                note = ("성공 판정" if placed_now else
                        f"그리퍼가 {self._open_streak} 프레임 열려 있다") + " — detach()"

        return LatchEvent(
            phase=self.phase,
            manipulated=self._manipulated.locked,
            destination=self._destination.locked,
            attach=attach,
            detach=detach,
            note=note,
        )

    def reset(self) -> None:
        """다음 과제로 넘어간다. 잠금 둘과 상태를 모두 푼다."""
        self._manipulated.release()
        self._destination.release()
        self.phase = GraspPhase.SEARCHING
        self._open_streak = 0


__all__ = ["CentroidIdentity", "GraspLatch", "GraspPhase", "LatchConfig", "LatchEvent"]
