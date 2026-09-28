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

## 파지 확인 (T22) — attach 는 "닫으라고 했다" 가 아니라 "잡았다" 에 붙는다

attach 는 "물체가 손에 붙어 함께 움직인다" 는 충돌 모델의 선언이다 (지침 §6.1). 그래서
`evidence=True` 로 지은 잠금은 그리퍼 **값 하나**로 attach 하지 않는다. 상태가 하나 늘어난다:

    SEARCHING ─(같은 물체 N 프레임)─► LATCHED  대상 확정
    LATCHED   ─(실행된 닫힘 명령)───► CLOSING  닫힘 시도       attach 안 함
    CLOSING   ─(파지 증거 전부)─────► HELD     파지 확인       attach
    CLOSING   ─(명령·개도 다시 열림)► LATCHED  시도 포기
    HELD      ─(개도 열림 N 프레임 · 또는 placed ∧ 열림)─► PLACED  놓임 확인  detach

**파지 증거** (배포에서 쓸 추정치 — 시뮬레이터 참값이 아니다, `observe_grasp`):

* 닫힘 명령이 **실제로 적용**됐다 — 실행 피드백의 `executed` 스텝 중 `applied_gripper` 가 문턱
  아래인 것. HOLD 로 취소된 청크(`n_exec == 0`)의 적용값은 현재 상태라 세지 않는다.
* 그 명령이 `settle_steps` 스텝 이상 적용됐다 — 측정 개도가 정착했다고 볼 수 있다 (T17: 2 스텝).
* 측정 개도가 **줄었다** (`opening_before − opening_after ≥ min_drop`) 그리고 문턱 아래다.
* 측정 개도가 **마지막 실행 명령보다 `blocked_gap` 이상 위에서 멈췄다** — 빈손이면 명령까지
  닫힌다 (위치 제어). 그리고 `empty_below` 보다 위다 (끝까지 닫힘 = 빈손).
* 그 순간 조작 대상(T20 manipulated) 이 있고 (`visible`/`occluded`), parent link 원점과 그
  centroid 의 거리가 `reach` 이하다.

`evidence=False`(기본) 는 T22 전의 규칙 그대로다 — 그리퍼 값이 문턱 아래면 attach. 이 모듈을
단독으로 쓰는 연구 스크립트와 기존 테스트가 그 계약을 읽는다. `SafePolicy` 는
`LatchConfig.legacy_gripper_attach=False`(기본) 일 때 `evidence=True` 로 짓는다.

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
    LATCHED = "latched"       # 대상 확정: 조작 대상이 정해졌다. 아직 쥐지 않았다
    CLOSING = "closing"       # 닫힘 시도: 닫힘 명령이 실제로 적용됐다. 파지는 미확인 (T22, evidence 전용)
    HELD = "held"             # 파지 확인: 쥐었다. `attach` 가 불렸어야 한다
    PLACED = "placed"         # 놓임 확인: 놓았다. `detach` 가 불렸어야 한다


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

    # --- T22: 파지 확인 (evidence) ---------------------------------------------------------
    #: **`SafePolicy` 전용.** True 면 T22 전의 경로 — latch 가 받은 그리퍼 값(피드백이 없으면
    #: 계획 청크 첫 행)이 문턱 아래면 attach. 옛 기록을 옛 규칙으로 재생할 때만 켠다. 기본 False:
    #: **계획값으로 attach 하는 경로가 꺼진다.** `GraspLatch` 단독 사용에는 영향이 없다
    #: (그쪽 모드는 생성자의 `evidence=` 가 정한다).
    legacy_gripper_attach: bool = False
    #: parent link 원점 ↔ 조작 대상 centroid 거리 상한 (m). **잠정값** — T17 MuJoCo FK 로
    #: `ee_finger_l1` body 원점 ↔ 사과 body 가 55 mm (t=117–160), grounding centroid 편차 중앙
    #: 17.6 mm (T17a) 라 여유를 두었다. verifier 가 재서 정한다. T14 의 636 mm 는 어떤 값으로도
    #: 거절돼야 한다. `AG3S.attach` 의 기본 관문(`attached.DEFAULT_ATTACH_REACH_M`)과 같은 값.
    reach: float = 0.12
    #: 닫힘 명령이 이만큼의 **실행된** 스텝 동안 적용돼야 측정 개도를 정착한 값으로 본다.
    #: T17: 명령 0.0 이 t=115 에 적용 → t=116 0.904 → t=117 0.712 (2 스텝). 1 스텝 여유.
    settle_steps: int = 3
    #: 닫힘 시도 직전 개도 − 지금 개도 ≥ 이것 (정규화, 1 = 열림). "줄어들었다".
    min_drop: float = 0.05
    #: 지금 개도 − 마지막 실행 명령 ≥ 이것 — 손가락이 명령까지 못 가고 **막혔다** (물체).
    #: 빈손은 위치 제어라 명령까지 닫힌다. T17: 개도 0.715 대 명령 ≈ 0.0 (파지 순간) · 0.57 (유지).
    #: 0.05 ≈ 손가락 하나 2.25 mm (`RBY1_GRIPPER_OPEN` = −0.045 m).
    blocked_gap: float = 0.05
    #: 개도가 이보다 작으면 끝까지 닫힌 것 = 빈손 (`gripper_finger_l1` 범위 [−0.05, 0] → 정규화 0 이
    #: 완전 닫힘). 0.10 ≈ 물체 폭 9 mm.
    empty_below: float = 0.10


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
    #: 파지 판단의 근거 (T22, `observe_grasp` 만 채운다). 기록(`summary_json.grasp`)에 그대로 간다.
    evidence: Optional[dict] = None


@dataclasses.dataclass(frozen=True)
class GraspSignal:
    """한 요청의 실행 피드백에서 **한 손**에 대해 뽑은 사실 (T22). 계획값은 여기 없다.

    `grasp_signal_from_feedback` 가 만든다. 옛 클라이언트(피드백 없음)면 `available=False` 이고
    `reason` 만 뜻이 있다.
    """

    available: bool
    reason: str = ""
    #: 어느 계획에 대한 사실인가 (`ag3s/exec_feedback.seq`).
    feedback_seq: Optional[int] = None
    n_steps: int = 0
    n_exec: int = 0
    #: 실행된 스텝 중 적용 개도가 문턱 아래였던 스텝 수 — **실제로 적용된 닫힘 명령**.
    closing_steps: int = 0
    #: 마지막 **실행된** 스텝의 적용 개도. 실행된 스텝이 없으면 `None` (HOLD 의 적용값은 현재
    #: 상태라 명령이 아니다).
    last_command: Optional[float] = None
    #: 이번 요청 촬영 시점의 측정 개도 (1 = 열림).
    opening: Optional[float] = None


def grasp_signal_from_feedback(feedback: Optional[dict], hand_index: int,
                               closed_below: float) -> GraspSignal:
    """`wire.unpack_exec_feedback` 의 결과에서 손 하나(`0` 왼, `1` 오른)의 사실을 뽑는다.

    **HOLD 스텝(`executed[i] == False`)의 적용값은 세지 않는다.** 그때 `d.ctrl` 에 들어간 것은
    `rby1_state()` — 지금 개도 — 이고, 물체를 쥔 채 HOLD 면 그 값이 0.72 라 "닫힘 명령" 처럼
    보인다 (T16: 빈손 HOLD 에서도 측정 −0.037 m = 0.82 < 0.85 가 나왔다).
    """
    fb = feedback or {}
    if not fb.get("available"):
        return GraspSignal(available=False,
                           reason=str(fb.get("reason") or "no execution feedback"))
    import numpy as np

    flags = [bool(v) for v in (fb.get("executed") or ())]
    applied = np.asarray(fb.get("applied_gripper", np.zeros((0, 2))), np.float64).reshape(-1, 2)
    col = applied[:, int(hand_index)] if applied.shape[0] == len(flags) else np.zeros(0)
    executed = [float(col[i]) for i, f in enumerate(flags) if f and i < col.shape[0]]
    measured = np.asarray(fb.get("measured_gripper", ()), np.float64).reshape(-1)
    opening = float(measured[int(hand_index)]) if measured.shape[0] > int(hand_index) else None
    return GraspSignal(
        available=True,
        feedback_seq=None if fb.get("seq") is None else int(fb["seq"]),
        n_steps=len(flags),
        n_exec=int(sum(flags)),
        closing_steps=int(sum(1 for v in executed if v < float(closed_below))),
        last_command=executed[-1] if executed else None,
        opening=opening,
    )


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

    def __init__(self, config: Optional[LatchConfig] = None, *, evidence: bool = False):
        self.config = config or LatchConfig()
        #: T22. True 면 attach/detach 는 `observe_grasp`(실행 피드백 + 조작 대상 + reach)만 낸다.
        #: `update` 는 대상 확정과 목적지 학습만 하고 그리퍼 값으로 상태를 바꾸지 않는다.
        self.evidence = bool(evidence)
        self._manipulated = _Confirm(self.config)
        self._destination = _Confirm(self.config)
        self.phase = GraspPhase.SEARCHING
        self._open_streak = 0
        self._reset_grasp()

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

        if self.evidence:
            # T22 — 그리퍼 값으로는 상태를 바꾸지 않는다. 파지·놓임은 `observe_grasp` 가 낸다.
            # 여기서 하는 것은 대상 확정(위)과 목적지 학습뿐이다.
            if self.phase is GraspPhase.HELD:
                self._destination.update(label, score, runner_up)
        elif self.phase is GraspPhase.LATCHED:
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

    # --- T22: 실행 피드백으로 파지 상태를 갱신한다 --------------------------------------------
    def observe_grasp(
        self,
        signal: GraspSignal,
        *,
        reach_m: Optional[float] = None,
        manipulated_id: Optional[int] = None,
        manipulated_state: Optional[str] = None,
        placed: bool = False,
    ) -> LatchEvent:
        """직전 청크의 **실행 사실**로 닫힘 시도 · 파지 확인 · 놓임 확인을 갱신한다 (`evidence=True`).

        호출자는 이것을 **제약을 짓기 전에** 부른다 (지침 §6.3: 피드백 → 파지 상태 → 기하 → TO).
        그래야 attach/detach 가 이번 프레임의 거리장·제약에 바로 들어간다.

        Args:
            signal: `grasp_signal_from_feedback` 의 결과 (그 손).
            reach_m: parent link 원점 ↔ 조작 대상 centroid 거리 (지금 자세). 모르면 `None` — 확인 불가.
            manipulated_id / manipulated_state: T20 `AG3S.manipulated` 의 id · 상태
                (`visible`/`occluded`/`lost`). 없으면 `None`.
            placed: 외부 성공 판정 (`placed_fn`). **측정 개도가 열렸을 때만** 풀기에 쓴다.
        """
        if not self.evidence:
            raise RuntimeError("observe_grasp needs GraspLatch(evidence=True)")
        cfg = self.config
        ev: dict = {
            "feedback_seq": signal.feedback_seq,
            "feedback_available": bool(signal.available),
            "n_exec": int(signal.n_exec),
            "n_steps": int(signal.n_steps),
            "closing_steps": int(signal.closing_steps),
            "opening_before": self._opening_before,
            "opening_after": signal.opening,
            "opening_previous": self._last_opening,
            "last_command": self._last_command,
            "settled_steps": int(self._settled_steps),
            "reach_mm": None if reach_m is None else round(float(reach_m) * 1000.0, 2),
            "reach_limit_mm": round(float(cfg.reach) * 1000.0, 2),
            "manipulated_id": manipulated_id,
            "manipulated_state": manipulated_state,
            "checks": {},
        }
        attach = detach = False
        note = ""

        if not signal.available:
            # 옛 클라이언트 · 첫 청크 · 모양 틀림. **계획값으로 대신하지 않는다** — 그것이 T22 의 요지다.
            note = (f"실행 피드백 없음 ({signal.reason}) — 파지 상태 그대로 ({self.phase.value}), "
                    "계획값으로 attach 하지 않는다")
            self._last_grasp = self._event(False, False, note, ev)
            return self._last_grasp

        opening = signal.opening
        if self.phase is GraspPhase.LATCHED and signal.closing_steps > 0:
            self.phase = GraspPhase.CLOSING
            self._opening_before = self._last_opening
            self._settled_steps = 0
            note = (f"닫힘 시도: 실행된 닫힘 명령 {signal.closing_steps} 스텝 "
                    f"(feedback seq {signal.feedback_seq})")
        elif self.phase in (GraspPhase.SEARCHING, GraspPhase.PLACED) and signal.closing_steps > 0:
            note = (f"닫힘 명령 {signal.closing_steps} 스텝이 실행됐지만 확정된 대상이 없다 "
                    f"({self.phase.value}) — attach 하지 않는다")

        if self.phase is GraspPhase.CLOSING:
            self._settled_steps += int(signal.closing_steps)
            if signal.last_command is not None:
                self._last_command = float(signal.last_command)
            ev["opening_before"] = self._opening_before
            ev["last_command"] = self._last_command
            ev["settled_steps"] = int(self._settled_steps)
            reopened = (signal.last_command is not None
                        and signal.last_command >= cfg.gripper_closed_below
                        and opening is not None and opening >= cfg.gripper_closed_below)
            if reopened:
                self.phase = GraspPhase.LATCHED
                note = (f"닫힘 시도 포기: 명령 {signal.last_command:.3f} · 개도 {opening:.3f} 가 "
                        "다시 열렸다 — attach 없음")
                self._reset_grasp(keep_opening=True)
            else:
                checks = self._grasp_checks(opening, reach_m, manipulated_id, manipulated_state)
                ev["checks"] = checks
                failed = [k for k, v in checks.items() if not v]
                if not failed:
                    self.phase = GraspPhase.HELD
                    self._open_streak = 0
                    attach = True
                    note = (f"파지 확인 — attach(manipulated id={manipulated_id}): 개도 "
                            f"{self._opening_before:.3f}→{opening:.3f}, 명령 {self._last_command:.3f}, "
                            f"reach {reach_m * 1000.0:.1f} mm ≤ {cfg.reach * 1000.0:.0f} mm, "
                            f"feedback seq {signal.feedback_seq}")
                else:
                    note = ((note + "; ") if note else "") + (
                        "닫힘 시도 — 파지 미확인: " + ", ".join(failed)
                        + (" (빈손 닫힘으로 보인다)" if "not_empty" in failed
                           or ("blocked" in failed and checks.get("settled")) else ""))

        elif self.phase is GraspPhase.HELD:
            open_now = opening is not None and opening >= cfg.gripper_closed_below
            self._open_streak = self._open_streak + 1 if open_now else 0
            placed_now = bool(placed) and open_now
            ev["open_streak"] = int(self._open_streak)
            ev["placed"] = bool(placed)
            if placed_now or self._open_streak >= cfg.release_frames:
                self.phase = GraspPhase.PLACED
                detach = True
                note = ((f"성공 판정 + 개도 {opening:.3f} 열림" if placed_now else
                         f"개도가 {self._open_streak} 프레임 열려 있다 ({opening:.3f})")
                        + f" — detach() (feedback seq {signal.feedback_seq})")

        if opening is not None:
            self._last_opening = float(opening)
        self._last_grasp = self._event(attach, detach, note, ev)
        return self._last_grasp

    def revert_attach(self, reason: str) -> LatchEvent:
        """`observe_grasp` 가 attach 를 냈는데 **마지막 관문**(`AG3S.attach` 의 reach 검사)이 거절했다.

        HELD 에 두면 잠금은 "쥐었다" 고 믿고 AG3S 는 아무것도 안 붙인 채 갈라진다. evidence 모드는
        CLOSING 으로 되돌려 다음 프레임에 다시 판정한다. 옛 모드(`evidence=False`)에는 CLOSING 이
        없으므로 LATCHED 로 — 다음 닫힘 값에 다시 시도한다.
        """
        if self.phase is GraspPhase.HELD:
            self.phase = GraspPhase.CLOSING if self.evidence else GraspPhase.LATCHED
        note = f"attach 거절 — {reason}; {self.phase.value} 로 되돌림"
        ev = dict((self._last_grasp.evidence if self._last_grasp else None) or {})
        ev["rejected"] = str(reason)
        self._last_grasp = self._event(False, False, note, ev)
        return self._last_grasp

    @property
    def last_grasp(self) -> Optional[LatchEvent]:
        """마지막 `observe_grasp`/`revert_attach` 의 결과 (기록용)."""
        return self._last_grasp

    def _grasp_checks(self, opening, reach_m, manipulated_id, manipulated_state) -> dict:
        """파지 확인의 조건. **전부 참이어야** attach 한다. 이름이 곧 기록의 키다."""
        cfg = self.config
        known = opening is not None
        return {
            "settled": self._settled_steps >= int(cfg.settle_steps),
            "closed": known and opening < cfg.gripper_closed_below,
            "dropped": (known and self._opening_before is not None
                        and self._opening_before - opening >= cfg.min_drop),
            "not_empty": known and opening >= cfg.empty_below,
            "blocked": (known and self._last_command is not None
                        and opening - self._last_command >= cfg.blocked_gap),
            "manipulated": (manipulated_id is not None
                            and manipulated_state in ("visible", "occluded")),
            "reach": reach_m is not None and float(reach_m) <= float(cfg.reach),
        }

    def _event(self, attach: bool, detach: bool, note: str, evidence: dict) -> LatchEvent:
        evidence = dict(evidence)
        evidence["state"] = self.phase.value
        return LatchEvent(
            phase=self.phase,
            manipulated=self._manipulated.locked,
            destination=self._destination.locked,
            attach=attach,
            detach=detach,
            note=note,
            evidence=evidence,
        )

    def _reset_grasp(self, *, keep_opening: bool = False) -> None:
        #: 닫힘 시도가 시작되기 직전의 측정 개도.
        self._opening_before: Optional[float] = None
        #: 닫힘 시도 뒤 **실행된** 닫힘 명령 스텝의 누적.
        self._settled_steps = 0
        #: 마지막으로 실행된 명령 (HOLD 청크는 갱신하지 않는다).
        self._last_command: Optional[float] = None
        if not keep_opening:
            #: 직전 요청의 측정 개도.
            self._last_opening: Optional[float] = None
            self._last_grasp: Optional[LatchEvent] = None

    def reset(self) -> None:
        """다음 과제로 넘어간다. 잠금 둘과 상태를 모두 푼다."""
        self._manipulated.release()
        self._destination.release()
        self.phase = GraspPhase.SEARCHING
        self._open_streak = 0
        self._reset_grasp()


__all__ = ["CentroidIdentity", "GraspLatch", "GraspPhase", "GraspSignal", "LatchConfig", "LatchEvent",
           "grasp_signal_from_feedback"]
