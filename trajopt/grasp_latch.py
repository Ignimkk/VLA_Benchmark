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

**T26 (2026-09-28) — 목적지는 밖에서 받을 수 있다.** T26 뒤 AG3S 의 target 은 파지 뒤에도 조작
대상(사과)이다 — crate 는 admissible 이 아니어서 target 이 될 수 없다. 그러면 위의 "잠금 뒤
grounding 이 내는 이름 = 목적지" 는 사과를 목적지로 잠근다. 그래서 `GraspLatch(...,
external_destination=True)` 는 이름으로 목적지를 배우지 않고 `set_destination(label)` 으로만
받는다 (`SafePolicy` 는 AG3S 가 등록한 destination 의 이름을 넘긴다). 잠금·파지 규칙(T22)은
그대로다 — 바뀐 것은 목적지를 받는 입구뿐이다. 기본(False) 은 예전 그대로 이름으로 배운다.

## 파지 확인 (T22) — attach 는 "닫으라고 했다" 가 아니라 "잡았다" 에 붙는다

attach 는 "물체가 손에 붙어 함께 움직인다" 는 충돌 모델의 선언이다 (지침 §6.1). 그래서
`evidence=True` 로 지은 잠금은 그리퍼 **값 하나**로 attach 하지 않는다. 상태가 하나 늘어난다:

    SEARCHING ─(같은 물체 N 프레임)─► LATCHED  대상 확정
    LATCHED   ─(실행된 닫힘 명령)───► CLOSING  닫힘 시도       attach 안 함
    CLOSING   ─(파지 증거 전부)─────► HELD     파지 확인       attach
    CLOSING   ─(명령·개도 다시 열림)► LATCHED  시도 포기
    HELD      ─(개도 열림 N 프레임 · 또는 placed ∧ 열림)─► PLACED  놓임 확인  detach
    HELD      ─(attach 뒤 N 실행 청크 안의 반증, T34)─► LATCHED  attach_revoked  detach
    HELD      ─(닫힌 채 개도가 명령까지 닫힘 N 요청, T43 Z3 · 기본 off)─► PLACED  slipped  detach

**파지 증거** (배포에서 쓸 추정치 — 시뮬레이터 참값이 아니다, `observe_grasp`):

* 닫힘 명령이 **실제로 적용**됐다 — 실행 피드백의 `executed` 스텝 중 `applied_gripper` 가 문턱
  아래인 것. HOLD 로 취소된 청크(`n_exec == 0`)의 적용값은 현재 상태라 세지 않는다.
* 측정 개도가 **멈췄다** — 연속 두 요청의 측정 개도 차가 `settle_epsilon` 미만이다 (T30 F4).
  예전(T22)에는 "닫힘 명령이 `settle_steps` 스텝 이상 적용됐다" 였다. 명령을 센 것이지 손가락을 본
  것이 아니어서, 청크 경계에서 손가락이 아직 움직이는 중(0.16–0.76)인 빈손 닫힘이 24 중 10 번
  attach 됐다 (T22 verify #4e). T28 E3b ep1808 r1 t=120 도 같은 모양이다 (0.997 → 0.364, 그 뒤
  명령 0.43 을 따라 0.43 으로 되돌아감 — 들린 적 없음).
* 측정 개도가 **줄었다** (`opening_before − opening_after ≥ min_drop`) 그리고 문턱 아래다.
* 측정 개도가 **마지막 실행 명령보다 `blocked_gap` 이상 위에서 멈췄다** — 빈손이면 명령까지
  닫힌다 (위치 제어). 그리고 `empty_below` 보다 위다 (끝까지 닫힘 = 빈손).
* 그 순간 조작 대상(T20 manipulated) 이 있고 (`visible`/`occluded`), parent link 원점과 그
  centroid 의 거리가 `reach` 이하다.

`evidence=False`(기본) 는 T22 전의 규칙 그대로다 — 그리퍼 값이 문턱 아래면 attach. 이 모듈을
단독으로 쓰는 연구 스크립트와 기존 테스트가 그 계약을 읽는다. `SafePolicy` 는
`LatchConfig.legacy_gripper_attach=False`(기본) 일 때 `evidence=True` 로 짓는다.

## 거짓 attach 회수 (T34 J2)

증거가 다 맞아도 attach 가 틀릴 수 있다 — T33 E3b ep1808 r1 은 개도가 명령을 느리게 따라가다
멈춘 순간에 attach 했고, 손이 들렸는데 사과는 테이블에 남았다 (t=232 에 열려서야 detach, 그동안
쥔 구가 허공을 들고 다녔다). 그래서 attach 뒤 `revoke_chunks` 실행 청크 동안 다시 본다: 개도가
계속 "쥐고 있음" 인가 (더 닫히지 않았나 · 명령에 막혀 있나), 조작 대상이 보이면 쥔 구 자리에
있나, 손을 들었으면 따라 올라왔나. 어긋나면 detach 하고 `attach_revoked` 를 남기며 LATCHED 로
돌아간다 (PLACED 가 아니다 — 대상은 그대로이고 다음 닫힘을 다시 판정한다).

## 닫힌 채 손에서 빠짐 (T43 Z3, `LatchConfig.slip_detach`, 기본 off)

놓기 규칙(열림 N 프레임 · placed ∧ 열림)은 그리퍼가 열려야 답한다. T43 W V5 1925 s19253 은 HOLD 중
gripper 명령이 닫힌 채(0.59) 사과가 t 369 에 빠졌고, 그 뒤 서버는 끝까지 attached — 쥔 구(body
cover)가 빈손에 붙은 채 crate 와 겹쳐 HOLD 를 냈다. 빠지면 손가락을 벌려 두던 것이 없어지므로
손가락이 마지막 실행 명령까지 닫힌다 (위치 제어 — T22 `blocked` 의 반대). 그래서 attach 순간 개도보다
`slip_drop` 이상 더 닫혔고 명령과의 차가 `slip_gap` 아래인 요청이 `slip_frames` 번 이어지면 detach 하고
PLACED 로 간다. 조작 대상 관측(target cluster)은 쓰지 않는다: 쥔 동안 self-filter · held_free 가 그 점을
지워 조작 대상은 `occluded` 로 attach 전 마지막 관측에 동결돼 있다 (T43 W V4 · V5 38 run 에서 HELD 로
끝난 요청 720 개 전부 — 관측 3 번은 모두 그 요청에서 attach 가 회수된 뒤다).

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
    #: **측정 개도가 멈췄다** 의 문턱 (T30 F4, 정규화 개도 — 1 = 열림). 이번 요청의 측정 개도와
    #: 직전 요청의 측정 개도 차가 이보다 작아야 `settled`. 명령 스텝을 세지 않는다.
    #:
    #: 0.02 = 손가락 하나 0.9 mm / 청크 (`RBY1_GRIPPER_OPEN` −0.045 m) — fine 격자 5 mm 보다 한참
    #: 작아서, 이 안에서 움직이는 손가락에 붙인 물체의 모델 오차는 무시할 만하다. 근거 (요청 간 차):
    #:
    #: | 기록 | 경우 | 차 |
    #: |---|---|---|
    #: | T17 실제 파지 t=120 → 128 | 쥔 뒤 정착 (0.715 → 0.720) | 0.0051 — 통과해야 한다 |
    #: | T17 t=128–216 | 쥔 채 유지 | ≤ 0.0006 |
    #: | T17 · T16 · T28 E3a·E3b 12 에피소드 (14 기록) | 청크 내내 스텝당 < 0.003 인 요청 793 개 | 중앙 0.0001 · p99 0.010 · 최대 0.0196 |
    #: | T22 빈손 sweep 거짓 attach 10 건 | 청크 경계에서 이동 중 | 0.24–0.84 |
    #: | T28 E3b ep1808 r1 t=120 거짓 attach | 이동 중 (0.997 → 0.364) | 0.633, 다음 요청 0.067 |
    #: | T28 E3b ep1808 r2 t=128 attach (옛 규칙) | 이동 중 (0.991 → 0.775 → 0.572) | 0.216 |
    #:
    #: 느린 추종(부분 명령 ≈ 0.43 을 따라 스텝당 ≤ 0.02 로 움직임)은 0.02 아래로 들어올 수 있다 —
    #: 차 하나로는 못 가른다. 그 경우 개도 ≈ 명령이라 `blocked` 가 거절한다 (빈손/쥔 손 구별은 T22 그대로).
    settle_epsilon: float = 0.02
    #: 닫힘 시도 직전 개도 − 지금 개도 ≥ 이것 (정규화, 1 = 열림). "줄어들었다".
    min_drop: float = 0.05
    #: 지금 개도 − 마지막 실행 명령 ≥ 이것 — 손가락이 명령까지 못 가고 **막혔다** (물체).
    #: 빈손은 위치 제어라 명령까지 닫힌다. T17: 개도 0.715 대 명령 ≈ 0.0 (파지 순간) · 0.57 (유지).
    #: 0.05 ≈ 손가락 하나 2.25 mm (`RBY1_GRIPPER_OPEN` = −0.045 m).
    blocked_gap: float = 0.05
    #: 개도가 이보다 작으면 끝까지 닫힌 것 = 빈손 (`gripper_finger_l1` 범위 [−0.05, 0] → 정규화 0 이
    #: 완전 닫힘). 0.10 ≈ 물체 폭 9 mm.
    empty_below: float = 0.10

    # --- T34 J2: 거짓 attach 회수 (`attach_revoked`) -------------------------------------------
    #: attach 뒤 **실행된** 청크 몇 개 동안 쥔 상태를 다시 확인하나. HOLD 청크(실행 0)는 세지 않는다
    #: — 로봇이 안 움직이면 새 증거도 없다. 근거 (T33 기록, `T34.impl.md` §2):
    #:
    #: | 기록 | 첫 반증이 보인 실행 청크 (attach 뒤 k 번째) |
    #: |---|---|
    #: | E3b ep1808 r1 거짓 attach | 개도 k=1 (0.684 → 0.555 = 명령 0.552) · 관측 k=3 (사과가 테이블에 보임, 쥔 구에서 215 mm) |
    #: | E3a ep1800 r1 (들다 떨어뜨림, 실패) | k=2 (개도 0.551 = 명령 0.550, GT 사과 다시 테이블) |
    #: | 성공 run 의 첫 정상 놓기 | k ≥ 8 (E3b ep1808 r2 k=8 crate 위 낙하 · 나머지 개도 열림 k ≥ 10) |
    #:
    #: 4 = 거짓 attach 의 마지막 증거(k=3) + 1, 정상 놓기(k ≥ 8)의 절반.
    revoke_chunks: int = 4
    #: attach 순간 개도 − 지금 개도 ≥ 이것이면 "쥐고 있음" 이 아니다 (손가락이 물체 없이 더 닫혔다).
    #: 참 attach 는 창 안에서 최대 0.015 (0.701 → 0.686) 움직였고, 거짓·떨어뜨림은 0.13–0.15.
    revoke_drop: float = 0.05
    #: 관측된 조작 대상 centroid 가 쥔 주 구 중심에서 (주 구 반지름 + 이것) 밖이면 쥔 물체가 아니다.
    #: 0.040 = 측정된 손 안 미끄럼 최대 24 mm (T32a) 의 두 배 가까이. 참 파지 뒤 창 안에서 조작 대상이
    #: 관측된 적은 T33 에서 **한 번도 없다** (self-filter 가 지운다); 거짓 attach 는 206–238 mm.
    revoke_centre_slack: float = 0.040
    #: 손이 이만큼 (m, 위로) 들렸는데 관측된 물체가 `revoke_follow_ratio` 만큼도 따라오지 않으면 회수.
    #: 참 들기는 첫 실행 청크에 75–88 mm, 들지 않은 흔들림은 ≤ 11 mm (T33).
    revoke_lift_min: float = 0.030
    revoke_follow_ratio: float = 0.5
    #: 개도 두 검사(`opening_dropped` · `not_blocked`)를 쓰나. 측정·대조군용 스위치 (관측 두 검사만 남는다).
    revoke_use_opening: bool = True

    # --- T43 Z3: 닫힌 채 손에서 빠짐 (`slip`) -------------------------------------------------------
    #: **기본 False = Z3 전과 비트 동일.** True 면 HELD 에서 (회수 창 · 열림 놓기가 답하지 않은 요청에)
    #: 측정 개도로 "물체가 손을 떠났다" 를 본다 — 쥔 물체가 손가락을 벌려 두던 것이 없어져 손가락이
    #: 마지막 실행 명령까지 닫혔다:
    #:
    #:     drop = attach 순간 개도 − 지금 개도 ≥ `slip_drop`   (물체 폭이 허락하는 것보다 더 닫혔다)
    #:     gap  = 지금 개도 − 마지막 실행 명령 < `slip_gap`     (명령에 막혀 있지 않다)
    #:
    #: 둘 다 `slip_frames` 요청 연속이면 detach 하고 PLACED 로 간다 (`LatchEvent.slipped`). 그 뒤 물체는
    #: 쥔 구 · body cover 없이 장면 물체다 (열림 놓기와 같은 뒤처리 — `SafePolicy` 가 `detach()` ·
    #: `set_grasp_active(False)` · `set_placed(True)` 를 부른다). 판정은 실행 피드백(측정 개도 · 실행
    #: 명령)만 본다 — 시뮬레이터 참값을 보지 않는다. 근거: T43 W V4 · V5 38 run (`T43Z3.impl.md`) —
    #: MuJoCo release 전 HELD 요청 368 개에서 drop ≤ 0.031 이고 두 조건이 함께 선 요청은 0 개. Y 의 닫힌 채
    #: 빠진 4 run 은 빠진 뒤 첫 요청 (1925 s19253 · B 1807 s18071 · H 1807 s18071: drop 0.108–0.135, gap
    #: 0.002–0.036) 또는 둘째 요청 (H 1807 s18073 — 첫 요청이 빠진 지 1 스텝 뒤: drop 0.165, gap −0.001)
    #: 부터 둘 다 선다. 마지막 실행 명령은 HOLD 청크(실행 0)에서 갱신하지 않고 들고 간다; attach 순간의
    #: 값은 닫힘 시도의 마지막 실행 명령이다.
    slip_detach: bool = False
    #: attach 순간 개도 − 지금 개도의 하한 (정규화, 1 = 열림). `revoke_drop` 과 같은 0.05 — 참 release
    #: 전 최대 0.031 (V5 1834 s18343, 빠지기 직전 사과가 손 안에서 천천히 미끄러질 때) 의 1.6 배.
    slip_drop: float = 0.05
    #: 지금 개도 − 마지막 실행 명령의 상한. `blocked_gap` 과 같은 0.05 — 닫힌 명령으로 쥔 동안 0.114 이상.
    #: (정책이 열기 시작하면 gap 은 음수가 되지만 그때 손가락은 벌어지므로 drop 이 음수다 — 그래서 AND.)
    slip_gap: float = 0.05
    #: 연속 몇 요청이어야 detach 하나. `release_frames` 와 같은 2 — 한 요청의 측정 튐으로 쥔 물체를
    #: 잃지 않는다. 대가는 한 청크(8 스텝) 늦은 detach.
    slip_frames: int = 2


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
    #: T34 J2: 이번 `detach` 는 놓기가 아니라 **거짓 attach 의 회수**다 (`attach_revoked`). 상태는
    #: `LATCHED` 로 돌아간다 — 대상은 그대로 잠겨 있고 다음 닫힘을 다시 판정한다.
    revoked: bool = False
    #: T43 Z3: 이번 `detach` 는 열림 놓기가 아니라 **닫힌 채 손에서 빠짐**이다 (`LatchConfig.slip_detach`).
    #: 상태는 열림 놓기와 같은 `PLACED` 다.
    slipped: bool = False


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


def _delta(opening, previous) -> Optional[float]:
    """연속 두 요청의 측정 개도 차 `|opening − previous|`. 하나라도 없으면 `None`."""
    if opening is None or previous is None:
        return None
    return round(abs(float(opening) - float(previous)), 6)


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

    def __init__(self, config: Optional[LatchConfig] = None, *, evidence: bool = False,
                 external_destination: bool = False):
        self.config = config or LatchConfig()
        #: T22. True 면 attach/detach 는 `observe_grasp`(실행 피드백 + 조작 대상 + reach)만 낸다.
        #: `update` 는 대상 확정과 목적지 학습만 하고 그리퍼 값으로 상태를 바꾸지 않는다.
        self.evidence = bool(evidence)
        #: T26. True 면 목적지를 grounding 이름으로 배우지 않고 `set_destination` 으로만 받는다.
        self.external_destination = bool(external_destination)
        self._external_destination: Optional[str] = None
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
        if self.external_destination:
            return self._external_destination
        return self._destination.locked

    def set_destination(self, label: Optional[str]) -> None:
        """목적지를 밖에서 준다 (T26, `external_destination=True` 전용). `None` = 아직 모른다."""
        if not self.external_destination:
            raise RuntimeError("set_destination needs GraspLatch(external_destination=True)")
        self._external_destination = None if label is None else str(label)

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
            if self.phase is GraspPhase.HELD and not self.external_destination:
                self._destination.update(label, score, runner_up)
        elif self.phase is GraspPhase.LATCHED:
            if closed:
                self.phase = GraspPhase.HELD
                attach = True
                self._open_streak = 0
                note = f"그리퍼가 닫혔다 — attach({self._manipulated.locked})"

        elif self.phase is GraspPhase.HELD:
            # 잠금이 걸린 뒤 grounding 이 내는 이름은 조작 대상이 아니라 **목적지**다 (F11 의
            # 뒤집기). 여기서만 목적지 잠금에 먹인다. (T26: 밖에서 받는 모드면 먹이지 않는다.)
            if not self.external_destination:
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
            destination=self.destination,
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
        held_observation: Optional[dict] = None,
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
            held_observation: T34 J2 — 직전 프레임에 조작 대상이 **관측됐을 때** 쥔 구와의 비교
                (`AG3S.held_observation()`: `centre_distance_m`, `held_radius_m`, `hand_rise_m`,
                `object_rise_m`). 관측이 없으면 `None` — 그 두 검사는 판단하지 않는다.
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
            "opening_delta": _delta(signal.opening, self._last_opening),
            "settle_epsilon": float(cfg.settle_epsilon),
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
                checks = self._grasp_checks(opening, self._last_opening, reach_m,
                                            manipulated_id, manipulated_state)
                ev["checks"] = checks
                failed = [k for k, v in checks.items() if not v]
                if not failed:
                    self.phase = GraspPhase.HELD
                    self._open_streak = 0
                    attach = True
                    # T34 J2: 회수 창을 연다 — attach 순간의 개도가 "쥐고 있음" 의 기준이다.
                    self._attach_opening = None if opening is None else float(opening)
                    self._held_exec_chunks = 0
                    # T43 Z3: 쥔 동안의 "마지막 실행 명령" 은 닫힘 시도의 마지막 실행 명령에서 시작한다.
                    self._hold_command = self._last_command
                    self._slip_streak = 0
                    note = (f"파지 확인 — attach(manipulated id={manipulated_id}): 개도 "
                            f"{self._opening_before:.3f}→{opening:.3f} (정지: 직전 요청과 차 "
                            f"{ev['opening_delta']:.4f} < {cfg.settle_epsilon:g}), "
                            f"명령 {self._last_command:.3f}, "
                            f"reach {reach_m * 1000.0:.1f} mm ≤ {cfg.reach * 1000.0:.0f} mm, "
                            f"feedback seq {signal.feedback_seq}")
                else:
                    note = ((note + "; ") if note else "") + (
                        "닫힘 시도 — 파지 미확인: " + ", ".join(failed)
                        + (" (빈손 닫힘으로 보인다)" if "not_empty" in failed
                           or ("blocked" in failed and checks.get("settled")) else "")
                        + (f" (손가락이 아직 움직인다: 직전 요청과 차 {ev['opening_delta']:.3f} ≥ "
                           f"{cfg.settle_epsilon:g})" if "settled" in failed
                           and ev["opening_delta"] is not None else ""))

        elif self.phase is GraspPhase.HELD:
            open_now = opening is not None and opening >= cfg.gripper_closed_below
            self._open_streak = self._open_streak + 1 if open_now else 0
            placed_now = bool(placed) and open_now
            ev["open_streak"] = int(self._open_streak)
            ev["placed"] = bool(placed)
            failed_hold = self._hold_checks(signal, opening, open_now, held_observation, ev)
            if failed_hold:
                # T34 J2 — 거짓 attach 회수. 놓기(PLACED)가 아니다: 대상은 잠긴 채 LATCHED 로 돌아가
                # 다음 닫힘을 처음부터 다시 판정한다 (PLACED 면 그 에피소드에서 다시 attach 할 수 없다).
                self.phase = GraspPhase.LATCHED
                self._open_streak = 0
                ev["revoke"] = {"reasons": failed_hold, "window": dict(ev.get("hold_window") or {})}
                note = (f"attach_revoked — 쥔 상태가 아니다: {', '.join(failed_hold)} "
                        f"(attach 뒤 실행 청크 {self._held_exec_chunks}/{cfg.revoke_chunks}) — "
                        f"detach(), 대상 잠금은 유지 (feedback seq {signal.feedback_seq})")
                self._reset_grasp(keep_opening=True)
                if opening is not None:
                    self._last_opening = float(opening)
                self._last_grasp = self._event(False, True, note, ev, revoked=True)
                return self._last_grasp
            slipped = self._slip_check(signal, opening, ev) if cfg.slip_detach else False
            if placed_now or self._open_streak >= cfg.release_frames:
                self.phase = GraspPhase.PLACED
                detach = True
                note = ((f"성공 판정 + 개도 {opening:.3f} 열림" if placed_now else
                         f"개도가 {self._open_streak} 프레임 열려 있다 ({opening:.3f})")
                        + f" — detach() (feedback seq {signal.feedback_seq})")
            elif slipped:
                # T43 Z3 — 닫힌 채 손에서 빠졌다. 열림 놓기와 같은 PLACED 로 간다: 그 뒤 물체는 장면
                # 물체이고 (쥔 구 · body cover 없음), 이 에피소드에서 다시 attach 하지 않는다.
                self.phase = GraspPhase.PLACED
                detach = True
                s = ev["slip"]
                note = (f"slip — 닫힌 채 손에서 빠졌다: 개도 {opening:.3f} (attach {s['attach_opening']:.3f}, "
                        f"drop {s['drop']:.3f} ≥ {cfg.slip_drop:g}), 마지막 실행 명령 "
                        f"{s['hold_command']:.3f} (gap {s['gap']:.3f} < {cfg.slip_gap:g}), "
                        f"{s['streak']}/{cfg.slip_frames} 요청 — detach() "
                        f"(feedback seq {signal.feedback_seq})")
                if opening is not None:
                    self._last_opening = float(opening)
                self._last_grasp = self._event(False, True, note, ev, slipped=True)
                return self._last_grasp

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

    def _grasp_checks(self, opening, previous, reach_m, manipulated_id,
                      manipulated_state) -> dict:
        """파지 확인의 조건. **전부 참이어야** attach 한다. 이름이 곧 기록의 키다.

        `settled` (T30 F4): 이번 요청의 측정 개도 `opening` 과 직전 요청의 측정 개도 `previous` 의
        차 < `settle_epsilon` — **손가락이 멈췄다.** 직전 측정이 없으면 거짓 (fail-closed).
        """
        cfg = self.config
        known = opening is not None
        delta = _delta(opening, previous)
        return {
            "settled": delta is not None and delta < float(cfg.settle_epsilon),
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

    def _hold_checks(self, signal: GraspSignal, opening, open_now: bool,
                     held_observation: Optional[dict], ev: dict) -> list:
        """T34 J2 — attach 뒤 `revoke_chunks` 실행 청크 동안 "정말 쥐고 있나" 를 본다.

        돌려주는 것은 어긋난 검사 이름 목록 (비면 유지). 네 가지다:

        * `opening_dropped` — attach 순간 개도 − 지금 ≥ `revoke_drop` (손가락이 더 닫혔다).
        * `not_blocked` — 이번 청크의 마지막 **실행** 명령이 attach 개도보다 `blocked_gap` 이상
          조이는데 개도가 그 명령까지 갔다 (`개도 − 명령 < blocked_gap`, T22 의 `blocked` 와 같은 뜻).
          명령이 attach 개도 근처이거나 푸는 쪽이면 판단하지 않는다 — 놓기는 `open_streak` 의 일이다.
        * `observed_elsewhere` — 관측된 조작 대상 centroid 가 쥔 주 구 밖 (`revoke_centre_slack`).
        * `not_following` — 손이 `revoke_lift_min` 이상 들렸는데 관측된 물체가 그 `revoke_follow_ratio`
          만큼도 안 올랐다.

        개도가 열렸으면(`open_now`) 판단하지 않는다 — 놓기다. 창이 지나면 아무것도 안 본다.
        """
        cfg = self.config
        if signal.n_exec > 0:
            self._held_exec_chunks += 1
        window = {"exec_chunks": int(self._held_exec_chunks), "limit": int(cfg.revoke_chunks),
                  "attach_opening": self._attach_opening}
        ev["hold_window"] = window
        if self._held_exec_chunks > int(cfg.revoke_chunks) or open_now:
            window["active"] = False
            return []
        window["active"] = True
        checks: dict = {}
        a = self._attach_opening
        if a is not None and opening is not None:
            checks["opening_dropped"] = round(a - float(opening), 4)
        cmd = signal.last_command
        tightening = (a is not None and cmd is not None
                      and float(cmd) <= a - float(cfg.blocked_gap))
        if tightening and opening is not None:
            checks["gap_to_command"] = round(float(opening) - float(cmd), 4)
        obs = held_observation or None
        if obs:
            checks["observed_distance_mm"] = round(float(obs["centre_distance_m"]) * 1000.0, 1)
            checks["observed_limit_mm"] = round(
                (float(obs["held_radius_m"]) + float(cfg.revoke_centre_slack)) * 1000.0, 1)
            checks["hand_rise_mm"] = round(float(obs["hand_rise_m"]) * 1000.0, 1)
            checks["object_rise_mm"] = round(float(obs["object_rise_m"]) * 1000.0, 1)
        ev["hold_checks"] = checks
        failed = []
        if cfg.revoke_use_opening:
            if "opening_dropped" in checks and checks["opening_dropped"] >= float(cfg.revoke_drop):
                failed.append("opening_dropped")
            if "gap_to_command" in checks and checks["gap_to_command"] < float(cfg.blocked_gap):
                failed.append("not_blocked")
        if obs:
            if float(obs["centre_distance_m"]) > (float(obs["held_radius_m"])
                                                   + float(cfg.revoke_centre_slack)):
                failed.append("observed_elsewhere")
            rise = float(obs["hand_rise_m"])
            if (rise >= float(cfg.revoke_lift_min)
                    and float(obs["object_rise_m"]) < float(cfg.revoke_follow_ratio) * rise):
                failed.append("not_following")
        return failed

    def _slip_check(self, signal: GraspSignal, opening, ev: dict) -> bool:
        """T43 Z3 — 닫힌 채 손에서 빠졌나 (`LatchConfig.slip_detach` 일 때만 부른다).

        `ev["slip"]` 에 근거를 남기고, `slip_frames` 요청 연속 참이면 True. 모르는 값(측정 개도 ·
        attach 개도 · 실행 명령 중 하나라도 없음)이면 그 요청은 거짓이다 (fail-safe: 쥔 채로 둔다 —
        Z3 전의 동작).
        """
        cfg = self.config
        if signal.last_command is not None:
            self._hold_command = float(signal.last_command)
        a, c = self._attach_opening, self._hold_command
        known = opening is not None and a is not None and c is not None
        drop = None if not known else round(a - float(opening), 4)
        gap = None if not known else round(float(opening) - c, 4)
        now = bool(known and drop >= float(cfg.slip_drop) and gap < float(cfg.slip_gap))
        self._slip_streak = self._slip_streak + 1 if now else 0
        ev["slip"] = {"attach_opening": a, "hold_command": c, "drop": drop, "gap": gap,
                      "now": now, "streak": int(self._slip_streak), "frames": int(cfg.slip_frames)}
        return self._slip_streak >= int(cfg.slip_frames)

    def _event(self, attach: bool, detach: bool, note: str, evidence: dict, *,
               revoked: bool = False, slipped: bool = False) -> LatchEvent:
        evidence = dict(evidence)
        evidence["state"] = self.phase.value
        return LatchEvent(
            phase=self.phase,
            manipulated=self._manipulated.locked,
            destination=self.destination,
            attach=attach,
            detach=detach,
            note=note,
            evidence=evidence,
            revoked=bool(revoked),
            slipped=bool(slipped),
        )

    def _reset_grasp(self, *, keep_opening: bool = False) -> None:
        #: 닫힘 시도가 시작되기 직전의 측정 개도.
        self._opening_before: Optional[float] = None
        #: 닫힘 시도 뒤 **실행된** 닫힘 명령 스텝의 누적. 기록용 (T30 F4 뒤 `settled` 는 이것을 보지 않는다).
        self._settled_steps = 0
        #: 마지막으로 실행된 명령 (HOLD 청크는 갱신하지 않는다).
        self._last_command: Optional[float] = None
        #: T34 J2: attach 순간의 측정 개도와 attach 뒤 실행된 청크 수 (회수 창).
        self._attach_opening: Optional[float] = None
        self._held_exec_chunks = 0
        #: T43 Z3: 쥔 동안 마지막으로 **실행된** gripper 명령 (HOLD 청크는 들고 간다) · slip 연속 요청 수.
        self._hold_command: Optional[float] = None
        self._slip_streak = 0
        if not keep_opening:
            #: 직전 요청의 측정 개도.
            self._last_opening: Optional[float] = None
            self._last_grasp: Optional[LatchEvent] = None

    def reset(self) -> None:
        """다음 과제로 넘어간다. 잠금 둘과 상태를 모두 푼다."""
        self._manipulated.release()
        self._destination.release()
        self._external_destination = None
        self.phase = GraspPhase.SEARCHING
        self._open_streak = 0
        self._reset_grasp()


__all__ = ["CentroidIdentity", "GraspLatch", "GraspPhase", "GraspSignal", "LatchConfig", "LatchEvent",
           "grasp_signal_from_feedback"]
