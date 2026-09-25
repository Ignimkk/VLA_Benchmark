# T2-record — 확정된 T2 결과와 사용자 판정 넷을 로그에 남긴다

> writer: lead (A0) · 2026-09-25 · 주 담당 **A3(scribe)** · 고치는 파일 **둘**: `docs/AG3S_T0T6_LOG.md` · `docs/AG3S_REVIEW_PLAN.md`

## 왜 지금 쓰나

`AG3S_T0T6_LOG.md` 의 T2 절이 아직 *"T2 측정은 진행 중이다 … 수치는 아직 쓰지 마라"* 로 끝난다.
**측정은 끝났다** — `handoff/T2.verify.json` 이 nine events 를 다 냈다. **이미 확정된 것을 지금 쓴다.**

## 출처는 하나다 — `handoff/T2.verify.json`

**`numbers` 에 없는 숫자는 한 자리도 쓰지 마라.** 없으면 "측정하지 않았다" 로 적는다.
`ep1800` 키(1 라운드, 운반 없음)와 `ep1807` 키(2 라운드, 운반 완결)를 **섞지 마라** — 어느
기록의 수치인지 문장마다 분명해야 한다.

## 쓸 것 1 — T2 절을 이어 쓴다 (`## T2 — 자산 확보 (ep1807, 2026-09-25)` 아래)

기존 절을 **지우지 말고 이어 쓴다.** 마지막 단락(*"T2 측정은 진행 중이다 … 쓰지 마라"*)만
측정 결과로 **바꾼다.** 담을 것:

1. **nine events 의 frame 번호 표** — `numbers.nine_events_cycle1_ep1807` 의 10 개 항목을
   frame · t_step · 판정 근거(`criterion`)로. **둘이 `None` 이다**:
   `destination_attention_locked` 와 `task_state_reset`.
2. **처음으로 놓기가 잡혔다** — `placement` 와 `detached` 가 같은 frame. detach 가 `placed_now`
   성공 경로로 울렸고 gripper open-streak fallback 이 아니라는 것(`criterion` 에 gripper 값이 있다).
   ep1800 에서 반쪽이던 뒤 절반이 닫혔다는 뜻.
3. **destination attention 이 한 번도 확정되지 않는다** — `criterion` 에 HELD 구간의 label 순서가
   그대로 있다. `confirm_frames = 3` 인데 연속 2 가 최대. **AG3S 의 destination 추적이 기여하지
   않은 채로 정책이 넣었다**는 것을 쓴다.
4. **detach 후 state 가 초기화되지 않는다** — `numbers.phase_stuck_after_detach_ep1807` 세 줄.
   acceptance 조건 *"detach 후 이전 상태가 제거된다"* 는 **불합격**. 단
   `numbers.cycle2_second_grasp_ep1807.present == False` 이므로 **그것이 다음 과제를 막는지는
   이 기록으로 증명할 수 없다** — 이 유보를 반드시 같이 쓴다.
5. **결함 3 재현 · 결함 2 미재현** — `numbers.runner_up_score_metrics_key_ep1807` 와
   `numbers.identity_ambiguity_ep1807`(`mismatch: False`), 그리고
   `numbers.identity_vs_runner_up_relation_ep1807` 이 둘의 관계를 적고 있다.
6. **object drift 표** — `numbers.object_displacement_full_75_frames_ep1807`: apple 408.5 mm,
   banana 0.1 · orange 0.2 · pear 0.6 · crate 0.0 mm. **대상만 움직였다.**
7. **아직 측정하지 않은 둘** — `not_measured` 를 그대로 옮긴다: (a) held object 가 robot collision
   geometry 에 들어가고 obstacle field 에서 빠지는가 (b) accumulated TSDF 의 residual 지속 frame 수.
   **`handoff/T2-b.task.md` 로 A2 에게 넘겼다**고 적는다.
8. **규칙 A · figure 링크** — `docs/figures/t2/` 의 여섯 장을 링크한다. `ep1807` 것 셋이
   `t2-scene-hand-trajectory-ep1807.png`(실제 씬) · `t2-timeline-ep1807.png`(그래프) ·
   `t2-events-table-ep1807.png`(표) 이고, 이름 없는 셋은 ep1800 이다. **어느 기록의 그림인지
   캡션에 쓴다.**

## 쓸 것 2 — 사용자 판정 넷 (2026-09-25)

새 소절 `### 사용자 판정 (2026-09-25) — T2 게이트와 T1 잔여 둘` 로 묶어 적는다:

| 물음 | 판정 |
|---|---|
| T2 를 어떻게 닫나 | **미측정 둘을 먼저 채운 뒤 판정한다** — `T2-b` 로 나갔다 |
| `runner_up_score` 키 부재 | **지금 고친다** — `T2-fix` 로 A1 에게 나갔다 |
| `gap_filling_capsules` 의 조용한 삼킴 | **소리를 내게 만든다.** 코드는 이미 그렇게 되어 있다 (`mujoco_source.py` 의 `_warn_no_capsules`) — T1 절의 *"이 삼킴은 아직 그대로다 — 판정 대기"* 를 **판정 결과로 갱신한다** |
| 손바닥(`ee_left`/`ee_right`)을 constraint model 에 | **넣지 않는다 — 기존 동작 유지.** `DEFAULT_CONTACT_LINKS`(`config.py:367-370`)와 constraint model(`rby1_transport.py:454`)이 어긋나는 것은 **알려진 상태로 남긴다** → **"되돌아올 지점"** 절에 전환 신호와 함께 |

T1 절의 **"판정 대기 둘"** 문장을 판정 결과로 갱신한다. **판정 대기라는 표현이 남아 있으면 안 된다.**

## 쓸 것 3 — **진행 현황** 표의 T2 행

지금 *"자산 확보(ep1807) — 측정 진행 중"* 이다. **nine events 가 나왔고 둘이 `None` 이며,
acceptance 두 항목이 `T2-b` 대기**라는 상태로 갱신한다. T1 행의 "판정 대기 둘" 도 갱신한다.

## 쓸 것 4 — 규칙 C · 용어 절

`## 용어` 에 없는 것만 더한다. 최소 이 넷은 이번 절에서 처음 쓰인다:

- **`runner_up_score`** — 1 등 후보의 점수와 2 등의 점수 차로 grounding 의 확신을 재는 값
- **destination attention lock** — 쥔 뒤 attention 이 목적지로 옮겨갔다고 `confirm_frames` 연속
  같은 label 로 확정하는 절차
- **`placed_ground_truth`** — MuJoCo 참값으로 판정한 "놓였다" (crate 안쪽 + rim 아래)
- **TSDF residual (잔상)** — 물체가 떠난 뒤에도 accumulated TSDF 에 남는 occupancy

## 쓸 것 5 — 규칙 F · `AG3S_REVIEW_PLAN.md`

수행한 것을 계획에도 반영한다. **이번 라운드는 이 파일의 writer 가 너 하나다** (lead 가 안 쓴다).

## 규칙

- **`verify.json` 의 `numbers` 에 없는 숫자를 쓰지 않는다.** 이것이 제일 중요하다.
- **`T2-b` 와 `T2-fix` 의 결과를 쓰지 마라** — 아직 돌고 있다. "나갔다" 까지만.
- 규칙 G — 발견 ID 를 그냥 쓰지 않는다. `E3(쥔 물체가 optimizer 에 도달하지 않는다)` 처럼 푼다.
- 규칙 I — technical term 은 영어로: `feasible` · `violated` · `clearance` · `latch` · `attach`/`detach` ·
  `obstacle field` · `collision geometry` · `TSDF`/`ESDF` · `confirm_frames`.
- 네 소유는 `docs/*.md` 다. **코드·figure·`handoff/*.verify.json` 을 고치지 않는다.**
- 큰 문서를 통째로 읽지 않는다 — 로그는 1836 줄이다. `sed -n`/`grep -n` 으로 필요한 줄만.
- 커밋하지 않는다. lead 가 한다.
