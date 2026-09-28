# T9b (A2 · 검증) — TO 는 어느 link 을 사과에 붙이려 하는가

기록: `/mnt/dev/work/outputs/live_test/20260926_t9/execute_ep1807/frames.jsonl`
**75 chunk 전부에 `actions_reference`(정책 원본)와 `actions`(TO 출력)가 나란히 있다.**
끊기지 않은 완주 기록이다. MuJoCo rollout 없이 T9a 의 `m0_state_tensor.py` 를 재사용하라.

서버 설정은 T9a 와 같다 (구 266 · `link_*_arm_5` 19.53 mm · 손가락 4.72 mm · 손바닥 8.92 mm
· `target_field_policy=exclude_authorized` · `esdf_margin 0.01`).

## 사용자 가설 (이것을 판정한다)

> *"action chunk 는 그리퍼를 사과에 닿게 한다. 그런데 TO 는 그리퍼가 아닌 **다른 부분**을
> 사과에 접근시키려는 것 같다."*

lead 가 코드로 확인한 것 (다시 하지 마라): chunk 열 → 관절 매핑은 **이름으로** 찾고
(`types.py:193-198`), 정·역방향이 **같은 `action_columns` 배열**을 쓴다 (`:120`, `:142`).
좌우 뒤바뀜이나 한 칸 밀림은 구조적으로 불가능하다.

lead 가 기록에서 잰 것 (재확인만 하고 되풀이하지 마라): 실행 창 8 step 에서 TO 의 편차는
`L0~L6` 최대 4.3~5.8° · chunk 별 최대의 중앙값 1.06°, **gripper 열은 정확히 0.000**,
오른팔은 0.04~0.10°.

## 재는 것

**N1 — 두 청크를 FK 로 풀어 link 별로 사과까지 거리를 낸다. 이것이 판정이다.**
chunk 마다 실행 창 8 step 에 대해, **정책 원본**과 **TO 출력** 각각을 FK 로 풀고
**link 별 사과 표면까지 최소 거리**를 내라. 그리고 각 청크에서 **가장 가까운 link 이 누구인가**를
두 경우로 나란히. 사용자 가설이 참이면 원본은 `ee_finger_*`, TO 출력은 **다른 link** 이 나온다.

**N2 — 손끝이 가려던 곳과 실제로 간 곳.** `ee_finger_l1`/`l2` 끝점의 위치를 두 청크로 풀어
**mm 단위 차이**를 시계열로. 어느 방향으로 갈라지는가 (x·y·z 성분). 그리고 그 차이가
`정책이 의도한 손끝 위치 − TO 가 만든 손끝 위치` 로 얼마나 누적되는가.

**N3 — 정책 원본을 끝까지 따라가면 사과에 닿는가.** 손이 가장 가까웠던 청크를 골라
**원본 50 step 전체**를 FK 로 풀어 손끝–사과 최소 거리를 내라. 이것이 *"정책은 잡을 줄 안다"*
를 이 실행의 기록으로 닫는다 (지금까지는 별도 shadow 실행이 근거였다).

**N4 — 어느 관절이 손끝을 가장 많이 옮겼나.** 도(degree)가 아니라 **손끝 변위 기여**로.
Jacobian 으로 관절별 기여를 갈라라. 1° 라도 어깨(`L0`)면 손끝에서 크고 손목(`L6`)이면 작다.
lead 의 도 단위 표는 이 구분을 못 한다.

**N5 — gripper 열이 정말 손대지 않는가.** 75 chunk × 50 step 전 원소에서
`actions[:, 7]` 과 `actions_reference[:, 7]` (그리고 `[:, 15]`)이 **bit 단위로 같은지**.
lead 는 실행 창 8 step 만 봤다.

## 내는 것

- `benchmark/ag3s/docs/handoff/T9b.verify.json` — 수치는 `numbers` 안에만
- figure: `benchmark/ag3s/docs/figures/t9b/` — **실제 씬에 두 궤적을 겹쳐 그려라**
  (정책 원본 손끝 경로 대 TO 손끝 경로, 사과와 함께). 그래프는 N2 의 mm 차이 시계열과
  N4 의 관절별 기여.
- raw·script 는 `outputs/verify/T9b/`

## 규칙

- **해석을 쓰지 마라.** 수치와 figure 만.
- 못 잰 것은 `not_measured` 에 이유와 함께.
- 포트 8123 서버는 건드리지 마라. 코드는 읽기만.
