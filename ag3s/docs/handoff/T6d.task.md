# T6d — 무엇이 손을 사과에 못 가게 하나 (세 단계 중 어디인가)

> writer: lead (A0) · 2026-09-26 · 주 담당 **A2(verifier)** · 산출 **하나**: `handoff/T6d.verify.json`

## 확정된 사실 — 여기서 출발한다

로컬 PC(GPU 렌더링)에서 돌린 **세 실행**이 있다. 서버·AG3S·field 설정은 같고 **변수가 하나씩만
다르다.** lead 가 참값(`object_poses`)으로 이미 센 값이다:

| 실행 | 경로 | HOLD | **apple 최대 들림** | apple↔crate 수평 |
|---|---|---|---|---|
| closed-loop / `approach` | `outputs/live_test/local_t6_from_pc/` | **0 / 75** | **0.0 mm** | 339.2 → 339.5 |
| **shadow** / `approach` | `outputs/live_test/local_shadow_from_pc/` | 6 / 75 | **245.2 mm** | 339.2 → **21.1** → 73.5 |
| closed-loop / `grasp` | `outputs/live_test/local_grasp_from_pc/` | **1 / 75** | **0.0 mm** | 339.2 → 339.5 |

**읽는 법 셋 (lead 가 확정했다):**

1. **HOLD 는 원인이 아니다.** closed-loop/approach 는 한 번도 안 멈췄는데 사과를 1 mm 도
   못 건드렸다. 멈춰서 못 잡은 것이 아니라 **멈추지 않고도 못 잡았다.**
2. **phase 는 원인이 아니다.** `approach → grasp` 로 바꿔도 결과가 완전히 같다 (둘 다 0.0 mm).
3. **로봇이 refined chunk 를 날리면 실패하고, 원본 chunk 를 날리면 성공한다.**
   → **TO 의 수정이 손을 사과에서 떼어 놓는다.** 그리고 그 밀어냄은 `violated` 로 안 잡힌다 —
   TO 는 위반이 안 나게 **성공적으로** 궤적을 바꿨고(위반 0.00 mm), 그 대가가 과제 실패다.

**이상한 점**: 세 실행 모두 최악 위반 link 가 **`link_left_arm_5` / `link_right_arm_5`(팔뚝)**
이고 손가락(`ee_finger_*`)이 아니다. 유일한 HOLD(grasp 실행 seq 5, t=32, 25.18 mm)의 위반 좌표
`max_violation_pair.point_m` 은 **가장 가까운 참값 물체인 apple 에서도 153 mm** 떨어져 있다.
**사용자가 그 자리를 직접 확인했고 아무것도 없다.**

## 재는 것 1 — **AG3S → cuRobo 단계**

`manipulated_link_margin`(target 에 대해 권한 있는 link 만 margin 을 완화하는 벡터)이
**실제로 만들어지고 값이 들어 있는가.**

- 세 실행의 기록·서버 로그에서 그 흔적을 찾아라. 없으면 `not_measured` 에 적고
  **무엇을 더 실어야 알 수 있는지** 한 줄로 (A1 에게 넘길 근거).
- **`is_authorized` 는 정확한 문자열 일치다** (`clearance.py:280`:
  `[name in authorized for name in links]`). `links` 는 `builder.sphere_link_names` 다.
  **그 두 집합을 실제로 찍어서 나란히 놓아라** — 겹치는 이름이 0 개면 완화가 한 link 도 안 걸린다.
  T1 에서 `DEFAULT_CONTACT_LINKS`(`config.py:367-370`)가 `ee_left`/`ee_right` 를 적는데
  constraint model 은 `ee_finger_` 만 본다는 어긋남을 이미 봤다.
- 오프라인으로 constraint set 을 한 frame 만들어 직접 찍어도 된다. **그것이 가장 빠르다.**

## 재는 것 2 — **cuRobo 단계**

**팔뚝이 피하는 자리에 참값으로 무엇이 있는가.**

- `max_violation_pair.point_m` 을 세 실행에서 모아, 그 좌표의 **참값 최근접 표면까지 거리**를
  재라 (`T5f` 의 `mj_geomDistance` `distmax` 함정을 피해라 — 0.1 m 이하를 쓴다).
- **아무것도 없는데 ESDF 가 점유로 답하면 field 의 문제다.** 유력한 후보는 **미관측 voxel 정책**
  (`esdf.unknown_policy`) 이다 — 카메라가 못 본 자리를 fail-closed 로 점유 취급하는 것.
  그 설정값과, 그 좌표가 카메라 시야 안이었는지를 확인해라.
- 사과 자리의 ESDF 부호와 값이 참값과 맞는지도 함께 (사과를 부풀리고 있지 않은가).

## 재는 것 3 — **TO 단계**

shadow 기록에는 `actions`(refined)와 `actions_reference`(원본)가 **둘 다** 있다. **네 라운드
연속 미측정이던 항목을 드디어 잴 수 있다.**

- chunk 별 **관절별 수정량** `refined − reference`. 어느 관절이 가장 많이 바뀌는가.
- **MuJoCo 참값으로 두 궤적의 최소 clearance 를 재라** — `refined` 가 `reference` 보다
  나빠지지 않는가 (T5 의 핵심 합격 조건). 나빠진 chunk 가 있으면 전부 나열해라.
- closed-loop 실행의 `actions`(refined)와 shadow 실행의 `actions_reference`(원본)를 같은 chunk
  축에 놓고, **손끝이 사과로 향하는 정도가 언제부터 갈라지는지**.

## 규칙

- 산출은 `handoff/T6d.verify.json` **하나**. 앞선 verify.json 을 덮지 마라.
- **figure 3 종**(규칙 A)을 `docs/figures/t6d/` 에. **실제 씬 하나는 세 실행의 같은 chunk 에서
  손끝·사과·팔뚝·위반 좌표를 한 그림에** — 사용자가 눈으로 본 것과 맞춰 볼 수 있게.
- raw 와 script 는 `outputs/verify/T6d/`. **새로 촬영하지 마라.**
- **코드를 고치지 않는다.** **포트 8000 서버(PID 413170)를 죽이지 마라** — A1 이 곧 고칠 것이고
  재기동은 lead 가 한다. `pkill -f` 금지.
- 해석·판정을 쓰지 않는다. 박힌 수치(0.0 / 245.2 mm · 339.2 → 21.1 → 73.5 mm · 25.18 mm ·
  153 mm · 0/6/1 hold) 밖의 값을 지어내지 마라.
- 규칙 I — technical term 은 영어로.

## 이 task 가 답하는 질문

**AG3S 가 넘기는 단계인가, cuRobo 인가, TO 인가.**
