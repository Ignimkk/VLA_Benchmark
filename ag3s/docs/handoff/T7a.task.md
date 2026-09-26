# T7a — 활성 제약 전체를 link 별로 (무엇이 손을 미는가)

> writer: lead (A0) · 2026-09-26 · 주 담당 **ag3s-verifier** · 산출 **하나**: `handoff/T7a.verify.json`
> 사용자 지시: *"어느 관절이 충돌 제약에 걸리는지 기록하세요."*

## 왜 — `max_violation_pair` 로는 모자란다

기록의 `max_violation_pair` 는 **최악 한 행**만 싣는다. 그런데 `T6d` 와 lead 측정이 보여 준 것은
**위반이 0 인데도 TO 가 손을 밀어낸다**는 것이다 — 제약이 *위반*이 아니라 *활성* 상태로 일한다.
**활성 행 전체의 분포**가 있어야 무엇이 미는지 안다.

배제된 것들 (다시 시험하지 마라, `handoff/T7.task.md` 참고): phase · 특정 link 제외 ·
실행 창 · `esdf_margin` 10 mm · `capsule_radius_scale` 0.8 — 전부 시험했고 전부 실패했다.

## 자산 — 이미 있다. **새로 촬영하지 마라**

| | |
|---|---|
| closed-loop / `approach` | `outputs/live_test/local_t6_from_pc/` (실패, apple 0.0 mm) |
| **shadow** / `approach` | `outputs/live_test/local_shadow_from_pc/` (**성공, apple 245.2 mm**) |
| closed-loop / `grasp` | `outputs/live_test/local_grasp_from_pc/` (실패) |
| closed-loop, 팔뚝 제외 | `outputs/live_test/local_noforearm_from_pc/` (실패) |

전부 `qpos` · `object_poses` · `actions` · `max_violation_pair` 를 담고 있다.

## 재는 것 — chunk 마다 **활성 제약 행 전체**

기록된 `qpos` 로 constraint set 을 **오프라인으로 다시 만들어** (`T6d` 에서 쓴 방법 그대로),
각 행에 대해:

| 무엇 | 왜 |
|---|---|
| `link` (그리고 그 link 를 움직이는 **관절**) | 사용자가 요구한 것이 이것이다 |
| 상대 — `obstacle` label / `candidate_id` / `esdf` | 무엇에 대해 걸렸나 |
| `clearance` 와 **`required_margin`** | 활성인지(clearance ≈ required) 위반인지(clearance < required) |
| `slack` 이 0 이 아닌 행인가 | QP 가 실제로 대가를 치른 행 |

**그리고 이것이 핵심이다 — `clearance − required_margin` 이 작은 순으로 상위 N 개를 chunk 마다.**
위반이 0 이어도 이 값이 0 에 가까운 행이 **손을 미는 행**이다.

**성공한 shadow 실행과 실패한 closed-loop 실행을 같은 chunk 축에 나란히 놓아라.** 둘이 갈라지는
chunk 에서 **어느 link 의 어느 행이 다른가**가 답이다.

## 같이 낼 것 — 관절 축으로

link → 관절 사슬을 되짚어, **어느 관절이 몇 번 활성 제약에 걸렸는지** 표로. 사용자가 요구한
형태다. `left_arm_0..6` · `ee_finger_l1/l2` 각각.

## 규칙

- 산출은 `handoff/T7a.verify.json` **하나**. 앞선 verify.json 을 덮지 마라.
- **figure 3 종**(규칙 A) 을 `docs/figures/t7a/`. **그래프 하나는 chunk 축 × link 별
  `clearance − required_margin`** 이어야 한다 — 무엇이 언제 조이는지가 한눈에.
- raw 와 script 는 `outputs/verify/T7a/`. `T6d` 의 script 를 넓혀 써도 된다.
- **코드를 고치지 않는다.** **포트 8000 서버(PID 1435228)를 죽이지 마라.** `pkill -f` 금지.
- 해석·판정을 쓰지 않는다. 숫자와 경로만.
- `T5f` 에서 잡은 `mj_geomDistance` 의 `distmax` 함정을 피해라 (0.1 m 이하).
- 규칙 C/I — technical term 은 영어로.

## 이 task 가 답하는 질문

**위반이 0 인데도 손을 미는 행은 어느 link 의 무엇에 대한 것인가.**
