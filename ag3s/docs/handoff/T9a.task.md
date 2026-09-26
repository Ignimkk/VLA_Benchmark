# T9a (A2 · 검증) — 충돌이 안 막는데 왜 못 잡았나

기록: `/mnt/dev/work/outputs/live_test/20260926_t8/execute_ep1807/frames.jsonl`
(observation 39 · planning 39 · control 312, t=304 에서 사용자가 중단). **MuJoCo rollout 없이**
T7a·T8a 의 오프라인 재구성과 `m0_clearance_tensor.py` 를 재사용하라. 코드를 고치지 않는다.

## 이번 실행이 이전과 다른 점

서버 설정: `--target-field-policy exclude-authorized` · `--capsule-extent-link link_*_arm_5=-0.10`
· `--sphere-spacing 0.4` · `--capsule-radius-scale 0.2` · `--max-spheres-per-capsule 32`
· `--esdf-margin 0.01`. 제약 구 **266 개**, `link_*_arm_5` **19.53 mm**, 손가락 **4.72 mm**,
손바닥 `ee_left`/`ee_right` **8.92 mm** 신규 포함.

결과: **39 chunk 전부 `safe`, `unsafe 0`.** 팔이 사과 앞 90 mm 에서 얼지 않고 바구니까지 갔다.
그런데 **사과를 못 집었고**, 헛잡은 채 latch 가 걸려 바구니로 갔고, **바구니에서 하강을 못 했다.**

## 재는 것 (다섯)

**M1 — 손끝이 사과에 얼마나 가까워졌나.** `qpos` + `object_poses` 로 chunk 마다
`ee_finger_l1`/`l2` 구 표면에서 사과 표면까지 최소 거리. 시계열과 전체 최소값.
**비교 기준 둘을 같은 표에 놓아라**: 이전 closed-loop 실패 **+89.86 mm**, 성공한 shadow **−17.96 mm**.
이번 값이 그 사이 어디인가가 "얼마나 가까워졌나" 다.

**M2 — gripper 가 닫힌 순간.** chunk 의 gripper 열(`wire.gripper_columns()` = 7, 15)이
0.85 아래로 처음 내려가는 `t_step`, 그 순간의 손–사과 거리, 그리고 그때 손이 사과의
**어느 쪽**에 있었나 (사과 중심 기준 offset 벡터, mm). 빗나갔다면 **어느 방향으로** 빗나갔는지가
다음 판정을 가른다.

**M3 — 사과가 움직였나 (참값).** `object_poses` 의 apple 위치 시계열. 들린 높이 최대값(mm).
0 이면 한 번도 안 잡힌 것이다. 이것이 "못 잡았다" 의 객관적 근거다.

**M4 — 바구니에서 하강이 멈춘 자리.** 손이 crate 위에 도달한 뒤의 chunk 들에서,
chunk 별 `max_violation_pair` 와 손 높이(z) 시계열. **무엇이 하강을 막았는지**를
(link, obstacle) 이름과 mm 로 내라. `verdict`·`hold_reason`·`ag3s_reason_codes` 도 함께.
쥔 것이 없으므로 `destination_margin`(쥔 물체 점에만 걸리는 얇은 margin)이 적용되지 않는다 —
그 가설이 맞는지 수치로 답하라.

**M5 — `max_violation_pair` 분포.** 39 chunk 전부에 대해 어느 (link, obstacle) 이 최악이었나.
`unsafe 0` 이므로 이것은 위반이 아니라 **가장 빡빡한 행**이다. 손바닥 `ee_left` 가 새로
들어왔으니 그것이 상위에 나타나는지 특히 보라.

## 내는 것

- `benchmark/ag3s/docs/handoff/T9a.verify.json` — 모든 수치는 `numbers` 안에만.
- figure: 실제 씬 + 그래프 + 표, `benchmark/ag3s/docs/figures/t9a/`.
  씬은 **M2 의 gripper 가 닫힌 순간**과 **M4 의 하강이 멈춘 순간** 두 자리를 그려라.
  그래프는 손–사과 거리 · 사과 높이 · 손 높이를 한 시간축에.
- raw·script 는 `outputs/verify/T9a/`.

## 규칙

- **해석을 쓰지 마라.** 수치와 figure 만. 판정은 lead 와 사용자가 한다.
- 못 잰 것은 `not_measured` 에 이유와 함께. 지어내지 않는다.
- **`actions_reference`(정책 원본 chunk)는 이 기록에 없다** — A1 이 지금 싣는 중이다.
  그러니 "TO 가 chunk 를 얼마나 바꿨나" 는 이번에 잴 수 없다. `not_measured` 에 그렇게 적어라.
- 포트 8123 서버와 A1 의 작업 트리는 건드리지 마라. 코드는 읽기만.
