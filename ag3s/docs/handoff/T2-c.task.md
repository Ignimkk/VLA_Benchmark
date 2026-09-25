# T2-c — `runner_up_score` 가 들어간 코드로 회귀 기준선을 다시 잰다

> writer: lead (A0) · 2026-09-25 · 주 담당 **A2(verifier)** · 산출 **하나**: `handoff/T2-c.verify.json`

## 왜 — 커밋을 막고 있는 것이 이것 하나다

`T2-b` 의 기준선 run 은 **14:44:37 종료**로 A1 변경(**14:48:13**) **전** 코드였다. 변경 후
코드로는 기준선을 **아직 한 번도 안 쟀다.** 사용자 판정(2026-09-25): **재서 보고 커밋한다.**

## 무엇을 재나 — `regression-baseline` skill 그대로

**skill 을 먼저 읽어라**: `.claude/skills/regression-baseline/SKILL.md`. 명령·판정 요령
(개수만 인용하는 이유, `--cameras all` 이 필요한 이유, 이 기준선이 재지 **않는** 것)이 거기 있다.

**비교 대상 셋** — 이 셋만이 기준이다:

| 무엇 | 기준값 |
|---|---|
| 위반으로 시작 | **14/15** |
| `has_target` | **9/15** |
| frame0 `clearance_before` | **+0.15718632962849477 mm** (`T2-b` 가 소수점까지 재현했다) |

**`feasible`/`violated`/`해소` 는 기준이 아니다** — `sqp.time_budget_ms` 가 벽시계 마감이라 같은
명령에서 흔들린다. 개수를 적기는 하되 **판정 근거로 쓰지 마라.**

## 변경이 무엇인지 — 그리고 무엇을 예상하면 안 되는지

`benchmark/ag3s/stages/target_grounding.py:417` 에 `runner_up_score` 키 하나가 생겼고,
`safe_policy.py:341` → `grasp_latch.py:142` 의 `confident = score >= score_ratio(1.3) *
max(runner_up, 1e-9)` 가 **처음으로 실제 값을 받는다.**

`T2-b` 가 이미 낸 사실 (이것만 물려받는다):
- 변경 후 코드가 내는 `runner_up_score` 는 75 프레임 **최대 0.2757589427371451**
- `confident` 판정이 갈리는 프레임은 **[30, 35, 67, 68, 71] 다섯 개**

**A1 의 impl.md 에 있는 "내가 기대하는 결과" 절을 측정이 끝나기 전에 읽지 마라.** 기대가
보이면 판정이 뒤집힌다 — 이 검토에서 두 번 그랬다 (C2 = 옛 측정법이 실제로는 대역폭을 재고
있었다 · D2 = fine layer 가 로봇을 안 덮은 한 프레임만 보고 값어치가 없다고 판정했다).

## 셋이 움직였다면

**코드를 고치지 마라.** `verify.json` 에 **무엇이 얼마로 바뀌었는지**와 `runner_up_score` 가
그 프레임에서 얼마였는지를 적고 넘겨라. 원인 찾기는 그다음이다.

## 규칙

- 산출은 `handoff/T2-c.verify.json` **하나**. `T2-b.verify.json` 을 덮지 마라 (재측정은 원 측정의
  파일을 덮어쓰지 않는다).
- raw 는 `outputs/verify/T2c/`. `outputs/verify/T2/regression_check/` 를 덮지 마라.
- `code_state` 에 `md5sum /mnt/dev/work/benchmark/ag3s/stages/target_grounding.py` 를 넣어라 —
  **`ee33a055a941811da6df094fbe985a94` 여야 한다** (변경 후). `caae04a1...` 이면 tree 가
  되돌아갔다는 뜻이니 멈추고 보고해라.
- figure 는 **이번엔 안 만든다** — 기준선 셋의 일치/불일치가 답의 전부다. 불일치가 나오면
  그때 lead 가 figure 를 요청한다.
- 해석·판정을 쓰지 않는다. 코드를 고치지 않는다. 커밋하지 않는다.
- `pkill -f` 금지. 긴 실행은 `python -u`.
