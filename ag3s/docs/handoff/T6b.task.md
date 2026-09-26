# T6b — 실행 모드(closed loop)가 언제 어디서 멈추나 — **지금 기록으로 잴 수 있는 것까지**

> writer: lead (A0) · 2026-09-26 · 주 담당 **A2(verifier)** · 산출 **하나**: `handoff/T6b.verify.json`

## 범위를 먼저 못 박는다 — 참값은 이 기록에 없다

이 기록에는 **`qpos` 도 물체 자세도 `actions` 도 없다.** 그래서 **이번에 못 재는 것**:

- T6 실행에서 사과가 **언제 바구니에 들어갔나** (물체 자세 없음)
- seq 38 이후 팔이 **정말 안 움직였나** (`qpos` 없음)
- 멈춘 자리의 **참값 clearance 와 가장 가까운 쌍** (`qpos` 없음)
- `violated` 가 **어느 제약**인가 (verdict 가 제약 신원을 안 싣는다)

**이 넷을 억지로 재려 하지 마라.** A1 이 기록기를 고치는 중이고(`T6a`), 그 뒤 재실행에서 잰다.
**`not_measured` 에 그대로 적어라.**

**offline replay 로 대신하려 하지 마라.** T6 는 hold 때문에 궤적이 shadow 와 **갈라졌다** —
원 기록을 재생해도 T6 가 실제로 지나간 자리가 아니다. 그 사실도 `not_measured` 에 적어라.

## 자산 — 세 실행을 나란히

| | 경로 | 조건 |
|---|---|---|
| shadow (threshold 0.25) | `outputs/live_test/20260925_t5shadow/shadow_ep1807/` | rank 1 이전 |
| shadow (rank 1 + confirm 3) | `outputs/live_test/20260925_t5e_shadow/shadow_ep1807/` | 판정만, 실행은 원본 |
| **execute (closed loop)** | `outputs/live_test/20260926_t6/execute_ep1807/` | **unsafe 면 멈춘다** |

클라이언트 로그는 `outputs/live_test/logs/` 의 `t5shadow_run.log` · `t5e_shadow_run.log` ·
`t6_execute_run.log`. 서버 로그는 `serve_safe_8000_shadow.log` · `_t5e.log` · `_t6.log`.

## 재는 것 1 — chunk 별 판정 (세 실행 나란히)

**lead 가 눈으로 센 것이다. 네 집계와 대조하고 다르면 네 수치가 맞다:**

- execute: safe **33** / hold **42**, completeness pass, timeout·stale·error 0
- hold 42 건의 분류: `(ag3s ok, violated)` **35** · `(degraded, violated)` **7**
- `max_violation_mm`: min **0.34** · p50 **21.82** · max **37.83**
- **hold 위치**: seq 3 · 22 · 26 · 28, 그리고 **seq 38~75 전부**
- **safe 33 건이 전부 seq 1~37 안에 있다** — 과제 구간은 37 중 33 통과(89 %)

chunk 별로 `safe` · `ag3s_status` · `trajopt_status` · `max_violation_m` ·
`ag3s_reason_codes` · `field.tiers` 개수를 **세 실행 나란히** 표로 내라.

## 재는 것 2 — **멈춤이 과제 구간인가 복귀 구간인가**

사용자 판정(2026-09-26): *"사과가 바구니에 들어간 순간 과제가 끝나고 준비자세로 돌아온다.
복귀 구간에는 최적화나 충돌회피가 필요 없다."*

**이 기록만으로 할 수 있는 것**: shadow(rank 1) 실행의 `first_closed_seq`/`first_reopen_seq`
(`T5d.verify.json` 의 `task_phase_rule` — 20 과 33)와 execute 의 hold 구간을 **같은 축에 놓고**
보여 준다. **execute 의 실제 phase 는 모른다** (참값 없음) — 그것을 분명히 적어라.

## 재는 것 3 — 위반이 왜 커졌나 (단서만)

shadow 에서 `max_violation` 이 0.14~5.69 mm 였는데 execute 는 p50 21.82 · max 37.83 mm 다.
**hold 가 시작된 뒤 위반이 커지는가** — seq 축으로 `max_violation_m` 을 그려라.
커진다면 *"멈추니까 빠져나오지 못한다"* 의 단서고, 처음부터 크면 다른 이야기다.
**해석은 쓰지 마라. 곡선만 낸다.**

## 재는 것 4 — 지연과 camera timing

세 실행의 `timing_ms` 단계별 P50·P95·max, 그리고 카메라 캡처 폭. shadow 에서
p50 이 93.61 → 74.94 ms 로 빨라진 것이 execute 에서는 어떤가.

## 규칙

- 산출은 `handoff/T6b.verify.json` **하나**. 앞선 verify.json 을 덮지 마라.
- **figure 3 종**(규칙 A)을 `docs/figures/t6b/` 에 새 이름으로. **그래프 하나는 세 실행의
  chunk 별 safe/hold 띠를 위아래로 나란히** — 어디서 갈라지는지가 한눈에 보이게.
- raw 와 script 는 `outputs/verify/T6b/`.
- **코드를 고치지 않는다.** **포트 8000 서버(PID 4004754)를 죽이지 마라.** `pkill -f` 금지.
- **새로 촬영하지 마라.** 재실행은 lead 가 `T6a` 뒤에 한다.
- 해석·판정을 쓰지 않는다. 박힌 수치 밖의 값을 지어내지 마라.
- 규칙 I — technical term 은 영어로.

## 이 task 가 답하는 질문

**closed loop 가 어디서부터 못 풀리나, 그리고 그것이 과제 구간인가 그 뒤인가.**
