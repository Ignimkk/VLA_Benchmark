# T5f — 1 등 + `confirm_frames 3` 이 실제로 무엇을 바꿨나

> writer: lead (A0) · 2026-09-26 · 주 담당 **A2(verifier)** · 산출 **하나**: `handoff/T5f.verify.json`

## 자산 — 두 실행을 **나란히** 둔다. 새로 촬영하지 마라

| | 경로 | 조건 |
|---|---|---|
| **이전** | `outputs/live_test/20260925_t5shadow/shadow_ep1807/` | `target_score_threshold = 0.25` |
| **이후** | `outputs/live_test/20260925_t5e_shadow/shadow_ep1807/` | **1 등 + `target_confirm_frames = 3`** (`T5e`) |
| 클라이언트 로그 | `outputs/live_test/logs/t5shadow_run.log` · `t5e_shadow_run.log` | |
| 서버 로그 | `outputs/live_test/logs/serve_safe_8000_shadow.log` · `serve_safe_8000_t5e.log` | |

**둘 다 같은 ep1807 · 600 step / 75 chunk · 같은 서버 설정**(cuRobo coarse 20 mm + fine 5 mm,
TSDF 5 mm, self-filter 218 sphere, constraint 120 sphere)이다. **바뀐 것은 target 선택 하나다.**

lead 가 눈으로 센 값이다 — **네 집계와 대조하고, 다르면 네 수치가 맞다**:

| | 이전 | 이후 |
|---|---|---|
| safe / unsafe | 14 / 61 | **68 / 7** |
| `no_target` | 54 | **0** |
| `degraded` | 6 | **0** |
| TO `violated` | 1 | **7** |
| fine layer 가 붙은 chunk | 16 / 75 | **75 / 75** |

## 재는 것 1 — **1 등이 옳은 물체였나** (이번 STEP 의 머리)

`T5e` 는 score threshold 를 없앴다. **score 가 "말도 안 되는 덩어리를 거부하는" 2 차 방어선
역할도 하고 있었고 그것이 사라졌다** (A1 이 `T5e.impl.md` 에 "대가" 로 적었다 — `exclude_mask`
없이 부르면 테이블로 번진 덩어리가 이제 target 이 된다).

**fine layer 가 75/75 붙었다는 것은 target 이 늘 있었다는 뜻이지 그 target 이 옳았다는 뜻이
아니다.** 그러니 chunk 마다:

- target cluster 를 **MuJoCo 참값 이름**으로 붙여라 (apple · banana · orange · pear · crate ·
  table · robot · unknown). `T5d` 에서 쓴 fusion provenance 방식을 그대로 쓴다.
- **`table` / `unknown` / 번진 덩어리가 target 이 된 chunk 를 전부 나열해라.** 개수만 적지 마라.
- target cluster 의 **크기(점 수)와 퍼짐(rms radius)** 을 chunk 별로. 번진 덩어리는 여기서 튄다.
- `T5d` 의 offline 계산과 대조: 예상 흐름은 **`apple 1-17` → `crate 18-51`** 이었다.
  **실제 live 흐름이 그것과 몇 chunk 에서 다른가.**

## 재는 것 2 — `violated` 7 건이 **진짜 관통인가**

이전 실행은 target 이 없어 *"all geometry held at full clearance"* 였다 — **위반할 제약 자체가
없었다.** 이제 candidate 가 만들어지니 SQP 가 비로소 진짜 위반을 보고할 수 있다.
**늘어난 것이 나빠진 것인지 정직해진 것인지 가려야 한다.**

- 7 chunk 각각의 `max_violation_m` · 어느 제약(어느 link 대 어느 물체)인가.
- **MuJoCo 참값으로 그 chunk 의 실제 최소 clearance 를 재라.** 참값이 음수면 진짜 관통이고,
  양수인데 `violated` 면 필드가 비관적인 것이다.
- 그 7 chunk 가 과제의 어느 국면인가 (`T5d` 의 phase 규칙: `first_closed_seq 20` ·
  `first_reopen_seq 33`).

## 재는 것 3 — 회귀 기준선

`regression-baseline` skill 을 읽고 돌려라. 기준값은 **위반으로 시작 14/15 · `has_target` 9/15 ·
frame0 `clearance_before` +0.15718632962849477 mm** 다.

**`has_target 9/15` 는 이 변경으로 바뀐다. 그것은 회귀가 아니라 의도다** — threshold 가 만든
값이었다. 나머지 둘이 움직였는지가 판정 대상이다. **움직였으면 코드를 고치지 말고 수치만 적어라.**

## 재는 것 4 — camera timing 이 정말 좋아졌나, 우연인가

이전 6 건 → 이후 0 건인데 **우리가 건드린 곳이 아니다.** 두 실행의 **세 카메라 캡처 폭**
(`T5c` 에서 p50 93.61 · p95 100.68 · max 107.22 ms, 한도 100 ms)을 나란히 내라.
**한도 근처에서 떨리는 값이 우연히 안 넘은 것인지**가 답이다.

## 측정할 수 없는 것 — 확인했고 적어 둔다

**`refined` 대 `reference` clearance 비교는 이 기록으로도 못 한다.** `frames.jsonl` 의
`executed_chunk` 는 `"reference"` 라는 **이름표**일 뿐이고 `actions` 배열은 여전히 없다
(`chunk_shape` 만 있다). qpos·물체 자세도 없다. **`not_measured` 에 그대로 적어라** —
재려면 기록기가 `actions`·`actions_reference`·`qpos` 를 남겨야 하고 그건 A1 의 일이다.

## 규칙

- 산출은 `handoff/T5f.verify.json` **하나**. `T5c`·`T5d` 의 것을 덮지 마라.
- **figure 3 종**(규칙 A)을 `docs/figures/t5f/` 에 새 이름으로. **그래프 하나는 이전/이후의
  chunk 별 target 이름을 위아래로 나란히 둔 띠 그림**이어야 한다 — 무엇이 바뀌었는지가 한눈에.
- raw 와 script 는 `outputs/verify/T5f/`. `T5d` 의 script 를 넓혀 써도 된다.
- **코드를 고치지 않는다.** **포트 8000 서버(PID 2814716)를 죽이지 마라.** `pkill -f` 금지.
- 해석·판정을 쓰지 않는다. 박힌 수치 밖의 값을 지어내지 마라.
- 규칙 I — technical term 은 영어로.

## 이 task 가 답하는 질문

**target 을 1 등으로 고르게 한 것이 옳은 물체를 고르고 있는가, 그리고 그 대가는 무엇인가.**
