# T5d — attention 이 무엇을 보고 언제 옮겨가나, 그리고 실패한 grounding 은 어떻게 생겼나

> writer: lead (A0) · 2026-09-25 · 주 담당 **A2(verifier)** · 산출 **하나**: `handoff/T5d.verify.json`
> 사용자 지시 2026-09-25. **이 STEP 은 결함을 세는 것이 아니라 grace window 를 정할 근거를 만드는 것이다.**

## 왜 — 전제를 바꾼다

앞선 `T5c` 는 *"75 chunk 중 grounding 성공 16"* 을 결함으로 셌다. **그 전제가 틀렸다** (사용자
판정): 물체를 **집기 직전·직후**, 바구니에 **놓기 직전·직후**는 attention 이 흔들리는 것이
당연한 구간이다. 완전무결을 요구하지 않는다. **몇 frame 을 버티면 되는지**를 정하면 된다.

그래서 이 STEP 이 답할 것은 둘이다:
1. **attention 이 시작부터 과제 성공까지 무엇을 보고 언제 무엇으로 옮겨가나** — 흐름 자체.
2. **실패한 grounding 이 실제로 어떻게 생겼나** — 그냥 "없음" 인가, 엉뚱한 물체를 짚었나,
   점수만 낮고 위치는 맞았나.

그 둘이 나오면 grace window 를 **수치로** 정할 수 있다.

## 범위 — `t_step ≤ 400`

과제는 **step 350~400 안에 성공한다** (사용자 판정). 그 뒤 구간은 판정에 쓰지 않는다.
`t_step ≤ 400` 은 **seq 1..51** 이다 (`t_step = (seq − 1) × 8`). 뒤 구간도 재는 것은 자유지만
**표와 판정은 seq 1..51 기준으로 낸다.**

## 자산 — 이미 있다. 새로 촬영하지 않는다

| | |
|---|---|
| live shadow 기록 | `outputs/live_test/20260925_t5shadow/shadow_ep1807/` (75 chunk / 600 step) |
| offline 재생 기록 | `outputs/live_test/20260925_ep1807/run_0000` |
| attention | `benchmark/ag3s/asset/data/attention_16d_ep1807.npz` |
| 네가 만든 script | `outputs/verify/T5c/measure_offline_grounding_ep1807.py` — **이것을 넓혀 쓴다** |

**live 경로는 `target_score` 를 와이어에 안 싣는다** (`benchmark/trajopt/safe_policy.py:294-301`).
그래서 점수 계열은 `T5c` 와 같이 offline 재생으로 낸다. **어느 경로의 수치인지 키 이름에 박아라.**

## 재는 것 1 — attention 의 흐름 (frame 별)

**frame 마다 모든 cluster 의 점수를 내고, 1 등이 무엇인지 적는다.** cluster 를 MuJoCo 참값으로
이름 붙여라 (apple · banana · orange · pear · crate · table · robot · unknown) — 각 cluster
중심에서 가장 가까운 참값 물체로.

| 내야 할 것 | 형태 |
|---|---|
| frame 별 **1 등 물체의 이름과 점수** | 75 행 표 |
| frame 별 **apple 의 점수** (1 등이 아니어도) | 75 행 |
| **구간 요약** | *"seq 1-9 는 apple(p50 0.79) → seq 10-21 은 robot(p50 0.31) → …"* 처럼 **1 등이 바뀌는 지점**으로 끊어서 |
| 각 구간이 **과제의 어느 국면인가** | gripper 열림/닫힘과 apple 의 참값 높이로 표시 (approach · grasp · transport · place · retreat) |

**이 표가 이 STEP 의 머리다.** 사용자가 보고 *"여기서 이렇게 옮겨가는 게 맞다/틀리다"* 를
바로 읽을 수 있어야 한다.

## 재는 것 2 — 실패한 grounding 은 어떻게 생겼나

seq 1..51 에서 grounding 이 실패한 **35 chunk** (no_target 32 + degraded 중 low_score 3) 각각에 대해:

| 무엇 | 왜 |
|---|---|
| 그 frame 의 **1 등 cluster 가 무엇이었나** (참값 이름) | apple 이 2 등이었나, 아예 후보에 없었나 |
| **apple 의 점수와 순위** | 점수만 낮은 건가, cluster 자체가 안 만들어진 건가 |
| **apple 점이 point cloud 에 몇 개 남아 있나** | self-filter 가 지웠는지 가른다 |
| 1 등 점수가 `threshold` 0.25 에 **얼마나 못 미쳤나** | 간신히 못 넘은 것과 한참 아래를 가른다 |
| `GraspLatch` 가 그 frame 에 **무엇을 쥐고 있었나** (`manipulated`) | latch 는 쥐고 있는데 grounding 만 놓쳤는지 |

**마지막 줄이 중요하다.** `pipeline.py:406` 은 `grounding.target is None` 만 보고
latch 를 참조하지 않는다 (lead 확인). latch 가 그 구간에 `obj0` 를 쥐고 있었다면 **정보는
있는데 안 쓰고 있는 것**이다.

## 재는 것 3 — grace window 는 몇 frame 이면 되나

재는 것 1·2 가 나오면 이것은 집계다.

- 실패의 **연속 길이 분포** (seq 1..51 안에서). 최장 몇 frame 인가.
- 각 실패 구간이 **과제의 어느 국면에 걸쳐 있나** — grasp 직전/직후, place 직전/직후에
  걸친 것과 그 밖의 것을 **갈라서** 센다.
- **"N frame 까지 버티면 실패 구간의 몇 %가 덮이나"** 를 N = 1..20 으로 낸 표.

**해석은 쓰지 마라.** 이 표만 있으면 lead 와 사용자가 N 을 고른다.

## 규칙

- 산출은 `handoff/T5d.verify.json` **하나**. `T5c.verify.json` 을 덮지 마라.
- **figure 3 종** (규칙 A) 을 `docs/figures/t5d/` 에 새 이름으로. 그중 **그래프 하나는
  "frame 축 × 1 등 물체" 의 띠 그림**이어야 한다 — 언제 무엇으로 옮겨가는지가 한눈에 보이게.
  `make-figure` skill 을 읽어라.
- raw 와 script 는 `outputs/verify/T5d/`.
- **코드를 고치지 않는다.** `benchmark/**/*.py` 는 A1 의 것이다.
- **포트 8000 의 서버(PID 1987912)를 죽이지 마라.** `pkill -f` 금지 — PID 로만.
- 새로 촬영하지 마라. 위 기록을 읽는다.
- 박힌 수치(75 · 51 · 16 · 35 · 32 · 3 · threshold 0.25 · t_step = (seq−1)×8) 밖의 값을
  지어내지 마라. 모르면 `not_measured` 에.
- **규칙 I — technical term 은 영어로 쓴다** (`threshold`·`clearance`·`grounding`·`latch`·
  `attention`·`cluster`·`pixel`). 한국어 번역어를 쓰지 마라.

## 이 task 가 답하는 질문

**attention 은 과제를 따라 제대로 옮겨가는가, 그리고 흔들리는 구간을 몇 frame 만 버티면 되는가.**
