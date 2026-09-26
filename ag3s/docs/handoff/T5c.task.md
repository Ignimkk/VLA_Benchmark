# T5c — shadow 루프를 판정한다 (ep1807 전 구간, 장애물 없는 씬)

> writer: lead (A0) · 2026-09-25 · 주 담당 **A2(verifier)** · 산출 **하나**: `handoff/T5c.verify.json`

## 자산 — **이미 돌려 놓았다. 다시 돌리지 마라**

lead 가 `pi05_TO_hybrid/logs/run_shadow_ep1807.sh` 로 띄웠다. 긴 실행을 agent 안에 넣지 않는다 —
**너는 산출 파일만 소비한다.**

| | |
|---|---|
| 기록 | `outputs/live_test/20260925_t5shadow/shadow_ep1807/` — `manifest.json` · `frames.jsonl` · `completeness.json` |
| 클라이언트 로그 | `outputs/live_test/logs/t5shadow_run.log` |
| 서버 로그 | `outputs/live_test/logs/serve_safe_8000_shadow.log` |
| 조건 | ep1807, **600 제어 스텝 = 75 청크**, `--safe-remote --safe-shadow` |
| 서버 | `serve_safe.py --shadow`, cuRobo backend coarse 20 mm + fine 5 mm, TSDF 5 mm, self-filter **218** sphere, constraint **120** sphere (arms) |

**shadow 의 뜻**: 서버가 AG3S·cuRobo·SQP·판정을 전부 돌리고 응답에 **refined `actions` 와 원본
`actions_reference` 를 함께** 싣는다. **로봇이 실제로 날린 것은 `actions_reference`(원본)** 다.
그래서 "수정이 여유거리를 나쁘게 만드는가" 를 로봇을 움직이기 전에 볼 수 있다.

## 재는 것 — 다섯

### 1. 여유거리(clearance) — **T5 의 핵심 합격 조건**

**MuJoCo 참값에서 `refined` 가 `reference` 보다 나빠지지 않는가.** 두 청크가 같은 기록에 나란히
있다. 청크마다:

- 각 청크의 두 궤적을 MuJoCo 로 굴려(또는 참값 기하로 평가해) **최소 clearance** 를 낸다.
- **`refined_min − reference_min` 의 분포**를 낸다: 음수(나빠짐) 청크 수 · 그 최댓값 · 중앙값.
- **나빠진 청크가 하나라도 있으면 그 청크 번호와 값을 전부 적어라.** 개수만 적지 마라.

### 2. HOLD 가 몇 번, 그리고 **왜**

`T5b`(2026-09-25)로 사유 코드가 프레임마다 실린다 — 응답의 `ag3s` 블록에
`reasons[{code, detail}]` 이 있고, 등록부는 `benchmark/ag3s/runtime/degradation.py` 의 `CODES`(12 개)다.

- **사유 코드별 집계**(코드 → 청크 수)를 내라. 이것이 이 측정의 알맹이다.
- **`degraded_without_reason` 이 하나라도 나오면 그것을 제일 먼저 적어라** — 사유를 못 붙인
  분기가 남아 있다는 뜻이고 `T5b` 의 구멍이다.
- 직전 smoke(ep1800, 2 청크)에서는 **2 중 1 개가 HOLD**였고 `ag3s_status=degraded` ·
  `geometry_certified=False` · `trajopt_status=violated` 인데 **`max_violation_m = 0.0`** 이었다.
  **75 청크에서 그 비율과 사유가 무엇인지**가 질문이다.
- **가설 하나를 확인해라** (추측이므로 수치로만 답해라): `multiview.py:378-396` 의
  **per-camera `max_points` cap** 분기가 원인인가. X3 는 *"상한은 카메라별이 아니라 합친 구름에
  걸린다"* 를 **offline 경로**에서 쟀는데 live 의 multiview 에는 카메라별 분기가 따로 있다.
  걸렸다면 **어느 카메라 · 몇 점 · voxel 이 얼마로 커졌는지**가 detail 에 있다.

### 3. 과제가 끝까지 가는가 (장애물 없는 씬)

**로봇이 날린 것은 원본 청크**이므로 이것은 정책 자체의 성공을 보는 것이기도 하다.
T2 가 같은 에피소드에서 낸 MuJoCo 참값과 **나란히** 둬라:

| | T2 (녹화, in-process 경로) | **이번 (live, 서버 경로)** |
|---|---|---|
| apple 들림 | 240.6 mm | ? |
| apple ↔ crate 수평거리 | 339 → 71 mm | ? |
| 다른 과일 셋 drift | banana 0.1 · orange 0.2 · pear 0.6 mm | ? |

**둘이 다르면 그것이 발견이다** — 같은 정책·같은 에피소드인데 경로가 달라 결과가 다르다는 뜻이다.
해석은 쓰지 말고 수치만.

### 4. 상태 일치 — 모순이 있는가

`ag3s_status` · `geometry_certified` · `trajopt_status` · `safe` · `should_execute` ·
`field.state` 가 **서로 모순되는 프레임**을 찾아라. 예: `safe=True` 인데 `geometry_certified=False`,
또는 `should_execute=False` 인데 `executed=True`. **모순 0 이면 그렇게 적어라.**
`completeness.json` 의 `unexplained_carried_or_stale` · `timestamp_reversals` ·
`duplicate_sequence_ids` 도 함께 옮겨라.

### 5. 지연 — 단계별로

`timing_ms` 를 단계별(캡처 · AG3S · ESDF · SQP · 서버 total · 왕복)로 **P50 · P95 · max**.
**첫 청크는 cuRobo kernel 컴파일이 섞이므로 따로 적어라** (smoke 에서 t=0 이 48.78 s,
t=8 이 3.77 s 였다). 청크 예산은 `8 / 15 Hz = 533 ms` 다.
**실시간성은 이번 판정 항목이 아니다**(사용자 판정: 범위 밖) — 수치만 남긴다.

## 규칙

- **해석·판정을 쓰지 않는다.** 수치와 경로만 `handoff/T5c.verify.json` 에. schema 는 `_SCHEMA.verify.json`.
- **figure 3 종**(규칙 A: 실제 씬 · 그래프 · 표)을 `docs/figures/t5c/` 에 **새 이름**으로.
  그래프는 **청크별 `refined − reference` clearance 차**를 75 청크에 걸쳐 보이는 것이 제일 중요하다.
  `make-figure` skill 을 읽고 스타일을 맞춘다.
- raw 와 측정 script 는 `outputs/verify/T5c/`.
- **코드를 고치지 않는다.** `benchmark/**/*.py` 는 A1 의 것이다.
- **포트 8000 의 서버를 죽이지 마라.** 필요하면 lead 에게 말해라. `pkill -f` 금지 — PID 로만.
- **기록을 다시 만들지 마라.** 위 경로의 것을 읽는다. 부족하면 `not_measured` 에 적고 lead 에게 요청.
- 위에 박힌 수치(600 · 75 · 218 · 120 · 240.6 mm · 339 → 71 mm · 0.1/0.2/0.6 mm · 48.78 s ·
  3.77 s · 533 ms · 0.0) 밖의 값을 지어내지 마라.

## 이 task 가 답하는 질문

**장애물이 없는 씬에서 서버-로컬 닫힌 루프가 서는가, 그리고 TO 의 수정이 여유거리를 나쁘게
만들지 않는가.**
