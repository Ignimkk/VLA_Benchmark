# AG3S + TO 실시간화 계획 — chunk pipelining

> writer: lead · 2026-10-09 · 상태: **계획만 (보류)**. 사용자 판정 (2026-10-09): "실시간화 (chunk pipelining) 를 위한 계획 문서만 작성해 놓고, 추후에 진행."
> 숫자 출처: `handoff/T43TP.impl.md` (V8 server 기록 분해, TP 통제 측정), `handoff/T43TK.verify.json` (client · sim 동작). 아래에서 **측정** 과 **요구치 / 추정** 을 구분해 적는다.

## 1. 목표와 기준

- **실시간 기준:** 정책 chunk 중 실행되는 8 행 (`execution_length = 8`) 을 15 Hz 로 실행하면 8 × 1/15 s = **533 ms** 이다. 다음 chunk 가 이 안에 준비돼야 로봇이 멈추지 않는다.
- **지금은 동기 방식이다:** client 가 관측을 보내고 응답을 기다린 뒤 8 행을 실행한다. simulation 은 기다리는 동안 멈춘다 (TK: `pi05_infer.py` 가 freeze 시각을 잡고 server 를 부른 뒤, 첫 행을 적용할 때까지 `mj_step` 을 돌리지 않는다 — snapshot v8 `:1662`, `:1696`, `:1985`, 주석 `:1903-1905`). 그래서 simulation 결과는 지연과 무관하고, **실제 로봇에서만** 문제가 된다.
- **목표:** 실제 로봇에서도 chunk 사이에 멈춤 없이, 지금과 같은 안전 판정 (AG3S → TO → gate) 을 유지한다.

## 2. 현재 지연 (측정)

V8 server 기록, run 별 중앙값의 중앙값 (`T43TP.impl.md`):

| 단계 | V8 기록 (ms) | T43 TP 권장 조합 기대 (ms) |
|---|---:|---:|
| π0.5 inference | 398 | 398 |
| AG3S (지각 · TSDF/ESDF) | 258 | 258 |
| TO | 76 | ≈ 47 (p90 ≈ 97, 추정) |
| client · 전송 · 기록 (왕복 − server_total) | 209 | 209 |
| **왕복** | **948** | **≈ 920** |

- TO 는 T43 TP (`--to-fast-eval --esdf-host-mirror`) 로 V8 수준이 되었다 (통제 측정 GPU 경로 중앙 38.9 · p90 69.2 ms; 온라인 값은 보정 추정).
- **TO 를 0 으로 해도 inference + AG3S 가 656 ms** — 동기 왕복으로는 533 ms 에 들어갈 수 없다.
- V11 라운드의 server TO 시간 (322–625 ms) 은 pod 부하 (load1 60–90) 와 섞여 있었다. 측정은 부하를 함께 기록한 조건에서만 비교한다.

## 3. 방식 — chunk pipelining

chunk k 의 8 행 실행을 **시작할 때** chunk k+1 을 요청한다. 실행 533 ms 동안 server 가 inference → AG3S → TO → gate 를 끝내면, chunk k 가 끝나는 순간 k+1 을 바로 실행한다.

```
시간 →
client 실행:  [ chunk k : 8 행, 533 ms ][ chunk k+1 : 8 행 ][ ... ]
server 계산:  [ k+1 계산 (≤ 533 ms) ]   [ k+2 계산 ]
               ↑ k 실행 시작 시 관측 전송
```

**필요 조건 (요구치):** server_total + 전송 ≤ 533 ms (여유 포함 목표 ≤ 450 ms). 지금 ≈ 920 ms 이므로 TO 밖에서 **약 390 ms** 를 줄여야 한다 (TP 의 계산, 측정 아님).

### 3.1 pipelining 이 새로 만드는 문제

| 문제 | 내용 | 대응 후보 |
|---|---|---|
| **관측 시점 어긋남** | k+1 은 chunk k 실행 **시작** 시점 관측으로 계산되는데, 실제로는 k 가 **끝난** 자세에서 실행된다 (8 행 뒤) | (a) 관측 시점 상태에 chunk k 의 명령을 적용해 k 끝 자세를 **예측** 하고 그 자세를 TO 의 시작점 (`q_now`) 으로 쓴다. (b) k+1 의 앞 몇 행을 k 의 끝과 이어 붙이는 연결 (T43 D, chunk 경계 연속성) |
| **chunk 경계 반전** | 지금도 경계에서 명령 방향 반전 29–36 % (TK). 시작점이 실제 측정 자세라 생긴다 | (a) 의 "명령 흐름에서 이어서 시작" 이 그대로 대응. TA 의 C 경계 속도 행 (`problem.py:418`) 재사용 |
| **장면 갱신 지연** | AG3S 거리장이 한 chunk (533 ms) 늦은 관측으로 만들어진다 | 거리장은 episode 동안 누적 (TSDF running sum, OC) 이라 정적 장애물에는 영향이 작다. 움직이는 물체 · 쥔 물체는 별도 확인 |
| **gate · HOLD 의미** | k+1 이 HOLD 면, k 가 끝난 뒤 무엇을 하나 | HOLD 는 지금처럼 측정 자세 유지 (`--safe-hold-mode measured`). 판정이 늦게 도착하면 HOLD 로 닫는다 (fail-closed) |
| **attach · latch 상태** | grasp latch · slip detach (Z3) 판정이 한 chunk 늦게 반영 | 상태 갱신은 실행 피드백 (exec_feedback) 기준이라 순서만 맞추면 된다 — 확인 항목 |

## 4. 지연 줄이기 — 단계별 후보 (측정 전, 후보)

| 단계 | 지금 | 목표 (요구치) | 후보 |
|---|---:|---:|---|
| inference | 398 | ≤ 250 | flow-matching step 수 축소 (정확도 영향 측정 필수), XLA compile 캐시 · autotune 고정, 이미 계산한 prefix KV 재사용 여부 (AttentionSampler 가 같은 cache 를 씀) |
| AG3S | 258 | ≤ 150 | TSDF 통합 · ESDF 생성의 GPU 경로 분해 측정, 변하지 않은 영역 재계산 생략, attention seed · clustering 비용 분해 |
| TO | ≈ 47 (p90 97) | p90 ≤ 100 | T43 TP 로 달성 (추정). 부하가 높으면 넘는다 — CPU 단일 thread 라 pod 부하에 민감 |
| 전송 · 기록 | 209 | ≤ 50 | 이미지 크기 · 압축, 같은 기계면 local socket, `--record-constraints` npz 기록을 응답 경로에서 분리 (비동기 기록) |

## 5. 단계별 진행 계획

| 단계 | 내용 | 산출 | 판정 기준 |
|---|---|---|---|
| **P0 측정 기반** | 단계별 지연을 부하와 함께 기록하는 profiler (server · client 양쪽), sim 을 멈추지 않는 **실시간 sim 모드** (wall clock 으로 진행) 를 client 에 flag 로 추가 | 지연 분해 표 (부하 bin 별) | 같은 부하에서 재현 가능한 숫자 |
| **P1 기록 분리** | 응답 경로에서 npz · 녹화를 비동기로 분리, 전송 포맷 축소 | 전송 · 기록 시간 | 209 → ≤ 50 ms, 결과 비트 동일 |
| **P2 AG3S 단축** | AG3S 단계 분해 → 큰 항목부터 | AG3S 시간 | 258 → ≤ 150 ms, 거리장 동일 (eikonal 자가진단, curobo-esdf skill) |
| **P3 inference 단축** | step 수 · compile · cache | inference 시간, 정책 출력 차이 | ≤ 250 ms, E0 성공률 유지 (학습 범위 episode) |
| **P4 pipelining** | client 비동기 요청 + 끝 자세 예측 시작점 + 경계 연결 | 실시간 sim 모드 closed-loop | chunk 사이 정지 0, 경계 반전 · 속도 초과 비교, 성공 · 접촉 (CT 규칙) 이 동기 방식과 동등 |
| **P5 검증** | 학습 범위 episode × 과일 4 종, 동기 vs pipelining 짝 비교 | verify.json · figure · 비교 영상 | McNemar 로 동등 이상 |

## 6. 검증 원칙

- 모든 변경은 flag (기본 off, off 면 비트 동일) — 지금까지의 T43 규칙 그대로.
- 지연 숫자는 **같은 시간대 · 같은 부하** 에서 짝으로만 비교한다 (V11 에서 부하가 섞여 판정 불가였던 교훈).
- episode 는 **학습 범위 (train split)** 에서 E0 가 확실히 성공하는 것만 쓴다 (2026-10-09 사용자 판정).
- 안전 판정 (gate · HOLD) 은 fail-closed — 늦게 온 판정은 HOLD 로 닫는다.

## 7. 열린 질문

1. 실제 로봇의 실행 주기 · 제어기 지연은 simulation 과 같은가 (sim 의 servo 지연 ≈ 2 행, TK — 실제 로봇은 별도 측정).
2. inference step 수를 줄였을 때 정책 품질 (특히 파지) 이 유지되는가.
3. 끝 자세 예측에 servo 모델 (`servo.py`, 예측 오차 중앙 0.02–0.08 mm, HX) 을 쓸 수 있는가 — TO 안이 아니라 **시작점 예측에만** 쓰는 것은 T43 SV 의 문제 (테이블 띠) 와 무관하다.
4. GPU 1 장에서 inference · AG3S · TO 를 겹쳐 돌릴 때 서로의 지연에 미치는 영향.
