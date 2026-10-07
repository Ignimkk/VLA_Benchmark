# T43TA — 구현: TO 경로 모양 A (look-ahead) · B (편향 변화율) · C (속도 + 가속도), 오프라인 비교

> writer: ag3s-implementer (TA2, 네트워크로 끊긴 TA 를 이어받음) · 읽는 쪽: lead, verifier, scribe · 2026-10-07
> 기준 benchmark `308ddfd` · task: [T43.task.md](T43.task.md) §"사용자 판정 (2026-10-07) — TA" + V10 보류 메모 + 15:40 중단 메모
> 오프라인 산출물: `outputs/impl/T43TA/` — `raw/` (V8 19 + V7g 18 run) · `agg/agg_V8.json` · `agg/agg_V7g.json` · `p/` (스크립트) · `fig/` (그림 3)

## 무엇을 했나 (평이한 요약 먼저)

TO 에 세 개의 flag 를 더했다. 셋 다 기본 off 이고, off 이면 출력이 바이트 단위로 같다 (HEAD `308ddfd` 와 실제 V8 기록 150 청크로 대조).
- **A**: 계획 창을 16 step 으로 늘린다. 실행 · 인증은 앞 8 step 그대로이고, 뒤 8 step 은 장애물만 보는 내다보기 창이다.
- **B**: 정책과의 편향 `D = Q − Q_ref` 의 step 간 변화량에 비용을 둔다. 지난 청크의 마지막 실행 편향에서 이어지게 한다.
- **C**: 명령의 2 차 차분에 servo 상수로 유도한 상한을 soft 로 건다. 청크 경계도 포함한다. 그리고 지난 명령 `u[−1]` 에서 새 청크 첫 행으로 가는 속도 행을 더한다.

이전 TA 작업물은 설계 골격 (A · B · C, 테스트 30, 오프라인 도구) 이 쓸 만해서 유지했다.
다만 오프라인 결과를 다시 보니 **A 가 파지 청크를 크게 움직였다** (손바닥이 정책에서 최대 88 mm 벗어남).
원인을 추적해 보니 A 자체가 아니라 **16 step QP 를 OSQP eps 1e-3 으로 느슨하게 푼 탓**이었다. 첫 SQP step 이 0.10 rad 로 엉뚱하게 나갔고, reference 가 위반 상태라 merit 가 그것을 받아들였다. eps 1e-4 에서는 그 step 이 0.019 rad 이고 base 와 같아진다.
그래서 `--qp-eps` 를 더했고, A 의 비교는 같은 QP 정확도의 기준선 `Q4` 대비로 한다. 그 밖에 바꾼 것은 다음과 같다.
- A 에 gate 를 넣었다. 꼬리 20 mm 안에 장애물이 없으면 8 step 풀이와 같게 푼다.
- 꼬리에서 지지 평면 행을 뺐다.
- 꼬리에서 잔상 판정에 쓰는 capture 부피를 배선했다.
- lookahead 는 esdf backend 에서만 허용한다.
- C 의 기본 가중치를 1.0 에서 0.01 로 낮췄다.
- 오프라인 도구에 평면 행 · outward · Q4 비교 · 반복 타이밍을 더했다.

오프라인 결과 요약 (V8 19 run 1037 청크, open loop):
- **A 단독**은 Q4 대비 허들 지표가 거의 그대로다. 파지 청크도 거의 그대로다 (사과 앞 72 청크 중 0.5° 넘게 움직인 것 2, 최대 outward +4.2 mm).
- **B** 를 넣으면 servo 예측 허들 clearance p10 이 −0.5 → +2.9 mm, 접촉 순간 clear 가 9 → 15/22 가 된다. 대신 청크 경계의 꺾임이 조금 늘고 파지 청크가 소폭 움직인다.
- **C** (relaxed 1°, w 0.01) 는 그 꺾임을 base 아래로 되돌리고 (경계 2 차 차분 p90 0.041 → 0.035, 반전 0.169 → 0.152), 경계 속도 초과를 0 으로 만든다.
- TO CPU 시간은 청크별 중앙값 기준 A 1.9× · A+B+C 2.1× 이다.

## 1. 속도 제한은 모든 행에 실제로 걸리나 (lead 요청, 먼저)

**결론.** 청크 안의 속도 행과 row 0 anchor 는 모든 행에서 지켜진다. HX 의 "0.32 rad/행 초과" 는 기준값을 잘못 잡은 데서 나왔다. 진짜 빈틈은 청크 경계다. 지난 명령 `u[−1]` 에서 새 청크 `Q0` 로 가는 한 step 에는 아무 속도 행도 없다.

| 무엇 | 코드 | 기록 (V8 19 run, 정제 청크 · 관절마다 자기 `max_step` 으로 나눈 비) |
|---|---|---|
| `max_step = v_URDF · velocity_scale(0.9) · Δt(1/15 s)` — 관절마다 다름: 0.189 (팔 0–3) · **0.377** (4–5) · 0.126 (6) rad/행 | `trajopt/config.py:306` · `trajopt/limits.py:360` | HX 는 모든 관절을 0.1885 rad (10.8°) 로 나눴다. 0.32 rad (V5) 는 0.377 관절의 0.85× 이다 |
| 청크 안 `|Q[k+1] − Q[k]| ≤ max_step` (soft, 10·w_slack) | `trajopt/problem.py:301` (행 수) · `:396` (행) — HEAD `:202` | 실행 행 7084 개 중 최대 **1.000×** (1 행 1.0000× = solver 허용오차) |
| row 0 anchor `|Q0 − q_now| ≤ max_step` (hard box) | `trajopt/problem.py:370` — HEAD `:256` | 1037 청크 중 최대 **1.000×** |
| **청크 경계 `|Q0 − u[−1]|`** | 행이 없다 — anchor 는 *측정된* `q_now` 기준이고, `q_now` 는 명령보다 τ = (kv + b)/kp ≈ 2.4–2.8 행 늦다 | V8: 999 경계 중 1 개가 넘음 (**1.20×**, `|u[−1] − q_now|` 최대 1.66×). V7g: 2 개 (1.25×). V1–V5 (W index): 4204 중 11 개 (최대 **2.60×**) |

C 를 켜고 실행 이력이 있으면 `|Q[:, 0] − u[−1]| ≤ max_step` 행이 더해진다. 무게는 다른 속도 행과 같은 10·w_slack 이다 (`problem.py:418`).
오프라인에서는 B 나 C 가 있는 모든 변형에서 경계 초과 청크가 0 이다 (base 2).
측정: `outputs/impl/T43TA/p/vel_audit.py` → `raw_aux/vel_audit.json` (같은 비를 오프라인 집계의 `vel_boundary_ratio` 로도 쓴다).

## 2. 설계 (flag · 식)

| | flag (serve_safe) | config | 식 / 동작 |
|---|---|---|---|
| A | `--plan-horizon 16 --lookahead obstacles` · `--lookahead-gate-mm 20` (기본 20, `always` = gate 없음) | `horizon.lookahead` · `horizon.lookahead_gate` (`config.py:102,104`) | 최적화는 P = 16 step 이고, 실행 · 인증 (`max_violation`, status, verdict · HOLD 분류) 은 0–7 만 한다. 꼬리 8–15 의 규칙: ESDF 행 중 최근접 표면이 **장애물인 것만** 남긴다 (target · 지지면 · 쥔 물체 자기 잔상은 뺀다, `linearize.py:829`). 지지 평면 행 자체도 뺀다 (`:1069`). slack 은 같은 w_slack 이고, merit = 창 최대 위반 + 꼬리 최대 위반이며, 꼬리 기록은 `to.metrics.lookahead` 에 남는다. **gate**: reference 꼬리의 장애물 행 최소 clearance 가 20 mm 이상이면 8 step 최적화기로 풀고 꼬리는 reference 그대로 둔다 (`sqp.py:522,540`) |
| — | `--qp-eps 1e-4` | `qp.eps_abs = eps_rel` | A 와 함께 쓴다. 이유는 §3 |
| B | `--w-deflection-rate W` | `cost.w_deflection_rate` (`config.py:247`) | `W·(Σ_k ‖D[:,k+1] − D[:,k]‖² + ‖D[:,0] − D_prev‖²)`, `D = Q − Q_ref`, `D_prev` = 마지막 실행 행 − 그 행의 이전 reference. HOLD 뒤나 이력이 없으면 경계 항은 뺀다 (`safe_policy.py:514`). QP 에는 `P += 2W MᵀM`, `q −= 2W Mᵀ(M r + c)` 로 넣는다 (`problem.py:278`). merit 에도 같은 항이 있다 |
| C | `--servo-accel relaxed --servo-accel-tolerance-deg 1 --servo-accel-weight 0.01` | `limits.servo_accel` · `_tolerance` · `_weight` (`config.py:387–394`) | 행 = `[u−2, u−1, Q0 … Q15]` 의 2 차 차분 (청크 경계 포함, `problem.py:135`). 상한 `a_C = ε Δt² / κ`, `κ = τ²/2 − I/kp = (2ζ²−1)/ωn²` (`servo.py:105,125`). relaxed: `b = min(max(a_C, |Δ²Q_ref|), max_step_change)` → TO 가 정책보다 더 날카롭게 만들 수 없다. strict: `b = min(a_C, …)`. 행마다 L1 slack, 무게 `w_slack · weight`. 경계 속도 행도 C 에 속한다 |

**C 상한의 근거 (V8 MJCF, `servo.py` 와 같은 상수).**
- 상수: kp 1500–2000 · kv + b 275–334 · I (armature + 관성 대각) 10.0–11.6 kg·m².
- 파생: ωn 12.2–13.9 rad/s · ζ 1.10–1.12 · τ 2.4–2.8 행 · κ 7.5–10.2 ms².
- 결과: ε = 1° 에서 `a_C` = **0.0077–0.0104 rad/행²** (1.7–2.3 rad/s²).
- 비교: 기존 가속도 행 (`max_step_change` = 0.04 rad/행², 청크 안만) 보다 4–5 배 좁다.
- 이 유도는 이산 servo (`ServoModel`, MuJoCo 와 같은 Euler) 로 검산했다 (`test_c_accel_bound_matches_the_discrete_servo`).
- servo 모델은 평가 (`ta_offline` 의 servo 예측 clearance) 에만 썼다. TO 안에는 상수에서 유도한 상한 하나만 들어간다 (T43SV).

**A 가 실행되지 않을 뒷부분에 회피를 미루지 못하는 이유 (T6d).**
1. 꼬리는 장애물만 본다. 그래서 target 접근과 테이블 접촉은 실행 창에서만 판정된다 (기본 `PLAN_EXECUTION_WINDOW` 와 같다). 이것이 T6d 의 미루기 (사과를 꼬리에서 접근) 를 원천적으로 막는다.
2. 꼬리 위반은 창 위반과 행당 같은 slack 값을 치른다. 그래서 위반을 꼬리로 옮겨도 이득이 없다.
3. 인증은 창만 한다. 꼬리 위반이 청크를 통과시키는 일은 없다.
4. 꼬리가 피하지 못한 것은 다음 청크의 창에서 기본과 같이 풀리고 인증된다.
5. gate 가 닫힌 청크는 정의상 기본과 같다.

## 3. 이어받은 부분 작업에서 무엇을 유지 · 고침 · 다시 했나

| 항목 | 판정 | 내용 |
|---|---|---|
| A · B · C 의 식, `servo.accel_bound`, `_ta_history`, verdict 의 창 한정 | 유지 | diff 를 spec 과 대조했다. B 의 QP 전개와 C 행이 직접 평가와 맞는지는 테스트가 확인한다 |
| `raw/` (13:36–13:52) | **다시 함** | 그때 코드로 만든 V8 기준선 결과는 맞았다. 그러나 아래 수정으로 코드가 바뀌어 전부 다시 돌렸다. 옛 것은 `raw_ta1_superseded/` 와 eps 1e-3 단독 2 차 시도 `raw_ta2a_eps3_superseded/` 에 남겼다 |
| A 가 파지 청크를 움직임 (옛 raw: 0.5° 넘게 움직인 청크 37/106, 최대 35°) | **고침** | (i) 원인 = OSQP eps 1e-3 의 부정확한 16-step 부분문제. 근거는 아래 표 → `--qp-eps` 를 추가했다. (ii) gate: 꼬리에 가까운 장애물이 없으면 기본과 같게 푼다 |
| 꼬리의 지지 평면 행 (온라인에는 평면 행이 있는데 꼬리 규칙은 ESDF 행만 봄) | 고침 | `_clearances_from` 에서 꼬리 평면 행을 `inf` 로 둔다 |
| 꼬리의 쥔 물체 잔상 (HM 이 꺼져 있으면 capture 부피가 씬에 없음 → 사과 자기 상이 장애물로 남음) | 고침 | lookahead 이면 `SafePolicy._held_capture_for_scene` 과 `scene_from_constraint_set` 이 capture 부피를 넣는다. 이 부피는 꼬리와 HM 만 읽는다 |
| primitive 후보 행에는 분류가 없음 → target 후보가 꼬리에 남음 | 고침 | lookahead 는 `collision.backend == "esdf"` 에서만 허용한다 (config 오류) |
| C 무게 기본 1.0 (첫 pass: 새 충돌 청크 19) | 고침 | 기본을 0.01 로 바꿨다. 0.1 에서도 새 충돌 청크가 9 개다 |
| 오프라인 도구 | 보강 | 평면 행 (온라인과 같게 offset + support margin), capture 부피, outward (손가락 중점 – target, row 7), Q4 비교, 3 청크마다 역순 반복 타이밍 |

16-step 부분문제의 정확도 (V8 H ep1967 s19672 chunk 17, 닫기 1 청크 전, reference 창 위반 19.9 mm):

| 변형 | 1 차 SQP step (rad) | 창 최대 편향 | outward (row 7) | CPU ms |
|---|---|---|---|---|
| base (eps 1e-3) | 0.033 | 0.99° | +10 mm | 113 |
| A16 (eps 1e-3) | **0.100** | 4.9° | **+68 mm** (palm 88 mm) | 111 |
| A16 eps 1e-4 | 0.019 | 0.68° | +3 mm | 126 |
| A16 eps 1e-5 | 0.019 | 0.57° | +3 mm | 137 |
| base eps 1e-4 | 0.019 | 0.56° | — | 62 |

polish 를 켜도 바뀌지 않았다. 기본 TO 도 eps 1e-3 에서는 이 청크를 필요보다 크게 (0.99° 대 0.56°) 민다. 기본값은 바꾸지 않았다 (열린 위험 2).

## 4. 오프라인 비교 (V8 기록, 장면 · 정책 chunk 고정, TO 만 재풀이)

`ta_offline` 이 V8 서버 명령줄을 서버 자신의 parser 로 읽어 config 를 만든다. 쥔 사과 행 (HM 분류 off = V8) 을 넣어 운반 청크도 다시 푼다.
변형 이름의 뜻:
- `q` = `--qp-eps 1e-4`, `u` = gate 없음.
- `B20` = W 20.
- `Cr1` = relaxed ε 1° w 0.01, `w1` = w 0.1, `Cs1` = strict.

교차 = 운반 · crate · 복귀 중 기준 명령이 장애물 50 mm 안에 든 226 청크다.

| 변형 | 명령 clearance p10 (mm) | servo 예측 p10 / 중앙 (mm) | servo < 0 청크 | 접촉 clear (V8 14 + V7g 8) | 손바닥 편향 중앙 / p90 (mm) | 경계 2 차 차분 p90 (rad/행²) | 반전 비율 | 경계 속도 > max | 새 / 해소 충돌 청크 | CPU × (중앙 / p90 / 합) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 14.8 | −0.56 / 24.6 | 24 | 6+3 = 9 | 0.6 / 21.7 | 0.041 | 0.169 | 2 | — | 1 |
| Q4 | 14.9 | −1.03 / 24.6 | 25 | 9 | 2.7 / 20.2 | 0.040 | 0.153 | 2 | 2 / 2 | 1.02 / 1.09 / 1.03 |
| A16 (eps 1e-3) | 14.8 | +0.39 / 24.5 | 22 | 9 | 0.6 / 23.1 | 0.043 | 0.174 | 2 | 2 / 3 | 1.87 / 2.10 / 1.61 |
| **A16q** | 14.9 | −0.49 / 24.2 | 24 | 9 | 2.7 / 20.2 | 0.040 | 0.153 | 2 | 4 / 2 | 1.91 / 2.31 / 1.68 |
| A16uq | 14.8 | −0.49 / 24.2 | 24 | 8 | 2.6 / 20.0 | 0.040 | 0.155 | 2 | 4 / 2 | 1.87 / 2.29 / 1.87 |
| A24q | 14.9 | −0.33 / 23.8 | 24 | 9 | 2.6 / 20.1 | 0.040 | 0.159 | 3 | 4 / 2 | 2.76 / 3.45 / 2.30 |
| A16qB5 | 14.8 | +1.94 / 24.1 | 17 | 10 | 2.4 / 18.9 | 0.043 | 0.200 | 0 | 3 / 4 | 1.60 / 2.31 / 1.70 |
| **A16qB20** | 14.7 | +2.94 / 23.9 | 14 | 10+5 = **15** | 2.5 / 19.8 | 0.046 | 0.219 | 0 | 3 / 4 | 1.93 / 2.37 / 1.75 |
| A16qB50 | 14.9 | +3.57 / 23.8 | 10 | 14 | 2.5 / 20.6 | 0.047 | 0.218 | 0 | 3 / 5 | 1.85 / 2.43 / 1.76 |
| **A16qB20Cr1** | 14.8 | +2.85 / 24.6 | 16 | 11+4 = **15** | 2.0 / 19.2 | **0.035** | **0.152** | 0 | 3 / 3 | 2.15 / 3.29 / 2.16 |
| A16qB20Cr1w1 | 14.8 | +3.06 / 24.6 | 16 | 16 | 2.0 / 19.1 | 0.035 | 0.148 | 0 | **9** / 3 | 2.08 / 3.30 / 2.06 |
| A16qB20Cr2 | 14.8 | +2.47 / 24.4 | 17 | 15 | 2.4 / 19.5 | 0.035 | 0.194 | 0 | 4 / 4 | 2.11 / 3.11 / 2.02 |
| A16qB20Cs1 | 14.9 | +2.35 / 24.5 | 15 | 15 | 3.0 / 21.7 | **0.010** | **0.099** | 0 | 3 / 2 | 2.21 / 3.61 / 2.26 |
| A16qB50Cr1 | 14.9 | +3.31 / 24.5 | 14 | 14 | 2.1 / 20.0 | 0.035 | 0.148 | 0 | 4 / 3 | 2.09 / 3.39 / 2.12 |

명령 쪽 TO clearance 는 접촉 순간 22 건 모두에서 거의 늘 ≥ 0 이다 (21–22/22, 모든 변형). 접촉의 원인은 servo 추종 지연이다 (HX · SV).

단계별 새 / 해소 충돌 청크 (`classify_violations`, 창 rows 0–7, base 대비). 청크 수는 접근 297 · closing 49 · 운반 172 · crate 57 · 복귀 462 이다.

| 변형 | 접근 | closing | 운반 | crate 진입 | 복귀 | 교차 명령 < 0 | VIOLATED |
|---|---|---|---|---|---|---|---|
| base | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0 | 8 |
| Q4 | 0/0 | 0/0 | 0/1 | 1/0 | 1/1 | 0 | 7 |
| A16q | 0/0 | 1/0 | 1/1 | 1/0 | 1/1 | 0 | 9 |
| A16qB20 | 0/0 | 0/0 | 2/2 | 0/0 | 1/2 | 0 | 5 |
| A16qB20Cr1 | 0/0 | 0/0 | 1/0 | 1/0 | 1/3 | 1 | 8 |
| A16qB20Cr1w1 | 0/0 | 1/0 | 4/1 | 2/0 | 2/2 | 6 | 14 |
| A16qB20Cs1 | 0/0 | 0/0 | 1/0 | 1/0 | 1/2 | 1 | 9 |

Q4 (QP 정확도만 바꾼 것) 자체가 2/2 를 만든다. 따라서 1–2 개 차이는 solver 잡음 수준이다.

### 파지 구간 (closing) 표 — V9 교훈

- "사과 앞" = 정책 손가락 중점이 row 7 에서 target 60 mm 안인 청크, 또는 attach 청크. 72 개다.
- "파지 집합" 106 = closing · attach · 첫 closing 앞 2 청크. 사과를 떨어뜨린 run 의 공중 closing 도 들어 있다.
- outward = 변형의 row 7 손가락 중점 – target 거리 − Q4 의 그 거리. > 0 이면 사과에서 멀어진 것이다.
- 비교 기준은 **Q4** 다. base 와 Q4 의 차이 자체가 0.5° 넘는 청크 14/72 를 만든다.

| 변형 | 사과 앞 n | 0.5° 넘게 움직임 | outward > 2 mm | outward p90 / 최대 (mm) | 파지 집합 n | 움직임 | outward > 2 | outward 최대 | 새 충돌 | gate 열림 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 72 | 0 | 2 | 1.0 / 7.5 | 106 | 0 | 8 | 101 | 0 | — |
| A16 (eps 1e-3) | 72 | 17 | 6 | 2.4 / **65.2** | 106 | 36 | 13 | 83 | 0 | 69 |
| **A16q** | 72 | **2** | 2 | 0.7 / **4.2** | 106 | 11 | 7 | 27 | 1 | 69 |
| A16uq | 72 | 3 | 2 | 0.8 / 4.2 | 106 | 16 | 10 | 27 | 1 | 106 |
| A24q | 72 | 5 | 3 | 0.9 / 7.6 | 106 | 15 | 8 | 77 | 0 | 72 |
| A16qB5 | 72 | 21 | 4 | 1.7 / 5.3 | 106 | 49 | 18 | 25 | 1 | 69 |
| **A16qB20** | 72 | 27 | 6 | 2.3 / 5.9 | 106 | 60 | 23 | 42 | 0 | 69 |
| A16qB50 | 72 | 30 | 8 | 2.7 / 6.7 | 106 | 64 | 25 | 45 | 0 | 69 |
| **A16qB20Cr1** | 72 | 26 | 4 | 1.0 / 5.8 | 106 | 59 | 20 | 190* | 0 | 69 |
| A16qB20Cr2 | 72 | 27 | 4 | 0.9 / 5.7 | 106 | 60 | 21 | 41 | 0 | 69 |
| A16qB20Cs1 | 72 | 34 | 9 | 3.3 / 6.3 | 106 | 67 | 26 | 185* | 0 | 69 |
| A16qB50Cr1 | 72 | 26 | 7 | 2.3 / 6.5 | 106 | 60 | 24 | 53 | 0 | 69 |

\* 파지 집합의 최대 outward 190 / 185 mm 는 사과 앞 청크가 아니다. 사과를 떨어뜨린 run (H ep1982 s19822) 의 공중 "closing" 청크로, 막대 옆에서 다시 닫는 중이고 target_centroid 는 옛 자리에 남아 있다. 사과 앞 72 청크의 최대는 5.8 / 6.3 mm 다.

읽는 법:
- **A (eps 1e-4) 는 파지를 바꾸지 않는다** (2/72, 최대 4.2 mm).
- B 는 사과 앞 청크의 약 1/3 을 0.5° 넘게 움직인다. 지난 청크의 편향을 이어받기 때문이다. 다만 손가락이 사과에서 멀어지는 양은 p90 1–2.3 mm, 최대 6 mm 다. V9 의 파지 손실 (closing outward > 2 mm 가 지속, 최대 24 mm) 보다 작지만 0 은 아니다 → 온라인 A+B 에서 G 를 확인해야 한다 (열린 위험 1).
- strict C 는 파지 청크를 가장 많이 움직인다 → relaxed 를 권한다.

### 그림 (규칙 A)
- `outputs/impl/T43TA/fig/t43ta-1-scene-carry.png` — 실제 장면. H ep1807 s18071 운반 청크 23–27 (t = 206 사과–막대 접촉 포함) 에서 쥔 사과 중심의 경로를 그렸다. 명령은 실선, servo 예측은 점선이다. 배경은 그 장면 ESDF 의 연직 단면이다. A+B+C 의 경로가 base · A 의 꺾인 지그재그 (z 1.27 m 까지 솟구침) 대신 완만해진다. `.json` sidecar 가 있다.
- `fig/t43ta-2-graph.png` — 변형별 지표 8 개.
- `fig/t43ta-3-table.png` — 위 표의 그림판. 모두 `.json` sidecar 가 있다.

## 5. 권장 값과 온라인 서버 flag (각각 V8 flag 위에)

V8 flag (`outputs/verify/T43/V8/p/start_server.sh`, `$FLAGS`):
`$E3B --obstacle-margin 0.010 --held-body-cover $T2 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both`
**HM (`--held-obstacle-margin`) 없음.** snapshot 은 이 커밋 이후의 benchmark 여야 한다.

| 단계 | 추가 flag |
|---|---|
| A | `--plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4` |
| A+B | A + `--w-deflection-rate 20` |
| A+B+C | A+B + `--servo-accel relaxed --servo-accel-tolerance-deg 1 --servo-accel-weight 0.01` |

값을 고른 이유:
- **H = 16**: 24 는 지표가 같은데 CPU 가 2.8× 다.
- **gate 20 mm**: 지표는 gate 없음과 같고 CPU 합이 1.87 → 1.68 로 준다.
- **eps 1e-4**: A 에 필수다.
- **B = 20**: 50 은 허들이 조금 낫지만 파지 청크 outward > 2 mm 가 6 → 8 이다.
- **C relaxed 1° w 0.01**: w 0.1 이면 새 충돌이 9, strict 는 파지를 가장 많이 움직인다.

온라인에서 볼 것:
- A 단독의 오프라인 이득은 거의 없다. open loop 라 앞 청크의 이른 회피가 다음 청크의 시작 상태로 이어지지 않기 때문이다. A 의 효과는 온라인에서만 보인다.
- `to.metrics.lookahead` (gate · 꼬리 위반), `to.metrics.deflection_rate`, `to.metrics.servo_accel` (overshoot · 경계 속도 비), `summary.ta_history`.
- TO 시간: 오프라인 CPU 중앙값 64 → 113 (A) → 133 ms (A+B+C). SQP 는 3 회 고정이라 시간 예산으로 끊기지 않는다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/trajopt/config.py:68–104` · `:136–161` · `:163` | `horizon.lookahead` · `lookahead_gate` (+ 문서표 · 검증) · `lookahead_from` | A |
| `…/config.py:247` | `cost.w_deflection_rate` | B |
| `…/config.py:350–394` | `limits.servo_accel` · `_tolerance` · `_weight` (기본 0.01) + 속도 결과 문서 | C |
| `…/config.py:866–882` | lookahead × servo_model/sweep_check 거부 · lookahead 는 esdf 만 | A |
| `benchmark/trajopt/problem.py:116–190` · `:278–290` · `:296–316` · `:409–426` · `objective` | B 항, C 행, 경계 속도 행, merit 의 같은 항 | B · C |
| `benchmark/trajopt/sqp.py:103–119` · `:266–277` · `:522–565` · `:805–825` · `:929` | gate 최적화기 · gate · 창/꼬리 merit · 창만 인증 · `metrics.lookahead/deflection_rate/servo_accel` | A · B · C |
| `benchmark/trajopt/linearize.py:812–867` · `:1064–1070` · `:1095` · `worst_row(steps=)` · `:1823–1830` | 꼬리 행 선택 · 꼬리 평면 행 제거 · `step_clearances` · 씬 지지면/capture | A |
| `benchmark/trajopt/safe_policy.py:514` · `:881` · `:1651` · `classify_violations(steps=)` | B/C 이력 · capture 배선 · verdict 창 한정 | A · B · C |
| `benchmark/trajopt/refiner.py:158` | context 의 이력 전달 (없으면 인자 없음) | B · C |
| `benchmark/trajopt/servo.py:105` · `:125` | `shape_coefficient` · `accel_bound` | C |
| `benchmark/trajopt/serve_safe.py:1171–1199` · `:1855–1925` | `--lookahead` · `--lookahead-gate-mm` · `--qp-eps` · `--w-deflection-rate` · `--servo-accel*` · 시작 로그 | CLI |
| `benchmark/trajopt/experiments/ta_offline.py` (새) | 오프라인 재풀이 (held 행 포함, 운반 청크도) | 비교 |
| `tests/trajopt/test_t43ta_path_shape.py` (새, 38) · `tests/ag3s/test_config_roundtrip.py` (+TA 키 왕복) | | |

## 단위 검증

```bash
cd /mnt/dev/work && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa \
  PYTHONPATH=/mnt/dev/work .venv-openpi-live/bin/python -m pytest -q -p no:cacheprovider tests/trajopt tests/ag3s
```

- TA 파일: 38 passed.
- 전체 `tests/trajopt` + `tests/ag3s` (CPU, 최종 코드): **2476 passed, 2 skipped**, 0 failed (559 s).
- **off ⇒ 비트 동일 (실기록)**: HEAD `308ddfd` 의 `git archive` 와 작업 트리로 V8 2 run · 150 청크의 기준 재풀이를 비교했다 (`outputs/impl/T43TA/p/ident.py`, 결과 `raw_aux/ident/`). 청크 · 궤적 바이트 · status · 위반 · cost 가 모두 같다 (0/150). B · C 이력을 context 에 넣고 flag 를 끈 경우도 0/150 이다.

## verifier 가 알아야 할 것

- 새 flag (모두 기본 off 이고 바이트 동일): `--lookahead {off,obstacles}` · `--lookahead-gate-mm MM|always` · `--qp-eps EPS` · `--w-deflection-rate W` · `--servo-accel {off,strict,relaxed}` · `--servo-accel-tolerance-deg` · `--servo-accel-weight`.
- C 의 config 기본 weight 는 1.0 에서 0.01 로 바뀌었다. 커밋된 적 없는 값이라 기록 호환 문제는 없다.
- 재생산이 필요한 산출물: 없음. 옛 기록과의 호환: 깨지지 않음.
- 오프라인 한계:
  - open loop 다. 변형의 출력이 다음 청크의 상태 · continuity · B/C 이력으로 이어지지 않는다 (이력은 V8 실행값을 쓴다).
  - 목적지 라벨은 근사다.
  - 기준 재풀이와 V8 기록의 차이는 중앙값 0.22°, p90 1.0°, 최대 16.6° 다 (온라인은 청크 간 QP warm start). 그래서 모든 비교는 같은 도구의 base / Q4 재풀이 대비로 했다.
- 온라인에서 gate 가 닫힌 청크는 "8 step 풀이와 같다" 고 했지만, 다른 QP 객체의 warm start 상태 때문에 OSQP 허용오차 수준에서만 같다. 오프라인 (청크마다 reset) 과 단위 테스트에서는 바이트 동일이다.

## 열린 위험

1. **B 는 파지 청크를 움직인다** (사과 앞 27/72 가 0.5° 넘게 움직였고 outward 는 최대 6 mm). 온라인 A+B 에서 G (파지 성공) 를 V8 과 짝으로 본다.
2. 기본 TO 의 eps 1e-3 도 위반 청크에서 필요 이상으로 민다 (§3 표: 0.99° 대 0.56°). 사용자 관찰 "V9 TO 과하게 작용" 과 관련이 있을 수 있으나 측정하지 않았다. `--qp-eps 1e-4` 를 기준선에 쓸지는 사용자가 판정할 일이다 (Q4 는 허들 지표가 base 와 같다).
3. 오프라인에서 A 단독은 이득이 거의 없다. A 의 가치는 온라인 closed loop 에서만 판정된다.
4. gate 20 mm 는 전체 청크의 52% 에서 열린다. 손바닥이 늘 5 mm 근처에 두는, 정체를 모르는 표면 (사과 옆 85–160 mm 높이, 창도 이미 피하고 있음) 이 많다. 그래서 CPU 이득이 작다.
5. 미리 있던 버그 (TA 와 무관, 고치지 않음): `scene_from_constraint_set` 에서 `obstacle_margin > 0` 이면 `obstacle = dict(...)` 가 앞서 넣은 `target_volume` 을 덮어쓴다 (`linearize.py` 의 T43 T step 2 블록 바로 아래). V8 flag 에는 `--target-volume-exempt` 가 없어 영향이 없다.

## 내가 기대하는 결과

(verifier 는 측정 전에 읽지 않는다.)
- 온라인 A+B+C 는 V8 대비 운반 사과–막대 · 복귀 손가락–막대 접촉이 줄고, 파지 G 는 같은 수준을 유지할 것이다.
- A 단독은 V8 과 거의 같을 것이다.
