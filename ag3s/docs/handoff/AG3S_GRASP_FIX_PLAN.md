# AG3S 파지 실패 수정 계획 (T18–T25)

> writer: lead (A0) · 2026-09-28 · 근거: [`AG3S_GRASP_FAILURE_IMPLEMENTATION_GUIDE.md`](AG3S_GRASP_FAILURE_IMPLEMENTATION_GUIDE.md) (이하 *지침*),
> [`T14.audit.md`](T14.audit.md), `T14a.verify.json`, `T17a.verify.json`, `T15.impl.md`.
> 이 문서는 **계획과 진행 현황**만 담는다. 무엇을 왜 고치는지의 근거는 지침에, 측정은 각 STEP 의 `verify.json` 에 있다.

## 0. 확정된 사실 (지침 §1–2 · T14.audit §0b)

```
손이 사과 50 mm 안 → self-filter 가 사과 관측을 지움 → grounding 이 다른 물체를 target 으로
→ 접촉 허용·fine/target-free 창이 그 물체를 따라감 → 사과가 손가락에게 다시 장애물
→ violated / 미인증 → 실행 게이트 HOLD → 닫힘 명령이 실행되지 않음 (+ 취소된 닫힘으로 attach)
```

| 실행 | TO 편차 (실행 창 최대) | unsafe chunk | 사과 상승 |
|---|---:|---:|---:|
| T16 충돌 OFF closed loop | 0.0002° | 25 / 34 | 0.4 mm |
| T17 shadow (정책 원본 실행) | (미실행) 8.6° | 1 / 75 | 241 mm |

**TO 궤적 변경은 실패의 필수 조건이 아니다.** 그러나 충돌 ON 의 TO 가 무해하다는 뜻도 아니다 (T17a: 접근 구간 손끝 변위 중앙값 19.5 mm) — 그래서 T24 를 따로 둔다.

## 1. 원칙 (지침에서 그대로 가져온다)

1. **사과 관측 보존과 쌍별 접촉 허용은 함께 고친다.** 보존만 고치면 사과가 더 확실한 장애물이 된다 (지침 §3.1).
2. **조작 대상(manipulated) 은 순간 attention target 과 다른 정보다.** 가려짐은 새 물체 확인이 아니다 (§4).
3. **fine/target-free 창은 실행 경로를 덮는다**, attention centroid 하나에 묶지 않는다 (§5).
4. **접촉 허용은 파지 확인 *전*에 필요하고, attach 는 파지 확인 *뒤*에 한다.** 취소된 닫힘 명령은 파지 증거가 아니다 (§6).
5. **HOLD 는 사유별로 나눈다.** 허용 접촉은 정상 실행이고, 예산 도달만으로 feasible 청크를 거부하지 않는다. HOLD 는 `q_hold` 를 한 번 잡고 유지하며 gripper 정책을 분리한다 (§8).
6. **완료 판정은 파지·들림·place·비대상 충돌·HOLD 빈도로 한다.** 단위 테스트 개수나 violation=0 이 아니다 (§10).
7. 숫자 상수(inflation, 반경, 시간 한도)는 **기록으로 정한다**. 임의 상수를 사양으로 박지 않는다.

## 2. STEP 목록

| STEP | 지침 | 무엇 | 주로 건드리는 파일 | 의존 |
|---|---|---|---|---|
| **T18** | §6.3–6.4, §10-1 | **실행 피드백 계약** — 로컬이 직전 청크의 실제 실행 사실(실행 step 수 · 적용 action/gripper · 측정 개도 · HOLD 사유)을 다음 요청에 싣고, 서버가 기록·사용한다. 연속성 참조를 실제 실행 수에 맞춘다. latch 의 gripper 신호를 **계획값**에서 **측정값**으로 | `trajopt/wire.py` `client.py` `safe_policy.py` `refiner.py`, `ag3s/runtime/frame_record.py`, `pi05_TO_hybrid/rby1_bringup/pi05_infer.py` | — |
| **T19** | §3, §10-2 | **self-filter 가 조작 대상을 보존** — link 그룹별 inflation + 확정 target 근방 보호. cloud 와 depth mask 양쪽. 실제 적용값을 기록 | `ag3s/config.py` `stages/robot_filter.py` `runtime/pipeline.py(_robot_mask_for·self-filter 호출)`, `trajopt/serve_safe.py(설정 통과·시작 로그)` | — |
| **T20** | §4, §10-2 | **조작 대상 정체** — 영속 ID · 관측 상태(visible/occluded/lost) · 가려짐에 target 을 넘기지 않음 · 전환은 확인+최소 score · 접촉 허용/제외 기하가 이 ID 를 참조 | `stages/target_grounding.py(TargetConfirm)` `runtime/pipeline.py` `constraints/to_adapter.py` | T19 |
| **T21** | §5, §10-3 | **fine/target-free 창을 실행 경로에** — 손 자세+실행 구간 FK 의 swept 영역으로 창 배치, 제외는 확정 조작 대상만, 후보의 창 이탈 검사, 어느 계층이 답했는지 기록 | `fields/curobo_builder.py` `fields/curobo_field.py` `runtime/pipeline.py(_target_ball·_build_esdf)` | T20 |
| **T22** | §6, §10-4 | **파지 확인 기반 attach** — 닫힘 시도/파지 확인/놓임 확인 분리, 손–대상 근접·측정 개도 변화·실행 피드백으로 확인, 제약 재생성 순서 | `trajopt/grasp_latch.py` `safe_policy.py(_run_latch)` `ag3s/runtime/pipeline.py(attach)` | T18, T20 |
| **T23** | §8, §10-5 | **실행 게이트·HOLD 정책** — 실패 사유 분리(허용 접촉=정상, 예산 도달+검증 통과=실행, 비허용 충돌, 미인증, 통신), `q_hold` 1 회 고정, gripper 정책 분리, 복구 한도 | `trajopt/client.py(should_execute·_hold)` `safe_policy.py(_verdict)` `wire.py` `pi05_infer.py(제어 루프)` | T18, T22 |
| **T24** | §7, §10-6 | **TO 고정 입력 ablation** — 기록된 입력으로 항별 재실행(충돌·평활·연속성·예산), `sqp` 초기 iterate 를 후보에 포함, 2차 차분 이름 정정, 파지 보존 지표 | `trajopt/sqp.py` `problem.py` `config.py`, `trajopt/experiments/` 신설 | — |
| **T25** | §10-7 | **반복 closed-loop 평가** — 같은 초기조건 N ≥ 10 × {VLA-only · shadow · closed-loop}, 지침 §10 시나리오 표 전부 | 실행 protocol · verifier | T18–T24 |

## 3. 병렬 lane

```
lane A (실행/게이트) : T18 ──────────────► T22 ──► T23 ─┐
lane B (지각/정체)   : T19 ──► T20 ──► T21 ─────────────┼──► T25
lane C (최적화)      : T24 ─────────────────────────────┘
```

**동시에 열 수 있는 것은 파일이 겹치지 않는 것뿐이다.** T18·T19·T24 는 서로 다른 파일을 건드리므로 지금 함께 연다. T20 은 T19 와 `runtime/pipeline.py` 를 공유하므로 T19 가 닫힌 뒤 연다. T22·T23 은 T18 의 피드백 계약 위에 선다.

## 4. 각 STEP 의 완료 기준 (지침 §10 시나리오와의 대응)

| STEP | 통과 기준 (verifier 가 잰다) | 시나리오 |
|---|---|---|
| T18 | 서버 기록의 planning 행에 직전 청크의 `executed_steps`·적용 gripper·측정 개도·HOLD 사유가 있고, T16 재현에서 t=176/208 닫힘이 "제안됐으나 미실행" 으로 구별된다. 연속성 참조가 HOLD 뒤 재설정된다 | 닫힘 제안 후 HOLD · 청크 중간 닫힘 · 부분 실행 |
| T19 | T14 t=88·96 자세를 재생했을 때 self-filter 통과 사과 px > 0 (지금 0), `base` 누수 0 px 유지(T1), 권한 없는 link 의 사과 충돌 검사 유지 | 사과 가까이 접근 |
| T20 | 접근 중 사과를 가리거나 attention 을 바구니로 옮겨도 manipulated ID 와 접촉 권한이 사과에 남는다; 새 물체로 넘어가지 않는다 | attention 이동 · 일시 가림 |
| T21 | attention 이 옮겨가도 손의 질의점이 fine 계층 안에서 답을 받는다 (T14 t=88–160 에서 지금 0/8); 사과 제외 질의에서도 table·crate 남음 | reference 창 안·후보 창 밖 |
| T22 | 빈손 닫힘·멀리 있는 target·취소된 닫힘에서 attach 없음; 실제 파지에서만 attach | 빈손 닫힘 · 파지 후 HOLD |
| T23 | T16 조건(충돌 OFF) closed loop 이 shadow 와 같은 결과(파지·들림)를 낸다; HOLD 중 목표·측정값이 기록되고 drift 가 평가된다 | 예산 종료 feasible · HOLD 중 외력 |
| T24 | 고정 입력에서 충돌·평활·연속성 OFF 면 reference 가 그대로 돌아온다; 첫 후보가 초기 궤적보다 나쁠 때 그것을 돌려주지 않는다; 항별 손끝 변위 표 | 첫 QP 후보가 나쁨 |
| T25 | 빈 씬 파지 성공률: closed-loop ≥ shadow − 허용 오차 (N 회); 비대상 충돌 0 | 전부 |

## 5. 진행 현황

| STEP | 상태 | task | impl | verify | 비고 |
|---|---|---|---|---|---|
| T18 | **구현 완료** (2026-09-28 09:45, tests 1130 passed) | [T18.task.md](T18.task.md) | [T18.impl.md](T18.impl.md) | 런타임 검증은 **새 closed-loop 실행** 필요 (서버에서 못 잰다) | 로컬 PC 에 `pi05_infer.py`·`trajopt/wire.py`·`client.py`·`ag3s/runtime/frame_record.py` 동기화 필요 (impl.md 의 diff) |
| T19 | **검증 완료 — 판정 대기** (2026-09-28 15:20) | [T19.task.md](T19.task.md) | [T19.impl.md](T19.impl.md) | [T19.verify.json](T19.verify.json) · `figures/t19/` | 사과를 지우는 것은 **`link_left_arm_5`** (gripper 그룹 아님) → 설정 B(gripper 10 mm) 는 A 와 동일. guard 를 사과 참값에 두면(D) t=96 wrist 1963/2497 · head 241/378 회복, 나머지는 bare sphere 안. 로봇 누수 0 px (base 는 이 자세에서 안 보여 미측정). guard 중심이 crate 로 가면(C) 효과 0 → T20 필요 |
| T20 | **검증 완료 — 값 판정 대기** (2026-09-28 16:20) | [T20.task.md](T20.task.md) | [T20.impl.md](T20.impl.md) | [T20.verify.json](T20.verify.json) · `figures/t20/` | T14 재생: `min_score ≥ 0.05` 면 seq 12–21 사과 id 0 · occluded 10/10, 전환 없음 (0.05/0.1/0.3 동일); 기본 0.0 은 seq 14 crate · 19 pear 로 전환. occluded 최대 age 14 → `lost_frames=10` 이면 seq 22–25 lost (사과는 seq 26 재관측), 20 이면 유지. 합성: 도전자 score ≥ min 이면 3 프레임에 전환(정체는 영구 잠금 아님) |
| T21 | **검증 완료** (2026-09-28 18:40) | [T21.task.md](T21.task.md) | [T21.impl.md](T21.impl.md) | [T21.verify.json](T21.verify.json) · `figures/t21/` — 손 행의 창 안 비율 0.06–0.10 → **1.00**, coarse 행 5472/3776 → **0** (T20 0.1/20 과 함께); T17 접근 손끝 변위 51/195 mm → q_now 창 35.7/169 · **execution_path 창 13.4/90 mm** (stand-in 필드, 계층 일치 0.92–0.97); 사과 ball 은 table·crate 점 0 제거 | 창 = 손 swept ∪ manipulated, overflow 3 종 기록, `tier`/`authorized_rows` 신원, guard 0 = 반지름+복셀. **후속 T21b**: (1) guard 끄는 값 없음(ablation 용 off 필요) (2) `serve_safe.announce_self_filter` 가 반경 0 을 '끔' 이라 찍음 (3) live 서버가 `execution_path` 를 안 넘김 — `refiner`/`safe_policy` 배선 필요 (T23 뒤) |
| T22 | **검증 완료 — 후속 T22b 필요** (2026-09-28 17:10) | [T22.task.md](T22.task.md) | [T22.impl.md](T22.impl.md) | [T22.verify.json](T22.verify.json) · `figures/t22/` | 새 상태 `closing`, 7 검사, `reach=0.12` 잠정 · `legacy_gripper_attach=False`. 순서 (a) 채택. lead 판정: `attach(reach=None)` 기본 유지 · slip 무-detach 유지 · 늦은 잠금 fail-closed · 0 카메라 요청은 피드백 제외 — 넷 다 수용 |
| T23 | **검증 완료** (2026-09-28 19:00) | [T23.task.md](T23.task.md) | [T23.impl.md](T23.impl.md) | [T23.verify.json](T23.verify.json) · `figures/t23/` — HOLD 드리프트 (T16 t=72 자세, 64 step) legacy **105.1 mm** → fixed **6.0 mm**, 팔 0.141 → 0.001 °/step; 닫힘 명령 HOLD 뒤 legacy 열림(1.0) · fixed 닫힘(0.0) 유지; T14 seq 22 → `occluded_target` 실행. **T16 옛 기록 재분류는 0/25 실행** — 기록의 target-free 층이 orange/crate 를 뺀 것이라 손가락 행이 coarse(`unverified`) 또는 target_free(`collision`) — 게이트의 이득은 T19–T21 이 든 새 실행에서만 보인다 | `verdict_reasons` 8 kind, 모든 위반 행 재분류, `HoldController`(q_hold 1 회·gripper 마지막 명령), `--safe-max-hold-chunks`. 판정 사항: coarse 계층 손가락 행은 `unverified`(HOLD) 유지(T21 규칙) · 서버 스위치 `verdict_policy` 는 CLI 미배선(T26) · collision HOLD 가 접촉 자세를 고정하면 밀기 지속(지침 §8.5, 범위 밖) |
| T24 | **검증 완료 — 판정 대기** (2026-09-28 15:50) | [T24.task.md](T24.task.md) | [T24.impl.md](T24.impl.md) | [T24.verify.json](T24.verify.json) · `figures/t24/` | 5 항목 전부 통과. 충실도 ≤ 5.9e-5 mm · 전부 OFF 면 편차 0 (T14 25/25 · T17 60/60) · 옛 코드가 `best_unaccepted` 를 돌려준 chunk T14 21/25 · T17 54/61 (옛 sqp 재실행으로 독립 확인). **T17 접근 구간 손끝 변위는 `collision` 항이 만든다** (중앙값 51 mm · 최대 195 mm; 예산 ×4 면 최대 56 mm) — 행은 전부 `ee_finger_l1/l2` **coarse 계층**, target-free=True 인데 reference 의 최소 clearance −34.7 mm → fine/target-free 창이 손을 안 덮음 (T21 의 근거) |
| T25 | 대기 | | | | protocol 초안은 T24 와 함께 |
| **T26** | **구현 완료** (2026-09-29, tests/ag3s+trajopt 1409 passed · 0 failed) | [T26.task.md](T26.task.md) | [T26.impl.md](T26.impl.md) | 검증은 T28 Phase 3 앞머리 (T14·T17 오프라인 재생) | max_opening **71.46 mm** (손가락 구 안쪽 면 @±0.05; MuJoCo mesh 99.32 mm) — 사과 여유 10.5 mm(참) / 5.5 mm(T14 PCA 최악). `NO_ADMISSIBLE` 상태 · `DestinationRegistry` · `_exclusion_gate` 불변식 · `metrics.exclusion` · inflation 0 · min_score 0.1 · execution_path 배선. 한계: 큰 물체의 평면 조각은 destination 등록 전엔 graspable. serve_safe 요청(T27 소유): exclude-authorized 로그 문구 · 목적지 주입 flag · announce_self_filter 문구 |
| **T27** | **구현 완료** (2026-09-28) | [T27.task.md](T27.task.md) | [T27.impl.md](T27.impl.md) | E1 smoke(port 8209, 3 청크): refined−reference 0.0°, comms HOLD 0 — 측정은 T28 | 새 테스트 64 통과 (tests/trajopt 641 passed · 5 failed = T26 편집 중인 `test_grasp_evidence.py`). 판정 사항: (1) `no_perception` 은 safe=True 이지만 reasons 게이트에서는 hold(fail-closed) — E1/E2 는 `--safe-gate off` 필수 (2) `--no-limits` 는 trust region·첫 스텝 anchor 를 남김 (`config.NO_LIMITS["keep_anchor"]`) — T24 후보 선택이 anchor 초과분 0.020 rad 중 0.015 만 당기는 경우 발견(미수정) (3) `--links` 기본 gripper → 기준선 재현엔 `--links arms`. 로컬 동기화: `pi05_infer.py` + `client.py` + `wire.py` |
| **T28** | **Phase 1 · 2R · 3 완료 — 원인 분석 → 사용자 판정 대기** | [T28.task.md](T28.task.md) | — | [T28.verify.partial.json](T28.verify.partial.json) · `figures/t28/` · 영상 30 개 | 성공: E0 7/12 · E1 6/12 · E2 6/6 · **E3a 0/6 · E3b 0/6**. lead 분석 (2026-09-29): **D1** client 가 `--safe-phase approach` 를 에피소드 내내 고정 → APPROACH 규칙이 권한 손가락에도 사과와 `0.05×0.4 = 20 mm` 요구 (`config.py:528`), `linearize.py:644-667` 이 target-free 층 행에 "사과 행" 판정(\|d−d_object\| ≤ voxel)이 나면 그 20 mm 를 되살림 — E3b t=88: 세 층 모두 +15.8–20.0 mm 인데 TO −2.25 mm; E3b 위반 154 청크 중 101 이 target_free·target, 여유 0 이면 101 전부 해소; E3a 초기 위반 34 중 28 은 AG3S(여유 0)로는 깨끗 — E3a ep1807 은 그 청크(t=80·88)에서 팔 6.3–6.7° 밀려 손–사과 5 mm → 78 mm. **D3** `camera_transform_stale`: head 이미지가 wrist 보다 102–142 ms 앞 (한도 100 ms) — osmesa 순차 렌더 시간, sim 은 정지 → E3b uncertified HOLD 125 청크. **D2** manipulated 가 이웃 과일로 재연결 (1808 t=96 orange · 1800 banana). **D4** 거짓 attach (1808 E3b t=120, 들림 없음). **D5** 기준선 inflation 0 → 14/15 · −29.031 mm (base 누수 복귀; T29 덮개에 base 없음) |
| **T30** | **검증 완료 — 사용자 판정 대기 (G1·G2·(c))** (2026-09-29) | [T30.task.md](T30.task.md) | [T30a](T30a.impl.md) · [T30b](T30b.impl.md) · [T30c](T30c.impl.md) | [T30.verify.json](T30.verify.json) · `figures/t30/` · 영상 15 개 | **E3a 0/6 → 2/6 (distinct) · E3b 0/6 → 2/6** — 성공은 전부 ep1807. F1 반사실: E3b reference 위반 159 → 13 (`target_free`·target 153 → 0). F4: 거짓 attach 0 (sweep 0/24). 기준선 14/15 · 15/15 · −29.031048 (T26/T29 뒤와 동일). **lead 분석 — 남은 실패(1800·1808)의 기전**: 접근 구간에 충돌 위반 0 인데 TO 가 `left_arm_6` 을 청크 끝까지 5–8.7° 틀음 — 정책 청크가 **URDF 기반 TO limit (2.685 rad) 을 넘고 MuJoCo 범위 (2.967) 안** (실패 run 최대 2.82–2.86 rad, 성공 run 1.91–2.13); SQP 는 50 ms 예산에 1 반복뿐 (linearize 67 ms) 이라 merit 이 10 배 나쁜 **거절된 후보**(`best_unaccepted`)를 반환 → 손목 8.7° 어긋난 채 파지 실패 |
| **T31** | **T31-diag 완료 · T31a/T31b 구현 중 · H1–H4 사용자 판정 대기** | [T31.task.md](T31.task.md) | — | [T31.diag.verify.json](T31.diag.verify.json) · `figures/t31/` (단면 4 · 3D · 애니메이션 2) | **place 요동 원인 = crate 아님.** 손·쥔 점이 crate 50 mm 안에 온 청크 0 (최소 67–78 mm). crate 는 구 아님 (`n_candidates` 0 · `n_static_shapes` 0 / 375 청크; 복셀 + destination 라벨). 요동: 관절 속도 부호 반전 E2 1.1–2.7/s vs E3a 14.6–18.3 · E3b 4.1–5.4; jerk 3–6 vs 26–55 rad/s³; refined−reference 거의 매 청크 **8.6° = trust radius 0.15 rad** (SQP 1 반복). 항별 ablation: collision off → 2.2–3.3°, limit/continuity/smooth off → 불변 ⇒ **충돌 행**. 행의 대상: GT 최근접 **쥔 사과** 7/8 · 23/24 · 23/24 · 30/31. 원인: 쥔 사과가 거리장에 남음 (attached 는 반지름 0 점 25/33 개, 그 복셀만 seed 제거; attached 중심이 GT 사과에서 12–46 mm 어긋남; 사과 중심 fine 값 < 0 최대 −24 mm) + **잔상** (실제 표면 30 mm 밖 점유 복셀 1.8k → 13.5k). crate 벽 16 mm 가 d<0 띠 55–58 mm (fine) |
| **T32** | **H4 측정 중 · H1–H3 는 T31b 뒤 착수** (2026-09-29 사용자 승인) | [T32.task.md](T32.task.md) | — | — | H1 쥔 물체 구를 self-filter 구 집합에 (cuRobo attach 방식 — 쥔 물체 픽셀이 TSDF 에 안 들어감) · H2 쥔 물체 = 관측 점에 맞춘 구, 부모 = 손바닥 (반지름 0 점 은퇴) · H3 잔상 (H1 뒤 측정으로 필요 여부) · H4 crate 벽 d<0 띠 55–58 mm 원인 측정 |
| **T33** | **검증 완료 — J1–J3 사용자 판정 대기** (2026-09-29) | (T31·T32 검증) | — | [T33.verify.json](T33.verify.json) · `figures/t33/` · 영상 12 | **place 요동 해소**: 부호 반전/s T30 E3a 18.25 · E3b 4.74 → **1.76 · 1.67** (E2 1.64), jerk 55.1 · 26.7 → **5.26 · 4.54** (E2 5.08), 편차 8.6° → 3.0°. `best_unaccepted` 0. 성공 **E3a 2/6 · E3b 3/6** (1808 첫 성공). 쥔 사과 점 잔류 0 (손 60 mm 안); H3 는 crate 벽 침식 0; fine crate 벽 24.5 mm. **남은 결함**: J1 E3b HOLD 59 청크 전부 `collision` = **쥔 사과 질의 구 ↔ 테이블** — attach 때 구 바닥이 테이블 아래 1.8–10.1 mm (fit + 5 mm), ep1800 r2 는 56 청크 HOLD 로 들지 못하고 교착; J2 거짓 attach (E3b 1808 r1, fit 중심 GT 에서 19 mm); J3 coarse nx 벽 폭 0·최소거리 14 mm (최대 40/75 청크). 기준선 불변 14/15 · −29.031 (esdf_rollout 은 legacy 경로) |
| **T34** | **검증 완료** (2026-09-30) | [T34.task.md](T34.task.md) | [T34.impl.md](T34.impl.md) | [T34.verify.json](T34.verify.json) · `figures/t34/` · 영상 12 | J1 쥔 질의 구를 받침 평면 + 테이블 ESDF 띠 (combined 8.5 mm) 위로 lift — 교착 56/56 → 0/56; J2 `attach_revoked` 라이브 2 건 (E3a 1808 r1 t=136 → 재파지 392 성공; E3b 1800 r1 t=160). **성공 E3a 5/6 · E3b 3/6** (T33 2/6 · 3/6). E3b HOLD 0, held↔table HOLD 0, comms 0. 요동 부호 반전 1.44/s · jerk 4.35–5.37 (E2 1.64 · 5.08). 기준선 불변. 남은 것: apple-top 잔상 복셀 (held 행 −0.7~−1.8 mm) · E3a 1800 r2 place 뒤 uncertified/invariant_violation 27 청크 (원인 미측정) · E3b 1808 r1/r2 · 1800 r1 실패 (HOLD 없음) |
| **T29** | **구현 완료** (2026-09-29, tests 1444 passed · 0 failed) | [T29.task.md](T29.task.md) | [T29.impl.md](T29.impl.md) | 공식 측정은 T28 Phase 3 (0c–0e) | 손가락 관절 = parameter (`param_joint_names` · `set_joint_parameters`), 역함수는 client 규약을 AST 로 읽음 (`RBY1_GRIPPER_OPEN=-0.045`); FK↔MuJoCo 손가락 0.0005 mm (전 51.0). TO 창: step0 측정, k≥1 = max(측정, 명령 누적 최대). **원인 정정: T17 사과 통째 삭제 17 프레임은 `link_left_arm_5` URDF capsule (r75·L250)** → self-filter 모델에서만 mesh-fit 덮개로 교체 (구 218 → 461). 자가 점검 inflation 0: 누수 36,037/54,106 → **1/2 px**, 사과 보존 T17 0.472 → 0.919, 통째 삭제 17 → 0. 한계: AG3S 자체 CasADi 제약은 생성 시 닫힘 상수 (trajopt 경로 미사용) |
| **T38** | **완료 — 브랜치 `o4-gpu-parallel`, main merge 는 사용자 판정** (2026-10-02) | [T38.task.md](T38.task.md) | [B1B5](T38-B1B5.impl.md) · [B2](T38-B2.impl.md) · [B3](T38-B3.impl.md) · [B4](T38-B4.impl.md) | [T38.profile.verify.json](T38.profile.verify.json) · [T38.verify.json](T38.verify.json) · `figures/t38/` | O4 실시간: AG3S+TO 청크당 중앙 2,935 → 314 ms (verifier, GPU 단독), 74/74 청크 ≤ 533 ms. tag 대비 기록 5 개 × 75 청크 14 항목 비트 동일 · 기준선 둘 불변 · CPU fallback 동일. 커밋 `1a35423` · `305dca1` · `044ecc4` · `5c72d37` |
| **T39** | **S1 완료 · S2 준비 중** (2026-10-02 사용자 승인) | [T39.task.md](T39.task.md) | [T39-S1](T39-S1.impl.md) | (S2 `T39.verify.json`) | O1 큰 N: 사과+왼팔 test 24 episode × seed 2 × {E0, E3b} = 96 run 짝 비교. S1 `--policy-seed` · `--warmup-steps` (main `e01b7bf` · pi05_TO_hybrid `22d8dae`), 서버 브랜치 `o1-eval` `9fab500`, XLA autotune 0, 불변 snapshot `/mnt/dev/work-o1` |

### 옛 "T26" 후속 목록 — 2026-09-28 17:40 재편: T21b(3)·config 기본값(0.1 · lost None)은 새 T26 으로, 나머지는 T28 결과 뒤에

| 출처 | 무엇 |
|---|---|
| T21b | guard off 값(ablation) · `announce_self_filter` 로그 정정 · live 에서 `execution_path`(reference 앞 K step FK) 를 `scene_fn` 으로 넘기는 배선 (`refiner`·`safe_policy`) |
| T23b | `serve_safe` 에 `verdict_policy`·`allow_unresolved_contact_tier` CLI 배선 · `--safe-max-hold-chunks` 기본값(측정 뒤) |
| T22b | `settled` 를 명령 누적 → 측정 개도 정지(연속 두 요청 차 < ε) |
| config 기본값 (사용자 판정 대기) | `target_switch_min_score 0.1` · `target_lost_frames 20` · `LatchConfig.reach 0.08` |

### T22 검증 요약과 후속 (T22b)

- 재생: T17 성공 파지 → attach 1 회 (t=120, reach 42.1 mm, 7 검사 전부 참), detach t=232 · T16 취소된 닫힘 → 0 · T14 phantom → 0 (`blocked`·`reach` 실패). 옛 경로가 attach 하던 6 경우를 `AG3S.attach(reach=0.12)` 에 넣으면 전부 거절 (306–624 mm).
- **미결**: MuJoCo 빈손 닫힘 sweep 에서 **10/24 attach** — chunk 경계에서 손가락이 아직 움직이는 중(개도 0.16–0.76)이라 `settled`(명령 3 step 누적)가 참이 됐다. T17 실제 파지는 2 step 에 정착(1.013→0.904→0.712).
  → **T22b**: `settled` 를 "명령 누적" 이 아니라 **측정 개도의 정지**(연속 두 요청의 개도 차 < ε)로 바꾼다.
- reach 실측: T17 파지 순간 `ee_finger_l1` 원점–사과 41–45 mm (여유 76 mm@0.12); T16 86–108 mm · T14 86–133 mm 는 0.12 를 통과하고 `blocked` 가 막았다.
  → 추천 **`reach = 0.08`** (T17 + centroid 오차 17 mm = 62 < 80; T14/T16 의 빈손 닫힘은 reach 로도 거절).

## 6. 되돌아올 지점

- T19 에서 link 별 inflation 만으로 `base` 누수가 돌아오면 → 렌더 기반 exact mask (지침 §3.3 두 번째 항).
- T20 의 "마지막 위치 유지" 가 물체 이동(밀림)에 틀리면 → 재관측 추적을 T21 의 fine 창과 묶는다.
- T23 에서 `q_hold` 고정으로도 drift 가 남으면 → 제어기 정지 기능/감속 궤적 (지침 §8.5).

## 7. 열린 문제 정리 (2026-10-01, T37 뒤 — 큰 N 평가 전에)

| # | 문제 | 근거 (측정) | 영향 | 할 일 | 우선 |
|---|---|---|---|---|---|
| O1 | **표본이 작아 성공률 차이를 판정할 수 없다** (+ 새 서버는 같은 난수 순서 → T37 4 run 이 T34 run 과 접촉 높이까지 동일: 표본이 조건 간 독립이 아님 → run 별 seed 명시, 조건 간 같은 seed 로 짝지은 비교) | 조건당 6 회, 정책 비결정적; T34 E3a 5/6 → T37 3/6 은 같은 설정 계열에서도 흔들림. 새로 띄운 서버는 같은 RNG 순서 → 병렬 실행은 중복 표본 (T30 E3a) | 모든 판단의 바닥 | E0 (VLA) vs E3b (T34 설정) 를 **같은 에피소드 · 한 서버 순차** 로 큰 N (예: 10 에피소드 × 2) | **1 — T39 진행 중** (seed 입구 완료, 24 episode × seed 2 짝 비교) |
| O2 | K1 (파지 중 continuity 끔) 효과 없음 | T37: 성공 E3a 3/6 · E3b 3/6, 손끝 편차는 0 으로 줄었으나 실패 run 접촉 높이 그대로 (+3.8 ~ +12.9 mm), place 부호 반전 1.44 → 2.47/s | 요동 소폭 증가 | 기본값 off (`grasp_continuity_off: false`), 코드는 유지 · 커밋 | **1** (O1 전에) |
| O3 | 남은 실패 = **정책이 사과를 높게 잡음** | 54 run 중 실패 18/21 이 접촉 중점 > +3 mm; 성공 31/33 은 첫 파지 유지; 1800·1808 은 VLA 단독 E0 도 ≤ 1/2 | E3 성공률 상한 | 코드로 고칠 대상 아님 — O1 에서 **같은 에피소드의 E0 대비** 로 판정 | 기록 |
| O4 | **실시간이 안 된다** (sim 에서는 가려짐) | 서버 청크당 ≈ 3.3 s (AG3S 2.9 s: grounding ≈ 1.5 s · constraint 0.9 s · reconstruction 0.5 s) vs 청크 주기 533 ms (6 배) | 실기 배포 불가 | AG3S 단계별 profile → 병목부터 (GPU 이전 · 중복 계산 제거) | **완료 (T38, 브랜치)** — 2,935 → 314 ms, 비트 동일 · merge 판정 대기 |
| O5 | coarse 20 mm 층의 얇은 벽 구멍 | T33: crate nx 벽 coarse 폭 0 · 최소거리 14 mm 인 청크 최대 40/75 | 손이 fine 창 밖일 때 벽을 못 봄 (안전) | coarse truncation/부호 규칙 측정 → 수정 | 2 |
| O6 | 테이블 ESDF 0-거리 띠 ≈ 8.5 mm (coarse 가 지배) | T34 offline: fine 2.5–4 mm · combined 8.5 mm | J1 이 쥔 구를 띠 위로 올림 → 사과 밑면이 질의 구 밖 최대 5.8–12.8 mm (구현자 자체 점검) | 받침면 행 전용 규칙 또는 coarse 띠 축소 (O5 와 함께) | 3 |
| O7 | 사과 윗면 잔상 복셀 | T34: held 행 −0.7 ~ −1.8 mm, GT = 사과 윗면 흔적 | 작은 위반, HOLD 0 | H3 비움 범위 확인 | 3 |
| O8 | identity 경계 사례 | 병합 잔존 40 청크 (T32b) · S3 reach 30 mm 여유 얇음 (실제 바나나 전환 2 건 억제) · J2 늦은 낙하 미검출 (실행 4 청크 창) | 드묾 | O1 큰 N 에서 빈도 측정 후 판단 | 3 |
| O9 | 놓은 뒤 `uncertified`/`invariant_violation` 27 청크 (E3a 1800 r2) | T34 — 원인 미측정 | 판정 off 라 영향 없음, E3b 였으면 HOLD | 원인 측정 | 3 |
| O10 | 로컬 PC 와 코드가 어긋남 | T18–T37 변경이 서버에만; `pi05_infer.py` 는 `client.py`·`wire.py`·`frame_record.py` 와 함께 옮겨야 함 | 로컬 실행 결과가 서버와 다름 | 동기화 목록 + diff 묶음 작성 | 2 |
| O11 | `.venv-ag3s` 가 NFS 에서 멈춤 | `.python/cpython-3.11.16…` 읽기 D 상태; VS Code 탐지 프로세스가 계속 재생성 | 오프라인 도구는 로컬 shim 으로 우회 중 | 사용자: NFS 확인 또는 venv 를 로컬 python 으로 재연결 / VS Code 인터프리터 변경 | 2 |
| O12 | 기록 미완 | T37 코드 미커밋 · 로그는 T34 까지 · 기준선은 gripper 범위·target-free·margin 0·쥔 물체를 덮지 않음 | 재현·추적 | O2 와 함께 커밋, scribe 로 T35–T37 이어쓰기 | 1 |
