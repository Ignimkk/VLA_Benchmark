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
| T21 | **구현 완료 · 검증 중** (2026-09-28 17:40, 1232 passed) | [T21.task.md](T21.task.md) | [T21.impl.md](T21.impl.md) | (A2 진행 중) | 창 = 손 swept ∪ manipulated, overflow 3 종 기록, `tier`/`authorized_rows` 신원, guard 0 = 반지름+복셀. **후속 T21b**: (1) guard 끄는 값 없음(ablation 용 off 필요) (2) `serve_safe.announce_self_filter` 가 반경 0 을 '끔' 이라 찍음 (3) live 서버가 `execution_path` 를 안 넘김 — `refiner`/`safe_policy` 배선 필요 (T23 뒤) |
| T22 | **검증 완료 — 후속 T22b 필요** (2026-09-28 17:10) | [T22.task.md](T22.task.md) | [T22.impl.md](T22.impl.md) | [T22.verify.json](T22.verify.json) · `figures/t22/` | 새 상태 `closing`, 7 검사, `reach=0.12` 잠정 · `legacy_gripper_attach=False`. 순서 (a) 채택. lead 판정: `attach(reach=None)` 기본 유지 · slip 무-detach 유지 · 늦은 잠금 fail-closed · 0 카메라 요청은 피드백 제외 — 넷 다 수용 |
| T23 | **구현 완료 · 검증 중** (2026-09-28 18:00, 1292 passed) | [T23.task.md](T23.task.md) | [T23.impl.md](T23.impl.md) | (A2 진행 중) | `verdict_reasons` 8 kind, 모든 위반 행 재분류, `HoldController`(q_hold 1 회·gripper 마지막 명령), `--safe-max-hold-chunks`. 판정 사항: coarse 계층 손가락 행은 `unverified`(HOLD) 유지(T21 규칙) · 서버 스위치 `verdict_policy` 는 CLI 미배선(T26) · collision HOLD 가 접촉 자세를 고정하면 밀기 지속(지침 §8.5, 범위 밖) |
| T24 | **검증 완료 — 판정 대기** (2026-09-28 15:50) | [T24.task.md](T24.task.md) | [T24.impl.md](T24.impl.md) | [T24.verify.json](T24.verify.json) · `figures/t24/` | 5 항목 전부 통과. 충실도 ≤ 5.9e-5 mm · 전부 OFF 면 편차 0 (T14 25/25 · T17 60/60) · 옛 코드가 `best_unaccepted` 를 돌려준 chunk T14 21/25 · T17 54/61 (옛 sqp 재실행으로 독립 확인). **T17 접근 구간 손끝 변위는 `collision` 항이 만든다** (중앙값 51 mm · 최대 195 mm; 예산 ×4 면 최대 56 mm) — 행은 전부 `ee_finger_l1/l2` **coarse 계층**, target-free=True 인데 reference 의 최소 clearance −34.7 mm → fine/target-free 창이 손을 안 덮음 (T21 의 근거) |
| T25 | 대기 | | | | protocol 초안은 T24 와 함께 |

### 마무리 STEP (T26) 에 모을 후속들 — T23 impl 뒤 한 구현자에게

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
