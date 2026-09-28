# 세션 상태 — 끊겨도 여기서 이어 간다

> writer: lead (A0). **매 turn 끝에 갱신한다.** 새 세션은 이 파일을 먼저 읽고, 아래 "이어 가는 절차" 를 따른다.
> 마지막 갱신: 2026-09-28 18:00 (T23 impl 완료 → 검증; 구현자 전부 종료, 전체 회귀 배경 실행 중, 그 뒤 커밋)

## 1. 지금 무엇을 하고 있나

[`AG3S_GRASP_FIX_PLAN.md`](AG3S_GRASP_FIX_PLAN.md) 의 STEP T18–T25 를 실행 중이다. 근거 문서는
[`AG3S_GRASP_FAILURE_IMPLEMENTATION_GUIDE.md`](AG3S_GRASP_FAILURE_IMPLEMENTATION_GUIDE.md) (지침) 과
[`T14.audit.md`](T14.audit.md) (측정). 계획의 §5 "진행 현황" 표가 STEP 별 상태의 정본이다.

| STEP | 상태 | 담당 | 산출물 (있으면 끝난 것) |
|---|---|---|---|
| T18 실행 피드백 계약 | **구현 완료** (impl.md 최종본, 1130 passed). 런타임 검증은 새 closed-loop 실행 뒤 | — | `T18.impl.md` |
| T19 self-filter 보존 | **검증 완료 — 사용자 판정 대기** (값: guard 채택 여부 · arm_5 inflation) | lead/사용자 | `T19.impl.md` · `T19.verify.json` · `figures/t19/` |
| T24 TO 고정입력 ablation | **검증 완료 — 사용자 판정 대기** (sqp 후보 선택 수정 유지 여부) | lead/사용자 | `T24.impl.md` · `T24.verify.json` · `figures/t24/` |
| T20 조작 대상 정체 | **검증 완료 — 값 판정 대기** (추천 min_score 0.1 · lost_frames 20) | lead/사용자 | `T20.impl.md` · `T20.verify.json` · `figures/t20/` |
| T22 파지 확인 attach | **검증 완료** — 후속 T22b(settled 를 개도 정지로) · reach 0.08 추천, 판정 대기 | lead/사용자 | `T22.impl.md` · `T22.verify.json` · `figures/t22/` |
| T23 실행 게이트·HOLD | **구현 완료 → 검증 중** (A2, `T23.verify.partial.json`) | verifier | `T23.impl.md` → `T23.verify.json` 대기 |
| T21 fine/target-free 창 | **구현 완료 → 검증 중** (A2, `T21.verify.partial.json`) | verifier | `T21.impl.md` → `T21.verify.json` 대기 |
| T25 | 대기 | — | T21·T23 뒤 |

**세 구현자는 파일이 겹치지 않게 나눴다** (각 task.md 의 "손대지 않을 것"). 커밋은 하지 않는다 — lead 가 모은다.

## 2. 이번 세션이 만든 것 (main agent)

| 무엇 | 어디 | 뜻 |
|---|---|---|
| TO 설계 명세 | [`../AG3S_TO_DESIGN.md`](../AG3S_TO_DESIGN.md) + `figures/to-design-*.png` | 로봇 모델·16D 매핑·objective·constraint·SQP. §7 에 고쳐진 결함 둘 (hold chunk 매핑 · 왼손가락 alias) |
| 실패 원인 감사 | [`T14.audit.md`](T14.audit.md) + `figures/t14/*.png` | self-filter → grounding 전환 → HOLD 사슬. §0b: T16(충돌 OFF, 0.00°, HOLD 25/34) vs T17(shadow, +239 mm) |
| 수정 계획 | [`AG3S_GRASP_FIX_PLAN.md`](AG3S_GRASP_FIX_PLAN.md) | T18–T25 |
| task | `T18.task.md` · `T19.task.md` · `T24.task.md` | 구현자에게 준 것 |
| 분석 스크립트 (재현용) | `/tmp/claude-0/.../scratchpad/*.py` — **세션 임시 공간, 사라질 수 있다** | 필요한 것은 figure 의 `.json` sidecar 에 숫자로 남아 있다 |

기록 규약 중 이번에 확인한 것: **server `chunk_k.npz` ↔ local `seq k+1` (t = 8k)**; `20260928_t12/execute_ep1807` 은 T11 로컬 기록의 사본 (T12 는 server 기록만 유효).

## 3. 이어 가는 절차 (새 세션이 할 일)

```bash
cd /mnt/dev/work/benchmark && git status --short          # 미커밋 변경 = 구현자들의 작업 (지우지 마라)
ls -t ag3s/docs/handoff | head                            # *.progress.md / *.impl.md 가 왔는지
```

1. **`T18/T19/T24.impl.md` 가 있으면** 그 STEP 은 끝난 것 → verifier(A2) 에게 task 의 "측정에 요청하는 것" 표를 재게 한다
   → `verify.json` 이 오면 계획 §5 갱신, scribe(A3) 에게 `AG3S_T0T6_LOG.md` 이어쓰기 요청 → lead 가 커밋 (skill `git-commit`).
2. **`*.progress.md` 만 있으면** 구현자가 중간에 끊긴 것 → 그 파일의 "남은 것" 부터 **같은 task.md 로 새 implementer 를 띄운다**
   ("`<STEP>.progress.md` 를 읽고 거기서부터 이어라" 를 프롬프트에 넣는다). `git diff` 로 실제 반영된 변경을 먼저 확인.
3. **둘 다 없으면** 시작 전에 끊긴 것 → task.md 로 새로 띄운다.
4. T19 가 닫히면 `T20.task.md` 를, T18 이 닫히면 `T22.task.md` 를 쓴다 (지침 §4 · §6 이 계약이다).
5. 회귀: `MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work src/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q`
   (이 세션 시작 시 **1026 passed**; T13 뒤 1040).

## 4. 세션을 넘어서도 유효한 판정 (사용자)

- 코드는 main agent 가 직접 고치지 않는다 — implementer 를 통하거나 사용자가 고친다.
- `plan_horizon=execution` (P=K=8) · `--no-collision` 은 진단 스위치 (T15) · 서버는 `esdf` backend 하드코딩.
- HOLD 는 지금 `action = rby1_state()` (현재 자세 재명령) — T23 에서 바꾼다.

## 5. 갱신 이력

| 시각 | 무엇 |
|---|---|
| 2026-09-28 (착수) | T18·T19·T24 를 implementer 3 에게 배경으로 맡김. 구현자에게 `*.progress.md` 를 milestone 마다 쓰라고 지시 |
| 2026-09-28 (+20 min) | **T19 impl 완료** (`T19.impl.md`, tests/ag3s 641 passed). 구현자가 올린 판정 사항 셋 — (a) `runtime/multiview.py` 도 고쳤다(서버 다중 카메라 경로의 cloud 필터가 거기 있음), (b) guard 안 margin 0 에서 로봇 표면이 fp-noise 수준으로 통과(합성 6,355/9,714 px) — 실제 depth 노이즈에서는 손가락 표면이 샐 수 있음, guard 전용 작은 margin 은 설정에 없음, (c) guard 는 `held_centroid` 를 따르므로 grounding 이 crate 로 가면 crate 를 보호 — T20 의 일. **T19 verifier 착수** (설정 A 기본 / B gripper 10 mm / C +guard 60 mm / D guard 를 사과 참값에). T18·T24 에 `*.progress.md` 지시 전송 |
| 2026-09-28 14:36 | **세션 끊김** (09:45 이후). 디스크: T18.impl.md 있음 · T24.progress.md 있음 · T19 verify 산출물 없음. 세 agent 에 SendMessage 로 재개 지시 (T24: progress §3 부터 · T19 verifier: partial.json 에 중간 저장 · T18: 최종본 확인) |
| 2026-09-28 14:45 | T18 impl 최종본 확인 (1130 passed). **T20 착수** (`T20.task.md`). 로컬 PC 동기화 4 파일은 사용자 몫 |
| 2026-09-28 15:00 | **T24 impl 완료** (tests/trajopt 488 passed). 핵심: chained replay 로 서버가 **거절된 QP 후보**를 돌려준 chunk T14 21/25 · T17 54/61 확인 → `sqp.best_merit` 초기 iterate 로(한계 안일 때만). T24 verifier 착수 |
| 2026-09-28 15:20 | **T19 verify 완료.** 핵심: 사과를 지우는 sphere 는 `link_left_arm_5`(inflation 0.05) — gripper 그룹 override 는 무효(B=A). guard 를 사과에 두면(D) 사과 px 80 % 회복, 나머지는 bare sphere 안. 로봇 누수 0. guard 가 이전 프레임 centroid 를 따르면(C) t≥88 에 crate 로 가서 효과 0 → T20 의 manipulated centroid 필요. 기준선: HEAD 와 동일(13/15 · +0.185978), **skill `regression-baseline` 의 기준값(14/15 · +0.157)은 planned_horizon 32 시절 것 — 갱신 필요** |
| 2026-09-28 15:30 | **T20 impl 완료** (1161 passed). T20 verifier 착수 (min_score·lost_frames sweep). **T22 착수** (`T22.task.md`; T21 은 pipeline.py 충돌로 T22 뒤). 사용자 판정 대기: T19 guard 채택 · T20 의 NO_SEED HOLD/lost_frames/min_score 값 |
| 2026-09-28 15:50 | **T24 verify 완료** (5/5 통과). 접근 구간 손끝 변위는 collision 항(coarse 계층의 손가락 행) — T21 근거. 기준선 HEAD 일치, skill 기준값 갱신 필요 (T19 와 같은 지적) |
| 2026-09-28 16:05 | **사용자 판정**: (1) T19 guard 채택 — 반경 = manipulated 반지름 + pad (구현은 T21 에 묶는다; 다음 실행은 `--self-filter-target-guard` 로 켠다) · (3) `target_lost_frames` 유한값 — 값은 T20 sweep 으로 · (4) T24 `sqp` 후보 선택 수정 유지 · (5) T18·T19·T20·T24 를 STEP 별 commit — **T22 가 `safe_policy.py`·`pipeline.py` 를 편집 중이라 T22 impl 이 닫힌 뒤 커밋** (섞임 방지). (2) min_score 는 sweep 대기 |
| 2026-09-28 16:20 | **T20 verify 완료** (tests/ag3s 673 passed). min_score ≥ 0.05 에서 사과 유지; lost_frames 10 은 seq 22–25 에서 잃고 20 은 유지 (max age 14). 추천: 0.1 / 20. skill 기준값 갱신함 (13/15 · +0.185978) |
| 2026-09-28 16:40 | **T22 impl 완료** (1188 passed). T22 verifier 착수, **T21 착수**. **커밋은 T22 verify 가 닫히면** — 공유 파일(safe_policy.py·pipeline.py·config.py)이 STEP 을 가로지르므로 검증된 상태에서 repo 별로 모은다 (benchmark 와 상위 repo 둘 다; skill `git-commit` 먼저) |
| 2026-09-28 17:10 | **T22 verify 완료** (재생 #1–#3·#5 통과, 빈손 MuJoCo sweep 10/24 attach → T22b 필요, reach 0.08 추천). **T23 착수** (T21 과 파일 겹침 없음). **커밋은 T21 impl 이 닫히면** — T21 이 pipeline.py·config.py·linearize.py 를 편집 중이라 지금 커밋하면 반쪽이 섞인다 |
| 2026-09-28 17:40 | **T21 impl 완료** (1232 passed). T21 verifier 착수. 후속(T21b·T22b·config 기본값)은 계획 §5 의 'T26' 표로 모음. **커밋은 T23 impl 뒤** — T23 이 safe_policy/wire/client/pi05_infer 편집 중 |
| 2026-09-28 18:00 | **T23 impl 완료** (1292 passed). T23 verifier 착수. 구현자 전원 종료 → 최종 tree 로 전체 회귀 배경 실행 → 통과하면 **repo 3 곳에 커밋** (benchmark · pi05_TO_hybrid · 루트 tests/.claude; skill `git-commit` 규칙: 출처 표기 없음, 소유 경로만 add) |
