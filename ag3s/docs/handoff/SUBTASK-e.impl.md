# SUBTASK-e — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-10-03
> task: [`SUBTASK-e.task.md`](SUBTASK-e.task.md) §0–§2 · 근거: [`SUBTASK-d.verify.json`](SUBTASK-d.verify.json) (해제 뒤 984/984 HOLD `uncertified`)
> 작업 tree: main checkout `/mnt/dev/work/benchmark` (HEAD `6b79846`) + 루트 `/mnt/dev/work/tests`. **커밋하지 않았다.**
> lead 의 미커밋 `SESSION_STATE.md` · `SUBTASK.audit.md` 와 다른 세션 파일은 건드리지 않았다. GPU 는 쓰지 않았다.

## 무엇을 했나 (평이한 요약 먼저)

target 이 없을 때 HOLD 하던 규칙에 예외 하나를 넣었다: **subtask gate 가 일부러 target 을 비웠고 (`grounding_status=subtask_gated`), validity 가 valid 이고,
carve 중인 것이 하나도 없으면** 그 기하는 인증된 것으로 본다. 판정 술어를 하나 (`safe_policy.geometry_certified`) 로 만들어 HOLD 를 만드는 두 곳 —
최적화기에 넘기는 `geometry_certified` (`_scene_fn` → refiner → `sqp.py:497`) 와 판정 사유 (`_verdict` → `_reasons`) — 가 같은 것을 읽게 했고,
in-process 경로 (`bringup.LivePipeline.scene_fn`) 도 같은 술어로 바꿨다. 실행되는 청크에는 새 사유 `subtask_no_target` (처리 = 실행) 이 붙어
"target 이 없는데 왜 실행했나" 가 응답 · 서버 로그 · `summary_json.verdict.reasons` 에 남는다. gate off 에서는 `subtask_gated` 가 생기지 않으므로 술어는 옛 `status == "ok"` 와 같다.

## 인증 판정이 내려지는 곳 (끝까지 따라간 결과)

| # | 곳 | 옛 규칙 | HOLD 로 이어지는 길 | 이번 수정 |
|---|---|---|---|---|
| 1 | `ag3s/types.py` `CollisionConstraintSet.geometry_certified` | `validity == VALID` | (직접은 없음 — `_certification_reason` 의 `occluded_target` 분기만 읽는다) | 그대로. subtask_gated + valid 프레임은 원래 `True` 였다 |
| 2 | `trajopt/safe_policy.py` `_scene_fn` (옛 709) | `status == "ok"` | → `refiner.py:125,157` → `sqp.py:497` `status=VIOLATED` (max_violation 0 인데 violated — SUBTASK-d 의 `trajopt_status violated · 0.0`) → `legacy` 판정 · 로컬 `--safe-gate legacy` 가 HOLD | `geometry_certified(cs)` |
| 3 | `trajopt/safe_policy.py` `_verdict` (옛 1239) | `cs is not None and status == "ok"` | → `_reasons` → `_certification_reason` → `uncertified` (HOLD) · 와이어 `geometry_certified=False` | `geometry_certified(cs)` + 인증된 subtask 프레임에 `subtask_no_target` 사유 |
| 4 | `trajopt/bringup.py` `LivePipeline.scene_fn` (옛 93) | `status == "ok"` | → refiner → sqp (#2 와 같은 길, in-process) | `geometry_certified(cs)` |
| 5 | `trajopt/client.py` · `wire.derive_verdict_reasons` | 서버의 `safe` · `geometry_certified` · `verdict_reasons` 를 읽기만 | 새 kind 를 모르면 HOLD (fail closed) | `wire.GATE_DEFAULT` 에 kind 등록 (실행) — client 코드는 그대로 |

`to_only_policy.py` (`--no-perception`) 는 AG3S 가 없어 해당 없음.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/types.py:1041-1091` | `subtask_no_target(cs)` (module 함수, duck-typed) · `_CARVE_FIELDS` | 조건 셋을 한 곳에: `status==no_target` ∧ `grounding==subtask_gated` ∧ `validity==valid` ∧ carve 없음 = `target` · `manipulated` · `attached` · `target_field_exclude` · `manipulated_link_margin` 모두 None, `metrics["manipulated"]` None, `metrics["exclusion"]` 이 **있고** `source=="none"` ∧ `active==False`. 못 읽는 필드는 False (fail closed) |
| `benchmark/ag3s/types.py:970-976` | `CollisionConstraintSet.subtask_no_target` property (위 함수 위임) | |
| `benchmark/ag3s/types.py:1115` | `__all__` 에 `subtask_no_target` | |
| `benchmark/trajopt/safe_policy.py:1486-1499` | `geometry_certified(cs)` = `status=="ok"` ∨ `subtask_no_target(cs)` | HOLD 를 만드는 두 곳이 쓰는 술어 하나 |
| `benchmark/trajopt/safe_policy.py:710-712` | `_scene_fn` 의 `certified` → `geometry_certified(constraint_set)` | #2 (sqp:497) |
| `benchmark/trajopt/safe_policy.py:1242` | `_verdict` 의 `certified` → `geometry_certified(cs)` | #3 |
| `benchmark/trajopt/safe_policy.py:1306-1308` | `_reasons`: 인증됐고 `subtask_no_target(cs)` 면 사유 `subtask_no_target` 추가 | 실행 청크에도 "왜" 를 남김 |
| `benchmark/trajopt/safe_policy.py:1378-1401` | `_subtask_no_target_reason(cs)` — evidence: `ag3s_status` · `grounding_status` · `validity` · `subtask_label` · `subtask_blocked` · `subtask_released` · `subtask_released_id` · `placed_seen` · `exclusion_source` · `exclusion_active` | 기록에서 gate 가 무엇을 보고 비웠나 |
| `benchmark/trajopt/safe_policy.py:25` | `from benchmark.ag3s.types import subtask_no_target` | |
| `benchmark/trajopt/bringup.py:29,94-95` | `scene_fn` 의 `certified` → `geometry_certified(constraint_set)` | #4 — 같은 조건으로 모두 |
| `benchmark/trajopt/wire.py:343-347` | `REASON_KINDS` 에 `subtask_no_target` (실행 쪽 자리) | `make_reason` 이 등록 안 된 kind 를 거절하므로 |
| `benchmark/trajopt/wire.py:360-362` | `GATE_DEFAULT["subtask_no_target"] = "execute"` | |
| `benchmark/trajopt/wire.py:139,141` | 머리말 사유 표에 행 추가 · `uncertified` 행에 "subtask_no_target 은 미인증이 아니다" | |
| `tests/trajopt/test_subtask_no_target.py` (신규, 391 줄) | (i)–(v) 155 개 | 아래 |
| `tests/trajopt/test_verdict_reasons.py:38-40` | 실행 kind 집합에 `subtask_no_target` 추가 | 의도된 변경 (kind 표를 고정하는 테스트) |

**건드리지 않은 것:** 위 조건 밖의 인증 규칙 (`_certification_reason` 본문 · `occluded_target` · `uncertified_waived`), gate · B3 해제 (`target_grounding.py`), admissibility, latch, `pipeline.py`, `sqp.py` (인자를 받는 쪽은 그대로 — 넘기는 값만 바뀜), `client.py`, `PipelineStatus` (상태는 여전히 `no_target` — 다시 쓰지 않는다).

### 설계 선택 (task 가 정하지 않은 것)

1. **상태를 `ok` 로 바꾸지 않았다.** AG3S 는 여전히 `status=no_target` · `grounding=subtask_gated` 를 내고 (와이어 `ag3s_status` · `ag3s` 블록도 그대로),
   인증은 소비 쪽 술어가 정한다. `status` 를 바꾸면 "target 이 있다" 로 읽는 다른 소비자 (has_target 집계, 기록 비교) 가 같이 바뀐다.
2. **최적화기에도 인증됨을 넘긴다** (`occluded_target` 과 다르다). `occluded_target` 은 sqp 에 미인증을 넘기고 사유 층에서만 실행으로 돌리므로
   `trajopt_status=violated` 가 남고 `legacy` 판정 · 로컬 `--safe-gate legacy` 에서는 HOLD 다. task §2 "HOLD 를 만드는 곳이 둘 이상이면 같은 조건으로 모두" 를 따라
   이번 예외는 sqp 상태 (`optimal`/`feasible`) · 와이어 `geometry_certified=True` · `reasons`/`legacy` 판정이 모두 같은 답을 낸다.
3. **사유 kind 는 실행 쪽에 등록했다** — `budget_only` 처럼 판정을 바꾸지 않는 기록용 사유. 인증된 청크인데도 붙으므로 `verdict_reasons` 가 `[]` 가 아니다.
   서버 로그에는 `budget_only` 가 아닌 사유가 있으면 한 줄 찍히는 기존 규칙 (`safe_policy.py:423`) 대로 매 청크 `verdict=execute — subtask_no_target: …` 가 찍힌다.
4. **exclusion 기록이 없으면 예외를 타지 않는다** (fail closed). 실제 파이프라인은 모든 정상 경로에서 `metrics["exclusion"]` 을 쓴다 (`pipeline.py:1499`).
5. **`metrics["manipulated"]` (identity) 도 None 이어야 한다** — `cs.manipulated` (기하) 가 None 이어도 `lost` 인 identity 가 살아 있으면 예외를 타지 않는다.
   gate 의 `subtask_gated` 는 `_held is None` 에서만 나오므로 (`target_grounding.py:1031-1041`) 실제로는 둘이 같이 None 이다.

## 단위 검증

```bash
# cwd /mnt/dev/work — GPU 를 잡지 않는 env (lead 지시)
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  src/openpi/.venv/bin/python -m pytest tests/trajopt/test_subtask_no_target.py -q
# 관련 파일 묶음
... -m pytest tests/trajopt/test_subtask_no_target.py tests/trajopt/test_verdict_reasons.py \
    tests/trajopt/test_subtask_wiring.py tests/ag3s/test_subtask_gate.py tests/trajopt/test_safe_policy.py -q
```

새 파일 **155 passed** (28 s). 관련 묶음 **273 passed, 1 skipped** (skip = SUBTASK-c A4 #7 GPU opt-in).

입력 프레임은 실제 `AG3S` 파이프라인 (planar 2-link fixture, gate on/off, `subtask_confirm_frames=1`) 이 만든 `CollisionConstraintSet` 이다 —
`first` (pick 전에 place 확정 → first 거절) 과 `released` (pick → adopt, PLACED + home → B3 해제) 두 길. 변형은 `dataclasses.replace`.

| # | 테스트 | 수 | 결과 |
|---|---|---|---|
| (i) | 두 길 모두 `subtask_no_target` · `geometry_certified=True`; `_verdict` → `safe=True` · `geometry_certified=True` · `ag3s_status` 는 `no_target` 그대로 · 사유 `["subtask_no_target"]` (evidence: label place/home, released, exclusion none) · 와이어 왕복 후 로컬 `gate_decision` = execute · `summary_json.verdict.reasons` 에 기록; `legacy` 판정도 실행; `_scene_fn` 이 refiner 에 넘기는 `certified` = True (`no_seed` 로 바꾸면 False); 실제 `TrajectoryOptimizer.solve(geometry_certified=geometry_certified(cs))` → safe · "could not certify" 노트 없음, `no_seed` 는 `VIOLATED` (sqp:497); `bringup.scene_fn` 이 같은 술어 | 8 | pass |
| (ii) | validity degraded (status no_target / degraded) · incomplete → `uncertified` · HOLD | 3 | pass |
| (iii) | `subtask_gated` 대신 `no_geometry` · `no_attention` · `no_seed` · `no_cluster` · `low_score` · `lost` · `no_admissible` → `uncertified` · HOLD, evidence 의 grounding 이 그 값 | 7 | pass |
| (iv) | gate off 파이프라인 세 프레임 (같은 label script) 은 `subtask_gated` 가 없고 술어 = `status=="ok"`; `subtask_gated` 를 뺀 grounding 8 × status 5 × validity 3 = 120 조합에서 술어 = `status is OK`; gate off 프레임 + 다른 no_target 7 개 × {reasons, legacy} × {optimal, violated} 에서 `_verdict().to_dict()` 와 `_last_reasons` 가 **옛 규칙 (술어를 `lambda: False` 로 바꾼 것) 과 동일**; 대조: gated 프레임에서는 옛 규칙이 SUBTASK-d 와 같은 문장 (`grounding=subtask_gated, validity=valid`) 의 `uncertified` 를 낸다 (비교가 공허하지 않음); `_scene_fn` 의 `certified` = 옛 규칙 | 127 | pass |
| (v) | subtask_gated + valid 인데 `manipulated` · `attached` · `target_field_exclude` · `manipulated_link_margin` · `metrics["manipulated"]` 중 하나가 있거나, exclusion `active=True` · `source="manipulated"` · 기록 없음 → 예외 아님 (미인증); occluded manipulated 가 살아 있으면 T20 `occluded_target` 경로 그대로 (`geometry_certified=False`) | 9 | pass |
| — | kind 등록 (`REASON_KINDS` · `GATE_DEFAULT=execute` · `make_reason`) | 1 | pass |

## 회귀

```bash
# cwd /mnt/dev/work
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  src/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q
```

**1958 passed, 2 skipped, 0 failed** (299 s). SUBTASK-c Part B 의 1803 + 이 STEP 의 새 테스트 155 = 1958 — 그 사이 다른 변화 없음.
skip 2 = SUBTASK-c A4 #7 GPU opt-in + 기존 skip 하나 (Part B 와 같음). 기존 테스트 중 바꾼 것은 `test_verdict_reasons.py` 의 kind 집합 하나 (위 표).
CPU 전용 env 로만 돌렸다 (GPU 미사용).

## verifier 가 알아야 할 것

- 새로 생긴 플래그·기본값 변경: **없음.** 새 사유 kind `subtask_no_target` (처리 = 실행), 새 함수 `ag3s.types.subtask_no_target` · `trajopt.safe_policy.geometry_certified`, 새 property `CollisionConstraintSet.subtask_no_target`.
- 재생산이 필요한 산출물(npz 등): 없음.
- 옛 기록과 호환: 깨지는 것 없음. 단 **클라이언트는 서버와 같은 `wire.py` 를 써야 한다** — `subtask_no_target` 을 모르는 옛 `GATE_DEFAULT` 는 그 kind 를 HOLD 한다 (fail closed).
  verifier 설비처럼 같은 snapshot 의 `benchmark` 를 client `PYTHONPATH` 로 쓰면 문제없다. local PC (`~/dev_ws/vla/pi0_TO_ws`) 클라이언트로 돌린다면 그쪽 `benchmark` 도 갱신해야 한다.
- gate on 에서 바뀌는 응답 모양: `subtask_gated` 프레임 (예외 조건 성립) 의 응답이 `geometry_certified: true` · `safe: true` · `trajopt_status` 는 최적화기 그대로 (`optimal`/`feasible`, 더는 강제 `violated` 아님) ·
  `verdict_reasons: [{kind: subtask_no_target, action: execute, ...}]`. `ag3s_status: no_target` 과 `ag3s` 블록은 그대로 실린다.
- 세는 법: 해제 뒤 청크의 사유는 `summary_json.verdict.reasons[*].kind` (서버) 와 클라이언트 `frames.jsonl` 의 `verdict_reasons` 에서 셀 수 있다.
- 예외는 `grounding_status == subtask_gated` 에서만 탄다. B3 해제 뒤라도 그 프레임의 grounding 이 `no_seed` · `no_cluster` 등이면 (attention 이 아무것도 못 잡은 프레임) **지금처럼 uncertified HOLD** 다 (task §1).
- gate on 에서 pick 전에 label 이 place/home 으로 잘못 확정돼 first 가 막힌 프레임도 같은 조건이라 **이제 실행된다** (carve 없는 field). 그런 프레임이 있으면 수는 위 사유로 셀 수 있다.

## 내가 기대하는 결과

<verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다>

- regression-baseline legacy · cuRobo (gate off): 기대값 그대로 (이 수정은 gate off 결정을 바꾸지 않는다 — 단위 테스트 (iv)).
- gate on, 해제 뒤: SUBTASK-d 의 `uncertified` HOLD 984 청크 중 grounding 이 `subtask_gated` 였던 것 (SUBTASK-d 는 전부) 이 `subtask_no_target` 실행으로 바뀐다.
  남는 HOLD 는 해제 뒤 grounding 이 다른 no_target 이거나 진짜 `collision` 인 청크.
- 해제 뒤 로봇은 정책 (home label) 을 따라 움직이기 시작한다 — 준비 자세로 돌아가는지는 정책의 몫이고 이 수정이 보장하지 않는다.
  carve 없는 field 라 놓인 사과 · crate 도 full clearance 장애물이므로 clearance 최소는 양수 쪽일 것.
- 해제 전 구간: 해제 전에 gate 가 first 를 막은 프레임이 없었다면 SUBTASK-d gate on 과 비트 동일. 있었다면 그 프레임부터 갈라진다 (HOLD → 실행).
