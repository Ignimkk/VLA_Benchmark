# SUBTASK-f — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-10-03
> task: [`SUBTASK-f.task.md`](SUBTASK-f.task.md) §0–§1 · 근거: [`SUBTASK-e.verify.json`](SUBTASK-e.verify.json) (해제 뒤 HOLD 232/984, 전부 `uncertified` · grounding `no_admissible`)
> 작업 tree: main checkout `/mnt/dev/work/benchmark` (HEAD `04b5b9c`) + 루트 `/mnt/dev/work/tests` (HEAD `ff8de30`). **커밋하지 않았다.**
> 미커밋 `SESSION_STATE.md` · 남의 untracked 파일 · `pi05_TO_hybrid` (T40) 는 건드리지 않았다. GPU 는 쓰지 않았다.

## 무엇을 했나 (평이한 요약 먼저)

SUBTASK-e 의 인증 예외 술어 `ag3s.types.subtask_no_target` 에 갈래를 **하나만** 더했다: grounding 이 `no_admissible` 이어도
`metrics["subtask"]` 가 **gate 가 켜져 있고 (`gate is True`) · 확정 label 이 `home` 이고 · latch 가 이번 episode 에 PLACED 에 도달했다 (`placed_seen is True`)**
고 말하면 `subtask_gated` 와 같이 인증된 기하로 본다. 나머지 조건 (`status == no_target` · `validity == valid` · carve 다섯 필드 None ·
`metrics["manipulated"]` None · exclusion 기록이 있고 `source=none`, inactive) 은 한 글자도 바꾸지 않았다.
HOLD 를 만드는 세 곳 (`SafePolicy._scene_fn` → sqp, `_verdict`/`_reasons`, `bringup.LivePipeline.scene_fn`) 은 이미 이 술어를
(`safe_policy.geometry_certified` 를 거쳐) 읽으므로 거기는 손대지 않았다. grounding 상태는 다시 쓰지 않는다 — 기록에는 `no_admissible` 이 남고,
사유 kind 는 `subtask_no_target` 그대로이며 evidence 의 `grounding_status` (`subtask_gated` | `no_admissible`) 와 detail 문장이 둘을 가른다.

`metrics["subtask"]` 에 `label` · `gate` · `placed_seen` 은 SUBTASK-c 부터 이미 실려 있다 (`ag3s/runtime/pipeline.py:1574-1589`,
`gate=bool(tc.subtask_gate)` · `placed_seen=bool(tc.placed_seen)`) — pipeline 은 고치지 않았다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/types.py:1055-1060` | `_placed_home(metrics)` (신규) — `metrics["subtask"]` 가 dict 이고 `gate is True` ∧ `label == "home"` ∧ `placed_seen is True` | 정확한 값만 통과 (`1` · `"true"` · `None` · 없음 · `"HOME"` → False, fail closed) |
| `benchmark/ag3s/types.py:1094-1099` | `subtask_no_target` 의 grounding 검사: `subtask_gated` **또는** (`no_admissible` ∧ `_placed_home`) | 넓힌 한 칸. 이어지는 validity · carve · manipulated · exclusion 검사는 그대로 (`metrics` 를 읽는 줄이 위로 올라간 것뿐) |
| `benchmark/ag3s/types.py:1063-1088` | docstring (SUBTASK-f 갈래와 "grounding 상태는 다시 쓰지 않는다") | |
| `benchmark/ag3s/types.py:971-973` | `CollisionConstraintSet.subtask_no_target` property docstring | |
| `benchmark/trajopt/safe_policy.py:1393-1405` | `_subtask_no_target_reason`: grounding 이 `no_admissible` 이면 detail = "nothing admissible is in view after the object was placed (placed_seen=…, label=…, gate=…, released=…)"; `subtask_gated` 는 SUBTASK-e 문장 그대로 | evidence 키는 그대로 (`grounding_status` · `subtask_label` · `placed_seen` · `subtask_released` … 이미 있음) |
| `benchmark/trajopt/safe_policy.py:1378-1383` | 같은 함수 docstring | |
| `benchmark/trajopt/safe_policy.py:1496-1505` | `geometry_certified` docstring (본문은 그대로) | |
| `benchmark/trajopt/wire.py:139` | 머리말 사유 표 `subtask_no_target` 행에 SUBTASK-f 문장 | 문서만. `REASON_KINDS` · `GATE_DEFAULT` 는 그대로 — **클라이언트 쪽 갱신 필요 없음** |
| `tests/trajopt/test_subtask_no_admissible_home.py` (신규, 456 줄) | (i)–(vii) 60 개 | 아래 |
| `tests/trajopt/test_subtask_no_target.py:218-222` | SUBTASK-e (iii) 에서 `no_admissible` 하나만 base 를 `released` → `first` (label place, PLACED 없음) 로 | **의도된 변경.** `released` 프레임은 gate on ∧ home ∧ placed_seen 이라 거기서의 `no_admissible` 이 바로 이번에 인증되는 칸이다. 그 칸은 새 파일 (i) 이 실행을, `first` 위의 같은 테스트가 "놓기 전 no_admissible 은 HOLD" 를 지킨다 |
| `tests/trajopt/test_subtask_no_target.py:249-253` | SUBTASK-e (iv) 120 조합: 기대값 `status is OK` 에 `(no_admissible ∧ no_target ∧ valid)` 한 칸 추가 | **의도된 변경**, 같은 이유 (그 테스트의 base 가 `released` 프레임). gate off 쪽 비트 동일은 새 파일 (iv) 가 따로 본다 |

**건드리지 않은 것:** `pipeline.py` · `target_grounding.py` (gate · B3 · latch · admissibility · destination), `_scene_fn` · `_verdict` · `_reasons` 본문 ·
`bringup.py` · `sqp.py` · `client.py`, `_certification_reason` (`occluded_target` · `uncertified_waived`), `GroundingStatus`/`PipelineStatus`.
SUBTASK-e 테스트는 위 두 줄 말고는 그대로이고 전부 통과한다 (155/155).

### 설계 선택 (task 가 정하지 않은 것)

1. **`gate` 를 `metrics["subtask"]["gate"]` 에서 읽는다.** 술어는 duck-typed module 함수라 config 를 모른다. pipeline 이 그 값을 `bool(tc.subtask_gate)` 로 싣는다.
   gate off 에서도 `placed_seen` 은 sticky 하게 `True` 가 되고 label 도 확정되므로 (gate off 프레임으로 확인), gate 검사가 없으면 gate off 에서 결정이 바뀐다 — 그래서 필수.
2. **`released` 를 조건에 넣지 않았다.** task 의 조건은 `placed_seen` 이다. 해제 (B3) 는 `_held is not None` 일 때만 일어나므로, 처음부터 아무것도 adopt 하지 않은 채
   PLACED 에 간 episode (예: 잡을 것을 못 찾다가 PLACED) 도 이 칸에 든다. 그때도 carve 는 없다 (다섯 필드 + manipulated identity None) — field 는 가장 보수적인 것.
3. **T26 invariant_violation 으로 생긴 `no_admissible` 은 예외를 타지 않는다.** pipeline 은 exclusion gate 가 manipulated 를 거절할 때도 grounding 을
   `no_admissible` 로 다시 쓰는데 (`pipeline.py:1208-1212`), 그 프레임은 `metrics["manipulated"]` 가 살아 있으므로 기존 carve 검사에서 걸린다 (테스트 (vii) 로 고정).

## 단위 검증

```bash
# cwd /mnt/dev/work — GPU 를 잡지 않는 env
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  src/openpi/.venv/bin/python -m pytest tests/trajopt/test_subtask_no_admissible_home.py -q
# 관련 묶음
... -m pytest tests/trajopt/test_subtask_no_admissible_home.py tests/trajopt/test_subtask_no_target.py \
    tests/trajopt/test_verdict_reasons.py tests/trajopt/test_subtask_wiring.py tests/ag3s/test_subtask_gate.py \
    tests/trajopt/test_safe_policy.py -q
```

새 파일 **60 passed**. 관련 묶음 **333 passed, 1 skipped** (SUBTASK-e 의 273 + 60; skip = SUBTASK-c A4 #7 GPU opt-in).

입력 프레임은 실제 `AG3S` 파이프라인 (planar 2-link fixture, `subtask_confirm_frames=1`) 이 만든 것이다. `gripper_max_opening=1 mm` 로 모든 cluster 를
inadmissible 하게 해 **진짜 `no_admissible`** (아무것도 adopt 안 됨, carve 없음, exclusion `none`) 을 얻었다 — SUBTASK-e 의 HOLD 프레임과 같은 모양.
스크립트: `placed_home` (pick → PLACED + home), `home_not_placed` (home, PLACED 없음), `pick`, `placed_place` (PLACED + place), 그리고 SUBTASK-e 의 B3 `released`
(정상 opening, 해제된 프레임의 grounding 을 `no_admissible` 로 바꿔 현장의 "해제 뒤 crate 만 보임" 을 흉내). 각각 gate on / off 두 벌.

| # | 테스트 | 수 | 결과 |
|---|---|---|---|
| — | 프레임이 말한 대로인가 (status · grounding · validity · carve 없음 · `metrics["subtask"]` 의 gate/label/placed_seen) | 1 | pass |
| (i) | `placed_home` · `released`→`no_admissible` 둘 다 `subtask_no_target` · `geometry_certified=True` · grounding 은 `NO_ADMISSIBLE` 그대로; `_verdict` → `safe=True` · `ag3s_status=no_target` · 사유 `["subtask_no_target"]` (action execute, evidence `grounding_status=no_admissible` · `subtask_label=home` · `placed_seen=True` · exclusion none/inactive · `subtask_released` 가 경로대로) · 와이어 왕복 후 로컬 `gate_decision`=execute · `summary_json.verdict.reasons` 기록; `subtask_gated` 사유 문장은 SUBTASK-e 그대로; `legacy` 판정 실행; `_scene_fn` 이 넘기는 `certified`=True (`home_not_placed` · `pick` 은 False); 실제 `TrajectoryOptimizer.solve` → safe · "could not certify" 없음, `home_not_placed` 는 `VIOLATED` | 8 | pass |
| (ii) | `placed_seen` 없음: 실제 `home_not_placed` · `False` · `None` · 키 없음 · `"true"` · `1` · `subtask` 기록 없음 · 기록이 dict 아님 → `uncertified` HOLD | 7 | pass |
| (iii) | label 이 home 아님: 실제 `pick` · 실제 PLACED + `place` · `pick` · `place` · `None` · `"HOME"` · `"home "` → HOLD | 7 | pass |
| (iv) | gate off 실제 프레임 넷 → HOLD; gate off `released` 에서 carve 를 다 지워도 `gate=False` 면 HOLD (gate 만 True 로 바꾸면 인증 — 대조); `gate` = `False` · `None` · `1` · `"true"` → HOLD; gate off 프레임 (스크립트 다섯 + grounding 9 종) × {reasons, legacy} × {optimal, violated} 에서 `_verdict().to_dict()` 와 `_last_reasons` 가 **SUBTASK-e 규칙과, 그리고 (도달 가능한 grounding 에서) SUBTASK-e 이전 규칙과 비트 동일**, `_scene_fn` 의 `certified` = `status=="ok"`; 대조: gate on `placed_home` 에서 SUBTASK-e 규칙은 `uncertified` (`grounding=no_admissible, validity=valid`) 로 갈린다; gate × label × placed_seen × grounding × status × validity **2160 조합** (2 × 4 × 2 × 9 × 5 × 3) 에서 새 술어가 SUBTASK-e 규칙과 다른 칸이 **정확히 1 개** (gate on ∧ home ∧ placed ∧ no_admissible ∧ no_target ∧ valid) | 11 | pass |
| (v) | validity degraded (status no_target / degraded) · incomplete (status no_target / geometry_incomplete) → HOLD | 4 | pass |
| (vi) | `placed_home` · `released` 위에서 grounding `no_seed` · `lost` · `low_score` · `no_cluster` · `no_geometry` · `no_attention` → HOLD, evidence 의 grounding 이 그 값 | 12 | pass |
| (vii) | `manipulated` · `attached` · `target_field_exclude` · `manipulated_link_margin` · `target` · `metrics["manipulated"]` · exclusion active · exclusion source manipulated · exclusion 기록 없음 → 예외 아님; invariant_violation 형 `no_admissible` (identity 살아 있음) → HOLD | 10 | pass |

## 회귀

```bash
# cwd /mnt/dev/work
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  src/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q -p no:cacheprovider
```

**2018 passed, 2 skipped, 0 failed** (356 s). SUBTASK-e 의 1958 + 이 STEP 의 새 테스트 60 = 2018 — 그 사이 다른 변화 없음. skip 2 는 SUBTASK-e 와 같다.
기존 테스트 중 바꾼 것은 `test_subtask_no_target.py` 의 두 단언 (위 표). CPU 전용 env 로만 돌렸다. `.venv-ag3s` 미사용.

## verifier 가 알아야 할 것

- 새로 생긴 플래그·기본값 변경: **없음.** 사유 kind 도 새로 생기지 않았다 (`subtask_no_target` 재사용) — `wire.py` 는 문서 한 줄만 바뀌었으므로 SUBTASK-e 와 같은 `wire` 를 가진 클라이언트면 그대로 실행된다.
- 재생산이 필요한 산출물(npz 등): 없음.
- 옛 기록과 호환: 깨지는 것 없음. `ag3s_status` · `target_grounding_status` · `ag3s` 블록은 그대로 `no_target` · `no_admissible` 로 실린다.
- 세는 법: 새로 실행된 청크는 `verdict_reasons[*].kind == "subtask_no_target"` ∧ `evidence.grounding_status == "no_admissible"` 로 SUBTASK-e 의 것 (`subtask_gated`) 과 나눠 셀 수 있다. detail 문장도 다르다 ("nothing admissible is in view after the object was placed").
- 이 칸은 `placed_seen` 만 본다 (`released` 아님). PLACED 에 갔지만 해제 (B3) 가 일어나지 않은 episode 의 home 구간 `no_admissible` 도 인증된다 — 단 carve 가 하나라도 살아 있으면 (manipulated identity 포함) 아니다.
- 해제 뒤라도 grounding 이 `no_seed` · `no_cluster` · `low_score` · `lost` 등이면 지금처럼 `uncertified` HOLD 다 (task §0).
- gate off 결정은 바뀌지 않는다 (단위 테스트 (iv)). gate off 에서도 `placed_seen` · label 은 기록되므로 술어가 막는 것은 `metrics["subtask"]["gate"]` 다.

## 내가 기대하는 결과

<verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다>

- regression-baseline legacy · cuRobo (gate off): 기대값 그대로.
- gate off 2 run: SUBTASK-e V0 와 action 비트 동일.
- gate on: SUBTASK-e 의 해제 뒤 HOLD 232 청크 (전부 `no_admissible`, home 구간) 가 `subtask_no_target` 실행으로 바뀐다. 끝까지 멈춘 4 run 은 풀린다.
  남는 HOLD 는 해제 뒤 grounding 이 `no_seed` · `no_cluster` 등이거나, label 이 아직 home 이 아닌 (place) 구간의 `no_admissible`, 또는 진짜 `collision`.
- SUBTASK-e on 과 처음 갈리는 chunk = 각 run 에서 PLACED 뒤 home 확정 상태의 첫 `no_admissible` 청크. 그 전은 비트 동일.
- carve 없는 field 이므로 새로 실행된 청크의 clearance 최소는 양수 쪽. 준비 자세 복귀는 정책의 몫 (보장하지 않음).
