# SUBTASK-c — 구현 (Part A)

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-10-02
> task: [`SUBTASK-c.task.md`](SUBTASK-c.task.md) · 이정표: [`SUBTASK-c.progress.md`](SUBTASK-c.progress.md)
> Part A (아래 본문) · **Part B 는 맨 아래 절** (T39 S1 커밋 `e01b7bf` 뒤, 2026-10-02 lead 지시로 시작 — B3 포함).

## 무엇을 했나 (평이한 요약 먼저)

SUBTASK-b 의 kv_L4 probe 를 같은 데이터·같은 설정으로 다시 학습해 numpy asset 으로 굳혔고 (A1), 서버의 attention 경로가
**이미 하는 prefix pass 의 KV cache** 에서 그 feature 를 꺼내 `result["subtask"]` 로 싣게 했다 (A2, 추가 forward 없음).
AG3S 쪽에는 label 을 받는 입구 (`AG3S.set_subtask`) 와 debounce, 그리고 `clustering.subtask_gate` 를 넣었다 (A3):
켜고 확정 label 이 place/home 이면 `TargetConfirm` 이 **새 target 을 채택(first)하지도, challenger 를 세지도(switch) 않는다.**
지금 manipulated 인 물체, admissibility, destination registry, latch freeze, attach/detach, `_exclusion_gate` 는 손대지 않았다.
기본은 off 이고, off 이면 결정은 지금과 같다 (label 은 `metrics["subtask"]` 에 기록만, "켰다면 무엇을 막았나" 와 함께).

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/experiments/tools/export_subtask_probe.py` (신규) | train 재학습 (StandardScaler + LR, C=0.01 = SUBTASK-b 선택) → asset export, parity 미달이면 exit 2 · 아무것도 안 씀 | A1 |
| `benchmark/ag3s/asset/subtask_probe/kv_L4_v1.npz` · `.json` (신규) | mean · scale · coef(3×512) · intercept · classes=[pick,place,home] · name / feature 정의 · C · 학습 파일 200 개 · parity · sha256 | A1 |
| `benchmark/ag3s/stages/subtask_probe.py` (신규) | `SubtaskProbe` (numpy softmax, sklearn 없음) · `kv_feature` (numpy 참조 구현) · `SubtaskDebounce` · `argmax_label` · `default_probe_path` | A2 · A3 |
| `benchmark/ag3s/experiments/sources/pi05_attention.py:93,108-123,169-206` | `AttentionSampler._kv_feature` (별도 jit, `_prefix` 의 kv 를 그대로 받음) · `attention_and_subtask_feature()` · `attention()` 은 공통 `_sample()` 의 [0] | A2 — `_prefix` 자체는 안 바꿈 → attention bit 동일 |
| `benchmark/trajopt/attention_policy.py:16-21,57-72,84-139,140-157` | `AttentionPolicy(subtask_probe="default")` · `infer` 가 `result["subtask"] = {p, argmax, probe}` · probe 로드/적용 실패는 경고만 | A2 |
| `benchmark/ag3s/config.py:207-222,309-321` | `clustering.subtask_gate=False` · `subtask_confirm_frames=None`(= target_confirm_frames) · `subtask_probe_path=None`(= 동봉 asset) + validate | A3 |
| `benchmark/ag3s/types.py:157-160` | `GroundingStatus.SUBTASK_GATED` | gate 가 first 를 막은 프레임의 상태 (gate on 에서만 나옴) |
| `benchmark/ag3s/stages/target_grounding.py:115,140,499-504,559-566,586-598,764-785,830-836,963-985,1038-1045,1532-1536` | `TargetConfirm.note_subtask` · `subtask` (debounce) · gate: held 없음 → `subtask_gated` (first 없음), held 있음 → challenger 프레임이 gap (frozen 과 같은 자리). frozen 이면 gate 없음. `ConfirmDecision.subtask_label/_blocked`. `ground_target` 이 `SUBTASK_GATED` 반환 | A3 |
| `benchmark/ag3s/runtime/pipeline.py:180-182,360,400-416,1141-1143,1212-1216,1442-1443,1541-1553` | `AG3S.set_subtask(p)` (dict 또는 `result["subtask"]` 통째) · grounding 직전에 `note_subtask` 로 넘기고 소비 · `metrics["subtask"]` · gate 가 막으면 note | A3 |
| `tests/ag3s/test_subtask_gate.py` (신규) | A4 #1–#7 (32 개, #7 은 `AG3S_GPU_TESTS=1` opt-in) | A4 |
| `benchmark/ag3s/experiments/tools/scan_placed_switch.py` (신규) | PLACED 뒤 switch 스캔 (읽기 전용) | §5 조사 |
| `tests/conftest.py` | `slow` marker 등록 | #7 의 `@pytest.mark.slow` 경고 제거 |

**건드리지 않은 것:** `trajopt/{wire,safe_policy,serve_safe,client}.py` · `pi05_infer.py` (T39 편집 중) · openpi submodule · docs · figures.

### 설계 선택 (task 가 정하지 않은 것)

1. **debounce 에서 None 도 하나의 class.** 요청에 label 이 없으면 (probe 실패 · Part B 전) 그 요청은 `None` 으로 센다.
   N 번 연속 None 이어야 확정 label 이 None (= gate 없음) 으로 돌아간다. 한 번 빠진 요청으로 확정이 풀리거나 생기지 않는다.
2. **frozen 이면 gate 는 비켜선다.** held 가 없는데 frozen 인 드문 경우에도 first 채택은 지금처럼 일어난다 — "쥐는 동안은 latch 가 정한다" (A4 #5 를 엄밀히 만족).
3. **label 은 grounding 호출마다 소비된다** (`note_hand` 와 같은 규칙). `set_subtask` 를 안 부른 요청은 None 으로 센다.
4. **`metrics["subtask"]`** = `{p, argmax, label, gate, blocked, would_block, streak, frames}`. task 의 다섯 키에 `would_block`
   (gate 를 켰다면 막았을 것, gate off 에서도 기록 — shadow) · `streak` · `frames` 를 더했다.
5. `clustering.subtask_probe_path` 는 config 에만 있다 — AG3S 는 확률만 받는다. probe 를 고르는 쪽은 `AttentionPolicy` 이고 지금은 동봉 asset 을 쓴다
   (serve_safe 에서 경로를 넘기는 것은 Part B).

## 단위 검증

```bash
# A1 (학습 · export · parity)
/mnt/dev/work/.venv-curobo/bin/python benchmark/ag3s/experiments/tools/export_subtask_probe.py
# A4 #1-#6
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work src/openpi/.venv/bin/python -m pytest tests/ag3s/test_subtask_gate.py -q
# A4 #7 (GPU, checkpoint)
AG3S_GPU_TESTS=1 OPENPI_DATA_HOME=/mnt/dev/work/pi05_TO_hybrid/cache/openpi XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  .venv-openpi-live/bin/python -m pytest tests/ag3s/test_subtask_gate.py -q -k gpu -s
# 회귀
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work src/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q
```

### A1 parity

| 항목 | 값 |
|---|---|
| train | 200 episode · 10,401 frame (pick 4000 / place 3200 / home 3201) · C = 0.01 |
| val (numpy, asset 로 적용) | **0.980777** (2,601 frame) |
| val (sklearn, 같은 모델) | 0.980777 |
| SUBTASK-b `verify.json` kv_L4 val normal | 0.9808 → 차 **2.3e-5** (허용 ±0.002) |
| confusion (numpy) | [[983,17,0],[32,768,0],[0,1,800]] — SUBTASK-b 와 동일 |
| max \|p_numpy − p_sklearn\| | 3.2e-8 (sklearn 이 float32 입력으로 계산) |

### A4 #7 — 새 `AttentionSampler` 경로의 kv_L4 vs SUBTASK-b `feat/closed`

frame: `outputs/live_test/20260925_ep1807/run_0000/step_00025.npz` (t_step 200, file index 25), 같은 checkpoint (29999, LoRA).
`AttentionPolicy._attention_and_feature` (= `infer` 가 쓰는 경로) 로 계산.

| state dtype | 상대오차 ‖a−b‖/‖b‖ | max abs | p (pick / place / home) |
|---|---|---|---|
| float64 (기록 그대로) | **2.03e-8** | 5.96e-8 | 0.00066 / 0.99792 / 0.00142 |
| float32 | 2.03e-8 | 5.96e-8 | 같음 |
| SUBTASK-b 저장 feature | — | — | 0.00066 / 0.99792 / 0.00142 |

같은 테스트에서: `infer` 결과에 `actions · attention · subtask` 가 다 있고, attention 은 feature 를 꺼낼 때와 안 꺼낼 때 **bit 동일**.
GPU 사용 전 `nvidia-smi`: 다른 compute app 하나 (1.6 GB, container 밖 pid 라 정체 미확인 · util 0 %), 140 GB 중 여유 충분해 그대로 썼다. 실행 약 90 s.

### 회귀

`tests/trajopt tests/ag3s` (`src/openpi/.venv`, cwd `/mnt/dev/work`): **1772 passed, 2 skipped, 0 failed** (917 s).
skip 2 = 이 STEP 의 GPU 테스트 (opt-in) + 기존 skip 하나. 주의: main checkout 이라 **T39 의 미커밋 변경 (wire · safe_policy · serve_safe · client · to_only_policy ·
`test_t39_policy_seed.py`) 이 함께 깔린 상태**에서 돈 결과다. 새 테스트만: `tests/ag3s/test_subtask_gate.py` 31 passed 1 skipped (8 s).

## 조사 — latch PLACED 뒤 TargetConfirm 이 다른 물체를 새 target 으로 채택하나? (task §5)

**예. 지금 code 에서 PLACED 뒤 switch 가 일어나고, 기록된 run 에서 실제로 흔했다.** gate 의 place/home 은 이것을 막는다.

code 경로: PLACED 에서 `detach` → `_attached = None`, latch phase 가 CLOSING/HELD 가 아니므로 `set_grasp_active(False)`
(`safe_policy.py:649-651`) → `TargetConfirm.freeze(False)` (`pipeline.py:1140`). `_held` (마지막으로 본 사과 centroid) 는
`reset()` 말고는 지워지지 않으므로 **first 는 다시 일어나지 않고**, 대신 **switch** 가 열린다: 사과에서 6 cm 넘게 떨어진 admissible
cluster 가 3 프레임 연속 score ≥ 0.1 이면 새 id 의 manipulated 가 된다 (`target_grounding.py` select 의 challenger 분기).
latch 는 PLACED 에서 다시 잠그지 않는다 (`grasp_latch.py:516` — PLACED 에서 닫힘이 와도 attach 안 함). 그래서 새 manipulated 는
attach 되지 않고, `_exclusion_gate` 를 통과하면 **target-free layer + contact margin 으로 carve** 된다.

기록 (server_constraints `summary_json`, T34 · T37 의 E3a/E3b, 24 episode). 읽기 전용 스캔 도구 `benchmark/ag3s/experiments/tools/scan_placed_switch.py` (docstring 의 명령 그대로 재현됨):

| 항목 | 값 |
|---|---|
| PLACED 에 도달한 episode | 14 / 24 |
| 그중 PLACED 뒤 switch 가 난 episode | **12 / 14** |
| PLACED → 첫 switch | 2 요청 (10 건) · 5 · 16 |
| 새 target 이 crate 안의 **놓인 사과** (crate centroid 와 xy ≤ 41 mm, 크기 38–56 × 65–67 × 66–73 mm) | 8 |
| 새 target 이 탁자 위 **다른 물체** (crate 에서 xy 0.24–0.30 m, 49 × 58–60 × 81–87 mm) | 4 |
| switch 뒤 chunk 의 `exclusion` | 433 중 **406 active (target_free_layer + contact_margin)** · 27 none |

해석: gate 를 켜면 PLACED 뒤 확정 label 은 place (놓는 동안 확정) → home 이므로 이 switch 들이 막힌다. 즉 home 이 실제로 바꾸는 것이 있다 —
놓은 사과 (crate 안) 와 옆 과일이 PLACED 뒤 carve 되는 것. 단 이 표는 **기록을 읽은 것**이지 gate 를 켠 재현이 아니다 —
막혔는지의 측정은 verifier 의 몫 (Part B 뒤 replay 또는 closed-loop).

## verifier 가 알아야 할 것

- 새 플래그·기본값: `clustering.subtask_gate=False` · `subtask_confirm_frames=None` · `subtask_probe_path=None` (모두 새 키, 기존 기본값 변경 없음).
  `AttentionPolicy` 는 **기본으로 probe 를 싣는다** (`subtask_probe="default"`) — serve_safe 가 만드는 attention 서버는 Part A 만으로 `result["subtask"]` 를 낸다
  (SafePolicy 는 Part B 전까지 읽지 않는다). 첫 요청에 작은 jit (kv reduction) 이 하나 더 컴파일된다.
- 새 `GroundingStatus.SUBTASK_GATED = "subtask_gated"` (gate on 에서만), 새 decision mode `subtask_gated`.
- 기록에 새 키: `metrics["subtask"]` (gate off 에서도 있음 — 결정은 같고 키만 늘었다).
- 재생산이 필요한 산출물: 없음 (asset 은 새로 만듦, 재학습 스크립트 동봉).
- 옛 기록과 호환: 깨지는 것 없음 (키 추가만).
- Part B 전에는 SafePolicy 가 `set_subtask` 를 부르지 않으므로 서버에서 label 은 늘 None → gate 를 켜도 아무것도 막지 않는다.

## 내가 기대하는 결과

<verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다>

- gate off: 기존 회귀 수치 · ESDF 와 bit 동일 (TargetConfirm 결정이 label 과 무관 — 단위 테스트로 확인).
- gate on (Part B 뒤): pick 단계는 같고, PLACED 뒤의 switch (위 12/14) 가 사라져 그 뒤 chunk 의 exclusion 이 `none` 이 된다.
  pick 전에 label 이 place/home 으로 잘못 확정되면 first 가 늦어진다 — closed-loop pick recall 1.0 이라 드물 것.


---

# Part B — safe_policy 배선 · serve_safe flag · B3 (home 뒤 해제)

> 시작 조건: T39 S1 이 benchmark `e01b7bf` · 루트 `48314a9` 로 커밋됨 (lead 확인). 시작 시 benchmark HEAD `5f38468` (그 위), T39 미커밋 변경 없음 확인.
> `/mnt/dev/work-o1` · `/mnt/dev/work-o4` · branch `o1-eval` · `o4-gpu-parallel` 은 건드리지 않았다. **GPU 는 쓰지 않았다** (T39 S2 진행 중).

## 무엇을 했나

SafePolicy 가 요청마다 `result["subtask"]` 의 `p` 를 `ag3s.set_subtask(...)` 로, latch 가 **PLACED** 인지를 `ag3s.set_placed(...)` 로
넘긴다 (`set_grasp_active` 바로 옆). serve_safe 에 `--subtask-gate` (기본 off) · `--subtask-probe` 를 달아 config 와 `AttentionPolicy` 로 보낸다.
B3: gate on ∧ 이번 episode 에 PLACED 를 봤음 ∧ 확정 label = home 이면 `TargetConfirm` 이 기존 manipulated 를 **해제**한다 →
manipulated 가 없으므로 exclusion 이 없다 (carve 없음). home 이 이어지는 동안 새 first 도 gate 가 막는다. pick 이 다시 확정되면 first 는 새 id 로 다시 열린다.

## 바뀐 파일 (Part B)

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/trajopt/safe_policy.py:373` | `_pending["subtask"] = result.get("subtask")` | 정책 결과의 label 을 `_scene_fn` 까지 |
| `benchmark/trajopt/safe_policy.py:654-663` | `set_grasp_active` 옆에서 `set_subtask(_subtask_probabilities(...))` · `set_placed(phase is PLACED)` | Part B 배선 · B3 입력 |
| `benchmark/trajopt/safe_policy.py:1456-1471` | `_subtask_probabilities` — 모양이 틀리거나 없으면 None, 예외 없음 | label 이 지각을 죽이지 않게 |
| `benchmark/trajopt/serve_safe.py:636-637,664-665,741-745` | `build_ag3s(subtask_gate=, subtask_probe_path=)` — 기본이면 `clustering` 키를 아예 안 넣음 | 끄면 config 글자 그대로 |
| `benchmark/trajopt/serve_safe.py:1067-1074` | `--subtask-gate` · `--subtask-probe NPZ` | |
| `benchmark/trajopt/serve_safe.py:1148-1156` | `--subtask-gate`/`--subtask-probe` 를 `--no-safe`·`--no-attention`·`--no-perception` 과 같이 주면 거절 | label 이 없는데 켰다고 믿는 실행 방지 |
| `benchmark/trajopt/serve_safe.py:1480-1488` | `AttentionPolicy(..., subtask_probe=args.subtask_probe or "default")` + 시작 로그 (probe · gate on/off) | `clustering.subtask_probe_path` 를 AttentionPolicy 로 |
| `benchmark/trajopt/serve_safe.py:1532-1533` | record manifest `meta["subtask"] = {gate, probe}` (항상) | 기록에 gate |
| `benchmark/trajopt/serve_safe.py:1598-1599` | `build_ag3s` 에 두 값 전달 | |
| `benchmark/ag3s/experiments/sources/constraint_record.py:159-161` | `summary_json` 에 `subtask` (= `metrics["subtask"]`) | 기록에 label · blocked · released |
| `benchmark/ag3s/stages/target_grounding.py:568-575,607-609,786-792,861-893,1018,1141` | B3: `note_placed` (sticky) · `_release_if_placed_home` (select · unseen 의 맨 앞) · `placed_seen` · `subtask_released`; `reset()` 이 지움 | B3 |
| `benchmark/ag3s/runtime/pipeline.py:418-424,1219-1222,1565-1567` | `AG3S.set_placed` · 해제 note · `metrics["subtask"]` 에 `released` · `released_id` · `placed_seen` | B3 |
| `tests/ag3s/test_subtask_gate.py` (B3 절 추가) | (i)–(v) + frozen · place 라벨 · pipeline (exclusion none) | |
| `tests/trajopt/test_subtask_wiring.py` (신규) | 배선 · PLACED 만 · revoke 는 아님 · reset · serve_safe flag/거절/config 모양 · 기록 키 | |

### B3 해제가 일어나는 조건 (전부)

`clustering.subtask_gate` ∧ `placed_seen` (latch 가 이번 episode 에 PLACED — sticky, `reset()` 만 지움) ∧ 확정 label == `home` ∧ **not frozen** ∧ 지금 manipulated 있음.
해제는 grounding 호출의 맨 앞 (`select` · `unseen`) 에서 일어나므로, 그 프레임의 grounding · exclusion 은 이미 manipulated 없음으로 계산된다.
id 는 계속 센다 (다음 first 는 새 id).

## B3 테스트 결과

| # | 테스트 | 결과 |
|---|---|---|
| (i) | home + PLACED → 해제 · 그 뒤 first 도 막힘 (`test_b3_i_*` 2 개) · pipeline: `metrics["subtask"].released=True`, `manipulated=None`, `exclusion.source="none"`, `active=False`, status `subtask_gated` (`test_b3_pipeline_release_metrics_and_no_exclusion`) | pass |
| (ii) | home 이지만 PLACED 아님 — 쥐기 전 · HELD 중 · attach_revoked 뒤 (`test_b3_ii_*[before_grasp/held/revoked]`), PLACED 지만 frozen; SafePolicy 가 `set_placed` 를 PLACED 에서만 True 로 (`test_set_placed_is_true_only_in_placed` 5 phase · `test_a_revoked_attach_is_not_placed`) | pass |
| (iii) | gate off → 해제 안 함 (`test_b3_iii_*`, pipeline 의 gate off 짝) | pass |
| (iv) | 해제 뒤 pick 확정 → first 다시, 새 id (`test_b3_iv_*`) | pass |
| (v) | `reset()` 이 placed 를 지운다 (`test_b3_v_*`, SafePolicy.reset 경유 `test_reset_clears_placed_through_safe_policy`) | pass |

새 테스트 파일만: `tests/ag3s/test_subtask_gate.py` 42 passed 1 skipped · `tests/trajopt/test_subtask_wiring.py` 19 passed.

## 회귀 (Part B 뒤)

```bash
# cwd /mnt/dev/work — lead 지시 env (T39 S2 와 GPU 를 나눠 쓰지 않도록)
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  src/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q
```

**1803 passed, 2 skipped, 0 failed** (756 s). CPU 전용 env 에서 실패한 테스트 (GPU 를 전제한 것) **없음**.
skip 2 = A4 #7 GPU 테스트 (opt-in) + 기존 skip 하나. Part A 때 1772 → +31 = `test_subtask_gate` +11 (B3) · `test_subtask_wiring` +19 · 나머지 +1 은 그 사이 main tree 변화 (T39 커밋 등, 출처 확인 안 함).

주의: 같은 결과를 env 없이 먼저 한 번 돌렸는데 (1803 passed · 2 skipped), 그 실행은 JAX 기본 preallocation 으로 GPU 를 105 GB 잡아
T39 S2 가 서버 병렬 수를 줄였다 (lead 알림). 그 뒤로 모든 실행에 위 env 를 붙였다.

- A4 #7 (GPU) 은 **T39 S2 중이라 생략** — Part A 때의 값 (rel_l2 2.03e-8) 그대로. Part B 는 `AttentionSampler` · `AttentionPolicy` 의 feature 경로를 바꾸지 않았다
  (바뀐 것은 serve_safe 가 `subtask_probe=` 를 넘기는 것뿐).

## 남은 위험

1. **legacy latch (`evidence=False`)** 는 latch 를 `process` 뒤에 돌리므로 PLACED 가 AG3S 에 한 요청 늦게 닿는다. 기본 (evidence) 은 `_scene_fn` 앞에서 돌아 같은 요청에 닿는다.
2. **해제 프레임의 self-filter target guard** (T19) 는 grounding 전에 직전 manipulated 로 정해지므로 해제된 그 한 프레임만 옛 물체 둘레를 지킨다. carve 가 아니라 점을 *남기는* 쪽이라 안전 방향.
3. **placed_seen 은 episode 동안 sticky** — 해제 뒤 pick 으로 다른 물체를 first 로 잡고 다시 home 이 확정되면 그것도 해제된다. 지금 latch 는 PLACED 에서 다시 잠그지 않으므로 (`grasp_latch.py:516`) 한 episode 의 두 번째 파지는 원래 attach 되지 않는다 — 두 번째 물체를 쥐는 과제가 생기면 다시 봐야 한다.
4. **label 지연:** closed-loop 에서 released→place 오분류가 전환 3 sample 안에 있었다 (SUBTASK-b). 그 동안 확정 label 은 place 라 switch 는 막히고 해제만 늦어진다 (보수적). 반대로 놓기 전 home 이 확정돼도 PLACED 전이면 해제하지 않는다 (B3 (ii)).
5. **gate off 의 shadow:** `would_block` 은 기록하지만 "켰다면 해제했을 것" 은 따로 기록하지 않는다 — `label=="home" ∧ placed_seen ∧ manipulated≠None` 으로 읽을 수 있다.
6. **클라이언트 기록 (`frames.jsonl`) 에는 label 이 없다** — 서버 `--record-constraints` 의 `summary_json.subtask` 와 manifest `meta.subtask` 에만 있다 (client · wire 는 이번 범위 밖).
7. probe 를 못 읽으면 서버는 뜨고 label 이 늘 None → gate 를 켜도 아무것도 막지 않는다. 시작 로그가 `subtask label: OFF` 로 크게 말한다.
