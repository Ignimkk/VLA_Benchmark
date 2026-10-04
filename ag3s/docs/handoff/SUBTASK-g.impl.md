# SUBTASK-g — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-10-04
> task: [`SUBTASK-g.task.md`](SUBTASK-g.task.md) §1 · 근거: [`SUBTASK-f.verify.json`](SUBTASK-f.verify.json) · 배경: [`SUBTASK-c.impl.md`](SUBTASK-c.impl.md) (gate · serve_safe flag · 기동 오류 규칙)
> 작업 tree: main checkout `/mnt/dev/work/benchmark` (HEAD `0f77081`) + 루트 `/mnt/dev/work/tests` (HEAD `033ff1a`). **커밋하지 않았다.**
> 미커밋 남의 파일 (`AG3S_REVIEW_PLAN.md` · `AG3S_T0T6_LOG.md` · `SESSION_STATE.md` · `T40.task.md` · `T40FG.verify.partial.json` · `figures/**`) 과 `pi05_TO_hybrid` 는 건드리지 않았다. GPU 는 쓰지 않았다.

## 무엇을 했나 (평이한 요약 먼저)

subtask gate 를 **기본 on** 으로 바꿨다. config `clustering.subtask_gate` 의 기본이 True 이고, `serve_safe` 는 flag 없이 띄우면 gate on,
`--no-subtask-gate` 로 끈다 (`--subtask-gate` 는 호환용으로 남김 — 받으면 on, 둘을 같이 주면 argparse 가 거절).
label 이 AG3S 에 오지 않는 조합 (`--no-safe` · `--no-attention` · `--no-perception`) 에서는 — 기본 on 이면 **오류 대신 gate 를 끄고 기동 로그에 한 줄**,
사용자가 `--subtask-gate` 를 **명시**했으면 SUBTASK-c 처럼 오류. 서버가 `build_ag3s` 에 넘기는 gate 는 늘 bool 이고 config 에 **켜든 끄든 명시적으로** 실린다.

in-process (`bringup.build_live_pipeline`) 와 offline replay (`safe_replay` · `a7_episode_walkthrough`) 는 config 기본을 따르므로 이제 gate on 이다.
그 경로의 정책은 `subtask` 를 싣지 않으므로 label 은 늘 None 이고, **label None 이면 gate on 의 결정이 gate off 와 비트 동일**하다 —
TargetConfirm · pipeline 의 constraint set 전체 · 인증 술어 세 층에서 테스트로 보였다. 달라지는 것은 기록 필드 `metrics["subtask"]["gate"]` (True/False) 하나뿐이다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/config.py:278` | `subtask_gate: bool = True` (was False) | 사용자 판정 2026-10-04 |
| `benchmark/ag3s/config.py:273-277` | 주석: 기본 on 의 근거 · label 없으면 False 와 비트 동일 (`metrics["subtask"]["gate"]` 제외) | |
| `benchmark/trajopt/serve_safe.py:624-635` | `subtask_config_section(subtask_gate, subtask_probe_path)` (신규) — gate 가 bool 이면 켜든 끄든 `clustering.subtask_gate` 로, None 이면 키 없음, probe 경로는 줄 때만 | `build_ag3s` 안의 dict 조각을 함수로 뽑음 — 테스트가 소스 문자열 대신 실제 조각을 검사하도록 |
| `benchmark/trajopt/serve_safe.py:650,678-682,760` | `build_ag3s(subtask_gate: bool \| None = None)` · docstring · `**subtask_config_section(...)` | task: "off 일 때도 config 에 명시적으로". 서버 gate 가 config 기본값에 기대지 않게 (기본이 다시 바뀌어도 서버는 CLI 가 정한 대로). 함수 기본 None = config 기본을 따른다 (기본값을 두 곳에 적지 않는다) |
| `benchmark/trajopt/serve_safe.py:1082-1097` | 상호 배타 group: `--subtask-gate` (store_const True) · `--no-subtask-gate` (store_const False), `default=None` | 세 값 None(안 줌) · True(명시) · False — 명시와 기본을 구별해야 기동 오류 규칙을 가를 수 있다 |
| `benchmark/trajopt/serve_safe.py:1145-1168` | `_label_unavailable(args)` · `resolve_subtask_gate(args) -> str \| None` (신규) | None → label 오는 조합이면 True, 아니면 False + 로그 한 줄 (`"subtask gate: off (--no-attention — subtask label 이 오지 않으므로 기본 on 인 gate 를 끈다. 명시한 --subtask-gate 였다면 오류)"`). bool 은 그대로. 멱등 |
| `benchmark/trajopt/serve_safe.py:1200-1205` | `reject_bad_flag_combinations`: `args.subtask_gate is True` (명시) 일 때만 거절 | 기본 on (None) 은 거절하지 않는다 |
| `benchmark/trajopt/serve_safe.py:1516-1519` | `main`: `logging.basicConfig` 바로 뒤 `gate_note = resolve_subtask_gate(args)` → `logging.info(gate_note)` | 이 뒤 (`AttentionPolicy` 기동 로그 · manifest `subtask.gate` · `build_ag3s(subtask_gate=)`) 는 전부 확정된 bool 만 본다. manifest `meta["subtask"] = {gate, probe}` (`:1606`) 는 그대로 |
| `tests/trajopt/test_subtask_gate_default.py` (신규, 329 줄) | 40 tests — 아래 | task §1 새 테스트 |
| `tests/ag3s/test_subtask_gate.py:157-160` | 아래 "바뀐 기대값" ① | |
| `tests/trajopt/test_subtask_wiring.py:6-10,138-143,161-167,172-187` | 아래 "바뀐 기대값" ②③④ (+ module docstring) | |

**건드리지 않은 것:** `target_grounding.py` · `pipeline.py` · `types.py` (gate · B3 · SUBTASK-f 술어) · `safe_policy.py` · `bringup.py` · `wire.py` · offline 스크립트.
`test_subtask_no_target.py` · `test_subtask_no_admissible_home.py` 는 **고칠 필요가 없었다** — 모든 AG3S 를 `clustering.subtask_gate` 를 명시해 만든다 (`_ag3s(arm, gate)`), 기본값을 가정하지 않는다.

## 바뀐 기대값과 이유

| # | 테스트 | 전 | 후 | 이유 |
|---|---|---|---|---|
| ① | `test_subtask_gate.py::test_gate_off_is_the_config_default_and_records_what_it_would_have_blocked` → `test_gate_off_records_what_it_would_have_blocked_and_on_is_the_config_default` | `ClusteringConfig().subtask_gate is False` | `is True` | 기본값 자체가 바뀌었다 (사용자 판정). 같은 테스트의 gate-off shadow 기록 검사 (`would_block="first"`, `blocked=None`) 는 그대로 — `_confirm()` 이 gate off 를 명시한다 |
| ② | `test_subtask_wiring.py::test_flags_default_off` → `test_flags_default` | parse 결과 `args.subtask_gate is False` | parse 결과 `is None` (= 안 줌), `resolve_subtask_gate` 뒤 `is True` | flag 가 store_true → 세 값 store_const 로 바뀌었다. 기본의 뜻이 off → on |
| ③ | `test_subtask_wiring.py::test_build_ag3s_puts_the_gate_in_the_config_only_when_asked` → `test_build_ag3s_puts_the_gate_in_the_config` | 소스에 `'"subtask_gate": True} if subtask_gate'` · `"if (subtask_gate or subtask_probe_path) else {}"` | 소스에 `"**subtask_config_section(subtask_gate, subtask_probe_path)"`. `main` 쪽 세 문자열 검사는 그대로 | 조각을 함수로 뽑았다. 조각의 모양은 ④ 가 실제 함수로 검사한다 |
| ④ | `test_subtask_wiring.py::test_the_config_shape_build_ag3s_produces` | 테스트 안에 조각을 다시 써서 검사 · `fragment(False, None) == {}` · 기본 config 의 gate `is False` | `serve_safe.subtask_config_section` 을 직접 검사 · `(None, None) == {}` · `(False, None) == {"clustering": {"subtask_gate": False}}` · `(True, None)` 도 명시 · 기본 config 의 gate `is True` | task: off 일 때도 명시적으로 싣는다 (SUBTASK-c 는 off = 당시 기본이라 키를 안 넣었다). 기본 config 가 on |

## 새 테스트 — `tests/trajopt/test_subtask_gate_default.py` (40)

| 묶음 | 테스트 | 무엇을 |
|---|---|---|
| CLI (6) | `test_config_default_is_on` · `test_server_default_is_on_and_written_into_the_config` · `test_no_subtask_gate_turns_it_off_and_writes_false` · `test_subtask_gate_flag_is_kept_for_compatibility` · `test_both_gate_flags_are_refused` · `test_resolve_is_idempotent` | 기본 on / `--no-subtask-gate` off (config 에 False 로 실림) / `--subtask-gate` on / 둘 다 → SystemExit |
| 기동 규칙 (9 + 2) | `test_default_gate_with_no_label_starts_with_the_gate_off_and_one_log_line[--no-safe·--no-attention·--no-perception]` | 기본 on + 그 flag → 오류 없음, gate False, 로그 한 줄 (`"subtask gate: off"` 로 시작, flag 이름 포함, 개행 없음) |
| | `test_explicit_subtask_gate_with_no_label_is_still_an_error[×3]` | 명시 `--subtask-gate` + 그 flag → SystemExit |
| | `test_no_subtask_gate_with_no_label_is_silent[×3]` | `--no-subtask-gate` + 그 flag → off, 로그 없음 |
| | `test_main_resolves_before_anything_reads_the_gate_and_logs_the_line` · `test_build_ag3s_signature_leaves_the_default_to_the_config` | `main` 순서: reject → basicConfig → resolve → (gate 를 읽는 첫 줄) · `logging.info(gate_note)` · manifest `subtask.gate` 유지 / `build_ag3s` 기본 None |
| in-process (1) | `test_in_process_paths_follow_the_config_default` | `build_live_pipeline` 소스에 `subtask_gate` 없음 · 그 config dict 로 만든 AG3S 의 `TargetConfirm.subtask_gate is True` |
| label None = 비트 동일 (18 + 2 + 1) | `test_target_confirm_label_none_gate_on_equals_gate_off[feed × freeze × placed]` | SUBTASK-c 의 SCRIPT (first · keep · hold · switch · occluded · destination) 를 label None (`note_subtask(None)` / 아예 안 부름) × freeze (없음 · 1 · 5) × PLACED (없음 · 0 · 8) 로 — 결정 key + `subtask_label/blocked/would_block/released` · `placed_seen` · manipulated id 가 gate on == off |
| | `test_pipeline_label_none_gate_on_equals_gate_off[none · absent]` | 실제 `AG3S.process` 13 프레임 (sphere 3 → box 4 → grasp active 2 → PLACED + sphere 4; manipulated id 0→1→2 switch 두 번이 실제로 일어남): `CollisionConstraintSet` 전체를 leaf 로 펼쳐 (배열은 dtype·shape·bytes) 비교 — 다른 leaf 는 `metrics["subtask"]["gate"]` **하나뿐**. wall-clock (`profile` · `timestamp`) 만 제외. `geometry_certified` 도 같다 |
| | `test_safe_policy_hands_none_when_the_policy_carries_no_label` | 기본 config 의 SafePolicy + `subtask` 없는 정책 → `_scene_fn` 이 `set_subtask(None)`, label None |
| 대조군 (1) | `test_control_the_comparison_sees_a_label_the_gate_acts_on` | 같은 비교를 확정 `home` label 로 → 차이가 **있어야** 한다 (on 에만 `subtask_gated`). 비교가 무감각하지 않음을 보인다 |

## 단위 검증

```bash
# cwd /mnt/dev/work — GPU 를 잡지 않는 env
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  src/openpi/.venv/bin/python -m pytest tests/trajopt/test_subtask_gate_default.py -q
# → 40 passed (49 s)
... -m pytest tests/ag3s/test_subtask_gate.py tests/trajopt/test_subtask_wiring.py \
    tests/trajopt/test_subtask_no_target.py tests/trajopt/test_subtask_no_admissible_home.py -q
# config 기본만 바꾼 직후: 2 failed (① · ④) 274 passed 1 skipped → 위 기대값 수정 뒤 전부 통과
```

## 회귀

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  src/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q
```

**2058 passed, 2 skipped, 0 failed** (451 s, 단일 프로세스). SUBTASK-f 의 2018 + 이 STEP 의 새 테스트 40 = 2058.
고친 기존 테스트 4 개는 이름만 바뀌고 개수는 같다. skip 2 는 SUBTASK-f 와 같다 (SUBTASK-c A4 #7 GPU opt-in + 기존 하나).
`.venv-ag3s` 는 쓰지 않았다.

## verifier 가 알아야 할 것

- **새로 생긴 플래그·기본값 변경:**
  - `clustering.subtask_gate` 기본 False → **True**.
  - `serve_safe --no-subtask-gate` (신규). `--subtask-gate` 는 남았지만 이제 기본과 같은 뜻 (단, 명시하면 label 없는 조합을 오류로 거절). 둘을 같이 주면 argparse 오류.
  - flag 없이 `--no-safe` / `--no-attention` / `--no-perception` 으로 띄우면 이제 **오류가 아니다** — 기동 로그에 `subtask gate: off (… 기본 on 인 gate 를 끈다 …)` 한 줄, manifest `subtask.gate=false`.
  - 서버 config 에 `clustering.subtask_gate` 가 **늘** 실린다 (on 이면 true, `--no-subtask-gate` 면 false).
- **재생산이 필요한 산출물:** 없음 (probe asset · npz 그대로).
- **옛 기록과 호환이 깨지는가:**
  - 플래그 없이 띄운 서버의 기록: `meta["subtask"]["gate"]` 가 이제 `true` (전에는 `false`). `metrics["subtask"]["gate"]` 도 마찬가지 — in-process / offline 에서도 (config 기본을 따르므로).
  - 옛 launch 명령 중 **`--subtask-gate` 없이 띄운 것은 이제 gate on 으로 뜬다.** T39 E3b (gate off) 와 같은 조건을 재려면 `--no-subtask-gate` 가 필요하다.
  - label 이 None 인 경로 (offline replay · `--no-attention` 서버 · probe 로드 실패) 에서는 결정이 비트 동일하다 — 달라지는 것은 위의 `gate` 기록 필드뿐.
  - **local client 쪽 (`pi05_infer.py`) 갱신 필요 없음** — wire 형식 · `REASON_KINDS` · `GATE_DEFAULT` 는 그대로다.

## 설계 선택 (task 가 정하지 않은 것)

1. **`build_ag3s` 함수 기본은 None (= config 기본 따름), `main` 은 늘 bool 을 넘긴다.** task 의 "off 일 때도 명시적으로" 를 만족하면서,
   함수 기본값을 True 로 적어 config 와 두 곳이 되는 것 (`build_ag3s` docstring 의 SUBTASK-c 규약이 피하려던 것) 을 피했다.
2. **기본 on 이 꺼질 때 로그는 `logging.info` 한 줄.** 오류가 아니라 정상 동작 (label 이 없으면 gate 는 원래 아무것도 안 한다) 이므로 warning 으로 올리지 않았다.
   `AttentionPolicy` 기동 로그 (`subtask label: … · gate ON/off`) 는 그대로 — `--no-attention` 에서는 그 로그가 없으므로 새 한 줄이 유일한 기록이다 (+ manifest).
3. **`--subtask-probe` + label 없는 조합은 SUBTASK-c 그대로 오류.** 명시한 flag 이므로 "명시 → 오류" 규칙과 같다.

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->
- regression-baseline legacy · cuRobo: 기대값 그대로 (label 없음 → gate 무동작). 다만 기록의 `metrics["subtask"]["gate"]` 는 true 일 것.
- 플래그 없이 띄운 서버 2 run = SUBTASK-f gate on 과 비트 동일 (config 가 같다: 그때 `{"subtask_gate": true}` 를 실었고 지금도 싣는다).
- `--no-subtask-gate` 2 run = T39 E3b 와 동작 동일. 단 config dict 에 `"clustering": {"subtask_gate": false}` 가 새로 실린다 (값은 그때의 기본과 같다).
