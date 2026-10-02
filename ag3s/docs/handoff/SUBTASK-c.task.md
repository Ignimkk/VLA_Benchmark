# SUBTASK-c — subtask label 로 attention target 의 carve 를 정한다 (pick 지움 / place 안 지움)

> writer: lead (A0), 2026-10-02. 담당: **implementer (A1)**. 사용자 승인: "좋습니다. 진행하세요" (2026-10-02).
> 근거: [`SUBTASK.audit.md`](SUBTASK.audit.md) §8 (probe 측정) · §9a (사용자 spec) · [`SUBTASK-b.verify.json`](SUBTASK-b.verify.json).
> **병렬 주의:** T39 implementer 가 지금 main checkout 에서 `trajopt/{wire,safe_policy,serve_safe,client}.py` · `pi05_infer.py` 를 고치는 중이다.
> 그래서 **Part A 는 그 파일들을 건드리지 않는다.** Part B 는 `T39.impl.md` 가 나온 뒤에만 한다 (그 전에 끝나면 `SUBTASK-c.progress.md` 에 "Part B 대기" 로 멈춘다).

## 0. spec (사용자 판정 — 이것이 전부다)

- subtask 가 **pick** 이면 attention target 을 ESDF 에서 지운다 (지금 동작).
- subtask 가 **place** 이면 attention target 을 지우지 않는다. **home** 도 place 와 같다.
- 쥔 사과는 지금처럼 latch **HELD → attach** 로 robot tree 에 들어간다. 사과의 carve → attach 전환은 **latch 가 정한다** (label 이 아니다).

## 1. 지금 code 에서 이 spec 이 닿는 곳 (lead 조사)

- attention 1 등이 carve 대상 (manipulated) 이 되는 길은 하나 — `TargetConfirm` 의 **first 채택** 과 **switch** (`ag3s/stages/target_grounding.py:497`, switch 는 `:971` 부근).
- 쥐는 동안 (latch CLOSING · HELD) 은 이미 `set_grasp_active` → `freeze` 로 challenger 를 세지 않는다 (`pipeline.py:387`, `safe_policy.py` 의 `set_grasp(...)`, `target_grounding.py:769`).
- 그래서 spec 의 구현은 **"label 이 place/home 이면 TargetConfirm 이 새 target 을 채택하지도 바꾸지도 않는다"** 하나다.
  지금 manipulated 인 물체 (쥐기 전 latch 가 잠근 사과, 쥔 사과) 는 그대로 — 그 carve · attach · detach 는 지금 규칙.

## 2. Part A — T39 와 겹치지 않는 파일

### A1. probe 를 asset 으로 굳힌다

SUBTASK-b 는 probe 를 학습만 하고 저장하지 않았다. 같은 데이터 · 같은 설정으로 다시 학습해 numpy 로 쓸 수 있게 export 한다.

- 입력: `outputs/verify/SUBTASK-b/feat/train/*.npz` 의 `kv_L4` 와 label. 설정은 `outputs/verify/SUBTASK-b/probe.py` · `probe_results.json` (StandardScaler + multinomial LogisticRegression, kv_L4 에 고른 C).
- 학습은 `/mnt/dev/work/.venv-curobo/bin/python` (sklearn 있음). **`.venv-ag3s` 는 NFS 에서 멈춘다, 쓰지 않는다.**
- export: `benchmark/ag3s/asset/subtask_probe/kv_L4_v1.npz` (mean · scale · coef · intercept · classes = [pick, place, home]) + 같은 이름 `.json`
  (feature 정의, C, 학습 파일 목록, 아래 parity 숫자). export 스크립트는 `benchmark/ag3s/experiments/tools/` 에.
- **parity**: numpy 로 적용한 val (`feat/val`) accuracy 가 `SUBTASK-b.verify.json` 의 kv_L4 val normal **0.981** 과 ±0.002 안이어야 한다. 아니면 멈추고 보고.

### A2. feature — 이미 도는 prefix pass 에서

- `kv_L4` 정의 (SUBTASK-b `extract.py:86-106` 과 **같아야 한다**): prefix KV cache 의 **V**, layer 4, batch 0, head 0 → (S, 256);
  유효 text token 평균 (text mask = prefix mask 의 768 번째 이후) ⊕ image token 768 개 평균 = 512 차원, float32.
- `ag3s/experiments/sources/pi05_attention.py` `AttentionSampler` 의 prefix pass 가 이미 이 KV cache 를 만든다 (`_prefix` 가 `kv` 를 반환). **추가 forward 없이** 여기서 feature 를 꺼내라.
- numpy probe 적용 모듈 (예: `ag3s/stages/subtask_probe.py`): asset 을 읽고 `p = softmax(((x − mean)/scale) @ coef.T + intercept)`. sklearn 의존 없음.
- `trajopt/attention_policy.py` `AttentionPolicy.infer` 가 `result["attention"]` 옆에 `result["subtask"] = {"p": {"pick","place","home"}, "argmax": str, "probe": "kv_L4_v1"}` 를 싣는다.
  실패해도 정책 호출은 죽지 않는다 (attention 과 같은 규칙).

### A3. gate — TargetConfirm

- 입력: 요청마다의 `p` (또는 None = label 없음).
- **debounce**: 확정 label 은 argmax 가 같은 class 로 `clustering.subtask_confirm_frames` (기본 = `target_confirm_frames`, 3) 번 연속일 때만 바뀐다. episode 시작 · `reset()` 뒤 첫 확정 전은 None.
- **gate** (`clustering.subtask_gate`, **기본 False** — 끄면 지금과 비트 동일):
  - 확정 label 이 place 또는 home → first 채택 없음, challenger 를 세지 않음 (frozen 과 같은 gap). 지금 manipulated 는 유지.
  - pick 또는 None → 지금 동작.
  - **건드리지 않는 것**: admissibility · destination registry · latch freeze · attach/detach · `_exclusion_gate`. label 은 carve 를 **늘리는 길이 없다**.
- `ag3s/runtime/pipeline.py`: label 을 받는 입구 (예: `set_subtask(p)`, `set_grasp_active` 와 같은 모양) + `metrics["subtask"]` =
  `{p, argmax, label (확정), gate, blocked: "first" | "switch" | None}`.
- `ag3s/config.py`: `clustering.subtask_gate` · `clustering.subtask_confirm_frames` · `clustering.subtask_probe_path`.

### A4. 테스트 (`tests/ag3s/`, 새 파일)

1. gate off → 기존 동작과 같다 (기존 TargetConfirm 테스트가 그대로 통과 + label 을 줘도 결정이 같다).
2. gate on · 확정 place → first 채택 안 됨, switch 안 됨, 기존 manipulated 유지; home 도 같다.
3. gate on · pick / None → 지금 동작.
4. debounce: 2 번 연속 place 로는 안 바뀌고 3 번째에 바뀐다; 섞이면 다시 센다; `reset()` 이 지운다.
5. frozen (grasp active) 일 때 label 이 무엇이든 결과가 같다.
6. probe numpy 적용 = asset json 의 parity 숫자 (feat/val 일부로 재계산).
7. (GPU 필요, slow 표시) 기록된 closed-loop frame 하나 (`outputs/live_test/20260925_ep1807/run_0000/step_00025.npz`) 에서
   새 `AttentionSampler` 경로의 kv_L4 가 `outputs/verify/SUBTASK-b/feat/closed/` 의 같은 frame feature 와 맞는지 (상대오차 보고). 이건 impl.md 에 숫자로.

회귀: `MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work src/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q` 전부 통과.

## 3. Part B — T39 뒤 (`T39.impl.md` 가 나온 다음)

- `trajopt/safe_policy.py`: 요청마다 `result["subtask"]` 의 `p` 를 `ag3s.set_subtask(...)` 로 넘긴다 (`set_grasp_active` 를 부르는 곳 근처, `_scene_fn`). 없으면 None.
- `trajopt/serve_safe.py`: `--subtask-gate` (기본 off) → `clustering.subtask_gate=True`. 기록 (record · manifest) 에 label 과 gate.
- **B3 — home 이면 놓인 target 을 해제 (사용자 판정 2026-10-02, 안 (b)).**
  조건: `clustering.subtask_gate` on **그리고** 확정 label = home **그리고** grasp latch 가 이번 episode 에서 **PLACED** 에 도달.
  그러면 TargetConfirm 의 기존 manipulated 를 해제한다 → exclusion 없음 (아무것도 carve 하지 않음). 해제 뒤 home 이 계속되면 first 채택도 gate 가 막는다.
  - 조건이 "detach" 가 아니라 **PLACED** 인 이유: T34 `attach_revoked` (거짓 attach 회수) 도 detach 지만 사과는 놓이지 않았다. 사용자의 의도는 "사과가 놓이기 전에 home 이 나와도 target 을 풀지 않는다".
  - PLACED 는 SafePolicy 가 안다 (`self._latch.phase`) → `set_grasp_active` 를 부르는 곳에서 AG3S 로 함께 넘긴다 (예: `set_placed(bool)`). `reset()` 이 지운다.
  - label 이 다시 pick 이 되면 (놓기 실패 뒤 재시도, 다음 물체) first 채택은 지금 규칙대로 다시 열린다.
  - `metrics["subtask"]` 에 `released: bool` (이번 frame 에 해제했나) 과 `placed_seen: bool`.
  - 테스트: (i) home + PLACED → 해제, exclusion none; (ii) home 이지만 PLACED 아님 (쥐기 전 · HELD 중 · attach_revoked 뒤) → 해제 안 함;
    (iii) gate off → 해제 안 함 (지금과 같음); (iv) 해제 뒤 pick 확정 → first 채택 다시 가능; (v) `reset()` 이 placed 를 지운다.
- 회귀 다시.

## 4. 손대지 않을 것

- openpi submodule (`pi05_TO_hybrid/openpi/**`).
- admissibility · destination registry · grasp latch 규칙 · `_exclusion_gate` 의 판정.
- Part A 동안 `trajopt/{wire,safe_policy,serve_safe,client}.py` · `pi05_TO_hybrid/rby1_bringup/pi05_infer.py` (T39 가 편집 중).
- 문서 (`docs/*.md`) · figure — 각각 scribe · verifier 의 것.

## 5. 넘길 것

- `SUBTASK-c.progress.md` (milestone 마다), 끝나면 `SUBTASK-c.impl.md`: 바뀐 파일, A1 parity 숫자, A4 #7 상대오차, 회귀 결과,
  그리고 **조사해서 보고할 것 하나** — 지금 code 에서 latch PLACED (detach) 뒤 TargetConfirm 이 다른 과일을 새 target 으로 채택할 수 있나 (gate 의 home 이 실제로 바꾸는 것이 있는지).
- 커밋하지 않는다 — lead 가 모은다.
