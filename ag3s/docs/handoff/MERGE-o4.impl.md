# MERGE-o4 — 구현 (merge, 커밋 전)

> writer: ag3s-implementer (A1) · 읽는 쪽: lead, verifier · 2026-10-02
> task: [`MERGE-o4.task.md`](MERGE-o4.task.md)

## 무엇을 했나 (평이한 요약 먼저)

`o4-gpu-parallel` (T38 B1–B5, tip `5c72d37`) 를 benchmark main (`5110dbc`, SUBTASK-c 커밋 위) 에 `git merge --no-ff --no-commit` 으로 합쳤다.
task 가 예상한 다섯 파일 가운데 **실제 충돌은 `trajopt/serve_safe.py` 하나** 였고, 나머지 넷 (`config.py` · `pipeline.py` ·
`target_grounding.py` · `constraint_record.py`) 은 서로 다른 줄을 고쳐 git 이 자동으로 합쳤다. 양쪽 기능을 다 살렸다.
**커밋하지 않았다** — `git status`: "All conflicts fixed but you are still merging." (HEAD `5110dbc`, MERGE_HEAD `5c72d37`).
남의 미커밋 (`ag3s/docs/handoff/SESSION_STATE.md` · untracked figure · `T39.verify.partial.json`) 은 건드리지 않았다 (add · stash 안 함).
`/mnt/dev/work-o1` · `/mnt/dev/work-o4` · branch `o1-eval` · `o4-gpu-parallel` 은 읽기만 했다.

## 충돌 · 자동 merge 별 해법

| 파일:줄 (merge 뒤) | main 쪽 | o4 쪽 | 해법 |
|---|---|---|---|
| `trajopt/serve_safe.py:1632-1642` (**충돌**) | T39 rng/xla 기록 (`describe_rng_owner` · `describe_xla_determinism`) | B5 `if recorder is not None: exit_cleanly_on_sigterm()` | 둘 다, **`9fab500` 과 같은 순서**: `exit_cleanly_on_sigterm()` → T39 rng/xla 기록 → `logging.info("serving on port …")`. `diff <(git show 9fab500:trajopt/serve_safe.py \| sed -n '/^def main/,$p') <(sed -n '/^def main/,$p' trajopt/serve_safe.py)` 의 차이는 SUBTASK-c 세 군데 (AttentionPolicy `subtask_probe=` + 시작 로그 · manifest `subtask` · `build_ag3s(subtask_gate=, subtask_probe_path=)`) 뿐 |
| `trajopt/serve_safe.py` 그 밖 (자동) | `--subtask-gate` · `--subtask-probe` · 거절 규칙 · `build_ag3s` 인자 | `exit_cleanly_on_sigterm` 정의 (`:1443`) · `recorder = None` (`:1476`) · `ConstraintRecordWriter(background=…)` | 겹침 없음 |
| `ag3s/config.py` (자동) | `clustering.subtask_gate/_confirm_frames/_probe_path` + validate (`:275-282`, `:369-`) | module 상수 `KDTREE_WORKERS` (`:41`) · `SELF_FILTER_DEVICE` (`:53`) · `RECON_DEVICE` (`:82`) · `SUPPORT_RANSAC_DEVICE` (`:88`) 등 | 겹침 없음 — o4 는 YAML 밖 module 상수, SUBTASK-c 는 `ClusteringConfig` 필드 |
| `ag3s/runtime/pipeline.py` (자동) | `set_subtask` (`:416`) · `set_placed` (`:434`) · `metrics["subtask"]` (`:1476`) · grounding 앞 `note_subtask` · 해제 note | device recon · device ESDF field (`:2080` 부근 등) | 겹침 없음 |
| `ag3s/stages/target_grounding.py` (자동) | subtask gate · B3 (`:116` import, `:871` `_release_if_placed_home` 등) | `KDTREE_WORKERS` import (`:101`) · dbscan/grow_region 의 `workers=` · `itertools.chain` | 겹침 없음 |
| `ag3s/experiments/sources/constraint_record.py` (자동) | summary 키에 `"subtask"` (`:224`) | `_DeferredGrid` (`:47`) · background writer (`:101`) | 겹침 없음 — subtask 키는 `record()` 안 summary 조립 (호출 스레드) 에 있으므로 background 에서도 같은 바이트 |

5 파일 모두 conflict marker 0, `py_compile` 통과. o4 쪽 나머지 파일 (`fields/{curobo_builder,device_field,esdf}.py` · `runtime/multiview.py` ·
`stages/{attention_lifting,device_recon,reconstruction,robot_filter,support_surface}.py`) 은 main 이 손대지 않은 파일이라 그대로 들어왔다.

## 확인

### 1. CPU 회귀

```bash
# cwd /mnt/dev/work
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  src/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q
```

**1803 passed, 2 skipped, 0 failed** (272 s; merge 전 SUBTASK-c Part B 와 같은 수, 시간은 756 → 272 s — T38 B1 `KDTREE_WORKERS=1` 등).
SUBTASK-c 의 `test_subtask_gate` · `test_subtask_wiring` 포함.

**주의 — 이 "CPU" 회귀는 GPU 를 조금 쓴다.** `JAX_PLATFORMS=cpu` 는 JAX 만 막는다. `src/openpi/.venv` 에는 torch 2.7.1+cu126 이 있고
`torch.cuda.is_available()` 가 True 라, merge 뒤 기본 `*_DEVICE="auto"` 경로 (self-filter · recon · RANSAC · ESDF field) 가 이 실행에서 **torch CUDA 로** 돈다.
완전한 CPU 회귀를 원하면 `CUDA_VISIBLE_DEVICES=` 를 더해야 한다 (이번엔 lead 지시 명령 그대로 돌렸다). 실행 전 `nvidia-smi`: 35.9 / 140 GB 사용.

### 2. `tests/o4` (GPU venv)

```bash
# cwd /mnt/dev/work
AG3S_ASSET_ROOT=/mnt/dev/work/src XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  .venv-openpi-live/bin/python -m pytest --import-mode=append tests/o4 -q -rs
```

**157 passed, 0 skipped, 0 failed** (57 s). merge 전 main 에서는 다섯 모듈 모두 가드로 skip 이었다 (T38-B4.impl: 0 passed · 5 skipped) —
이제 GPU 모듈이 main 에서 import 되어 **skip 0**.
실행 전 `nvidia-smi`: 35.9 GB / 140 GB 사용 (T39 S2 서버 둘 각 17.9 GB 보임) → 남은 ≈ 104 GB, 10 GB 기준 위.

### 3. regression-baseline (GPU)

하지 않았다 — task §2.3 대로 verifier 의 몫.

## verifier 가 알아야 할 것

- 새 플래그 · 기본값 변경: merge 로 들어온 T38 의 device 상수 (`*_DEVICE="auto"`, `KDTREE_WORKERS=1`) — main 의 기본 경로가 GPU (torch 가 CUDA 를 보면) 가 된다.
  SUBTASK-c 는 `clustering.subtask_gate=False` 기본이라 결정에 영향 없음.
- 재생산이 필요한 산출물: 없음.
- 옛 기록과 호환: 깨지는 것 없음 (o4 의 background writer 는 `--record-constraints` 에서만, 기록 바이트 같다고 T38 이 검증).

## 내가 기대하는 결과

<verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다>

regression-baseline 두 개가 T38 숫자 (legacy `40798fb0a4d2` · cuRobo `fe73bd7a6ba6`) 와 비트 동일. gate 기본 off 이고 SUBTASK-c 의 기본 경로 변화는
`metrics["subtask"]` 키 추가 · `AttentionPolicy` 의 kv reduction (baseline 은 attention 서버를 쓰지 않으면 무관) 뿐이다.
