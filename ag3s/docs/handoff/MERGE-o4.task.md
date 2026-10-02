# MERGE-o4 — `o4-gpu-parallel` (T38 GPU 가속) 를 main 에 merge

> writer: lead (A0, SUBTASK session work-dd), 2026-10-02. 사용자 승인: "좋습니다. 그대신 둘 다 안전 장치로 백업은 해놓고 진행합시다".
> T39/O1 lead (work-ef) 와 합의: **merge 는 work-dd 가 한다, 끝날 때까지 work-ef 는 benchmark main 에 커밋하지 않는다.**

## 0. 전제 (lead 가 확인한 뒤 시작)

- SUBTASK-c 가 main 에 커밋돼 있다 (lead). main checkout `/mnt/dev/work/benchmark` 의 미커밋 코드 변경이 없다
  (`ag3s/docs/handoff/SESSION_STATE.md` 의 미커밋 행 · 남의 untracked figure 는 그대로 둔다 — 건드리지 않는다).
- 백업 tag: `backup/20261002-pre-subtask-merge/{main, o4-gpu-parallel, o1-eval}` (benchmark) · `…/root-main` (루트) +
  SUBTASK-c 커밋 직후 `backup/20261002-pre-merge/main` (lead).
- **건드리지 않는 것:** branch `o1-eval` · `/mnt/dev/work-o1` (T39 S2 snapshot) · `/mnt/dev/work-o4` worktree · branch `o4-gpu-parallel` 자체 (읽기만).

## 1. merge (implementer)

```bash
cd /mnt/dev/work/benchmark
git merge --no-ff --no-commit o4-gpu-parallel     # 커밋은 lead 가 한다
```

충돌 예상 5 파일 — **양쪽을 다 살린다** (어느 쪽 기능도 버리지 않는다):

| 파일 | main 쪽 (SUBTASK-c · T39) | o4 쪽 (T38) | 해법 |
|---|---|---|---|
| `trajopt/serve_safe.py` | T39 seed (`describe_rng_owner` · `describe_xla_determinism`) · SUBTASK-c `--subtask-gate` · `--subtask-probe` | B5 `exit_cleanly_on_sigterm()` · device 옵션 | `main()` 끝은 **9fab500 과 같은 순서**: `if recorder is not None: exit_cleanly_on_sigterm()` → T39 rng/xla 기록 → `logging.info("serving on port …")`. 참고 `git show 9fab500:trajopt/serve_safe.py` (work-ef 가 o1-eval 에서 푼 것). SUBTASK-c flag 는 그 위에 더한다 |
| `ag3s/config.py` | `clustering.subtask_*` 3 키 | device 키 (`RECON_DEVICE` · `SELF_FILTER_DEVICE` · `SUPPORT_RANSAC_DEVICE` …) | 둘 다 |
| `ag3s/runtime/pipeline.py` | `set_subtask` · `set_placed` · `metrics["subtask"]` | GPU 경로 (device field · recon) | 둘 다 |
| `ag3s/stages/target_grounding.py` | subtask gate · B3 해제 | 16 줄 변경 | 둘 다 |
| `ag3s/experiments/sources/constraint_record.py` | subtask 기록 키 | B5 기록 경로 밖으로 | 둘 다 |

## 2. 확인

**implementer:**
1. CPU 회귀 — `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work src/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q` (cwd `/mnt/dev/work`). SUBTASK-c 의 새 테스트 (`test_subtask_gate` · `test_subtask_wiring`) 포함.
2. `tests/o4` — GPU venv, `--import-mode=append`, `AG3S_ASSET_ROOT=/mnt/dev/work/src`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`. main 에서는 GPU 모듈이 import 돼야 skip 되지 않는다 — **skip 수를 보고**.
   **GPU 를 쓰기 전에 `nvidia-smi` 로 빈 공간을 본다** — T39 S2 서버 (PREALLOCATE=false, 메모리를 필요할 때 잡는다) 가 돌고 있다. 남은 공간이 10 GB 미만이면 멈추고 보고.

**verifier (implementer 뒤):**
3. skill `regression-baseline` 두 개, GPU venv (merge 뒤 main 의 기본 경로가 GPU 다). 기대값 (work-ef, T38 검증):
   - legacy: **14/15 · 15/15 · −29.031048280806342 · `40798fb0a4d2`**
   - cuRobo: **10/15 · 15/15 · −9.171877401271193 · `fe73bd7a6ba6`**
   gate 기본 off 이므로 이것이 "SUBTASK-c 를 더해도 T38 결과가 비트 동일" 확인이다. 하나씩, `nvidia-smi` 먼저.

## 3. 넘길 것

- implementer: `MERGE-o4.impl.md` — 충돌 파일별로 어떻게 풀었나 (file:line), 1 · 2 결과 (passed/failed/skipped). **커밋하지 않는다** (merge 상태 그대로 둔다 — `git status` 가 "All conflicts fixed but you are still merging").
- verifier: `MERGE-o4.verify.json` — 3 의 숫자.
- lead: 1–3 이 통과하면 merge 커밋 → work-ef 에 "끝났다" 통지. 실패하면 `git merge --abort` 로 merge 전 (SUBTASK-c 커밋) 으로 되돌리고 보고 — 백업 tag 가 있다.
