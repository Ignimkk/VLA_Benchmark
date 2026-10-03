# SUBTASK-f — 놓은 뒤 "바구니만 보임 (`no_admissible`)" 도 인증 (해제 뒤 남은 HOLD 고리 제거)

> writer: lead (A0, work-dd), 2026-10-03. 사용자 승인: "진행해" (2026-10-03).
> 근거: [`SUBTASK-e.verify.json`](SUBTASK-e.verify.json) — 수정 후에도 해제 뒤 HOLD **232/984 chunk (10 run)**, 전부 `uncertified` · grounding `no_admissible`
> (보이는 cluster 가 crate 뿐 — `not_graspable`, destination overlap 1.0). HOLD 중엔 팔이 멈춰 화면이 같으므로 같은 판정이 되풀이되어 4 run 은 끝까지 (35–41 chunk) 못 풀렸다.
> gate off 에도 PLACED 뒤 같은 종류 86.

## 0. 무엇을

SUBTASK-e 의 인증 예외 (`ag3s.types.subtask_no_target`) 를 **한 칸만** 넓힌다:

> gate on ∧ 확정 subtask label = **home** ∧ grasp latch 가 이번 episode 에 **PLACED** 도달 (`placed_seen`) 이면,
> grounding 이 `no_admissible` 이어도 `subtask_gated` 와 같이 인증된 기하로 본다.
> 조건은 그대로: `status == no_target` · `validity == valid` · carve 중인 것 없음 (SUBTASK-e 의 다섯 필드 + exclusion inactive).

놓은 뒤 home 이면 잡을 게 없는 것이 정상이다. 그 밖 (놓기 전 · gate off · label 이 home 아님 · `no_seed` · `lost` · `low_score` 등) 은 **지금 그대로 HOLD**.

## 1. 구현 (implementer)

- 판정은 **SUBTASK-e 의 술어 하나** 를 넓히는 것으로 한다 — HOLD 를 만드는 곳 (SafePolicy `_scene_fn` → sqp, `_verdict`/`_reasons`, bringup) 이 이미 그 술어를 쓰므로 거기만 고친다.
- 술어가 label · placed_seen · gate 를 읽을 수 있어야 한다: `metrics["subtask"]` (`label`, `placed_seen`, `gate`) 에 이미 있는지 확인하고, 없으면 pipeline 이 싣게 한다.
- **grounding 상태는 바꾸지 않는다** — 기록에는 `no_admissible` 이 그대로 남는다. reason 은 `subtask_no_target` 그대로 쓰되 evidence 에 `grounding_status` 를 남겨 둘이 구분되게.
- 건드리지 않는 것: admissibility · destination · latch · gate/B3 로직, 위 조건 밖의 인증 규칙.
- 테스트 (`tests/`): (i) home ∧ placed_seen ∧ gate on ∧ no_admissible ∧ valid ∧ carve 없음 → 인증 · 실행; (ii) placed_seen 없음 → HOLD; (iii) label 이 home 아님 (place · pick · None) → HOLD;
  (iv) gate off → HOLD (지금과 같음, 결정 비트 동일); (v) validity degraded/incomplete → HOLD; (vi) 다른 grounding (`no_seed` · `lost` · `low_score` · `no_cluster` · `no_geometry`) → HOLD; (vii) carve 중인 것이 있으면 예외 없음. SUBTASK-e 테스트 전부 그대로 통과.
- 회귀: `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work src/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q` (cwd `/mnt/dev/work`). `.venv-ag3s` 금지. GPU 를 잡지 않는다.
- 병렬: T40 implementer 가 `pi05_TO_hybrid` (`pi05_infer.py` · scenes · `pick_place_obstacles.py`) 를 고치는 중 — **그 파일들은 건드리지 마라.** benchmark main checkout 의 미커밋 `SESSION_STATE.md` 와 남의 untracked 파일도 그대로.
- 커밋하지 않는다. `SUBTASK-f.impl.md`.

## 2. 검증 (verifier, lead 커밋 뒤)

SUBTASK-e 와 같은 방식: 새 snapshot (`/mnt/dev/work-sf`, 커밋 HEAD + rby1_bringup `22d8dae`), baseline, gate off 2 run 재확인, **gate on 48 run**, 포트 8230–8239 (T40 은 8240–8249).
보는 것: 해제 뒤 HOLD (SUBTASK-e 232/984 → ?), 끝까지 멈춘 run (4 → ?), 새로 실행된 chunk 의 clearance · MuJoCo 접촉, 준비 자세 복귀, 성공 · 짝 2×2 (off · SUBTASK-e on 과), SUBTASK-e on 과 처음 갈리는 chunk.
산출: `SUBTASK-f.verify.json`, `figures/subtask-f/` (실제 씬 · 그래프 · 표), raw `outputs/verify/SUBTASK-f/`.
