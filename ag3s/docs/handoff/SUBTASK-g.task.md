# SUBTASK-g — subtask gate 를 기본 on 으로

> writer: lead (A0, work-54), 2026-10-04. 사용자 판정: "네 on 을 기본값으로 합니다" (2026-10-04).
> gate = "subtask label 을 보고 attention target 을 지울지 말지 정하는 규칙" (pick → 지움, place/home → 안 지움, home + PLACED → 놓인 target 해제).
> 근거: [`SUBTASK-f.verify.json`](SUBTASK-f.verify.json) — gate on 에서 놓은 뒤 엉뚱한 carve 사라짐, 성공 · pick 단계 그대로.

## 1. 구현 (implementer)

- `ag3s/config.py:275` `clustering.subtask_gate` 기본 **True**.
- `trajopt/serve_safe.py`: `--subtask-gate` (store_true) 대신 기본 on + **`--no-subtask-gate`** 로 끈다 (`--subtask-gate` 는 호환용으로 남겨도 된다 — 받으면 on). `build_ag3s(subtask_gate=…)` 가 off 일 때도 config 에 명시적으로 실려야 한다 (지금은 기본이 아닐 때만 얹는 규약 — 기본이 바뀌었으니 다시 맞춘다). manifest `subtask.gate` 는 그대로 남긴다.
- **기동 오류 규칙**: `--no-safe` · `--no-attention` · `--no-perception` 과 함께일 때 —
  사용자가 `--subtask-gate` 를 **명시**했으면 지금처럼 오류, **기본 on 이면** 오류 대신 gate 를 끄고 (label 이 없으니 어차피 아무것도 하지 않는다) 기동 로그에 한 줄 남긴다.
- in-process 경로 (`trajopt/bringup.py` 등) 와 offline 실험 스크립트가 config 기본을 따라 on 이 되는지, label 이 None 이면 결정이 예전과 같은지 확인 (테스트로).
- 테스트: gate 기본값을 가정한 테스트 (`tests/ag3s/test_subtask_gate.py` · `tests/trajopt/test_subtask_wiring.py` · `test_subtask_no_target.py` · `test_subtask_no_admissible_home.py`) 를 새 기본에 맞게 고친다 — **기대값을 바꾸는 곳마다 이유를 impl.md 에.** 새 테스트: 기본 on, `--no-subtask-gate` 로 off, 기본 on + `--no-attention` 은 기동되고 gate off, 명시 `--subtask-gate` + `--no-attention` 은 오류, label None 이면 gate on 이어도 결정이 off 와 비트 동일.
- 회귀: `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work src/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q`. `.venv-ag3s` 금지. GPU 를 잡지 않는다.
- 병렬: 다른 세션 (T40) 의 미커밋 파일 (`T40FG.verify.partial.json`, `figures/t40/*`, `SESSION_STATE.md`) 과 lead/scribe 의 미커밋 `AG3S_REVIEW_PLAN.md` · `AG3S_T0T6_LOG.md` 는 건드리지 마라. `pi05_TO_hybrid` 도.
- 커밋하지 않는다. `SUBTASK-g.impl.md`.

## 2. 검증 (verifier, lead 커밋 뒤 · T40 과 시점 조율 뒤)

- regression-baseline legacy · cuRobo — 기대값 그대로여야 한다 (label 없음 → gate 무동작): 14/15 · 15/15 · −29.031048280806342 / 10/15 · 15/15 · −9.171877401271193.
- 플래그 없이 띄운 서버 (= 기본 on) 로 2 run (ep1807 s18071 · ep1800 s18001) → SUBTASK-f gate on 과 비트 동일.
- `--no-subtask-gate` 로 2 run → T39 E3b (gate off) 와 동작 동일.
- 산출: `SUBTASK-g.verify.json`, `figures/subtask-g/` (표 하나면 충분), raw `outputs/verify/SUBTASK-g/`.
