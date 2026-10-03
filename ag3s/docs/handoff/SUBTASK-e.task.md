# SUBTASK-e — subtask gate 가 일부러 비운 target 은 "인증됨" (해제 뒤 HOLD 제거)

> writer: lead (A0, work-dd), 2026-10-03. 사용자 승인: "네 진행하세요" (2026-10-03).
> 근거: [`SUBTASK-d.verify.json`](SUBTASK-d.verify.json) — gate on 에서 B3 해제 (home + latch PLACED → target 해제) 뒤 **984/984 chunk 가 HOLD (`uncertified`)**,
> 서버 이유 `AG3S could not certify the geometry (status=no_target, grounding=subtask_gated, validity=valid)`, trajopt_status `violated` · max_violation 0.0.
> [`SUBTASK.audit.md`](SUBTASK.audit.md) §11 (문제 2 는 철회 — 이것 하나만 고친다).

## 0. 무엇을

target 이 없으면 (`status=no_target`) 지금은 "기하 미인증" 으로 HOLD 한다 — 잡을 대상을 모르면 움직이지 않는다는 원래 규칙 (`safe_policy.py:1338-1370` `_certification_reason`, `sqp.py:497`).
**gate 가 일부러 target 을 비운 경우** (`grounding_status == subtask_gated`) 는 다르다: 대상을 모르는 것이 아니라 지울 대상이 없다고 정한 것이고, field 는 아무것도 carve 하지 않은
가장 보수적인 field 다. 이 경우를 **인증된 기하** 로 본다.

## 1. 조건 (모두 참일 때만)

- `grounding_status == subtask_gated` (gate on 에서만 생긴다 → gate off 는 비트 동일해야 한다)
- `validity == valid`
- 이번 제약 집합에 carve 중인 대상이 없다 (`cs.manipulated is None`, exclusion inactive)

그 밖의 `no_target` (`NO_SEED` · `LOW_SCORE` · lost · degraded · incomplete) 은 **지금 그대로 uncertified**.

## 2. 구현 (implementer)

- 인증 판정이 어디서 내려지는지 끝까지 따라가라: AG3S `geometry_certified` (pipeline) → SafePolicy `_certification_reason` → TO `geometry_certified` 인자 (`sqp.py:497`). HOLD 를 만드는 곳이 둘 이상이면 같은 조건으로 모두.
- 새 reason 종류 (예: `subtask_no_target`) 를 기록에 남긴다 — 실행되는 chunk 에서도 "왜 target 이 없는데 실행했나" 가 보이게.
- **건드리지 않는 것**: 위 조건 밖의 인증 규칙, gate · B3 해제 로직, admissibility, latch.
- 테스트 (`tests/`): (i) subtask_gated + valid + carve 없음 → 인증됨 · HOLD 아님; (ii) 같은 상황에서 validity 가 valid 아님 → uncertified; (iii) NO_SEED 등 다른 no_target → uncertified (지금과 같음);
  (iv) gate off 경로는 결정이 비트 동일; (v) carve 중인 대상이 있으면 이 예외를 타지 않는다.
- 회귀: `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work src/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q` (cwd `/mnt/dev/work`). `.venv-ag3s` 금지.
- 커밋하지 않는다. `SUBTASK-e.impl.md` 로 넘긴다.

## 3. 검증 (verifier, implementer 뒤)

- 새 불변 snapshot (main + 이 수정의 working tree — lead 가 커밋한 뒤의 HEAD) 에서 SUBTASK-d 설비 (`outputs/verify/SUBTASK-d/p/`) 그대로:
  - regression-baseline legacy · cuRobo (gate off 기본) — 기대값 그대로 (14/15 · −29.031048280806342 / 10/15 · −9.171877401271193).
  - **gate on 48 run 다시** (같은 24 episode × seed). gate off 쪽은 T39 E3b 그대로 재사용 (V0 은 이미 통과, 이 수정은 gate off 를 바꾸지 않는다 — 2 run 으로만 다시 확인).
- 보는 것: 해제 뒤 HOLD (SUBTASK-d 984/984 → ?), 해제 뒤 로봇이 준비 자세로 돌아가나 (관절 궤적), 해제 뒤 clearance 최소 · 충돌, 성공 · 짝 2×2 (SUBTASK-d gate on 과 T39 off 둘 다와), 해제 전 구간이 SUBTASK-d gate on 과 비트 동일한가.
- 산출: `SUBTASK-e.verify.json`, `figures/subtask-e/` (실제 씬: 해제 전후 frame + ESDF 단면 / 그래프: chunk 별 HOLD · latch · label, gate on 수정 전 vs 후 / 표), raw `outputs/verify/SUBTASK-e/`.
- GPU: `nvidia-smi` 먼저, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, 포트 8230–8239, 다른 세션 서버를 건드리지 않는다.
