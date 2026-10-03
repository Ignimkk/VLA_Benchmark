# SUBTASK-d — subtask gate closed-loop 검증 (gate on vs off, 같은 seed 짝)

> writer: lead (A0, work-dd), 2026-10-02. 담당: **verifier (A2)**. 사용자 승인: SUBTASK-c closed-loop 비교는 T39 S2 뒤 (MuJoCo 승인 포함), "T39 S2 끝난 거 같습니다 … 우리 진행할 거 해보자" (2026-10-02).
> 근거: [`SUBTASK.audit.md`](SUBTASK.audit.md) §9a · §10, [`SUBTASK-c.impl.md`](SUBTASK-c.impl.md), T39 S2 실행 설비 (`outputs/verify/T39/p/`, [`T39.task.md`](T39.task.md) S2 절).
> **코드는 고치지 않는다.** 측정 스크립트는 `outputs/verify/SUBTASK-d/`.

## 0. 질문

SUBTASK-c gate (`--subtask-gate`: 확정 place/home → 새 target 채택 · 교체 없음, home + latch PLACED → 기존 target 해제) 를 켜면 closed-loop 에서
(a) PLACED 뒤의 carve 가 실제로 사라지나, (b) pick 단계 (파지 전) 는 그대로인가, (c) 성공률 · HOLD 가 나빠지지 않나.

## 1. 설비 — T39 S2 를 그대로 다시 쓴다

- **불변 snapshot** `/mnt/dev/work-sd` — T39 S2-0 와 같은 방식 (`T39.verify.partial.json` 의 `s2_0.snapshot`):
  benchmark = `git -C /mnt/dev/work/benchmark archive <main HEAD>` (지금 `73dcb37` — 시작 시 HEAD 를 기록) + `ag3s/asset/data` symlink,
  pi05_TO_hybrid = `rby1_bringup` 은 `archive 22d8dae`, 나머지 최상위는 symlink, `/mnt/dev/work-sd/src` → `pi05_TO_hybrid` (T39 의 extra symlink 와 같은 이유).
  `ag3s/asset/subtask_probe/` 는 git 에 있으므로 archive 에 들어간다 — 들어갔는지 확인. 서버 기동 전 code md5 확인 (T39 `code_md5.sh` 처럼).
- 스크립트: `outputs/verify/T39/p/` 를 `outputs/verify/SUBTASK-d/p/` 로 복사해 경로만 바꾼다. **E3b 조건만** 쓴다 (서버 flag · client flag 는 T39 E3b 그대로).
  gate on 은 서버에 `--subtask-gate` 하나만 더한다. `XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"` (T39 와 같음).
- 포트 **8230–8239** (T39 의 8220–8229 · 8123 금지). 서버는 `PYTHONPATH=/mnt/dev/work-sd`, cwd `/mnt/dev/work-sd`. import 된 `benchmark.*` 의 `__file__` 이 snapshot 아래인지 첫 run 에서 기록.
- GPU: `nvidia-smi` 먼저, `XLA_PYTHON_CLIENT_PREALLOCATE=false`. E3b 서버 peak 17.5–19 GB (T39 측정) → 4 개 병렬 가능하면 4 개. 총 사용 ≤ 130 GB 규칙 (T39 env.sh 와 같음).

## 2. 측정

**V0 — gate off 동일성 (먼저, 실패하면 멈추고 보고).** merge 된 main (snapshot) 에서 **gate off** E3b 를 2 run:
ep1807 seed 18071 (T39 S2-1 의 E3b_a 와 짝) · ep1800 seed 18001 (S2-2 의 E3b 와 짝). T39 쪽 run 과 비교:
- client `--trajectory-out` npz · 실행된 actions · frames 의 latch 상태 · 매 chunk 의 manipulated id / exclusion — **비트 동일**이어야 한다.
- server 기록은 SUBTASK-c 가 키를 더했으므로 (`subtask`, manifest `subtask`) **그 키들을 뺀 나머지**가 동일한지.
- 동일하면 **T39 S2-2 의 E3b 48 run 을 gate-off 쪽으로 쓴다** (다시 돌리지 않는다). 아니면 어디서 처음 갈리는지 (chunk · 필드) 보고하고 멈춘다.

**V1 — gate on 48 run.** T39 S2-2 와 같은 24 episode × seed `10·ep + rep` (rep 1, 2), 목록은 `outputs/verify/T39/p/lists/E3b_*.txt`. client 공통 flag 도 같다.

**V2 — 분석 (gate on vs T39 gate off, (episode, seed) 짝):**

| 항목 | 무엇 |
|---|---|
| 성공 | grasp · place · success (T39 `analyze_t39.py` 규칙 그대로), 짝 2×2 + exact McNemar p |
| PLACED 뒤 carve | PLACED 도달 run 수, PLACED 뒤 `switch` 수, exclusion active chunk 수 / 전체, `released` 이벤트 수와 PLACED → 해제까지 chunk 수 |
| pick 단계 영향 | PLACED 전에 gate 가 막은 것 (`subtask_blocked` = first / switch) 의 수와 위치, 첫 target 채택 chunk (gate on vs off) |
| label | chunk 별 확정 label vs latch 상태 (SEARCHING/LATCHED/CLOSING/HELD/PLACED) 의 교차표, pick→place · place→home 전환 지연 |
| 갈림 | 같은 (ep, seed) 에서 actions 가 처음 다른 chunk 와 그때의 latch 상태 |
| 안전 · 기타 | HOLD chunk 수, verdict 분포, clearance 최소, chunk 시간 (AG3S · TO · policy) |

## 3. 산출 (규칙 A)

- `outputs/verify/SUBTASK-d/` — raw · 스크립트 · 로그. 끊기면 이어가도록 `handoff/SUBTASK-d.verify.partial.json` 을 run 마다 갱신.
- `figures/subtask-d/` (json sidecar 포함):
  1. **실제 씬** — PLACED 전후 frame card 2–3 개 (gate on vs off 나란히): 카메라 image + attention target · manipulated · exclusion 표시 (가능하면 ESDF 단면).
  2. **그래프** — 대표 episode 의 chunk 별 label · latch · exclusion active (on vs off), 전체 run 의 PLACED 뒤 exclusion active 비율 분포.
  3. **표** — 짝 2×2 · McNemar, V2 표의 숫자 전부.
- `handoff/SUBTASK-d.verify.json` — `_SCHEMA.verify.json`, numbers 와 경로만.

## 4. 추가 (CPU, 아무 때나) — SUBTASK (a) 의 숫자를 verify.json 으로

첫 측정 (text decode · AUROC, 2026-09-30) 은 숫자가 figure sidecar 에만 있다. scribe 가 쓰려면 verify.json 이 필요하다.
`outputs/verify/subtask_probe/{smoke,discover,episode,auroc}.json` 과 `figures/subtask/*.json` 에서 **다시 계산해** `handoff/SUBTASK.verify.json` 을 만든다
(template discovery 표, 생성 정확도 · action-token 비율 · object 이름 정확도, AUROC 4 episode × 3 variant, FAST 영역 비율 97.4 %, latency 21.0 ms/token, LLM weight 상대오차 1.7e-3).
값이 audit 문서와 다르면 verify.json 의 값이 정본이고 차이를 적는다.

## 5. 넘길 것

끝나면 lead 에게 보고 (SendMessage `main`): V0 결과, V1 완료 수, V2 핵심 숫자 표, figure 경로, 실패 · 생략 항목.
