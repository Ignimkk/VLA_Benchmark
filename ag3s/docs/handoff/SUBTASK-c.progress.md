# SUBTASK-c progress (implementer)

> writer: ag3s-implementer · 재개용 이정표. 최신이 아래.

| 시각 (KST) | 이정표 | 상태 |
|---|---|---|
| 2026-10-02 | task · audit §8/§9a · SUBTASK-b verify/extract/probe 읽음. 코드 조사 (TargetConfirm.select · ground_target · pipeline freeze/metrics · AttentionSampler · AttentionPolicy · grasp_latch PLACED) | done |
| 10-02 08:00 | A1: `experiments/tools/export_subtask_probe.py` (.venv-curobo) → `asset/subtask_probe/kv_L4_v1.{npz,json}`. parity val numpy 0.98078 vs verify 0.9808 (diff 2.3e-5), confusion 동일 | done |
| 10-02 08:10 | A2: `stages/subtask_probe.py` (신규: SubtaskProbe · kv_feature · SubtaskDebounce) · `pi05_attention.AttentionSampler.attention_and_subtask_feature` (기존 prefix KV 에서 jit reduction) · `attention_policy.AttentionPolicy` result["subtask"] | done (GPU 확인 전) |
| 10-02 08:20 | A3: config `clustering.subtask_gate/_confirm_frames/_probe_path` · `GroundingStatus.SUBTASK_GATED` · `TargetConfirm.note_subtask` + gate (first · switch, frozen 이면 무시) · `AG3S.set_subtask` + `metrics["subtask"]`. 기존 target_confirm/config/manipulated/grounding 123 passed | done |
| 10-02 08:35 | A4: `tests/ag3s/test_subtask_gate.py` (신규) 31 passed 1 skipped (GPU opt-in) · `tests/conftest.py` 에 `slow` marker 등록 | done |
| 10-02 08:45 | A4 #7 (GPU, .venv-openpi-live, AG3S_GPU_TESTS=1): ep1807 step_00025 (t_step 200) kv_L4 rel_l2 2.03e-8 · max_abs 5.96e-8 vs feat/closed; attention 은 feature 유무와 무관하게 bit 동일 | done |
| 10-02 09:00 | §5 조사: T34·T37 E3a/E3b server_constraints 24 episode 스캔 — PLACED 14 중 12 가 PLACED 뒤 switch (8 = crate 안의 사과, 4 = 탁자 위 다른 물체), 그 뒤 433 chunk 중 406 이 exclusion active | done |
| 10-02 09:20 | 전체 회귀 (tests/trajopt tests/ag3s, T39 미커밋 변경 포함) 1772 passed · 2 skipped · 0 failed | done |
| 10-02 09:25 | `SUBTASK-c.impl.md` (Part A) 작성. §5 스캔을 `experiments/tools/scan_placed_switch.py` 로 옮겨 재현 확인 | done |
| 10-02 09:25 | **Part B 대기** — `T39.impl.md` 없음. safe_policy · serve_safe 는 손대지 않았다 | waiting |
| 10-02 (Part B) | lead 지시로 Part B 시작 (T39 S1 커밋 e01b7bf 확인, HEAD 5f38468). safe_policy 배선 · serve_safe `--subtask-gate`/`--subtask-probe` · 기록 키 · B3 (TargetConfirm.note_placed · 해제) | done |
| 10-02 (Part B) | 새 테스트: test_subtask_gate 42 passed 1 skipped · test_subtask_wiring 19 passed. 회귀 (GPU env 없이) 1803 passed 2 skipped. CPU env (`JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false`) 회귀 1803 passed · 2 skipped · 0 failed. impl.md Part B 절 작성 | done |
