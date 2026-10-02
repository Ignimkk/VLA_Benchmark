# SUBTASK-b — hidden-state probe: 우리 checkpoint 의 VLM 안에서 phase 를 읽을 수 있나

> writer: lead (A0), 2026-10-01. 담당: **verifier (A2)**. 사용자 승인: "b 를 gpu 에 올려 테스트" (2026-10-01).
> 배경: [`SUBTASK.audit.md`](SUBTASK.audit.md) §3 (text decode 는 불안정) · §7 (추론 경로, probe 제안).
> **코드는 고치지 않는다.** 측정 스크립트는 `outputs/verify/SUBTASK-b/` 에 둔다. MuJoCo 는 쓰지 않는다.

## 0. 질문

우리 16D checkpoint 의 hidden state 위에 작은 classifier (probe) 를 학습하면 frame 마다 **pick / place / home** 을 가를 수 있나.
VLA 는 그대로 두고 읽기만 한다. 특히 **State 문자열 속 gripper 값 없이도** 갈리는지를 따로 잰다
(pi0.5 는 state 를 prompt text 에 넣으므로, probe 가 gripper 숫자만 읽으면 지금 rule 과 같은 것이다).

## 1. 기준선

**생략한다 (lead 판단).** 이 측정은 AG3S · trajopt 코드를 한 줄도 거치지 않고 VLA forward 만 쓴다. `regression-baseline` 이 보는 숫자와 무관하다.

## 2. 고정 사항

| 항목 | 값 |
|---|---|
| checkpoint | `pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999` (`pi05_infer.py:99` 가 serving 하는 것). config `pi05_rby1_randomized_pick_place_16d_lora`. **LoRA on 그대로** |
| feature 추출 venv | `/mnt/dev/work/.venv-openpi-live/bin/python` + `OPENPI_DATA_HOME=/mnt/dev/work/pi05_TO_hybrid/cache/openpi XLA_FLAGS="--xla_gpu_enable_command_buffer=" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/dev/work` (cv2 · av · lerobot 있음, sklearn 없음) |
| probe 학습 venv | `/mnt/dev/work/.venv-curobo/bin/python` (sklearn 있음) — feature 를 npz 로 넘긴다. **`.venv-ag3s` 는 NFS 에서 멈춘다, 쓰지 않는다** |
| 입력 변환 | `policy._input_transform({"state", "images": {cam: CHW uint8}, "prompt"})` → `Observation.from_dict` → `preprocess_observation(None, obs, train=False)` (참고: `benchmark/ag3s/experiments/sources/pi05_attention.py:observation_from_step`) |
| prefix forward 참고 | `outputs/verify/subtask_probe/subtask_probe.py` 의 `TextHead` (nnx split/merge + jit), `pi05_attention.py` 의 `AttentionSampler` (suffix pass, `return_attn_probs=True`) |

## 3. 데이터

| 집합 | 출처 | 표본 | label |
|---|---|---|---|
| **train** | dataset `pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1`, train split 0–1599 | episode 8 개마다 1 개 (200 ep) × 6 frame 마다 1 개 | parquet `phase_index` |
| **val** | 같은 dataset, validation split 1600–1799 | episode 4 개마다 1 개 (50 ep) × 6 frame 마다 1 개 | parquet `phase_index` |
| **closed-loop** | `outputs/live_test/20260925_ep1807/run_0000` · `20260925_train/{ep0,ep500,ep1200}/run_0000` 의 `step_*.npz` (이미지 = policy 가 실제로 본 tensor) | 전부 (75 + 57×3 = 246) | MuJoCo 참값 — `outputs/verify/subtask_probe/subtask_episode.py:ground_truth()` 와 같은 규칙 |

- dataset 이미지는 `videos/chunk-XXX/observation.images.<cam>/episode_XXXXXX.mp4` (224×224, h264, 15 fps). parquet 의 `frame_index` 와 맞춰 decode 한다. state 는 `observation.state` (16), prompt 는 `task_index` → `meta/tasks.jsonl`.
- `phase_index` 정의는 `pi05_TO_hybrid/rby1_manipulation/src/rby1_manipulation/tasks/transport_atomic.py:105` (`Phase`). 20 이상 (recovery · basket) 이 나오면 따로 세고 학습에서 뺀다.

**3-class 묶음** (closed-loop 참값과 맞춘 것 — gripper 는 phase 4–8 에서 닫혀 있다):

| class | `phase_index` | closed-loop 참값 |
|---|---|---|
| pick | 0 INITIAL_HOLD · 1 APPROACH · 2 PREGRASP_ALIGN · 3 GRIPPER_CLOSE | reach |
| place | 4 GRASP_VERIFY · 5 LIFT_FROM_TABLE · 6 TRANSPORT · 7 LOWER_INTO_BASKET · 8 RELEASE | held |
| home | 9 RELEASE_VERIFY · 10 RETREAT · 11 TARGET_SETTLE · 12 RETURN_TO_READY · 13 TERMINAL_HOLD | released |

## 4. feature (코드 수정 없이 얻을 수 있는 것만)

| 이름 | 무엇 | 얻는 곳 |
|---|---|---|
| `prefix_last` | Gemma 최종 hidden, 마지막 유효 prompt token | `PaliGemma.llm([prefix_tokens, None], …)` 의 첫 출력 (`sample_actions` 는 버리지만 반환은 된다) |
| `prefix_img` | Gemma 최종 hidden, image token 768 개 평균 | 같은 호출 |
| `kv_L{l}` | prefix KV cache 의 V, layer l ∈ {0, 4, 8, 12, 17}, 유효 text token 평균과 image token 평균을 이어 붙인 것 | 같은 호출의 `kv_cache` — **action expert 가 실제로 읽는 것** |
| `ae_final` | action expert 최종 hidden (`suffix_out`), action token 평균, denoise step 0, noise seed 0 | suffix pass (AG3S attention 과 같은 조건) |
| `attn_L8` | layer 8 attention probs, head 8 개 × (camera 3 개 mass + text mass), `last` agg | `return_attn_probs=True` suffix pass — AG3S 가 쓰는 바로 그 층 |

## 5. 측정

**M1 — probe 정확도.** feature 마다 standardize + logistic regression (L2, C 는 train 안에서 episode 단위 group CV 로 고른다).
train 에서 학습 → **val** 과 **closed-loop** 에서 평가. 3-class accuracy · class 별 recall · confusion · one-vs-rest AUROC.
fine 14-way probe 도 하나 (최적 feature 로) — confusion 만.

**M2 — gripper 를 가린 control.** State 문자열의 gripper 두 값 (state index 7 · 15) 을 **normalize 전에 "열림" 상수 1.0 (dataset 의 열린 gripper 값; 닫힘은 ≈0.666) 으로 덮어쓰고** feature 를 다시 뽑는다.
(a) 정상 feature 로 학습 → 가린 feature 로 평가, (b) 가린 feature 로 학습 · 평가. val 과 closed-loop 둘 다.
image 에서 phase 를 읽는지, State 숫자를 읽는지를 가른다.

**M3 — rule 기준선.** 같은 frame 에서 rule: gripper 닫힘 (`min(state[7], state[15]) < 0.9`) → place; 열림이면 그 episode 에서 이전에 닫힌 적이 있으면 home, 없으면 pick.
M1 과 같은 지표로.

**M4 — 시간.** feature 추출 1 frame 당 ms (prefix pass · suffix pass 따로, JIT 첫 호출 제외), probe 추론 ms.

## 6. 산출물

- raw: `outputs/verify/SUBTASK-b/` — 추출 스크립트, feature npz, probe 학습 스크립트, 예측 csv (frame 마다 set · episode · t · label · class 확률).
- figure (`figures/subtask-b/`, json sidecar 포함, 규칙 A):
  1. **실제 씬** — ep1807 의 reach · held · released 대표 frame 과 val 예시 frame 에 probe 확률 막대 (최적 feature, 정상 · 가림 둘 다).
  2. **그래프** — closed-loop 4 run 의 frame 별 class 확률 timeline 과 참값 띠; feature × layer 별 val accuracy 막대 (정상 vs 가림).
  3. **표** — feature × {val, closed-loop} × {정상, 가림 a, 가림 b} 의 accuracy · AUROC, rule 기준선, 14-way confusion.
- `handoff/SUBTASK-b.verify.json` — `_SCHEMA.verify.json` 따름. **numbers 와 경로만.**
- 끝나면 lead 에게 SendMessage.
