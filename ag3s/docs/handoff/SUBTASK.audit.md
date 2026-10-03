# SUBTASK audit — pi0.5 가 말하는 subtask 로 "지금 attention 이 무엇을 위한 것인가" 를 알 수 있는가

> writer: lead (A0). 2026-09-30 10:16. **측정 기록이고 판정 전이다.** `AG3S_T0T6_LOG.md` 이어쓰기는 사용자 판정 뒤 A3 가 한다.
> raw · 재현 script: [`outputs/verify/subtask_probe/`](../../../../outputs/verify/subtask_probe/) — figure 의 숫자는 각 `.json` sidecar 에 있다.

## 0. 배경 — 왜 묻나

AG3S 는 attention 이 가리키는 target 을 ESDF 에서 지운다(carve). place 할 때는 지우지 않는다는 약속이
지금은 **rule** (gripper 닫힘 / attach 상태) 로 걸려 있다. pi0.5 논문의 hierarchical inference 는
먼저 **subtask 문장**("pick up the pillow")을 text 로 생성하고 그 문장에 조건부로 action 을 만든다.
그 문장을 읽을 수 있으면 "지금 attention 이 pick 을 위한 것인가, place 를 위한 것인가" 를 model 에게서 직접 얻는다.

## 1. 답 (요약)

| 질문 | 답 | 근거 |
|---|---|---|
| openpi 에 subtask 생성 코드가 있나 | **없다** | openpi 소스 grep 0 건. GitHub issue #647 · #664 · #679 · #701 · #813 모두 같은 질문이고 maintainer 답 없음 |
| 우리가 구현할 수 있나 | **있다 — model 수정 없이** | Gemma 의 LM head 는 embedding table 을 공유(`Embedder.decode = x @ E.T`)하므로 weight 가 이미 checkpoint 안에 있다. prefix-LM decode ~100 줄로 영어 subtask 생성 확인 |
| released `pi05_base` 가 subtask 를 말하나 | **말한다, format 도 드러낸다** | `caption en\n` → `Subtask: pick up apple`. 학습 format 이 `Subtask: …` 임을 model 이 스스로 보인다 |
| issue #679 의 "garbled words" 는 무엇인가 | **FAST action token** | openpi 원형 prompt `Task: …, State: …;\nAction: ` 뒤 생성 token 의 **97.4 %** (192 개 중) 가 FAST 영역 id 254,976–257,023 (vocab 2048). 고장이 아니라 discrete action 을 말하는 중 |
| 16D checkpoint 안에 그 language head 가 남아 있나 | **LLM 은 그대로, 눈(SigLIP)은 바뀜** | LoRA freeze filter 가 `.*llm.*` 를 얼림 → LLM base weight 상대오차 1.7e-3 (bf16 저장 반올림 수준). SigLIP 은 freeze 대상이 아니라 최대 17 % 변경 |
| 그 신호로 pick / place 를 frame 마다 가를 수 있나 | **zero-shot 으로는 못 쓴다** | 아래 §3. 자유 생성은 object 를 지어내고, served model 은 held 구간에서 action token 으로 무너지며, score 기반 분리도(AUROC)는 episode 마다 0.50–1.00 로 흔들린다 |

## 2. 방법

- **decoding**: prefix = 3 camera SigLIP token (768) + text token. image·prompt 는 bidirectional, 생성 token 은 causal
  (prefix-LM). logits = final-norm hidden · embedding.T. greedy, 최대 16–40 token. KV cache 없이 매 token 전체 forward.
- **variant 3 개**: `base` (pi05_base) · `ft16d` (served 16D, LoRA on) · `ft_nolora` (16D 에서 `lora_b` = 0 → released LLM + fine-tuned SigLIP).
- **template**: 14 개를 3 frame 에서 탐색 → 3 개를 전체 episode 에 사용 (`Task: {p};\n` · `{p}\n` · `Task: {p}. Subtask: ` causal).
- **score**: 후보 subtask `pick up the X` / `place the X in the basket` / `return to home position` 의 log-likelihood (`Subtask: ` 뒤 causal).
  Δ = log p(place) − log p(pick).
- **입력**: 기록된 rollout 4 개 — `20260925_ep1807` (apple, 75 frame) + `20260925_train/ep0·ep500·ep1200`
  (orange · banana · pear, 57 frame 씩) = **246 frame**. **네 개 모두 우리 16D checkpoint (`…_xla_retry_20260923/29999`, `pi05_infer.py:99` 가
  가리키는 것) 의 headless MuJoCo closed-loop 실행 기록**이다 (`meta.json` argv: `--model rby1_randomized_pick_place_16d --headless`).
  ep1807 은 test split (1800–1999), 나머지 셋은 train split (0–1599) layout. dataset 재생이 아니다.
- **참값 phase** (MuJoCo qpos): reach = gripper 열림·과일 crate 밖 / held = gripper 닫힘 / released = gripper 열림·과일 crate 안.
  기대 subtask: reach→pick, held→place, released→home.

## 3. 결과

| figure | 무엇 |
|---|---|
| [`subtask-template-discovery.png`](../figures/subtask/subtask-template-discovery.png) | 14 template × 3 frame × 3 variant — 어느 prompt 에서 영어가 나오나 |
| [`subtask-scene-ep1807.png`](../figures/subtask/subtask-scene-ep1807.png) | 실제 씬 5 frame (reach → held → released) 과 각 variant 가 말한 subtask |
| [`subtask-timeline.png`](../figures/subtask/subtask-timeline.png) | 4 episode 의 Δ 추이 + 생성 문장 class 띠 vs 참값 phase |
| [`subtask-accuracy-table.png`](../figures/subtask/subtask-accuracy-table.png) | AUROC · 생성 정확도 · action-token 비율 · object 이름 정확도 |

**(a) score 분리도 — AUROC (0.5 = 무작위, 1.0 = 완전 분리)**

| variant | ep1807 apple Δ held vs reach | ep0 orange | ep500 banana | ep1200 pear | released vs rest (4 ep) |
|---|---|---|---|---|---|
| pi05_base | 0.63 | 0.96 | 1.00 | 0.87 | 0.94 · 0.49 · 0.33 · 0.35 |
| 16D ft (served) | 0.50 | 0.57 | 0.75 | 0.85 | 0.95 · 0.85 · 0.88 · 0.87 |
| 16D ft, LoRA off | 0.53 | 0.68 | 0.59 | 0.83 | 0.88 · 0.69 · 0.80 · 0.92 |

**(b) 자유 생성 (246 frame)**

| variant | 정확도 `{p}\n` | 정확도 `Task: {p}. Subtask: ` | action-token 비율 `Task: {p};\n` | object 이름 맞음 `Task: {p};\n` |
|---|---|---|---|---|
| pi05_base | 21 % | 20 % | 13 % | 54 % |
| 16D ft (served) | 35 % | 11 % | 48 % | 0 % |
| 16D ft, LoRA off | 48 % | 39 % | 29 % | 1 % |

- 3-way argmax(pick/place/home 중 최고 score)는 **세 variant 모두 246/246 frame 에서 `home`** — 짧은 후보가 유리한 length bias.
  정확도 64.2 % 는 majority baseline(released 158/246)과 같아 정보가 없다. 버린 지표다.
- `pi05_base` 는 자유 생성에서 거의 항상 `pick up …` 이다 (place 로 넘어가지 않음). object 는 "red energy drink", "blue pen" 처럼 지어낸다.
- served `ft16d` 는 held 구간에서 영어 대신 FAST action token 을 낸다 — template `Task: {p};\n` 에서 held 36 frame 중 30
  (ep1807 13/13 · ep0 7/7 · ep500 3/8 · ep1200 7/8). **place 를 말해야 할 바로 그 구간이 비어 있다.**
- released 뒤 `return to home position` 쪽 score 는 served `ft16d` 에서 네 episode 모두 높다 (released vs rest AUROC 0.85–0.95).
- latency: token 당 forward median **21.0 ms** (naive, KV cache 없음, H200, bf16). KV cache 판은 미측정.

## 4. 해석 — 구조적 한계 하나

pi0.5 원 설계에서 attention 이 "그 subtask 를 위한 것" 인 이유는 **action expert 가 생성된 subtask token 에 조건부**이기 때문이다.
우리 16D 는 full task prompt 로만 학습됐고 subtask 를 한 번도 보지 않았다. 그래서 zero-shot 으로 읽은 subtask 는 VLM 쪽의 장면 해석일 뿐,
AG3S 가 쓰는 action-expert attention (layer 8 · head 2) 이 무엇을 위한 것인지에 대한 **인과적 보장이 없다.**

## 5. 선택지 (판정 요청)

| | 무엇 | 비용 | 보장 |
|---|---|---|---|
| A | zero-shot subtask 를 그대로 사용 | 코드 ~100 줄 | §3 의 수치로 **비추천** |
| B | **subtask co-training** — dataset 의 frame 별 `phase_index` (14 phase: APPROACH … TRANSPORT · LOWER_INTO_BASKET · RELEASE … RETURN_TO_READY) 를 subtask 문장으로 바꿔 text CE loss 를 추가, inference 에서 subtask 를 먼저 생성하고 action 을 그것에 조건부로 | openpi `compute_loss`·tokenizer·inference 수정 + 16D 재학습 | pi0.5 원 설계와 같은 인과 보장 |
| C | **phase probe** — 얼린 VLM prefix hidden 위에 작은 classifier 를 `phase_index` 로 학습 | 재학습 없음, 작음 | text 아님, 인과 보장 없음 (A 보다 정확할 것 — 미측정) |
| D | 지금 rule 유지 (gripper / attach) | 0 | 이 측정의 참값 phase 정의가 곧 gripper 상태라, 이 지표로는 rule 을 이길 수 없다 |

## 6. 사용자 판정 (2026-09-30)

- **우리 checkpoint (16D, LoRA on, 29999) 를 쓴다.** → §3 표의 `16D ft (served)` 행이 해당 수치다.
- 질문: "headless MuJoCo 로 돌려 테스트하면 zero-shot 이 아니지 않나" → 답: 장면(image)은 in-distribution 이지만
  **subtask text 출력은 여전히 zero-shot** 이다. 16D 학습 loss 는 action flow-matching MSE 뿐이고 (`pi0.py` `compute_loss`),
  prompt 는 full instruction 12 종뿐이며, LM head·LLM base weight 는 LoRA 가 얼린 pi05_base 그대로다. §3 의 frame 이 이미
  headless closed-loop 기록이므로, read-only 로 subtask 를 읽는 한 새 headless 실행은 같은 조건의 표본을 늘릴 뿐이다.

## 7. "instruction 이 Gemma 에서 subtask 로 나뉘고 attention 이 action expert 로 가는 것 아닌가" (사용자 질문, 2026-09-30)

도식: [`subtask-inference-paths.png`](../figures/subtask/subtask-inference-paths.png)

우리 checkpoint 의 실제 경로에는 **subtask 가 나뉘는 단계가 없다.**

1. prompt 는 `Task: {instruction}, State: {discretized state};\nAction: ` 하나 (`tokenizer.py:28`).
2. Gemma 2B 는 image 768 token + 이 prompt 를 **forward 한 번** 하고 **KV cache 만** 남긴다. 출력 hidden 은 버린다 (`pi0.py:245`: `_, kv_cache = …`). text 를 한 token 도 생성하지 않는다.
3. action expert 300M 이 10 denoise step 동안 그 KV cache 에 attend 해서 action 을 만든다 (`pi0.py:277`). AG3S 가 읽는 attention 이 이 단계의 layer 8 · head 2 다.
4. 학습 loss 는 action MSE 뿐 (`pi0.py:216`) — text 를 내도록 학습된 적이 없다.

pi0.5 논문의 hierarchical inference 는 2 와 3 사이에 **AR text decode (subtask 생성) → subtask token 을 prefix 에 붙임** 이 있다. openpi 공개 코드와 우리 학습은 이 단계를 뺐다.

**그래도 VLM 안에 phase 정보는 있다 (논리):** pi0.5 는 state 를 prompt text 에 넣으므로, action expert 가 받는 입력은 prefix KV · noise · timestep 뿐이다.
그런데 action 은 phase 마다 다르다 → phase 를 가를 정보는 prefix KV 안에 **있어야 한다.** 문제는 그것이 *text 로* 나오도록 학습되지 않았다는 것.
그래서 text decode (a) 대신 **hidden-state probe (b)** 로 읽는 것이 다음 측정 후보다 — 선형으로 읽히는지는 미측정.

**probe 측정 제안 (승인 대기 — checkpoint 사용, MuJoCo 불필요, offline):**

| 항목 | 내용 |
|---|---|
| feature | ① Gemma prefix hidden (마지막 text token · image token 평균, 몇 layer) ② action-expert hidden, layer 8 · denoise step 0 (AG3S 가 attention 을 읽는 바로 그 query) |
| label | dataset `phase_index` → pick (0–5) / place (6–9) / home (10–13) |
| 학습 / 평가 | train split 표본 episode 로 logistic regression → validation split (1600–1799) + 우리 closed-loop 기록 4 run 으로 평가 |
| 지표 | accuracy · AUROC, rule (gripper 상태) 와 나란히 |

## 8. SUBTASK-b 결과 — hidden-state probe (2026-10-01, verifier 측정)

정본 숫자: [`SUBTASK-b.verify.json`](SUBTASK-b.verify.json). figure: [`subtask-b-scene.png`](../figures/subtask-b/subtask-b-scene.png) ·
[`subtask-b-graph.png`](../figures/subtask-b/subtask-b-graph.png) · [`subtask-b-table.png`](../figures/subtask-b/subtask-b-table.png). raw: `outputs/verify/SUBTASK-b/`.

| feature (3-class acc / macro AUROC) | val normal | val gripper 가림(b) | closed246 normal | closed246 가림(b) |
|---|---|---|---|---|
| prefix_img (Gemma image hidden) | 0.980 / 0.999 | 0.978 / 0.999 | 0.955 / 0.997 | 0.955 / 0.994 |
| **kv_L4** (best, action expert 가 읽는 KV) | 0.981 / 0.999 | 0.980 / 0.999 | 0.959 / 0.997 | 0.955 / 0.998 |
| ae_final (action expert hidden) | 0.980 / 0.999 | 0.980 / 0.999 | 0.963 / 0.998 | 0.967 / 0.993 |
| attn_L8 (AG3S 가 읽는 층의 attention) | 0.974 / 0.999 | 0.975 / 0.999 | 0.955 / 0.994 | 0.951 / 0.996 |
| rule (gripper + 이력) | 0.942 / 0.959 | — | 1.000 / 1.000 | — |
| control: frame_index 만 | 1.000 | — | — | — |
| control: arm joint 14 (gripper 제외) | 0.697 | — | 0.569 | — |

run 별 (kv_L4 normal, closed414, **lead 가 `predictions.csv` 에서 계산 — verify.json 에 없음**): ep1807 (test split) 0.960 · ep0 0.973 · ep500 0.965 · ep1200 0.938.
오분류는 전부 전환 지점: held→pick (잡는 순간) 과 released→place (놓는 순간). val 50 개 전부 전환 1 sample 안, closed414 17 개 전부 3 sample 안.

**읽는 법 — 무엇이 확인됐고 무엇이 아닌가**

- 확인: 우리 checkpoint 의 hidden state 에서 phase 가 **선형으로** 읽힌다. Gemma 쪽 · action expert 쪽 · attention 자체 모두.
- 확인: State 문자열의 gripper 값을 가려도 그대로다 (0.981 → 0.980). arm joint 만으로는 0.70 → image 에서 읽는다.
- 한계 1: dataset `phase_index` 는 scripted 라 frame 시각만으로 1.000 이 나온다 (248/250 episode 가 같은 phase 열). val 은 쉬운 시험이다. closed-loop (policy 자신의 속도) 의 0.93–0.97 이 더 정직한 숫자다.
- 한계 2: closed-loop 참값은 gripper 로 정의했으므로 rule 은 거기서 정의상 1.000 이다. **이 데이터로는 probe 가 rule 을 이긴다고 말할 수 없다.** probe 가 rule 을 이길 수 있는 곳은 gripper 는 닫혔는데 사과가 손에 없는 경우 (T35-diag 의 미끄러짐) — 그 run 들은 policy image 를 저장하지 않아 아직 못 쟀다.
- 한계 3: held-out closed-loop 은 ep1807 하나. ae_final · attn_L8 은 noise seed 하나.
- 통합 비용: kv_L4 = layer 4 V cache (text 평균 256 + image 평균 256). `pi05_attention.py` `AttentionSampler._prefix` 가 AG3S attention 용으로 이미 계산하는 KV cache 에서 나온다 → 추가 forward 없음, probe 0.09–0.13 ms (CPU).

## 9. label 을 carve 판정에 어떻게 쓰나 — lead 제안 (2026-10-01, 사용자와 논의 중, 미판정)

**지금의 약속은 geometry 다.** attention 1 등의 역할은 크기로 정해진다 — 잡을 수 있으면 manipulated 후보
(`pipeline.py:_exclusion_gate`), 못 잡으면 destination (`admissibility.py:DestinationRegistry`), attach/detach 는
물리 증거 (`grasp_latch.py`). crate 는 "너무 커서" carve 에서 빠진다. 크기로 갈리지 않는 destination
(bowl · plate · 쌓기 · 건네기) 에서는 깨진다.

**제안 — 세 권한 분리, label 은 carve 를 줄일 수만 있다:**

1. label → **역할**: pick 이면 attention 1 등 = manipulated 후보, place 면 = destination 후보.
2. geometry → **carve 허용** (admissibility · destination overlap 그대로, safety invariant).
3. latch → **attach/detach**. CLOSING · HELD 동안 label 은 역할을 바꾸지 못한다.

| latch | label (3 frame 연속) | attention 1 등 역할 | carve | 지금 대비 |
|---|---|---|---|---|
| SEARCHING / LATCHED | pick | manipulated 후보 | geometry 허용 시 | 같음 |
| SEARCHING / LATCHED | place | destination 후보 | 안 함 | 바뀜 — 잡을 수 있는 크기의 destination 도 보호 |
| CLOSING / HELD | (무시) | latch 가 잠근 쥔 물체 | attach 경로 | 같음, 불일치는 기록 |
| PLACED 이후 | home | 없음 | 새 target 없음 | 미확인 — 지금 새 과일을 잡는지 측정 필요 |

- 측정된 오류 (closed-loop) 는 전부 늦은 전환 → 이 설계에서 둘 다 보수적. 위험한 방향은 val 의 GRIPPER_CLOSE 경계 pick→place 17 개 — 규칙 3 이 막아야 한다 (closed-loop 에선 0 개).
- 지금 과제에서는 거의 바뀌는 것이 없다 (crate 가 크기로 갈림). T35-diag 실패 (들 때 미끄러짐) 는 역할 판정 문제가 아니라 label 로 안 고쳐진다. 가치는 일반성과 거짓 attach 교차검증.
- 순서 제안: ① shadow — label 을 기록만 하고 "썼다면 달라졌을 판정" 을 남김 (implementer) ② 미끄러짐 재현 run 을 shadow 로 (MuJoCo 승인 필요) ③ 그 뒤 veto-only 켜기 판정.

### 9a. 사용자 판정 (2026-10-02) — spec 은 단순하다

> "phase 가 pick 이면 target 을 지운다, place 면 지우지 않는다. 이게 전부다."

§9 의 표 · 세 권한 분리 · shadow 단계는 이 spec 으로 대체한다. 구현 전에 확인 요청한 두 가지 (미판정):
1. place 중 **쥔 사과**는 지금처럼 attach 로 field 에서 뺀다 — "지우지 않는다" 는 attention target (바구니) 에 적용.
2. pick → place 전환은 grasp 확인 (latch HELD) 까지 pick 을 유지 — label 이 닫히는 도중 먼저 바뀌는 경우 (val 17 frame, closed-loop 0) 대비.
home phase 는 place 와 같이 처리 (지우지 않음).
- (2026-10-02) 1 번 **확정** — 쥔 사과는 attach 로 robot tree 에 포함 (지금 code 그대로).
- 2 번 설명: 사과의 처리 방식이 carve (pick) → attach (place) 로 바뀌는 순간을 무엇이 일으키나. label 이 일찍 바뀌면 실제 사과가 field 로 돌아오고 gripper 엔 가짜 사과 (T34 거짓 attach 와 같은 모양), 늦게 바뀌면 쥔 사과가 충돌 계산에서 빠진다.
  lead 추천: **사과의 carve → attach 전환 = latch HELD (지금처럼, 물리 증거)**, **label = attention target 을 carve 할지만** (pick 지움 / place 안 지움). 사용자 확인 대기.
- (2026-10-02) 2 번 **확정** — 사과의 carve → attach 전환은 latch HELD, label 은 attention target 의 carve 만. → 구현 task [`SUBTASK-c.task.md`](SUBTASK-c.task.md) (implementer 착수).
  구현 = "확정 label 이 place/home 이면 TargetConfirm 이 새 target 을 채택 · 교체하지 않는다", `clustering.subtask_gate` 기본 off.

## 10. SUBTASK-c Part A 완료 · 남은 spec 빈칸 하나 (2026-10-02)

- Part A: [`SUBTASK-c.impl.md`](SUBTASK-c.impl.md). probe asset parity val 0.980777 (SUBTASK-b 0.9808, 차 2.3e-5) · 새 AttentionSampler 경로 kv_L4 상대오차 2.03e-8 ·
  회귀 1772 passed / 2 skipped / 0 failed. gate 기본 off. Part B (safe_policy 배선 · serve_safe flag) 는 `T39.impl.md` 대기.
- **PLACED 뒤 carve — 기록 (T34 · T37 E3a/E3b, server_constraints, 읽기만):**

| PLACED 뒤 chunk | exclusion active | 누가 carve 되나 | gate 가 막나 |
|---|---|---|---|
| switch 전 (기존 manipulated = 놓인 사과) | **129 / 129** | 놓인 사과 (원래 id) | **아니오** — gate 는 새 채택 · 교체만 막는다 |
| switch 뒤 (12 / 14 episode) | 406 / 433 | 새 id: 놓인 사과 8 · 탁자 위 다른 과일 4 | 예 |

  (switch 뒤 숫자는 implementer `scan_placed_switch.py`, switch 전 숫자는 lead `outputs/verify/subtask_probe/placed_excl.py`.)
- 빈칸: spec "home 이면 지우지 않는다" 를 다 지키려면 **"home 확정 + 이번 episode 에서 detach 가 있었음 → 기존 manipulated 해제 (carve 없음)"** 가 하나 더 필요. 사용자 판정 대기.
- (2026-10-02) **사용자 판정: 안 (b)** — "사과가 놓이기 전에 home 이 나올 수도 있는 상황에 대비". 규칙: gate on ∧ 확정 home ∧ latch 가 이번 episode 에 **PLACED** 도달 → 기존 manipulated 해제 (carve 없음).
  "detach" 대신 PLACED 로 좁힌 이유: T34 attach_revoked (거짓 attach 회수) 도 detach 지만 사과는 놓이지 않았다. → `SUBTASK-c.task.md` Part B · B3.

## 11. SUBTASK-d 해석 정정 (2026-10-03, 사용자 지적)

lead 가 "문제 2 — PLACED 전 LATCHED 에서 label place/home 으로 switch 를 막아 ep1828 을 잃었다" 고 보고했으나 **틀렸다.**
기록 (server_8232 · T39 server_8226, chunk 14–35): ep1828 s18281 은 chunk 14–27 CLOSING (파지 확인 안 됨) → 28 LATCHED, label 은 16 부터 place.
chunk 30 에 gate off 는 manipulated 를 사과 (id 0, ≈(0.56, 0.32)) 에서 **id 1 (0.464, −0.299) — 탁자 반대편, 사과에서 약 0.6 m 떨어진 다른 물체** 로 바꿨고,
gate on 은 그 교체를 막아 사과를 계속 carve 했다. 사용자 원칙 ("쥐기 전 + place 면 사과는 이미 지워져 있고, 바구니는 안 지운다") 그대로의 동작이다.
불일치 1 쌍은 그 뒤 재파지 결과가 갈린 것 (McNemar p 1.0). → 문제 2 와 "HELD 이후에만 막기" 제안은 철회. 남은 수정은 문제 1 (해제 뒤 uncertified HOLD) 하나.
