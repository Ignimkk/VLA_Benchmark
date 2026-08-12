# Phase 6 — Training / Inference Pipelines

## 1. 학습 파이프라인 — VLA에 대해서는 **존재하지 않음**

이 논문에는 VLA 정책의 학습 파이프라인이 없다. 이것이 결함이 아니라 **핵심 주장**이다.

> "we treat $\pi_\theta$ as a black box; the model itself is neither fine-tuned nor modified"
> [PAPER] §3.1

따라서 다음 항목은 **전부 해당 없음**: 데이터셋 버전, train/val 분할, 배치 크기, 옵티마이저,
학습률, 스케줄러, 스텝 수, weight decay, gradient clipping, LoRA/어댑터, 학습 시드.

### 예외 — 학습되는 유일한 구성요소

| 모듈 | 학습 내용 | 명세 | 가용 |
|---|---|---|---|
| YOLOe [34] | "We finetune YOLOe to segment manipulable objects in the scene" [PAPER] §4.1 | **데이터·에폭·LR·증강 전부 미기재** | 가중치 미공개 |

재현자는 이 파인튜닝을 스스로 해야 하며 논문에는 재현에 필요한 어떤 정보도 없다 → OPEN-Q 9.
런타임 표에 등장하는 구체 변형은 **YOLOe-11m-seg** [PAPER] Table 2.

> **주의**: "training-free"는 VLA 정책에 한정된 주장이다. 시스템 전체는 학습된 세그멘테이션
> 모델에 의존하며, 그 품질이 타원체 기하 → 안전성으로 직결된다(논문 §5 "Perception error").

---

## 2. 추론 파이프라인

```text
관측 (agent RGB, wrist RGB, depth, proprioception) + 언어 지시
   │
   ├─────────────────────────────┬────────────────────────────┐
   ↓                             ↓                            ↓
전처리 (resize 224², pad)   YOLOe 세그멘테이션          proprioception
   ↓                             ↓                       (c_R, R_R)
Frozen π0.5                깊이 역투영 + centroid 갱신        │
 ├── 액션 청크 a_{t:t+H}         ↓                            │
 └── attention (L12/H3)     E_j = (p_j, Q_j)                  │
   ↓                             │                            │
역정규화 (Unnormalize)           │                            │
   ↓                             ↓                            │
물리 단위 (δc^nom, δθ^nom)  타깃 식별 Eq.(2)-(4)              │
   │                             ↓                            │
   │                        O^obs = O \ {τ_t}                 │
   └──────────────┬──────────────┴────────────────────────────┘
                  ↓
          CBF-QP (OSQP), Eq.(12)
                  ↓  비가능 → zero delta
          â_t (+ gripper 통과)
                  ↓
          OSC 컨트롤러 (블랙박스)
                  ↓
          관절 명령 → 로봇
```

### 2.1 타이밍 파라미터

| 항목 | 값 | 증거 |
|---|---|---|
| 제어율 | **20 Hz** (스텝 예산 50 ms) | [PAPER] §4.3, §6 |
| 액션 청크 $H$ | **8** | [PAPER] Table 2 캡션 (243/8 ≈ 30 ms/step) |
| 실행 horizon $K_{\mathrm{exec}}$ | **미기재** | → OPEN-Q 12 |
| 재계획 주기 | $K_{\mathrm{exec}}$ 스텝마다 | [DERIVED] |
| attention 갱신 주기 | 정책 질의 시에만 = $K_{\mathrm{exec}}$ 스텝마다 | [DERIVED] |
| 타깃 식별 주기 | **매 스텝** (마스크가 매 스텝 갱신되므로 $m_{i,t}$도 매 스텝 변함) | [PAPER] §3.3 + [DERIVED] |
| 지각/추적 주기 | 매 스텝 | [PAPER] §3.2 |
| QP 주기 | 매 스텝 | [PAPER] §3.5 |
| 에피소드 스텝 예산 | 300 (SPATIAL/OBJECT/GOAL), 550 (LONG) | [PAPER] §4.1 |
| 에피소드 수 | 태스크당 50 | [PAPER] §4.1 |
| 동적 장애물 속도 | 두 waypoint 사이 직선, **30 제어 스텝 = 1.5 s**, 이후 정지 | [PAPER] §4.1 |
| agent view 렌더 해상도 | $640\times640$ (지연 측정 조건) | [PAPER] §4.3 |

### 2.2 지연 예산 [PAPER] Table 2 (200 제어 스텝 평균)

| 구성요소 | 지연 (ms) |
|---|---:|
| VLA 정책 추론 (청크 1회) | 243 |
| └ attention 추출 | 0.8 |
| YOLOe-11m-seg | 19.3 |
| 깊이 + centroid | 9.1 |
| 타깃 식별 | 9.4 |
| 안전 QP (OSQP) | 11.4 |
| **wrapper 오버헤드 합** | **49.3** |
| 정책 상각분 ($243/8$) | ~30 |
| **스텝 총합** | **~79** |
| 제어 예산 (20 Hz) | 50 |

> **논문 서술과 수치의 불일치** [DERIVED]. §4.3은 "The full wrapper totals 49 ms — within the
> 50 ms budget of LIBERO's 20 Hz control rate — and action chunking ($H{=}8$) amortizes the
> policy forward to ~30 ms/step, so the system holds control rate"라고 적는다. 그러나
> wrapper 49.3 ms와 정책 상각 30 ms는 **같은 스텝에서 순차 실행**되므로 합이 79 ms이고
> 50 ms 예산을 초과한다. 두 항이 병렬(정책 서버 GPU / 클라이언트 CPU 분리 — §4.1에서 실제로
> 그렇게 구성했다고 밝힘)로 겹친다는 전제라야 문장이 성립한다. 논문은 이 겹침을 명시하지
> 않는다 → OPEN-Q 14. 재현 시 **동기 지연과 파이프라인 지연을 구분해 측정**해야 한다
> (SEAM의 [metrics/](../../seam_vla/metrics/)에 이미 synchronized latency 개념이 있음).

### 2.3 병목과 감축 여지

지각(YOLOe 19.3 + 깊이/centroid 9.1 = 28.4 ms)이 wrapper의 58%를 차지한다. 논문은
"더 가벼운 detector, 낮은 입력 해상도, 컴퓨트 최적화로 쉽게 줄일 수 있다"고 한다 [PAPER] §4.3.

attention 추출이 0.8 ms인 것이 이 논문의 핵심 주장이다 — VLM 질의(수백 ms~초 단위)를
**실질적으로 공짜인 신호**로 대체했다는 것. 단 §5.1에서 밝혔듯 우리 스택에서는 이 값이
그대로 적용되지 않는다(openpi는 `probs`를 이미 만들어 두므로 재계산 자체가 없다).

### 2.4 청크 경계 처리 — 미기재

$K_{\mathrm{exec}}$ 스텝 동안 attention 그리드는 고정이고 마스크만 갱신된다. 따라서
청크 경계에서 attention이 급변하면 $\tau_t$가 튈 수 있다. 논문은 다음을 다루지 않는다:

- 타깃 전환 시 히스테리시스 유무
- $\tau_t$가 바뀔 때 해당 객체의 가상 normal $n^{(j)}$ 초기화 방식
- temporal ensembling / 액션 스티칭 유무

Eq. (3)의 $K$ 프레임 누적이 사실상 저역통과 필터 역할을 하지만, 이는 attention 노이즈에
대한 것이지 청크 경계 불연속에 대한 것이 아니다 [DERIVED] → OPEN-Q 15.

---

## 3. 우리 스택으로의 대응

| 논문 단계 | openpi / seam_vla 대응 | 성격 |
|---|---|---|
| 정책 forward | `Pi0.sample_actions` (jitted) | 그대로 |
| attention 추출 | `probs[layer12, head3, :, 0:256]` 직접 슬라이스 | **Layer B 어댑터** — 부록 §7.2 불필요 |
| 역정규화 | openpi `Unnormalize` 변환 | 그대로 |
| CBF-QP | `DecodedChunkRefiner` 구현체 | Layer A |
| 지각/추적 | 신규 (LIBERO는 sim 깊이 사용 가능) | Layer A |
| 컨트롤러 | LIBERO OSC / RB-Y1 | Layer B |

**QP는 반드시 역정규화 이후**에 놓여야 한다 — 부록 §7.1이 "scaled to physical units"라고
명시하고, $h$의 단위가 미터이기 때문이다 [PAPER]+[DERIVED].
