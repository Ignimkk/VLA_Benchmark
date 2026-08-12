# Phase 2–3 — Architecture Reconstruction & I/O Contract

## 1. 모듈 표

| Module | Input | Output | Dimensions | Trainable? | Train | Inference | Evidence |
|---|---|---|---|---:|---:|---:|---|
| Frozen VLA $\pi_\theta$ (π0.5) | $o_t$, $\ell$ | 액션 청크 $a_{t:t+H}$ + 내부 attention | 액션 $H \times 7$ (LIBERO 유효), attention `[B,K,G,T,S]` | **No** (frozen) | — | O | [PAPER] §3.1 |
| Attention 추출 | 선택 layer/head의 attention | $A_t \in \mathbb{R}^{g\times g}$ | $g{=}16$, $g^2{=}256$ | 비학습 | — | O | [PAPER] §3.3, 부록 §7.2 / $g$는 [DERIVED] |
| 지각: 세그멘테이션 (YOLOe) | agent RGB | 객체별 이진 마스크 | $N$개 마스크 | **Yes** (파인튜닝됨) | O | O | [PAPER] §4.1 |
| 지각: 타원체 적합 (MVEE) | 마스크 + 깊이 + intrinsic/extrinsic | $\mathcal{E}_j = (p_j, Q_j)$ | $p\in\mathbb{R}^3$, $Q\in\mathbb{R}^{3\times3}$ | 비학습 | — | **$t{=}0$만** | [PAPER] §3.2 |
| 지각: 추적기 | 마스크 + 이전 트랙 | 갱신된 centroid $p_{i,t}$ | $\mathbb{R}^3$ | 비학습 | — | O (매 스텝) | [PAPER] §3.2 |
| 타깃 식별 (Eq. 2–4) | $A_t$, 마스크 $M_i$ | $\tau_t \in \mathcal{O}\cup\{\varnothing\}$ | 스칼라 인덱스 | 비학습 | — | O | [PAPER] §3.3 |
| CBF-QP 필터 (Eq. 5–12) | $a_t$, $\mathcal{E}_R$, $\{\mathcal{E}_j\}_{j\in\mathcal{O}^{\mathrm{obs}}_t}$, $\{n^{(j)}\}$ | $\hat a_t$ | 변수 $6 + 3|\mathcal{O}^{\mathrm{obs}}|$ | 비학습 | — | O | [PAPER] 부록 §7.1 |
| OSC 컨트롤러 | $\hat a_t$ | 관절 명령 | — | 비학습 | — | O (블랙박스) | [PAPER] §5 |

**학습 가능한 모듈은 YOLOe 하나뿐이다.** "training-free"는 VLA 정책에 대한 주장이지 시스템
전체에 대한 주장이 아니다. → [01-problem.md](01-problem.md) §6

## 2. 파이프라인

```text
                        ┌─────────────── 에피소드 초기화 (t=0)에만 ───────────────┐
                        │  YOLOe 세그멘테이션 → 깊이 역투영 → 멀티뷰 융합         │
                        │  → MVEE 적합 → 형상행렬 Q_{1..N} 고정                  │
                        └───────────────────────┬────────────────────────────────┘
                                                │  Q_j (고정)
  o_t (agent RGB, wrist RGB, state)             │
  ℓ (언어 지시)                                  │
        │                                       │
        ├──────────────┐                        │
        ↓              ↓                        ↓
 ┌──────────────┐  ┌────────────────┐   ┌──────────────────┐
 │ Frozen π0.5  │  │ YOLOe (매 스텝)│   │ 추적기 (매 스텝) │
 │ (블랙박스)   │  │  마스크 M_i    │──▶│ centroid p_i 갱신│
 └──────┬───────┘  └───────┬────────┘   │ 가림→freeze      │
        │                  │            │ HSV Bhattacharyya│
        │                  │            │ →identity 복구   │
   ┌────┴─────┐            │            └────────┬─────────┘
   ↓          ↓            │                     │
 a_{t:t+H}  attention      │                     │ E_j = (p_j, Q_j)
 (명목)     layer12/head3  │                     │
   │          │            │                     │
   │          └────┬───────┘                     │
   │               ↓                             │
   │      ┌──────────────────────┐               │
   │      │ 타깃 식별 Eq.(2)-(4) │               │
   │      │ mass → density → gap │               │
   │      └──────────┬───────────┘               │
   │                 │ τ_t (또는 ∅)              │
   │                 ↓                           │
   │      O^obs_t = O \ {τ_t}   ◀────────────────┘
   │      (τ_t = ∅ 이면 O^obs_t = O, 즉 전부 장애물)
   │                 │
   ↓                 ↓
 ┌───────────────────────────────┐
 │ CBF-QP  Eq.(5)-(12) / OSQP    │  가상 normal n^(j) 를
 │ min ‖δc−δc^nom‖² + W‖δθ−δθ^nom‖²│ 결정변수로 함께 최적화
 │ s.t. 장애물당 선형 CBF 제약 1개│  (스텝 간 warm start)
 └───────────────┬───────────────┘
                 │ 비가능 → emergency stop (zero delta)
                 ↓
              â_t (+ gripper 그대로 통과)
                 ↓
           OSC 컨트롤러 (블랙박스)
                 ↓
              관절 명령
```

[PAPER] §3, 부록 §7.1

## 3. I/O 계약

### 3.1 액션

| Symbol | Meaning | Shape | Unit / Frame | Normalization |
|---|---|---:|---|---|
| $a_t$ | 정책 명목 액션 | 7 (LIBERO 유효) | 아래 참조 | 정책 출력은 정규화 공간 |
| $(\Delta x, \Delta\theta)$ | EEF delta pose | $\mathbb{R}^6$ | m, rad | [PAPER] §3.1 |
| gripper | 그리퍼 명령 | 1 | — | QP 통과 [PAPER] 부록 §7.1 |
| $(\delta c_R^{\mathrm{nom}}, \delta\theta^{\mathrm{nom}})$ | **물리 단위로 스케일된** 명목 delta | $\mathbb{R}^3 + \mathbb{R}^3$ | m, rad | **역정규화 완료 상태** [PAPER] 부록 §7.1 |
| $\hat a_t$ | 필터링된 액션 | 7 | 동일 | — |
| $A_t^{\mathrm{chunk}}$ | 액션 청크 | $H \times 32$ (모델) / $H \times 7$ (유효) | — | [CODE] `action_dim=32`, `LiberoOutputs`가 `[...,:7]` |

**프레임은 논문에 명시되지 않았다.** LIBERO/robosuite의 OSC_POSE 컨트롤러 규약상
delta position은 base(world) 프레임, delta orientation은 축각(axis-angle) 3벡터이다
[DERIVED, robosuite 규약]. 부록 §7.1이 $\delta\theta \in \mathbb{R}^3$라고만 하므로 축각으로
읽는 것이 정합적이지만, **논문이 이를 진술하지 않았다** → OPEN-Q 2.

이 구분은 중요하다. Eq. (10)의 회전 gradient
$\nabla_{R_R}h_j = (n \times Q_R n)/\sqrt{n^\top Q_R n}$는 **body-frame 각속도 형태의 증분**을
가정한다. 프레임을 틀리면 부호/축이 뒤바뀌어 필터가 충돌 방향으로 밀 수 있다.

### 3.2 상태 · 기하

| Symbol | Meaning | Shape | Unit / Frame |
|---|---|---:|---|
| $c_R$ | EEF 타원체 중심 | $\mathbb{R}^3$ | m, world |
| $R_R$ | EEF 회전 | $SO(3)$ | world |
| $Q_R$ | EEF 타원체 형상행렬 | $3\times3$ SPD | 반축은 **오프라인 캘리브레이션** [PAPER] §3.1 |
| $c_j, Q_j$ | 장애물 $j$의 중심·형상 | $\mathbb{R}^3$, $3\times3$ | $Q_j$는 $t{=}0$ 고정 [PAPER] §3.2 |
| $n^{(j)}$ | 가상 분리 초평면 normal | $\mathbb{R}^3$, $\|n\|{=}1$ | world. **QP 결정변수이자 스텝 간 유지되는 가상 상태** |
| $\gamma$ | 초평면 offset | 스칼라 | Eq. (6)에서 **소거됨** |

### 3.3 Attention

| Symbol | Meaning | Shape | 비고 |
|---|---|---:|---|
| $A_t$ | agent view attention 그리드 | $g\times g = 16\times16$ | [PAPER] $\mathbb{R}^{g\times g}$ / $g$는 [DERIVED] |
| $\bar A_t[r,c]$ | 패치 $(r,c)$의 attention | 스칼라 | Eq. (2). 바(bar)는 정규화/집계를 시사하나 정의 없음 → OPEN-Q 10 |
| $M_i$ | 객체 $i$의 이미지 평면 마스크 | $H_{\mathrm{img}}\times W_{\mathrm{img}}$ | 타원체 볼록껍질의 래스터화 [PAPER] §3.3 |
| $\alpha_{i,t}$ | 비가림 픽셀 면적 $|M_i|$ | 스칼라 | [PAPER] §3.3 |
| $c_i(r,c)$ | 패치 $(r,c)$의 객체 $i$ 커버리지 비율 | $[0,1]$ | Eq. (2) |
| $m_{i,t}$ | 객체 $i$의 attention mass | 스칼라 | Eq. (2) |
| $d_i$ | attention density | 스칼라 | Eq. (3) |

### 3.4 우리 스택에서의 텐서 위치 (실측 근거)

π0.5 + LIBERO 구성에서 [CODE]:

| 양 | 값 | 근거 |
|---|---:|---|
| 카메라당 vision token | 256 | So400m/**14**, 224² → $(224/14)^2$ |
| $g$ | 16 | 동일 |
| prefix 이미지 슬롯 | 3 (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`) | [libero_policy.py:57](../../../src/openpi/src/openpi/policies/libero_policy.py#L57) |
| prefix vision token 총합 | 768 | $3\times256$. `right_wrist_0_rgb`는 zero + `image_mask=False`지만 **토큰은 시퀀스에 존재**하고 attention mask로만 배제 [pi0.py:114-126](../../../src/openpi/src/openpi/models/pi0.py#L114) |
| 언어 토큰 | 200 | `max_token_len = 200 if pi05` [pi0_config.py:39](../../../src/openpi/src/openpi/models/pi0_config.py#L39) |
| prefix 총 길이 | **968** | $768 + 200$ |
| suffix 길이 | $H$ = **10** | pi05는 state 토큰 없음 (`if not self.pi05`) [pi0.py:152](../../../src/openpi/src/openpi/models/pi0.py#L152) |
| `probs` 축 | `[B, K=1, G=8, T, S]` | [gemma.py:217-228](../../../src/openpi/src/openpi/models/gemma.py#L217) |
| 디노이징 스텝의 $T$, $S$ | $T = H = 10$, $S = 968 + 10 = 978$ | suffix가 query, prefix는 KV 캐시 |

**agent view 키 블록 = 인덱스 [0:256]** → 추출 슬라이스:

```python
# layer 12, head 3, action-query × agentview-key
A_raw = probs[0, 0, 3, :, 0:256]      # [H, 256]
A_t   = aggregate_over_H(A_raw).reshape(16, 16)   # [g, g]
```

이 인덱스들은 Phase 8 프로브에서 실측 검증한다.

## 4. 재구성에서 드러난 명세 공백

논문에서 복구되지 않아 구현자가 결정해야 하는 항목:

**(a) $H$개 액션 쿼리를 어떻게 하나의 그리드로 집계하는가.** 부록 §7.2는 블록이
$H \times g^2$라고 하고 본문은 $A_t \in \mathbb{R}^{g\times g}$라고 한다. 평균인지 합인지
마지막 쿼리만 쓰는지 명시가 없다 → OPEN-Q 10.

**(b) 어느 디노이징 스텝의 attention인가.** π0.5는 청크 하나를 만들 때 $N$번(openpi 기본 10회)
Euler 스텝을 돌고 매 스텝 suffix forward pass가 있다 → 청크당 attention 그리드가 $N$개 나온다.
논문은 "During each policy query we obtain an attention grid"라고 단수로만 말한다 → OPEN-Q 11.
초기 스텝(노이즈가 큰 시점)과 마지막 스텝의 attention은 다를 수 있으므로 실질적 영향이 있다.

**(c) "매 스텝"의 의미.** 타깃 식별은 매 제어 스텝(9.4 ms)이지만 정책은 $K$ 스텝마다 한 번만
돈다. 따라서 attention 그리드는 $K$ 스텝 동안 고정이고 마스크 $M_i$만 갱신된다 → $m_{i,t}$는
매 스텝 바뀐다 [DERIVED]. Eq. (3)의 "last $K$ frames"가 제어 스텝인지 정책 질의인지 불명이며,
논문이 실행 horizon과 sliding window에 **같은 기호 $K$**를 쓰고 있어 더 혼동된다 → OPEN-Q 12.

**(d) 멀티뷰 융합.** §3.2는 "fuse it across the available camera views into a single point
cloud"라고 하나, LIBERO에서 깊이가 있는 뷰가 몇 개인지·외부 파라미터를 어떻게 얻는지 미기재.

**(e) 정규화 경계.** 부록 §7.1은 QP 입력이 "scaled to physical units"라고만 한다. openpi에서
역정규화는 `Unnormalize` 변환에서 일어나므로 QP는 그 **이후**에 놓여야 한다 [DERIVED].
[normalization_and_shapes.md](../../seam_vla/docs/normalization_and_shapes.md)의
SEAM 분석과 동일한 경계다.
