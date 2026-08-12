# Phase 1 — Problem Definition

## 1. 압축 진술

> **주어진** 것: 고정(frozen)된 VLA 정책 $\pi_\theta$, 관측 $o_t$(3인칭 RGB + 손목 RGB +
> proprioception), 언어 지시 $\ell$, 그리고 장면 안의 강체 객체 집합 $\mathcal{O}$.
>
> **계산하는** 것: 로봇에 실제로 실행될 안전한 액션 $\hat a_t$.
>
> **방법**: 정책 자신의 attention(π0.5 layer 12 / head 3 / agent view)에서 매 스텝 현재
> 접근 중인 타깃 $\tau_t$를 읽어 장애물 집합에서 제외하고, 나머지 객체를 타원체로 추적하여
> 이산시간 CBF-QP로 명목 액션 $a_t$를 안전 집합에 투영한다.
>
> **제약**: 엔드이펙터 타원체 $\mathcal{E}_R$가 모든 장애물 타원체 $\mathcal{E}_j$
> ($j \in \mathcal{O}^{\mathrm{obs}}_t$)와 분리 — Eq. (1). 실패 시 emergency stop.
>
> **목적**: 태스크 성공률을 희생하지 않으면서 충돌률을 낮추되, **20 Hz 제어 루프 안에서**
> 동작할 것 — 즉 **동적 장애물에 대응**할 것.

[PAPER] §3, §3.1

## 2. 해결하려는 실패 모드

이 논문이 겨냥하는 것은 "VLA가 안전하지 않다"가 아니라 **기존 안전 필터가 느려서 생기는 실패**다.

기존 추론시 안전 필터([9] VLSA, [10] Brunke et al.)는 어떤 객체가 "타깃"이고 어떤 것이
"장애물"인지를 알아야 한다. 타깃까지 피하면 태스크를 수행할 수 없기 때문이다. 이 판단을
외부 VLM에 묻는데, VLM 질의가 제어 루프에 들어가기엔 너무 느려서 **에피소드 초기화 때 한 번만**
실행된다. [PAPER] §1

결과적으로:
- 장애물이 움직이면 초기 시점의 장애물 위치가 낡는다,
- 로봇이 움직이면서 주변과의 공간 관계가 계속 바뀐다,
- 타깃/장애물 배정 자체가 낡는다(예: 다단계 태스크에서 phase가 바뀌면 타깃도 바뀐다).

논문의 관찰: **이 정보는 이미 정책 안에 있다.** frozen VLA의 소수 attention head가 정책이
지금 향하고 있는 객체를 일관되게 국소화하며, 이는 추가 forward pass 없이 읽을 수 있다.
[PAPER] 초록, §1

## 3. 가정

| 가정 | 근거 | 위험 |
|---|---|---|
| 정책은 블랙박스이고 파인튜닝·수정하지 않는다 | [PAPER] §3.1 "we treat $\pi_\theta$ as a black box; the model itself is neither fine-tuned nor modified" | 없음 — 핵심 주장 |
| 장면은 강체 객체의 집합 $\mathcal{O} = \{1,\dots,N\}$ | [PAPER] §3.1 | 변형체·관절체 미지원 |
| EEF와 모든 객체를 **3D 타원체**로 근사 | [PAPER] §3.1 | 오목 형상 과대근사 → 보수성 |
| 타원체 형상행렬 $Q_{1..N}$은 $t=0$에 고정, 이후 centroid만 갱신 | [PAPER] §3.2 | 회전하는 비구형 객체에서 부정확 |
| EEF 타원체 반축은 오프라인 캘리브레이션 | [PAPER] §3.1 | 절차 미기재 → OPEN-Q 3 |
| 카메라 intrinsic/extrinsic 기지 | [PAPER] §3.2 (깊이 역투영에 사용) | 실로봇 이식 시 캘리브레이션 필요 |
| 깊이 관측 가용 | [PAPER] §3.2 | LIBERO는 sim 깊이. 실로봇은 센서 필요 |
| **EEF만** 보호하고 팔 링크는 모델링하지 않음 | [PAPER] §5 "the rest of the arm is unmodeled" | 상완·팔꿈치 충돌 발생 — 저자 스스로 한계로 명시 |
| 하위 OSC 컨트롤러는 블랙박스, 추종 오차는 미해석 | [PAPER] §5, 부록 §7.1 "reduced-order safety" | ROM 안전이 전체 시스템 안전을 보장하지 않음 |

## 4. 입력 / 출력

**입력** [PAPER] §3.1, §3.2
- $o_t$: 3인칭(agent) RGB, 손목 RGB, proprioceptive state
- $\ell$: 언어 지시
- 깊이 + 카메라 intrinsic/extrinsic (지각 모듈용)
- EEF 자세 $(c_R, R_R)$ — proprioceptive state에서 매 스텝 읽음

**출력**
- $\hat a_t$: 필터링된 EEF delta pose + gripper 명령. gripper는 QP를 **그대로 통과**한다
  [PAPER] 부록 §7.1

**중간 산출**
- $A_t \in \mathbb{R}^{g \times g}$: 단일 layer/head의 action-query × vision-key attention 그리드
- $\tau_t \in \mathcal{O} \cup \{\varnothing\}$: 현재 타깃 (또는 판정 불가)
- $\mathcal{O}^{\mathrm{obs}}_t = \mathcal{O} \setminus \{\tau_t\}$: 장애물 집합
- $\{\mathcal{E}_j\}$: 객체 타원체 (형상 고정, 중심 갱신)
- $n^{(j)}$: 장애물별 가상 분리 초평면 normal (QP의 결정변수 일부)

## 5. 베이스라인 vs 제안 방법

논문은 세 조건을 비교한다 [PAPER] §4.1:

| 조건 | 안전 레이어 | 장애물 추정 | 타깃 식별 | 갱신 주기 |
|---|---|---|---|---|
| **No CBF** | 없음 | — | — | — |
| **Naive** | CBF-QP | 단일 고정 장애물 타원체, **ground-truth 세그멘테이션**으로 $t=0$에 배치 후 **갱신 없음** | 시뮬레이터 특권 상태(oracle), $t=0$ 1회 | 에피소드당 1회 |
| **KNOWS** | CBF-QP | 추적된 **모든** 객체가 후보 장애물, 매 스텝 재국소화 | **정책 attention**, 매 스텝 | 매 제어 스텝 |

중요한 점 두 가지:

1. **Naive는 [9]의 실제 구현이 아니라 저자의 대리 구현이다.** 논문은 "a strong stand-in for
   prior init-only filters"라고 표현한다 [PAPER] §4.1. 게다가 Naive는 **ground-truth
   세그멘테이션과 특권 시뮬레이터 상태**를 쓰므로, 지각 오차가 없는 상한(oracle) 베이스라인이다.
   따라서 "KNOWS가 Naive와 대등하다(Level I/II)"는 것은 *지각 오차를 감수하고도 oracle과
   대등하다*는 더 강한 주장이다. [DERIVED]

2. **Naive는 장애물이 하나뿐이다.** KNOWS는 추적된 모든 비타깃 객체를 장애물로 쓴다. 즉 두
   조건은 타깃 식별 방식만 다른 것이 아니라 **장애물 집합의 크기도 다르다**. 이는 통제되지 않은
   변수이며, `reproduction.md` §8("한 번에 하나의 변수만 바꿀 것")에 어긋난다. 재현 시
   "다중 장애물 + init-only" 조건을 추가하면 기여를 분해할 수 있다. → OPEN-Q 6, 8

## 6. 학습 시점 / 추론 시점 변화

| | 변화 |
|---|---|
| **학습 시점** | **없음.** 정책은 frozen, 파인튜닝·재학습·어댑터 전부 없음. [PAPER] §3.1, 초록 "training-free" |
| **추론 시점** | 전부. attention 추출, 타깃 식별, 객체 추적, CBF-QP 투영이 모두 매 제어 스텝에 추가된다 |

**예외 하나**: 세그멘테이션 모델 YOLOe는 "조작 가능한 객체를 세그멘트하도록 파인튜닝"된다
[PAPER] §4.1. 즉 "training-free"는 **VLA 정책에 한정된 주장**이며, 지각 모듈은 학습된다.
재현자는 이 파인튜닝을 직접 해야 하고 가중치는 공개되지 않았다. → OPEN-Q 9

## 7. 주장하는 기여 (논문 §1)

1. frozen VLA 정책 안에 정책의 현재 타깃을 가리키는 소수의 attention head가 존재함을 식별.
   추가 학습·감독 불필요.
2. 이 head를 경량 객체 추적기와 결합해 매 스텝 타깃/장애물 분해를 유지하고 CBF-QP에 공급하는
   training-free 안전 프레임워크 KNOWS.
3. SafeLIBERO를 동적 장애물로 확장하고, KNOWS가 정적 장면에서는 특권 상태 oracle과 대등하며
   동적 장면에서는 크게 앞선다는 것을 보임.

부수적 발견(§4.4): 같은 attention density가 **에피소드 성공의 추론시 예측자**이기도 하다.
초기 구간(첫 물체를 집기 전 또는 에피소드 1/3 중 이른 쪽)으로 제한하면 AUC 0.70 → 0.89.
타깃이 아닌 목적지(destination)에 대한 density는 AUC 0.55(우연 수준)이므로, 이 신호는
일반적 saliency가 아니라 **의미적으로 현재 명령 객체에 특정적**이다. [PAPER] §4.4

## 8. 평가 지표

[PAPER] §4.1
- **SR** (success rate): 태스크 완료
- **CR** (collision rate): 에피소드가 비타깃 장애물과 접촉했는가 (에피소드 단위 이진)
- **SSR** (safe-success rate): 완료 **그리고** 충돌 없음 — **주 지표**

SSR이 주 지표인 이유는 안전/유능함의 트레이드오프를 함께 포착하기 때문이다. CR이 에피소드당
접촉 횟수가 아니라 이진 플래그라는 점은 재현 시 반드시 맞춰야 한다 (`reproduction.md` §6).

## 9. 이 워크스페이스에서의 위치

KNOWS는 SEAM과 **같은 지점**에 개입한다 — 정책 출력 이후, 컨트롤러 이전. 그러나:

| | SEAM (VLS) | KNOWS |
|---|---|---|
| 개입 위치 | 디노이징 ODE **내부** (post-Euler guidance) | 디코딩된 액션 **이후** (외부 투영) |
| 목적 | 청크 경계 평활도 | 충돌 회피 |
| 정보원 | 이전 청크의 미실행 tail | 정책 attention + 지각 |
| 미분 필요 | 없음 (closed-form) | 없음 (QP는 외부) |

따라서 KNOWS는 [refinement/](../../seam_vla/refinement/)의 `DecodedChunkRefiner`
슬롯에 정확히 대응하며, SEAM과 **동시 적용 가능**하다(VLS가 청크를 매끄럽게 만들고 KNOWS가
그 결과를 안전 집합에 투영). 다만 두 방법의 상호작용은 어느 논문에도 없다 → 향후 과제.
