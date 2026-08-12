# Phase 8–10 — Minimal Reproduction, Implementation Plan, Verification

---

# Phase 8 — 최소 재현

## 8.1 P0 프로브 — 이 논문 전체가 얹혀 있는 관찰 하나

논문의 나머지 전부(CBF-QP, 추적기, 벤치마크)는 **"attention head가 타깃을 가리킨다"**는
관찰 위에 서 있다. 이것이 우리 체크포인트에서 성립하지 않으면 나머지를 구현할 이유가 없다.
따라서 벤치마크·시뮬레이터·QP 없이 이것만 먼저 검증한다.

> **가설 H1**: frozen π0.5(`pi05_libero`)의 (layer 12, head 3, agent view) attention density
> $d_i$가, 정책이 현재 접근 중인 객체를 다른 객체보다 유의하게 높게 지목한다.

| 항목 | 내용 |
|---|---|
| **환경** | 시뮬레이터 불필요. JAX + 로컬 체크포인트만 |
| **데이터** | [data/policy_records/](../../../data/policy_records/)의 저장 스텝 (출처·형식 먼저 확인 → OPEN-Q 16). 부족하면 LIBERO 데모 에피소드에서 프레임 추출 |
| **모델** | `pi05_libero`, frozen. 학습 없음 |
| **필요 구현** | attention 슬라이스 + Eq. (2)(3) 뿐. **Eq. (4)의 $\delta$, CBF, 추적기 전부 불필요** |
| **객체 마스크** | 이 단계에서는 YOLOe 대신 **수동/GT 마스크**로 대체 — 지각 오차와 신호 품질을 분리하기 위함 |
| **베이스라인** | (a) 같은 레이어의 다른 7개 헤드, (b) layer별 평균, (c) 무작위 선택 |
| **지표** | ① 타깃 적중률(= $\arg\max_i d_i$가 실제 조작 대상), ② $d_{(1)}-d_{(2)}$ gap 분포, ③ layer×head 히트맵(논문 Fig. 2 재현) |
| **PASS 조건** | (layer 12, head 3)의 적중률이 헤드 평균과 무작위를 뚜렷이 상회하고, gap 분포가 $\delta$를 정할 수 있을 만큼 성공/실패 케이스로 분리됨 |
| **FAIL 시** | **여기서 중단.** 헤드 인덱스가 체크포인트마다 다를 가능성이 크므로, 논문 §3.4 절차대로 layer×head 스윕을 우리 체크포인트에서 다시 수행해 우리만의 헤드를 선택한 뒤 재판정 |

**부수 검증** (같은 프로브에서 공짜로 얻음):
- [02-architecture.md](02-architecture.md) §3.4의 슬라이스 인덱스 실측 확인
  (prefix 968, agent view [0:256], `probs` 축 `[B,1,8,T,S]`, $g{=}16$)
- OPEN-Q 10($\bar A_t$ 집계 방식)과 OPEN-Q 11(디노이징 스텝 선택)을 **경험적으로 결정** —
  평균/합/마지막 쿼리, 그리고 Euler 스텝 1..N 각각에 대해 적중률을 재어 가장 좋은 것을 택하고
  그 선택을 `[ASSUMPTION]`으로 기록

로컬 CPU에서 청크당 ~8초이므로 수십~수백 스텝 규모는 로컬에서도 가능하다. H200에서는 즉시.

## 8.2 P1 — CBF-QP 단위 검증 (모델 불필요)

논문 수식만으로 성립하는 순수 수치 검증. 정책과 완전히 독립이므로 P0와 **병렬 진행 가능**.

| 테스트 | 내용 | 기대 |
|---|---|---|
| 지지함수 정합 | 무작위 $Q\succ0$, $n$에 대해 $\max_{y\in\mathcal{E}}n^\top y = n^\top c + \sqrt{n^\top Qn}$ | 수치 최적화와 일치 |
| $h$ 부호 | 분리된/겹친 타원체 쌍 | 분리 시 $h>0$, 겹침 시 $h<0$ (최적 $n$ 기준) |
| Eq. (9)–(11) gradient | 유한차분 대조 | 상대오차 $<10^{-5}$ |
| **구형 $Q_R$ 특수케이스** | $Q_R = rI$ | Eq. (10)의 $n\times Q_Rn = 0$ → 회전 gradient 항등적 0 ([03-math.md](03-math.md)) |
| QP 항등 | 장애물 0개 | $\delta c = \delta c^{\mathrm{nom}}$ 정확히 |
| QP 분리 방향 | 충돌 코스 1개 장애물 | $\delta c$가 명목 대비 장애물 반대 방향 성분을 가짐 |
| 회복 동작 | $h<0$에서 시작 | Eq. (7)이 $\Delta h>0$ 강제 |
| 비가능 폴백 | 상충 제약 | zero delta 반환, 예외 없음 |
| $\epsilon$ 완화 효과 | $\epsilon$ 스윕 | 실효 제약이 $\epsilon\|\nabla_nh\|_1$만큼 느슨해짐 ([03-math.md](03-math.md) Eq. 12 분석) 수치 확인 |
| 지연 | 장애물 수 $\times$ QP 시간 | 논문 11.4 ms와 비교 (하드웨어 상이 감안) |

## 8.3 확장 순서

```text
P0 attention 프로브        ─┐
                            ├─▶ P2 오프라인 통합 (저장 에피소드에 필터 적용, 폐루프 아님)
P1 CBF-QP 단위 검증        ─┘        │
                                     ▼
                            P3 SafeLIBERO Level I/II 폐루프  (vlsa-aegis @57b1aef, H200)
                                     │
                                     ▼
                            P4 Level III 자체 재구현 + 평가
```

**P3에서 반드시 지킬 것** (`reproduction.md` §4 "베이스라인 먼저"):
No CBF → Naive → KNOWS 순으로 돌리고, **먼저 No CBF가 논문 Table 1의 No CBF 열과
맞는지 확인**한다. 여기서 어긋나면 환경·지표 문제이지 KNOWS 문제가 아니다.

**추가 조건 제안** [Proposed improvement, 논문에 없음]: Naive는 *init-only* + *단일 장애물*
두 가지가 동시에 다르다([01-problem.md](01-problem.md) §5). 기여를 분해하려면
**"다중 장애물 + init-only"** 조건을 추가해야 한다. 이것은 논문 재현이 아니라 우리 확장이므로
분리해 표기한다.

**P4 Level III 재구현 사양** — 논문에 있는 것만 사용:
- 두 waypoint 사이 직선 이동, 30 제어 스텝(20 Hz에서 1.5 s), 이후 정지 [PAPER] §4.1
- waypoint 선정 규칙, 장애물 개수, 시작 타이밍 **전부 미기재** → `[ASSUMPTION]`, 반드시 문서화

---

# Phase 9 — 구현 계획

## 9.1 배치 원칙

새 트리를 만들지 않는다. attention 캡처는 SEAM의 기존 주입 지점을 확장하고, CBF는 이미
비어 있는 `DecodedChunkRefiner` 슬롯에 들어간다. KNOWS 고유 로직만 새 패키지로 분리한다.

```text
benchmark/
├── seam_vla/
│   ├── integration/openpi_jax.py     ← attention 반환 추가 (기존 guidance_fn 패턴 확장)
│   ├── refinement/                   ← KnowsCbfRefiner 추가 (DecodedChunkRefiner 구현)
│   └── metrics/                      ← SR/CR/SSR 추가
└── knows_vla/                        ← 신규. KNOWS 고유 로직
    ├── attention/                    슬라이스 + 집계 (Eq. 2 전처리)
    ├── target/                       Eq. (2)(3)(4)
    ├── perception/                   MVEE, centroid 추적, HSV 재연결
    ├── cbf/                          Eq. (5)-(12), OSQP 래퍼
    ├── configs/                      하이퍼파라미터 스윕 정의
    └── experiments/                  P0 프로브, P1 단위, P3/P4 평가
```

## 9.2 모듈 표

| Module | Responsibility | Inputs | Outputs | Depends On | Test | 성격 |
|---|---|---|---|---|---|---|
| `integration` 확장 | jitted sampler가 layer 12 `probs` 슬라이스를 함께 반환 | observation | 액션 청크 + `[H,256]` | openpi | 실측 shape 검증 | **Layer B** |
| `attention` | $H$ 쿼리 집계 → $\bar A_t$ $[16,16]$ | `[H,256]` | $[g,g]$ | integration | 결정론성, 합=1 여부 | **Layer B** |
| `target` | Eq. (2)(3)(4) | $\bar A_t$, 마스크 | $\tau_t$ 또는 $\varnothing$ | attention | 합성 마스크로 mass 검산 | **Layer A** |
| `perception` | MVEE 적합($t{=}0$), centroid 갱신, freeze, HSV 재연결 | RGB, depth, intrinsics | $\{(p_j,Q_j)\}$ | — | 합성 포인트클라우드 | **Layer A** |
| `cbf` | Eq. (5)–(12), OSQP, 폴백 | 명목 delta, 타원체, $n$ | $\hat a_t$ | perception | P1 전체 | **Layer A** |
| `KnowsCbfRefiner` | 위를 `DecodedChunkRefiner`로 묶음 | 디코딩된 청크 | 안전 청크 | target, cbf | 통합 | **Layer A** |
| `metrics` | SR / CR(에피소드 이진) / SSR | 롤아웃 로그 | 지표 | — | 정의 대조 | **Layer A** |
| RB-Y1 설정 계층 | 관절/프레임/충돌 기하 | — | 설정 | 전부 | 후속 | **Layer B** |

Layer A/B 구분은 `reproduction.md` §11 — A는 논문과 동등해야 하는 과학적 핵심, B는 우리
임베디먼트에 맞추기 위한 통합 코드.

## 9.3 의존 순서

```text
1. integration 확장  ─┐
2. attention          ├─▶ 3. target ─┐
                      │              ├─▶ 6. KnowsCbfRefiner ─▶ 7. metrics ─▶ 8. 평가
4. perception  ───────┴─▶ 5. cbf ────┘
```

1–3은 P0 프로브에 필요하고 4–5는 P1에 필요하므로 **두 갈래를 병렬 진행**할 수 있다.
6 이후는 P0/P1이 모두 PASS한 뒤에만 착수한다.

## 9.4 SEAM과의 관계

`KnowsCbfRefiner`는 SEAM의 VLS와 **직교**한다 — VLS는 디노이징 ODE 내부, KNOWS는 디코딩
이후다([01-problem.md](01-problem.md) §9). 따라서 동시 적용이 구조적으로 가능하다.
다만 **두 방법의 상호작용은 어느 논문에도 없다**. 조합은 "Proposed improvement"로 분류하고
KNOWS 단독 재현이 끝난 뒤에만 다룬다.

`integration/openpi_jax.py`를 수정할 때 SEAM의 기존 보증(guidance_fn=None이면 baseline과
byte-for-byte 동일)을 깨지 않아야 한다 — attention 반환은 **기본 비활성**이어야 한다.

---

# Phase 10 — 검증 계획 (실행은 이후)

## 10.1 목표별 판정

| 대상 | 논문 값 | 판정 가능성 | 근거 |
|---|---|---|---|
| Fig. 2 layer×head 분포 | layer 12 최고, layer 8 차순 | **PARTIAL 기대** | 절차는 재현 가능하나 논문은 GT 타원체 안전필터로 프로파일링했고 우리는 그 조건을 갖추기 어려움 |
| Table 1 No CBF | suite별 SR/CR/SSR | **PASS 기대** | 필터 없는 π0.5 — 체크포인트 차이만 남음 |
| Table 1 Naive | — | PARTIAL | 대리 구현이라 저자 구현과 동일 보장 없음 |
| Table 1 KNOWS Level I/II | — | **PARTIAL 기대** | 미상 하이퍼파라미터 5개 → 절대 수치 일치 기대 불가. **경향**(KNOWS ≈ Naive) 판정이 현실적 목표 |
| Table 1 KNOWS Level III | Naive 대비 CR 43.9pt 감소 | PARTIAL | 벤치마크 자체가 재구현이므로 절대 비교 불가. **부호와 크기 순서**만 판정 |
| Table 2 지연 | wrapper 49.3 ms | **UNVERIFIABLE** | 하드웨어·추출 경로 상이 |
| Fig. 3 성공 예측 AUC | early-window density 0.89 | PASS/FAIL 판정 가능 | P0 프로브의 자연스러운 확장. **논문에서 가장 독립적으로 검증 가능한 주장** |

## 10.2 판정 기준을 미리 고정할 것

- **CR은 에피소드 단위 이진 플래그** (접촉 횟수 아님) — [01-problem.md](01-problem.md) §8
- **"43%"는 절대 퍼센트포인트** — [00-evidence.md](00-evidence.md) §3.
  PASS 조건을 상대 감소로 쓰면 다른 결론이 나온다
- **SSR이 주 지표**
- 확률적 요소(정책 샘플링, 장애물 배치)가 있으므로 시드 고정 + 시행 수·표준편차 보고
  (`reproduction.md` §5). 논문은 태스크당 50 에피소드

## 10.3 불일치 시 디버깅 순서

`reproduction.md` §9를 따른다. 하이퍼파라미터 튜닝을 **먼저 하지 않는다**:

1. 입출력 의미 — 액션 프레임/회전 표현 (OPEN-Q 2)
2. 정규화 — QP가 역정규화 이후인가 ([04-pipelines.md](04-pipelines.md) §3)
3. 좌표 프레임 — 타원체가 world인가 base인가, EEF 자세 출처
4. attention 슬라이스 — 헤드/레이어/뷰 인덱스, 집계 방식
5. 체크포인트 — $H$ 불일치의 영향
6. 타이밍 — $K_{\mathrm{exec}}$, attention 신선도
7. 목적/제약 부호 — Eq. (7)의 $-\gamma_h h$ 방향, Eq. (10) 외적 축
8. 솔버 설정 — OSQP 허용오차, warm start
9. 지표 정의 — CR 이진 여부
10. 수치 정밀도

## 10.4 재현 매니페스트 (실행 전 채울 것)

| Category | Paper | Reproduction | Match? |
|---|---|---|---|
| 논문 버전 | arXiv:2606.09749v1 | 동일 | O |
| 코드 리비전 | **없음** | — | — |
| 벤치마크 리비전 | 미기재 | vlsa-aegis @57b1aef | ? |
| 정책 체크포인트 | 미기재 ($H{=}8$) | `pi05_libero` ($H{=}10$) | **X** |
| 세그멘테이션 | YOLOe 파인튜닝(미공개) | 미정 | **X** |
| $K,\delta,\gamma_h,W,\epsilon$ | 미기재 | 스윕 | **X** |
| 하드웨어 | RTX PRO 6000 ×2 | H200 | X (허용) |
| 평가 프로토콜 | 50 ep/task, 300·550 스텝 | 동일 목표 | ? |
