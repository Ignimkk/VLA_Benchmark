# P1 결과 — CBF-QP 단위 검증

정책도 시뮬레이터도 쓰지 않는 순수 수치 검증. [06-repro-plan.md](06-repro-plan.md) §8.2의 항목을
모두 다룬다.

**결론: 18/18 PASS.** Eq. (5)–(12)가 논문 서술대로 구현·검증되었고, 분석 단계에서 [DERIVED]로만
주장했던 세 가지가 수치로 확인되었다.

| 항목 | 내용 |
|---|---|
| 구현 | [cbf/ellipsoid.py](../cbf/ellipsoid.py) (Eq. 5–11), [cbf/filter.py](../cbf/filter.py) (Eq. 7, 8, 12) |
| 테스트 | [tests/test_cbf.py](../tests/test_cbf.py) |
| 솔버 | OSQP 1.1.3 |
| 실행 | `src/openpi/.venv/bin/python -m pytest benchmark/knows_vla/tests/test_cbf.py -q` |

---

## 1. 검증 결과

| 테스트 | 확인 내용 | 결과 |
|---|---|---|
| 지지함수 (Eq. 5) | $\max_{y\in\mathcal{E}} n^\top y = n^\top c + \sqrt{n^\top Qn}$, 표면 2만 점 브루트포스 대조 | PASS |
| barrier 부호 (Eq. 6) | 최적 $n$에서 $h>0 \iff$ 서로소 | PASS |
| barrier 정확값 | 구 두 개에서 $h = \|c_R-c_O\| - r_R - r_O$ (abs 1e-9) | PASS |
| **$\gamma$ 소거의 성격** | $h(n)\ge0 \Rightarrow \exists\gamma: h_R\ge0 \wedge h_O\ge0$ | PASS |
| $\nabla_{c_R}h$ (Eq. 9) | 유한차분 대조 | PASS |
| **$\nabla_{R_R}h$ (Eq. 10)** | $Q_R \to RQ_RR^\top$ 하에서 유한차분 대조 | PASS |
| $\nabla_n h$ (Eq. 11) | 유한차분 대조 | PASS |
| 구형 $Q_R$ 특수케이스 | $n \times Q_Rn = 0$ → 회전 gradient 항등적 0 | PASS |
| QP 항등성 | 장애물 0개 → $\delta c = \delta c^{\mathrm{nom}}$ 정확히 | PASS |
| 원거리 장애물 | 제약이 이미 느슨 → 명목값 불변 | PASS |
| 충돌 코스 편향 | 접근 성분 감소, $n$ 방향 성분 증가 | PASS |
| 제약 만족 | 해가 Eq. (8)을 모든 장애물에 대해 만족 | PASS |
| 침범 회복 | $h<0$에서 명목이 "정지"여도 바깥으로 밂 | PASS |
| 비가능 폴백 | zero delta 반환, 예외 없음 | PASS |
| **$\epsilon$ 단조 완화** | $\epsilon\uparrow$ → 전진량 단조 증가 | PASS |
| $\gamma_h$ 보수성 | $\gamma_h\uparrow$ → 전진량 단조 증가 | PASS |
| 가상 normal 상태 | 스텝 간 유지, 항상 단위 노름 | PASS |
| 지연 | 아래 §3 | PASS |

## 2. 분석 단계의 [DERIVED] 주장 세 가지가 확인되었다

### 2.1 Eq. (10)이 전제하는 회전 규약을 확정했다 — OPEN-Q 2 부분 해소

[03-math.md](03-math.md)에서 Eq. (10)이 "특정 회전 매개변수화를 전제한다"고만 적었던 부분을
해석적·수치적으로 확정했다. $Q_R \to RQ_RR^\top$이고 $R \approx I + [w]_\times$일 때

$$d(n^\top Qn) = 2\,w\cdot(Qn \times n) \;\Rightarrow\;
d\!\left(-\sqrt{n^\top Qn}\right) = \frac{w\cdot(n \times Qn)}{\sqrt{n^\top Qn}}$$

이것이 정확히 Eq. (10)이다. 즉 논문의 $\delta\theta$는 **world 프레임 축각 증분**이고,
엔드이펙터 타원체는 그리퍼와 강체로 함께 회전한다. **논문은 둘 다 명시하지 않는다.**
`test_grad_rotation_matches_finite_difference_under_R_Q_Rt`가 유한차분으로 확인한다.

남은 부분: LIBERO/robosuite OSC_POSE가 실제로 이 규약으로 delta를 해석하는지는 P2(오프라인
통합)에서 확인해야 한다. 회전 규약이 어긋나면 필터가 잘못된 축으로 밀 수 있다.

### 2.2 $\gamma$ 소거는 충돌 없음 인증을 약화시키지 않는다

논문 부록 §7.1은 Eq. (6)을 "formal collision-free certificate가 아닌 practical safety margin"
이라고 낮춰 말한다. 무작위 타원체 쌍에서 $h(n)>0$인 모든 경우에 대해

$$\gamma \in [\,n^\top c_O + \sqrt{n^\top Q_On},\; n^\top c_R - \sqrt{n^\top Q_Rn}\,] \ne \emptyset$$

이고 그 $\gamma$에서 $h_R \ge 0 \wedge h_O \ge 0$이 성립함을 확인했다
(`test_gamma_elimination_is_not_a_relaxation_of_collision_freeness`).

**즉 고정된 단위 $n$에 대해 $h(n)\ge0$은 분리의 충분조건이다.** 집합 포함 관계로는 [36]의
결합 조건을 완화한 것이 맞지만, 잃은 것은 충돌 없음이 아니라 임의의 $\gamma$에 대한 보수성뿐이다.
[03-math.md](03-math.md)의 분석이 유지된다.

### 2.3 $\epsilon$은 안전 제약을 단조 완화한다

$\delta n$이 목적함수에 없으므로 QP는 박스 전체를 제약 완화에 쓴다. 실효 제약은

$$\nabla_{c_R}h_j\cdot\delta c_R + \nabla_{R_R}h_j\cdot\delta\theta
+ \epsilon\|\nabla_{n^{(j)}}h_j\|_1 \ge -\gamma_h h_j$$

$\epsilon \in \{0, 0.02, 0.05, 0.1\}$ 스윕에서 전진량이 **단조 증가**함을 확인했다
(`test_eps_normal_relaxes_the_constraint_monotonically`). 논문은 $\epsilon$을 "초평면 추정을
매끄럽게 유지하는 상한"으로만 설명하고 이 트레이드오프를 언급하지 않는다.

**함의**: $\epsilon$은 [OPEN-Q 1](OPEN-QUESTIONS.md)의 미상 파라미터 중 하나인데, 안전에 직접
영향을 주므로 스윕 시 $\gamma_h$와 **함께** 다뤄야 한다. 둘은 독립 축이 아니다.

## 3. 지연

| 장애물 수 | QP 지연 |
|---:|---:|
| 1 | 0.75 ms |
| 3 | 0.86 ms |
| 6 | 0.77 ms |
| 10 | 0.97 ms |

논문 Table 2는 **11.4 ms**를 보고한다. 우리는 10개 장애물에서도 ~1 ms로 **약 10배 빠르다.**

직접 비교는 불가하다 — 하드웨어가 다르고(우리 CPU vs 논문의 Xeon w5-3425), 논문 수치에 타원체
기하 준비나 상태 갱신이 포함되는지 불명이다. 다만 **QP 자체가 20 Hz 예산의 병목이 아니라는
점은 분명하다.** [04-pipelines.md](04-pipelines.md) §2.2에서 지적한 지연 합산 문제
(wrapper 49.3 ms + 정책 30 ms > 50 ms)는 QP가 아니라 지각(YOLOe 19.3 + 깊이 9.1 ms)에서 온다.

## 4. 솔버 설정을 조였다 (논문 미기재)

OSQP 기본 허용오차(`eps_abs=eps_rel=1e-3`)에서는 Eq. (8)의 primal residual이 약 $4\times10^{-6}$
남았다. 물리적으로는 마이크로미터 수준이라 무해하지만, **안전 필터가 파이프라인에서 가장 느슨한
고리가 되어서는 안 된다.** `eps_abs = eps_rel = 1e-9`, `max_iter = 20000`으로 설정했다.

이 문제 크기에서 비용은 무시할 만하다(§3의 지연은 조인 뒤 측정값이다). 논문은 솔버만 밝히고
설정은 밝히지 않으므로 `[ASSUMPTION]`으로 기록한다.

## 5. 한계

1. **엔드이펙터 타원체 하나만** 다룬다 — 논문 §5의 범위 그대로다. 팔 링크는 미모델링.
2. **$Q_j$ 고정 가정**을 그대로 따랐다. 회전하는 비구형 객체에서는 부정확하다.
3. `optimal_normal`은 테스트 판정용 보조 함수다. 실제 필터는 논문대로 가상 normal을
   $\|\delta n\|_\infty \le \epsilon$로 점진 갱신할 뿐 최적화하지 않는다.
4. **지각·추적(MVEE, centroid, HSV 재연결)은 아직 구현하지 않았다.** P1은 기하와 QP만 다룬다.

## 6. 다음 단계

P0·P1 통과. 남은 것은 [06-repro-plan.md](06-repro-plan.md) 기준:

- **P2 오프라인 통합** — P0b가 수집한 에피소드에 attention 타깃 식별 + CBF-QP를 붙여, 폐루프
  없이 필터가 만드는 액션 변화를 관찰. 여기서 §2.1의 회전 규약을 실제 LIBERO 액션으로 확인한다.
- **지각·추적 모듈** — 깊이 역투영, MVEE 적합, centroid 추적. `depth_full`이 이미 수집돼 있다.
- **[OPEN-Q 18](OPEN-QUESTIONS.md)** — 배치 단계 타깃 식별 열화. P3 이전에 해소.
