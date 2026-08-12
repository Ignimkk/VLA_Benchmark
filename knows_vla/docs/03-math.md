# Phase 4–5 — Equation & Algorithm Reconstruction

기호는 논문 표기를 따른다. 모든 결론에 증거 라벨을 붙인다.

---

## Eq. (1) — 안전 목표

$$\mathcal{E}_R \cap \mathcal{E}_j = \emptyset \qquad \forall j \in \mathcal{O}^{\mathrm{obs}}_t$$

**역할**: 필터가 달성하려는 최종 조건. EEF 타원체가 모든 장애물 타원체와 서로소.
**출력**: 명목 $a_t$를 최소한으로 수정한 $\hat a_t$. 존재하지 않으면 emergency stop.
**미분가능성**: 집합 조건 자체는 비미분. Eq. (6)의 $h$로 매끄럽게 대체된다.
[PAPER] §3.1

---

## Eq. (2) — Attention mass

$$m_{i,t} = \sum_{(r,c)} \bar A_t[r,c]\, c_i(r,c), \qquad
c_i(r,c) = \frac{|M_i \cap \mathrm{patch}(r,c)|}{|\mathrm{patch}(r,c)|}$$

**역할**: 객체 $i$가 이번 프레임에서 받은 attention 총량.
**입력**: attention 그리드 $\bar A_t \in \mathbb{R}^{g\times g}$, 객체 마스크 $M_i$.
**출력**: 스칼라.

**설계 의도** [PAPER] §3.3: 마스크가 겹칠 수 있으므로 각 패치의 attention을 **커버리지 비율로
분배**한다. 가려진 객체도 자기 몫을 받으며, 앞의 객체가 패치 전체를 독식하지 않는다.
→ 구현상 $\sum_i c_i(r,c) > 1$일 수 있고, 이는 의도된 것이다. 정규화하면 방법이 바뀐다.

**수치 사항**: $M_i$는 이미지 해상도(예: $640\times640$), 패치 그리드는 $16\times16$.
$c_i$는 마스크를 패치 격자로 **면적 가중 다운샘플**한 것과 동일하다 [DERIVED].

```python
# patch_grid: (g, g) 패치 경계. masks: (N, H_img, W_img) bool
cov = area_pool(masks, g, g)          # (N, g, g), 값 ∈ [0,1] — 패치 내 마스크 면적 비율
m   = (A_t[None] * cov).sum((1, 2))   # (N,)
```

**미분가능성**: $\bar A_t$에 대해 선형(미분가능). $M_i$는 이산 → 비미분. 추론 전용이므로 무관.
**증거**: [PAPER] §3.3

### 미해결: $\bar A_t$의 바(bar)
논문은 $A_t$(부록 §7.2)와 $\bar A_t$(Eq. 2)를 구분해 쓰지만 바의 정의가 없다.
$H$개 액션 쿼리에 대한 평균일 가능성이 가장 높다 [ASSUMPTION] → OPEN-Q 10.

---

## Eq. (3) — Attention density

$$d_i = \Big(\sum_K m_{i,t}\Big)\Big(\sum_K \alpha_{i,t}\Big)^{\beta}, \qquad \beta = -1$$

**역할**: 최근 $K$ 프레임에 걸쳐 누적한 mass를 누적 면적으로 나눈 **단위 면적당 attention**.
$\beta=-1$이므로 실질적으로

$$d_i = \frac{\sum_K m_{i,t}}{\sum_K \alpha_{i,t}}$$

**설계 의도** [PAPER] §3.3: 면적 정규화가 없으면 크거나 가까운 객체가 단지 패치를 많이
차지한다는 이유로 이긴다. 슬라이딩 윈도는 단일 프레임 attention의 노이즈를 억제한다.

**중요**: 이것은 "프레임별 비율의 평균"이 **아니라** "누적합의 비율"이다. 두 값은 다르며,
가림으로 $\alpha_{i,t}$가 작아지는 프레임에서 크게 갈린다. 논문 형태를 그대로 따를 것.

**수치 사항**: $\sum_K \alpha_{i,t} = 0$(윈도 내내 완전 가림 또는 미검출)이면 0으로 나눈다.
논문에 처리 규정 없음 → 구현 시 $d_i = 0$ 또는 후보 제외 필요 [ASSUMPTION].

```python
# 링버퍼 길이 K
d = m_hist.sum(0) / np.maximum(alpha_hist.sum(0), eps)   # (N,)
```

**증거**: [PAPER] §3.3 ($\beta=-1$ 포함). $K$ 값은 **미상** → OPEN-Q 1.

---

## Eq. (4) — 타깃 확정 (gap 판정)

$$\tau_t = \arg\max_i d_i \quad \text{if } d_{(1)} - d_{(2)} \ge \delta, \qquad
\text{else } \tau_t = \varnothing$$

**역할**: 1위가 2위를 $\delta$ 이상 앞설 때만 타깃으로 인정. 아니면 **아무것도 제외하지 않고
장면 전체를 장애물로 취급**한다.

**이것이 이 방법의 안전/유능함 스위치다** [DERIVED]:
- $\delta \uparrow$ → $\varnothing$ 빈발 → 타깃까지 회피 → CR ↓, SR ↓
- $\delta \downarrow$ → 오분류 시 **실제 장애물을 타깃으로 오인해 장애물 집합에서 빼버림** → CR ↑

$\delta$ 값이 논문에 없다는 것은 Table 1의 SR/CR 균형을 재현할 수 없다는 뜻이다 → OPEN-Q 1.

**수치 사항**: $d_{(1)}, d_{(2)}$는 절대 스케일이 attention 분포에 의존하므로 $\delta$는
**스케일 의존적**이다. $\bar A_t$의 정규화 방식(OPEN-Q 10)이 바뀌면 $\delta$도 함께 바뀐다.
따라서 $\delta$를 스윕할 때는 정규화 규약을 먼저 고정해야 한다.

```python
order = np.argsort(d)[::-1]
tau = order[0] if (d[order[0]] - d[order[1]]) >= delta else None
obstacles = [j for j in objects if j != tau]      # tau is None -> 전부 장애물
```

**증거**: [PAPER] §3.3. $\delta$ **미상**.

---

## Eq. (5) — 분리 초평면 CBF (Wu & Liu [36])

$$h_R(n,\gamma) = n^\top c_R - \gamma - \sqrt{n^\top Q_R n}, \qquad
h_O(n,\gamma) = -n^\top c_O + \gamma - \sqrt{n^\top Q_O n}$$

**역할**: 초평면 $\{y : n^\top y = \gamma\}$가 두 타원체를 분리하는지 인증.
$h_R \ge 0$은 $\mathcal{E}_R \subseteq \{n^\top y \ge \gamma\}$, $h_O \ge 0$은
$\mathcal{E}_O \subseteq \{n^\top y \le \gamma\}$를 뜻한다.

$\sqrt{n^\top Q n}$은 타원체의 **지지함수(support function)** 항이다:
$\max_{y\in\mathcal{E}} n^\top y = n^\top c + \sqrt{n^\top Q n}$ [DERIVED].

$(n,\gamma)$는 **가상 상태**로 취급되며 $\|n\|=1$이 항상 강제된다 [PAPER] 부록 §7.1.

---

## Eq. (6) — 결합 barrier

$$h(n) = n^\top(c_R - c_O) - \sqrt{n^\top Q_R n} - \sqrt{n^\top Q_O n}$$

**유도**: $h = h_R + h_O$이며 $\gamma$가 상쇄된다 [DERIVED, 확인함]. 장애물이 비제어
대상이므로 offset을 결정변수로 유지할 이유가 없다.

### 논문의 자기평가는 과도하게 보수적이다

논문은 [PAPER] 부록 §7.1에서 이렇게 적는다:

> "enforcing $h \ge 0$ is a relaxation of the joint condition $h_R \ge 0 \wedge h_O \ge 0$ ...
> We therefore treat eq. (6) as a practical safety margin rather than a formal
> collision-free certificate."

**앞 절은 맞고 뒷 절은 과하다.**

원출처 [36](Wu & Liu, arXiv:2505.20847)을 대조했다 [CODE 아님 — 원논문 확인]:
[36]은 두 barrier를 **개별 제약으로 유지하며 절대 합하지 않고**(그들의 Eq. 27b–c), 초평면
normal $n_{ij}$와 offset $\gamma_{ij}$를 각각 고유 동역학을 갖는 **가상 상태**로 둔다
(그들의 Eq. 19: $\dot n_{ij} = (I_d - n_{ij}n_{ij}^\top)\eta_{ij}$). KNOWS가 $\gamma$를
소거하고 둘을 합친 것은 [36]에 대한 KNOWS 고유의 변경이 맞다.

집합 포함 관계로는 실제로 완화다: $\{h_R\ge0 \wedge h_O\ge0\} \subseteq \{h\ge0\}$.
그러나 **잃어버린 것은 충돌 없음이 아니라 특정 $\gamma$에 대한 보수성뿐이다.**
지지함수 논증으로 보면 **고정된 단위 $n$에 대해 $h(n) \ge 0$은 분리의 충분조건이다**
[DERIVED]:

$$\min_{y\in\mathcal{E}_R} n^\top y - \max_{y\in\mathcal{E}_O} n^\top y
= n^\top(c_R-c_O) - \sqrt{n^\top Q_R n} - \sqrt{n^\top Q_O n} = h(n)$$

$h(n)\ge0$이면 $\gamma \in [\,n^\top c_O + \sqrt{n^\top Q_O n},\; n^\top c_R - \sqrt{n^\top Q_R n}\,]$
가 비어 있지 않으므로 그런 $\gamma$를 고르면 $h_R \ge 0 \wedge h_O \ge 0$이 성립한다.
즉 $h(n) \ge 0 \iff \exists\gamma$ 로 두 조건이 동시 성립. **$\gamma$ 소거는 "최적 $\gamma$를
암묵적으로 선택"하는 것과 같으며, 인증을 약화시키지 않는다.**

오히려 [36]의 방식은 $\gamma$가 나쁘게 배치되면 실제로는 분리되어 있는데도 제약이 걸리는
보수성을 갖는다. KNOWS는 그 보수성을 제거한 것이고, 대신 $\gamma$ 상태를 추적할 필요가 없어
QP 변수가 장애물당 1개 줄어든다 [DERIVED].

**실제로 인증이 깨지는 지점은 따로 있다** [DERIVED]:
1. Eq. (8)에서 $h$를 **선형화**한다 — 1차 근사이므로 실제 $h(t+1)$이 예측보다 작을 수 있다.
2. $n$이 매 스텝 $\|\delta n\|_\infty \le \epsilon$로 **제한적으로만** 갱신된다 — 최적 분리
   방향에 못 미치면 $h$가 실제 여유를 과소평가(보수적)하거나, 급격히 상대 자세가 바뀌면
   부적절한 $n$에 갇힌다.
3. 하위 OSC의 추종 오차(§7.1 reduced-order safety).

→ 재현 시 이 차이를 확인할 것. 논문의 서술이 지나치게 보수적인 것으로 보이나, 우리가 놓친
전제가 있을 수 있으므로 **방법 자체는 논문대로 구현**하고 이 분석은 기록만 남긴다
(`reproduction.md` §3 "Must Match" 원칙).

**수치 사항**: $Q$가 특이(납작한 객체)하면 $\sqrt{n^\top Q n} \to 0$이고 Eq. (10)(11)의 분모가
0에 접근한다. $Q$ 고유값 하한(floor)이 필요 [ASSUMPTION].

---

## Eq. (7) — 이산시간 CBF 조건

$$\Delta h_j \ge -\gamma_h h_j, \qquad \gamma_h \in (0,1]$$

즉 $h_j(t+1) \ge (1-\gamma_h)h_j(t)$ [PAPER].

**해석**:
- $h_j > 0$(안전): $h$가 스텝당 $(1-\gamma_h)$ 비율보다 빨리 줄지 못한다. $\gamma_h \to 0$이면
  거의 감소 불가(매우 보수적), $\gamma_h = 1$이면 $h(t+1)\ge0$만 요구.
- $h_j < 0$(이미 침범): $-\gamma_h h_j > 0$이므로 $\Delta h_j > 0$ — **회복 방향을 강제**한다.
  이 성질 덕분에 지각 오차로 잠시 침범해도 필터가 빠져나온다 [DERIVED].

**$\gamma_h$ 미상** → OPEN-Q 1. 이 값이 보수성을 직접 지배한다.
**증거**: [PAPER] 부록 §7.1, 원출처 [15].

---

## Eq. (8)–(11) — 선형화된 제약과 gradient

$$\nabla_{c_R}h_j \cdot \delta c_R + \nabla_{R_R}h_j\cdot\delta\theta
+ \nabla_{n^{(j)}}h_j\cdot\delta n^{(j)} \ge -\gamma_h h_j \tag{8}$$

$$\nabla_{c_R}h_j = n^{(j)} \tag{9}$$

$$\nabla_{R_R}h_j = \frac{n^{(j)} \times Q_R n^{(j)}}{\sqrt{n^{(j)\top}Q_R n^{(j)}}} \tag{10}$$

$$\nabla_{n^{(j)}}h_j = (c_R - c_j)
- \frac{Q_R n^{(j)}}{\sqrt{n^{(j)\top}Q_R n^{(j)}}}
- \frac{Q_j n^{(j)}}{\sqrt{n^{(j)\top}Q_j n^{(j)}}} \tag{11}$$

**Eq. (9)**: $h$가 $c_R$에 선형이므로 자명. 물리적으로 "초평면 normal 방향으로 EEF를 밀면
여유가 그만큼 늘어난다".

**Eq. (10)**: 회전 gradient는 $Q_R$이 EEF 자세에 의존하기 때문에 생긴다
($Q_R = R_R \Sigma R_R^\top$ 형태). 논문은 [36] eq. 26의 $d{=}3$ 형태라고 밝힌다 [PAPER].

> **중요한 특수 케이스** [DERIVED]: $n$이 $Q_R$의 고유벡터이면 $n \times Q_R n = 0$이다.
> 특히 $Q_R$이 **구형(등방)**이면 모든 방향이 고유벡터이므로 회전 gradient가 항등적으로 0이고,
> QP는 회전으로 안전을 개선할 수 없다. 물리적으로 타당하다(구를 돌려도 형상이 안 변함).
> 구현 검증에 쓸 수 있는 좋은 단위 테스트다.

**Eq. (11)**: $\delta n$이 목적함수에 없다는 점이 핵심이다 → §Eq.(12) 분석 참조.

**미분가능성**: $n \ne 0$이고 $Q \succ 0$인 한 매끄럽다.
**수치 위험**: 세 분모 모두 $\sqrt{n^\top Q n}$. $\|n\|=1$ 유지 + $Q$ 고유값 하한 필수.
$\delta\theta$가 축각 증분이라는 전제(→ [02-architecture.md](02-architecture.md) §3.1)가
틀리면 Eq. (10)의 외적 항이 잘못된 축을 가리킨다.

---

## Eq. (12) — 안전 QP

$$\begin{aligned}
\min_{\delta c_R,\, \delta\theta,\, \{\delta n^{(j)}\}}\quad
& \|\delta c_R - \delta c_R^{\mathrm{nom}}\|^2 + W\|\delta\theta - \delta\theta^{\mathrm{nom}}\|^2\\
\text{s.t.}\quad & \text{Eq. (8)} && \forall j \in \mathcal{O}^{\mathrm{obs}}_t\\
& \|\delta n^{(j)}\|_\infty \le \epsilon && \forall j \in \mathcal{O}^{\mathrm{obs}}_t
\end{aligned}$$

**결정변수**: $\delta c_R \in \mathbb{R}^3$, $\delta\theta \in \mathbb{R}^3$,
$\delta n^{(j)} \in \mathbb{R}^3$ (장애물당) → 총 $6 + 3|\mathcal{O}^{\mathrm{obs}}_t|$.
**제약 수**: 장애물당 부등식 1개 + 박스 $6|\mathcal{O}^{\mathrm{obs}}_t|$.
**볼록성**: 목적은 볼록 이차, 제약은 선형 → 볼록 QP [PAPER]. 솔버 **OSQP** [39].

### 구조적 관찰 — $\epsilon$은 안전 제약을 직접 완화한다 [DERIVED]

$\delta n^{(j)}$는 **목적함수에 등장하지 않고** 오직 자기 장애물의 제약과 박스에만 나타난다.
따라서 QP는 $\delta n$을 제약 만족에 가장 유리하게 자유롭게 고른다. $\delta n$을 소거하면
실효 제약은

$$\nabla_{c_R}h_j\cdot\delta c_R + \nabla_{R_R}h_j\cdot\delta\theta
+ \epsilon\,\|\nabla_{n^{(j)}}h_j\|_1 \ge -\gamma_h h_j$$

이 된다. 즉 **$\epsilon$이 클수록 제약이 $\epsilon\|\nabla_n h\|_1$만큼 느슨해진다.**

$\epsilon$은 논문에서 "초평면 추정을 스텝 간 매끄럽게 유지하는 상한"으로만 설명되지만
[PAPER], 실제로는 **안전성과 초평면 적응 속도를 동시에 지배하는 파라미터**다. 논문은 이
트레이드오프를 논하지 않는다. 값도 미상 → OPEN-Q 1.

### 해 후 처리 [PAPER] 부록 §7.1

1. 가상 normal 재정규화: $n^{(j)} \leftarrow (n^{(j)} + \delta n^{(j)}) / \|n^{(j)} + \delta n^{(j)}\|$
2. $(\delta c_R, \delta\theta)$를 명령 액션에 적용
3. **gripper 명령은 수정 없이 통과**
4. **비가능(infeasible)이면 병진·회전 delta를 0으로 하는 emergency stop**

> **한계** [DERIVED]: emergency stop이 "정지"인 것은 EEF 기준이며, **장애물이 움직여 다가오는
> 경우 정지는 안전하지 않다**. Level III(동적 장애물)에서 KNOWS의 잔여 충돌 일부가 여기서
> 나올 수 있다. 논문 §5의 한계 목록에는 이 항목이 없다.

### 그 밖의 미기재 사항
- $\delta c_R$, $\delta\theta$ 자체의 크기 제한이 **없다**. 목적함수가 명목값에서 멀어지는 것을
  벌하지만 상한은 없으므로, 제약이 강하면 정책의 정상 액션 범위를 넘는 delta가 나올 수 있다.
  OSC/액션 공간 클리핑 여부 미기재 → OPEN-Q 13.
- 초기 $n^{(j)}$: 중심 간 방향 $c_R - c_j$로 초기화 [PAPER] 부록 §7.1. 새 객체가 추적에
  등장할 때도 동일하게 초기화한다고 가정 [ASSUMPTION].

---

## Phase 5 — 알고리즘 재구성

### A. 에피소드 초기화 ($t=0$) [PAPER] §3.2

```text
1. YOLOe(파인튜닝)로 조작 가능 객체별 이진 마스크 생성
2. 마스크 영역의 깊이를 카메라 intrinsic/extrinsic로 3D 역투영
3. 가용한 여러 카메라 뷰를 단일 포인트클라우드로 융합
4. 각 객체에 MVEE(최소부피 외접 타원체) 적합 → E_i = (p_i, Q_i)
5. 형상행렬 Q_{1..N} 을 고정 (이후 절대 재적합하지 않음)
6. 장애물별 가상 normal n^(j) ← normalize(c_R − c_j)
7. attention 링버퍼(길이 K) 초기화
```

### B. 매 제어 스텝 [PAPER] §3.2–3.5, 부록 §7.1

```text
# --- 정책 (K_exec 스텝마다 1회; 그 사이에는 청크에서 꺼내 씀) ---
if 새 청크가 필요:
    a_{t:t+H}, probs ← π_θ(o_t, ℓ)            # frozen, 단일 forward pass
    A_t ← aggregate(probs[layer=12, head=3, action_q, agentview_keys])   # [g,g]
    # 어느 디노이징 스텝의 probs인지 미기재 → OPEN-Q 11

# --- 지각 (매 스텝) ---
masks ← YOLOe(agent_rgb)                       # 지배적 비용: 19.3 ms
for i in objects:
    if 가려짐(팔 개입):  트랙을 마지막 위치로 freeze
    else:                p_i ← centroid(역투영(masks[i]))   # Q_i 는 그대로
    identity swap 의심 시: HSV 색히스토그램 Bhattacharyya 매칭으로 재연결

# --- 타깃 식별 (매 스텝) ---
for i in objects:
    cov     ← area_pool(타원체 볼록껍질 래스터화 마스크, g, g)
    m_i     ← Σ_{r,c} Ā_t[r,c] · cov_i[r,c]                     # Eq. (2)
    α_i     ← |M_i|                                              # 비가림 픽셀 면적
링버퍼에 (m, α) 푸시 (길이 K)
d_i ← Σ_K m_i / Σ_K α_i                                          # Eq. (3), β=−1
τ_t ← argmax d  if  d_(1) − d_(2) ≥ δ  else  ∅                   # Eq. (4)
O_obs ← O \ {τ_t}

# --- 안전 QP (매 스텝) ---
(c_R, R_R) ← proprioception
(δc^nom, δθ^nom) ← 물리 단위로 스케일된 명목 액션        # 역정규화 이후
for j in O_obs:
    h_j   ← n_jᵀ(c_R − c_j) − √(n_jᵀQ_R n_j) − √(n_jᵀQ_j n_j)   # Eq. (6)
    ∇_c   ← n_j                                                   # Eq. (9)
    ∇_R   ← (n_j × Q_R n_j)/√(n_jᵀQ_R n_j)                        # Eq. (10)
    ∇_n   ← (c_R − c_j) − Q_R n_j/√(n_jᵀQ_R n_j) − Q_j n_j/√(n_jᵀQ_j n_j)  # Eq. (11)
해 ← OSQP(Eq. 12)
if 비가능:
    δc, δθ ← 0, 0                                                 # emergency stop
else:
    n_j ← normalize(n_j + δn_j)   ∀j                              # 가상 상태 갱신 (warm start)
â_t ← (δc, δθ, gripper 원본)
로봇에 실행
```

### 종료 조건 / 폴백 정리

| 상황 | 동작 | 증거 |
|---|---|---|
| gap < $\delta$ | 타깃 미확정, 전 객체를 장애물로 | [PAPER] Eq. (4) |
| QP 비가능 | zero delta(emergency stop) | [PAPER] 부록 §7.1 |
| 팔이 객체를 가림 | 트랙 freeze (마지막 위치 유지) | [PAPER] §3.2 |
| identity swap | HSV Bhattacharyya로 재연결 | [PAPER] §3.2 |
| $\sum_K \alpha_i = 0$ | **미기재** | [ASSUMPTION] |
| 객체 신규 등장/소멸 | **미기재** | [ASSUMPTION] |
| 하이퍼파라미터 $K,\delta,\gamma_h,W,\epsilon$ | **전부 미기재** | [ASSUMPTION] → OPEN-Q 1 |
