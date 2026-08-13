# RB-Y1 이식 계획

KNOWS를 RB-Y1 **transport 시나리오**로 옮기기 위한 계획. `reproduction.md` §11에 따라
**Layer A(충실 재현)** 와 **Layer B(임베디먼트 적응)** 를 분리하되, 이번에는 Layer A 자체를
고쳐야 하는 항목이 여럿 있다.

> **대상 모델**: 학습 중인 transport 모델(`pi05_rby1_mobile_lora`, 17-D). 블록 pick-and-place
> 모델(`pi05_rby1_lora`, 14-D)은 사용하지 않는다.
>
> **시나리오** ([TRANSPORT_SCENARIO_KO.md](../../../src/docs/TRANSPORT_SCENARIO_KO.md)):
> 1. 양팔로 크레이트 파지 → 리프트 → **베이스 주행** → 3단 선반 배치
> 2. 한 손으로 과일 4종 파지 → 크레이트에 담기 → 시나리오 1

## 0. transport가 논문 전제를 벗어나는 지점

논문은 **고정 베이스 단일 팔 탁상 조작**을 상정한다. transport는 셋 다 다르다.

| | 논문 | transport | 영향 |
|---|---|---|---|
| 베이스 | 고정 | **주행** (17-D의 `base_x/y/yaw`) | 카메라가 움직인다. 장면이 방 전체로 확장 |
| 팔 | 1개 | **양팔** (크레이트를 함께 든다) | Eq. (4)의 단일 타깃 전제 붕괴 |
| 목적지 | 탁상 평면 | **3단 선반** | 오목 구조 — 타원체가 내부를 메운다 |
| 운반물 | 없음 | **크레이트를 계속 쥔 채 이동** | [OPEN-Q 19](OPEN-QUESTIONS.md)가 상시 발생 |

**셋 다 완화가 아니라 악화 방향이다.** §4의 admissibility 점검이 LIBERO 때보다 훨씬 중요해진다.

---

## 1. 최대 장벽 — 액션 공간이 다르다 (Layer A 변경 필요)

| | 논문 (LIBERO/Panda) | RB-Y1 transport |
|---|---|---|
| 액션 | EEF delta pose $(\Delta x, \Delta\theta) \in \mathbb{R}^6$ + gripper | **17차원 절대값** `[L6, Lgrip, R6, Rgrip, base_x, base_y, base_yaw]` |
| 하위 제어 | OSC (블랙박스) | 관절 위치 + 베이스 |
| $H$ / $K$ | 8 / 미기재 | **50 / 8** (블록 모델 기준, transport 모델에서 재확인 필요) |

베이스 3차원은 **절대 world 좌표**이며, 문서 §8이 지적하듯 다른 시작 포즈로 일반화되지 않는다.
필터가 베이스를 건드리려면 이 표현을 먼저 이해해야 한다.

**Eq. (12)의 결정변수 $\delta c_R$, $\delta\theta$가 RB-Y1에는 존재하지 않는다.** 정책이 관절
목표를 내므로 EEF delta를 직접 명령할 수 없다. 논문 수식을 그대로 적용할 수 없다.

### 해결 — 자코비안 다리

결정변수를 관절 증분 $\delta q$로 바꾸고, EEF 운동을 자코비안으로 연결한다:

$$\delta c_R = J_v(q)\,\delta q, \qquad \delta\theta = J_\omega(q)\,\delta q$$

Eq. (8)의 선형 제약은 그대로 살아난다:

$$\underbrace{\left(\nabla_{c_R}h_j^\top J_v + \nabla_{R_R}h_j^\top J_\omega\right)}_{\text{1}\times n_q}\delta q
\;+\; \nabla_{n^{(j)}}h_j\cdot\delta n^{(j)} \;\ge\; -\gamma_h h_j$$

목적함수는 정책 명령에서의 이탈을 벌한다:

$$\min_{\delta q,\,\{\delta n\}} \;\|\delta q - \delta q^{\mathrm{nom}}\|_W^2$$

여기서 $\delta q^{\mathrm{nom}} = q^{\mathrm{target}}_{\text{policy}} - q_{\text{current}}$ 이다.

**베이스 포함**: transport에서는 $\delta q$에 베이스 3차원도 넣을 수 있다. 그러면 자코비안이
$J = [J_{\text{arm}} \; J_{\text{base}}]$로 확장되고, **베이스 충돌 회피가 같은 QP에서 처리된다** —
방 안의 벽·테이블·선반이 장애물로 들어온다. 논문에는 없는 확장이지만 transport에서는 필수다.

**이것은 논문 이탈이다.** `reproduction.md` 분류로 **Proposed improvement**가 아니라
**Engineering adaptation**에 가깝다 — 방법의 의도(EEF를 안전 집합에 투영)는 보존하되, 제어
권한이 다른 로봇에 맞춰 매개변수화만 바꾼다. 문서에 반드시 그렇게 표기한다.

### 뜻밖의 이득 — 논문 §5의 한계가 사라진다

논문 §5는 스스로 이렇게 적는다:

> "The CBF protects a single ellipsoid approximating the end-effector; the rest of the arm is
> unmodeled. ... Extending the filter to cover the whole kinematic chain is the natural next step;
> however, our safety filter assumes control of only the end effector position and pose."

**그 제약은 EEF delta만 명령할 수 있었기 때문이다.** RB-Y1은 관절을 직접 명령하므로 그 이유가
사라진다. 링크마다 타원체를 붙이고 (링크 $\times$ 장애물) 쌍마다 제약을 추가하면 된다:

$$\left(\nabla_{c_{L_i}}h_{ij}^\top J_v^{(L_i)} + \nabla_{R_{L_i}}h_{ij}^\top J_\omega^{(L_i)}\right)\delta q
\ge -\gamma_h h_{ij} \quad \forall (i, j)$$

QP 크기는 (링크 수 × 장애물 수)로 커지지만, [09-p1-results.md](09-p1-results.md) §3에서 10개
장애물에 ~1 ms였으므로 여유가 있다. 링크 6개 × 장애물 6개 = 36 제약이면 수 ms 수준이다.

**즉 RB-Y1 이식은 논문 방법의 후퇴가 아니라 확장이다.** 이 점을 명시적으로 평가 항목에 넣는다.

## 2. 양팔 — Eq. (4)가 상정하지 않은 구조

논문은 EEF 하나, 타깃 하나를 상정한다. RB-Y1은 팔이 둘이다.

| 항목 | 문제 | 대응 |
|---|---|---|
| EEF 타원체 | 2개 (좌/우 그리퍼) | 각각 별도 로봇 타원체. 제약이 2배 |
| 타깃 | Eq. (4)는 $\tau_t$ 하나만 고른다 | 팔마다 다른 물체를 다룰 수 있다 → **팔별 타깃이 필요** |
| 자기충돌 | 논문 미고려 | 좌팔 링크 ↔ 우팔 링크 제약 추가 |

**팔별 타깃 분해는 미해결 문제다.** attention은 정책 전체의 것이지 팔별로 나뉘지 않는다.
현실적 선택지:

- (a) 두 팔이 같은 타깃을 공유한다고 가정 (단일 물체 조작에는 타당)
- (b) 손목 카메라 attention을 팔별 신호로 쓴다 — 논문 §3.4는 손목 뷰가 agent 뷰보다 약하지만
  같은 경향이라고 보고한다. 검증 필요
- (c) 팔별로 가장 가까운 물체를 타깃으로 (attention 미사용, 논문 이탈)

**transport에서는 (a)가 성립하지 않는다.** 시나리오 1은 양팔이 **하나의 크레이트를 함께** 들므로
타깃은 공유되지만, 시나리오 2는 한 손이 과일을 집는 동안 다른 손이 놀거나 크레이트를 잡는다.

권고: **시나리오 2의 단일 팔 과일 파지 구간부터** 착수한다. 팔 하나, 타깃 하나로 논문 전제에
가장 가깝다. 크레이트 양팔 운반(시나리오 1)은 그 다음이며, 거기서는 (a)를 쓰되 **크레이트가
`held`로 분류되어야** 한다([OPEN-Q 19](OPEN-QUESTIONS.md)) — 두 그리퍼 모두 크레이트와 상시
접촉하므로 장애물로 두면 $h<0$이 전 구간 지속된다.

## 3. 지각 — 시뮬레이션은 LIBERO와 같고, 실로봇만 어렵다

**중요**: transport 시나리오는 MuJoCo에서 돈다(`model_transport.xml`). 즉 **시뮬레이션에서는
LIBERO와 똑같이 GT 세그멘테이션·깊이를 얻을 수 있다.** 지각 문제는 실로봇 단계에서만 발생한다.

이는 LIBERO에서 쓴 2단계 전략을 그대로 반복할 수 있다는 뜻이다 — 먼저 GT로 방법을 검증하고,
그 다음 실제 검출기로 바꿔 성능 하락분을 지각 오차로 귀속한다.

| 단계 | 마스크 | 깊이 | 외부 파라미터 |
|---|---|---|---|
| **sim (먼저)** | MuJoCo segmentation 렌더 | MuJoCo depth | 씬에서 직접 |
| real (나중) | 검출기 필요 | RealSense / ZED | ROS TF |

MuJoCo는 robosuite와 달리 세그멘테이션 API가 다르므로(`mjv_updateScene` +
`mjRND_SEGMENT`, body id 기반), `dump_camera_params.py`와 `libero_env.py`에 해당하는
transport용 어댑터를 새로 써야 한다. 분량은 작다 — 카메라 파라미터와 id↔이름 매핑뿐이다.

### 실로봇 단계

| 필요한 것 | LIBERO | RB-Y1 | 상태 |
|---|---|---|---|
| 객체 마스크 | GT instance seg | **없음** | 검출기 필요 |
| 깊이 | sim depth | RealSense / ZED | 하드웨어 있음 ([src/realsense-ros](../../../src/realsense-ros)) |
| 카메라 내부 | `dump_camera_params.py` | 캘리브레이션 파일 | 확인 필요 |
| 카메라 외부 | robosuite API | **ROS TF** (`cam_high` → base) | 확인 필요 |
| 객체 id 일관성 | GT 불변 | 프레임 간 추적 필요 | 논문 §3.2의 HSV Bhattacharyya가 여기서 실제로 필요해진다 |

**검출기 선택이 갈림길이다.** 논문은 YOLOe를 파인튜닝했지만 레시피도 가중치도 공개하지 않는다
([OPEN-Q 9](OPEN-QUESTIONS.md)). 선택지:

1. **YOLOe 제로샷 + 프롬프트** — 논문과 같은 계열, 파인튜닝 없이 시작
2. **SAM 계열 + 텍스트 프롬프트** — 더 무겁지만 마스크 품질이 좋다
3. **색 분할** — "파란 블록 / 갈색 상자"는 색이 확연히 다르다. 첫 통합에는 이걸로 충분하고,
   지각 오차와 방법 문제를 분리할 수 있다

**3번으로 시작할 것을 권한다.** LIBERO에서 GT 마스크가 했던 역할(지각 오차 배제)을 대신한다.
여기서 동작을 확인한 뒤 1번으로 교체하면, 성능 하락분이 곧 지각 오차의 기여분이다.

## 4. Admissibility 사전 점검 — 코드 작성 전에

[12-p3a-results.md](12-p3a-results.md) §3에서 얻은 판정식을 **줄자로** 먼저 적용한다:

$$r_R^{\max} + r_j^{\max} < \|c_\tau - c_j\| \qquad \forall j \ne \tau$$

> **실행 완료 — 결과: NOT ADMISSIBLE (조건부).** [14-transport-admissibility.md](14-transport-admissibility.md)
> 인접 과일 쌍(9 cm 간격)이 마진 −0.2~−1.9 cm로 차단되고, 크레이트는 손잡이 때문에 z반축이
> 36 cm로 부풀어 −4.6~−7.1 cm. 다만 보수적 근사(외접 타원체, 그리퍼를 타깃 중심에 배치)를
> 썼으므로 MVEE 재적합으로 뒤집힐 여지가 크다.

**MuJoCo 씬에서 직접 계산할 수 있다** — 줄자도 필요 없다.
`config/transport_layout.json`에 크레이트·과일·선반 포즈가 있고, geom에서 크기를 읽을 수 있다.

재야 할 쌍:

| 국면 | 타깃 $\tau$ | 검사할 비타깃 $j$ | 위험도 |
|---|---|---|---|
| 과일 파지 | 해당 과일 | 다른 과일 3종, 크레이트, 테이블 | 중 — 문서 §8이 "최소 중심 간격 0.080 m" 확인 |
| 크레이트 담기 | 크레이트 | 나머지 과일, 테이블 | 높음 — 크레이트는 **속이 빈 용기** |
| 선반 배치 | 선반 칸 | 다른 선반 칸, 벽, 크레이트 | **매우 높음** — 3단 선반은 오목 구조 |

**과일 최소 간격 0.080 m가 이미 경고 신호다.** 과일 반축이 3~4 cm씩이면 두 과일의 반축 합만
6~8 cm로 간격과 비슷하고, 여기에 그리퍼 반축이 더해지면 §4의 부등식이 깨진다. LIBERO에서
접시 하나 때문에 9/9 실패했던 것과 정확히 같은 구조다.

**선반이 가장 심각하다.** 3단 선반을 타원체 하나로 근사하면 **선반 전체 부피가 장애물**이 되어
어느 칸에도 넣을 수 없다. 크레이트도 마찬가지로 내부가 메워져 과일을 담을 수 없다
([02-architecture.md](02-architecture.md) §4). Eq. (4)가 배치 단계에 목적지를 타깃으로 제외해
가리지만, gap이 $\delta$ 미만이 되는 순간 드러난다.

> **이것이 착수 전에 답해야 할 첫 질문이다.** 깨진다면 타원체 표현 자체를 바꿔야 하고
> (선반을 칸별로, 크레이트를 벽 4장으로 볼록 분해), 그러면 더 이상 논문 방법이 아니다.

## 5. Attention 헤드 재선택 — 저렴하고 필수

transport 모델은 `pi05_base`의 LoRA 파인튜닝이다. [08-p0b-results.md](08-p0b-results.md)에서
`pi05_libero`는 layer 12가 정점임을 확인했지만 **다른 체크포인트에서 같으리라는 보장은 없다** —
LoRA가 attention을 직접 수정한다.

> **기존 `data/policy_records/`는 쓸 수 없다.** 그것은 블록 pick-and-place 모델의 기록이고,
> 사용할 모델이 아니다. transport 씬에서 새로 수집해야 한다.

다만 LIBERO에서 만든 파이프라인이 거의 그대로 재사용된다:

| 항목 | LIBERO | transport |
|---|---|---|
| 수집 | `collect_p0b.py` (websocket 클라이언트) | 동일 구조, MuJoCo 씬만 교체 |
| 마스크 | robosuite GT seg | MuJoCo GT seg (§3) |
| 타깃 정답 | BDDL 목표 | `evaluation/transport.py`의 판정 로직 재사용 |
| 스윕 | `probe_p0b.py` | 거의 그대로 |

[probe_p0b.py](../probe_p0b.py)에서 바꿀 것:

- `AGENTVIEW_KEYS = (0, 256)` — RB-Y1도 `cam_high`가 첫 이미지이므로 **그대로**
  (ALOHA 규약: `cam_high` → `base_0_rgb`)
- suffix 길이가 $H=50$이므로 `[L, G, T, 256]`의 $T$가 50 — 코드는 shape에 무관
- 마스크 소스를 GT에서 색 분할로

## 6. 실행 순서

```text
0. Admissibility 1차 (§4)           ← 완료. 조건부 실패 → 1번으로
1. MVEE 재적합 + 실제 파지자세 재검사 ← 유형 A가 통과하는지 확정. 여기서 갈린다
2. Attention 헤드 재선택 (§5)        ← 기존 policy_records + 색 분할. 서버 1회 실행
3. 지각 스택 (§3)                   ← 색 분할 + 깊이 + TF 외부파라미터
4. 자코비안 CBF (§1)                ← Layer A 변경. 단위 테스트 먼저
5. 전신 링크 확장 (§1 이득)          ← 논문 §5 한계 해소. 선택
6. 오프라인 통합 → 폐루프           ← LIBERO에서와 같은 순서
```

**1번을 먼저 하는 이유**: 장면이 admissible하지 않으면 나머지가 전부 무의미하다. LIBERO에서
그걸 모르고 P3a까지 가서 9/9 실패를 본 뒤에야 알았다.

**2번이 두 번째인 이유**: 헤드가 이식되지 않으면 방법의 전제가 무너진다. 비용이 낮으므로 먼저 친다.

**4번 전에 단위 테스트**: [tests/test_cbf.py](../tests/test_cbf.py)가 EEF 공간 CBF를 검증한 것처럼,
자코비안 버전도 유한차분으로 검증한다 — $\delta q$에 대한 $h$의 gradient가
$\nabla_c h^\top J_v + \nabla_R h^\top J_\omega$와 일치하는지.

## 7. 이식하면서 유지할 것

| 항목 | 상태 |
|---|---|
| Eq. (2)–(4) 타깃 식별 | **그대로** — 액션 공간과 무관 |
| Eq. (5)–(7), (11) barrier와 normal | **그대로** — 순수 기하 |
| Eq. (9)–(10) EEF gradient | **그대로** — 자코비안 앞단에 그대로 들어간다 |
| Eq. (12) 결정변수·목적함수 | **변경** — §1 |
| $Q_R$ 캘리브레이션 | RB-Y1 그리퍼로 재측정. [12-p3a-results.md](12-p3a-results.md) §2의 교훈 — 마스크 적합은 손목까지 삼킨다 |
| $\gamma_h, \epsilon, W, K, \delta$ | 여전히 미상. $\epsilon$–$\gamma_h$ 공동 스윕 필수 ([11-p2b-results.md](11-p2b-results.md) §4) |

## 8. 알려진 위험

1. **§4가 깨지면 재설계**다. 타원체를 볼록 분해(상자를 벽 4장으로)로 바꾸면 해결되지만 그것은
   논문 방법이 아니다.
2. **양팔 타깃 분해(§2)에 정답이 없다.** 단일 팔 과제로 범위를 좁히는 것이 현실적이다.
3. **실로봇 안전.** LIBERO에서 최대 181 cm 명령이 나왔다([11-p2b-results.md](11-p2b-results.md) §3).
   실로봇에는 **액션 한계를 반드시 걸어야 한다** — 논문에 없지만 타협 불가.
   그러면 emergency stop이 발동하므로 정지 시 거동을 미리 정의해야 한다.
4. **$H=50$, $K=8$** 이므로 attention 그리드가 8 스텝 동안 고정된다. LIBERO($K=5$)보다 신선도가
   낮다 — 동적 장애물 대응력에 영향.
