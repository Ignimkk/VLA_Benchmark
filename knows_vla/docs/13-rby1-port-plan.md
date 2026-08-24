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

### 해결 — 자코비안 다리 &nbsp;&nbsp;✅ **구현 완료**

> [cbf/filter.py](../cbf/filter.py)의 `ArticulatedBody` / `Link`.
> 체인 룰이 **실제 정기구학 유한차분**과 일치함을 검증했다(자기 자신의 자코비안과 대조하면
> 아무것도 증명하지 못하므로, 3관절 체인의 FK를 직접 미분해 대조했다).
> 6링크 × 6장애물 = 36 제약에서 **5.8 ms**(몸체가 움직이는 현실적 조건)로 Table 2의
> 11.4 ms 예산 안이다.
> [16-filter-variants.md](16-filter-variants.md) §6.

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

QP 크기는 (링크 수 × 장애물 수)로 커진다. **실측(몸체 이동 중): 36 제약 5.8 ms, 60 제약
9.1 ms**로 Table 2의 11.4 ms 예산 안이다. 처음 두 번의 측정이 각각 종료 조건 결함과 정지 자세
반복 인공물로 틀렸고, 실제 해법은 쌍 전체를 벡터화하는 것이었다 —
[16-filter-variants.md](16-filter-variants.md) §6.

**단위 테스트로 검증한 것**: 체인 룰이 실제 정기구학 유한차분과 일치하고(자기 자신의 자코비안과
대조하면 아무것도 증명하지 못하므로 3관절 체인의 FK를 직접 미분했다), 항등 자코비안($J_v=I$,
$J_\omega=0$)에서 task-space 필터와 결과가 같으며, 관절 한계가 지켜지고, 관절 가중치로 어느
관절이 보정을 흡수할지 조절된다.

**§5 한계 해소의 직접 검증**: 팔꿈치만 위협받는 자세를 만들어(팔꿈치 여유 +3.0 → −5.7 cm,
EEF는 +9.4 cm 아래로 내려가지 않음) EEF만 모델링하면 명령이 그대로 통과해 팔꿈치가 충돌하고,
체인 전체를 모델링하면 막힌다는 것을 확인했다.

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

> **실행 완료 — 결과: ADMISSIBLE (조건 2개).**
> [14-transport-admissibility.md](14-transport-admissibility.md)
>
> 판정이 두 번 뒤집혔고 두 번 다 도구가 원인이었다(같은 문서 §0). 최종 조건은:
> **(1) 크레이트를 벽·바닥·손잡이로 분해할 것**, **(2) EEF를 과일 중심 +3 cm에서 닫을 것.**
> 그리퍼는 이제 가정이 아니라 RB-Y1 메시 실측이며, 손 본체가 7.6×4.5×5.8 cm로 이전 가정
> (4×4×7)보다 크다.

`check_admissibility.py`가 MJCF와 OBJ를 직접 읽으므로 줄자도, 모델 컴파일도 필요 없다.

국면별 결과:

| 국면 | 타깃 $\tau$ | 결과 |
|---|---|---|
| 과일 파지 | 해당 과일 | **통과** — 단 EEF를 과일 중심 **+3 cm**에서 닫을 것. 오프셋 0에서는 orange −1.0, banana −1.2 cm |
| 크레이트 파지 | 크레이트(손잡이) | **통과** — 손잡이 분리 시. 단 타깃 제외가 파생 부품까지 덮어야 한다 |
| 크레이트 담기 | 크레이트 | **통과** — 크레이트를 **벽·바닥·손잡이로 분해**할 때만. 통짜면 림 위 10 cm에서도 −4.7 cm |

(선반 배치는 국면에서 빠졌다 — 현재 학습 중인 시나리오는 과일 담기와 크레이트 들기까지다.)

**담기 국면이 기하로 해결됐다.** 크레이트를 껍질로 분해하면 내부가 실제로 열린다(바닥 근처에서도
$h=+0.9$ cm). 얇은 판을 감싼 타원체도 얇은 방향은 얇게 유지되기 때문이다.

> **이전 판에서 "이식의 성패를 가르는 단일 실험"이라고 했던 attention 국면 실험은 게이트에서
> 내려간다.** 크레이트가 장애물로 남아 있어도 담을 수 있으므로 Eq. (4)의 제외에 의존하지 않는다.
> 오히려 분해 쪽이 낫다 — 통째 제외는 운반 경로에서 실제로 부딪히는 손잡이($h=-5.0$ cm)까지
> 무방비로 만든다.

주의: 위 판정은 **지각 오차를 0으로 놓은 것**이다. 한 시점 깊이에서 적합하면 타원체가 작아지므로
(orange z반축 3.1 → 2.0 cm) D6의 불확실성 마진을 넣고 재검사해야 한다.

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
0. Admissibility (§4)                ← 완료. 조건 2개를 붙여 통과
1. 필터의 다중 파트 지원 (D1/D2/D3)   ← 체크포인트 불필요. 지금 가능
2. 지각 스택 (§3) + 불확실성 마진 D6  ← 색 분할 + 깊이 + TF. 넣고 §4 재검사
3. 자코비안 CBF (§1)                 ← Layer A 변경. 단위 테스트 먼저
4. Attention 헤드 재선택 + 국면 채점   ← 체크포인트 대기. 품질 항목 (게이트 아님)
5. 전신 링크 확장 (§1 이득)           ← 논문 §5 한계 해소. 선택
6. 오프라인 통합 → 폐루프            ← LIBERO에서와 같은 순서
```

**순서가 다시 바뀌었다.** 이전 판은 attention 국면 실험을 1번에 놓았는데, 담기 국면이 기하로
해결되면서 그 실험이 게이트가 아니게 됐다(§4). 대신 **체크포인트 없이 지금 할 수 있는 것들**이
앞으로 온다 — 필터의 다중 파트 지원, 가상 normal 제거, 슬랙 QP는 전부 합성 타원체와 기존
LIBERO 롤아웃으로 검증된다([15-development-plan.md](15-development-plan.md) 우선순위 표).

**2번에서 §4를 반드시 재검사한다.** 현재 판정은 GT 메시 기준이고 마진이 mm~cm 단위라, 지각
오차가 들어오면 뒤집힐 수 있다.

LIBERO에서 기하를 모르고 P3a까지 가서 9/9 실패를 본 것이 "먼저 판정하고 착수한다"는 원칙의
근거다. 그 원칙은 유지되고, 순서만 바뀌었다.

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

1. **담기 국면이 attention에 전적으로 달려 있다**(§4). 크레이트가 타깃으로 지목되지 않는 스텝이
   있으면 그 스텝은 반드시 개입을 받는다. 완화책은 국면 인지형 제외나 볼록 분해인데, 후자는
   더 이상 논문 방법이 아니다.
2. **양팔 타깃 분해(§2)에 정답이 없다.** 단일 팔 과제로 범위를 좁히는 것이 현실적이다.
3. **실로봇 안전.** LIBERO에서 최대 181 cm 명령이 나왔다([11-p2b-results.md](11-p2b-results.md) §3).
   실로봇에는 **액션 한계를 반드시 걸어야 한다** — 논문에 없지만 타협 불가.
   그러면 emergency stop이 발동하므로 정지 시 거동을 미리 정의해야 한다.
4. **$H=50$, $K=8$** 이므로 attention 그리드가 8 스텝 동안 고정된다. LIBERO($K=5$)보다 신선도가
   낮다 — 동적 장애물 대응력에 영향.
