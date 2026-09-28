# AG3S trajectory optimization — 설계 명세

*작성 2026-09-28. `benchmark/trajopt/` 의 코드를 읽고, 이 문서에 실린 수치는 그 자리에서 실행해
측정한 것이다. 추정치는 하나도 없다.*

---

## 0. 이 문서가 답하는 것

| 질문 | 절 |
|---|---|
| 로봇이 제대로 modeling 되어 있는가 | [1](#1-robot-model--무엇이-충돌-기하인가) |
| 16D action chunk 를 어떻게 받아 initial guess 로 쓰는가 | [2](#2-16d-action-chunk-가-decision-variable-이-되는-경로) |
| objective function 식은 무엇인가 | [3](#3-objective-function) |
| 어떤 constraint 를 쓰는가 | [4](#4-constraint) |
| 그것을 어떻게 푸는가 | [5](#5-sqp--rti-loop) |
| MuJoCo 가 spawn 하는 robot 과 TO 출력 joint 값이 정상 매핑인가 | [6](#6-mujoco-↔-to-매핑-검증) |
| 발견한 결함, 그리고 그것이 고쳐졌는지 | [7](#7-발견--둘-다-고쳐졌다-2026-09-28) |

**Figure**

| | 파일 | 무엇 |
|---|---|---|
| Fig 1 | [`figures/to-design-model.png`](figures/to-design-model.png) · [json](figures/to-design-model.json) | collision model 이 MuJoCo 로봇 위에 놓인 모습 + URDF↔MuJoCo FK 일치도 |
| Fig 2 | [`figures/to-design-mapping.png`](figures/to-design-mapping.png) · [json](figures/to-design-mapping.json) | action column → decision variable → joint → actuator 전체 매핑 |
| Fig 3 | [`figures/to-design-solve.png`](figures/to-design-solve.png) · [json](figures/to-design-solve.json) | 실제 solve 한 번 — clearance, deviation, timing |
| Fig 4 | [`figures/to-design-fix-20260928.png`](figures/to-design-fix-20260928.png) · [json](figures/to-design-fix-20260928.json) | 아래 [7](#7-발견--둘-다-고쳐졌다-2026-09-28) 의 두 결함을 고친 뒤의 재측정 |

> **Fig 1 · Fig 2 는 수정 *전*의 측정이다.** 그날 오후 commit `7a1cdc5` 가 둘을 고쳤고, 고친 뒤의
> 숫자는 Fig 4 에 있다. 두 그림을 지우지 않는 이유는 [7](#7-발견--둘-다-고쳐졌다-2026-09-28) 이
> 그것을 결함의 증거로 링크하기 때문이다.

**용어** — 이 문서에서 처음 쓰는 말은 그 자리에서 풀어 쓴다.

| 용어 | 뜻 |
|---|---|
| chunk | 정책이 한 번의 inference 로 내놓는 action 묶음. 여기서는 `[50, 16]` |
| planned window (P) | chunk 중 optimizer 가 실제로 다듬는 앞부분 스텝 수 |
| execution window (K) | 다음 chunk 를 요청하기 전에 로봇이 실제로 실행하는 스텝 수 |
| 여유거리 (clearance) | `h`. 양수면 떨어져 있고 음수면 파고든 깊이(m) |
| ESDF | Euclidean Signed Distance Field. 공간의 각 점에서 가장 가까운 표면까지의 부호 있는 거리 |
| query point | 거리를 묻는 점. 로봇의 collision sphere 중심 + 쥔 물체의 점 |
| slack | constraint 를 못 지킬 때 그 위반량을 담는 보조 변수. 목적함수가 비싸게 벌한다 |
| warm start | 지난 solve 의 해와 factorization 을 다음 solve 의 출발점으로 재사용하는 것 |
| RTI | Real-Time Iteration. 수렴까지 돌지 않고 **주기당 고정 예산**만큼만 도는 SQP 방식 |
| `violated` / `feasible` | TO 가 내는 상태. `violated` 는 최선을 다했지만 위반이 남았다는 **보고**이지 예외가 아니다 |

---

## 1. Robot model — 무엇이 충돌 기하인가

### 1.1 세 개의 좌표계가 아니라 하나

이 pipeline 에는 로봇이 세 번 등장한다. **셋이 같은 로봇이어야 한다.**

```
MuJoCo MJCF   model_transport.xml     ← 물리를 돌리고 로봇을 spawn 한다 (local PC)
URDF          rby1a/urdf/model.urdf   ← AG3S 와 TO 의 forward kinematics (GPU server)
ChunkLayout   types.ChunkLayout       ← action chunk 의 어느 열이 어느 joint 인가
```

`SafePolicy` 는 AG3S 가 constraint 를 쓴 것과 **같은 모델 객체**를 optimizer 에 넘긴다
([safe_policy.py:145-159](../../trajopt/safe_policy.py#L145-L159)) — 인자로 받지 않고
`ag3s.constraint_robot_model` 에서 꺼낸다. 두 모델이 다르면 optimizer 는 한 곳의 기하를 피하면서
constraint 는 다른 곳을 설명하게 되고, 아무도 그 불일치를 보고하지 않기 때문이다.

### 1.2 q 는 20-DoF 다

`DEFAULT_RBY1_JOINTS` = `torso_0..5` (q0–5) · `right_arm_0..6` (q6–12) · `left_arm_0..6` (q13–19).

Wheel 과 head 는 빠진다. Wheel 은 base 를 통째로 옮기므로 base frame 으로 표현하는 것으로 처리되고,
head 는 collision capsule 을 갖지 않는다. Gripper finger 는 prismatic 이동이 mm 단위라 3.5 cm
capsule radius 앞에서 무시된다.

**Head joint 는 FK 에는 필요하다** — ZED camera 가 `link_head_2` 에 달려 있어서, `head_0`/`head_1`
을 `fixed_joint_values` 로 고정해 넣지 않으면 head camera 의 FK 가 항상 수평인 머리를 따라간다
([mujoco_source.py:62-66](../../ag3s/experiments/sources/mujoco_source.py#L62-L66)).

### 1.3 Capsule → sphere

URDF 는 torso 와 arm link 0–5 에만 collision capsule 을 준다. Base, wheel, wrist(`link_*_arm_6`),
gripper palm(`ee_left`/`ee_right`), finger 는 `<collision>` element 가 아예 없다. 그래서
`gap_filling_capsules(mujoco_model)` 이 MuJoCo mesh 를 재서 그 빈자리를 채운다
([mujoco_source.py:465-484](../../ag3s/experiments/sources/mujoco_source.py#L465-L484)).

각 capsule 은 축을 따라 sphere 로 이산화되고, **sphere 는 inflate 된다.** 중심 간격 `s`, 반지름
`r` 이면 두 중심의 중간에 있는 capsule 표면 점은 가장 가까운 중심에서 `sqrt(r² + (s/2)²)` 떨어져
있다. 각 sphere 를 그 반지름으로 키워야 union 이 capsule 을 **포함**한다. 안 그러면 constraint 가
로봇의 일부를 보호하지 못하면서 보호한다고 말하게 된다.

Fig 1 왼쪽 패널이 그 결과다 — teleop keyframe 자세에서 **29 개 link 위 218 개 sphere**, 그리고
MuJoCo body origin(`+` 표시)이 그 안에 정확히 놓여 있다. 가운데 패널은 `wrist_cam_r` render 위에
같은 sphere 를 투영한 것으로, gripper sphere 가 실제 gripper 위에 얹혀 있다.

> 218 spheres 는 `gap_filling_capsules` 를 전부 넣은 모델이다. 실제 server 는 `--links`(기본
> `arms`)로 constraint model 을 줄여 쓰므로 배포 시 sphere 수는 더 작다.

### 1.4 URDF FK 와 MuJoCo FK 가 같은가 — 측정

16 개 자세(영점 1 + 관절 한계 안 무작위 15)에서 base frame 기준 link 원점을 두 kinematics 로
각각 계산해 비교했다. Fig 1 오른쪽 패널.

| link | worst \|p_urdf − p_mujoco\| (mm) |
|---|---:|
| `link_torso_3` | 2.3e-10 |
| `link_torso_5` | 0.00045 |
| `link_left_arm_3` | 0.00056 |
| `link_left_arm_6` | 0.00065 |
| `link_right_arm_3` | 0.00069 |
| `link_right_arm_6` | 0.00090 |
| `ee_left` | 0.00078 |
| `ee_right` | 0.0010 |
| `ee_finger_r1` · `ee_finger_r2` | 0.0011 |
| `ee_finger_l1` · `ee_finger_l2` | 0.00084 |

**판정: 두 kinematics 가 측정한 모든 link 에서 1.1 µm 안에서 일치한다.** 왼쪽 finger 둘은 수정
전에 6.001 mm 였다 — 기하가 아니라 **이름표**가 어긋나 있었고, [7.2](#72-urdf-ee_finger_l1-은-mjcf-ee_finger_l2-다) 에서
다룬다. 위 표는 수정 뒤의 값이다 (Fig 4).

---

## 2. 16D action chunk 가 decision variable 이 되는 경로

### 2.1 무엇을 받는가

π0.5 checkpoint `pi05_rby1_randomized_pick_place_16d_lora` 가 내놓는 것은
`chunk[50, 32]` 인데 앞 16 열만 뜻이 있다. 나머지는 openpi 의 padding 이다.

```
16 = [ left_arm_0..6 (7) , left gripper (1) , right_arm_0..6 (7) , right gripper (1) ]
```

이 layout 은 상수로 박혀 있지 않고 `ARM_JOINT_DIM = 7` **하나**에서 나온다
([wire.py:190-208](../../trajopt/wire.py#L190-L208)). openpi 가 delta mask 를
`make_bool_mask(N, -1, N, -1)` 로 만들기 때문에 총 차원은 항상 `2·(N+1)` 이고 gripper 열은
`(N, 2N+1)` 이다. 14D 는 `(6, 13)`, 16D 는 `(7, 15)`.

값은 **절대 관절 각도**다 — server 의 `AbsoluteActions` transform 이 모델 내부의 delta 예측을
이미 되돌려 놓은 뒤이므로, 로컬은 `d.ctrl` 에 그대로 넣는다
([pi05_infer.py:517-537](../../../pi05_TO_hybrid/rby1_bringup/pi05_infer.py#L517-L537)).

### 2.2 `ChunkLayout` — 무엇이 변수이고 무엇이 아닌가

**Optimizer 는 로봇에게 실제로 지령할 수 있는 것만 움직여도 된다.** Action format 이 가리키지
못하는 자유도를 최적화하면 실행할 수 없는 아름다운 궤적이 나오고, 그 차이는 로봇이 optimizer 가
생각한 곳이 아닌 다른 곳으로 가는 형태로 드러난다.

`ChunkLayout.rby1(joint_names, arm_joint_dim=7)` 이 세 가지로 가른다:

| | 열 / q | 왜 |
|---|---|---|
| **decision variable** (`action_to_q`) | 14 개 — 양팔 `arm_0..6` | action channel 이 있고 joint 다 |
| **parameter** (`fixed_q`) | q0–5 `torso_0..5` | action channel 이 없다. **FK 에는 그대로 들어간다** — 빼면 모든 collision sphere 가 로봇이 똑바로 선 것처럼 놓인다 |
| **passthrough** | 열 7, 15 (gripper) + padding 16–31 | 거의 binary 한 open/close 이고, 그것을 smoothing 하는 것은 이산적 결정을 흐리는 일이다 |

Fig 2a 가 이 매핑 전체다. q index 는 이름으로 찾으므로(`order[f"left_arm_{i}"]`)
`DEFAULT_RBY1_JOINTS` 가 바뀌어도 매핑이 조용히 한 칸 밀리지 않는다.

**Decision vector 의 순서는 q 순서가 아니라 action 열 순서다.** z0–6 = 왼팔(q13–19),
z7–13 = 오른팔(q6–12). `full_q()` 가 `q_indices` 로 scatter 하므로 FK 로 갈 때 제자리를 찾는다.

### 2.3 Initial guess — chunk 가 곧 출발점이다

세 자리에서 chunk 가 쓰인다. **셋의 역할이 다르다.**

```python
reference = layout.chunk_to_trajectory(chunk)[:, :planned]   # (14, P)
```

| 쓰임 | 무엇 | 어디 |
|---|---|---|
| **reference** | objective 의 tracking 항이 붙잡는 대상. 끝까지 안 바뀐다 | `problem.build_problem(reference=...)` |
| **initial iterate** | linearization 을 취하는 자리. 반복마다 갱신된다 | `sqp.solve` 의 `iterate` |
| **template** | 결과를 되쓸 원본. gripper 열과 padding 이 byte 단위로 그대로 통과하는 근거 | `layout.trajectory_to_chunk(traj, template)` |

첫 iterate 는 **정책 chunk 그 자체**다 (`iterate = reference.copy()`). 여기에 warm start 가
붙는 경우가 한 가지 있다:

```python
if warm_start and self._previous is not None and execution_length < horizon:
    k = min(execution_length, horizon - 1)
    shifted = concat([previous[:, k:], tile(previous[:, -1:], k)])
    iterate = 0.5 * (iterate + shifted)
```

지난 chunk 의 해를 그동안 실행된 스텝 수만큼 **밀어서** 현재 chunk 와 평균 낸다. 꼬리는
extrapolate 하지 않고 마지막 값을 유지한다 — horizon 너머로 궤적을 외삽하는 것은 정책이 제안한
적 없는 움직임을 지어내는 일이다.

**기본 배포에서는 이 분기가 실행되지 않는다.** `plan_horizon = PLAN_EXECUTION_WINDOW` 이면
`execution_length >= horizon` 이라 지난 계획과 이번 계획 사이에 겹침이 없다. 겹침이 없는데 밀면
실행 창 **이전**의 자세가 섞여 들어와 첫 iterate 를 뒤로 당긴다. 겹침이 없으면 reference 가
최선의 출발점이다.

### 2.4 몇 스텝을 다듬는가 — P = K = 8

| | 값 | 출처 |
|---|---:|---|
| `horizon` H | 50 | 정책의 action_horizon |
| `execution_length` K | 8 | `pi05_infer.OPEN_LOOP_HORIZON` |
| `control_hz` | 15 | `pi05_infer.CTRL_HZ` → `dt = 66.7 ms` |
| `plan_horizon` | `PLAN_EXECUTION_WINDOW` → **P = 8** | `serve_safe --plan-horizon` 기본값 `execution` |

`refine()` 이 chunk 를 P 스텝으로 잘라 넘기고, `trajectory_to_chunk` 가 그 P 스텝만 되쓰며,
`sqp._finish` 가 같은 구간에서 `max_violation` 을 잰다. **"다듬는 창을 줄인다" 와 "plan horizon
을 줄인다" 는 같은 편집이다.**

> **이 기본값은 실측으로 바뀐 것이다 (T6f).** 그전 기본값 32 는 실행 창 뒤로 24 스텝의 예지력을
> 샀지만 **회피를 미룰 자리도 함께 샀다**: refined chunk 가 실행되는 8 스텝에서는 사과를
> 손끝에서 +89.86 mm 피해 두고, 절대 실행되지 않는 스텝에서만 −17.61 mm 로 내려왔다. 매 chunk
> 같은 회피가 다시 계획되고 다시 버려져 사과는 closed loop 4 회에서 0.0 mm 움직였다 (정책 자신의
> chunk 는 245.2 mm). `PLAN_EXECUTION_WINDOW` 는 숨을 자리를 없앤다 — 예지력도 함께 잃는,
> 의도된 trade 다.

`benchmark/trajopt/configs/rby1.yaml` 에는 아직 `plan_horizon: 32` 가 적혀 있지만 **server 는
그 파일을 읽지 않는다.** `serve_safe.main()` 이 flag 로 `TrajOptConfig.from_dict(...)` 를 직접
짓는다 ([serve_safe.py:975-984](../../trajopt/serve_safe.py#L975-L984)).

---

## 3. Objective function

### 3.1 식

Decision vector 는 step-major 로 편 궤적 전체와 slack 이다.

```
z = [ vec(Q) ; s_collision ; s_kinematic ] ,    Q ∈ R^{14 × P} ,    z[k·14 + j] = Q[j, k]
```

Step-major 는 장식이 아니다. 모든 collision constraint 는 한 시간 스텝에서 작용하므로 이 순서에서
각 collision row 가 **연속된 14 개 변수 블록 하나**만 건드린다. 그것이 constraint block 의
sparsity pattern 을 고정시키고, 그래야 QP solver 가 반복 사이에 factorization 을 유지한다.

```
minimize   w_track  · Σ_k γ^k ‖Q[:,k] − Q_ref[:,k]‖²
         + w_smooth · ‖Δ²Q‖²
         + w_cont   · Σ_{k<L} ‖Q[:,k] − Q_prev[:,k]‖²
         + w_slack  · ‖s_collision‖₁
         + 10·w_slack · ‖s_kinematic‖₁
```

네 항 중 **task 에 대한 것은 첫째뿐이다.**

| 항 | 기본 weight | 왜 있는가 |
|---|---:|---|
| `w_track` | 1.0 | SEAM/π0.5 의 chunk 가 곧 task 다. Optimizer 의 일은 collision avoidance 가 허용하는 만큼만 그것을 바꾸는 것이다. **Tracking weight 가 너무 작으면 안전하지만 task 를 더 이상 수행하지 않는 궤적이 나오고, 어떤 collision metric 도 그것을 알아채지 못한다.** 그래서 `w_track <= 0` 은 config 가 거부한다 |
| `w_smooth` | 0.05 | 2차 차분 = jerk. SEAM 이 없애려는 바로 그 양이다. 무시하면 SEAM 이 제거한 불연속을 그대로 되돌려 준다 |
| `w_continuity` | 0.5 | [3.3](#33-w_continuity-가-왜-있는가) |
| `w_slack` | 1e3 | `w_track` 을 **지배해야 한다** — 아니면 optimizer 가 tracking 정확도를 collision 으로 산다. 그러나 "가능한 한 크게" 는 아니다: 1e4 에서 QP 가 ADMM 175 반복을 썼고 1e3 에서는 25 반복이다. 4 자릿수 차이가 solver 의 scaling 이 싸워야 할 conditioning 을 망가뜨린다 |
| `track_decay` γ | 1.0 | 뒤 스텝을 덜 중요하게 볼 수 있는 손잡이. 1.0 이 모든 스텝을 동등하게 두는 기본 동작 |

Kinematic slack 은 collision slack 의 **10 배**로 벌한다 — 로봇을 부수는 것이 장애물을 스치는
것보다 나쁘고, collision 과 달리 예측이 아니라 확실하기 때문이다
([problem.py:177-205](../../trajopt/problem.py#L177-L205)).

### 3.2 Slack 이 L1 인 이유

비음수 변수에 대한 L1 penalty 는 유한한 weight 위에서 **exact** 하다. 그래서 feasible 한 문제는
`s = 0` 을 얻지, 모든 row 에 조금씩 번진 위반을 얻지 않는다. Quadratic penalty 였다면 항상 작은
위반이 남는다.

### 3.3 `w_continuity` 가 왜 있는가

SEAM 은 다음 chunk 의 prior 를 **정제 전** model-space chunk 로 짓는다
(`seam_policy.py` 의 `with_chunk(model_chunk_np[0], ...)`). 즉 **TO 의 수정이 SEAM 에 되먹지
않는다.** 방치하면 정책이 매 chunk 같은 충돌 궤적을 다시 제안하고 TO 가 매번 밀어내면서, SEAM 이
없애려던 경계 jerk 가 TO 계층에서 되살아난다.

완화책이 `context["previous_physical_chunk"]` 에 대한 continuity 항이다. **완화이지 해결이
아니다** — 근본 해결은 physical→model 역변환과 SEAM 수정을 수반한다.

정렬이 중요하다: 이전 chunk 의 앞 K 스텝은 이미 실행됐으므로, 이번 chunk 와 겹치는 부분은
**index K 에서 시작한다** ([refiner.py:187-211](../../trajopt/refiner.py#L187-L211)).
0 에서 읽으면 로봇이 이미 지나간 스텝과 비교하게 된다.

> P = K 인 지금은 이전 chunk 가 0..K−1 스텝에서만 refine 됐고 그것이 전부 실행됐으므로, 이 항이
> 읽는 겹침은 **정책의 손대지 않은 꼬리**다. 이전 수정이 아니라 정책이 제안한 것 쪽으로 당기게
> 되어 전보다 약하지만, 올바르게 정렬된 유일한 reference 다.

### 3.4 Objective 는 두 번 구현되어 있다

`problem.build_problem` 이 행렬로 짓고, `problem.objective` 가 같은 식을 직접 계산한다. 중복이
아니라 **의도**다: SQP 의 step acceptance 가 실제 감소와 예측 감소를 비교하는데, QP 를 짓는 코드가
스스로의 증인이 되면 둘 다 틀렸을 때도 서로 동의한다.

---

## 4. Constraint

### 4.1 전체 목록

QP 에 실제로 들어가는 row block 은 다섯이다.

| block | 식 | soft? | 출처 |
|---|---|---|---|
| **box** (position + trust region + anchor) | `max(q_min, Q_k − r) ≤ Q ≤ min(q_max, Q_k + r)`, 첫 스텝은 추가로 `\|Q[:,0] − q_now\| ≤ v_max·dt` | hard | URDF + config |
| **velocity** | `\|Q[:,k+1] − Q[:,k]\| ≤ v_max·dt·0.9` | slack | URDF |
| **acceleration** | `\|Q[:,k+2] − 2Q[:,k+1] + Q[:,k]\| ≤ a_max·dt²·0.9` | slack | URDF |
| **collision** | `h(Q_k) + ∇h·(Q − Q_k) + s ≥ backoff`, `s ≥ 0` | slack | AG3S |
| **slack 비음수** | `s ≥ 0` | hard | — |

### 4.2 Box — 셋을 한 블록으로 교집합

Position box, trust region, 첫 스텝 anchor 는 **모두 같은 변수의 bound** 이므로 세 개의 row block
이 아니라 하나로 교집합한다 ([problem.py:203-228](../../trajopt/problem.py#L203-L228)).
그러면 trust region 이 뜻 그대로 읽힌다: *이번 반복은 linearization 을 취한 자리에서 어떤 joint 도
r 보다 멀리 움직이지 않는다.*

**Anchor 는 pin 이 아니다.** `Q[:,0] = q_now` 로 고정하면 첫 제어 주기 동안 로봇이 움직이는 것을
금지하게 된다. 한 스텝만큼의 이동은 허용한다.

교집합이 **빈 경우**는 로봇이 자기 한계 밖에서 시작했을 때 생긴다 — fault 나 잘못된 hand-off 뒤의
실제 상황이다. 그때는 중점으로 붕괴시켜 QP 를 풀 수 있게 두고 궤적이 걸어서 안으로 들어오게 한다.
실패하고 호출부에 아무것도 안 주는 것보다 낫다.

### 4.3 운동학 한계는 soft 다

Demonstration 으로 학습한 정책은 로봇의 가속 한계를 넘는 chunk 를 쉽게 내놓는다. 그것을 hard
constraint 로 두면 QP 가 `primal infeasible` 이 되고, **infeasible 한 QP 는 컨트롤러가 가장
무언가를 필요로 하는 순간에 아무것도 주지 않는다.** 그래서 row 를 slack 으로 부드럽게 하고 위반량을
`limit_report` 로 **보고**한다. KNOWS 의 CBF filter 가 같은 결론에 도달했다 (D3).

양측 bound 하나에 slack **하나**를 쓴다 — 한 row 가 양쪽에서 동시에 위반될 수는 없다:
`a·x − s ≤ b` 와 `a·x + s ≥ −b`.

`enforce_velocity=False` 는 constraint 를 제거하지 않고 optimizer 가 절대 닿을 수 없는 값으로
넓힌다. Ablation 이 문제의 **구조**가 아니라 숫자 하나만 바꾸게 하려는 것이다.

### 4.4 Collision — 배포에서는 ESDF 가 답한다

**Backend 가 두 개 있고, 지금 서버는 ESDF 를 쓴다**
([serve_safe.py:976-978](../../trajopt/serve_safe.py#L976-L978) 이
`backend="esdf"`, `use_support_planes=False` 를 박아 넣는다).

#### (a) primitive backend — candidate sphere + support plane

```
h = ‖p(q) − c‖ − ( r_robot + r_candidate + d_safe[sphere, slot] )
∇h = (p − c)ᵀ/‖p − c‖ · ∂p/∂q
```

`d_safe` 는 AG3S 의 `ClearancePolicy` 가 **(robot sphere, candidate slot) 쌍마다** 채운 행렬이다.
Target/obstacle 구분과 phase 별 완화가 전부 그 숫자 안에 들어 있고, TO 는 그것을 다시 계산하지
않는다.

Support plane: `h = n·p − offset − r_robot`.

#### (b) esdf backend — query point 당 한 행 (배포 기본)

```
h = d_esdf(p(q)) − r_query − margin(p, query)
∇h = ∇d_esdf(p)ᵀ · ∂p/∂q
```

Slot 을 열거할 필요가 없다 — field 는 **점**에 대해 답한다. 그래서 primitive 가 `S × M` 행일 때
이것은 `S` 행이다. ESDF gradient 는 eikonal 방정식 때문에 free space 에서 이미 단위 크기이므로
정규화하지 않는다. 일부러 그렇게 둔다 — 정규화하면 망가진 field 를 숨기게 된다. 장애물 깊숙한
곳(field 가 평평한 곳)에서만 구제한다.

`margin` 이 한 숫자가 아니다. 네 가지가 순서대로 덮어쓴다
([linearize.py:489-594](../../trajopt/linearize.py#L489-L594)):

| 층 | 무엇 | 왜 |
|---|---|---|
| 기본 `esdf_margin` = 50 mm | 모든 query point | 보수적 기본값 |
| `target_free_mask` (T8b) | 권한 있는 query point 는 **target 이 빠진 field 계층**에 되묻는다 | 성공한 grasp 는 손끝 구가 사과 표면을 −17.96 mm 파고든다. `m ≥ 0` 이라 margin 으로는 표면 안쪽을 허용할 수 없어서 `m` 이 아니라 `d` 를 바꾼다. **row 를 끄는 것과 다르다** — 끄면 table 에 대한 보호까지 잃는다 |
| `destination_margin` (20 mm) | 가장 가까운 표면이 **목적지**이고 그 query point 가 **쥔 물체**일 때만 | 쥔 사과의 점이 바구니 안벽을 32.9 mm 까지 지난다 (F18). 팔꿈치까지 완화하면 로봇 구 최악 여유가 −77.8 → −47.8 mm 로 망가지는 것을 실측했다 |
| `manipulated_link_margin` | 그 query point 의 가장 가까운 field 장애물이 **조작 대상 자신**일 때 (E1) | AG3S 가 조작 대상을 field 에서 파내지 않으므로, 권한 있는 link 만 자기 margin 을 쓴다. 판정은 **양방향**이다 (`|d − d_object| ≤ voxel`) — 단방향이면 물체가 field 에 없을 때 훨씬 먼 구까지 잘못 완화한다 (F13) |

#### (c) 쥔 물체는 로봇 쪽 query point 다

Grasp 가 닫히면 물체는 세계의 일부이기를 그만두고 로봇의 일부가 된다. 그래서 field **안**이 아니라
field 에 **묻는 쪽**에 놓인다 — 반지름 0 인 query point 로 로봇 sphere 뒤에 붙는다
([linearize.py:353-381](../../trajopt/linearize.py#L353-L381)).

점이지 fitting 된 primitive 가 아닌 이유 (F19): 사과의 관측 점구름에 sphere 하나를 맞추면 꼭지와
잎까지 덮느라 반지름이 57.4 mm 가 되어 바구니에 대한 여유를 평균 13.2 mm 먹고, 한 프레임에서는
없는 충돌을 만들어 낸다.

**QP 의 sparsity 는 움직이지 않는다.** Block 은 `P × rows_per_step` 이고 query point 를 늘리는
것은 그 행들을 **두고 경쟁하는 대상**을 늘리는 것이지 행 수를 늘리는 것이 아니다. 그래서 grasp 가
rollout 중간에 시작하고 끝나도 factorization 이 무효화되지 않는다.

Symbolic graph 도 다시 짓지 않는다. `_link_probe_map` 이 link 마다 네 점(원점 + 세 단위축)만
미분해 두면, rigid transform 이 affine 이므로 모든 attached point 가 그 넷의 정확한 affine
결합이다. 새 grasp 는 CasADi graph 가 아니라 numpy matmul 한 번이다.

### 4.5 Linearization backoff — 5 mm

QP 가 `h + ∇h·d ≥ 0` 을 만족시켰다고 `h(Q + d) ≥ 0` 이 되는 것은 **아니다.** 관절 공간의 직선을
따라 거리는 오목하므로 참값이 자기 접선 아래에 놓인다. RB-Y1 실측: linearized row 를 정확히
만족시키며 수렴한 SQP 에 **0.327 mm 의 실제 관통**이 남았다. Linearized row 에 몇 mm 를 더
요구하면 닫힌다.

대안은 gap 이 사라질 만큼 step 이 작아질 때까지 반복하는 것인데, 그 반복은 real-time loop 에 없다.

### 4.6 행을 줄이는 두 가지 — 그리고 그것이 안전한 이유

H=50 · 46 sphere · 32 slot 이면 약 78,000 행 대 600 변수다. **행이 병목이지 solver 가 아니다.**

1. **Activation band (0.35)** — 현재 clearance 가 band 를 넘는 row 는 trust region 한 스텝 안에
   위반이 될 수 없으므로 QP 해에 영향이 없다. Config 가 `activation_band > trust_radius` 를
   강제한다.
2. **Step 당 예산 (`rows_per_step = 24`)** — 각 스텝이 가장 빡빡한 24 개를 가져간다. 전역이 아니라
   **step 당** 인 것이 핵심이다: row `i` 는 언제나 step `i // 24` 에 속하므로 **QP 의 sparsity 가
   고정**되고 solver 가 factorization 을 재사용한다. Pattern 이 매 반복 움직였을 때 solve 당
   193 ms 였다.

**둘 다 안전한 이유는 하나다** — `full_violation` 이 해에서 **모든** 행을 원해상도로 재평가한다.
축소는 수렴을 잃을 수 있어도 충돌을 숨길 수는 없다. Selection 은 optimizer 가 무엇을 최적화하는지를
정하고, 이 재평가가 호출자에게 무엇을 말할지를 정한다.

쓰이지 않는 row 는 **제거되지 않고 자명하게 만족되도록** 채워진다 — gradient 0, clearance `+inf`.
AG3S 가 빈 candidate slot 을 끄는 것과 같은 장치다.

---

## 5. SQP + RTI loop

### 5.1 왜 IPOPT 가 아닌가

1. **Interior-point method 는 warm start 가 구조적으로 어렵다.** IPM 은 barrier parameter 로
   매개된 central path 를 따라간다. Margin 을 스치는 좋은 회피 궤적은 constraint **경계 근처**에
   있고, 그것은 어떤 적당한 μ 에서도 central path 에서 먼 나쁜 시작점이라 solver 가 μ 를 다시
   annealing 한다. Active-set · ADMM 계열에는 그 병리가 없다.
2. **반복 횟수에 상한이 없다.** IPOPT 는 허용오차까지 반복하고 restoration phase 에 들어갈 수
   있다. 실시간 제어는 **주기당 고정 예산**이 필요하고, 그것이 RTI 가 주는 것이다.

### 5.2 루프

```
started = now
iterate ← reference (+ warm-start blend)
반복 (최대 max_iterations = 3):
    반복 ≥ 1 이고 경과 ≥ time_budget_ms(50) 이면 중단
    rows    ← linearize(iterate)                    # 평가 · 선택 · 미분
    problem ← build_problem(reference, iterate, r)  # cost + box/vel/acc
    problem ← append_collision_rows(problem, rows, backoff)
    sol     ← osqp.solve(problem, warm_start)
    QP 실패면 r ← r·0.5, r < r_min 이면 중단, 아니면 재시도
    candidate, slack ← split(sol)
    merit ← objective(candidate) + w_slack · max(0, −full_violation(candidate))
    merit < best_merit 이면 best ← candidate
    merit ≤ iterate_merit 이면 iterate ← candidate,  r ← min(r·2, r_max)
    아니면                                          r ← max(r·0.5, r_min)
    step_size < 1e-5 이면 중단
finish: worst_row(best) 로 전체 행 재검사 → 상태 판정
```

**시간 예산은 첫 반복 뒤에만 본다.** 설정만으로 예산이 소진되는 상황에서 reference 를 손도 안 대고
성공이라 보고하는 것은 safety layer 가 내서는 안 되는 유일한 결과다. 한 스텝은 RTI 의 최소 작업
단위이고 언제나 취한다.

**Merit function 은 QP 의 linearized row 에서 계산하지 않는다.** 같은 코드가 만든 merit 은 둘 다
틀렸을 때도 QP 와 동의한다. 여기서는 기하를 다시 평가하므로 **동의하지 않을 수 있고**, 그것이
step acceptance 가 나쁜 linearization 을 잡아내는 유일한 길이다.

### 5.3 상태 판정

| status | 뜻 | `.safe` |
|---|---|---|
| `OPTIMAL` | 예산 안에서 수렴, 위반 없음 | ✅ |
| `FEASIBLE` | 예산/반복 상한이 먼저 걸렸지만 위반 없음 | ✅ |
| `VIOLATED` | 최선을 다했고 위반이 남았다. **예외가 아니라 보고** | ❌ |
| `SOLVER_FAILED` | 모든 QP 가 실패. reference 를 그대로 돌려준다 | ❌ |
| `UNCONSTRAINED` | AG3S 가 피할 것을 아무것도 내지 않았다 | ❌ |

`UNCONSTRAINED` 가 일부러 safe 가 **아니다.** "피할 것이 없었다" 와 "확인했고 아무것도 없었다" 는
여기서 똑같이 생겼지만 같은 주장이 아니다 — 빈 constraint set 은 카메라가 실패했다는 뜻일 수 있다.

`require_certified_geometry=True` 이면, 궤적이 AG3S 가 기술한 모든 것을 비껴도 AG3S 가 본 것을 다
설명하지 못했다고 말했으면 `VIOLATED` 로 내린다 — "clear" 는 모델에 대한 진술이지 세계에 대한
진술이 아니기 때문이다.

**멈출지 실행할지는 여기서 정하지 않는다.** `on_result` 콜백으로 호출자에게 넘긴다. AG3S 가
`GEOMETRY_INCOMPLETE` 에 대해 무엇을 할지 정하지 않는 것과 같은 계약이고, 이유도 같다 — 무엇이
허용 가능한 위험인지는 지각도 최적화도 알 수 없다.

### 5.4 QP backend

측정으로 고른 기본값이 `osqp` 다. `eps_abs = eps_rel = 1e-3` 은 1e-5 보다 게을러 보이지만,
subproblem 이 비볼록 문제의 **linearization** 이므로 모델 자체의 충실도를 넘어 정밀하게 풀 이유가
없다 (inexact SQP). 그리고 **해에서 전체 행을 원해상도로 재검사**하므로 느슨하게 푼 subproblem 은
조용한 오류가 아니라 보고되는 위반으로 드러난다.

`ipqp` 가 `SOLVERS` 목록에 없는 이유는 timing 이 아니다 — RB-Y1 subproblem 에서 heap corruption
으로 **인터프리터를 abort 시켰다.** 제어 루프를 함께 죽일 수 있는 backend 는 타이밍과 무관하게
후보가 아니다.

---

## 6. MuJoCo ↔ TO 매핑 검증

**이것이 이 검토의 핵심 질문이었다.** 결과부터: **정상이다.**

### 6.1 검증한 사슬

```
d.qpos[ left_arm_i ]                     (MuJoCo 가 spawn 한 로봇의 상태)
   → TransportScene.robot_state()         q[13+i]   (DEFAULT_RBY1_JOINTS 순서)
   → ChunkLayout.q_indices                z[i]      (decision vector)
   → optimizer 가 푼다
   → trajectory_to_chunk                  chunk[:, i]
   → apply_action("rby1_16d")             d.ctrl[ act["left_a"][i] ]
   → actuator left_arm_{i+1}_act          → joint left_arm_i        ✅ 같은 joint
```

MuJoCo actuator 이름이 1-based 이고 joint 이름이 0-based 라 `left_arm_1_act` 가 `left_arm_0` 을
구동한다. `LEFT_ARM_ACTS = [f"left_arm_{i+1}_act" for i in range(7)]` 이 그 off-by-one 을 정확히
흡수한다. MuJoCo model 을 열어 `actuator_trnid` 로 확인했다 — 14 개 팔 actuator 전부 일치.

### 6.2 확인된 매핑 (Fig 2a)

| action col | z | q | joint | MuJoCo actuator |
|---:|---:|---:|---|---|
| 0–6 | 0–6 | 13–19 | `left_arm_0..6` | `left_arm_1..7_act` |
| 7 | — | — | left gripper (passthrough) | `gripper_l_act` |
| 8–14 | 7–13 | 6–12 | `right_arm_0..6` | `right_arm_1..7_act` |
| 15 | — | — | right gripper (passthrough) | `gripper_r_act` |
| — | — | 0–5 | `torso_0..5` (parameter) | action channel 없음 |

### 6.3 Gripper 열이 두 번 보호된다

`ChunkLayout` 이 이미 gripper 를 decision variable 에서 빼고 `template` 로 되돌려 놓지만,
`SafePolicy._preserve_grippers` 가 한 번 더 강제한다. 이 두 열이 조용히 바뀌면 손이 엉뚱한 순간에
열리고, 그것은 궤적 오차와 달리 눈에 띄지 않는다.

열 번호는 **layout 에서 유도한다** (`wire.gripper_columns(nq_opt // 2)`). 박아 두면 안 되는
이유를 이 프로젝트가 이미 밟았다: `_run_latch` 가 16D 에서 열 6(= `left_arm_6` 손목)을 gripper
개도로 읽고 있었고, 손목 각도가 우연히 문턱을 넘으면 latch 가 grasp 로 오인한다. 증상이 궤적
오차가 아니라 **권한이 엉뚱한 물체에 붙는 것**으로 나오므로 조용하다.

### 6.4 차원 불일치는 hold 다

Client 는 `actions.shape[1] != 16` 이면 자르거나 채우지 않고 hold 한다
([client.py:172-180](../../trajopt/client.py#L172-L180)). 자르거나 채워 통과시키면 joint 가 한
칸씩 밀린 chunk 가 실행된다 — **형태는 맞고 뜻은 틀린, 가장 위험한 실패다.**

### 6.5 한 번의 실제 solve (Fig 3)

배포 기본값 그대로, 오른손이 지나는 자리에 반지름 60 mm candidate 를 놓고 풀었다.

| | 값 |
|---|---|
| planned window | 8 of 50 (`execution_length = 8`) |
| reference violation | **191.25 mm** |
| violation after TO | **0.0000 mm** (tolerance 0.10 mm) |
| status | `feasible` (반복 1 / 상한 3) |
| ‖Q_TO − Q_ref‖ | 0.6094 rad |
| solve time, 첫 회 | 63.2 ms (OSQP setup 포함) |
| solve time, 정상 상태 | **20.6 ms** (6 회 재solve 의 중앙값, 예산 50 ms) |
| 단계별 (첫 회) | linearize 0.2 · assemble 2.7 · qp 58.2 · check 0.7 ms |
| QP collision rows | 8 × 24 = 192 |
| worst row 신원 | `candidate` / step 4 / `link_right_arm_5` |

Fig 3 가운데 패널이 objective 가 하는 일을 보여 준다: **오른팔 7 개 joint 만 움직이고 왼팔은
거의 그대로다.** Tracking 항이 필요하지 않은 수정을 막고 있다.

`max_violation_pair` 는 T5f 가 *"`violated` 가 **어느** constraint 인가"* 에서 막힌 것을 여는
키다. 판정에 `max_violation_m` 한 숫자만 있으면 팔꿈치가 탁자를 스친 것과 손끝이 사과를 파고든
것을 구별할 수 없고, 대응도 다르다.

### 6.6 회귀

```
python -m pytest tests/trajopt -q              →   415 passed (28.4 s)
python -m pytest tests/trajopt tests/ag3s -q   →  1026 passed (187.5 s)   # 7 절의 수정 뒤
```

---

## 7. 발견 — 둘 다 고쳐졌다 (2026-09-28)

이 검토가 찾은 것은 둘이고, 같은 날 commit `7a1cdc5`
(*"거리장을 계층별로 기록하고, hold 청크와 왼손가락 이름표를 고친다"*) 가 둘을 고쳤다. 아래 수치는
**그 commit 을 읽고 적은 것이 아니라 고친 코드를 다시 돌려 잰 것**이다 (Fig 4).

| | 발견 | 수정 전 | 수정 후 |
|---|---|---|---|
| 7.1 | `_current_state()` 의 hold chunk | 14/14 joint 열이 엉뚱한 joint · gripper `0.0`(닫힘) | **0/14** · gripper `1.0`(열림) |
| 7.2 | URDF ↔ MJCF 왼손가락 이름표 | `ee_finger_l1` FK 오차 **6.001 mm** | **0.00084 mm** |

회귀: `tests/trajopt` + `tests/ag3s` **1026 passed**. Collision sphere 는 218 개 × 16 자세에서
**하나도 움직이지 않았다** (최대 이동 0.000000 mm).

### 7.1 `SafeRemoteClient._current_state()` 가 20-vector 를 16D layout 으로 읽었다

**위치** [`client.py:415-500`](../../trajopt/client.py#L415-L500) · **Fig 2b**(전) · **Fig 4 오른쪽**(후)

`self._scene` 은 `TransportScene.attach(m, d)` 이고 그 `robot_state()` 는 `DEFAULT_RBY1_JOINTS`
순서의 **20-vector** (`torso_0..5, right_arm_0..6, left_arm_0..6`) 를 돌려준다. 그런데
`_current_state()` 는 그 앞 14 개를 "왼팔 7 + 오른팔 7" 로 **읽고 있었다** (수정 전 코드):

```python
arms = state[:2 * n]                                    # = torso_0..5 + right_arm_0..6 + left_arm_0
return np.concatenate([arms[:n], [0.0], arms[n:2*n], [0.0]])
```

결과: **16 열 중 14 개 joint 열이 전부 엉뚱한 joint 값을 받았다.** 열 0(왼팔 어깨)에 `torso_0`
이, 열 14(오른손목)에 `left_arm_0` 이 들어갔다. 부수적으로 gripper 열에 채우던 `0.0` 은 코드
주석이 말하는 "열림" 이 아니라 **닫힘**이었다 (`build_obs` 의 `grip_norm = |qpos|/|OPEN|` 규약에서
1.0 이 열림, `apply_action` 은 `ctrl = grip · RBY1_GRIPPER_OPEN`).

**로봇에 도달하지는 않았다.** 이 chunk 는 server 응답이 **아예 없을 때**만 만들어지고
(`_hold(..., "error")`), `pi05_infer.py` 의 제어 루프는 hold 프레임에서 client 의 chunk 대신
자기 `rby1_state()` 를 쓴다 ([pi05_infer.py:1637-1648](../../../pi05_TO_hybrid/rby1_bringup/pi05_infer.py#L1637-L1648)).
그쪽은 올바르다.

**그래도 결함이었다.** 이 값은 `{"actions": hold}` 로 반환되어 형태 계약을 지키는 척하고, 다른
호출부(분석 script, 다른 bringup)가 그것을 실행하면 팔이 torso 각도로 지령된다. `_hold` 의
docstring 이 "호출부가 `chunk[chunk_step]` 을 무조건 인덱싱한다" 를 근거로 이 chunk 를 만드는
이상, 그 chunk 는 맞아야 한다.

#### 고쳐진 것

`_current_state()` 가 이제 **`ChunkLayout.rby1` 의 이름 매핑을 그대로 쓴다** — 매핑을 두 곳에
따로 두면 한쪽이 조용히 한 칸 밀리기 때문이다. Gripper 열은 씬이 `gripper_norm()` 을 내놓으면
그 값을, 못 내놓으면 `HOLD_GRIPPER_NORM = 1.0`(열림)을 쓴다. **길이가 20·16·14 중 어느 것도
아니면 추측하지 않고 `ValueError` 로 거절한다.**

재측정 (씬이 `q[i] = i` 를 주는 가짜 20-vector):

| | 전 | 후 |
|---|---|---|
| joint 열 중 틀린 것 | **14 / 14** | **0 / 14** |
| 열 0 (`left_arm_0`) 이 받는 값 | `torso_0` | `left_arm_0` |
| 열 14 (`right_arm_6`) 이 받는 값 | `left_arm_0` | `right_arm_6` |
| gripper 채움값 | `0.0` = **닫힘** (주석은 "열림" 이라 적혀 있었다) | `1.0` = **열림** |
| 18-D 를 주면 | 조용히 잘라 읽음 | `ValueError` |

> **열림이 무조건 옳은 것은 아니다.** 물건을 쥔 채 hold 가 걸리면 열림 값은 그것을 **놓는다.**
> 그래서 씬이 실제 개도를 내놓는 쪽(`gripper_norm()`)이 정상 경로이고 상수는 마지막 수단이다.
> 현재 `TransportScene` 에는 `gripper_norm()` 이 없으므로 이 경로는 상수 1.0 으로 간다.

### 7.1b 같은 commit 이 함께 막은 조용한 실패 두 개

이 검토의 발견은 아니지만 같은 commit 에 들어 있고, 성질이 같아서 적어 둔다 — **켰다고 믿은
기능이 아무 일도 하지 않는데 아무 데도 안 찍히는** 종류다.

- **기록기가 cuRobo backend 에서 75 chunk 전부 실패하고 빈 디렉터리만 남겼다**
  (`CuroboEsdfField` 에 `max_distance` 가 없다). `print` 한 줄로 삼켜져 실행이 끝까지 갔다.
  이제 지원 안 되는 backend 는 **시작할 때** `SystemExit`, 런타임 실패는 첫 번째에 크게 외친다.
- **`link_pose_error` 가 `ee_left`/`ee_right` 를 한 번도 재지 않고 있었다** — MJCF 이름이
  `EE_BODY_L`/`EE_BODY_R` 인데 alias 를 안 쓰고 `KeyError` 로 조용히 건너뛰었다. *"frame 이
  맞는가"* 를 묻는 유일한 도구가 손바닥 둘을 빼놓고 있었던 것이다. 이제 alias 를 쓰고, 그래도
  못 찾으면 **못 쟀다고 경고한다** — 반환 dict 에 키가 없는 것과 0 mm 는 전혀 다른 뜻이다.

### 7.2 URDF `ee_finger_l1` 은 MJCF `ee_finger_l2` 다

**Fig 1 오른쪽 패널**(전) · **Fig 4 왼쪽·가운데**(후)

처음 측정에서 URDF FK 와 MuJoCo FK 가 어긋난 link 는 왼쪽 finger 뿐이었고, 16 개 자세 전부에서
값이 **일정했다**:

```
Δp = (−6.000, 0.000, 0.0005) mm        상대 회전 = z 축 180°
```

자세에 따라 변하지 않으므로 joint 축이 아니라 frame 의 문제다. **나는 이것을 "왼쪽 finger 의
frame 정의가 다르다" 로 읽었는데, 교차 비교가 더 단순한 답을 냈다** — **두 파일이 왼손가락 둘의
이름을 서로 반대로 붙였다:**

| 비교 | 위치 오차 | 회전 |
|---|---:|---:|
| URDF `ee_finger_l1` vs MJCF `ee_finger_l1` | 6.001 mm | 180° |
| **URDF `ee_finger_l1` vs MJCF `ee_finger_l2`** | **0.00084 mm** | **0.000°** |
| **URDF `ee_finger_l2` vs MJCF `ee_finger_l1`** | **0.00084 mm** | **0.000°** |
| URDF `ee_finger_r1` vs MJCF `ee_finger_r1` | 0.0011 mm | 0.000° |

`link_*_arm_6` 기준으로 URDF 는 양팔 모두 `finger1` 을 `x = +3 mm` 에 두는데 MJCF 는 **왼손만**
거울이다 (`l1` 이 `x = −3 mm`). 6.000 mm 는 `|+3 − (−3)|` 이다.

**어느 파일도 틀리지 않았다.** joint 축과 limit 까지 양쪽이 각자 일관되어 있어
(`finger1` = axis `−x` · `[−0.05, 0]`, `finger2` = axis `+x` · `[0, 0.05]`) frame 과 축이 함께
뒤집힌 것이고, **물리적 개폐 동작은 같다.** 고칠 기하가 없고 고칠 것은 두 이름표의 짝이다.

#### 고쳐진 것

`MJCF_BODY_ALIASES` 에 두 줄이 추가됐다 — `ee_finger_l1 ↔ ee_finger_l2` 가 서로를 가리킨다
([mujoco_source.py](../../ag3s/experiments/sources/mujoco_source.py)). 이미 `ee_left → EE_BODY_L`
이 하던 것과 같은 일이다.

vendored robot description(URDF) 쪽 joint origin 을 MJCF 에 맞추는 길을 **안 고른 이유**: URDF 를
고치면 FK 가 바뀌어 **제약 모델의 sphere 위치가 전부 움직이고 회귀 기준선이 또 이동한다.**
이름표 두 줄로 끝나는 일에 그 대가를 낼 이유가 없다.

#### 확인: collision model 은 하나도 움직이지 않았다

이것이 이 수정에서 확인해야 할 유일한 안전 관련 질문이다 — alias 를 바꾸면
`gap_filling_capsules` 가 **다른 MJCF body 의 mesh 를 재게** 되기 때문이다.

| | 값 |
|---|---|
| sphere 수 | 218 (전) → 218 (후), link 순서 동일 |
| finger sphere 수 | 44 |
| **최대 중심 이동** | **0.000000 mm** (218 sphere × 16 자세) |

네 finger body 가 같은 mesh 를 자기 frame 의 같은 자리에 놓으므로 local capsule 이 동일하고,
따라서 어느 이름으로 재든 같은 capsule 이 나온다. Fig 4 가운데 패널이 두 모델의 finger sphere 를
겹쳐 그린 것이다 — 굵은 주황(전) 위에 가는 파랑(후)이 정확히 얹힌다.

**판정: 기하는 처음부터 옳았고, 틀린 것은 진단 도구가 읽던 이름이었다.** 즉 이 수정으로 로봇의
거동은 바뀌지 않고, 바뀐 것은 *"frame 이 맞는가"* 를 물었을 때 돌아오는 답이다. 왼손·양팔 과제로
갈 때 6 mm 를 의심할 필요가 없어졌다.

---

## 8. 알려진 한계 (코드가 스스로 적어 둔 것)

- **SQP 는 지역해만 준다.** Collision avoidance 는 비볼록이라 초기값에 따라 다른 지역해에 간다.
  정책 chunk 를 초기값으로 쓰는 것이 합리적 선택이지만 전역 최적을 주장하지 않는다.
- **운동학 한계가 soft 하다.** Reference 가 로봇의 가속 한계를 넘으면 해에도 잔여 위반이 남을 수
  있고 `limit_report` 가 그 값을 보고한다. Hard constraint 가 아니다.
- **SEAM 이 TO 의 수정을 되먹지 않는다.** `w_continuity` 는 완화이지 해결이 아니다 ([3.3](#33-w_continuity-가-왜-있는가)).
- **P = K 는 예지력을 버린 것이다.** 실행 창만 계획하면 회피를 미룰 자리가 없어지지만, 8 스텝
  (533 ms) 너머를 못 보므로 빠져나올 수 없는 구석으로 들어갈 수 있다. T6f 가 의도적으로 택한
  trade 다.
- **`configs/rby1.yaml` 은 server 가 읽지 않는다.** `plan_horizon: 32` 등 그 파일의 값은
  `serve_safe` 의 flag 기본값과 다르다 ([2.4](#24-몇-스텝을-다듬는가--p--k--8)).

---

## 9. 재현

```bash
# 매핑 · FK 일치도 · solve figure
cd /mnt/dev/work
PYTHONPATH=/mnt/dev/work MUJOCO_GL=osmesa \
  src/openpi/.venv/bin/python <각 figure script>

# 회귀
PYTHONPATH=/mnt/dev/work src/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q
```

각 figure 의 `.json` sidecar 에 그림이 쓴 숫자가 전부 들어 있다.

[7](#7-발견--둘-다-고쳐졌다-2026-09-28) 의 재측정은 두 가지를 **코드를 읽지 않고** 확인한다.
둘 다 스스로 확인할 수 있게 방법만 적어 둔다:

1. **hold chunk 의 매핑** — `robot_state()` 가 `q[i] = i` 를 주는 가짜 씬을 물려
   `SafeRemoteClient._current_state()` 를 부르면, 각 열에 들어온 값이 곧 그 열이 받은 q index 다.
   열 이름과 `DEFAULT_RBY1_JOINTS[값]` 이 같아야 한다.
2. **alias 가 sphere 를 움직였는가** — `MJCF_BODY_ALIASES` 에서 `ee_finger_*` 두 줄을 뺀 사본으로
   `gap_filling_capsules` 를 한 번 더 돌려 두 `UrdfSphereChain` 을 만들고, 같은 q 에서
   `sphere_centers_numeric` 을 비교한다. **이것이 이 수정의 유일한 안전 관련 질문이다** —
   alias 는 어느 MJCF body 의 mesh 를 재는지를 바꾸기 때문이다. 측정값은 0.000000 mm 였다.
