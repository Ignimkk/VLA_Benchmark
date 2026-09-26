# T6f — 구현

> writer: ag3s-implementer (A1) · 2026-09-26 · 읽는 쪽: verifier, scribe, lead

## 무엇을 했나 (평이한 요약 먼저)

**TO 가 다듬는 구간을 실행되는 창으로 줄였다.** 이 패키지에서 "계획 지평" 과 "다듬는 구간" 은
두 숫자가 아니라 **한 숫자**였다 — `HorizonConfig.planned` 하나가 결정 변수 개수·충돌 행이
걸리는 스텝·chunk 에 되쓰는 행·`max_violation` 을 재는 범위를 전부 정한다. 그래서 새 기구를
만들지 않고 그 값을 `execution_length` 로 따라가게 했고(`plan_horizon` 의 새 기본값
`PLAN_EXECUTION_WINDOW`), 8 을 다시 박지 않도록 **sentinel** 로 뒀다. `--plan-horizon full` 이나
숫자를 주면 예전 동작으로 되돌아간다.

**팔 굵기는 설정으로 조절 가능하게 하고 기본값은 건드리지 않았다.** `UrdfSphereChain` 에
`capsule_radius_scale` · `max_sphere_radius` 를 더하고 기존 `sphere_spacing` ·
`max_spheres_per_capsule` 을 서버 flag 로 꺼냈다. 넷 다 안 주면 구 집합이 글자 그대로 같다.
**덮개를 잃는 설정이면 생성 시점에 크게 찍고**(`coverage_report()`) 그 목록을
`coverage_shortfall` 로 들고 있는다. self-filter 모델에는 이 통로를 **아예 만들지 않았다**.

**관절 매핑은 읽고 확인하고 테스트로 고정했고, 그 과정에서 gripper 열 별건이 실제로 물어야 할
곳을 물고 있었다.** `safe_policy.py:411` 의 `column = 6 if left else 13` 은 grasp latch 가 읽는
열이고, 16D 에서 열 6 은 `left_arm_6`(손목) · 열 13 은 `right_arm_5` 다. latch 는 **값이 0.85
아래면 "닫혔다"** 로 읽으므로(`grasp_latch.py:71`), 손목 각도가 0.85 rad 아래인 프레임은 손이
열려 있어도 파지로 보인다 — 궤적 오차가 아니라 **권한이 엉뚱한 물체에 붙는** 증상이라 조용하다.
`wire.gripper_columns` 로 유도하게 고쳤고, 같은 이유로 `_preserve_grippers` 와
`safe_replay.py` 의 검사도 모듈 상수 대신 레이아웃에서 가져온다.

타원체는 **구현하지 않았다** — 무엇을 바꿔야 하는지와, 이 자리에서 계산한 기하로 "한 link 에
타원체 하나" 가 오히려 두꺼워진다는 점을 아래 한 절로 적었다.

---

## 고칠 것 1 — TO 를 실행 창에만 건다

### 왜 "다듬는 구간만 줄인다" 가 별도의 선택지가 아닌가 (근거)

코드를 읽고 정한 결과다. `planned` 한 값이 아래 넷을 동시에 정한다:

| 무엇 | 어디 |
|---|---|
| 참조 궤적을 자르는 길이 | `refiner.py:105` `planned = min(config.horizon.planned, chunk.shape[0])` |
| 결정 변수·충돌 행의 스텝 수 | `refiner.py:75-80` → `TrajectoryOptimizer.horizon` (`sqp.py:66`) |
| chunk 에 되쓰는 행 | `types.py:142` `out[:planned, action_columns] = trajectory.T` |
| `max_violation` 을 재는 범위 | `sqp._finish` → `linearizer.worst_row(trajectory, ...)` |

즉 **"다듬는 구간" 이라는 별개의 창은 존재하지 않는다.** 그것만 줄이려면 "결정 변수는 32 스텝,
충돌 행은 앞 8 스텝" 이라는 새 기구를 만들어야 하는데, 그것은 **더 나쁜 선택**이다:

* 예지력의 실체가 곧 *앞선 스텝의 충돌 행*이다. 뒤 24 스텝에 행을 안 걸면 예지력은 그대로
  사라진다 — 즉 얻는 것이 없다.
* 비용은 거의 그대로 남는다. 가장 비싼 것이 32 스텝 FK 와 야코비안이다(`linearize.py:438-446`).
* 그런데도 뒤 24 스텝이 chunk 에 **되쓰인다** — 실행되지 않는 값이 다음 chunk 의 연속성 기준으로
  들어간다.

그래서 **둘을 같이 줄였다.** `plan_horizon = PLAN_EXECUTION_WINDOW` → `planned = execution_length`.

### 어떻게 8 을 다시 박지 않았나

`OPEN_LOOP_HORIZON = 8` 은 클라이언트(`pi05_TO_hybrid/rby1_bringup/pi05_infer.py:142`)에 있고
**와이어로 오지 않는다.** 서버 쪽에서 그 수를 들고 있는 것은 `horizon.execution_length` 하나이고,
sentinel 이 그 값을 따라간다(`config.py` `planned` property). 테스트가 이것을 지킨다 —
`execution_length=5` 로 지으면 `planned == 5` 다.

### 함께 고친 것: 겹치는 스텝이 없으면 warm start 를 쓰지 않는다

`sqp.py` 의 warm start 는 지난 궤적을 `execution_length` 만큼 shift 하고 꼬리를 복제해 섞는다.
`execution_length >= horizon` 이면 **지난 계획과 겹치는 스텝이 하나도 없고**, shift 는 실행 창보다
앞선 자세(`_previous[:, -1]`)를 첫 iterate 에 섞어 넣는다. 그때는 참조가 유일하게 옳은 시작점이라
건너뛴다. 이 가드는 `execution_length < horizon` 인 예전 설정에서는 **한 번도 걸리지 않는다**.

### 되돌리는 법

```bash
--plan-horizon 32     # T6f 이전 기본값
--plan-horizon full   # chunk 50 스텝 전부
```

---

## 고칠 것 2 — 팔을 실제 굵기로

### 손잡이 넷, 기본값은 그대로

| 인자 / flag | 기본 | 무엇이 내려가나 | 덮개 |
|---|---|---|---|
| `sphere_spacing` / `--sphere-spacing` | 1.0 | 팽창 항 `(간격/2)²` | **유지** |
| `max_spheres_per_capsule` / `--max-spheres-per-capsule` | 8 | (위를 실제로 듣게 한다) | 유지 |
| `capsule_radius_scale` / `--capsule-radius-scale` | 1.0 | capsule 반지름 자체 | **잃는다 → 경고** |
| `max_sphere_radius` / `--max-sphere-radius` | 없음 | 팽창까지 끝난 최종 반지름의 상한 | **잃는다 → 경고** |

간격은 **줄인 반지름이 아니라 URDF 반지름**으로 정한다. 줄인 반지름으로 정하면 반지름을 내릴 때
구가 자동으로 늘어나 "가늘게" 와 "촘촘하게" 가 한 손잡이에 묶이는데, 둘은 값이 다르다 — 하나는
제약 행을 늘리고 하나는 덮개를 잃는다.

### 이 자리에서 계산한 기하 (측정이 아니다 — URDF 만 읽는 정적 계산)

`link_*_arm_5` 의 URDF capsule 은 **r=75 mm, L=250 mm** 이고 기본 설정의 구 5 개가
`sqrt(75² + 31.25²) = 81.25 mm` 다 — T6d 가 인용한 **81.2 mm** 가 여기서 나온다. `--links arms`
범위(gap-filling capsule 제외, URDF 만)에서:

| 설정 | 구 수 | `arm_4` | `arm_5` | 덮개 |
|---|---:|---:|---:|---|
| 기본 | 46 | 38.41 mm | **81.25 mm** | 전부 |
| `--sphere-spacing 0.4 --max-spheres-per-capsule 16` | 94 | 35.65 mm | **76.28 mm** | 전부 |
| `--sphere-spacing 0.25 --max-spheres-per-capsule 24` | 140 | 35.27 mm | 75.53 mm | 전부 |
| `0.4 / 16` + `--max-sphere-radius 0.068` | 94 | 35.65 mm | **68.00 mm** | **12 중 4 capsule 부족** |

읽을 점 둘:

1. **촘촘하게 하는 것만으로는 76.3 mm 가 하한이다.** 팽창 항은 0 으로 갈 수 있지만 URDF capsule
   자신의 75 mm 는 남는다. MJCF mesh 실측 **65.4~68.4 mm** 에 닿으려면 반지름 자체를 줄여야
   하고, 그것은 **URDF capsule 을 덮지 않는 설정**이다.
2. 그러나 "URDF capsule 을 안 덮는다" 가 곧 "실제 팔을 안 덮는다" 는 아니다 — URDF 가 mesh 보다
   두껍기 때문이다. `coverage_report()` 는 **아는 것만 말한다**: 기준이 URDF capsule 이라고 밝히고,
   mesh 에 대해서는 아무 주장도 하지 않는다.

### 덮개를 잃으면 어떻게 말하나

경고는 `UrdfSphereChain.__init__` 에서 찍는다 — 서버·실험 스크립트·테스트가 각자 이 클래스를
짓기 때문에 진입점에 두면 한 곳이 빠지고, 빠진 실행이 기록에 "예전과 같은 모델" 로 남는다.
`serve_safe` 는 그것을 한 번 더 찍고, self-filter 는 손대지 않았다는 줄을 같이 찍는다.

```
!!! THIS SPHERE MODEL DOES NOT COVER THE ROBOT !!!
    4 of 12 capsule(s) are thinner than the URDF says, worst by 8.3 mm (sphere_spacing=0.4,
    max_spheres_per_capsule=16, capsule_radius_scale=1, max_sphere_radius=68.0 mm).
    빼는 것과 같은 성질의 flag 다 — ... 기준은 **URDF capsule** 이고, URDF 자체가 mesh 보다 두꺼울 수 있다.
    - link_left_arm_5: 구 10 개 × 68.0 mm, 덮으려면 76.3 mm 필요 (8.3 mm 부족; URDF capsule r=75.0 L=250.0 mm)
    - link_left_arm_0: 구 4 개 × 68.0 mm, 덮으려면 75.9 mm 필요 (7.9 mm 부족; URDF capsule r=75.0 L=70.0 mm)
```

`--no-safe` 와 함께 주면 서버가 시작하지 않는다 (`--exclude-links` 와 같은 규약 — 아무 일도 안
하는 flag 를 받아 놓고 뜨면 효과가 있었다고 믿는다).

### self-filter 는 건드리지 않았다

`build_robot_model(scene)` 에는 `sphere_options` 통로가 **없다**. `build_ag3s` 안에서 dict 는
`build_constraint_robot_model(...)` 두 호출로만 간다. 소스로 그 경계를 고정하는 테스트를 뒀다 —
`--exclude-links` 가 쓰는 것과 같은 방식이다.

---

## 고칠 것 3 — `esdf_margin`

**바꾸지 않았다.** `--esdf-margin` (기본 `0.05` = **50 mm**) 하나로 조절된다
(`serve_safe.py` → `to_config.collision.esdf_margin`). 제약은
`d_esdf(구 중심) - 구 반지름 - esdf_margin >= 0` 이므로 기본 팔뚝에서 중심 기준 요구 자유공간이
`81.2 + 50 = 131 mm` 다. flag 에 그 설명을 붙이고, 서버 시작 로그에 값을 찍게 했다. **값은
사용자가 정한다.**

---

## 확인할 것 4 — 관절 매핑 (TO 안쪽)

### 읽어서 확인한 길

```
chunk[H, 32] --chunk_to_trajectory--> Q[nq_opt=14, planned] --full_q--> q[nq_model=20, planned]
        --sphere_centers_(numeric|symbolic)--> 구 중심 --linearize--> QP 행
        --split--> Q' --trajectory_to_chunk(template=chunk)--> chunk'[H, 32]
```

* 열 → q 는 **이름으로** 찾는다 (`types.py:193-198`, `order[name]`). `DEFAULT_RBY1_JOINTS` 는
  `torso 0-5 · right_arm 6-12 · left_arm 13-19` 순서라, 열 0(=`left_arm_0`)이 q 13 으로 간다 —
  열 번호를 q 번호로 믿으면 왼팔 명령이 torso 로 간다. 지금 코드는 그러지 않는다.
* 야코비안 열 순서도 같은 곳에서 온다: `linearize.py:307-308` 이 `q[layout.q_indices]` 로 미분하고,
  그 순서가 결정 변수 벡터의 순서다. **한 출처에서 나오므로 갈라질 수 없다.**
* 되쓰기는 `template` 복사 뒤 `out[:planned, action_columns] = trajectory.T` 다 — gripper 와
  padding 은 만지지 않으므로 바이트 동일하다.

**왕복 동일성만으로는 부족하다.** 두 변환이 같은 열 목록을 쓰므로 좌우가 뒤바뀌어 있어도 왕복은
항등이다. 그래서 테스트는 셋을 따로 본다: 열↔관절 이름표 · 왕복 · **FK 국소성**(왼팔 열을 흔들면
오른팔 구는 1 μm 도 안 움직인다, 야코비안도 같이 확인). 세 번째가 좌우 교환을 잡는다.

결론: **chunk↔결정변수↔chunk 길은 맞다.** 틀린 것은 그 옆의 gripper 열이었다.

### 별건 — gripper 열 (고쳤다)

| 자리 | 전 | 후 |
|---|---|---|
| `safe_policy.py:411` (grasp latch 가 읽는 열) | `6 if left else 13` = 16D 에서 `left_arm_6` · `right_arm_5` | `wire.gripper_columns(nq_opt//2)` = `(7, 15)` |
| `safe_policy._preserve_grippers` | `wire.GRIPPER_COLUMNS` (기본 차원 상수) | `self.gripper_columns` |
| `safe_replay.py:242` (매 프레임 검사) | `wire.GRIPPER_COLUMNS` | `safe.gripper_columns` |

latch 는 `< 0.85` 를 "닫혔다" 로 본다. 손목 각도는 rad 이고 그 문턱 아래에 있는 것이 흔하므로,
**target 이 3 프레임 확인되면 손이 열려 있어도 `attach` 가 불릴 수 있는 배선**이었다. 그러면 쥐기
전에 조작 대상이 attached 로 넘어가 장애물에서 빠진다. 실제로 몇 프레임에서 그렇게 됐는지는
기록을 봐야 안다 — **여기서는 코드 사실만 적는다.** (fixture 도 같은 14D 잔재를 갖고 있었다:
`tests/trajopt/fixtures.py` 가 열 6·13 에 "gripper" 값을 넣으면서 실제 gripper 열은 0 으로 뒀다.
레이아웃에서 유도하게 고쳤다.)

---

## 보고만 할 것 5 — 타원체 (구현하지 않았다)

### 무엇을 바꿔야 하나

ESDF 질의가 **점 기반**이라는 것이 핵심이다. 지금 행은
`d_esdf(p) - r - margin >= 0` 이고 기울기는 `∇d(p) · ∂p/∂q` 다 (`linearize.py:843-856`).
타원체의 자연스러운 일반화는 **지지함수**다 — `benchmark/knows_vla/cbf/ellipsoid.py` 의 규약
`E = {c + Q^{1/2}u : ||u|| <= 1}`, `support(n) = n·c + sqrt(n'Qn)` 로 쓰면

```
d_esdf(c) - sqrt(n' Q(q) n) - margin >= 0,    n = ∇d_esdf(c)
```

구는 `Q = r²I`, `|n| = 1` 이라 `sqrt(n'Qn) = r` — 지금 행이 이것의 특수한 경우다. 그래서 질의
방식을 안 바꾸고도 갈 수 있다. 값이 드는 곳은 그 뒤다:

| 바꿀 곳 | 크기 | 무엇 |
|---|---|---|
| `RobotCollisionModel` protocol (`ag3s/types.py`) | 중 | `(centre, radius)` → `(centre, Q)`. 이 Protocol 을 만족하는 모든 구현이 따라온다 |
| `UrdfSphereChain.sphere_centers_(numeric\|symbolic)` | 중 | link 회전이 필요하다: `Q(q) = R(q) Q_local R(q)'`. `_link_transforms` 가 이미 4×4 를 들고 있어 자료는 있다 |
| `CollisionLinearizer` | **큼** | `robot_radii` 가 (S,) 스칼라 벡터로 열 군데 넘게 흐른다 — `_esdf_clearance` · candidate 행 · plane 행 · `d_safe` (S×M) · `SceneSnapshot` · `scene_from_constraint_set` |
| 기울기 | **가장 큼** | `-sqrt(n'Qn)` 의 회전 미분은 `(n × Qn)/sqrt(n'Qn)` (KNOWS Eq. (10), `ellipsoid.rotate_shape` 의 주석이 유도해 둔 그 식)이고 link 의 **각속도 야코비안**이 필요하다. `linearize.py` 는 지금 중심 위치만 미분한다 — 없는 기구다 |
| AG3S self-filter | 중 | `robot_sphere_mask` 가 점-구 판정이다. `(x-c)' Q^{-1} (x-c) <= 1` 로 바뀐다 |
| 기록·그림 | 소 | `sphere_radii` 키, verifier 의 씬 그림 |

**candidate/plane 행에는 근사가 하나 들어간다.** 구-구 행의 분리 방향은 중심 차이로 정확했지만,
타원체-구에서는 최적 분리 방향이 최적화의 답이다 (`knows_vla` 의 `optimal_normal`). 선형화
시점에 방향을 고정하는 지금의 규약과 맞물리면 실용적으로는 쓸 수 있지만, "정확한 행" 이라는 말은
못 쓴다.

### 그런데 한 link 에 타원체 하나는 **더 두껍다** (이 자리에서 계산한 기하)

`arm_5` capsule(r=75, L=250 mm)을 담는 회전 타원체의 최소 허리 반지름:

| 장축 반지름 c | 200 mm (겨우 감쌈) | 225 | 250 | 300 | 400 | 600 |
|---|---:|---:|---:|---:|---:|---:|
| 최소 허리 a | 122.6 mm | 95.0 | 88.8 | 83.4 | 79.2 | 76.8 |

최소 부피 해는 `a=89.4, c=246.5 mm` 다. **지금의 구 사슬 81.25 mm 보다 허리가 두껍다** — 허리를
81 mm 아래로 내리려면 타원체가 link 보다 ~330 mm 이상 길어져야 하고, 그러면 양 끝으로 삐져나온다.
즉 **"긴 link 는 타원체가 잘 맞는다" 는 capsule 을 감쌀 때는 성립하지 않는다.** 타원체가 이기는
것은 `knows_vla` 가 한 대로 **mesh 점군에 직접 MVEE 를 맞출 때**이고, 그러면 이득의 출처는
primitive 가 아니라 **원본 기하**(mesh 65.4~68.4 mm 대 URDF 75 mm)다.

**얼마나 큰 일인가**: protocol · 두 FK 경로 · linearizer 의 스칼라 반지름 가정 · 각속도 야코비안 ·
self-filter 까지 손대야 하고, 각속도 야코비안은 새로 만드는 기구다. 같은 노력의 **작은 앞단계**가
있다 — 이번에 만든 `capsule_radius_scale` / `max_sphere_radius` 로 mesh 실측값을 넣어 보는 것.
그것이 "primitive 를 바꿔야 하는가" 와 "기하 출처를 바꿔야 하는가" 를 먼저 가른다.

---

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/trajopt/config.py:23-28` | `PLAN_EXECUTION_WINDOW` sentinel | 8 을 두 번째 자리에 박지 않는다 |
| `benchmark/trajopt/config.py:65` | `plan_horizon` 기본 `32` → sentinel | 고칠 것 1 |
| `benchmark/trajopt/config.py:86-126` | `validate` 가 sentinel 을 알고, `planned` 가 그것을 푼다 · `plans_execution_window_only` | 오타 sentinel 이 "전체 계획" 으로 안 떨어진다 |
| `benchmark/trajopt/sqp.py:116-127` | 겹치는 스텝이 없으면 warm start 를 건너뛴다 | 실행 창보다 앞선 자세를 첫 iterate 에 섞지 않는다 |
| `benchmark/trajopt/refiner.py:12-25, 187-200` | 머리말·연속성 기준 주석 | 무엇이 바뀌었고 정렬은 왜 그대로인가 |
| `benchmark/ag3s/robot_models/urdf_sphere_chain.py:37-73` | `CoverageShortfall` · `_COVERAGE_TOL` | 덮지 못한 것을 **자료로** 들고 있는다 |
| `…/urdf_sphere_chain.py:395-415` | 굵기 인자 넷 + 검증 | 고칠 것 2 |
| `…/urdf_sphere_chain.py:462-479` | 구를 지으며 shortfall 을 모으고 생성 시점에 경고 | 진입점이 여럿이라 클래스가 찍는다 |
| `…/urdf_sphere_chain.py:507-570` | `_capsule_sphere_centres` 가 `(centres, radius, required)` · `coverage_report()` | 기준(URDF capsule)을 밝히고 말한다 |
| `benchmark/ag3s/robot_models/__init__.py` | `CoverageShortfall` export | — |
| `benchmark/ag3s/experiments/reports/grounding_report.py:93-122` | `build_constraint_robot_model(..., sphere_options=None)` | **제약 모델만**. 기본값을 두 번 적지 않는다 |
| `benchmark/trajopt/serve_safe.py:80-115` | `sphere_options(args)` · `resolve_plan_horizon(value)` | 준 flag 만 넘긴다 / 문자열을 config 값으로 |
| `benchmark/trajopt/serve_safe.py:118-200` | `build_ag3s(..., constraint_sphere_options=)` + 굵기 로그 | 시작할 때 크게 말한다 |
| `benchmark/trajopt/serve_safe.py:331-361` | flag 5 개 (`--plan-horizon` · `--sphere-*`) + `--esdf-margin` 설명 | 되돌릴 수 있게 |
| `benchmark/trajopt/serve_safe.py:422-434` | `--no-safe` 와의 조합 거절 · `--plan-horizon` 을 파싱 단계에서 검사 | GPU 에 체크포인트를 올리기 전에 죽는다 |
| `benchmark/trajopt/serve_safe.py:481-489` | `plan_horizon` 을 기록 meta 에 **항상**, `sphere_options` 는 줬을 때만 | T6d 기록과 구별되어야 한다 |
| `benchmark/trajopt/safe_policy.py:152-157` | `self.gripper_columns` 를 레이아웃에서 유도 | 확인할 것 4 별건 |
| `benchmark/trajopt/safe_policy.py:417-420` | latch 가 읽는 열 `6/13` → 유도된 `(7, 15)` | **손목을 gripper 로 읽고 있었다** |
| `benchmark/trajopt/safe_policy.py:497-502` | `_preserve_grippers` 가 유도된 열을 쓴다 | 14D 에서도 맞다 |
| `benchmark/trajopt/experiments/safe_replay.py:240-244` | 같은 자리에서 열을 가져온다 | 검사와 서버가 갈라지지 않는다 |
| `tests/trajopt/fixtures.py:22, 168-178` | gripper 열을 레이아웃에서 유도 | 14D 잔재 (열 6·13) |
| `tests/ag3s/test_robot_models.py:270` | `_capsule_sphere_centres` 3-튜플 + `required == radius` | 기본값에서는 덮개를 잃지 않는다 |
| `tests/trajopt/test_execution_window.py` | **새 파일** 18 개 | 창의 경계·되돌리기·warm start·flag |
| `tests/trajopt/test_joint_mapping.py` | **새 파일** 14 개 | 이름표·왕복·FK 국소성·gripper 열 |
| `tests/trajopt/test_sphere_thickness.py` | **새 파일** 18 개 | 기본값 고정·가늘어짐·경고·flag·경계 |

## 단위 검증

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -u -m pytest tests/ -q
```

**841 passed** (직전 791 + 새 50). 기존 791 개는 하나도 고치지 않았다 — 손댄 두 곳
(`fixtures.py` · `test_robot_models.py`)은 14D 잔재와 튜플 폭이고 단정은 그대로다.

기본값이 실제로 옛 공식과 같은지는 테스트가 **공식을 따로 다시 계산해서** 맞춘다
(`test_sphere_thickness.py::_old_formula`) — 같은 코드를 불러 비교하면 어떤 변경도 통과한다.

## verifier 가 알아야 할 것

* **새 flag**: `--plan-horizon`(기본 `execution`) · `--sphere-spacing` · `--max-spheres-per-capsule`
  · `--capsule-radius-scale` · `--max-sphere-radius`(뒤 넷은 기본 "안 줌" = 예전과 같음).
* **기본값이 바뀐 것은 하나**: `horizon.plan_horizon` 32 → `execution`(=8). `--plan-horizon 32` 가
  T6d 조건이다. **구 집합·`esdf_margin` 기본값은 안 바뀌었다.**
* **회귀 기준선 셋은 SQP 이전 값이라 영향이 없어야 한다** (`위반으로 시작 14/15` ·
  `has_target 9/15` · `frame0 clearance_before +0.157 mm`). 그런데 **`feasible`/`violated`/`해소`
  개수는 달라진다** — `max_violation` 을 재는 범위가 32 → 8 스텝이다.
* **기록 형식**: `meta` 에 `plan_horizon` 키가 **항상** 생긴다(줬든 안 줬든). npz 의 `clearance` ·
  `clearance_per_step_min_mm` 가 32 행 → **8 행**이 된다. `sphere_*` 키의 길이는 기본 설정에서 그대로다
  (`--sphere-*` 를 주면 늘어난다). 옛 기록을 읽는 코드는 행 수를 박아 두지 않았는지 확인이 필요하다.
* **latch 가 다르게 움직인다.** gripper 열 수정으로 `attach`/`detach` 시점이 옛 실행과 달라질 수
  있다 — 기록의 `latch` 상태·`held_points`·`carved` 를 옛 run 과 그대로 비교하면 안 된다.
* **`.planned` 를 읽는 실험 스크립트 둘이 같이 따라간다**:
  `benchmark/ag3s/experiments/studies/a1_horizon_visibility.py:119` 와
  `benchmark/trajopt/experiments/esdf_rollout.py:219` — 둘 다 이제 8 스텝을 본다. 그 study 를
  32 스텝으로 재려면 `plan_horizon` 을 명시로 줘야 한다 (지금 flag 는 없다).
* 재생산이 필요한 npz: **없음** (TSDF/ESDF 자산은 안 건드렸다).
* 나는 rollout 을 돌리지 않았고 서버(PID 717165)도 건드리지 않았다.

## 내가 기대하는 결과

> **verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.**

* 실행 창만 다듬으므로 refined chunk 의 앞 8 스텝이 정책 chunk 에 훨씬 가까워지고, **사과가
  움직이기 시작할 것**으로 본다. 다만 그러면 참값 clearance 가 음수인 프레임이 늘어난다 —
  `violated` 가 늘고 로컬이 HOLD 를 더 자주 부를 수 있다. 그 둘 중 무엇이 나오는지가 이번 측정의
  실제 질문이다.
* 계산량: 결정 변수 32×14 → 8×14, 충돌 행 최대 768 → 192. 서버 `trajopt` 시간이 내려갈 것으로
  **예상**한다 (얼마나는 측정이다). AG3S 지각은 그대로이므로 total p50 3445.6 ms 의 대부분은
  안 움직일 것으로 본다.
* 굵기 flag 는 **아무것도 안 주면 T6d 와 같은 모델**이어야 한다. 구 수·반지름이 다르게 찍히면
  그것은 내가 뭔가 놓친 것이다.
