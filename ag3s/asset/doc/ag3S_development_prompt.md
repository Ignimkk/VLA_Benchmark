# AG3S 개선 구현 요청

## 0. 작업 목적

`/home/mk/dev_ws/vla/pi0_TO_ws`의 기존 AG3S 구현을 기반으로, **VLA attention 기반 target grounding과 geometry-complete collision constraint generation을 분리한 안전한 3D collision scene pipeline**으로 개선하라.

작업 대상은 다음과 같다.

- 구현: `benchmark/ag3s/`
- 테스트: `tests/ag3s/`
- 참고 설계 문서: `/home/mk/dev_ws/vla/pi0_TO_ws/benchmark/ag3s/asset/doc/AG3S_development_guide.md`
- 연동 코드:
  - `src/openpi/`
  - `src/rby1_manipulation/`
  - `src/rby1_bringup/`

참고 문서는 **설계 자료**이며 실행 명세가 아니다.

반드시 실제 코드와 테스트를 먼저 조사하고, 현재 구현과 문서가 충돌하면 **검증된 현재 코드와 안전 invariant를 우선 확인한 뒤 필요한 부분만 수정**하라.

이미 구현된 기능을 중복 구현하거나 전체 구조를 불필요하게 재작성하지 말라.

---

# 1. 현재 테스트 기준선

현재 AG3S 테스트 기준선은 다음과 같다.

```text
src/openpi/.venv/bin/python -m pytest tests/ag3s -q
287 passed
```

작업 시작 시 반드시 이 baseline을 실제로 재현한다.

작업 트리에 사용자 변경 사항이 존재하므로:

- 관련 없는 파일 수정 금지
- 관련 없는 파일 삭제 금지
- 사용자 변경 사항 restore/reset 금지
- repository 전체 formatter 적용 금지
- AG3S와 직접 관련된 코드, 테스트, config, 문서만 수정

---

# 2. 구현 전 조사

구현 전에 현재 코드와 테스트를 조사하고 아래만 간단히 보고한다.

1. 관련 기존 클래스 / 함수 / 파일
2. 이미 구현되어 재사용할 기능
3. 현재 safety invariant와 충돌하는 구현
4. 변경 예정 파일
5. 단계별 구현 순서

조사 후 사용자 확인을 기다리지 말고 바로 구현을 계속한다.

---

# 3. 절대 유지해야 할 핵심 설계 원칙

다음은 AG3S의 invariant이다.

변경하거나 약화하지 말라.

## 3.1 Semantics와 geometry의 분리

### 무엇이 target인가

VLA attention은 **target grounding에만 사용한다.**

고-attention point에서 seed를 추출하고 해당 seed가 가리키는 3D target을 찾는 데까지만 사용한다.

- seed 없음
- grounding 실패
- score threshold 미달

이면:

```text
target=None
```

과 명시적인 failure reason/status를 반환한다.

target을 추측하거나 만들어내지 않는다.

---

### 무엇이 충돌할 수 있는가

재구성된 모든 valid physical geometry는 collision candidate가 될 수 있다.

따라서:

- low-attention geometry
- zero-attention geometry
- semantic classification 실패 geometry
- unknown geometry

도 collision scene에서 제거하지 않는다.

`UNKNOWN_GEOMETRY`로 남겨야 한다.

`generate_candidates()`는 attention을 입력으로 받지 않는다.

이 API invariant를 유지하고 테스트한다.

---

## 3.2 Target은 collision geometry에서 삭제하지 않는다

Target 역시 물리적인 물체이므로:

```text
TargetGeometry
    ↓
CollisionCandidate(source_type=TARGET)
```

으로 collision scene에 남긴다.

GRASP phase라고 해서 target candidate 전체를 제거하거나 비활성화하지 않는다.

접촉 허용 여부는:

```text
Phase
×
Active Manipulator
×
Candidate Source
×
Robot Link
```

에 따른 clearance policy로만 결정한다.

---

## 3.3 Attention은 collision candidate 생성에 영향을 주지 않는다

다음 구조를 유지한다.

```text
Attention
   ↓
Target Grounding
```

하지만:

```text
Geometry
   ↓
Collision Candidate Generation
```

에는 attention이 영향을 주지 않는다.

`CollisionCandidate`에 attention field를 추가하지 않는다.

---

## 3.4 과소근사를 허용하지 않는다

모든 geometry approximation은 보수적으로 수행한다.

목표는 tight approximation이 아니라:

> 원본 geometry를 완전히 포함하는 conservative approximation

이다.

과대근사는 우회 경로를 만들 수 있지만,
과소근사는 실제 충돌을 만들 수 있다.

따라서 primitive fitting, sphere chain, overflow aggregation 모두 containment를 우선한다.

---

## 3.5 Phase는 외부 입력이다

AG3S 내부에 새로운 phase FSM 또는 task-state inference를 만들지 않는다.

`Phase`는 외부에서 명시적으로 주입된다.

---

## 3.6 Active manipulator도 외부 입력이다

`GRASP` phase 자체만으로 어느 arm/gripper가 target과 접촉 가능한지 추론하지 않는다.

접촉 허용 주체는 phase와 별도의 외부 context로 전달한다.

예:

```python
ContactPolicyContext(
    phase=Phase.GRASP,
    active_manipulators={Manipulator.RIGHT},
)
```

또는 기존 코드 구조와 더 잘 맞는 동등한 typed contract를 설계한다.

양손 grasp도 향후 다음처럼 표현 가능해야 한다.

```python
active_manipulators={
    Manipulator.LEFT,
    Manipulator.RIGHT,
}
```

active manipulator가 없거나 잘못된 경우 자동으로 접촉 relaxation을 하지 않는다.

fail-closed로 full margin을 적용한다.

---

# 4. 구현 단계

구현 순서는 반드시 다음을 따른다.

```text
Phase A
현재 코드 조사 + baseline 재현

Phase B
Link-wise / manipulator-aware clearance policy

Phase C
Geometry-complete overflow / fail-closed contract

Phase D
3-camera reconstruction / fusion

Phase E
Multi-view attention lifting

Phase F
POST_GRASP attached collision geometry

Phase G
문서/config 정리 + 전체 regression
```

각 단계에서는 관련 unit test를 먼저 실행한다.

해당 단계 완료 후:

```bash
src/openpi/.venv/bin/python -m pytest tests/ag3s -q
```

전체 regression을 수행한다.

이전 단계가 깨진 상태로 다음 feature를 덧붙이지 않는다.

---

# 5. Phase × Active Manipulator × Source × Link Clearance

현재 GRASP에서 target candidate에:

```python
collision_enabled=False
```

가 적용되어 target과 모든 robot collision sphere 간 제약이 제거되는 문제를 가장 먼저 수정한다.

---

## 5.1 Robot sphere link metadata

`RobotCollisionModel`이 각 collision sphere에 대해 다음 중 하나를 안정적으로 제공하도록 확장한다.

- link name
- stable link ID

`UrdfSphereChain`의 다음 세 요소는 항상 동일한 순서를 가져야 한다.

```text
numeric sphere FK
symbolic sphere FK
sphere link metadata
```

index mismatch를 허용하지 않는다.

---

## 5.2 Clearance policy

기본 형태:

```python
margin = clearance_policy.lookup(
    phase=phase,
    active_manipulators=active_manipulators,
    candidate_source=candidate.source_type,
    robot_link=link_name,
)
```

구현 구조상 더 적절한 typed context가 있다면 사용해도 된다.

---

## 5.3 Contact margin의 의미

다음을 명확하게 구분한다.

### 일반 안전 거리

```text
surface_distance >= d_safe
```

### 허용된 contact

```text
surface_distance >= d_contact
```

기본 `d_contact = 0`.

중요:

> margin이 0이라는 것은 constraint를 disable한다는 의미가 아니다.

접촉은 허용하더라도 **penetration은 금지**한다.

즉 target constraint row는 CasADi graph에서 계속 존재해야 한다.

---

## 5.4 기본 정책

예를 들어 RIGHT GRASP일 때:

```text
RIGHT fingertip / target
    → contact margin

RIGHT gripper allowed contact links / target
    → contact margin

RIGHT forearm / target
    → full margin

LEFT arm / target
    → full margin

Torso / target
    → full margin
```

`OBJECT`와 `UNKNOWN_GEOMETRY`는 모든 phase에서 full margin을 유지한다.

명시적인 successful target grounding이 있는 `TARGET`만 contact relaxation 후보가 될 수 있다.

다음 source들은 자동으로 relaxation하지 않는다.

```text
OBJECT
UNKNOWN_GEOMETRY
OVERFLOW
UNCLASSIFIED
기타 unknown source
```

알 수 없는:

```text
phase
source
link
manipulator
```

조합은 모두 fail-closed full margin이다.

---

## 5.5 Grounding failure

다음 경우:

```text
target=None
```

이면 어떠한 robot link도 target/contact relaxation을 받아서는 안 된다.

---

# 6. Fixed CasADi Constraint Graph 유지

CasADi graph는 frame, phase, target 여부, grasp 여부에 따라 다시 생성하지 않는다.

구조는 고정한다.

기존 candidate 단위 `d_safe` parameter가 있다면 이를:

```text
robot sphere/link
×
candidate slot
```

pair별 parameter로 확장한다.

예:

```text
margin[sphere_idx, candidate_slot]
```

CasADi:

- constraint row 수 고정
- sparsity 고정
- slot 수 고정
- solver structure 고정

이어야 한다.

frame별로 변경되는 것은 parameter vector뿐이어야 한다.

---

# 7. 실제 RB-Y1 3-Camera Reconstruction / Fusion

현재:

```text
HEAD
LEFT_WRIST
RIGHT_WRIST
```

세 카메라를 독립 pipeline처럼 사용하고 있다면 하나의 fused physical scene으로 확장한다.

---

## 7.1 Camera contract

최소 다음 타입을 추가하거나 기존 타입을 확장한다.

```python
CameraID.HEAD
CameraID.LEFT_WRIST
CameraID.RIGHT_WRIST
```

`CameraObservation`은 최소 다음 정보를 표현해야 한다.

```text
camera_id
depth 또는 point cloud
intrinsics
T_base_cam 또는 이를 계산할 정보
capture timestamp
attention(optional)
```

기존 single-camera `AG3S.process()` 호출은 compatibility wrapper로 유지한다.

---

# 8. Capture-Time Transform 사용

각 camera observation은 자신의 capture timestamp를 기준으로 처리한다.

세 카메라에 현재 시각의 하나의 `q_now`를 재사용하지 않는다.

각 카메라에 대해:

```text
observation timestamp
        ↓
q(t)
        ↓
T_base_cam(t)
```

을 사용한다.

특히 wrist camera는 robot motion에 따라 extrinsic이 변하므로 반드시 capture-time FK를 사용한다.

---

## 8.1 권장 처리 순서

각 카메라에 대해:

```text
Depth / Point Cloud
        ↓
Reconstruction
        ↓
T_base_cam(t)
        ↓
Base-frame Point Cloud
        ↓
Robot Collision Model @ q(t)
        ↓
Self Filtering
```

을 독립적으로 수행한다.

이후 self-filter된 cloud들을:

```text
base frame
```

에서 fusion한다.

---

## 8.2 Timestamp validity

다음 값을 config로 둔다.

예:

```text
max_state_age_sec
max_transform_age_sec
max_camera_skew_sec
```

실제 이름은 기존 config style에 맞춘다.

코드에 임의의 magic number를 넣지 않는다.

stale state 또는 transform이 발생하면:

- silent fallback 금지
- status
- notes
- metrics

에 기록한다.

---

# 9. Multi-View Geometry Fusion

Geometry와 image observation provenance를 분리한다.

최종 fused point는 단순히:

```text
xyz
```

만 가지면 안 된다.

최소한 다음 정보를 추적 가능해야 한다.

```text
xyz_base
contributing camera_id
(u,v)
timestamp
```

Python object-per-point가 지나치게 비싸다면:

```text
point table
+
observation table
+
index/offset arrays
```

같은 CSR-like representation을 허용한다.

---

## 9.1 동일 geometry 판단

기본적으로 base-frame voxel key를 geometry association 단위로 사용한다.

동일 voxel에 여러 camera observation이 들어오면:

```text
하나의 fused geometry
+
여러 observation provenance
```

로 표현한다.

---

## 9.2 Representative point

voxel 대표값을 synthetic centroid로 만들 경우 image `(u,v)` provenance가 사라질 수 있다.

따라서 최종 representative는 가능하면:

> 실제 관측점 중 하나

를 사용한다.

geometry representative와 observation provenance는 별개로 관리한다.

---

# 10. Camera 역할

기본 역할은 다음과 같이 가정한다.

```text
HEAD
→ global workspace geometry
→ table / shelf / large obstacle
→ global target observation

LEFT_WRIST
→ left manipulation local geometry
→ head-camera occlusion 보완

RIGHT_WRIST
→ right manipulation local geometry
→ head-camera occlusion 보완
```

하지만 collision geometry 생성 시 세 camera를 의미적으로 차등 제거하지 않는다.

관측된 valid physical geometry는 모두 collision candidate pipeline에 남긴다.

---

# 11. Camera Dropout / Incomplete Coverage

다음 상황을 명시적으로 처리한다.

```text
camera dropout
missing depth
missing attention
stale transform
stale robot state
registration failure
```

최소 한 카메라가 정상이라면 geometry pipeline 자체는 가능한 범위에서 동작할 수 있다.

하지만:

> 일부 camera가 빠진 scene을 정상적인 fully-observed clear scene으로 보고해서는 안 된다.

coverage degradation을 status/metrics로 전달한다.

AG3S 내부에 emergency stop FSM을 만들지 않는다.

대신 호출자가 안전 정책을 결정할 수 있도록 상태를 명시적으로 전달한다.

---

# 12. Multi-View Attention Lifting

Attention은 target grounding에만 사용한다.

각 camera의 attention은 동일 normalization mode/config를 사용한다.

카메라마다 서로 다른 normalization 방식이 암묵적으로 적용되지 않게 한다.

---

## 12.1 기본 aggregation

동일 fused geometry에 여러 camera observation이 있으면 기본:

```python
attention_fused = max(
    normalized_attention_from_each_valid_observation
)
```

을 사용한다.

이번 작업에서는 `max` aggregation만 필수 구현한다.

불필요하게 mean / weighted / learned fusion까지 구현 범위를 확장하지 않는다.

필요하면 config enum으로 향후 확장 가능하게만 설계한다.

---

## 12.2 Missing attention

attention이 없는 camera observation은 target scoring에:

```text
0
```

을 제공할 수 있다.

하지만 geometry 자체를 제거하면 안 된다.

---

## 12.3 Invariant

Attention lifting 전후:

```text
physical geometry point coverage
```

가 감소하면 안 된다.

특히 low-attention point를 삭제하지 않는다.

`CollisionCandidate`에 attention field를 추가하지 않는다.

---

# 13. Target Geometry Provenance

`TargetGeometry`에는 target 판단을 추적할 수 있도록 필요한 최소 provenance를 추가한다.

예:

```text
seed camera
supporting cameras
observation indices
seed pixels
```

단, API와 memory cost를 불필요하게 키우지 않는다.

target debugging/validation에 필요한 범위로 제한한다.

---

# 14. Geometry-Complete Fail-Closed Contract

현재 다음 경로에서 geometry가 조용히 사라질 가능성을 조사한다.

```text
unknown_min_points 미만 residual
max_clusters 초과
max_candidates 초과
constraint sphere overflow
max_points cap
voxel downsampling
camera dropout
stale state
```

모든 valid non-support geometry는 최종적으로 다음 세 상태 중 하나여야 한다.

```text
1. 정상 CollisionCandidate

2. Conservative Overflow Aggregate

3. 명시적 GEOMETRY_INCOMPLETE / DEGRADED 상태
```

조용히 삭제하지 않는다.

---

# 15. Fail-Closed 상태의 의미

`GEOMETRY_INCOMPLETE`, `DEGRADED`, `UNSAFE` 등의 상태는 단순 diagnostic 문자열이어서는 안 된다.

`to_adapter` 또는 최종 `CollisionConstraintSet`까지 전달되어야 한다.

예:

```python
CollisionConstraintSet(
    ...,
    validity=ConstraintValidity.VALID,
)
```

또는 현재 architecture에 더 적절한 동등한 typed contract를 사용한다.

중요:

> geometry completeness를 보장할 수 없는 frame을 정상적인 empty / clear constraint set으로 변환하지 않는다.

다만 AG3S 내부에 stop/hold/retry FSM을 새로 만들지 않는다.

AG3S의 책임은:

```text
"I cannot certify this scene as geometry-complete."
```

를 호출자에게 전달하는 것까지다.

---

# 16. Constraint Capacity Overflow

현재 constraint slot overflow 시 작은 sphere부터 삭제하는 방식이 있다면 제거한다.

고정 CasADi graph는 유지한다.

overflow geometry는 가능한 경우 conservative aggregate로 병합한다.

---

## 16.1 Overflow strategy

하나의 거대한 sphere가 과도한 free-space loss를 만들 수 있으므로 가능하면:

```text
bounded number of reserved overflow slots
```

을 사용한다.

예:

```text
normal candidate slots
+
reserved overflow aggregate slots
```

형태를 검토한다.

overflow groups는 가능하면 spatial grouping 후 각각 conservative bounding primitive로 만든다.

구현이 과도하게 복잡해지면 단순한 conservative aggregate를 사용할 수 있지만:

- containment 우선
- excessive conservatism metrics 기록

을 유지한다.

---

## 16.2 Overflow invariant

모든 overflow aggregate는 자신이 대표하는 모든 original point를 포함해야 한다.

---

# 17. max_points / Downsampling 처리

`max_points` 문제는 candidate overflow와 다르다.

candidate overflow는 원본 geometry가 아직 존재하지만,
초기 point truncation은 geometry 자체를 잃어버린다.

따라서:

> `max_points` 때문에 valid points를 arbitrary tail truncation하지 않는다.

가능하면:

- deterministic
- spatial coverage preserving

selection을 사용한다.

coverage를 보존할 수 없으면:

```text
GEOMETRY_INCOMPLETE
```

상태로 처리한다.

downsampling도 collision coverage를 잃는 방향으로 작동해서는 안 된다.

---

# 18. Geometry Uncertainty Budget

다음 오차를 safety geometry에 반영한다.

```text
depth noise
voxel/downsampling error
camera registration error
hand-eye calibration error
```

예:

```text
r_conservative =
    r_fit
    + uncertainty_margin
```

또는:

```text
effective_clearance =
    nominal_clearance
    + perception_uncertainty
```

중 하나의 일관된 방식으로 처리한다.

중요:

> uncertainty를 primitive inflation과 d_safe 양쪽에 중복 적용하지 않는다.

double counting을 피한다.

각 uncertainty component는 config와 diagnostics에서 추적 가능하게 한다.

---

# 19. Support Surface

기존 설계를 유지한다.

Support plane은 primitive cloud로 변환하지 않는다.

analytic half-space:

```text
nᵀ p >= d + margin
```

형태를 유지한다.

구조화된 planar geometry는 analytic constraint,
일반 unstructured geometry는 conservative primitive로 처리한다.

---

# 20. Primitive Fitting

기존:

```text
sphere
capsule
box
ellipsoid
```

구조를 보존한다.

목표는 fitting error 최소화보다 containment 보장이다.

최소 diagnostics:

```text
containment_rate
excess_volume 또는 동등한 conservatism metric
```

을 제공한다.

capsule → sphere chain 변환 역시 원본 capsule volume을 보수적으로 덮어야 한다.

---

# 21. POST_GRASP Attached Collision Geometry

이 기능은 마지막 단계에서 구현한다.

AG3S 내부에서 grasp 성공을 추론하지 않는다.

외부 grasp confirmation을 입력받는다.

```text
Target / fitted target primitives
        ↓
External grasp confirmation
        ↓
AttachedCollisionGeometry
```

---

## 21.1 Snapshot

grasp confirmation 시점의 target collision geometry / fitted primitives를 snapshot한다.

grasp 후 target이 camera에서 보이지 않거나 grounding에 실패해도 attached object가 임의로 사라져서는 안 된다.

---

## 21.2 Transform convention

attached object transform 방향을 명시적으로 정의한다.

예:

```text
T_parent_object
```

즉 parent link frame에서 object frame으로 가는 고정 transform을 저장한다.

향후:

```text
T_base_object(q)
=
T_base_parent(q)
*
T_parent_object
```

로 symbolic FK와 함께 움직여야 한다.

transform 방향을 unit test로 검증한다.

---

## 21.3 Attached geometry contract

예:

```python
AttachedCollisionGeometry(
    parent_link=...,
    T_parent_object=...,
    primitives=...,
    allowed_contact_links=...,
)
```

정확한 API는 기존 architecture에 맞게 설계한다.

---

# 22. Attached Object Collision

grasp 이후 최소 다음 충돌을 검사한다.

```text
held object ↔ environment
held object ↔ support surface
```

가능하면:

```text
held object ↔ opposite arm
held object ↔ torso
held object ↔ non-contact same-arm links
```

도 검사한다.

---

## 22.1 Contact allowlist

held object는 parent gripper / fingers와 의도적으로 접촉한다.

따라서 attached object와 robot 간 모든 self-collision을 무조건 검사하면 안 된다.

명시적인 allowlist를 사용한다.

예:

```text
right gripper
right fingertip links
```

만 허용.

다음은 계속 collision constraint를 유지한다.

```text
forearm
torso
opposite arm
기타 non-contact links
```

잘못된 link 이름이나 unknown link는 fail-closed 처리한다.

---

# 23. Attach / Detach State

tracking 또는 target grounding 실패 때문에 attached geometry를 자동으로 detach하지 않는다.

detach도 반드시 외부 confirmation을 요구한다.

즉:

```text
attach
→ explicit external event

detach
→ explicit external event
```

이다.

AG3S 내부에 grasp FSM을 만들지 않는다.

---

# 24. Attached Slots도 Fixed Graph에 포함

grasp 전후로 CasADi solver를 재생성하지 않는다.

fixed number의 attached-object primitive slots와 active parameter를 graph에 미리 포함한다.

```text
attached active = 0
```

또는:

```text
attached active = 1
```

parameter로 활성화한다.

graph:

```text
shape
row count
sparsity
```

는 attach/detach 전후 동일해야 한다.

---

# 25. 테스트 요구사항

기존 올바른 테스트는 모두 보존한다.

기존 테스트 중:

```text
GRASP에서 target 전체 비활성화
```

를 올바른 동작이라고 가정하는 테스트는 새 계약에 맞게 수정한다.

GPU나 실물 카메라는 필수 조건으로 만들지 않는다.

합성 fixture 기반 unit/integration test를 우선한다.

---

## 25.1 Existing invariants

반드시 테스트:

- `generate_candidates()`에 attention 인자가 없음
- zero-attention geometry 보존
- unknown geometry 보존
- isolated valid geometry 보존
- target은 collision candidate에 남음
- phase는 AG3S 내부에서 추론되지 않음

---

## 25.2 Clearance tests

추가:

- target grounding 실패 시 clearance relaxation 없음
- target은 모든 phase에서 active candidate slot으로 유지
- GRASP에서도 target constraint row 존재
- `contact margin = 0`에서도 penetration은 금지
- RIGHT active manipulator일 때 RIGHT allowed contact link만 완화
- RIGHT grasp에서 LEFT gripper–target은 full margin 유지
- target–forearm full margin 유지
- target–torso full margin 유지
- target–opposite-arm full margin 유지
- missing/invalid active manipulator → full margin
- unknown source/link/phase → fail-closed full margin

---

## 25.3 Robot model tests

- numeric sphere FK와 link metadata 정렬
- symbolic sphere FK와 link metadata 정렬
- numeric/symbolic sphere order 동일
- self-filter와 TO가 동일 robot collision model 사용

---

## 25.4 Multi-camera tests

- 세 camera base-frame registration
- camera provenance 보존
- `(u,v)` provenance 보존
- timestamp 보존
- wrist camera capture에 capture-time FK 사용
- 서로 다른 camera timestamp에서도 각 transform으로 올바른 base geometry 생성
- 하나의 `q_now`를 세 camera에 재사용하지 않음
- 한 camera dropout 시 pipeline이 명시적으로 degraded 상태를 반환
- stale transform/state status 반영
- 최소 한 camera 정상일 때 보수적인 geometry pipeline 동작

---

## 25.5 Fusion tests

- 동일 voxel을 두 camera가 관측할 때 하나의 geometry representative 생성
- contributing camera provenance 모두 유지
- representative가 synthetic centroid 때문에 provenance를 잃지 않음
- geometry coverage가 fusion 후 감소하지 않음

---

## 25.6 Attention tests

- camera별 동일 normalization config 사용
- camera별 normalization 후 max fusion
- high-attention secondary camera observation이 voxel fusion 때문에 사라지지 않음
- attention 없는 camera geometry 유지
- multi-view lifting 전후 geometry count/coverage invariant
- `CollisionCandidate`에 attention field가 없음
- `TargetGeometry`에서 필요한 provenance 추적 가능

---

## 25.7 Geometry-complete tests

- `unknown_min_points` 이하 residual이 조용히 사라지지 않음
- max_clusters overflow 보존
- candidate slot overflow 보존
- overflow aggregate가 모든 원본 point를 포함
- arbitrary `max_points` tail truncation 없음
- spatial coverage 보존 불가능 시 `GEOMETRY_INCOMPLETE`
- degraded/incomplete 상태가 empty clear constraint set으로 변환되지 않음
- camera dropout/stale state도 geometry validity에 반영
- primitive containment
- capsule sphere-chain containment

---

## 25.8 Fixed graph tests

- phase 변경 전후 graph shape 동일
- active manipulator 변경 전후 graph shape 동일
- camera frame 갱신 전후 graph shape 동일
- candidate slot 내용 변경 시 parameter-only refresh
- 동일 solver object 재사용
- constraint row/sparsity 불변

---

## 25.9 Attached geometry tests

- attach transform 방향 검증
- held object가 parent FK와 함께 이동
- held object–table collision
- held object–shelf collision
- held object–opposite arm collision
- held object–parent forearm collision 유지
- parent gripper/contact links allowlist만 제외
- grounding 실패가 attached object를 삭제하지 않음
- explicit detach 전까지 attached geometry 유지
- attach/detach 전후 graph shape 불변

---

# 26. Diagnostics / Metrics

최소 다음 정보를 가능한 범위에서 기록한다.

```text
input point count per camera
post-filter point count
fused point count
camera dropout
stale camera/state count
support point count
target grounding status
target confidence
candidate count
unknown point count
overflow point count
overflow primitive/sphere count
containment rate
excess volume / conservatism metric
geometry completeness status
constraint slot usage
attached slot usage
```

기존 metrics 구조가 있다면 확장하고 별도 시스템을 중복 구현하지 않는다.

---

# 27. 구현 범위 제한

이번 작업에서 다음을 새로 만들지 않는다.

- 새로운 VLA
- 새로운 learned segmentation model
- 새로운 grasp detector
- 새로운 task planner
- AG3S 내부 phase FSM
- AG3S 내부 emergency-stop FSM
- TSDF/ESDF 전체 시스템
- 새로운 trajectory optimizer
- learned multi-view fusion
- 불필요한 perception framework

현재 AG3S + CasADi architecture 안에서 필요한 최소 변경으로 구현한다.

---

# 28. 문서 업데이트

최종 구현 완료 후 실제 API 기준으로:

```text
benchmark/ag3s/README.md
config example
필요한 design documentation
```

을 갱신한다.

development guide의 문구를 기계적으로 복사하지 않는다.

실제 구현과 테스트된 API를 source of truth로 사용한다.

---

# 29. 완료 조건

다음을 모두 만족해야 완료로 간주한다.

### 1.

구현 전 현재 코드와 테스트를 조사하고 baseline을 확인했다.

### 2.

기존 올바른 safety invariant를 보존했다.

### 3.

link/manipulator-aware target contact clearance를 구현했다.

### 4.

GRASP에서도 target 전체 collision constraint를 삭제하지 않는다.

### 5.

geometry overflow/truncation이 silent drop으로 이어지지 않는다.

### 6.

3-camera가 capture-time pose를 이용해 하나의 base-frame collision scene을 구성한다.

### 7.

multi-view attention은 target grounding에만 사용된다.

### 8.

POST_GRASP object collision을 fixed graph 안에서 처리한다.

### 9.

다음 명령이 통과한다.

```bash
src/openpi/.venv/bin/python -m pytest tests/ag3s -q
```

### 10.

최종 보고에 다음을 명확히 구분한다.

```text
변경 파일
핵심 API/contract 변화
추가 테스트
전체 테스트 결과
성능 영향
남은 limitation
실물 RB-Y1에서 검증해야 할 항목
```

---

# 30. 최우선 안전 규칙

테스트를 통과시키기 위해 다음을 하지 말라.

- safety margin 임의 감소
- target collision candidate 삭제
- unknown geometry 삭제
- overflow geometry silent drop
- stale camera를 정상 데이터로 처리
- geometry-incomplete scene을 clear scene으로 처리
- constraint row 삭제
- CasADi graph 동적 재생성으로 문제 회피
- contact 허용을 constraint disable로 구현
- attached object를 perception failure로 자동 제거
- failing safety test 삭제 또는 의미 약화

테스트가 기존 설계 오류를 정답으로 간주하고 있다면 테스트를 새 contract에 맞게 수정하되, **안전 invariant 자체를 테스트에 맞춰 약화하지 않는다.**

---

# 최종 설계 원칙

구현 전체에서 아래 원칙을 유지한다.

```text
Attention decides:
"What is the target?"

Geometry decides:
"What can physically collide?"

Phase + externally supplied contact context decides:
"Which contact is currently allowed?"

Trajectory Optimization decides:
"How can the robot move while satisfying those constraints?"
```

AG3S는 semantics와 physical safety geometry를 혼합하지 않는다.
