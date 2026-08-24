# AG3S Development Guide

> **목적**  
> 본 문서는 RB-Y1 기반 VLA 안전성 연구에서 사용되는 **AG3S(3D 안전 장면 생성 및 충돌제약 생성 모듈)**의 구현 방향을 정리한 개발 문서다.  
> 현재 구현된 8단계 파이프라인을 유지하면서, 다중 RGB-D 카메라(Head / Left Wrist / Right Wrist), VLA attention 기반 target grounding, geometry-complete collision candidate generation, conservative primitive fitting, phase-conditioned collision constraint를 Trajectory Optimization(TO)에 안정적으로 연결하는 것을 목표로 한다.

---

## 1. 핵심 설계 철학

AG3S의 가장 중요한 원칙은 **semantic reasoning과 physical collision reasoning을 분리**하는 것이다.

### 1.1 Attention은 target만 결정한다

Attention의 역할은 다음 한 가지다.

> **“현재 VLA가 어느 물체를 target으로 보고 있는가?”**

고-attention 3D point에서 seed를 뽑아 target candidate를 찾는다. 신뢰할 수 있는 seed가 없거나 grounding score가 임계값보다 낮으면 `target=None`과 명확한 `GroundingStatus`를 반환한다.

Attention은 다음 용도로 사용하지 않는다.

- collision candidate 삭제
- obstacle 여부 판정
- unknown geometry 제거
- collision constraint 활성/비활성 결정

이를 타입과 API 수준에서 강제한다.

```python
TargetGeometry
    - points
    - attention_score
    - grounding_score
    - source_camera / observation info

CollisionCandidate
    - geometry
    - source_type
    - cluster_id
    - confidence / provenance
    # attention 필드 없음
```

특히 다음 원칙을 유지한다.

```python
generate_candidates(...)
```

는 **attention 인자를 절대 받지 않는다.**

---

### 1.2 Geometry가 충돌 가능성을 결정한다

재구성된 물리적 geometry는 semantic label의 존재 여부와 무관하게 collision candidate가 될 수 있다.

즉:

```text
관측된 geometry
    ↓
TARGET / OBJECT / UNKNOWN_GEOMETRY / 기타
    ↓
CollisionCandidate
```

`UNKNOWN_GEOMETRY`도 삭제하지 않는다.

핵심 원칙:

> **Attention decides the target, not physical existence.**

> **Geometry decides what can collide.**

---

### 1.3 Target도 collision candidate에 남긴다

Target은 물리적으로 존재하는 물체이므로 collision scene에서 삭제하지 않는다.

```text
TargetGeometry
    ↓
CollisionCandidate(source_type=TARGET)
```

Target을 scene에서 삭제하면 다음 문제가 생긴다.

- grasp 직전 가장 중요한 geometry가 collision model에서 사라짐
- target 주변의 최소거리 정보 손실
- grasp phase 진입 시 constraint를 다시 삽입하면 barrier가 불연속적으로 변함
- target과 허용되지 않은 robot link 간 충돌을 검사할 수 없음

따라서 **target 여부는 collision candidate 생성 여부가 아니라 required clearance를 결정하는 입력**이어야 한다.

---

### 1.4 Phase는 외부에서 주입된다

AG3S는 phase를 추론하지 않는다.

```python
Phase = APPROACH | PREGRASP | GRASP | POSTGRASP | PLACE | ...
```

AG3S 내부에는 phase transition state machine을 만들지 않는다.

이유:

- task planner와 safety module에 서로 다른 상태 기계가 존재하는 문제 방지
- phase 판단 오류가 collision constraint를 비활성화하는 문제 방지
- unit test 가능한 deterministic rule table 유지

핵심 원칙:

> **Phase decides which contact is admissible now.**

---

### 1.5 Geometry approximation은 보수적으로 수행한다

모든 근사는 **collision false negative를 만들지 않는 방향**으로 틀려야 한다.

허용:

- 실제 물체보다 큰 sphere
- 실제 물체보다 큰 capsule
- 약간 팽창된 box
- self-filter 영역을 실제 robot보다 약간 크게 설정

금지:

- cluster 일부를 primitive가 포함하지 못함
- sphere chain 사이에 uncovered gap 존재
- robot surface가 self-filter 영역 밖으로 노출됨

즉 최적화 기준은 tightness보다 먼저 **containment**다.

---

## 2. 최종 시스템 아키텍처

```text
          HEAD RGB-D          LEFT WRIST RGB-D       RIGHT WRIST RGB-D
              │                      │                       │
              └──────────────┬───────┴──────────────┬────────┘
                             ▼
             [01] Per-camera Scene Reconstruction
                  - depth validation
                  - back-projection
                  - T_base_cam(t)
                  - camera_id / (u,v) / timestamp 보존
                             │
                             ▼
                  Multi-view Base-frame Cloud
                             │
                             ▼
                [02] Robot Self Filtering
                  - 동일 RobotCollisionModel
                  - 동일 FK
                  - conservative sphere removal
                             │
                  ┌──────────┴───────────┐
                  ▼                      ▼
          [03] Support Surface      Remaining Geometry
                  │                      │
                  │                      ▼
                  │             [04] Attention Lifting
                  │                      │
                  │                      ▼
                  │             [05] Target Grounding
                  │                      │
                  │                      ▼
                  │                TargetGeometry
                  │                      │
                  └────────────┬─────────┘
                               ▼
                    [06] Collision Candidates
                 TARGET / OBJECT / UNKNOWN_GEOMETRY
                               │
                               ▼
                    [07] Primitive Fitting
                sphere / capsule / box / ellipsoid
                               │
                               ▼
                  Phase × Link Clearance Rules
                               │
                               ▼
                   [08] Constraint Generation
                      CasADi symbolic constraints
                               │
                    VLA Action / Goal Prior
                               │
                               ▼
                    Trajectory Optimization
                               │
                               ▼
                         Safe Trajectory
```

---

# 3. Stage 01 — Scene Reconstruction

**파일:** `reconstruction.py`

## 3.1 현재 역할

입력:

```text
depth 또는 pointcloud + camera calibration + T_base_cam
```

출력:

```text
PointCloud(base frame)
```

Depth camera의 pinhole back-projection:

\[
X = \frac{(u-c_x)D}{f_x}, \qquad
Y = \frac{(v-c_y)D}{f_y}, \qquad
Z = D
\]

이후:

\[
p_B = {}^BT_C p_C
\]

으로 base frame에 정렬한다.

---

## 3.2 유지해야 할 현재 설계

- `0`, `NaN`, `Inf` depth 제거
- depth valid range 검사
- voxel downsampling
- `max_points` cap
- 대표 voxel centroid가 아니라 **실제 observation point를 representative로 유지**
- `(u,v)` 유지

특히 `(u,v)`는 이후 VLA attention lifting에 반드시 필요하다.

---

## 3.3 3-camera 지원을 위해 추가할 데이터

현재 point 표현에 다음 항목을 추가하는 것을 권장한다.

```python
PointObservation:
    xyz_base: float[3]
    camera_id: CameraID
    u: int
    v: int
    timestamp: float
```

더 좋은 구조는 하나의 fused 3D point가 여러 카메라 observation을 보존하는 것이다.

```python
FusedPoint:
    xyz_base: float[3]
    observations: list[Observation]

Observation:
    camera_id
    u
    v
    timestamp
```

이 구조를 사용하면 **geometry fusion과 attention fusion을 분리**할 수 있다.

---

## 3.4 카메라 역할 분담

### Head Camera

주 역할:

- global workspace reconstruction
- table / shelf / large obstacle
- 장거리 target observation
- support surface detection

### Left Wrist Camera

주 역할:

- left gripper 주변 local geometry
- head camera occlusion 보완
- left-arm manipulation detail

### Right Wrist Camera

주 역할:

- right gripper 주변 local geometry
- head camera occlusion 보완
- right-arm manipulation detail

권장 설계:

> **Head = global backbone, wrists = local refinement / occlusion recovery**

---

## 3.5 Wrist camera transform

Head camera가 고정되어 있다면:

\[
{}^BT_{C_H} \approx \text{constant}
\]

Wrist camera는 매 timestep마다 변한다.

\[
{}^BT_{C_L}(q)
=
{}^BT_{EE_L}(q){}^{EE_L}T_{C_L}
\]

\[
{}^BT_{C_R}(q)
=
{}^BT_{EE_R}(q){}^{EE_R}T_{C_R}
\]

따라서 다음이 필요하다.

- URDF/FK
- hand-eye calibration
- camera timestamp에 대응하는 joint state

가능하면 각 frame에 대해

```text
q(t_head)
q(t_left)
q(t_right)
```

를 사용한다.

### 이유

빠른 wrist motion에서 모든 frame을 단일 `q_now`로 transform하면 ghost geometry와 self-filter mismatch가 발생할 수 있다.

---

## 3.6 참고 논문 활용

### Grasping Trajectory Optimization with Point Clouds

활용:

- depth → point cloud → robot frame reconstruction의 선행근거
- perception 결과를 trajectory optimization collision formulation으로 연결하는 선행근거

### MobileH2R

활용:

- global/front perception과 wrist local perception의 역할 분리
- wrist camera가 arm occlusion을 보완하는 근거

### AhaRobot / Mobile ALOHA

활용:

- head/top + left wrist + right wrist의 3-camera 구성이 실제 bimanual manipulation에서 사용된다는 근거

---

# 4. Stage 02 — Robot Self-Filter

**파일:** `robot_filter.py`

## 4.1 현재 역할

입력:

```text
PointCloud + RobotCollisionModel + q
```

출력:

```text
robot geometry가 제거된 PointCloud
```

RobotCollisionModel의 sphere chain에 대해 ball query를 수행한다.

---

## 4.2 반드시 유지할 원칙

> **Self-filter와 TO collision constraint는 동일한 FK와 동일한 RobotCollisionModel을 사용한다.**

다른 robot geometry를 사용하면 perception과 optimizer 사이의 geometric consistency가 깨진다.

예:

```text
self-filter: robot이라고 판단하여 제거
TO model: 해당 위치를 robot surface로 간주하지 않음
```

또는 반대 상황이 발생할 수 있다.

---

## 4.3 Conservative self-filter

self-filter sphere는 실제 collision sphere보다 약간 팽창시키는 방향을 권장한다.

```text
r_filter = r_robot + m_filter
```

단, 지나치게 크게 지우면 robot 주변의 실제 obstacle geometry까지 제거될 수 있으므로 별도 validation이 필요하다.

---

## 4.4 3-camera 처리 방식

권장 순서:

```text
Head cloud  ── self-filter(q(t_head)) ──┐
Left cloud  ── self-filter(q(t_left)) ──┼─→ fusion
Right cloud ── self-filter(q(t_right)) ─┘
```

또는 base-frame fusion 후 각 point의 timestamp/provenance를 이용하여 self-filter할 수 있다.

초기 구현에서는 **camera별 self-filter 후 fusion**이 단순하고 디버깅이 쉽다.

---

## 4.5 참고 논문 활용 — cuRoboV2

가져올 내용:

- current robot state의 FK를 사용한 robot collision geometry 배치
- depth / point cloud에서 robot body 제거
- self-filter 이후 environment representation 구축
- high-DoF robot에서도 일관된 robot collision representation 사용

가져오지 않을 내용:

- TSDF / ESDF 자체
- GPU distance field pipeline 전체

현재 AG3S에서는 **self-filter 철학과 geometric consistency만 가져온다.**

---

# 5. Stage 03 — Support Surface

**파일:** `support_surface.py`

## 5.1 현재 설계 유지

입력:

```text
filtered cloud
```

출력:

```text
SupportSurface[] + inlier mask
```

테이블과 같은 넓은 support surface는 point cluster / primitive가 아니라 plane constraint로 처리한다.

\[
n^Tp \ge d + m
\]

---

## 5.2 이 방식을 유지해야 하는 이유

테이블을 object primitive로 근사하면:

- 매우 큰 box 발생
- 지나치게 많은 spheres 필요
- optimizer constraint 수 증가
- target cluster connectivity가 tabletop 전체로 범람할 수 있음

반면 plane은 한 개의 analytic constraint로 표현할 수 있다.

권장 연구 표현:

> **Structured geometry → analytic constraint**  
> **Unstructured geometry → conservative primitive approximation**

---

## 5.3 순서 유지

실제 구현 순서:

```text
Scene reconstruction
    ↓
Self-filter
    ↓
Support plane extraction
    ↓
Attention lifting / target grounding
```

Target grounding 전에 plane을 제거하거나 mask로 전달하는 현재 결정은 유지한다.

---

# 6. Stage 04 — Attention Lifting

**파일:** `attention_lifting.py`

## 6.1 역할

입력:

```text
A(u,v) + PointCloud
```

출력:

```text
AttentionPointCloud
```

VLA attention은 `AttentionAdapter` 뒤에 격리한다.

---

## 6.2 핵심 invariant

```python
len(lift(cloud, attention)) == len(cloud)
```

저-attention geometry도 절대 삭제하지 않는다.

이것이 다음 원칙의 구현이다.

> **Low attention does not mean non-collidable.**

---

## 6.3 Multi-camera attention fusion

같은 3D geometry가 여러 camera에 보이면, 각 camera의 attention을 geometry와 별도로 결합해야 한다.

초기 구현 추천:

\[
a(p)=\max_{c\in V(p)} \tilde A_c(u_c,v_c)
\]

여기서:

- \(V(p)\): point \(p\)를 관측한 camera 집합
- \(\tilde A_c\): camera별 rank-preserving normalized attention

### `max`를 추천하는 이유

한 camera에서 occlusion 또는 view ambiguity로 attention이 낮더라도, 다른 camera가 target을 명확하게 본다면 seed를 살릴 수 있다.

---

## 6.4 주의사항

multi-view voxel fusion이 attention보다 먼저 수행되면 원래 high-attention observation이 삭제될 수 있다.

따라서 다음 중 하나를 사용한다.

### 방법 A — observation provenance 보존

```text
fused geometry point
  ├─ head observation
  ├─ left observation
  └─ right observation
```

### 방법 B — camera별 lift 후 geometry fusion

초기 구현에서는 A가 장기적으로 더 적합하다.

---

# 7. Stage 05 — Target Grounding

**파일:** `target_grounding.py`

## 7.1 역할

```text
AttentionPointCloud
    ↓
seed extraction
    ↓
3D connectivity growth
    ↓
DBSCAN
    ↓
cluster metrics
    ↓
TargetGeometry or failure
```

---

## 7.2 현재 핵심 결정 유지

### 전체 cloud로 connectivity growth

seed 내부에서만 cluster를 만드는 것이 아니라 **전체 geometry cloud로 성장**한다.

이유:

- target backside는 attention이 약할 수 있음
- visible front surface만 사용하면 centroid가 camera 방향으로 치우침
- collision / grasp geometry가 target 전체를 대표하지 못함

---

## 7.3 실패는 정상적인 출력이다

다음 상황에서 target을 지어내지 않는다.

- seed 없음
- valid cluster 없음
- grounding score 미달

출력:

```python
TargetResult(
    target=None,
    status=GroundingStatus.NO_SEED | NO_CLUSTER | LOW_SCORE,
)
```

이 원칙은 safety-critical하다.

잘못된 target을 만들면 phase rule이 해당 물체의 margin을 줄이면서 **collision constraint를 잘못 완화**할 수 있다.

---

## 7.4 Lan-o3dp에서 가져올 내용

Lan-o3dp의 중요한 아이디어:

> task semantics를 이용하여 target object와 collision-relevant object를 구분한다.

AG3S에서는 이 중 **target grounding 철학만** 가져온다.

차이:

```text
Lan-o3dp
Language → LLM/open-vocabulary detector → Target

AG3S
VLA internal attention → 3D seed → TargetGeometry
```

AG3S의 장점:

- 별도 object detector가 없어도 VLA 내부 task relevance를 활용 가능
- target을 못 찾으면 명시적으로 failure 반환 가능

---

# 8. Stage 06 — Collision Candidates

**파일:** `collision_candidates.py`

## 8.1 현재 구조를 유지해야 한다

입력:

```text
residual cloud + target + support planes
```

출력:

```text
CollisionCandidate[]
```

핵심 질문은 다음이다.

> **“이 geometry가 obstacle인가?”**가 아니라  
> **“이 공간에 물리적으로 충돌 가능한 geometry가 존재하는가?”**

---

## 8.2 Semantic label은 삭제 조건이 아니다

예:

```text
TARGET
OBJECT
UNKNOWN_GEOMETRY
```

모두 candidate가 될 수 있다.

추천 타입:

```python
CollisionCandidate:
    geometry_points
    source_type: TARGET | OBJECT | UNKNOWN_GEOMETRY | ...
    cluster_id
    provenance
    confidence
```

Attention field는 포함하지 않는다.

---

## 8.3 Lan-o3dp의 obstacle-only 접근은 따르지 않는다

Semantic detector가 obstacle을 놓치면 geometry가 collision model에서 사라질 수 있다.

AG3S는 반대로:

```text
physical observation
    ↓
CollisionCandidate
```

를 기본으로 한다.

Semantic reasoning은 **candidate existence**가 아니라 이후 **clearance policy**에만 관여한다.

---

# 9. Stage 07 — Primitive Fitting

**파일:** `geometry.py`

## 9.1 현재 primitive 기반 접근 유지

지원 primitive:

```text
sphere → capsule → box → ellipsoid
```

ESDF로 변경하지 않는 것을 권장한다.

---

## 9.2 Primitive를 유지하는 이유

CasADi 기반 NLP에서는 closed-form geometry가 유리하다.

예를 들어 robot sphere \(i\)와 obstacle sphere \(j\) 사이 constraint:

\[
g_{ij}(q)
=
\|c_i(q)-o_j\|_2
-(r_i+r_j+m)
\ge 0
\]

여기서 \(c_i(q)\)는 FK의 symbolic expression으로 표현할 수 있다.

CasADi가 이를 직접 자동미분할 수 있다.

---

## 9.3 ESDF를 사용하지 않는 이유

ESDF 방식:

```text
q
 ↓
FK
 ↓
xyz
 ↓
ESDF grid lookup
 ↓
interpolation
 ↓
distance
```

장점:

- dense environment representation
- arbitrary geometry 표현력 높음

단점:

- 매 frame map reconstruction / update 필요
- grid lookup / interpolation 필요
- CasADi symbolic formulation과 직접적인 궁합이 낮음

따라서 AG3S에서는:

> **ESDF가 해결하는 “scene geometry → differentiable collision measure” 문제를 analytic conservative primitives로 해결한다.**

---

## 9.4 Conservative fitting metric

primitive fitting 평가는 IoU나 Chamfer distance만으로 수행하지 않는다.

### 1순위 — Containment Rate

\[
R_{contain}
=
\frac{
|\{p\in C\mid p\in\mathcal P\}|
}{|C|}
\]

목표:

```text
R_contain = 1.0 또는 설정한 safety tolerance 만족
```

### 2순위 — Excess Volume

\[
V_{excess}=V(\mathcal P)-V(C)
\]

순서:

```text
1. containment 확보
2. excess volume 최소화
```

---

## 9.5 Capsule sphere-chain containment

capsule을 discrete sphere chain으로 변환할 때 adjacent sphere 사이의 gap을 방지한다.

샘플 간격이 \(s\), 원 capsule radius가 \(r\)이면 각 sphere radius를 다음처럼 팽창시킬 수 있다.

\[
r_{sphere}=\sqrt{r^2+(s/2)^2}
\]

목표는 sphere chain union이 capsule을 포함하도록 만드는 것이다.

---

# 10. Stage 08 — Constraint Generation

**파일:**

- `constraint_builder.py`
- `to_adapter.py`

## 10.1 현재 고정 슬롯 설계 유지

constraint graph는 한 번 구축하고 frame마다 numerical parameter만 갱신한다.

```text
build once
    ↓
fixed CasADi graph
    ↓
per-frame parameter update
```

현재 측정 예:

```text
structure build : 약 6.1 ms (1회)
per-frame generation : 약 0.08 ms
```

실시간성을 위해 이 구조를 유지한다.

---

## 10.2 GraspTrajOpt에서 가져올 내용

활용:

- point-cloud-derived environment geometry를 nonlinear trajectory optimization에 연결하는 연구적 근거
- CasADi + IPOPT 기반 trajectory optimization 사례

차이:

GraspTrajOpt 계열:

```text
Point Cloud → SDF → collision cost → NLP
```

AG3S:

```text
Multi-view Point Cloud
    ↓
Collision Candidates
    ↓
Conservative Primitives
    ↓
Analytic Distance
    ↓
Hard Collision Constraint
    ↓
NLP
```

---

# 11. Phase-Conditioned Collision Clearance

현재 AG3S에서 가장 중요한 추가 개발 항목이다.

Collision candidate의 existence를 phase로 바꾸지 않는다.

phase가 결정하는 것은 다음이다.

> **candidate–robot-link pair의 required clearance**

---

## 11.1 권장 rule table

```text
Phase × Candidate Source × Robot Link → Required Clearance
```

예: right hand로 target cup을 grasp하는 상황

| Phase | Candidate | Robot part | Required clearance |
|---|---|---|---:|
| APPROACH | TARGET | all links | `d_safe` |
| PREGRASP | TARGET | right gripper | reduced margin |
| PREGRASP | TARGET | right forearm | `d_safe` |
| PREGRASP | TARGET | left arm | `d_safe` |
| GRASP | TARGET | right fingertips | `0` 또는 contact margin |
| GRASP | TARGET | right forearm | `d_safe` |
| GRASP | TARGET | left arm | `d_safe` |
| ANY | UNKNOWN_GEOMETRY | all links | `d_safe` |
| ANY | OBJECT | all non-contact links | `d_safe` |

---

## 11.2 API 제안

```python
margin = clearance_policy.lookup(
    phase=phase,
    candidate_source=candidate.source_type,
    robot_link=link_name,
)
```

Constraint:

\[
d(q; L_i,C_j) \ge m_{phase,source,link}
\]

---

## 11.3 왜 target 삭제보다 좋은가

Target을 삭제하는 방식:

```text
PREGRASP → target constraint 존재
GRASP    → target constraint 삭제
```

문제:

- barrier discontinuity
- link별 허용 contact 표현 불가

Clearance 방식:

```text
candidate는 계속 존재
margin만 변화
```

장점:

- constraint topology 유지
- optimization 구조 유지
- link-specific contact permission 가능
- phase transition에서 변화가 연속적이고 제어 가능

---

# 12. POST-GRASP — Attached Collision Geometry

향후 반드시 추가할 것을 권장한다.

## 12.1 문제

물체를 잡은 이후에는 target이 environment에 고정된 obstacle이 아니다.

```text
Target object
    ↓ grasp
robot과 함께 움직이는 collision body
```

따라서 단순 robot ↔ environment collision만 검사하면 부족하다.

검사해야 할 것:

```text
RB-Y1 links ↔ environment
held object ↔ environment
```

---

## 12.2 권장 설계

```text
TargetGeometry
    ↓ grasp confirmed
AttachedCollisionGeometry
    ↓
RobotCollisionModel extension
```

POST_GRASP 시:

```text
Robot collision body = RB-Y1 + grasped object
```

---

## 12.3 Attached geometry transform

grasp 시점의 object-to-EE transform을 고정한다.

\[
{}^{EE}T_O
=
({}^{B}T_{EE})^{-1}{}^{B}T_O
\]

이후 trajectory optimization 동안:

\[
{}^{B}T_O(q)
=
{}^{B}T_{EE}(q){}^{EE}T_O
\]

으로 attached object primitive를 이동시킨다.

---

# 13. 논문별 실제 활용 전략

## 13.1 Grasping Trajectory Optimization with Point Clouds

### 사용 목적

**Point cloud perception을 TO collision formulation으로 연결하는 가장 직접적인 선행연구**

### AG3S에서 가져올 부분

- depth / point cloud 기반 scene representation
- robot geometry와 scene geometry 사이 distance 기반 collision handling
- nonlinear constrained trajectory optimization
- CasADi / IPOPT 사용 근거

### AG3S와의 차별점

```text
GraspTrajOpt:
Point Cloud → SDF → collision cost

AG3S:
Multi-view point cloud
→ task-aware target grounding
→ geometry-complete collision candidates
→ conservative analytic primitives
→ phase/link-conditioned hard collision constraints
```

---

## 13.2 cuRoboV2

### 사용 목적

**실물 RGB-D perception에서 robot self geometry를 제거하고 high-DoF robot collision model과 일관되게 연결하는 근거**

### 가져올 부분

- FK 기반 robot collision geometry placement
- robot self-filter
- high-DoF manipulation safety
- current robot state와 depth scene의 일관성

### 가져오지 않을 부분

- TSDF / ESDF map representation
- CUDA 기반 distance-field optimizer 전체

---

## 13.3 Lan-o3dp

### 사용 목적

**task semantics를 이용해 manipulation target을 구분하는 연구적 근거**

### 가져올 부분

- task-relevant object identification 필요성
- target geometry와 environment collision geometry의 semantic role 분리

### AG3S에서 변경한 부분

Lan-o3dp:

```text
Language → semantic detector → target / obstacle
```

AG3S:

```text
VLA attention → target grounding
physical geometry → collision candidate
```

즉 collision candidate generation은 semantic detector failure에 의존하지 않는다.

---

## 13.4 MobileH2R

### 사용 목적

**global camera와 wrist camera의 역할 분담 근거**

AG3S 적용:

```text
Head camera = global scene coverage
Wrist cameras = local geometry / occlusion recovery
```

---

## 13.5 AhaRobot / Mobile ALOHA

### 사용 목적

**Head/Top + Left Wrist + Right Wrist의 3-camera bimanual configuration 근거**

AG3S에서는 이 카메라 구성을 단순 policy observation이 아니라 **safety scene reconstruction**에도 사용한다.

---

## 13.6 Multi-View Fusion for Multi-Level Robotic Scene Understanding

### 사용 목적

**multiple camera observation을 unified 3D scene으로 fusion하는 근거**

AG3S 적용:

- base-frame registration
- view provenance 보존
- occlusion 보완
- fused geometry를 collision scene으로 사용

---

# 14. 현재 AG3S의 연구적 포지셔닝

단순히 다음처럼 정의하면 novelty가 약하다.

> Point-cloud-based collision avoidance for VLA

대신 다음 방향을 사용한다.

> **Task-aware but geometry-complete collision constraint generation for VLA-guided manipulation**

또는:

> **Attention-guided target grounding with geometry-complete, phase-conditioned collision constraints for safe VLA manipulation**

핵심 구조:

```text
Semantics
    ↓
What is the TARGET?

Geometry
    ↓
What can physically COLLIDE?

Phase
    ↓
Which CONTACT is admissible now?

Trajectory Optimization
    ↓
How should the robot move safely?
```

세 가지를 의도적으로 분리한다.

---

# 15. 핵심 Contribution 후보

## C1. Decoupled Semantic Target Grounding and Collision Geometry

VLA attention은 target grounding에만 사용하고, collision candidate generation은 attention과 독립적인 full observed geometry에서 수행한다.

효과:

- low-attention obstacle 누락 방지
- unclassified geometry 안전성 유지
- VLA attention failure가 collision scene deletion으로 이어지는 문제 방지

---

## C2. Conservative Multi-View 3D Collision Scene for VLA Manipulation

Head + bilateral wrist RGB-D를 robot base frame으로 fuse하여 global + local collision geometry를 구성한다.

특징:

- head global coverage
- wrist local occlusion recovery
- camera provenance 유지
- same-FK robot self-filter

---

## C3. Phase- and Link-Conditioned Collision Constraints

Target을 collision scene에서 삭제하지 않고, phase와 robot link에 따라 required clearance를 변경한다.

효과:

- grasp contact 허용
- target–forearm collision은 계속 방지
- target–non-grasp arm collision 방지
- barrier topology 유지

---

## C4. Conservative Analytic Primitive Constraints for Real-Time TO

Dense ESDF 대신 conservative primitives를 사용하여 CasADi symbolic hard constraints를 생성한다.

효과:

- closed-form gradient
- fixed constraint graph
- fast per-frame parameter update
- containment-oriented safety

---

# 16. 개발 우선순위

## Priority 1 — 3-Camera Reconstruction

구현:

- `CameraID`
- camera별 intrinsics/extrinsics
- wrist timestamped FK
- base-frame fusion
- `(camera_id, u, v, timestamp)` 보존

테스트:

- 동일 static object를 세 camera로 관측했을 때 base frame registration error
- wrist motion 중 ghost geometry 유무
- camera dropout 대응

---

## Priority 2 — Multi-View Attention Lifting

구현:

- camera별 attention normalization
- observation provenance 유지
- same geometry의 multi-view attention aggregation
- 초기 aggregation = `max`

테스트:

- head occlusion + wrist visibility
- wrist occlusion + head visibility
- 한 camera attention failure 시 target grounding 유지 여부

---

## Priority 3 — Self-Filter Consistency

구현:

- TO와 동일 `RobotCollisionModel`
- camera timestamp별 FK
- conservative inflation margin

테스트:

- robot residual point ratio
- nearby obstacle accidental deletion rate
- left/right wrist view에서 self geometry removal

---

## Priority 4 — Phase × Link Clearance Policy

구현:

```python
clearance_policy.lookup(phase, source_type, link)
```

테스트:

- APPROACH에서 target collision 금지
- GRASP에서 fingertip contact 허용
- GRASP에서도 forearm/left arm collision 유지
- target `None`일 때 margin이 절대 감소하지 않음

---

## Priority 5 — Attached Collision Geometry

구현:

- grasp confirmation input
- `TargetGeometry → AttachedCollisionGeometry`
- object-to-EE transform 저장
- robot collision model 확장

테스트:

- held object ↔ shelf collision
- held object ↔ table collision
- dual-arm scenario에서 held object ↔ opposite arm collision

---

# 17. 필수 Unit / Integration Tests

## 17.1 Semantic-Geometry Separation

```text
[ ] generate_candidates()가 attention 인자를 받지 않는다.
[ ] attention=0인 geometry도 CollisionCandidate가 된다.
[ ] UNKNOWN_GEOMETRY가 삭제되지 않는다.
[ ] target=None이어도 collision candidates가 정상 생성된다.
```

---

## 17.2 Target Grounding Safety

```text
[ ] seed 없음 → target=None
[ ] low score → target=None
[ ] cluster 없음 → target=None
[ ] target 실패 시 clearance relaxation 없음
[ ] 전체 cloud connectivity growth가 target backside를 포함한다.
```

---

## 17.3 Geometry Containment

```text
[ ] sphere primitive가 cluster를 포함한다.
[ ] capsule primitive가 cluster를 포함한다.
[ ] box primitive가 cluster를 포함한다.
[ ] sphere-chain union이 original capsule을 포함한다.
[ ] primitive fitting 실패 시 under-approximation을 반환하지 않는다.
```

---

## 17.4 Self-Filter

```text
[ ] RobotCollisionModel이 TO와 동일하다.
[ ] FK implementation이 TO와 동일하다.
[ ] wrist camera frame마다 timestamped q를 사용한다.
[ ] robot point residual rate를 측정한다.
[ ] nearby external geometry deletion rate를 측정한다.
```

---

## 17.5 Multi-Camera

```text
[ ] HEAD / LEFT / RIGHT 모두 base frame으로 transform된다.
[ ] camera provenance가 유지된다.
[ ] 한 camera dropout 시 pipeline이 실패하지 않는다.
[ ] duplicate geometry fusion이 안전성을 저해하지 않는다.
[ ] attention high-view observation이 voxel fusion으로 사라지지 않는다.
```

---

## 17.6 Phase Constraint

```text
[ ] phase는 외부 입력이다.
[ ] AG3S 내부에 별도 phase transition FSM이 없다.
[ ] TARGET candidate는 모든 phase에서 candidate set에 남는다.
[ ] phase에 따라 margin만 변경된다.
[ ] contact 허용은 link-specific하게 적용된다.
```

---

## 17.7 Optimizer Integration

```text
[ ] CasADi graph 구조가 frame마다 재생성되지 않는다.
[ ] primitive parameter만 update된다.
[ ] inactive slot이 optimizer에 잘못된 constraint를 만들지 않는다.
[ ] worst-case candidate count에서 solve time 측정
[ ] perception update와 optimizer update rate mismatch 테스트
```

---

# 18. 권장 평가 지표

## 18.1 Perception / Reconstruction

- multi-view registration error
- target grounding success rate
- target centroid error
- robot residual point rate after self-filter
- external geometry accidental deletion rate
- collision geometry coverage / containment rate

## 18.2 Collision Scene

- collision candidate recall
- unknown geometry retention rate
- false-negative collision geometry rate
- primitive excess volume
- primitive count per frame

## 18.3 Trajectory Optimization

- collision rate
- minimum robot–environment distance
- minimum held-object–environment distance
- solve time
- constraint generation time
- trajectory duration
- path length
- jerk
- intervention rate
- infeasible optimization rate

## 18.4 Ablation

필수 비교:

```text
A. Head only
B. Head + Right Wrist
C. Head + Left + Right Wrist
```

```text
D. Attention-filtered collision geometry
E. Geometry-complete collision candidates (AG3S)
```

```text
F. Target removed from collision scene
G. Target retained + phase-conditioned clearance (AG3S)
```

```text
H. Tight primitive fitting
I. Conservative containment fitting (AG3S)
```

---

# 19. 구현 시 권장하지 않는 변경

현재 단계에서는 다음 변경을 권장하지 않는다.

## 19.1 ESDF/TSDF로 전체 전환

이유:

- 현재 CasADi symbolic architecture와 맞지 않음
- 구현 복잡도 증가
- 현재 일정에서 핵심 novelty와 직접 관계가 약함

ESDF는 baseline 또는 후속 연구로 남긴다.

---

## 19.2 Semantic obstacle detector 의존

```text
semantic obstacle detected → collision candidate
```

형태로 바꾸지 않는다.

그렇게 하면 perception false negative가 safety false negative로 직접 연결된다.

---

## 19.3 Target 제거

Target은 candidate set에서 삭제하지 않는다.

Phase/link policy로 contact admissibility만 조정한다.

---

## 19.4 AG3S 내부 Phase 추론

Task planner와 별도 phase state machine을 만들지 않는다.

---

# 20. 구현 로드맵

## Step 1

현재 single-camera pipeline을 그대로 유지한 상태에서 `PointCloud` 타입에:

```text
camera_id
u
v
timestamp
```

추가.

## Step 2

Head / Left Wrist / Right Wrist를 각각 독립적으로 reconstruction.

## Step 3

camera별 timestamped FK 적용.

## Step 4

camera별 self-filter 후 base-frame fusion.

## Step 5

multi-view observation provenance 유지.

## Step 6

attention lifting을 multi-view 방식으로 확장.

초기 rule:

```text
same geometry → max normalized attention
```

## Step 7

기존 target grounding 테스트 재실행.

## Step 8

`CollisionCandidate` 생성이 attention과 완전히 독립적인지 API/test 재확인.

## Step 9

`Phase × SourceType × RobotLink → margin` policy 추가.

## Step 10

CasADi fixed-slot parameter에 link-specific / phase-specific margin 반영.

## Step 11

POST_GRASP `AttachedCollisionGeometry` 구현.

## Step 12

Head-only vs 3-camera ablation과 target-delete vs target-retain ablation 수행.

---

# 21. 참고 논문

1. **Grasping Trajectory Optimization with Point Clouds**  
   IROS 2024.  
   Point cloud 기반 scene representation을 collision-aware nonlinear trajectory optimization에 연결하는 핵심 reference.

2. **RAMP: Hierarchical Reactive Motion Planning for Manipulation Tasks Using Implicit Signed Distance Functions**  
   IROS 2023.  
   Current scene point cloud와 robot configuration을 이용한 collision-aware manipulation planning reference.

3. **cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots**  
   2026.  
   Depth scene, robot self-removal, high-DoF collision geometry, real-time motion generation reference.

4. **Lan-o3dp: Language-Guided Object-Centric Diffusion Policy for Generalizable and Collision-Aware Manipulation**  
   ICRA 2025.  
   Task semantics를 이용한 target / collision-relevant object reasoning reference.

5. **MobileH2R**  
   Head/front camera와 wrist camera를 함께 사용하여 global / local perception을 보완하는 reference.

6. **Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation**  
   Top + Left Wrist + Right Wrist camera 구성의 대표적인 bimanual manipulation reference.

7. **AhaRobot**  
   Head + bilateral wrist camera 구성을 사용하는 humanoid manipulation reference.

8. **Multi-View Fusion for Multi-Level Robotic Scene Understanding**  
   Multiple camera views를 unified 3D scene representation으로 fusion하는 reference.

9. **Coupled Mobile Manipulation via Trajectory Optimization with Free Space Decomposition**  
   ICRA 2021.  
   Point cloud에서 free-space geometry를 추출하여 trajectory optimization constraint로 연결하는 reference.

10. **nvblox: GPU-Accelerated Incremental Signed Distance Field Mapping**  
    ICRA 2024.  
    RGB-D 기반 TSDF/ESDF reconstruction 및 collision planning용 distance-field mapping reference.

---

# 22. 최종 설계 문장

AG3S는 다음 세 문장으로 요약한다.

> **Attention decides the target, not physical existence.**

> **Geometry decides what can collide.**

> **Phase decides which contact is admissible.**

그리고 이 결과를 conservative analytic geometry와 CasADi hard constraints로 변환하여 VLA가 제안한 action / goal을 **물리적으로 안전한 trajectory**로 최적화한다.

---

## Appendix A. 권장 모듈 인터페이스 예시

```python
class AG3S:
    def reconstruct(
        self,
        observations: list[CameraObservation],
        robot_state: RobotState,
    ) -> PointCloud:
        ...

    def self_filter(
        self,
        cloud: PointCloud,
        robot_state: RobotState,
    ) -> PointCloud:
        ...

    def detect_support_surfaces(
        self,
        cloud: PointCloud,
    ) -> tuple[list[SupportSurface], PointMask]:
        ...

    def lift_attention(
        self,
        cloud: PointCloud,
        attention: dict[CameraID, AttentionMap],
    ) -> AttentionPointCloud:
        ...

    def ground_target(
        self,
        cloud: AttentionPointCloud,
    ) -> TargetResult:
        ...

    def generate_candidates(
        self,
        cloud: PointCloud,
        target: TargetGeometry | None,
        surfaces: list[SupportSurface],
    ) -> list[CollisionCandidate]:
        # 중요: attention 인자 없음
        ...

    def fit_primitives(
        self,
        candidates: list[CollisionCandidate],
    ) -> list[Primitive]:
        ...

    def build_constraints(
        self,
        primitives: list[Primitive],
        surfaces: list[SupportSurface],
        phase: Phase,
        robot_model: RobotCollisionModel,
    ) -> CollisionConstraintSet:
        ...
```

---

## Appendix B. 가장 먼저 추가할 테스트

```python
def test_generate_candidates_has_no_attention_argument():
    ...


def test_zero_attention_geometry_is_preserved_as_collision_candidate():
    ...


def test_unknown_geometry_is_not_dropped():
    ...


def test_target_remains_collision_candidate():
    ...


def test_low_confidence_target_does_not_relax_clearance():
    ...


def test_multiview_point_preserves_camera_provenance():
    ...


def test_wrist_point_uses_timestamped_fk():
    ...


def test_primitive_contains_all_cluster_points():
    ...


def test_grasp_phase_relaxes_only_allowed_target_link_pairs():
    ...


def test_postgrasp_object_becomes_attached_collision_geometry():
    ...
```
