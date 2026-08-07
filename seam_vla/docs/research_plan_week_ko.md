# 이번 주 연구 계획: 문제 정의 / RQ / 프레임워크 구조

> 대상: `benchmark/seam_vla`
> 목적: "VLA의 Safety를 개선한다"는 막연한 목표를, 실제로 관측/재현 가능한 문제(Jerk, Collision)로
> 재정의하고, 이를 검증할 RQ와 프레임워크 구조를 설계한다.
> 범위: 이 문서는 RQ와 실험/프레임워크 설계 단계다. 안전 제약을 구현하는 구체적 알고리즘(예:
> collision-avoidance)은 아직 결정하지 않았고, 다음 단계 과제로 남긴다. 실제 실험 데이터도
> 별도로 수집·제공될 예정이며, 이 문서는 그 데이터를 어떻게 분석할지의 틀만 정의한다.

---

## 1. 문제 정의

### 1.1 기존 프레이밍의 문제

"VLA의 안전성을 개선한다"는 목표는 무엇이 안전하지 않은지, 어떤 지표로 측정하는지, 무엇을
바꾸면 개선되는지를 특정하지 못한다. 아래 두 가지는 이 저장소에서 **실제로 확인된, 구체적인**
문제다.

### 1.2 문제 1 — Jerk: SEAM으로도 완전히 해결되지 않음

`docs/professor_review_ko.md` §9.2의 RB-Y1 114개 chunk 오프라인 재현 결과:

| 지표 | Baseline | SEAM(decoded) | 변화 |
|---|---:|---:|---:|
| BJ (boundary jerk) | 0.0226 | 0.0221 | −2.4% |
| AVb (boundary jerk 분산) | ~0.0001 | ~0.0001 | −8.8% |
| overlap residual | 0.1275 | 0.1254 | −1.7% |
| **CD (경계 불연속)** | 0.0180 | 0.0181 | **+0.1% (미개선)** |

BJ/AVb는 줄었지만 **CD는 개선되지 않았다**. 즉 SEAM(VLS)은 이전 chunk의 tail을 향해 boundary
근처 action을 nudge하지만, 경계에서의 절대적 불연속(`‖a_t - a_(t-1)‖`) 자체는 줄이지 못하는
사례가 존재한다.

이는 다음 두 가지로 설명 가능하다.
- `docs/reproduction_notes.md`가 문서화한 대로, 배포된 checkpoint는 논문(H=50, K=10, 사용 가능한
  overlap L=40)보다 훨씬 좁은 guided window(H=10, K=5, L=5)에서 동작한다. Overlap 자체가 짧아
  보정이 걸 수 있는 여지가 작다.
- SEAM은 "이전 예측 tail 방향"만 prior로 사용할 뿐, 장면의 실제 기하(장애물, 목표 위치)를
  전혀 참조하지 않는다. Jerk가 유발되는 원인이 순수한 재계획 불연속이 아니라 장면 대응(예:
  갑작스러운 회피/정지)일 경우 SEAM은 이를 반영할 수 없다.

### 1.3 문제 2 — Collision: 파이프라인에 전혀 다뤄지지 않음

`refinement/base.py`의 docstring은 다음과 같이 명시한다.

> "Future implementations may apply trajectory optimization, MPC, joint/velocity/acceleration
> limits, collision or CBF safety filtering, etc. For this work only `IdentityChunkRefiner` is
> provided."

실제로 이 저장소의 두 `DecodedChunkRefiner` 구현(`IdentityChunkRefiner`, `OverlapSteerRefiner`)은
모두 물리 공간 action chunk만 입력받을 뿐, 오브젝트 위치·충돌 상태 등 **공간 정보를 전혀 읽지
않는다.** MuJoCo 자산(`src/rby1_description/models/*/mujoco/assets/*_collision_*.obj`)에 충돌
지오메트리는 존재하지만, 이는 물리 엔진이 접촉을 계산하기 위한 것일 뿐 이를 회피하는 로직은
어디에도 없다. 즉 collision-avoidance는 "구현했지만 부족한" 상태가 아니라 **처음부터 예약만
되어 있고 전무한 상태**다.

### 1.4 가설

Jerk의 CD 미개선과 Collision의 부재는 같은 원인에서 기인한다: **SEAM/refinement 파이프라인이
장면 기하(오브젝트/장애물 위치, 접촉 상태)를 전혀 참조하지 않고, 오직 "이전에 예측한 action"만
prior로 사용한다.** 따라서 smoothness 제약(SEAM)과 별개로 장면 기하를 참조하는 안전 제약을
추가하면, (a) 충돌 위험을 직접 줄이고 (b) 장면 대응이 필요한 구간의 잔여 jerk(CD가 개선되지
않는 사례)도 함께 개선될 가능성이 있다. 이 안전 제약이 구체적으로 어떤 알고리즘(예:
collision-avoidance/CBF, 속도·가속도 제한, 다른 형태의 장면 기하 제약 등)이 될지는 아직
결정하지 않았다 — 이번 단계에서는 "장면 기하를 참조하는 안전 제약"이라는 요구사항만 확정한다.

### 1.5 Novelty

- 기존 SEAM(VLS)은 denoising-space에서 "이전 predicted tail"만을 prior로 쓰는 smoothness 기법이며
  장면 기하를 모른다 — 이는 이미 구현/검증된 선행 연구 기여다.
- 이번 연구의 신규성은, `refinement/base.py`가 예약해 둔 physical-space 훅에 **장면 기하 기반
  안전 제약**(구체적 알고리즘 미정 — collision-avoidance는 후보 중 하나)을 추가하고, 이를 SEAM의
  smoothness 제약과 **함께** 적용했을 때 task semantics(성공률)를 유지하면서 안전성(jerk CD +
  충돌 위험)을 개선할 수 있는지를 실증적으로 검증하는 것이다. 어떤 안전 제약을 택할지는 RQ-a의
  진단 결과(§2)를 보고 결정한다.

---

## 2. Research Question

### RQ (본 연구)

> 장면 기하 기반 안전 제약(구체적 알고리즘 미정)과 smoothness 제약(SEAM/VLS)을 함께 적용했을
> 때, baseline 및 SEAM-only 대비 VLA의 task 성공률(semantics)을 유지하면서 안전성
> (chunk-boundary discontinuity, 충돌 위험)을 향상시킬 수 있는가?

### 하위 RQ 분리

- **RQ-a (선행, 진단적 — 안전 제약 알고리즘 결정 없이 검증 가능)**: baseline과 SEAM은
  공간적으로 *어디서* 실패하며, 그 실패는 (i) 모델이 만든 trajectory 자체의 실패인가, (ii)
  컨트롤러/시뮬레이션 등 외부 요인에 의한 실패인가? 이 진단 결과가 §1.4 가설의 근거가 되고,
  어떤 형태의 안전 제약(RQ-b)이 적합한지를 결정하는 데 쓰인다.
- **RQ-b (다음 단계, 알고리즘 선정 및 구현 필요)**: RQ-a의 진단 결과를 바탕으로 선택한 안전
  제약(예: collision-avoidance는 후보 중 하나이며, `refinement/base.py`가 예약해 둔
  physical-space 훅에 구현될 예정)을 SEAM과 결합했을 때, RQ 본문의 성공률 유지 + 안전성 개선이
  실제로 일어나는가?

### 검증에 필요한 실험/프레임워크

| RQ | 필요 실험 | 데이터 |
|---|---|---|
| RQ-a | LIBERO/RBY1 baseline·SEAM rollout에서 위치별 성공/실패, 충돌 근접 여부, action 시계열을 확보해 실패 궤적을 진단 | 별도로 수집·제공 예정 (§3.3) |
| RQ-b | 선정된 안전 제약 알고리즘 구현 + baseline/SEAM/SEAM+안전제약 비교 | RQ-a 결과 확정 후 설계 |

---

## 3. 프레임워크 구조 설계

### 3.1 시스템 구조

```
observation
   │
   ▼
DenoisingGuidance (model space, denoising ODE 내부)      ← 기존 SEAM(VLS) 기여
   │  IdentityGuidance | VLSGuidance
   ▼
Unnormalize / output transform (physical space 변환)
   │
   ▼
DecodedChunkRefiner (physical space, 실행 직전)
   │  IdentityChunkRefiner | OverlapSteerRefiner   ← 기존 기여
   │  Safety Refiner Hook (알고리즘 미정)           ← RQ-b, 다음 단계
   ▼
ChunkExecutor (K-of-H 실행 cadence, baseline/SEAM 동일)
   │
   ▼
env (LIBERO/robosuite/MuJoCo | RBY1 native MuJoCo)
```

`DenoisingGuidance`(model space)와 `DecodedChunkRefiner`(physical space)는 이미 저장소에
분리되어 있는 두 개의 개입 지점이다. VLS는 전자를 사용하며, RQ-b의 안전 제약은 (구체적
알고리즘과 무관하게) 후자의 physical-space 훅을 사용할 것으로 설계한다 — 서로 다른 좌표계에서
동작하므로 원칙적으로 독립적으로 켜고 끌 수 있다. `refinement/collision_avoidance.py`는 이
훅에 맞춘 참고용 인터페이스 프로토타입(스켈레톤, `refine()`은 미구현)으로 이미 작성해 두었으나,
RQ-a 진단 결과 전까지는 채택 여부가 확정된 구현이 아니다.

### 3.2 연구 기여 vs 인용/재사용 모듈

| 모듈 | 상태 | 귀속 |
|---|---|---|
| `guidance/vls.py` (VLS) | 구현·오프라인 검증됨 | 기존 연구 기여 (SEAM) |
| `refinement/overlap_steer.py` | 구현·오프라인 검증됨 | 기존 연구 기여 |
| `metrics/motion.py` (BJ/IJ/CD/AVb, Eq. 9–13) | 구현됨 | SEAM 논문 수식 인용, 코드는 자체 구현 |
| RQ / 문제 정의 / 프레임워크 설계 (본 문서) | 작성됨 | 이번 단계 기여 |
| `refinement/collision_avoidance.py` | 참고용 인터페이스 프로토타입 (미채택, `refine()` 미구현) | 참고 자료 — RQ-b 알고리즘 후보 중 하나 |
| 안전 제약 알고리즘 선정 + 구현 | **미정** | RQ-a 진단 결과 이후 결정 |
| 실패 궤적 진단 / 위치별 성공률 분석 등 실험 파이프라인 | 설계만, 데이터 확보 후 구현 | 데이터 제공 이후 진행 |
| openpi π0.5 모델, LoRA 학습 파이프라인 | — | Physical Intelligence 인용 |
| LIBERO / robosuite / MuJoCo | — | 인용 (벤치마크/시뮬레이터) |
| RBY1 MJCF 모델 자산 | — | 재사용 (하드웨어 벤더 제공) |

**범위 제한 명시**: 이번 단계의 산출물은 문제 정의, RQ, 프레임워크 구조(본 문서)까지다.
`refinement/collision_avoidance.py`는 physical-space 안전 훅의 인터페이스가 어떤 모습일 수
있는지 보여주는 참고 프로토타입일 뿐, 채택이 확정된 구현이 아니다(`refine()`은
`NotImplementedError`). 실제 안전 제약 알고리즘 선정·구현과 실험 데이터 수집·분석은 모두
다음 단계(RQ-a 데이터 확보 이후)로 남긴다 — 보고서에 "충돌 회피를 구현했다"는 과장된 서술이
들어가지 않도록 범위를 명시적으로 남긴다.

### 3.3 실험 시나리오 (개요 — 세부 파이프라인은 데이터 확보 후 설계)

- **LIBERO**: task 별 오브젝트/타겟 위치가 다양하므로 위치별 성공률·실패 궤적 진단(RQ-a)에
  적합할 것으로 예상.
- **RBY1**: 이미 확보된 controller/qpos 로깅 인프라(`pi05_infer.py`,
  `plot_rby1_jerk_comparison.py`)를 활용해 컨트롤러 root-cause 분석과 jerk-peak 구간 분석에
  적합할 것으로 예상.
- 두 시나리오 모두 실제 실험 실행과 로깅 확장은 사용자가 별도로 데이터를 수집해 제공한 뒤
  진행한다.
