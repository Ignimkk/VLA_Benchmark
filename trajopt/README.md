# trajopt — AG3S 제약 + SEAM 참조 궤적 → 안전한 action chunk

> **Attention은 target을 정하고, 기하는 무엇이 충돌할 수 있는지를 정하며, phase와 주입된 접촉 컨텍스트는
> 어떤 접촉이 허용되는지를 정한다. 그리고 TO는 그 제약을 만족하며 로봇이 어떻게 움직일지를 정한다.**

```
π0.5 → chunk[50,14] → SEAM → 참조 궤적 ─┐
                                        ├→ [trajopt] → 안전한 chunk[50,14] → 실행
AG3S(3-카메라 융합) → CollisionConstraintSet ─┘
```

`benchmark/seam_vla/refinement/base.py`가 예약해 둔 `DecodedChunkRefiner` 자리를 채웁니다.
**SEAM 코드는 수정하지 않습니다** — import해서 상속할 뿐입니다.

## 실행 환경 (선택이 아니라 배포의 성질)

| 항목 | 값 | 출처 |
|---|---|---|
| 제어 주기 | CTRL_HZ = 15 → **66.7 ms** | [pi05_infer.py:103](../../src/rby1_bringup/pi05_infer.py#L103) |
| 청크 / 실행 길이 | H = 50, K = 8 → 청크당 533 ms | [seam_rby1.yaml](../seam_vla/configs/seam_rby1.yaml) |
| Action | `14 = [L 6 abs joint, L grip, R 6 abs joint, R grip]` | [pi05_infer.py:77](../../src/rby1_bringup/pi05_infer.py#L77) |
| 최적화 자유도 | **12** (양팔 arm_0..5). `arm_6`·토르소는 action 채널이 없어 파라미터 | — |

TO는 **청크당 1회, 블로킹**으로 돕니다. 한 제어 주기(66.7 ms)를 넘기면 루프가 멈춥니다.

## 방법 — SQP + 희소 QP, Real-Time Iteration

### IPOPT를 쓰지 않는 이유

1. **내점법은 warm-start가 구조적으로 어렵다.** IPM은 barrier 파라미터로 매개된 central path를
   따라간다. 마진을 스치는 좋은 충돌회피 궤적은 제약 *경계 근처*에 있고, 그것은 어떤 적당한 μ에서도
   central path에서 먼 나쁜 시작점이라 솔버가 μ를 다시 annealing한다. Active-set·ADMM 계열에는
   그 병리가 없다.
2. **반복 횟수에 상한이 없다.** IPOPT는 허용오차까지 반복하고 restoration phase에 들어갈 수 있다.
   실시간 제어는 **주기당 고정 예산**이 필요하고, 그것이 RTI가 주는 것이다.

### 제약을 얻는 방법 — 측정으로 결정

RB-Y1 (61구 × 32슬롯 × H=50 = 103,700행):

| 방법 | 빌드(1회) | h + 기울기 |
|---|---|---|
| AG3S의 단일 CasADi 그래프 (`to_casadi`) | ~19 s | ~19 ms |
| 스텝별 그래프 + `map(50)` | 0.31 s | 12.5 ms |
| **sphere FK + 위치 Jacobian** | **0.066 s** | **1.07 ms** |

세 번째는 근사가 아닙니다. `h = ‖p−c‖ − R`, `∇h = (p−c)ᵀ/‖p−c‖ · ∂p/∂q`가 닫힌 형태이고
`∂p/∂q`는 **32개 슬롯 전체가 공유**합니다 — 단일 그래프는 그 중복을 32번 지불합니다.

**그래서 trajopt는 `to_casadi()`가 아니라 AG3S가 이미 채운 파라미터 벡터를 layout으로 읽습니다.**
clearance 정책·overflow 집계·attached 블록이 전부 그 안에 있으므로 정책 중복이 없고, 솔브 중
AG3S 호출도 없습니다. `to_casadi()`는 일반 NLP 솔버용으로 그대로 유효합니다.

### QP 부분솔버 — 실측 (1,884 변수 · 4,116행 · nnz 16,980)

| backend | 첫 solve | 재-solve | 판정 |
|---|---|---|---|
| **osqp** | 97.0 ms | **2.6 ms** | **기본값** |
| highs | 335.0 ms | 332.3 ms | 정확하나 120× 느리고 warm-start 이득 없음 |
| proxqp | 5,508 ms | 5,630 ms | 정확하나 여기서는 사용 불가 |
| qrqp | > 200 s | — | 제외 |
| **ipqp** | — | — | **인터프리터를 abort시킴** (heap corruption) |
| qpOASES | — | — | 구조적 제외 (밀집 active-set, 제약 수에 초선형) |

`ipqp`가 `SOLVERS`에 없는 이유가 마지막 줄입니다 — 제어 루프를 함께 죽일 수 있는 backend는
타이밍과 무관하게 후보가 아닙니다.

> **재-solve 열은 낙관적 수치입니다.** 같은 문제를 다시 준 경우로, warm start가 이미 답 위에 있습니다.
> SQP 루프 안에서는 매 반복이 다른 곳에서 선형화하므로, 정직한 수치는 아래의 청크당 총합입니다.

`src/openpi/.venv/bin/python -m benchmark.trajopt.qp --benchmark`로 대상 장비에서 재현합니다.

## 문제 정식화

```
min  w_track ‖Q − Q_seam‖²  +  w_smooth ‖Δ²Q‖²  +  w_cont ‖Q − Q_prev‖²  +  w_slack ‖s‖₁

s.t. h_AG3S(Q_k) + ∇h·(Q − Q_k) + s ≥ backoff,   s ≥ 0
     q_min ≤ Q ≤ q_max                      URDF
     |ΔQ| ≤ v_max·dt,  |Δ²Q| ≤ a_max·dt²    URDF (softened)
     |Q[:,0] − q_now| ≤ v_max·dt            anchor, not a pin
     ‖Q − Q_k‖_∞ ≤ trust_radius
```

**Slack은 편의가 아니라 필수입니다.** 이미 접촉 중이거나 지각이 한 프레임 튀면 QP가 실행 불가능해지고,
그러면 컨트롤러가 아무것도 내놓지 못합니다. KNOWS가 같은 결론에 도달했습니다
([filter.py](../knows_vla/cbf/filter.py) D3). **운동학 제약도 마찬가지입니다** — 정책이 로봇의
가속 한계를 넘는 청크를 제안하면 QP가 `primal infeasible`이 되므로, 속도·가속 행도 slack으로
부드럽게 하고 위반량을 보고합니다.

**Back-off는 선택이 아닙니다.** QP가 `h + ∇h·Δ ≥ 0`을 만족시켜도 `h(Q+Δ) ≥ 0`은 아닙니다 —
관절 공간의 직선을 따라 거리는 오목하므로 참값이 접선 아래에 놓입니다. 실측: back-off 없이 수렴한
SQP에 **0.327 mm의 실제 관통**이 남았습니다. 5 mm를 더 요구하면 닫힙니다.

## 실측 성능

RB-Y1, 53 mm 근접 시나리오, 12 자유도. **모든 조합이 위반 0.00 mm에 도달**:

| plan_horizon | SQP 2회 | SQP 3회 |
|---|---|---|
| 50 | — | 92.8 ms |
| **32** | **22.4 ms** | 29.3 ms |
| 24 | 20.0 ms | 27.4 ms |
| 16 | 15.0 ms | 21.4 ms |

기본값 `plan_horizon=32`에서 **정상 상태 28.4 ms** (제어 주기 66.7 ms). 단계별:

| 단계 | ms |
|---|---|
| linearize (선택 + 기울기) | 2.5 |
| assemble (QP 조립) | 5.6 |
| **qp** | **32.3** |
| check (전체 행 재검사) | 7.1 |

성능을 만든 네 가지 수정 — 전부 프로파일이 지목한 것:

| 원인 | 전 | 후 |
|---|---|---|
| `sp.bmat`으로 차분 연산자 조립 | 8 ms | 0.24 ms |
| CasADi DM→numpy 조밀 변환 | 5.7 ms | 2.2 ms |
| OSQP `eps` 1e-5 | 70.7 ms | 4.6 ms (1e-3) |
| `w_slack` 1e4 (조건수 파괴) | 175 반복 | 25 반복 (1e3) |

`eps=1e-3`이 안전한 이유는 속도가 아닙니다. SQP 부분문제는 비볼록 문제의 *선형화*이므로 모델 자체의
충실도를 넘어 정밀하게 풀 이유가 없고(inexact SQP), **해에서 전체 행을 원해상도로 다시 검사**하므로
느슨하게 푼 부분문제는 조용한 오류가 아니라 보고되는 위반으로 드러납니다.

## 행을 줄이는 두 가지

H=50 · 46구 · 32슬롯이면 약 78,000행 대 600 변수 — 행이 병목입니다.

1. **활성 밴드**: clearance가 밴드를 넘는 행은 한 신뢰 영역 스텝 안에 위반이 될 수 없으므로 QP에
   영향이 없습니다. 밴드는 `trust_radius`를 초과해야 하며 config가 강제합니다.
2. **스텝당 예산**: 행 `i`는 항상 스텝 `i // rows_per_step`에 속합니다 → **QP의 희소성이 고정**되어
   솔버가 인수분해를 재사용합니다. 패턴이 매 반복 움직였을 때 solve당 193 ms였습니다.
   예산이 걸리면 보고합니다.

**둘 다 안전한 이유는 하나입니다** — `full_violation`이 해에서 **모든** 행을 원해상도로 재평가합니다.
축소는 수렴을 잃을 수 있어도 충돌을 숨길 수는 없습니다.

## 사용법

```python
from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS, load_rby1
from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.linearize import SceneSnapshot
from benchmark.trajopt.refiner import TrajOptChunkRefiner
from benchmark.trajopt.types import ChunkLayout

robot = load_rby1()          # AG3S가 제약을 만든 것과 **같은** 모델이어야 합니다
layout = ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)

def scene_fn(context):
    constraint_set = ag3s.process_multi(observations, phase="approach",
                                        active_manipulators=[Manipulator.RIGHT])
    scene = SceneSnapshot.from_spec(constraint_set.constraints, robot.radii)
    return scene, q_now, constraint_set.geometry_certified

refiner = TrajOptChunkRefiner(
    robot, layout, scene_fn,
    TrajOptConfig.from_yaml("benchmark/trajopt/configs/rby1.yaml"),
    on_result=lambda r: None if r.safe else my_stop_policy(r),   # 정지 여부는 호출자가 결정
)
safe_chunk = refiner.refine(physical_chunk, context={"previous_physical_chunk": previous})
```

SEAM 정책에 꽂을 때는 `SeamPolicy(..., refiner=refiner)`.

## 검증

```bash
src/openpi/.venv/bin/python -m pytest tests/trajopt -q          # 79 passed
src/openpi/.venv/bin/python -m pytest tests/ag3s -q             # 422 passed (회귀)
src/openpi/.venv/bin/python -m benchmark.trajopt.qp --benchmark
```

## 발견한 것 — SEAM은 TO의 수정을 되먹이지 않는다

[seam_policy.py:194](../seam_vla/policy/seam_policy.py#L194):

```python
new_state = seam_state.with_chunk(model_chunk_np[0], physical_chunk, ...)
#                                 ^^^^^^^^^^^^^^ 정제 **전** 모델 공간 청크가 다음 prior가 된다
```

정제된 physical chunk는 로깅용으로만 저장됩니다. 방치하면 SEAM이 매 청크 같은 충돌 궤적을 다시
제안하고 TO가 매번 밀어내면서, SEAM이 없애려던 경계 jerk가 TO 계층에서 되살아납니다.

완화책은 `context["previous_physical_chunk"]`를 쓰는 **연속성 항**입니다(`w_continuity`).
SEAM 수정 없이 됩니다. **완화이지 해결이 아닙니다** — 근본 해결은 physical→model 역변환과
SEAM 수정을 수반하며 이번 범위 밖입니다.

## 알려진 한계

- **정확도·성능 수치는 전부 합성 시나리오 기준**입니다. 실제 π0.5 청크로는 미검증입니다.
- **`plan_horizon=32`는 청크의 뒤 18스텝을 손대지 않고 통과시킵니다.** 실행되지 않으므로 안전에는
  영향이 없지만, 이음매에 불연속이 생기고 그것이 다음 청크의 연속성 항에 들어갑니다.
  실측 0.02 rad이며 `metrics["plan_join_discontinuity_rad"]`로 보고됩니다.
- **SQP는 지역해만 줍니다.** 충돌 회피는 비볼록이라 초기값에 따라 다른 지역해에 갑니다.
  SEAM 청크를 초기값으로 쓰는 것이 합리적 선택이지만 전역 최적을 주장하지 않습니다.
- **운동학 한계가 부드럽습니다.** 참조가 로봇의 가속 한계를 넘으면 해에도 잔여 위반이 남을 수 있고,
  `limit_report`가 그 값을 보고합니다. 하드 제약이 아닙니다.
- **MuJoCo 재생 실험과 `pi05_infer.py` 배선은 아직입니다.**
