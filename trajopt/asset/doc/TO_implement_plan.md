# TO 구현 — AG3S 제약 + SEAM 참조 궤적 → 안전한 action chunk

## Context

AG3S(제약)와 SEAM(참조 궤적)이 완성됐고, 둘을 소비하는 **궤적 최적화기가 비어 있다.**
`benchmark/seam_vla/refinement/collision_avoidance.py`의 `refine()`은 아직
`NotImplementedError`이고, AG3S의 `to_adapter.to_casadi()`는 "미래의 TO가 소비할 계약"으로
설계만 되어 있다. 이번 작업이 그 자리를 채운다.

```
π0.5 → chunk[50,14] → SEAM → 참조 궤적 ─┐
                                        ├→ [TO] → 안전한 chunk[50,14] → 실행
AG3S(3-카메라 융합) → CollisionConstraintSet ─┘
```

### 조사에서 확인된 실행 환경

| 사실 | 값 | 출처 |
|---|---|---|
| 제어 주기 | **CTRL_HZ = 15** (66.7 ms/step) | [pi05_infer.py:103](src/rby1_bringup/pi05_infer.py#L103) |
| 청크 지평 / 실행 길이 | **H = 50, K = 8** (→ 533 ms/청크) | [seam_rby1.yaml](benchmark/seam_vla/configs/seam_rby1.yaml) |
| Action 포맷 | `14 = [L 6 abs joint, L grip, R 6 abs joint, R grip]` — **절대 관절 목표** | [pi05_infer.py:77](src/rby1_bringup/pi05_infer.py#L77) |
| 고정 관절 | `arm_6`(손목 마지막)은 고정, 토르소는 action에 채널 없음 | [pi05_infer.py:320](src/rby1_bringup/pi05_infer.py#L320) |
| TO 호출 지점 | `SeamPolicy` 내부 `self.refiner.refine(physical_chunk, context)` | [seam_policy.py:181](benchmark/seam_vla/policy/seam_policy.py#L181) |
| 관절/속도/가속 한계 | URDF `<limit lower upper velocity acceleration>` 전부 존재 | `model.urdf` |
| 충돌 sphere | 팔만 46개, 팔+토르소 61개 | 실측 |

**따라서 TO는 청크당 1회, 블로킹으로 돈다.** 예산은 π0.5 추론을 뺀 나머지이며,
한 제어 주기(66.7 ms)를 넘기면 루프가 멈춘다. **목표 < 30 ms, 하드 캡으로 강제.**

### 조사에서 발견한 것 — SEAM은 TO의 수정을 되먹이지 않는다

[seam_policy.py:194](benchmark/seam_vla/policy/seam_policy.py#L194):

```python
new_state = seam_state.with_chunk(model_chunk_np[0], physical_chunk, ...)
#                                 ^^^^^^^^^^^^^^ 정제 **전** 모델 공간 청크가 다음 prior가 된다
```

다음 청크의 SEAM prior는 **정제 전** 모델 공간 청크에서 만들어지고, 정제된 physical chunk는
로깅용으로만 저장된다. 즉 **TO의 수정은 SEAM의 prior에 전혀 반영되지 않는다.**

방치하면 SEAM이 매 청크 같은 충돌 궤적을 다시 제안하고 TO가 매번 밀어내면서, SEAM이 없애려던
경계 jerk가 TO 계층에서 되살아난다. SEAM을 수정하지 않고 고치는 방법은 있다 —
`refine()`이 이미 `context["previous_physical_chunk"]`를 받으므로, **TO의 비용 함수에
연속성 항을 넣는다.**

---

## 솔버 선택과 근거

### 방법: SQP + 희소 QP 부분문제, Real-Time Iteration 방식

**IPOPT를 쓰지 않는 이유** (사용자 지적대로 실시간성 문제):

1. **내점법은 warm-start가 구조적으로 어렵다.** IPM 반복은 barrier 파라미터 μ로 매개된
   central path를 따라간다. 이전 해가 제약 경계 근처에 있으면 — 마진을 스치는 좋은
   충돌회피 궤적이 정확히 그렇다 — 어떤 적당한 μ에서도 central path에서 멀어, 솔버가 μ를
   다시 annealing해야 한다. Active-set·ADMM·증강 라그랑주 계열에는 이 문제가 없다.
2. **반복 횟수에 상한이 없다.** IPOPT는 수렴 허용오차까지 반복하고 restoration phase에
   들어갈 수 있다. 실시간 제어는 **주기당 고정 연산 예산**이 필요하고, 그것이 RTI 방식
   (Diehl et al. 2005)이 주는 것이다 — 주기당 SQP 1회, 수렴은 피드백 루프가 담당.
3. **규모.** 기존 AG3S 하네스는 H=5, nq=2, 1,360행짜리 장난감이다. RB-Y1은 H=50, nq=12,
   약 78,000행 — 행 57배, 변수 6배.

**QP 부분솔버: 측정 후 확정.** 이 CasADi 빌드에서 실측 확인된 가용 솔버 —
`conic`: osqp · qpoases · **proxqp** · qrqp · highs · ipqp / `nlpsol`: ipopt · sqpmethod ·
feasiblesqpmethod · blocksqp. 별도로 pip `osqp` 1.1.3도 설치돼 있다.

| 후보 | 근거 | 판단 |
|---|---|---|
| **OSQP** (ADMM) | KKT 행렬을 **1회 분해하고 재사용** — ADMM 반복 간에도, 희소성 패턴이 불변인 한 솔브 간에도. 그 불변성이 정확히 AG3S가 보장하는 것이다. warm-start 네이티브. KNOWS가 이미 [filter.py](benchmark/knows_vla/cbf/filter.py)에서 같은 패턴을 쓴다 | **기본값 후보 1** |
| **ProxQP** (proximal AL) | 로보틱스용 설계(Bambade et al. RSS 2022), **퇴화·중복 제약에 강함**. AG3S는 정확히 그런 행을 낸다 — 한 후보에 대한 sphere-chain 행들은 거의 공선이고, `active=0` 행은 상수라 Jacobian이 0행이 된다 | **기본값 후보 2** |
| qpOASES | 밀집 online active set, 제약 수에 초선형. 수천 행은 사정권 밖 | 구조적 제외 |
| acados / HPIPM / cuRobo | 미설치. acados는 C codegen + 빌드 툴체인, cuRobo는 CUDA 필요. 저장소 의존성 정책("있는 것만 쓴다")에 위배 | 예산 미달 시 재검토 |

**측정 먼저, 최적화 나중.** AG3S에서 self-filter가 257 ms → 13.5 ms가 된 것은 추측이 아니라
프로파일링 결과였다. 같은 순서를 따른다 — 전체 행으로 정직한 baseline을 만들고 재본 뒤,
프로파일이 지목하는 곳만 줄인다.

---

## 설계

### D1. 결정 변수는 action 포맷이 실제로 쓸 수 있는 것만

`Q ∈ R^{12×50}` — `[L arm_0..5, R arm_0..5]`. 그리퍼(dim 6, 13)는 손대지 않고 통과,
`arm_6`는 `pi05_infer.apply_action`과 동일하게 고정, 토르소는 action 채널이 없으므로
**변수가 아니라 파라미터**다. TO가 쓸 수 없는 자유도를 최적화하면 실행 불가능한 해가 나온다.

### D2. 비용 — 참조 추종 · 평활 · 연속성 · slack

```
min  w_track ‖Q − Q_seam‖²_W        SEAM 청크가 참조
   + w_smooth ‖Δ²Q‖²                 jerk — SEAM이 존재하는 이유
   + w_cont  ‖Q[:, :L] − Q_prev‖²    이전 정제 청크와의 연속성 (위 발견 대응)
   + w_slack ‖s‖₁

s.t. h_AG3S(Q) + s ≥ 0,  s ≥ 0
     q_min ≤ Q ≤ q_max               URDF
     |ΔQ| ≤ v_max·dt                 URDF velocity
     |Δ²Q| ≤ a_max·dt²               URDF acceleration
     Q[:, 0] = q_now                 초기 조건
     ‖Q − Q_k‖_∞ ≤ trust_radius      SQP 신뢰 영역
```

**slack은 편의가 아니라 필수다.** 이미 접촉 중이거나 지각이 한 프레임 튀면 QP가 실행 불가능해지고,
그러면 컨트롤러가 아무것도 내놓지 못한다. KNOWS가 같은 결론에 도달했다 —
`filter.py` D3: "QP가 항상 풀리도록 barrier 행을 부드럽게 하고, 비상 정지를 솔버 실패가 아니라
slack에 대한 명시적 임계값으로 만든다" + 논문의 비상정지 분기가 5,879 스텝 중 0.0%로
도달 불가능했다는 측정.

**사용자 결정**: slack이 임계치를 넘으면 **최선 결과를 실행하고 상태로 보고**한다.
AG3S와 같은 철학 — TO는 "이 청크를 안전하다고 인증할 수 없다"까지만 말하고,
정지 여부는 상위 계층이 정한다.

### D3. 행 수를 줄이는 세 가지 — 프로파일이 지목하는 순서대로

H=50 · 팔 46구 · 32슬롯 = **73,600 sphere 행 + 4,600 plane 행**, 변수는 600개뿐이다.
행이 병목이지 솔버가 병목이 아니다.

1. **활성 밴드(activation band).** `h(Q)`를 Jacobian 없이 수치로만 평가하고(싼 Function 호출 1회),
   `h < band`인 행만 QP에 넘긴다. 멀리 떨어진 행은 QP 해에 영향이 없다. 안전은 두 가지로 유지한다 —
   (a) band ≥ SQP 스텝당 최대 관절 변위 × h의 Lipschitz 상한, (b) **해에서 전체 행을 다시 평가**해
   실제 최소값을 보고. TrajOpt(Schulman et al. RSS 2013)이 하는 그대로이고,
   **AG3S의 고정 그래프는 건드리지 않는다** — pruning은 TO 자신의 선택이다.
2. **시간 stride.** 50 스텝 전부를 결정 변수로 두어 **전방 시야를 유지**하되(근시안 방지),
   충돌 행은 `s` 스텝마다 건다. URDF 속도 한계 × dt로부터 연속 스텝 간 기하 이동량이 유계이므로,
   stride 2–3에서 미검사 sweep이 안전 마진보다 작다. 그 상한을 측정해 보고한다.
3. **링크 필터.** AG3S는 이미 `constraint_robot_model`을 `robot_model`과 분리해 받는다.
   오른팔 작업이면 오른팔+토르소 ~30구로 줄어든다.

### D4. RTI — 고정 예산

주기당 SQP 1–3회, 이전 청크에서 warm-start. `max_iterations`와 `time_budget_ms`로
**하드 캡**을 걸고, 캡에 걸리면 현재 반복의 최선해를 반환한다(그것이 RTI의 철학이다).
반환값에는 반복 수·소요 시간·최종 위반량이 실린다.

---

## 패키지 구조

```
benchmark/trajopt/
├── types.py          TrajOptResult · TrajOptStatus · ReferenceChunk
├── config.py         frozen dataclass + YAML (AG3S의 패턴 재사용)
├── limits.py         URDF에서 position/velocity/acceleration 한계
├── problem.py        변수 레이아웃 · 비용 · 선형 제약 · 신뢰 영역
├── linearize.py      h·∇h 평가 + 활성 밴드 + stride
├── qp.py             솔버 무관 QP 어댑터 (osqp / proxqp / qrqp / ipqp)
├── sqp.py            RTI 루프: 선형화 → QP → 스텝 수용 → 재평가
├── refiner.py        TrajOptChunkRefiner(DecodedChunkRefiner)   ← SEAM 훅
├── bringup.py        pi05_infer용 팩토리 (bringup 코드와 분리)
├── configs/rby1.yaml
├── experiments/rby1_replay.py
└── README.md
```

**재사용할 것** (새로 만들지 않는다):

| 필요 | 이미 있는 것 |
|---|---|
| 제약 벡터 · 파라미터 갱신 | `to_adapter.to_casadi()` / `refresh()` / `warm_start_map()` |
| 심볼릭 FK | `UrdfSphereChain.sphere_centers_symbolic` |
| 씬 → 제약 | `AG3S.process_multi()` |
| 단계별 계측 | `benchmark.ag3s.runtime.profiler.StageProfiler` |
| 희소 QP 패턴 | `benchmark/knows_vla/cbf/filter.py`의 osqp + scipy.sparse 사용법 |
| 설정 검증 패턴 | `benchmark.ag3s.config`의 frozen dataclass + unknown-key 하드 에러 |

**AG3S에 대한 유일한 변경(가산적)**: `UrdfJoint`가 현재 `lower`/`upper`만 파싱한다.
URDF에 있는 `velocity`·`acceleration`도 읽고 `velocity_limits()` / `acceleration_limits()`를
노출한다. 동작 변경 없음.

**SEAM은 수정하지 않는다.** `TrajOptChunkRefiner`가 `DecodedChunkRefiner`를 import해 상속할 뿐이다.
기존 스켈레톤 `refinement/collision_avoidance.py`는 그대로 두고(다른 `distance_fn` 기반 설계),
README에 trajopt가 그 자리를 대신한다고 적는다.

---

## 구현 순서

각 단계 후 `tests/trajopt` 전체 + `tests/ag3s` 회귀를 돌린다.

**T0 — 골격과 한계**
`types.py` · `config.py` · `limits.py`. URDF 속도/가속 한계 파싱(AG3S 가산 변경).

**T1 — 문제 조립**
`problem.py`: 12×50 변수 레이아웃, 추종/평활/연속성 비용, 박스·속도·가속 제약.
아직 충돌 없음. 충돌 없는 QP가 SEAM 청크를 그대로 되돌려주는지부터 확인.

**T2 — 선형화와 QP 어댑터**
`linearize.py`(전체 행, 축소 없음) + `qp.py`. **여기서 baseline을 측정한다** —
h 평가 / Jacobian / QP 솔브를 솔버 4종(osqp · proxqp · qrqp · ipqp)에 대해.

**T3 — SQP/RTI 루프**
`sqp.py`: 신뢰 영역, 스텝 수용, warm-start, 하드 캡. 해에서 전체 행 재평가 후 보고.

**T4 — 축소**
T2 프로파일이 지목한 것만: 활성 밴드 → stride → 링크 필터. 각 단계마다
"축소 전 해 == 축소 후 해"를 작은 씬에서 확인.

**T5 — SEAM 훅**
`refiner.py`. `context["previous_physical_chunk"]`로 연속성, 그리퍼·`arm_6` 불변 확인.

**T6 — 배선 (사용자 결정: 둘 다)**
- `experiments/rby1_replay.py`: TransportScene + AG3S 3-카메라 융합 씬 + 참조 청크
  (`data/policy_records`가 있으면 실측, 없으면 합성) → TO → MuJoCo 재생.
  충돌 여부·위반량·latency 보고. **src/를 건드리지 않고 전체 검증이 된다.**
- `bringup.py` + `pi05_infer.py`에 **`--trajopt` 플래그로 게이트된 최소 패치**.
  플래그가 없으면 기존 루프는 바이트 단위로 동일하다.

**T7 — 문서 · 회귀**
`benchmark/trajopt/README.md`(측정된 솔버 비교표 포함), AG3S·SEAM·KNOWS 회귀.

---

## 테스트 (`tests/trajopt/`, 합성 전용 · GPU/실기 불필요)

| 파일 | 검증 |
|---|---|
| `test_limits.py` | URDF 한계 파싱, 관절 순서 정렬 |
| `test_problem.py` | 충돌 없는 문제가 참조를 그대로 반환, 박스/속도/가속이 실제로 구속 |
| `test_linearize.py` | ∇h가 유한차분과 일치, **활성 밴드 pruning이 해를 바꾸지 않음**(작은 씬에서 전체 대조), stride의 미검사 sweep 상한 |
| `test_qp.py` | 솔버 4종이 같은 QP에서 허용오차 내 일치, warm-start가 반복 수를 줄임 |
| `test_sqp.py` | 충돌 궤적이 안전해짐 + **해에서 전체 행 재평가 시 위반 없음**, 하드 캡에서 최선해 반환, 실행 불가능 입력이 예외 대신 상태 반환 |
| `test_refiner.py` | shape 보존, 그리퍼·`arm_6` 불변, 연속성 항이 경계 jerk를 줄임, SEAM 코드 미수정 확인 |

---

## 검증

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws

src/openpi/.venv/bin/python -m pytest tests/trajopt -q
src/openpi/.venv/bin/python -m pytest tests/ag3s -q          # 회귀 (현재 422 passed)

# 솔버 벤치마크 — 실제 RB-Y1 문제에서 측정 후 기본값 확정
src/openpi/.venv/bin/python -m benchmark.trajopt.qp --benchmark

# 3-카메라 융합 씬 + TO + MuJoCo 재생
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.trajopt.experiments.rby1_replay \
    --json benchmark/trajopt/asset/rby1_replay.json

JAX_PLATFORMS=cpu src/openpi/.venv/bin/python -m pytest benchmark/seam_vla/tests/ -q
src/openpi/.venv/bin/python -m pytest benchmark/knows_vla/tests/test_cbf.py -q
```

**완료 기준**: 신규 테스트 전부 통과 + AG3S 422개 회귀 유지 / 솔버 4종 실측 비교표와
그로부터 확정한 기본값 / 청크당 latency가 목표 30 ms 대비 어디인지 수치로 / 축소 전후 해 동일성 /
MuJoCo 재생에서 충돌 궤적이 안전해지는 것을 GT로 확인 / 한국어 최종 보고서.

## 리스크

- **78,000행이 30 ms에 안 들어갈 수 있다.** T2 측정이 이 계획의 분기점이다.
  미달이면 순서대로: 활성 밴드 → stride → 링크 필터 → horizon 축소(50→K+lookahead).
  그래도 미달이면 acados/HPIPM 도입을 **측정 근거와 함께** 다시 제안한다.
- **SEAM prior 되먹임 부재는 완화이지 해결이 아니다.** 연속성 항은 경계 jerk를 줄이지만,
  SEAM은 여전히 매 청크 충돌 궤적을 제안한다. 근본 해결은 physical→model 역변환이 필요하고
  SEAM 수정을 수반하므로 이번 범위 밖 — 보고서에 명시한다.
- **참조 청크가 없을 수 있다.** `data/policy_records` 유무를 T6에서 확인하고, 없으면 합성
  참조로 검증하되 "실제 π0.5 청크에서는 미검증"임을 명시한다.
- **SQP는 지역해만 준다.** 충돌 회피는 비볼록이라 신뢰 영역 초기값에 따라 다른 지역해에 간다.
  SEAM 청크를 초기값으로 쓰는 것이 그 자체로 합리적 선택이지만, 전역 최적을 주장하지 않는다.
