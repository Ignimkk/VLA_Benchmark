# 연구 실험 runbook — contribution/novelty 확보용

> 대상 하드웨어: 사용자 GPU 서버 (π0.5 추론 필요). 실험 0만 로컬에서 돌아갑니다.
> 목적: "모듈을 구현했다"가 아니라 **"Y가 참임을 보인다"**로 논문의 축을 옮기기 위한 실험 계획.
> 상태 표기: ✅ 실행 완료 · 🔧 코드 필요 · ▶ 커맨드 준비됨

---

## 0. 실험 순서와 예산

| # | 실험 | GPU | 소요 | 산출물 | 상태 |
|---|---|---|---|---|---|
| 0 | 청크 위상별 risk 진단 | 불필요 | 15분 | 논문 Fig 1 | ✅ |
| A | Semantic budget ε*(phase) | 필요 | ~30 GPU-h | 논문 Fig 3, TO의 trust region | 🔧 |
| D | 장애물 씬 4-way ablation | 필요 | ~40 GPU-h | 논문 Fig 4 (Pareto) | ▶ |
| C | Closed-loop prior feedback | 필요 | ~15 GPU-h | 논문 Fig 2 | 🔧 |

**A를 D보다 먼저 하십시오.** A가 내놓는 ε*(phase)가 D에서 쓸 `w_track`·trust radius를
결정합니다. 순서를 뒤집으면 D를 두 번 돌리게 됩니다.

기준 처리량 (2026-08-03 그리드 실측): 216 trial · 조건당 약 67 s/trial → **조건당 약 4 GPU-h**.

---

## 1. 실험 0 — 청크 위상별 risk 진단 ✅

```bash
src/openpi/.venv/bin/python src/scripts/analyze_boundary_risk.py \
    --baseline data/rby1_grid_eval_baseline \
    --seam     data/rby1_grid_eval_seam \
    --out      docs/assets/boundary_risk.md \
    --json-out docs/assets/boundary_risk.json

# 컨테이너(= place 동작 자체)를 뺀 "닿으면 안 되는 것들"만
src/openpi/.venv/bin/python src/scripts/analyze_boundary_risk.py \
    --baseline data/rby1_grid_eval_baseline --seam data/rby1_grid_eval_seam \
    --channels table,self \
    --out docs/assets/boundary_risk_nocontainer.md
```

기록된 `measured_qpos`에서 AG3S/trajopt와 **같은 충돌 모델**로 clearance를 재계산합니다.
정책도 시뮬레이터 동역학도 GPU도 필요 없습니다.

### 결과 — 원래 가설은 기각, 더 나은 것이 나왔다

원래 가설("위험은 청크 경계에 몰린다")은 **틀렸습니다.** 216쌍 × 2조건에서 나온 것은:

| 관측 | 값 | 유의성 |
|---|---|---|
| 청크 내 위상별 clearance 프로파일 | **단조 감소** (phase 0 → 7) | — |
| 침식 기울기 (table+self) | baseline **−0.176** / SEAM **−0.172** mm/step | 사실상 동일 |
| SEAM의 수준 이동 | **+6.99 mm**, 위상 간 std **0.06 mm** | 완전히 균일한 평행이동 |
| 경계 스텝 vs 내부 스텝 | 경계가 **+1.0 mm 더 안전** | p ≈ 3e-37 |
| trial 최저 clearance의 위상 | **phase 7**에 집중 (51/216, 균등이면 27) | p = 4e-5 |
| 경계 불연속 크기 ↔ clearance 손실 | **부호가 반대** — 큰 불연속일수록 clearance를 **회복** | rho −0.19, 최상위 decile은 +0.45 mm 획득 |

읽는 법:

1. **청크를 open-loop로 실행하는 동안 clearance가 단조적으로 침식된다.** K=8 스텝에 걸쳐
   약 1.1 mm. 최악 지점은 경계가 아니라 **경계 직전(phase 7)**이다.
2. **재계획이 그것을 되돌린다.** 그리고 경계 불연속이 클수록 더 많이 되돌린다 — 큰 점프는
   위험 요인이 아니라 **교정**이다. 정책이 낡은 계획을 버리는 순간이기 때문이다.
3. **SEAM은 기울기를 건드리지 않는다.** clearance 수준을 +7 mm 평행이동시킬 뿐 침식률은
   −0.176 → −0.172로 사실상 그대로다.

### 따라서 논문의 주장은 이렇게 바뀝니다

> VLA 청킹의 안전 문제는 **경계의 거칠기가 아니라 open-loop 지평 K**다. 안전 계층은 이음매가
> 아니라 **청크 전체**에 작용해야 한다. Smoothing은 clearance의 *수준*을 바꾸고, 기하를 아는
> 최적화만이 *기울기*를 바꿀 수 있다.

이것이 TO 계층을 SEAM 위에 얹는 이유를 처음으로 **데이터로** 정당화합니다. 실험 D가 검증할
예측: TO는 프로파일의 기울기를 평탄화해야 하며, 단지 수준만 올린다면 SEAM과 중복이다.

### 한계 (그대로 적어두십시오)

- 이 데이터셋에는 **피할 장애물이 없습니다.** clearance는 테이블·컨테이너·자기 자신 상대이고,
  컨테이너는 place 동작 자체라 `--channels table,self` 변형이 주 결과입니다.
- 운동학 재계산이며 URDF `arm_5` 캡슐은 보수적 bounding cylinder라 절대 mm 값은 신뢰 구간이
  넓습니다. 모든 검정이 trial 내 paired이므로 상수 오프셋은 상쇄됩니다.
- **상관이지 인과가 아닙니다.** 인과는 실험 A(ε 스윕)와 D(장애물 ablation)가 만듭니다.

산출물: [`docs/assets/boundary_risk.md`](../../../docs/assets/boundary_risk.md) (전 채널) ·
[`docs/assets/boundary_risk_nocontainer.md`](../../../docs/assets/boundary_risk_nocontainer.md)
(**주 결과**, table+self).

---

## 2. 실험 A — Semantic Budget ε*(phase) 🔧

### 주장

> Safety filter는 정책 출력을 수정한다. **얼마나** 수정해도 태스크가 안 깨지는지를 아무도
> 재지 않았다. 그 양은 phase마다 자릿수가 다르며, 따라서 균일한 `w_track`은 틀렸다.

$$\varepsilon^*(\phi) = \max\{\varepsilon : P(\text{success} \mid \|Q - Q_\pi\|_\infty \le \varepsilon,\ \phi) \ge P_0 - \delta\}$$

### 필요한 코드 — `PerturbationRefiner`

`benchmark/seam_vla/refinement/base.py`의 `DecodedChunkRefiner`를 상속해 **통제된 크기의
실행 가능한 섭동**을 주입합니다. 랜덤 노이즈는 안 됩니다 — 그것은 진동을 측정할 뿐 "safety
filter가 밀어낸 궤적"을 측정하지 않습니다. 섭동은 다음 성질을 만족해야 합니다.

1. 크기가 `ε`로 정확히 통제될 것 (EE 공간 또는 관절 공간, 하나로 고정)
2. 로봇의 속도·가속 한계를 지킬 것 → `benchmark/trajopt/limits.py` 재사용
3. 청크 경계에서 연속일 것 → 그렇지 않으면 저크가 교란변수로 들어옴
4. **phase를 알 것** → 그리퍼 채널로 approach/grasp/transport/place 라벨링
   (`benchmark/seam_vla/docs/root_cause_analysis_ko.md`가 쓴 것과 같은 방법)

가장 깔끔한 구현: **이미 만든 TO를 가짜 장애물로 구동**합니다. 목표 clearance를 인위적으로
키우면 TO가 실제 회피 방향으로, 한계를 지키며, 연속으로 궤적을 밀어냅니다. 그 결과의
`‖Q* − Q_π‖`를 재고 ε 빈으로 나눕니다. 새 섭동 모델을 발명할 필요가 없습니다.

### 실행

```bash
# ε 스윕. ID 영역만 — x=0.625 열은 OOD라 신호가 묻힙니다.
for EPS in 0.000 0.005 0.010 0.020 0.040 0.080; do
  src/openpi/.venv/bin/python -m rby1_bringup.pi05_ex_infer \
      --model pi05_rby1_lora --grid-experiment --headless \
      --grid-config <grid_config_006698dac3ef.json> \
      --grid-repeats 3 --trial-max-steps 600 \
      --refiner perturbation --perturb-eps "$EPS" \
      --grid-output-dir "data/eps_sweep/eps_${EPS}" \
      --grid-save-trajectories
done
```

> `--refiner` / `--perturb-eps` 는 아직 없습니다. `pi05_ex_infer.py`의 SeamPolicy 생성부에
> `refiner=` 를 넘기는 한 줄이 추가돼야 합니다 ([seam_policy.py:181](../../seam_vla/policy/seam_policy.py#L181)이
> 이미 훅을 갖고 있습니다).

### 분석

phase × ε 격자에서 성공률 곡선 → `ε*(φ)`. **예상되는 그림**: GRASP는 mm 단위에서 무너지고
TRANSPORT는 수 cm까지 견딥니다. 그 차이가 크면 논문이 되고, 없으면 그것도 결과입니다
("VLA 청크는 phase와 무관하게 균일한 여유를 갖는다" — 역시 아무도 모르는 사실).

### 되먹임

측정된 `ε*(φ)`를 TO의 phase별 trust region으로 넣습니다:
`benchmark/trajopt/configs/rby1.yaml`의 `trust_radius`를 스칼라에서 phase 딕셔너리로.

---

## 3. 실험 D — 장애물 씬 4-way ablation ▶

### 주장

> smoothness 제약과 safety 제약을 **순차로 쌓으면** 서로의 목표를 훼손한다. 하나의 문제로
> 풀면 Pareto 지배한다.

### 장애물은 이미 있습니다

`src/rby1_manipulation/src/rby1_manipulation/config/pick_place_obstacles.json`:

| profile | 성격 |
|---|---|
| `right_bollard` / `left_bollard` | 접근 경로 위 기둥 |
| `right_divider` / `left_divider` | 테이블 가로지르는 칸막이 |
| `dual_bollards` | 양쪽 기둥 |

`PickPlaceObstacleManager`가 trial마다 `robot_collision`(이진), `min_robot_clearance_m`(연속),
`contact_pairs`를 기록합니다. **종속변수가 이미 계측되어 있습니다.**

> **중요**: `--obstacle-stop-distance 0`으로 두십시오. 기본값 0.02는 근접 시 로봇을
> **정지**시키므로 "정책이 얼마나 충돌하는가"를 측정할 수 없습니다. 0이면 실제 접촉에서만
> 멈추고, 그 접촉이 곧 관측하려는 사건입니다.

### 조건 5개

| 조건 | guidance | refiner | 무엇을 분리하는가 |
|---|---|---|---|
| `baseline` | Identity | Identity | 대조군 |
| `seam` | VLS | Identity | smoothness 단독 |
| `to` | Identity | TrajOpt | safety 단독 |
| `seq` | VLS | TrajOpt | 순차 stacking |
| `joint` | VLS | TrajOpt (`w_track`에 tail-consistency 포함) | 통합 정식화 |

`joint`는 새 코드가 거의 필요 없습니다 — [problem.py](../problem.py)가 이미
`w_track‖Q−Q_seam‖² + w_smooth‖Δ²Q‖² + w_cont‖Q−Q_prev‖²`를 갖고 있고, `seq`는 그 가중치를
0으로 둔 특수 케이스입니다.

### 실행

```bash
for PROFILE in clear right_bollard left_divider dual_bollards; do
for COND in baseline seam to seq joint; do
  src/openpi/.venv/bin/python -m rby1_bringup.pi05_ex_infer \
      --model pi05_rby1_lora --grid-experiment --headless \
      --obstacle-profile "$PROFILE" --obstacle-stop-distance 0 \
      $( [ "$COND" = baseline ] || [ "$COND" = to ] || echo --seam ) \
      --grid-repeats 3 --trial-max-steps 600 \
      --grid-output-dir "data/ablation/${PROFILE}__${COND}" \
      --grid-save-trajectories
done; done
```

> `to`/`seq`/`joint`는 `--refiner trajopt` 스위치가 필요합니다 (실험 A와 같은 훅).
> `--obstacle-profile`은 `--grid-experiment`와 함께 쓸 때 `--model pi05_rby1_lora` 계열만
> 허용됩니다 ([pi05_infer.py:635](../../../src/rby1_bringup/pi05_infer.py#L635) 참고).

### 분석

축 3개로 Pareto frontier: **(경계 불연속 CD, 최소 clearance, 성공률)**.
주장할 것: `seq`는 지배당하고 `joint`는 지배한다. 216쌍 paired McNemar로 성공률이 보존됨을
같이 보입니다 — `src/scripts/compare_grid_conditions.py`가 이미 그 검정을 합니다.

**함정**: `clear` 프로파일도 반드시 포함하십시오. 장애물이 없을 때 TO가 성공률을 깎지
않는다는 것을 보이지 못하면, 개선이 아니라 보수화로 읽힙니다.

---

## 4. 실험 C — Closed-loop prior feedback 🔧

### 주장

> Post-hoc safety filter는 action-chunked VLA에서 **구조적으로 불안정**하다. 정책은 매 청크
> 같은 충돌 궤적을 다시 제안하고 필터는 매번 밀어낸다.

### 측정 — 새 실험 없이 D의 데이터로 가능

D의 `seq` 조건에서 연속 청크의 보정량 `‖Q* − Q_π‖`의 시계열을 뽑습니다. 닫힌 루프라면
감쇠해야 하고, 상수로 유지되면 정책과 필터가 싸우고 있다는 직접 증거입니다.
이것만으로 논문 Figure 2가 나옵니다.

### 해법 두 가지

| | 비용 | 내용 |
|---|---|---|
| 약 | 낮음 | 정제된 physical chunk → model space 역변환 → 다음 SEAM prior. [seam_policy.py:194](../../seam_vla/policy/seam_policy.py#L194)의 `with_chunk(model_chunk_np[0], ...)`를 정제 후 값으로. **역변환이 손실 없는지 먼저 검증할 것** |
| 강 | 높음 | denoising ODE 안에서 충돌 gradient guidance. `guidance_fn` 훅이 [pi0.py](../../../src/openpi/models/pi0.py) `sample_actions`에 **이미 있습니다** |

강 버전의 방어 논리는 속도입니다. Euler step이 10회뿐이고 제어 주기가 66.7 ms이므로
gradient가 1 ms 안에 나와야 합니다. [README](../README.md)의 실측표 —
CasADi 단일 그래프 19 ms vs **닫힌 형태 sphere FK + 위치 Jacobian 1.07 ms** — 가
엔지니어링 각주가 아니라 **method enabler**가 되는 지점입니다.

---

## 5. 체크리스트 — 실험 전에 확인할 것

- [ ] **OOD 열을 빼십시오.** x=0.625는 학습 범위(x ≤ 0.60) 밖이고 성공률 18~20%이며 사실상
      확률적입니다 ([root_cause_analysis_ko.md](../../seam_vla/docs/root_cause_analysis_ko.md) §0).
      거기서는 어떤 method도 유의한 신호를 못 냅니다.
- [ ] **paired 설계를 유지하십시오.** 같은 `grid_config` fingerprint와 같은 반복 인덱스를 쓰면
      `reset_and_place_trial()`이 초기 상태를 결정론적으로 재현합니다. 이 통계적 엄밀성이
      이 분야 논문 대부분보다 강한 점입니다 — 버리지 마십시오.
- [ ] **trial마다 `seam_reset`을 보내십시오.** 에피소드 간 SEAM 상태 누수를 막습니다.
- [ ] **KNOWS의 CBF를 baseline으로 한 줄 넣으십시오.** 리뷰어는 반드시 "왜 CBF가 아니라
      TO인가"를 묻습니다. `benchmark/knows_vla/cbf/filter.py`가 이미 있습니다.
- [ ] **TO의 실측 수치는 전부 합성 시나리오 기준**입니다 ([README](../README.md) 알려진 한계).
      실제 π0.5 청크로 28.4 ms가 유지되는지가 D의 부수적 검증 항목입니다.
