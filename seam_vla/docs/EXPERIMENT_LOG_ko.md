# RB-Y1 SEAM 평가 — 전체 실험 기록

> 기간: 2026-08-03 ~ 2026-08-05
> 대상: π0.5 + LoRA(`pi05_rby1_lora`, 30k steps), RB-Y1 dual-arm, MuJoCo
> 총 실행: **972 trials** (baseline 216 + SEAM 216 + OOD SEAM 540), 데이터 833 MB
>
> 이 문서는 수행한 모든 실험의 **무엇을 / 어떻게 / 결과가 어땠는지**를 하나로 정리한 기록입니다.
> 개별 상세는 각 절 끝의 링크를 참조하세요.

---

## 0. 한 장 요약

| | 내용 |
|---|---|
| **출발 질문** | SEAM(VLS)이 task 성공률을 유지하면서 chunk 경계 jerk를 줄이는가? |
| **답 (모션)** | **예.** commanded BJ −23.4%, 경계/내부 변위비 1.295 → 1.006, 추론 비용 +0.4% |
| **답 (성공률)** | **차이 없음.** 78.7% → 80.1%, McNemar p=0.629 |
| **예상 밖 발견** | 실패의 근본 원인은 SEAM과 무관. **평가 격자 x=0.625가 팔 최대 도달거리를 3.3% 초과**하고 학습 분포(x ≤ 0.60) 밖 |
| **결론** | 현 벤치마크는 SEAM을 성공률로 변별할 수 없음. 주장을 **모션 품질 + 성공률 비열화**로 재구성해야 함 |

---

## 1. 구축한 실험 인프라

### 1.1 실행 코드 수정 — `src/rby1_bringup/pi05_ex_infer.py`

기존 그리드 루프에 4가지 기록 기능을 추가했습니다(모두 기존 스키마에 **추가만**, 하위 호환 유지).

| 추가 | 내용 |
|---|---|
| 정면뷰 녹화 | `--grid-record`. `--view`가 `--grid-experiment`에서 `front`로 자동 기본값 |
| 3카메라 녹화 | `--grid-record-inputs`. cam_high / cam_left_wrist / cam_right_wrist, **정책이 실제로 본 프레임만**(추론 시점, 15Hz÷8 = 1.875 fps) |
| 정량 지표 | `compute_trial_metrics()` — BJ/IJ/CD/AVb를 commanded·measured 양쪽에서 계산해 `results.jsonl`에 인라인 저장 |
| 히트맵용 좌표 | `grid_cell_indices()` — 실좌표에서 (col, row) 유도, `control_hz`/`execution_length`/`inference_ms`/`num_vls_chunks` 추가 |
| 편의 | `--grid-record-all` (위 3개 한번에) |

**검증 방법**: 스텁 정책(관측 state를 그대로 반환)으로 24 trials 드라이런 → 정면뷰 MP4, 3카메라 MP4, NPZ, results.jsonl, 히트맵까지 전 경로 확인. BJ 0.0129 vs IJ 0.0016으로 경계 아티팩트가 정상 측정됨을 확인.

### 1.2 분석 스크립트 (신규 5종)

| 스크립트 | 역할 |
|---|---|
| `scripts/plot_grid_heatmap.py` | 테이블 평면도(top-down) 성공률/지표 히트맵. `--merge-colors`, `--metric BJ\|IJ\|CD\|AVb`, `--condition` |
| `scripts/compare_grid_conditions.py` | paired 비교. NPZ에서 지표 **재계산** + Wilson CI + McNemar + Wilcoxon |
| `scripts/plot_condition_summary.py` | 논문용 핵심 1장 (성공률 CI + 지표 변화율) |
| `scripts/plot_jerk_comparison.py` | chunk 경계 정렬 프로파일 + 지표 막대 + 수치표 |
| `scripts/replay_trial.py` | 정책 없이 저장된 action만 MuJoCo에 재주입. `--mode chunk`로 계획 전체 실행 가능 |

모든 지표는 **arm 12관절**(그리퍼 제외, 근사 이진값이라 저크 노름을 지배)에서 계산하며, `benchmark/seam_vla/metrics/motion.py`의 논문 정의(Eq. 9–13)를 사용합니다.

### 1.3 데이터

```
data/rby1_grid_eval_baseline/     149 MB   216 trials  (grid 006698dac3ef)
data/rby1_grid_eval_seam/         142 MB   216 trials  (grid 006698dac3ef)
data/rby1_grid_eval_false_scene/  542 MB   540 trials  (grid 023080190bbd, SEAM only)
```

각 trial마다: `results.jsonl` 1줄 + 정면뷰 MP4 + 3카메라 MP4 + 궤적 NPZ
(NPZ 키: `executed_actions`, `measured_qpos`, `predicted_chunks`, `chunk_start_steps`, `inference_states`, `inference_ms`, `used_vls`)

---

## 2. 실험 1 — 기본 그리드 baseline vs SEAM

### 무엇을

좌/우 각 12지점(x ∈ {0.475, 0.525, 0.575, 0.625} × |y| ∈ {0.178, 0.235, 0.292}) × 3색 × 3반복 = **조건당 216 trials**.

### 어떻게

- 서버(`serve_seam_policy`)의 `--seam-config`를 `seam_rby1.yaml` ↔ `baseline_rby1.yaml`로 교체해 조건 전환
- 클라이언트는 `--seam` 유무로 라벨링 + 매 trial 시작 시 `seam_reset` 전송(에피소드 간 SEAM 상태 누수 차단)
- `reset_and_place_trial()`이 결정론적으로 초기 상태를 재현 → **216쌍 완전 매칭**(trial_id 불일치 0) → paired 검정 가능
- 상한 600 step(40 s), 성공 = 블록이 컨테이너 ±0.075 m 내에서 8 step 연속 정지

### 결과

**성공률: 차이 없음**

| 조건 | 성공 | 성공률 | 95% CI (Wilson) |
|---|---:|---:|---|
| baseline | 170/216 | 78.7% | [72.8%, 83.6%] |
| SEAM | 173/216 | 80.1% | [74.3%, 84.9%] |

Paired 분할표: 둘 다 성공 163, baseline만 7, SEAM만 10, 둘 다 실패 36 → **McNemar 정확검정 p = 0.629**

**모션 품질: 전 지표 유의하게 개선** (NPZ 재계산, Wilcoxon signed-rank, n=216)

| 지표 | commanded | measured |
|---|---:|---:|
| BJ | 0.02802 → 0.02147 (**−23.4%**) | 0.00362 → 0.00296 (−18.1%) |
| IJ | 0.01274 → 0.00919 (**−27.9%**) | 0.00318 → 0.00267 (−15.9%) |
| CD | 0.02289 → 0.01810 (**−20.9%**) | 0.01381 → 0.01551 (**+12.4%**) |
| AVb | −31.7% | −25.9% |
| overlap residual | 0.07549 → 0.06355 (−15.8%) | — |

전부 p<0.001. **measured CD만 역전**(§5.3에서 규명).

**비용**: 추론 지연 평균 218.6 → 219.5 ms (**+0.4%**), 중앙값 341 vs 339 ms. 최대치(6.2 s / 12.9 s)는 첫 호출·서버 재시작 직후 JIT 이상치.

**공간 분포**

| 구역 | n | baseline | SEAM |
|---|---:|---:|---:|
| x < 0.625 | 162 | 160 (98.8%) | 162 (**100%**) |
| x = 0.625 | 54 | 10 (18.5%) | 11 (20.4%) |

전체 78~80%는 **x=0.625 한 열이 만든 수치**입니다.

📄 상세: [`grid_evaluation_report_ko.md`](grid_evaluation_report_ko.md)
🖼 [summary_baseline_vs_seam.png](../../../docs/assets/summary_baseline_vs_seam.png) · [cmp_success_merged.png](../../../docs/assets/cmp_success_merged.png) · [jerk_boundary_profile.png](../../../docs/assets/jerk_boundary_profile.png) · [jerk_metrics_bars.png](../../../docs/assets/jerk_metrics_bars.png)

---

## 3. 실험 2 — 실패 구역 고표본 (false_scene)

### 무엇을

실험 1에서 셀당 n=9로 결론 불가였던 x=0.625를 **n=90**으로 재측정. 기존 |y| ∈ {0.178, 0.235}에 **보간 지점 0.206을 신규 추가**.

`src/rby1_manipulation/block_false_grid.json` — 6지점(±0.178, ±0.206, ±0.235) × 3색 × 30반복 = **540 trials**, SEAM 단일 조건.

### 어떻게

기존 코드 그대로(`--grid-config`, `--grid-repeats 30`). `grid_fingerprint`가 달라 기존 실험과 섞이지 않음. distractor 배치는 원본과 동일하게 유지.

### 결과 — 성공률이 |y|에 대해 **단조적이지 않음**

| y | 성공률 | 95% CI |
|---:|---:|---|
| +0.235 | 31.1% | [22.5%, 41.3%] |
| **+0.206** | **67.8%** | [57.6%, 76.5%] |
| +0.178 | 48.9% | [38.8%, 59.0%] |
| **−0.178** | **0.0%** | [0.0%, 4.1%] |
| **−0.206** | **76.7%** | [66.9%, 84.2%] |
| −0.235 | 17.8% | [11.2%, 26.9%] |

|y|로 묶으면 0.178 → 24.4%, **0.206 → 72.2%**, 0.235 → 24.4%.
**|y|=0.206 vs 나머지: Fisher exact p = 1.5 × 10⁻²⁶.** 좌우 양쪽에서 동시에 나타남.

기존 n=9 추정치와 비교하면 4개 중첩 위치 전부 유의차 없음(p > 0.16) → 기존 추정은 **편향되지 않았으나 부정확**했음(+0.178이 78% → 48.9%, 전형적 평균 회귀).

📄 상세: [`false_scene_report_ko.md`](false_scene_report_ko.md)
🖼 [false_scene_success.png](../../../docs/assets/false_scene_success.png)

---

## 4. 분석 A — 실패 메커니즘 규명

### 어떻게

궤적 NPZ의 **그리퍼 채널**(action dim 6/13, 1=열림 0=닫힘) + MuJoCo **순기구학**으로 EE 궤적 복원 + 녹화 영상 대조.

### 결과 1: 2단계 분해

| | 비율 | 이후 태스크 성공률 |
|---|---:|---:|
| 파지 성공 | 456/540 (84.4%) | 218/456 (47.8%) |
| 파지 실패 | 84/540 (15.6%) | **0/84 (0%)** |

파지 실패는 태스크 실패의 **완벽한 예측자**. −0.178을 빼면 파지는 어디서나 93~100%이므로 **성공률 차이는 거의 전부 배치 단계**에서 발생.

### 결과 2: 배치 실패 = 헛파지 후 밀기

증거 6가지가 한 방향을 가리킴:

1. **파지 순간 기하가 성공/실패 동일** — EE↔블록 수평오차 0.0330 vs 0.0322 (+0.235)
2. **팔은 양쪽 다 들어올림** — 파지 직후 40 step에 8~9 cm 상승, 2 cm 초과 비율 100%
3. **그런데 블록은 안 따라옴** — 실패 322건의 최종 z = **0.8449**(테이블면), 컨테이너 진입 **0.0%**
4. **블록이 항상 같은 곳에서 멈춤** — 실패의 **82%가 |y| = 0.135 ± 0.02**
   (MuJoCo 실측: 벽 바깥면 0.110 + 블록 반폭 0.025 = 밀착 시 중심 0.135)
5. **릴리스는 컨테이너 위에서 발생**(88~96%) — 옮겼다고 판단하고 **빈 손으로 놓음**
6. **영상 확인** — `red_left_g03_r2` 프레임 100→160: 블록은 벽 바깥, 팔은 컨테이너 안

> 그리퍼가 닫히지만 블록을 확보하지 못한 채 팔만 올라가고, 횡이동하며 블록을 **밀어** 벽에 걸리게 한 뒤 빈 손으로 릴리스한다.

### 결과 3: y = −0.178은 유일하게 다른 모드

| 지표 | −0.178 | 다른 위치 |
|---|---:|---:|
| 파지율 | **27.8%** | 93~100% |
| 첫 파지 시점 | **step 113** | step 54~56 |
| 파지 직후 EE 상승 | **0.005 m** | 0.084~0.096 m |
| 파지 성공 시 태스크 성공 | **0/25** | 19~79% |

📄 상세: [`root_cause_analysis_ko.md`](root_cause_analysis_ko.md) §1

---

## 5. 분석 B — SEAM 저크 역전 구간

### 어떻게

216 paired trials에서 **경계별** 저크를 짝지어 비교하고, overlap residual과의 관계를 조건 내부에서 상관 분석.

### 결과 1: 체계적 역전 구간은 없다

| 공간 | SEAM이 더 큰 경계 | 평균 BJ가 SEAM>baseline인 trial |
|---|---:|---:|
| commanded | 1788/6407 (27.9%) | **4/216** |
| measured | 2241/6407 (35.0%) | 20/216 |

chunk 인덱스별로 나눠도 전 구간 평균 차이가 음수(SEAM 우세). **경계 정렬 프로파일 17개 offset 전부에서 SEAM의 저크가 더 낮음(0/17).**

### 결과 2: 진짜 한계는 overlap residual

| | residual 중앙값 | residual→jerk Spearman ρ |
|---|---:|---:|
| baseline | 0.0787 | +0.397 |
| SEAM | 0.0606 (**−23%**) | **+0.395** |

**SEAM은 residual 분포를 아래로 밀지만 residual→jerk 기울기는 전혀 바꾸지 못함.** 결과적으로 SEAM 최악 사분위(0.0281) > baseline 최선 사분위(0.0196).

정책이 정당한 이유로 계획을 크게 바꾸는 순간(관측 갱신)에는 개입 여력이 없음 — **SEAM의 구조적 상한**. 이를 넘으려면 이전 tail이 아니라 장면 기하를 참조하는 prior가 필요.

### 결과 3: measured CD +12.4%는 속도 산물

성공 trial 163건 기준:

| | baseline | SEAM | 변화 |
|---|---:|---:|---:|
| 평균 스텝 변위(속도) | 0.0163 | 0.0172 | **+5.1%** |
| 에피소드 길이 | 189.2 | 177.7 | **−6.1%** |
| 총 이동 거리 | 1.5255 | 1.5239 | −0.1% |

**같은 경로를 6.1% 빨리 주파.** CD는 1차 차분(속도)이라 산술적으로 상승. 속도 불변 지표(경계/내부 변위비)로 보면 commanded **1.295 → 1.006**, measured는 양 조건 모두 1 미만.

### 결과 4: 명령 vs 실측의 차이

| | baseline 피크 | 내부 바닥 | 피크/바닥 |
|---|---:|---:|---:|
| commanded | 0.03636 | 0.00865 | **4.20** |
| measured | 0.00640 | 0.00216 | **2.96** |

**명령 피크가 실측 피크의 5.7배** — 위치 액추에이터와 관성이 불연속의 대부분을 이미 흡수. SEAM이 고치는 문제는 상당 부분 **명령 레벨**에 있으므로 실기 검증이 필요.

🖼 [residual_vs_jerk.png](../../../docs/assets/residual_vs_jerk.png)

---

## 6. 분석 C — 근본 원인: 도달 한계 초과 + 학습 분포 이탈

### 어떻게

`scene_utils.py`의 데이터 수집 범위 확인 + MuJoCo에서 shoulder roll 회전중심 특정 + 관절 한계 내 4000회 샘플링으로 최대 도달거리 추정.

### 결과 1: 학습 분포

```python
RIGHT_ARM_REACH = dict(x=(0.45, 0.60), y=(-0.32, -0.15))
LEFT_ARM_REACH  = dict(x=(0.45, 0.60), y=( 0.15,  0.32))
```

**학습 시 x 최대가 0.60.** 평가 격자 x=0.625는 2.5 cm 밖. y는 전부 학습 범위 안.

### 결과 2: 물리적 도달 한계

shoulder roll 회전중심 = `[0, ±0.22, 1.37]` (pitch/roll/yaw = `arm_0/1/2`가 모두 이 점에 모인 구면 관절)
왼팔 최대 도달거리(관절 한계 내 4000샘플) ≈ **0.7900 m**

| x | shoulder roll까지 | 최대 도달 대비 | 성공률 |
|---:|---:|---:|---:|
| 0.475 | 0.708~0.712 | 89.6~90.1% | 100% |
| 0.525 | 0.743~0.746 | 94.0~94.4% | 100% |
| 0.575 | 0.779~0.782 | 98.6~99.0% | 100% |
| **0.625** | **0.816~0.817** | **103.3~103.5%** | **20%** |

**x=0.625는 팔 최대 도달거리를 3.3% 초과합니다.** 몸통을 굽혀야 닿는 지점인데 정책은 팔 12관절만 출력하고 torso는 teleop 자세에 고정되어 있어, 실제로는 **완전 신전(full extension)** 상태에서 간신히 닿습니다. 완전 신전 근처에서는 자코비안이 특이해져 미세 관절 오차가 말단에서 크게 증폭됩니다.

### 결과 3: 거리로는 sweet spot을 설명 못함

| \|y\| | shoulder roll까지 | 성공률 |
|---:|---:|---:|
| 0.235 | 0.8164 | 24.4% |
| 0.206 | **0.8164** | **72.2%** |
| 0.178 | 0.8173 | 24.4% |

**편차 0.9 mm.** 0.235와 0.206은 소수점 4자리까지 동일한데 성공률은 3배 차이. sweet spot은 거리가 아니라 **완전 신전 상태에서의 자세(configuration) 차이**로 남습니다.

📄 상세: [`root_cause_analysis_ko.md`](root_cause_analysis_ko.md) §0

---

## 7. 분석 D — 색상 효과

### 어떻게

CIELAB 색차 계산 + 학습 데이터 구성 확인 + 위치별 유의성 검정 + 색상 간 궤적 차이 vs 같은 색 내부 산포 비교.

### 결과: 실재하나 궤적에는 흔적이 없음

| 구간 | red | green | blue | χ² p |
|---|---:|---:|---:|---:|
| **x ≤ 0.575 (분포 안)** | 99.1% | 100.0% | 99.1% | 0.60 |
| x = 0.625 (분포 밖) | 27.8% | 13.9% | 16.7% | 0.29 |
| OOD 풀링(−0.178 제외) | 58.7% | 47.3% | 39.3% | **0.0035** |

**학습 분포 안에서는 세 색이 모두 99~100%로 동일.** 위치별로는 6곳 중 2곳만 유의(+0.178 p=0.039, −0.235 p=0.024), 풀링해야 p=0.0035.

**(0.625, +0.235)에서 색상별 궤적 비교** (n=30씩)

| 비교 | 평균 차이 |
|---|---:|
| red vs green | 0.01080 rad |
| red vs blue | 0.00470 rad |
| green vs blue | 0.00835 rad |
| **같은 색 내부 산포(표준편차)** | **0.0219~0.0235 rad** |

**색상 간 차이 < 같은 색 반복 시 샘플링 산포 (2~5배).** BJ/IJ/CD, 가동폭, 파지 시점, 그리퍼 실측값 모두 색상별로 동일.

### 기각된 가설 3가지

| 가설 | 기각 근거 |
|---|---|
| 파란 바닥과 파란 블록 혼동 | CIELAB ΔE — 테이블 대비 red 66.3 / green 85.4 / **blue 133.2**. 파란 블록이 **가장 잘 구분**되는데 성적은 최하. 예측이 정반대 |
| 학습 데이터 개수 불균형 | `meta/episodes.jsonl` — 6 태스크 각각 **정확히 200 에피소드** |
| 색상별 배치 prior (red→우, green/blue→좌) | 예측은 "red가 오른쪽에서 유리"이나 실측 red +2.2pp / green −9.4pp / blue +1.1pp로 무관 |

---

## 8. 검증 — 리플레이 실험

### 무엇을 / 어떻게

`scripts/replay_trial.py` — 정책을 호출하지 않고 저장된 action만 MuJoCo에 재주입.

1. `reset_and_place_trial()`로 초기 상태 재현: teleop 키프레임 → 블록 3개 배치 → 액추에이터 고정 → **1.5 s(mj_step 750회) 안정화**
2. 매 action마다 `d.ctrl`에 절대 관절 목표 기입 후 **mj_step 33회**(15 Hz ÷ 0.002 s)
3. 원본과 동일한 성공 판정

### 결과 1: 완전 재현

| trial | 원래 | 리플레이 | 블록 최종 위치 오차 |
|---|---|---|---:|
| SEAM blue_left_g08_r1 | 실패 | 실패 | **0.0000 m** |
| SEAM blue_left_g08_r2 | 실패 | 실패 | **0.0000 m** |
| SEAM blue_left_g08_r3 | 성공 | 성공 | **0.0000 m** |

**시뮬레이션은 action 시퀀스에 대해 결정론적** → 성패는 전적으로 정책 출력이 결정. 컨트롤러·시뮬레이션 환경 등 외부 요인 개입 없음.
→ 원래 계획의 M1/M2 판별에서 **M2(모델 출력 문제)** 로 확정.

### 결과 2: chunk 전체 실행으로 파지 실패 시점 특정

실패 trial r1에서 각 chunk를 실행분(K=8)이 아니라 **전체 50 step** 실행:

| chunk | 시작 step | 블록 최종 y | 해석 |
|---:|---:|---:|---|
| 6 | 48 | +0.2400 | 파지 전, 블록 그대로 |
| **7** | **56 (파지 직후)** | **+0.2366** | **계획을 끝까지 돌려도 블록이 안 움직임** |
| 8 | 64 | +0.1418 | 이 구간에서 벽까지 밀림 |

**파지 순간 블록이 그리퍼에 들어오지 않았음**이 통제된 실험으로 확인.

---

## 9. 사례 연구 — `blue_left_g08` (0.625, +0.235)

초기 블록 위치가 6회 모두 `[0.625, 0.235, 0.84493]`로 동일한데 SEAM은 실패·실패·성공, baseline은 성공·성공·실패.

**전 구간 지표는 "실패가 거칠다"로 보이나 착시**(실패 600 step vs 성공 123 step). **동일 구간(첫 123 step)으로 맞추면:**

| 조건 | trial | 결과 | BJ | 가동폭 | 파지 step | 개폐 |
|---|---|---|---:|---:|---:|---:|
| SEAM | r2 | 실패 | 0.01783 | 3.435 | 54 | 1 |
| SEAM | r3 | **성공** | 0.01781 | 3.445 | 55 | 1 |
| baseline | r3 | 실패 | **0.02414** | 3.286 | 63 | 3 |
| baseline | r1 | 성공 | 0.02581 | 3.315 | 61 | 1 |

SEAM은 소수점 5자리에서야 갈리고, **baseline은 실패가 오히려 더 부드러움.**

**성패를 가른 실제 차이: 파지 순간 관절당 0.003~0.014 rad = 0.2°~0.8°.** 실패끼리의 차이(0.0047)도 실패-성공 차이(0.0117)와 같은 자릿수 → 분기가 계통적이지 않음.

반면 **모션 품질은 성패와 무관하게 일관** — 6회 전부 SEAM BJ ≈ 0.018, baseline ≈ 0.025 (25% 격차 유지).

📄 상세: [`case_study_blue_left_g08_ko.md`](case_study_blue_left_g08_ko.md)

---

## 10. 종합 — 원인과 증상의 구분

```
근본 원인:  평가 격자 x=0.625 가
              (a) 팔 최대 도달거리(0.79 m)의 103.3%  → 완전 신전, 자코비안 특이
              (b) 학습 분포(x ≤ 0.60) 밖              → 외삽
   │
   ├─ 증상 1: 성공률 절벽 (98~100% → 18~20%)
   ├─ 증상 2: 파지 확보 실패 → 밀기 → 빈 손 릴리스
   ├─ 증상 3: |y|=0.206 비단조 sweet spot (거리는 0.9mm 차이뿐)
   ├─ 증상 4: 색상 효과 (분포 안에서는 0, OOD에서만 발현)
   ├─ 증상 5: (0.625, −0.178) 완전 사각지대 0/90
   └─ 증상 6: 0.2~0.8° 노이즈가 성패를 뒤집는 확률적 결과

SEAM과 무관: 위 전부.
SEAM의 효과: 모션 품질 축에서만 일관 (BJ −23.4%, 변위비 1.295→1.006, 비용 +0.4%)
```

### 확립된 결론

1. **태스크 의미 보존** — 78.7% → 80.1%, McNemar p=0.629. 개선도 훼손도 없음.
2. **명령 스트림의 chunk 경계 아티팩트 제거** — 경계/내부 변위비 1.295 → 1.006. SEAM의 가장 명확한 기여.
3. **절대 저크 전면 감소** — commanded BJ −23.4%, IJ −27.9%, AVb −31.7%; measured BJ −18.1%, IJ −15.9%, AVb −25.9%. 전부 p<0.001, 17개 offset 전부에서 SEAM 우세.
4. **비용 없음** — 추론 지연 +0.4%, 재학습 불필요.
5. **부수 효과** — 같은 경로를 6.1% 빨리 주파.
6. **구조적 상한** — overlap residual이 클 때는 개입 여력 없음. 장면 기하 참조 prior가 필요.

### 한계

- SEAM의 개선은 주로 **명령 레벨**. 물리 레벨에서는 액추에이터가 이미 5.7배 흡수하고 있어 실기 검증 필요.
- 성공률 차이 없음은 양날 — "대가 없음"의 근거이자 "성공률 개선" 주장 불가.
- false_scene은 **SEAM 단일 조건** — 그 구역에서 SEAM 효과는 여전히 미확인.
- 성공/실패 간 모션 차이는 에피소드 길이(600 vs 120 step) 교란으로 인과 해석 불가.

---

## 11. 이전 보고서 정정 사항

분석이 진행되며 초기 해석 4건을 정정했습니다.

| 위치 | 초기 서술 | 정정 |
|---|---|---|
| `grid_evaluation_report_ko.md` §2.1 | x=0.625 저조를 **"도달 거리 한계"** 로 해석 | 도달·파지 모두 됨(파지율 93~100%). 실제로는 **완전 신전 + OOD** 로 인한 파지 확보 실패 |
| `false_scene_report_ko.md` §3.3 | **"lift-over-rim 단계 실패"** | 팔은 8~9 cm 실제로 들어올림. 블록이 애초에 잡히지 않은 것 |
| `false_scene_report_ko.md` §2.1 | **"오른팔 쪽 정책/기구학 문제"** | −0.178 한 칸 제외 시 좌 96.3%/51.2% vs 우 95.0%/49.7%로 동일. **단일 자세의 문제** |
| `grid_evaluation_report_ko.md` §4.3 | BJ/IJ 비율 악화를 **"IJ가 더 크게 줄어서"** | % 기준만 맞음. **절대 감소량은 BJ −0.00656 > IJ −0.00355로 BJ가 1.85배 더 감소.** 비율은 오해를 유발하는 지표 |

---

## 12. 남은 작업 (우선순위)

| 순위 | 항목 | 근거 |
|---|---|---|
| 1 | **평가 격자를 x ≤ 0.60으로 재설계** | §6. 현 실패의 유일한 근본 원인 |
| 2 | SEAM 평가를 **non-inferiority + 모션 품질** 설계로 재구성 | 성공률 축이 SEAM과 무관함이 확인됨 |
| 3 | OOD/완전신전 일반화를 별도 주제로 분리 | x=0.625는 "실패 구간"이 아니라 "의도된 극한 조건" |
| 4 | 세 위치의 IK 해·자코비안 조건수 비교 | sweet spot이 자세 차이인지 확정 |
| 5 | (0.625, −0.178) IK 해 존재 확인 | 0/90의 유일한 사각지대 |
| 6 | 실기 RB-Y1 A/B | 명령/실측 5.7배 격차 → 시뮬레이터 액추에이터 모델 의존성 해소 |
| 7 | false_scene baseline 540 trials | 해당 구역 SEAM 효과 미확인 (명령: [`false_scene_report_ko.md`](false_scene_report_ko.md) §6) |

---

## 부록 — 재현 명령

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws

# 비교 지표 (NPZ 재계산 + 통계 검정)
src/openpi/.venv/bin/python scripts/compare_grid_conditions.py \
  --out docs/assets/comparison_summary.md

# 핵심 요약 그림
src/openpi/.venv/bin/python scripts/plot_condition_summary.py

# 경계 정렬 프로파일 + 지표 막대 + 수치표
src/openpi/.venv/bin/python scripts/plot_jerk_comparison.py

# 히트맵 (병합 / 색상별 / 다른 지표)
cat data/rby1_grid_eval_baseline/results.jsonl data/rby1_grid_eval_seam/results.jsonl > /tmp/all.jsonl
src/openpi/.venv/bin/python scripts/plot_grid_heatmap.py --results /tmp/all.jsonl --merge-colors \
  --out docs/assets/cmp_success_merged.png

# 리플레이 (결정론 검증 / 계획 전체 실행)
MUJOCO_GL=osmesa src/openpi/.venv/bin/python scripts/replay_trial.py \
  --results data/rby1_grid_eval_seam/results.jsonl --trial-id blue_left_g08_r1
MUJOCO_GL=osmesa src/openpi/.venv/bin/python scripts/replay_trial.py \
  --results data/rby1_grid_eval_seam/results.jsonl --trial-id blue_left_g08_r1 \
  --mode chunk --chunk-index 7
```
