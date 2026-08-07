# RB-Y1 그리드 평가: Baseline vs SEAM 최종 결과

> 실험일: 2026-08-03 · 그리드 리비전 `006698dac3ef`
> 데이터: `data/rby1_grid_eval_baseline/`, `data/rby1_grid_eval_seam/` (각 216 trials)
> 재현: `scripts/compare_grid_conditions.py`, `scripts/plot_grid_heatmap.py`, `scripts/plot_condition_summary.py`

---

## 1. 실험 설계

| 항목 | 값 |
|---|---|
| 로봇 / 시뮬레이터 | RB-Y1 dual-arm, MuJoCo |
| 정책 | π0.5 + LoRA (`pi05_rby1_lora`, 30k steps) |
| 태스크 | "put the {color} block in the brown box with your {side} hand" |
| 그리드 | 좌/우 각 12지점 (x ∈ {0.475, 0.525, 0.575, 0.625} × \|y\| ∈ {0.178, 0.235, 0.292}) |
| 조건 | 3색 × 2손 × 12위치 × 3반복 = **216 trials/조건** |
| chunk 실행 | H=50, K=8 (K-of-H), 15 Hz |
| 상한 | 600 step (40 s) |
| 성공 판정 | 블록이 컨테이너 내부(±0.075 m)에서 8 step 연속 정지 |

**Paired 설계**: 두 조건이 동일 그리드·동일 반복 인덱스를 공유하며, `reset_and_place_trial()`이 결정론적으로 초기 상태를 재현합니다. 216쌍이 완전 매칭되어(`trial_id` 불일치 0) paired 검정이 가능합니다.

SEAM on/off는 서버 측 `seam_enabled` (YAML)로 전환했고, 클라이언트는 매 trial 시작 시 `seam_reset`을 보내 에피소드 간 상태 누수를 차단했습니다.

---

## 2. 결과 1 — 태스크 성공률: 차이 없음

| 조건 | 성공 | 성공률 | 95% CI (Wilson) |
|---|---:|---:|---|
| Baseline | 170/216 | 78.7% | [72.8%, 83.6%] |
| SEAM | 173/216 | 80.1% | [74.3%, 84.9%] |

Paired 분할표:

| | SEAM 성공 | SEAM 실패 |
|---|---:|---:|
| **Baseline 성공** | 163 | 7 |
| **Baseline 실패** | 10 | 36 |

불일치 17쌍에 대한 **McNemar 정확검정 p = 0.629** → 성공률 차이는 통계적으로 유의하지 않습니다.

**해석**: SEAM은 태스크 성공률을 **훼손하지도 개선하지도 않습니다.** 이는 "task semantics 보존"이라는 설계 목표가 충족되었다는 뜻이며, smoothing 기법이 정책의 의도를 왜곡하지 않았음을 보여줍니다.

### 2.1 공간 분포 — 실패는 도달 한계에 집중

![success heatmap](../../../docs/assets/cmp_success_merged.png)

| 구역 | n | Baseline | SEAM | 차이 |
|---|---:|---:|---:|---:|
| 근거리 (x < 0.625) | 162 | 160 (98.8%) | 162 (**100.0%**) | +1.2 pp |
| 원거리 (x = 0.625) | 54 | 10 (18.5%) | 11 (20.4%) | +1.9 pp |

전체 78~80%라는 수치는 **x=0.625 한 열이 만든 것**입니다. 앞 3열은 두 조건 모두 사실상 완전 성공이며, SEAM은 근거리 18칸 전부 100%를 달성한 반면 baseline은 2칸(x=0.575/y=+0.178, x=0.525/y=+0.235)에서 89%로 새어나갔습니다.

x=0.625 열 세부 (셀당 n=9):

| y | +0.292 | +0.235 | +0.178 ‖ −0.178 | −0.235 | −0.292 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0% | 67% | 33% ‖ 0% | 11% | 0% |
| SEAM | 0% | 22% | **78%** ‖ 0% | 22% | 0% |

두 조건이 서로 다른 칸에서 강합니다(baseline은 +0.235, SEAM은 +0.178). n=9에서는 이 정도 변동이 우연 범위이므로, **이 구간은 현재 표본으로 결론을 내릴 수 없습니다** — §6의 추가 실험이 필요한 이유입니다.

---

## 3. 결과 2 — 모션 품질: 전 지표 유의하게 개선

![summary](../../../docs/assets/summary_baseline_vs_seam.png)

모든 값은 저장된 요약이 아니라 **trajectory NPZ에서 재계산**했으며, arm 12관절만 사용합니다(그리퍼는 near-binary라 저크 노름을 지배). 216쌍 전부 사용, Wilcoxon signed-rank.

### 3.1 Commanded (정책이 내보낸 관절 목표)

| 지표 | Baseline | SEAM | 변화 | p |
|---|---:|---:|---:|---:|
| BJ (경계 저크) | 0.02802 ± 0.00407 | 0.02147 ± 0.00366 | **−23.4%** | <0.001 |
| IJ (내부 저크) | 0.01274 ± 0.00108 | 0.00919 ± 0.00088 | **−27.9%** | <0.001 |
| CD (경계 불연속) | 0.02289 ± 0.00396 | 0.01810 ± 0.00290 | **−20.9%** | <0.001 |
| AVb (경계 저크 분산) | 0.00020 ± 0.00008 | 0.00014 ± 0.00006 | **−31.7%** | <0.001 |
| overlap residual | 0.07549 | 0.06355 | **−15.8%** | <0.001 |

### 3.2 Measured (실제 물리 응답, qpos)

| 지표 | Baseline | SEAM | 변화 | p |
|---|---:|---:|---:|---:|
| BJ | 0.00362 ± 0.00062 | 0.00296 ± 0.00048 | **−18.1%** | <0.001 |
| IJ | 0.00318 ± 0.00041 | 0.00267 ± 0.00038 | **−15.9%** | <0.001 |
| CD | 0.01381 ± 0.00332 | 0.01551 ± 0.00273 | **+12.4%** | <0.001 |
| AVb | — | — | **−25.9%** | <0.001 |

**저크(2차 차분)는 명령·물리 양쪽에서 모두 감소했으나, 물리 공간의 CD(1차 차분)만 12.4% 증가했습니다.** 이 역전이 이번 실험의 핵심 논점입니다.

---

## 4. 심층 분석 — measured CD 역전의 정체

CD는 경계에서의 `‖a_t − a_(t−1)‖`, 즉 **1차 차분(속도)** 이고 BJ/IJ는 2차 차분(저크)입니다. 물리 공간에서 속도가 커지면 CD는 저크와 무관하게 상승합니다. 성공 trial 163개로 검증했습니다:

| 항목 | Baseline | SEAM | 변화 | p |
|---|---:|---:|---:|---:|
| 평균 스텝 변위 (속도) | 0.0163 | 0.0172 | **+5.1%** | 6e-24 |
| 에피소드 길이 (스텝) | 189.2 | 177.7 | **−6.1%** | 1e-19 |
| 총 이동 거리 | 1.5255 | 1.5239 | −0.1% | 0.012 |

**SEAM은 같은 경로를 6.1% 더 짧은 시간에, 5.1% 더 빠른 속도로 주파합니다.** 총 이동 거리가 사실상 동일하므로 경로가 바뀐 것이 아니라 **더 빠르게 실행**된 것입니다. measured CD의 +12.4%는 이 속도 상승이 만든 산술적 결과이며, 움직임이 거칠어진 것이 아닙니다.

속도에 불변인 지표(경계 변위 ÷ 내부 변위)로 다시 보면:

| 공간 | Baseline | SEAM | 변화 | 의미 |
|---|---:|---:|---:|---|
| Commanded | 1.2953 | **1.0058** | −22.4% | 경계 step이 내부보다 29.5% 컸던 것이 **거의 완전히 사라짐** |
| Measured | 0.9102 | 0.9394 | +3.2% | 양쪽 모두 **1 미만** — 물리적으로 경계는 원래 이상점이 아님 |

**가장 강한 결과**: baseline의 commanded 경계 변위비 1.295는 chunk 경계에서 명령이 튀는 아티팩트 그 자체이고, SEAM은 이를 **1.006** — 통계적으로 내부 step과 구별되지 않는 수준까지 낮췄습니다.

Measured 비율이 양 조건 모두 1 미만인 것도 중요합니다. 위치 액추에이터와 관성이 이미 명령을 저역통과시켜, 물리 레벨에서는 경계가 애초에 이상점이 아니었습니다. 즉 SEAM이 해결한 문제는 주로 **명령 스트림**에 존재합니다.

### 4.1 Chunk 경계 정렬 프로파일 — 아티팩트의 직접 관찰

![boundary profile](../../../docs/assets/jerk_boundary_profile.png)

216 trial의 **모든 chunk 경계(각 trial 약 23개)를 step 0에 정렬해 평균**한 그림입니다. Baseline과 SEAM의 에피소드 길이가 다르므로(§4) 단일 trial 겹쳐 그리기로는 만들 수 없고, 이 정렬 평균이 유일하게 공정한 비교입니다.

네 패널이 각각 다른 이야기를 합니다.

- **좌상 (commanded jerk)**: K=8 주기로 뚜렷한 스파이크. Baseline 피크 0.0361 → SEAM 0.0278 (**−23.1%**). 내부 바닥도 0.0087 → 0.0056으로 함께 내려갑니다.
- **좌하 (commanded displacement)**: **가장 결정적인 패널.** Baseline은 step 0에서 0.0245로 솟구치는 반면 SEAM은 0.0192로 평탄합니다. §4 표의 경계/내부 변위비 1.295 → 1.006이 눈으로 확인되는 지점입니다.
- **우상 (measured jerk)**: 물리 응답에서도 SEAM이 전 구간에서 아래. 피크 0.00635 → 0.00515 (−19.0%).
- **우하 (measured displacement)**: SEAM 곡선이 **전 구간에서 위**에 있고, 두 조건 모두 step 0에 스파이크가 없습니다. measured CD 증가가 경계 아티팩트가 아니라 **전반적 속도 상승**임을 보여줍니다.

**방법론적 관찰**: 저크 피크는 실제로 offset **−1**에 있으나, `motion.py`의 BJ는 `t % K == 0`, 즉 offset 0에서 샘플링합니다(SEAM 논문 Eq. 10 정의를 그대로 따름). 피크 지점에서 재보면 −23.1%(commanded) / −19.0%(measured)로, offset 0 기준 −23.4% / −18.3%와 사실상 같습니다. 즉 **어느 지점에서 재도 결론은 동일**하며, BJ 정의는 개선폭을 과대평가하지 않습니다.

### 4.2 지표 요약 그래프

![jerk metrics](../../../docs/assets/jerk_metrics_bars.png)

수치 표: [`docs/assets/jerk_table.md`](../../../docs/assets/jerk_table.md) (mean ± sd, Wilcoxon p)

### 4.3 BJ/IJ 비율은 오히려 악화

| 공간 | Baseline | SEAM | 변화 |
|---|---:|---:|---:|
| Commanded | 2.1951 | 2.3247 | +5.9% (p<0.001) |
| Measured | 1.1393 | 1.1151 | −2.1% (p<0.001) |

Commanded BJ/IJ가 나빠진 것은 **내부 저크(−27.9%)가 경계 저크(−23.4%)보다 더 크게 줄었기 때문**이지, 경계가 나빠져서가 아닙니다. 절대값은 둘 다 감소했습니다. 이 현상은 오프라인 예비 실험(`professor_review_ko.md` §9.2, BJ/IJ 1.75→1.91)에서도 동일하게 관찰되었으며, VLS의 보정이 경계 근처 M step에 국한되지 않고 chunk 전체 형상에 영향을 준다는 해석과 일치합니다.

---

## 5. 결과 3 — 계산 비용

| 조건 | 평균 지연 | 지연 중앙값(최대) | chunk/trial |
|---|---:|---:|---:|
| Baseline | 218.6 ms | 341 ms | 34.8 |
| SEAM | 219.5 ms | 339 ms | 32.8 |

**추론 오버헤드 +0.4% (사실상 0).** VLS의 보정은 closed-form `−2(x−r)`의 element-wise 연산이라 backward pass가 없고, 기존 ODE 루프 안에서 흡수됩니다. 보고된 최대치(6.2 s / 12.9 s)는 첫 호출 및 서버 재시작 직후의 JIT 컴파일 이상치이며 중앙값은 두 조건이 동일합니다.

SEAM의 chunk/trial이 적은 것(32.8 vs 34.8)은 에피소드가 더 빨리 끝났기 때문입니다(§4).

---

## 6. 종합 평가

### 6.1 확립된 결론

1. **태스크 의미 보존** — 성공률 78.7% → 80.1%, McNemar p=0.629. 개선도 훼손도 없음.
2. **명령 스트림의 chunk 경계 아티팩트 제거** — 경계/내부 변위비 1.295 → 1.006. 이것이 SEAM의 가장 명확한 기여.
3. **절대 저크 전면 감소** — commanded BJ −23.4%, IJ −27.9%, AVb −31.7%; measured BJ −18.1%, IJ −15.9%, AVb −25.9%. 전부 p<0.001.
4. **비용 없음** — 추론 지연 +0.4%, 재학습 불필요.
5. **부수 효과: 실행 시간 6.1% 단축** — 같은 경로를 더 빠르게. 의도한 효과는 아니나 재현성 있게 관측됨(p=1e-19).

### 6.2 한계와 반론 가능 지점

- **measured CD +12.4%** 는 액면 그대로는 SEAM에 불리합니다. §4에서 속도 효과로 설명했으나, "속도가 올라간 것 자체가 안전성 관점에서 바람직한가"는 별개 문제입니다. 안전성을 속도 상한과 함께 논한다면 이 지점은 방어가 필요합니다.
- **물리 공간에서 경계는 원래 이상점이 아니었습니다**(변위비 < 1). 즉 SEAM이 고친 문제의 상당 부분은 명령 레벨에 있고, 실제 하드웨어에서 체감되는 개선은 시뮬레이터의 액추에이터 모델에 의존할 수 있습니다. 실기 검증이 남아 있습니다.
- **성공률 차이 없음**은 양날입니다. 안전성 개선의 대가가 없다는 근거인 동시에, "성공률을 높인다"는 주장은 불가능합니다.
- **원거리 구역(x=0.625)은 결론 불가** — 셀당 n=9에서 baseline과 SEAM이 서로 다른 칸에서 우세하며, 이는 우연 범위입니다.
- **BJ/IJ 비율 악화**(commanded +5.9%)는 논문에 그대로 적시하는 편이 안전합니다. 절대값 감소와 함께 제시하면 오해 소지가 없습니다.

### 6.3 한 문장 결론

> π0.5의 재학습 없이 denoising 단계에 closed-form guidance를 추가하는 것만으로, RB-Y1 216쌍 paired 실험에서 **태스크 성공률을 유지한 채(p=0.629)** 명령 스트림의 chunk 경계 아티팩트를 사실상 제거하고(경계/내부 변위비 1.295→1.006) 절대 저크를 16~32% 감소시켰으며, 추론 비용 증가는 0.4%에 불과했다.

---

## 7. 남은 작업

- 원거리 실패 구역(x=0.625)에 대한 고표본 추가 실험 → `block_false_grid.json`, §8
- 실기(real RB-Y1) A/B — 시뮬레이터 액추에이터 모델 의존성 해소
- x=0.625에서 좌우 비대칭(왼팔 33.3% vs 오른팔 7.4%)의 원인 규명 — IK 해 존재 여부 / 관절 한계 / 복귀 경로
- 실패 궤적의 M1/M2 분류(모델 문제 vs 외부 요인)

---

## 8. 추가 실험 실행 절차

`src/rby1_manipulation/block_false_grid.json` — x=0.625의 6개 지점(±0.178, ±0.206, ±0.235). ±0.206은 기존 그리드에 없던 보간 지점입니다.

**칸당 30회** 기준 반복 수는 색상 사용 방식에 따라 달라집니다:

| 목표 | 명령 | trials | 소요(추정) |
|---|---|---:|---|
| 위치당 30회 (권장) | `--grid-colors red --grid-repeats 30` | 180 | 2~3 h |
| (색상×위치)당 30회 | `--grid-repeats 30` | 540 | 6~9 h |

이 구역은 실패가 잦아 대부분 600 step 상한(40 s)까지 도달하므로 trial당 40~60초로 추정했습니다.

SEAM 조건 (서버는 `--seam-config seam_rby1.yaml`로 기동되어 있어야 함):

```bash
cd /home/mk/dev_ws/vla/pi0_TO_ws
tmux new -A -s gridfalse

src/openpi/.venv/bin/python src/rby1_bringup/pi05_ex_infer.py \
  --model rby1 --remote localhost:8123 --seam \
  --grid-experiment --headless --grid-record-all \
  --grid-config src/rby1_manipulation/block_false_grid.json \
  --grid-colors red --grid-repeats 30 \
  --grid-output-dir data/rby1_grid_eval_false_scene
```

나중에 baseline도 받으려면 서버를 `--seam-config baseline_rby1.yaml`로 재시작한 뒤 `--seam`을 빼고 `--grid-output-dir data/rby1_grid_eval_false_scene_baseline`으로 실행합니다.

분석:

```bash
src/openpi/.venv/bin/python scripts/plot_grid_heatmap.py \
  --results data/rby1_grid_eval_false_scene/results.jsonl --merge-colors \
  --out docs/assets/false_scene_success.png

# baseline까지 확보한 뒤
cat data/rby1_grid_eval_false_scene_baseline/results.jsonl \
    data/rby1_grid_eval_false_scene/results.jsonl > /tmp/false_scene_all.jsonl
src/openpi/.venv/bin/python scripts/plot_grid_heatmap.py \
  --results /tmp/false_scene_all.jsonl --merge-colors \
  --out docs/assets/false_scene_compare.png
src/openpi/.venv/bin/python scripts/compare_grid_conditions.py \
  --baseline data/rby1_grid_eval_false_scene_baseline \
  --seam     data/rby1_grid_eval_false_scene \
  --far-x 0.625 --out docs/assets/false_scene_summary.md
```

**주의사항**

- 이 그리드의 `grid_fingerprint`는 기존(`006698dac3ef`)과 다르므로 resume이 두 실험을 섞지 않습니다. 다만 히트맵 스크립트는 한 파일에 여러 리비전이 섞이면 거부하므로 **출력 폴더를 분리**하세요.
- 코드 수정은 **불필요**합니다. `--grid-config`, `--grid-repeats`, `--grid-colors` 모두 기존 CLI 인자이며 위치 개수는 하드코딩되어 있지 않습니다.
- SEAM 조건에서 `--seam`은 필수입니다. 서버 프로세스가 전체 trial 동안 살아있으므로, 이 플래그가 없으면 이전 trial의 마지막 chunk가 다음 trial의 prior로 누수됩니다.
- 검정력: 위치당 n=30이면 20% vs 45% 수준의 차이를 α=0.05에서 검출할 수 있습니다. 기존 n=9로는 불가능했던 해상도입니다.
