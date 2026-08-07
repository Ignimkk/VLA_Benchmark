# 실험 결과 분석 템플릿 (섹션 4)

> `docs/research_plan_week_ko.md`의 RQ-a를 검증하기 위한 실험 결과를 채우는 문서.
> 실행 전 골격만 먼저 작성하고, `plot_success_heatmap.py` / `failure_trajectory_analysis.py` /
> `metrics/motion.py::find_jerk_peak_regressions` 실행 후 수치·그림을 채운다.
>
> 실행 로그: (실행 시 커맨드와 wall-clock 시간을 여기 기록)

---

## 4a. 위치별 성공률 히트맵

- 데이터 소스: `evaluate_baseline.py --save-per-episode`, `evaluate_seam.py --save-per-episode`가
  생성하는 `*.per_episode.jsonl`
- 스크립트: `experiments/libero/plot_success_heatmap.py`
- 3색 스키마: 성공(녹색) / 실패-충돌없음(황색) / 실패-충돌있음(적색)
  - 주의: 이번 주는 baseline과 SEAM 2개 조건만 실행하므로, 색상은 조건 비교가 아니라
    **실패 유형 구분**을 나타낸다. 3-way(collision-avoidance 포함) 비교는 RQ-b 단계에서 진행.

**[결과 삽입 위치]**

- baseline heatmap: `TODO`
- SEAM heatmap: `TODO`
- 관찰: `TODO`

---

## 4b/4c. 실패 궤적 판별 (M1/M2)

- 스크립트: `experiments/libero/failure_trajectory_analysis.py`
- **M1**: `classify_failure()`가 실패 에피소드를 "external"(EE가 목표로 계속 수렴 중이었으나
  타임아웃) 또는 "trajectory_failure"(궤적 자체가 목표에서 벗어남)로 분류. External로 분류된
  비율이 높다면, 모델보다 컨트롤러/환경 쪽 원인일 가능성을 시사한다(§4d와 연결).
- **M2**: `compare_failure_vs_success_timeseries()`로 동일 task의 성공/실패 에피소드 action·jerk
  시계열을 겹쳐 그려 실패 시점 근처에서의 차이를 관찰.

**[결과 삽입 위치]**

- M1 분류 비율 (external / trajectory_failure): `TODO`
- M2 대표 사례 그림: `TODO`
- 관찰: `TODO`

---

## 4d. 외부 요인 분석 (컨트롤러 / 시뮬레이션 환경) — 가설 문서화만, 재실험 없음

### 컨트롤러 (RBY1 `pi05_infer.py`)

- `pi05_infer.py`는 예측된 joint delta를 `d.qpos` 인덱스 기반으로 `d.ctrl[...]`에 직접
  기입한다. Position actuator이므로, 큰 delta가 명령되면 실제 관절이 그 목표를 즉시 따라가지
  못하고 tracking error가 발생할 수 있다 — 이 tracking error 자체가 원인이 되어 jerk/불안정한
  접촉이 발생할 가능성이 있다.
- 검증 방법(재실험 없이): 새로 로깅되는 `collision_trace`와 기존 `measured_qpos`(명령된 delta
  적용 후 실측값)를 비교해, collision이 발생한 스텝 근처에서 명령-실측 tracking error가 크게
  벌어지는지 확인. **[결과 삽입 위치: TODO]**

### 시뮬레이션 환경 — 바닥 재질(파란색) 가설

- `src/rby1_description/models/{rby1a,rby1m,rby1t5}/mujoco/rby1.xml`의 floor 텍스처가
  slate-blue 계열(체커 텍스처 `rgb1=".2 .3 .4" rgb2=".1 .15 .2"`, 두 색 모두 B 채널이 가장
  높음)로 확인됨. 반면 LIBERO arena는 기본값 `floor_style="light-gray"`로 파란색과 무관하다
  (`src/openpi/third_party/libero/.../style.py`).
- **가설**: RBY1 환경에서 바닥과 대비가 낮은 색상의 오브젝트(또는 그 반대)가 있을 경우
  perception(segmentation/depth cue)에 영향을 줄 수 있다. 이번 주는 이 가설을 문서화하는
  수준까지만 진행하며, 바닥 재질을 교체한 재실험은 진행하지 않는다(범위 결정 사항).
- **주의**: 이 결과는 RBY1에만 해당한다. 이번 주 §4a/4b/4c의 실제 실행 데이터는 LIBERO 기준이므로
  이 가설은 LIBERO 결과와 직접 연결되지 않는다 — RBY1 쪽 실패/충돌 데이터가 확보되면 별도로
  검증한다.

---

## 4e. Jerk peak regression 구간 분석

- 데이터 소스: RBY1 기존 녹화 데이터(114개 chunk, `data/policy_records/`) + `collision_trace`가
  추가된 신규 RBY1 녹화 데이터(있다면)
- 함수: `metrics/motion.py::find_jerk_peak_regressions(baseline_jerk, seam_jerk, window)`
- 시각화: `scripts/plot_rby1_jerk_comparison.py --flag-regressions`
- 후보 원인(이미 문서화됨, `docs/reproduction_notes.md`): 배포된 checkpoint가 논문 대비 훨씬 좁은
  guided window를 사용 — LIBERO는 H=10/K=5(L=5) vs 논문 H=50/K=10(L=40); RBY1은 H=50/K=8(L=42,
  M=20 사용 가능하나 실제 서버 통합 미확인). Overlap이 짧을수록 SEAM의 보정이 걸릴 수 있는
  구간이 짧아, 특정 구간에서는 보정이 오히려 새로운 불연속을 만들 수 있다는 가설.

**[결과 삽입 위치]**

- SEAM peak > baseline peak인 구간 목록: `TODO`
- 해당 구간과 chunk-boundary/guided-window 파라미터의 상관관계: `TODO`
- 결론: `TODO`
