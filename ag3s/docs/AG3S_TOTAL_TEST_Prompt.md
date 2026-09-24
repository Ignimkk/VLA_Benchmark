# AG3S–cuRobo 전체 파이프라인 통합 테스트 (T0~T6)

## 목적

구현된 AG3S–cuRobo 파이프라인을 T0부터 T6까지 **처음부터 다시 실행**하여 검증한다. 각 단계에서 실패가 발생하면 기록하고, 원인을 분석하고, 코드를 직접 수정한 뒤 재시험한다. 최종 목표는 T6 실제 통합 closed loop까지 통과하는 동작하는 파이프라인이다. T0~T3 재실험만 하고 작업을 종료하지 않는다.

---

## 1. 공통 원칙

1. **기존 결과는 합격 근거가 아니다.** 기존 seed 17/23, run_0004, 기존 T0~T3 표는 모든 프레임과 중간 단계가 기록되지 않았으므로 재검증 대상이다. 삭제하지 말고 다음과 같이 표시한 뒤 회귀 비교용으로만 사용한다.
   > 과거 참고 결과 — 현재의 전 프레임 검증 기준을 충족하지 않으므로 합격 근거로 사용할 수 없음
2. **새 세션, 새 seed로 실행한다.** 기존 seed와 저장된 결과를 시험 입력으로 재사용하지 않는다.
3. **전 프레임 기준.** 대표 프레임, 성공 프레임, 일정 간격 샘플링만으로 합격시키지 않는다. 실패 프레임과 상태 전이 경계 프레임을 포함한 전체 프레임이 결과에 포함되어야 한다. 단일 프레임 성공, 단일 eikonal 수치, adapter round-trip 성공만으로는 어떤 단계도 통과시키지 않는다.
4. **값을 만들어내지 않는다.** 갱신되지 않은 프레임의 값을 새로 계산한 것처럼 기록하지 않는다. 모든 field와 상태는 `new / carried / stale / unavailable` 중 하나로 표시하고, carried·stale이면 마지막 실제 갱신 frame ID, timestamp, age를 함께 기록한다.
5. **합격 기준을 낮춰서 통과시키지 않는다.** 임계값·허용오차·판정 로직을 변경해야 한다면 변경 전후 값과 근거를 실패 기록에 남기고, 변경 이전 결과와 함께 보고한다.

---

## 2. 실행 순서와 게이트

```
T0 환경·배선 → T1 연속 프레임 AG3S → T2 pick-place 상태 전이
→ T3 전 프레임 TSDF/ESDF → T4 fail-closed 및 AG3S–cuRobo 결합
→ T5 shadow closed loop → T6 실제 통합 closed loop
```

- 각 단계는 이전 단계가 통과해야 시작한다.
- T4의 필수 fail-closed 조건을 통과하면 같은 작업 안에서 T5, T6까지 계속 진행한다.
- T4~T6의 세부 합격 기준은 `AG3S_CUROBO_LIVE_TEST_PLAN.md`에 정의된 것을 따른다. 문서에 기준이 없거나 모호하면 기준을 임의로 만들지 말고, 초안을 문서에 추가한 뒤 해당 사실을 최종 보고에 명시한다.

---

## 3. 실패 처리 루프

각 단계에서 실패가 발생하면 다음 순서를 따른다.

1. **기록** — 실패 ID, 단계, run/seed, 발생 frame ID, 증상, 증거(로그·그림·수치 경로)를 기록한다.
2. **분석** — 근본 원인을 규명하고 원인 위치를 다음 중 하나로 분류한다: 구현 결함 / 테스트 하네스 결함 / 설정·환경 문제 / 기준 자체의 문제.
3. **수정** — 원인을 직접 수정한다. 증상만 가리는 수정(예외 무시, 조건 우회, 값 clamp로 은폐)은 하지 않는다.
4. **재시험** — 수정이 영향을 주는 **가장 앞 단계부터** 다시 실행한다. 예: T3에서 수정한 코드가 AG3S lifting에도 쓰인다면 T1부터 다시 실행한다. 재시험도 새 run ID로 기록한다.
5. **중단 조건** — 같은 원인의 실패가 3회 수정 후에도 해결되지 않거나, 해결에 설계 결정(인터페이스 변경, 알고리즘 교체, 합격 기준 변경)이 필요하면 임의로 진행하지 말고 현재까지의 분석과 선택지를 정리하여 보고한다.

실패 기록 표 형식:

| ID | 단계 | run/seed | frame | 증상 | 증거 | 원인 분류 | 근본 원인 | 수정 내용(commit) | 재시험 run | 결과 |
|---|---|---|---|---|---|---|---|---|---|---|

---

## 4. 단계별 요구사항

### T0. 전 프레임 환경·배선 검증

시작 시 import 확인으로 끝내지 않고, 실행 시작부터 종료까지 계속 기록한다.

**manifest (불변 설정, 1회 기록 후 각 프레임이 참조)**
- Python, NumPy, PyTorch, CUDA, Warp, cuRobo 버전과 commit
- GPU, 모델 경로, 설정 파일, 좌표계 정의
- 선택된 AG3S backend와 ESDF backend
- 렌더 주기 등 사전에 정한 기록 주기

**프레임별 기록 (실행 중 변하는 값)**
- 각 observation / planning / control frame에 실제 사용된 backend provenance
- legacy `EsdfBuilder` 호출 여부
- cuRobo 요청·응답 횟수와 field sequence
- 각 field의 생성 시각, 적용 시각, age, stale 여부
- 카메라 timestamp와 robot-state timestamp
- TSDF, tracker, target latch, SQP warm start의 reset 시점
- IPC timeout, 누락, 중복, 순서 역전, 예외

**즉시 실패**: legacy backend 호출, 출처가 불분명한 field, timestamp 역전, stale field를 정상으로 처리한 경우.

### T1. 연속 프레임 AG3S 기능 검증

새로 생성한 MuJoCo 씬에서 로봇과 물체가 실제로 움직이는 짧은 연속 구간을 **최소 3개의 새 seed**로 실행한다.

모든 observation frame에서 저장·평가할 항목:
- 모든 카메라의 RGB, depth, mask, intrinsic/extrinsic, timestamp
- 실제 policy attention의 원본과 정규화 결과, 통계, 프레임 간 변화량
- depth 역투영 점군과 base-frame 변환 결과, 카메라별·병합 점군
- robot self-filter 전후 점군
- attention lifting 결과와 point별 provenance
- target 후보, runner-up, confidence, grounding 결과, target 중심·표면 오차
- 지지면, 장애물, 미관측 영역, 정적 기하
- AG3S status, geometry certification, degraded 원인
- 이전 프레임에서 전달된 상태와 이번 프레임에서 새로 계산된 상태의 구분

모든 프레임에 대해 실제 씬, third-person view, 카메라 영상, attention, lifting, grounding, 점군을 한 번에 볼 수 있는 **diagnostic frame card**를 생성한다. 프레임별 오류와 상태 변화가 설명되어야 통과한다.

### T2. pick-place 전 구간 AG3S 상태 전이 검증

접근 → 파지 → 운반 → 놓기 → 초기화를 포함하는 연속 동작을 새 실행으로 생성한다.

모든 관련 프레임에서 기록·시각화할 항목: attention 대상과 공간 분포, target 및 후보 ID, latch 상태, attach/detach 상태, 파지 물체 query point, 허용 접촉 링크, 목적지 전환, 누적 기하와 잔상, tracking confidence, degraded/no-target/hold 상태, reset 이후 상태 제거 여부.

다음 이벤트의 **정확한 발생 프레임**을 표시한다.

```
target acquired → target latched → grasp contact → attached
→ destination attention → transport → placement → detached → task state reset
```

전체 프레임 시계열로 다음을 검증한다.
- attention이 목적지로 이동해도 조작 대상 ID가 바뀌지 않는다.
- attach 후 물체가 robot collision geometry에 포함된다.
- detach 후 이전 상태가 제거된다.

### T3. 전 관측 프레임 cuRobo TSDF/ESDF 검증

T1과 T2에서 생성되는 **모든** observation frame을 같은 라이브 세션에서 cuRobo로 전달한다.

각 update frame에서 기록할 항목:
- 입력 카메라와 depth frame ID, robot/object masking 결과
- TSDF 적분 전후 변화: 새로 관측된 voxel 수, 누적·감쇠·삭제된 voxel 수
- coarse ESDF, fine ESDF와 ROI, surface voxel 또는 mesh
- 각 query의 선택 계층, signed distance, gradient와 gradient norm
- field coverage와 미관측 영역
- field sequence, timestamp, age
- reset 및 reintegration 결과
- MuJoCo 참값 대비 거리 오차

갱신이 없는 control frame은 원칙 4에 따라 상태를 표시한다.

시각화:
- 모든 update frame에 TSDF/ESDF slice 또는 3D 시각화와 이전 field 대비 변화량
- 거리, 기울기, coverage, 오차, 처리시간의 전체 프레임 시계열 그래프

### T4. fail-closed 및 AG3S–cuRobo 결합 검증

`AG3S_CUROBO_LIVE_TEST_PLAN.md`의 T4 정의를 따른다. T0에서 정의한 즉시 실패 조건(stale·출처 불명 field, timestamp 역전 등)이 실제 결합 경로에서 계획·제어를 차단하는지 반드시 포함하여 확인한다.

### T5. shadow closed loop / T6. 실제 통합 closed loop

`AG3S_CUROBO_LIVE_TEST_PLAN.md`의 T5, T6 정의를 따른다. T0~T3의 프레임별 기록과 completeness 기준을 해당되는 항목에 동일하게 적용한다.

---

## 5. Completeness 표 (각 run마다 작성)

| 항목 | 요구사항 |
|---|---|
| expected observation frames | 실행에서 기대된 전체 수 |
| captured observation frames | expected와 동일 |
| attention / lifting / grounding records | 모든 observation frame |
| TSDF / ESDF updates | 예정된 모든 update frame |
| planning records | 모든 planning frame |
| control records | 모든 control frame |
| third-person frames | 모든 control frame 또는 manifest에 사전 명시한 렌더 주기 |
| diagnostic cards | 모든 observation / update / planning frame |
| planned missing frames | 0 |
| duplicate sequence IDs | 0 |
| timestamp reversals | 0 |
| unexplained carried/stale state | 0 |

---

## 6. 문서화 규칙

- **작업 시작 시**: `benchmark/ag3s/docs/AG3S_CUROBO_LIVE_TEST_PLAN.md`의 상태 표에서 T0~T3를 `재검증 필요`로 변경한다. 상태는 새 실험이 끝난 뒤에만 갱신한다.
- **진행 중**: 실행 과정, run별 completeness 표, 실패 기록 표, 시각화 링크, 수정 이력을 모두 `AG3S_CUROBO_LIVE_TEST_PLAN.md`에 계속 기록한다. 새로운 통합 실험 계획 MD 파일은 만들지 않는다.
- **완료 후**: 실제 구현과 검증이 최종 확인된 내용만 `AG3S_REVIEW_LOG.md`에 요약한다.
- 모든 run의 산출물(로그, 프레임 기록, card, 그래프)은 run ID별 디렉터리에 저장하고 그 경로를 manifest와 문서에 기록한다.

---

## 7. 최종 보고

작업 종료 시 다음을 보고한다.
1. 단계별 최종 상태(통과 / 실패 / 중단)와 근거가 된 run ID
2. 발생한 모든 실패와 원인 분류, 수정 내용, 재시험 결과 요약
3. 기준·임계값을 변경한 경우 그 내역과 근거
4. 해결하지 못한 문제와 필요한 결정 사항(중단 조건에 걸린 경우)