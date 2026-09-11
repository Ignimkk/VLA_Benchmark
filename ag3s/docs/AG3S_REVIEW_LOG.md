# AG3S 공동 코드 검토 기록

> **목적** — 씬(장애물) 쪽 primitive 근사를 걷어내고 **attention + TSDF/ESDF** 로 대체하기
> 위해, 현재 코드를 스텝별로 함께 읽고 기능적 오류를 판정한다. 로봇 모델 자체의 primitive
> (sphere chain) 는 유지된다.
>
> **규칙** — 사용자와 함께 판정하기 전까지 어떤 발견도 "확정" 으로 쓰지 않는다.
> 각 스텝은 `설명 → 사용자 질문 → 공동 판정 → 기록` 순서로 진행하고, 스텝을 합치지 않는다.

> **검토 계획서** — [AG3S_REVIEW_PLAN.md](AG3S_REVIEW_PLAN.md). 스텝 분해, 삭제 경계표,
> 사전 조사에서 찾은 발견 목록의 근거가 그쪽에 있다. 이 파일은 그 계획의 **실행 기록**이다.

---

## 진행 현황

| Step | 대상 | 상태 | 완료 |
|---|---|---|---|
| 0 | 로그 생성 + `ag3s` ↔ `trajopt` 경계 지도 | 완료 | 2026-09-07 |
| 1 | `esdf.py` — TSDF/ESDF 코어 | 완료 | 2026-09-07 |
| 2 | `pipeline.py` ESDF 조립부 | 완료 | 2026-09-07 |
| 3 | `trajopt/linearize.py` ESDF 소비부 | 완료 | 2026-09-07 |
| 4 | `reconstruction.py` + `robot_filter.py` | 완료 | 2026-09-07 |
| 5 | `attention_lifting.py` | 대기 | — |
| 6 | `target_grounding.py` | 대기 | — |
| 7 | `multiview.py` | 대기 | — |
| 8 | `support_surface.py` + 평면 행 | 대기 | — |
| 9 | `collision_candidates.py` + `geometry.py` | 대기 | — |
| 10 | `clearance.py` + `constraint_builder.py` + `to_adapter.py` + `attached.py` | 대기 | — |
| 11 | `tests/ag3s/` + `tests/trajopt/` | 대기 | — |

---

## 누적 발견

상태는 `미확정` → (`확정` | `기각` | `보류`). 심각도는 전환 설계에 미치는 영향 기준.

| ID | 심각도 | 상태 | 요약 | 위치 | Step |
|----|--------|------|------|------|------|
| E1 | 심각 | **확정, 수정됨** | GRASP 의 target carving 이 **전신에 대해** target 을 지운다 | `pipeline.py:684-696` → `esdf.py:339` | 2 |
| E2 | 심각 | 미확정 | 테이블이 무한 half-space. 홈 자세에서 26/61 sphere 위반, 최악 −0.677 m | `constraint_builder.py:216` | 8 |
| E3 | 심각 | **확정**(Step 3), 수정 미완 — Step 10 | attached object 가 실제 optimizer 에 도달하지 않는다 | `linearize.py:126` (읽는 코드 없음) | 3 확정 / 10 수정 |
| E4 | 중간 | 확정 · 수정됨(부분) → **백엔드 교체로 소멸 예정** | ESDF 격자 밖은 무조건 자유(0.5m)로 오답 | `esdf.py:400`, `esdf.py:516` | 1 |
| E5 | 중간 | **확정, 수정됨** | per-camera 통계가 세 대 모두 `"camera"` 로 붕괴 | `pipeline.py:658` ↔ `esdf.py:466` | 2 |
| E6 | 중간 | 확정 · 수정됨(부분) → **백엔드 교체로 소멸 예정** | TSDF 적분이 전 복셀 중심을 매 카메라·매 프레임 투영 | `esdf.py:127-141` | 1 |
| E7 | 낮음 | **기각/재분류**(Step 4) | "로봇 마스크가 클라우드를 재구성" — 중복이 아니라 **다른 해상도의 다른 계산**. 재사용하면 로봇 픽셀 89.3% 를 놓친다 | `pipeline.py:597-625` | 2 제기 / 4 기각 |
| G1 | 심각 | **확정, 수정됨**(Step 4) | `pointcloud.depth_scale` 이 TSDF 경로에 적용되지 않아, mm depth 에서 충돌 필드가 **통째로 비고도 `status: ok`** | `pipeline.py:_depth_cameras_from` | 4 |
| G2 | 심각 | **확정, 수정됨**(Step 4) | 미관측 100% ESDF 가 `validity: valid` / `certified: True` 로 보고됨 — 노트만 남고 검증 플래그엔 안 닿음 | `pipeline.py:455-460` | 4 |
| G3 | 중간 | **확정, 수정됨**(Step 4) | 카메라 3대 중 2대가 **아무것도 기여 안 해도** 프레임이 정상으로 보고됨 | `pipeline.py:_esdf_coverage` | 4 |
| G4 | 중간 | **확정, 수정됨**(Step 4) | 로봇 구가 ESDF 격자 **밖**이면 무조건 자유로 읽히는데 아무도 경고 안 함 (E4 의 미완 부분) | `pipeline.py:_esdf_coverage` | 4 |
| F2 | 중간 | 미확정 | attention 정규화가 카메라별 — 융합 `max` 가 스케일을 고른다 | `attention_lifting.py:280` | 5 |
| F8 | 중간 | 미확정 | 희소 attention 에서 p95 컷이 0 이 되어 0값 시드 허용 | `target_grounding.py:183` | 6 |
| F9 | 중간 | 미확정 | `lift()` 의 `image_hw` 기본값이 다운샘플 후 어긋난다 | `attention_lifting.py:265` | 5 |
| F10 | 낮음 | 미확정 | `DEGRADED` 가 `NO_TARGET` 을 가린다 | `to_adapter.py:176-179` | 10 |
| X1 | 정보 | 미확정 | primitive 폴백 순서가 문서와 반대 (삭제 대상) | `geometry.py:255` | 9 |
| X2 | 정보 | 미확정 | overflow 집합체가 빈 슬롯과 구분 불가 (삭제 대상) | `constraint_builder.py:466` | 9 |


> **2026-09-11 갱신** — `esdf.py` 를 cuRoboV2 `Mapper` 로 교체하기로 하면서 **E4 와 E6 의 지위가
> 바뀌었다.** 둘 다 우리 dense-fixed-box 구현에 고유한 문제이고, block-sparse 구조에서는
> 애초에 생기지 않는다(벗어날 고정 상자가 없고, block discovery 가 전 복셀 투영을 대체한다).
> Step 1 의 수정은 교체 전까지 유효하고, 교체가 끝나면 이 두 항목은 닫힌다.
> **나머지 발견은 백엔드와 무관하게 그대로 우리 몫이다** — E1·E2·E3 는 접촉 정책/기하 표현의
> 문제이고, G1~G4 는 배선과 검증 플래그의 문제라 어느 필드 구현을 쓰든 필요하다.

---

## 모듈 처분

`필요` = ESDF 전환 후에도 그대로 / `축소` = 일부만 남음 / `삭제` = 제거.
각 스텝에서 확정한다.

| 모듈 | 처분 | 근거 |
|---|---|---|
| `esdf.py` | **대체** (2026-09-11) | cuRoboV2 `Mapper` 로 교체. API 검증 완료 — 같은 TSDF 에서 해상도 다른 ESDF 를 여러 번 뽑는 것이 stock API 로 된다. Step 1 에서 고친 E4·E6 는 그쪽 구조에서 애초에 안 생긴다. 우리 구현은 교체 완료 전까지 참조 구현으로 남긴다 |
| `pipeline.py` | 필요 (백엔드만 교체) | 배선·정책은 그대로 — E1 target carving 제거, E5 카메라 이름, G1 depth_scale, G2~G4 검증 플래그는 백엔드가 바뀌어도 전부 필요하다. `_build_esdf` 안의 필드 구현만 cuRobo 로 바뀐다 |
| `trajopt/linearize.py` | 필요 | 실제 optimizer 의 소비부. Step 3 에서 전체 검토 완료 — E1 수정과 정합 확인. E3(attached 미소비) 확정, 수정은 Step 10 |
| `reconstruction.py` | 필요 | ESDF 는 raw depth 를 먹지만 클라우드는 **지지면·attention lifting·target grounding** 이 계속 쓴다. `uv` 보존·결정론적 다운샘플이 그 셋의 전제 |
| `robot_filter.py` | 필요 (일부 대체 검토) | 클라우드용 self-filter 와 ESDF 용 이미지 마스크 양쪽에 쓰인다. **cuRobo 에 `RobotSegmenter` 가 있어** ESDF 용 이미지 마스크 쪽은 대체 후보다 — E7 의 성능 문제가 그것으로 사라지는지 확인 필요 |
| `attention_lifting.py` | (Step 5) | — |
| `target_grounding.py` | (Step 6) | — |
| `multiview.py` | (Step 7) | — |
| `support_surface.py` | (Step 8) | — |
| `collision_candidates.py` | (Step 9) | — |
| `geometry.py` | (Step 9) | — |
| `clearance.py` | (Step 10 — 정식 검토 전) | 수정 없음 — E1이 `ClearancePolicy.margin_matrix`를 그대로 재사용. 기존 primitive 정책 로직이 정확하다는 전제를 그대로 물려받았으므로 Step 10에서 이 전제 자체를 검토할 것 |
| `constraint_builder.py` | (Step 10) | — |
| `to_adapter.py` | (Step 10 — 정식 검토 전) | E1 수정으로 `build_constraint_set`에 `target_link_margin` 계산이 추가됐다 (2026-09-07). Step 10에서 이 모듈을 볼 때 같이 훑을 것 |
| `attached.py` | (Step 10) | — |
| `robot_models/urdf_sphere_chain.py` | 필요 | 로봇 primitive 는 유지. 자체 캡슐→구 이산화, `geometry.py` 미의존 |
| *(신규)* cuRobo `Mapper` | **도입 예정** | ESDF 인프라. `/tmp/curobo_src` 에 clone·설치·검증 완료 (commit `78fd485`). trajopt 어댑터는 미착수 |

---

## 참고 문헌과 적용 내역

수정마다 관련 논문을 찾아 그 해법을 참조한다 (2026-09-07 사용자 지시). 원문은 `/papers` 에 둔다.
어떤 논문의 무엇을 **실제로 가져왔고 무엇을 안 가져왔는지**를 여기 모아, 스텝 로그에 흩어지지
않게 한다.

| 문헌 | 위치 | 무엇을 다루나 |
|---|---|---|
| **cuRoboV2** (Sundaralingam et al., 2026) | `papers/cuRoboV2_ Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots.pdf` (52p, 발췌 텍스트 `papers/curobov2.txt`) | GPU-native TSDF/ESDF 지각 파이프라인. §5.1 block-sparse TSDF, §5.2 voxel-centric projection, §5.3 frustum decay, §5.4 PBA+ 로 dense ESDF |
| **cuRobo v1** (원 리포트) | `papers/curobo_report_v1.pdf` (내가 curobo.org 에서 받음, 사용자 업로드분 아님) | 부록 D — 잡은 물체를 구로 근사해 **로봇 쪽** 충돌 모델에 편입 (attached object) |
| **MoveIt** Allowed Collision Matrix | (온라인 문서, 원문 없음) | (링크, 물체) 쌍 단위 충돌 허용 테이블 |

### cuRoboV2 에서 가져온 것 / 안 가져온 것

**E6 (Step 1) — TSDF 적분 범위.** cuRoboV2 §5.2 의 4단계 중 1단계(block discovery: "이번 depth
프레임이 건드리는 블록만 먼저 찾는다")의 **발상만** 가져왔다. 우리 구현은 카메라 절두체의
AABB 하나로 자르는 성긴 컬링이고, cuRoboV2 는 8³ 복셀 블록 단위 해시 테이블 + CAS 로 훨씬
세밀하게 솎아낸다 (활성 블록 10만 기준 dense 대비 메모리 11× 절감 보고).

가져오지 **않은** 것과 그 이유:
- 블록 단위 희소 저장(해시 테이블, free-list, tombstone) — 자료구조를 통째로 바꾸는 일이라
  이번 수정 범위 밖. **GPU 전환 때 함께 볼 항목** (아래 참고).
- PBA+ 기반 dense ESDF 생성(§5.4) — 우리는 `scipy.ndimage.distance_transform_edt` + dirty-block
  국소 갱신. 판단 결과는 같고 속도만 다르다.
- frustum decay / block recycling(§5.3) — 동적 장애물이 사라졌을 때 빨리 잊는 장치. 우리 씬은
  아직 정적이라 필요가 확인되지 않았다.
- GPU 전반(CUDA graph capture, warp primitives) — 아래 참고.

**E4 (Step 1) — 격자 밖 처리.** cuRoboV2 §5.4 Stage 3 을 읽고 얻은 것은 **해법이 아니라
한계의 확인**이다: cuRoboV2 도 "절단대역 밖은 양(exterior), 미할당 영역도 exterior" 로 둔다 —
즉 "관측 안 됨 = 자유" 는 이 기법 계열의 공통 근사이고 풀린 문제가 아니다. 대신 cuRoboV2 가
쓰는 완화책은 §5.1 의 **두 채널 `min()` 합성** (depth 로 만든 SDF 와 알려진 analytic 기하의
SDF 중 작은 쪽)이고, AG3S 는 이미 지지면(평면) 행에서 부분적으로 같은 일을 한다. 이것을
일반화(작업공간 외곽·고정 장애물을 analytic 채널로 따로 강제)하는 것은 더 큰 설계 작업이라
하지 않았다 — **E4 의 진짜 해결은 아직 남아 있다.**

**E1 (Step 2) — 접촉 허용.** cuRobo v1 부록 D + MoveIt ACM 에서 가져온 것은 "**공유된 world
표현에서 지우지 말고 질의 쪽에서 예외를 준다**" 는 원칙. cuRobo 는 잡은 물체를 그리퍼에 붙은
구로 바꿔 로봇 쪽 모델에 넣고 world SDF 는 건드리지 않으며, MoveIt 은 (링크, 물체) 쌍 표로
관리한다. 우리는 AG3S 에 이미 있던 `ClearancePolicy` 의 (sphere, slot) 마진 행렬이 같은
발상이었으므로 **그 계산을 그대로 재사용**해 ESDF 경로로 옮겼다 (`target_link_margin`).
cuRobo 의 "잡은 물체를 로봇 쪽 구로 편입" 은 우리 **E3** 이 요구하는 것과 정확히 같은 구조인데,
E3 은 아직 미수정이다 — 고칠 때 이 문헌을 다시 볼 것.

### 실시간성 — 지금은 다루지 않는다 (2026-09-07 결정)

이 스택은 CPU/numpy 기반이라 실시간에 못 미친다 (실측 AG3S 한 청크 2 s 이상, 예산 66.7 ms).
**그 격차를 이 검토에서 좁히려 하지 않는다.** 나중에 cuRoboV2 / nvblox 의 GPU 가속 방식을
참조해 별도로 개선한다 — 그때 가져올 후보가 위 "가져오지 않은 것" 목록이다. 그때까지 성능은
"판단 결과를 안 바꾸면서 개선되면 좋음" 정도로만 다루고, **판단의 정확성**(무엇을 장애물로 볼
것인가, 언제 완화할 것인가)을 우선한다. 계획서 Context 4번 항목과 같은 내용.

---

## Step 0 — 회귀 기준선 + 경계 지도

- 시작: 2026-09-05 23:53 KST / 완료: 2026-09-07
- 처분: —
- attention 의 역할: —

### 회귀 기준선

검토 시작 시점의 상태. 나중에 코드를 고칠 때 비교 대상이 된다.

```
$ src/openpi/.venv/bin/python -m pytest tests/ag3s/ tests/trajopt/ -q
560 passed, 217 warnings in 49.34s      # ag3s 445 + trajopt 115
```

```
$ src/openpi/.venv/bin/python -m benchmark.ag3s.pipeline --frames 12 --profile
points: 256884 raw -> 41333 voxel -> 40057 after self-filter
candidates: 2 object, 0 unknown, 1 target, 1 support

AG3S per-stage latency (11 frame(s) after 1 warm-up)
  stage                        mean ms     std    n
  -------------------------- --------- ------- ----
  scene_reconstruction           39.08    3.58   11
  robot_self_filter              11.75    2.16   11
  support_surface                46.77    3.34   11
  attention_lifting               5.28    0.44   11
  target_grounding               37.66    5.19   11
  collision_candidates           11.97    2.08   11
  primitive_fitting               0.24    0.02   11
  constraint_generation           0.18    0.02   11
  -------------------------- --------- ------- ----
  TOTAL                         152.94
```

이 표에서 이미 읽히는 것 하나 — **삭제 대상인 primitive 경로는 전체 시간의 8% 밖에 안 쓴다**
(`collision_candidates` 11.97 + `primitive_fitting` 0.24 + `constraint_generation` 0.18).
비용이 큰 것은 `support_surface` 46.8 ms 와 `target_grounding` 37.7 ms 이고, 둘 다 **남는** 쪽이다.
(단 이 fixture 는 합성 씬이고 ESDF 는 꺼져 있다. RB-Y1 실측에서는 ESDF 자체가 1213 ms 중
대부분을 쓴다 — Step 1 에서 확인한다.)

### 경계 지도 — 누가 무엇을 하는가

- **AG3S (`benchmark/ag3s/`)**: 카메라 관측 → 포인트클라우드 → attention lifting → target
  grounding → (primitive: candidate 구/평면 fitting, 또는 esdf: `EsdfBuilder`) → `ConstraintSpec`
  /`EsdfField` 를 담은 `CollisionConstraintSet`. `pipeline.py:265` `_run()` 이 8단계 스테이지를
  고정 순서로 돈다. `_build_esdf` (`pipeline.py:668`) 는 8번 "constraint_generation" 안의 하위
  스테이지이고 target grounding **이후**에 돈다 — carving 여부가 grounded target 에 의존하기
  때문.
- AG3S 안의 `ConstraintBuilder` (`constraint_builder.py:65`) 는 **CasADi 심볼릭** 그래프를 한 번
  짓고 매 프레임 파라미터 벡터만 갱신한다 — 일반 NLP 솔버(IPOPT류)를 겨냥한 경로.
- 실제로 도는 optimizer 는 이게 아니라 `benchmark/trajopt/` 의 SQP (4.2k LOC).
  `linearize.py:531` `scene_from_constraint_set` 이 AG3S 의 `ConstraintSpec` (파라미터 벡터 +
  layout) 을 읽어 numpy `SceneSnapshot` 으로 디코드하고, `linearize.py:397` `linearize()` 가
  sphere FK + Jacobian 으로 **수치 선형화** 한다 (CasADi 그래프 재사용 없음 — docstring 벤치마크:
  monolithic graph 19s/19ms, per-step graph 0.31s/12.5ms, sphere FK+Jacobian 0.066s/1.07ms).
- ESDF 는 이미 이 수치 경로에 붙어 있다: `linearize.py:324` `_esdf_clearance` =
  `d_esdf(p) - r_robot - margin`, gradient 는 필드의 `∇d` 를 그대로 쓴다
  (`linearize.py:502-512`).
- `ConstraintBuilder` 의 심볼릭 기계장치는 **candidate(씬 primitive) 와 평면 블록 전용** 이고
  ESDF 는 그걸 거치지 않는다 — 심볼릭 브리지(`to_adapter.to_casadi`)는 씬 primitive 를 지우면
  통째로 죽는 대상.
- 평면(지지면) 행은 두 backend 모두에서 살아있다 — primitive 에서도 ESDF 에서도 별도 선형 행
  (`linearize.py:545` 주석: "표면 하나는 선형 행 하나로 정확한데 복셀로 옮기면 수천 행이 되고
  근사도 나빠진다"). 이게 E2 가 걸리는 지점.
- candidate 와 esdf 는 **동시에 켤 수 있다** (`backend: both`, ablation 목적).
  `scene_from_constraint_set` 이 `esdf` 모드에서는 candidate 슬롯을 전부 비활성화해 이중 제약을
  막는다 (`linearize.py:540-543`).

**계획서와 어긋나는 점 (미해결)**: 계획서는 "평면 행이 심볼릭 파라미터 벡터 안에 얹혀 있어
후보 블록을 지우면 운반체를 잃는다"고 적었는데, `SceneSnapshot` 에서는 `plane_normal/offset/
active` 가 candidate 와 별도 필드로 이미 분리되어 있다. AG3S 쪽 `ConstraintSpec.parameter_
values`/`layout` 이 candidate 와 plane 을 한 벡터에 같이 패킹하는지, trajopt 가 디코드할 때만
분리하는지는 아직 확인하지 않았다 — Step 8/10 에서 다시 본다.

### 사용자 질문과 답

- Q: (없음, Step 1로 진행하기로 함) / A: —

### 발견

(Step 0 에서는 위 "누적 발견" 표를 미확정 후보로 등재하는 것까지만 한다.)

---

## Step 1 — `esdf.py` (TSDF/ESDF 코어)

- 시작: 2026-09-07 / 완료: 2026-09-07
- 처분: 필요 (E4·E6 수정 적용 — 아래 "수정 적용" 참고. 완전한 해결은 아님, 한계 명시)
- attention 의 역할: —

### 이 모듈이 하는 일

세 계층: `VoxelGrid`(base 프레임 축 정렬 격자) → `TsdfVolume`(카메라별 투영 적분, `sdf = 관측깊이
- 복셀깊이`, 표면 뒤 절단 밖은 갱신 안 함) → `EsdfField`/`EsdfBuilder`(점유 3상태 분류 → EDT →
삼선형 보간되는 거리·기울기 조회). `EsdfBuilder.update()`가 프레임마다 적분→점유→(target/support
carve)→dirty 복셀 계산→국소 EDT 순으로 돈다. `default_bounds()`가 base 프레임 앞쪽만 자른 작업
공간 상자를 기본값으로 쓴다(비용 때문 — 사방으로 열면 14.8M 복셀/6.5s, 앞쪽만 자르면 4.3M/1.8s).

### 공부한 것

- 국소(incremental) 갱신은 "점유 상태가 실제로 바뀐 복셀"을 기준으로 dirty 판정하지, "TSDF가
  닿은 복셀"을 기준으로 하지 않는다 — 후자로 하면 카메라가 매 프레임 훑는 시야 전체가 dirty가
  되어 국소 갱신이 이름만 남는다(주석에 실패 기록: 1.82s → 1.77s로 이득 없었음).
- 이산화 편향은 항상 실제보다 **작게**(보수적으로) 나오도록 설계됨 — 표면이 점유 복셀 "중심"에
  찍히기 때문. 제약 부호상 안전한 방향.

### 사용자 질문과 답

- Q: (Step 0 설명에서 질문 없이 Step 1로 진행) / A: —
- Q: "뭐가 문제라는 건지 명확하게 말하세요" / A: 아래 "판정 근거"의 E4·E6 설명으로 답함 —
  E4는 관측 실패를 안전(0.5m 여유)으로 오답하는 **정확성** 문제, E6은 incremental 설계의 핵심
  절감 대상(EDT)이 실제 지배 비용(TSDF 투영)에는 적용되지 않는다는 **성능** 문제.

### 발견

| ID | 심각도 | 상태 | 요약 | 위치 |
|----|--------|------|------|------|
| E4 | 중간 | **확정** | ESDF 격자 밖은 무조건 자유(0.5m)로 오답 — 관측 실패와 실제 안전을 구분 못 함 | `esdf.py:400`, `esdf.py:516` |
| E6 | 중간 | **확정** | TSDF 적분이 전 복셀 중심을 매 카메라·매 프레임 투영 — incremental 최적화가 안 닿는 고정 비용 | `esdf.py:127-141` |

### 판정 근거

**E4** — 격자 상자(`default_bounds()`: x∈[-0.3,1.2], y∈[-0.9,0.9], z∈[0,1.6]) 밖의 임의의 점은
실제 장애물 유무와 무관하게 정확히 `outside_distance == max_distance`를 반환한다. "관측 안 됨"이
아니라 "0.5m 떨어져서 안전함"으로 답하는 것. `unknown_fraction` 등 어떤 통계에도 안 잡히므로
실패가 조용하다. 로봇이 상자 밖으로 움직이는 상황(베이스 회전, 팔이 y>0.9 로 뻗음 등)에서
안전하지 않은 방향으로 조용히 실패한다.

```
$ src/openpi/.venv/bin/python -c "..."
distance at (100,100,100) [격자 훨씬 밖]: [0.5]   (== outside_distance == max_distance)
distance at (-5.0, 0.0, 0.5) [x=-5, 하한 -0.3 밖]: [0.5]
```

**E6** — 국소 갱신이 완벽히 작동해 EDT를 완전히 스킵하는 상황(`n_occupancy_changed=0,
n_blocks=0`, 즉 "정적인 씬에서 공짜"라는 주석의 조건 그대로)에서도 `update()`가 1.24s 걸렸다.
`integrate()`만 분리해서 재면 카메라당 ~400-450ms(3대 ~1.2-1.4s)로 `update()` 시간의 대부분을
차지한다. 국소 갱신 최적화는 EDT 단계에만 적용되고, 지배적 비용인 TSDF 투영(전체 435만 복셀 ×
카메라 수)은 씬이 정적이든 아니든 매 프레임 고정 비용으로 남는다 — RB-Y1 실측 1213ms/frame,
`live-integration.md`의 청크 예산 533ms의 2배 이상.

```
$ src/openpi/.venv/bin/python -c "..."
grid shape (151, 180, 160) n_voxels 4348800
trial 0: 3x integrate() = 1364.0 ms  (454.7 ms/camera)
trial 1: 3x integrate() = 1266.5 ms  (422.2 ms/camera)
trial 2: 3x integrate() = 1204.3 ms  (401.4 ms/camera)
steady-state update() unchanged scene: 1243.9 ms, n_occupancy_changed=0, n_blocks=0
```
(voxel_size=0.010, 합성 카메라 3대, 480×640, RB-Y1 기본 config)

**모듈 처분**: `esdf.py` 자체는 ESDF 전환 후 **필요** — TSDF/EDT 핵심 로직 자체는 대체 대상이
아니다.

### 수정 적용 — 2026-09-07

사용자 지시로 검토를 잠깐 멈추고 E4·E6를 바로 고쳤다 (`ag3s/esdf.py`만 수정, Step 1 모듈 경계
안). 관련 문헌: `papers/cuRoboV2_...pdf` (Sundaralingam et al., cuRoboV2 — depth-fused
TSDF/ESDF, 매니퓰레이터 대상, block-sparse 투영 + 알려진 기하와의 min() 합성).

**E6 수정** — `TsdfVolume.integrate()`가 전체 격자 대신 그 카메라 절두체(근/원 평면 8 코너)의
base 프레임 AABB만 투영한다 (`_frustum_index_bounds`, 새 메서드). 절두체는 볼록체라 8 코너의
AABB가 정확한 경계이고, 절단대역만큼 패딩한다. cuRoboV2의 block-discovery 단계와 같은 발상을
해시 테이블 없이 numpy 슬라이싱으로 구현 — 다만 cuRoboV2는 8³ 복셀 블록 단위로 더 세밀하게
솎아내는데(격자 전체 대비 최대 11배 메모리 절감 보고) 여기서는 카메라당 하나의 AABB 상자
단위라 성긴 컬링이다. 완전한 block-sparsity는 하지 않았다는 걸 분명히 해 둔다.

**E4 수정** — `EsdfBuilder.update()`가 `outside_distance`를 `unknown_policy`에 맞춰 정한다
(`free`→`+max_distance`, `occupied`→`-max_distance`; 이전엔 정책과 무관하게 항상 `+`였다).
그리고 `EsdfField`에 `outside_query_count`/`outside_query_fraction`을 추가해 `distance()`
호출마다 격자 밖 조회 비율을 센다 — `unknown_fraction`과 같은 자리에 두는 관측성 훅이고,
`pipeline.py`/`ConstraintValidity`에 실제로 연결하는 건 Step 2의 몫으로 남겨 뒀다 (여기서
imports/consumers를 건드리지 않음). **정직하게 남기는 한계**: cuRoboV2도 절단대역 밖·완전
미할당 영역은 여전히 자유로 가정한다 — "관측 안 됨=자유" 기본값 자체는 이 기법으로 없어지지
않는다. cuRoboV2가 실제로 쓰는 완화책(알려진 정적 기하 채널과의 `min()` 합성)은 AG3S가 이미
평면(지지면) 행에서 부분적으로 하고 있는 패턴이고, 일반화하려면 더 큰 설계 작업이라 이번
수정 범위 밖으로 남긴다.

**검증**:
- 정확성(단위) — 합성 카메라로 새 `integrate()`와 옛 전체격자 투영을 직접 대조, `tsdf`/
  `weight` 배열이 완전히 일치(`np.allclose`, atol=1e-5), `n_updated`도 일치.
- 정확성(실측) — `run_0002` 5프레임을 수정 전/후로 각각 `esdf_rollout` 돌려 비교. 위반값
  (`-28.2→-0.8`, `-134.1→-1.4` mm), feasible/violated 개수(1/4) **완전히 동일** — 수치가
  아니라 계산량만 바뀌었다는 확인.
  ```
  $ git stash push -- ag3s/esdf.py   # 수정 전으로 되돌려 기준선
  $ MUJOCO_GL=osmesa .../python -m benchmark.trajopt.experiments.esdf_rollout \
      --records run_0002 --attention attention_step1_run0002.npz --frames 5 \
      --voxel 0.020 --esdf-margin 0.05 --support-surfaces field --constraint-links arms
  AG3S 2073 ms  TO 71 ms   # 수정 전
  $ git stash pop
  (같은 명령)
  AG3S 2015 ms  TO 70 ms   # 수정 후 — 5프레임/20mm에서는 차이가 노이즈 수준
  ```
- 성능 — **voxel_size=0.02(이 실측 스크립트 기본값)에서는 개선폭이 거의 안 보인다.** 이유는
  `esdf-full-scenario.md`가 이미 적어 둔 대로 이 설정에서 "AG3S" 시간의 대부분이 target
  grounding이지 TSDF 적분이 아니기 때문 — 20mm 격자(543k복셀)는 애초에 435만복셀(10mm)보다
  8배 작아 E6가 지목한 고정비 자체가 작다. E6를 처음 확정할 때 쓴 voxel_size=0.010(435만
  복셀) 조건에서 다시 재면:
  ```
  steady-state update() 3카메라, 수정 전: 1243.9 ms (E6 판정 근거와 동일)
  steady-state update() 3카메라, 수정 후 (현실적 head cam, FOV~70°, depth_max=2.0): 937.9 ms
  단일 wrist cam(FOV~45°, depth_max=2.0, 근접): n_considered 30.5%만 처리 (435만→132만복셀)
  ```
  즉 **개선폭은 카메라 기하(FOV·거리)에 크게 의존한다** — 넓은 화각으로 상자 대부분을 보는
  카메라는 절두체 AABB도 상자에 가까워 이득이 작고, 좁은 화각·근접 손목 카메라는 크게 준다.
  cuRoboV2가 보고하는 자릿수 개선(블록 단위 희소화, GPU)에는 못 미친다 — 이건 카메라당 상자
  하나짜리 성긴 컬링이지 블록 단위 희소 구조가 아니기 때문이고, 그 차이를 위 "E6 수정"에
  적어 뒀다.

**부수 발견**: 이 검증 과정에서 이 GPU 서버(`/mnt/dev/work`)의 `benchmark` 체크아웃에는
`tests/ag3s`·`tests/trajopt` 디렉터리가 **존재하지 않는다** (`git ls-files | grep '^tests/'`
빈 결과). Step 0에 적힌 "560 passed" 회귀 기준선은 이 서버에서 재현·검증할 수 없다 — 아마
로컬 머신(`/home/mk/dev_ws/vla/pi0_TO_ws`)에만 있다. Step 11("445개가 왜 E1~E3을 못 잡았는가")
에서 이 test 스위트를 실제로 열어봐야 할 텐데, 그때 로컬 머신에서 받아오거나 실행해야 한다는
뜻이다.

---

## Step 2 — `pipeline.py` ESDF 조립부

- 시작: 2026-09-07 / 완료: 2026-09-07
- 처분: 필요 (E1·E5 수정 적용, E7은 Step 4로 이월)
- attention 의 역할: `_build_esdf`는 target grounding **이후**에 돈다 — carving 여부(수정 전)와
  이제는 `target_link_margin` 계산(수정 후, `to_adapter.py`)이 grounded target 존재 여부에
  의존하기 때문. attention이 target을 못 잡으면(`target=None`) 이 스텝 전체가 조용히 "완화 없음,
  전부 안전 마진"으로 fail-close 된다 — Step 10에서 다시 확인할 지점.

### 이 모듈이 하는 일

`_run()`의 8번째 스테이지 안에서 세 함수가 순서대로 돈다: `_robot_mask_for`(597, 카메라별 로봇
자기 마스크) → `_depth_cameras_from`(627, 관측 → `CameraDepth` 리스트) → `_build_esdf`(668,
TSDF/ESDF 적분 + 예전엔 target carving).

### 사용자 질문과 답

- Q: "현재 step2에서 문제라고 밝힌 부분 이해 안갑니다. 정리해서 설명하세요" / A: E1을 "GRASP
  phase가 되면 사과가 있던 자리를 로봇 전체에게 지운다"는 구체적 시나리오로 재설명. E5·E7은
  각각 "로그가 거짓말", "같은 계산을 두 번"으로 요약.
- Q: "이 문제도 역시 해결하고 넘어가야합니다... 논문을 찾아보고... 필요하면 말해주세요" / A:
  MoveIt ACM + cuRobo attached-object 패턴을 찾아 보고, `/papers`에 있는 cuRoboV2와 별개로
  cuRobo 원 리포트(curobo.org)를 받아 확인함 (`papers/curobo_report_v1.pdf`는 사용자가 직접
  올리지 않았고 이쪽에서 받음 — 필요시 사용자가 `/papers`에도 넣어두겠다고 함).
- Q: (AskUserQuestion) "E1을 지금 완전히 고치려면 Step 3·10 코드도 같이 건드려야 하는데
  어떻게 할까요?" / A: "지금 셋 다 고친다" 선택.

### 발견

| ID | 심각도 | 상태 | 요약 | 위치 |
|----|--------|------|------|------|
| E1 | 심각 | **확정, 수정됨** | GRASP의 target carving이 전신에 대해 target을 지운다 | `pipeline.py:684-696` → `esdf.py:339` |
| E5 | 중간 | **확정, 수정됨** | per-camera 통계가 세 대 모두 `"camera"`로 붕괴 | `pipeline.py:658` |
| E7 | 낮음 | Step 4 로 이월 -> **거기서 기각됨** | 로봇 마스크 생성이 클라우드를 통째로 재구성 | `pipeline.py:597-625` |

### 판정 근거 — E1

**문헌**: MoveIt의 Allowed Collision Matrix(링크-물체 쌍 단위 허용 테이블)와 cuRobo(원 리포트,
`papers/curobo_report_v1.pdf` §"objects... attached to a gripper")가 같은 답을 준다 — 잡힌 물체를
**공유된 world 표현에서 지우지 않는다.** cuRobo는 잡힌 물체를 그리퍼에 rigid하게 붙은 구로
바꿔 로봇 쪽 충돌 모델에 편입시키고, world SDF는 그대로 둔다. MoveIt의 ACM은 아예 (링크, 물체)
쌍 단위로 허용 여부를 표로 관리해 질의 쪽에서 예외를 준다. 공통점: **전역 지도를 고치지 않고
질의 쪽에서 봐준다.**

**적용**: AG3S가 이미 갖고 있던 `ClearancePolicy`(primitive 시절의 (sphere, slot) 마진 행렬)가
정확히 이 발상이었으므로, 그 정책 계산을 그대로 재사용해 ESDF 경로에 옮겼다 — 새 정책 로직을
만들지 않았다.

1. **`pipeline.py:_build_esdf`** — `carve()` 호출 제거. 필드는 이제 target을 **항상** 담는다.
2. **`to_adapter.py:build_constraint_set`** — `builder.clearance_policy.margin_matrix(sphere_link_names, [TARGET], context=ctx)`의 TARGET 열을 그대로 읽어
   `CollisionConstraintSet.target_link_margin` (`(S,)`, 새 필드, `types.py`)에 담는다. 권한 없는
   링크는 이 배열에서도 `safety_margin` 그대로 — "허용된 손끝만" 이라는 성질이 값 자체에 있다.
3. **`trajopt/linearize.py`** — `SceneSnapshot`에 `target_points`/`target_link_margin` 필드
   추가. `_esdf_clearance`가 각 구에 대해 `d_field`(필드가 답하는 최근접 거리)와
   `d_target`(target 점군까지의 실제 최근접 거리, cKDTree)을 비교해 `d_field >= d_target -
   voxel_size`(그 구의 최근접 장애물이 사실 target이라는 뜻, 이산화 편향만큼 여유)이면
   `target_link_margin`을, 아니면 평소의 `esdf_margin`을 쓴다.

**검증**:

- **정책 계산 자체(단위)**: `run_0004` step 9(grasp phase)에서 `active_manipulators=['right']`로
  직접 호출.
  ```
  authorized links: ['ee_finger_r1', 'ee_finger_r2', 'ee_right']
  n relaxed spheres: 22 / 120, relaxed link names: ['ee_finger_r1', 'ee_finger_r2']
  margin value range: 0.0 0.05
  field distance at target centroid: -0.02   # carve 전이면 +0.4(=max_distance)였을 값
  ```
  `ee_right`는 권한은 있지만 그 이름의 sphere가 모델에 없어 relaxed에 안 잡힘 — 해는 없지만
  안전 쪽 실패(그 항목이 그냥 아무 효과가 없을 뿐)이므로 E1과는 별개로 기록만 해 둔다.
- **fail-closed**: `active_manipulators` 생략(아무도 권한 없음) → relaxed 0/120. `['left']`만
  → 왼쪽 손끝만 22개 relaxed, 오른쪽은 그대로. 의도대로 작동.
- **`_esdf_clearance` 결정 로직(단위)**: 합성 필드로 `is_target_closest` 산식을 직접 검증 —
  target 위치(d_target=0.0)의 구는 `True`(margin 0.0 적용, row=0.28), target에서 먼 구
  (d_target=0.687)는 `False`(margin 0.05 적용, row=0.23) — 계산이 설계한 그대로 갈린다.
- **실측 전체 파이프라인**: `run_0004` 15프레임(grasp phase 포함, chunk 10·14)을
  `esdf_rollout`으로 끝까지 돌려 예외 없이 완주, feasible 8/violated 7 — E2/E3 등 아직 안 고친
  다른 결함들이 남아 있어 완전히 clean하진 않지만, 이 fix 자체가 파이프라인을 깨지 않았다는
  확인.

### 판정 근거 — E5

`CameraObservation`은 `camera_id`(`CameraID` enum)로 자신을 가리키는데, 옛 코드는
`camera`/`name` 속성을 찾았고 둘 다 존재한 적이 없어 항상 fallback `"camera"`로 붕괴했다.
`pipeline.py:_depth_cameras_from`에서 `camera_id`를 먼저 시도하도록 고쳤다.

```
process_multi(head/left_wrist/right_wrist 3대) 후:
per_camera keys in esdf stats: ['head', 'left_wrist', 'right_wrist']   # 수정 전엔 ['camera'] 하나
```

### 판정 근거 — E7 (미수정)

`filter_robot_points`(Stage 2, 융합된 클라우드 전체에 대해 한 번)와 `_robot_mask_for`(ESDF용,
카메라마다 `backproject`를 다시 돌려 또 한 번) 가 같은 sphere ball-query를 두 번 한다는 것은
확정. 제대로 고치려면 `robot_filter.py`(필터가 `inside` 마스크와 uv를 버리지 않고 반환하게)·
`multiview.py`(`CameraResult`가 그 마스크를 들고 있게)·`pipeline.py`(재계산 대신 그걸 읽게) 세
파일을 같이 고쳐야 한다 — E1과 달리 아직 안 읽은 모듈(Step 4: `reconstruction.py` +
`robot_filter.py`)을 실제로 바꾸는 일이라, Step 4에서 그 모듈들을 제대로 본 뒤에 하는 게 맞다고
판단해 로그에 확정만 남기고 이월한다. 성능 문제일 뿐 정확성엔 영향 없다.

> **후속 (Step 4)**: 여기서 가정한 "같은 일을 두 번 한다" 가 **틀렸다.** 두 경로는 서로
> 다른 해상도에서 다른 것을 계산하며, 2단계 결과를 재사용하면 로봇 픽셀의 89.3% 를 놓친다.
> E7 은 Step 4 에서 **기각**되었다 — 근거는 그쪽 "판정 근거 — E7" 참고.

### 부수 발견

`DEFAULT_CONTACT_LINKS["right"]`에 `"ee_right"`가 있는데 (`config.py`) 이 이름을 가진 sphere가
실제 RB-Y1 constraint 모델(`ARM_LINKS` 필터)에는 없다 — `ee_finger_r1`/`ee_finger_r2`만 진짜
링크 이름과 일치한다. 해는 없지만(그 항목이 그냥 무효), 설정과 실제 모델 이름이 어긋난다는
뜻이라 어딘가 다른 곳(예: 원래 `ee_right`가 가리키려던 손목/손바닥 sphere)에 남은 사각지대가
있을 수 있다. Step 10에서 `resolve_link_names`/`sphere_link_names`를 볼 때 같이 확인.

---

## Step 3 — `trajopt/linearize.py` (ESDF 소비부)

- 시작: 2026-09-07 / 완료: 2026-09-07
- 처분: 필요 (E1 수정으로 이미 손을 탄 상태. E3 확정, 수정은 Step 10 으로 이월)
- attention 의 역할: 없음 — 이 모듈은 attention 을 보지 않는다. AG3S 가 이미 attention 으로
  정한 것(무엇이 target 인가, 어느 링크가 완화되는가)을 `target_points`/`target_link_margin`
  으로 받아서 쓸 뿐이다. **attention 이 틀리면 이 모듈은 그걸 알 방법이 없다** — 그것이 앞단
  (Step 5·6)의 정확성이 전환 후 오히려 중요해지는 이유다.

### 이 모듈이 하는 일

AG3S 가 "주변에서 뭘 봤는지"를 받아, 로봇이 지금 움직임으로 뭔가에 부딪히는지를 실제로 계산하고,
부딪힌다면 어느 방향으로 얼마나 피해야 하는지를 QP 가 풀 수 있는 선형 부등식으로 바꾸는 곳.
AG3S 가 눈이라면 이쪽이 그 눈이 본 것을 쓰는 뇌에 해당한다.

- `SceneSnapshot` (72) — AG3S 의 `ConstraintSpec.parameter_values` 를 numpy 로 디코드한 한 프레임.
  **CasADi 를 쓰지 않는다.**
- `CollisionLinearizer` (243) — 로봇 모델당 한 번 sphere FK + Jacobian 을 CasADi 로 짓고
  (`__init__` 258-265), 매 프레임은 `map()` 으로 horizon 전체를 한 번에 평가 (66 ms).
- `_esdf_clearance` (332) — `d_esdf(p) - r - margin`. E1 수정 후 margin 은 스칼라가 아니라
  (H, S) 배열 — 그 구의 최근접 장애물이 target 인지에 따라 갈린다.
- `_clearances_from` (336) — candidate / plane 클리어런스. **평면 행이 실제로 계산되는 곳**
  (365-370): `centres @ n.T - offset - r`, 유한 상판이 아니라 무한 half-space (E2 의 현장).
- `linearize` (397) — 세 블록을 한 배열로 합쳐 activation band 로 거르고 step 당
  `rows_per_step` 만큼 argpartition 으로 뽑는다. 뽑히지 않은 나머지는 버려지는 것이 아니라
  `full_violation` (373) 이 매 프레임 전수 재검사한다 — QP 는 부분집합을 최적화하지만 **안전
  판정은 항상 전부**.
- `append_collision_rows` (633) — QP 행렬에 행을 붙이는 배선. 설계 판단 없음.

### 공부한 것

- **선택(reduction)과 판정이 분리되어 있다.** 행을 줄이는 것은 QP 를 싸게 만들기 위한 것이고,
  "안전한가"를 답하는 `full_violation` 은 그 축약을 쓰지 않는다. 축약이 수렴을 나쁘게 할 수는
  있어도 충돌을 숨길 수는 없다는 것이 이 설계의 보증.
- **ESDF 행이 candidate 행보다 구조적으로 싼 이유** — 필드는 점에 대해 답하므로 슬롯을 열거할
  필요가 없다. candidate 는 `S x M` 행, ESDF 는 `S` 행.
- ESDF 의 기울기는 정규화하지 않고 그대로 쓴다 (502-512) — eikonal 성질상 자유공간에서 크기가
  1 이어야 하므로, 정규화하면 필드가 망가진 것을 숨기게 된다.

### 사용자 질문과 답

- Q: "우선 뭘한건지 뭘 분석한건지 모르겠어" / A: 파일 전체(688줄)를 읽고 세 가지를 확인했다고
  평이하게 재설명 — (1) 앞 스텝 수정이 제대로 연결됐는지, (2) 잡은 물체가 계산에 들어가는지,
  (3) 테이블 관련 코드의 위치. 이 피드백으로 앞으로 모든 설명은 **평이한 요약 먼저, 코드
  근거는 그 다음** 순서로 하기로 함.
- Q: "AG3S 먼저 다 하고 trajopt 를 거기 맞춰 고치는 게 낫지 않나? 지금 방식과 뭐가 다른가" /
  A: 지금 방식은 패키지 경계가 아니라 **데이터가 흐르는 순서**를 따른다. 근거 둘 — (1) Step 0
  에서 AG3S 의 CasADi 브리지가 실제로 안 쓰인다는 것이 trajopt 를 봤기 때문에 드러났다,
  (2) E1 은 AG3S 만 봐서는 고칠 수 없었고 pipeline.py + linearize.py + to_adapter.py 를 동시에
  고쳐야 했다. 사용자가 현행 유지를 선택.

### 발견

| ID | 심각도 | 상태 | 요약 | 위치 |
|----|--------|------|------|------|
| E3 | 심각 | **확정**, 수정은 Step 10 이월 | attached object 가 실제 optimizer 에 도달 안 함 | `trajopt/` 전체에 attached 소비 코드 없음 |
| E2 | 심각 | 미확정 (판정은 Step 8) | 평면 행이 무한 half-space — 현장은 `linearize.py:365-370` | Step 8 에서 판정 |

### 판정 근거 — E3

`attached` 라는 단어가 `linearize.py` 에 두 번 나오는데 **둘 다 주석이고 코드가 아니다**:
78 행은 "the clearance policy, the overflow aggregation and the **attached-object block** all end
up in that vector, so a change on the AG3S side reaches the optimizer without a second
implementation" — 즉 **docstring 이 주장하는 내용 자체가 사실이 아니다.** 328 행은 "no field is
attached" 로 ESDF 를 가리키는 무관한 용례.

```
$ grep -rln "attached\|Attached" benchmark/trajopt --include="*.py"
benchmark/trajopt/linearize.py      # 위 두 주석뿐
```

`SceneSnapshot.from_spec` (126-154) 이 파라미터 벡터에서 읽는 블록은 `pos`, `radius`, `active`,
`d_safe`, `plane_normal`, `plane_offset`, `plane_active` 뿐이다. AG3S 가
`constraint_builder._attached_rows` 로 채워 넣는 attached 블록을 **읽는 코드가 없다.**

**결과**: 로봇이 물체를 쥐고 팔을 움직일 때, 그 물체가 테이블·다른 물체에 부딪히는지를 이
시스템은 계산하지 않는다. 로봇 본체만 피한다. 표현의 문제가 아니라 **배선의 문제** — AG3S 는
이미 필요한 것을 만들어 내보내고 있고 소비 쪽이 그것을 무시한다.

**수정을 Step 10 으로 미루는 이유**: 제대로 고치려면 `attached.py` (아직 안 읽음, Step 10) 의
`AttachedCollisionGeometry` -> FK -> 구 위치 경로를 알아야 하고, ESDF 경로에서는 그 구들을
"로봇 구의 연장"으로 다뤄야 한다 — cuRobo v1 부록 D 가 정확히 그 구조다 (위 "참고 문헌" 참고).
E1 처럼 지금 당장 고치려면 아직 안 읽은 모듈의 설계를 추측해야 하므로, Step 10 에서 그 모듈을
읽은 뒤에 고치는 것이 맞다고 판단. **심각 등급 결함을 미수정 상태로 들고 간다는 점을 명시해
둔다** — E7(낮음, Step 4 이월)과 달리 이것은 안전에 직접 영향이 있다.

### 이 스텝에서 수정한 것

없음. E1 수정(Step 2)으로 이 파일의 `SceneSnapshot`·`_esdf_clearance`·
`scene_from_constraint_set` 이 이미 바뀐 상태였고, 이 스텝은 그 변경이 모듈 전체와 정합하는지
확인하는 것이 목적이었다 — 정합했다. E2 는 Step 8, E3 는 Step 10 에서 다룬다.

---

## Step 4 — `reconstruction.py` + `robot_filter.py`

- 시작: 2026-09-07 / 완료: 2026-09-07
- 처분: 둘 다 **필요**
- attention 의 역할: 간접적이지만 결정적. `reconstruction` 이 `(u, v)` 를 끝까지 보존하기
  때문에 attention lifting 이 "이 3D 점은 어느 픽셀에서 왔는가" 를 답할 수 있다. voxel
  다운샘플이 **중심점이 아니라 대표점**을 남기는 이유가 이것이다 — 중심점에는 픽셀이 없다.
  즉 이 모듈은 attention 을 직접 다루지 않지만, attention 이 3D 로 올라올 수 있게 하는 전제를
  유지하는 쪽이다.

### 이 모듈이 하는 일

**`reconstruction.py`** — depth 이미지(또는 이미 3D 인 점군)를 받아 로봇 base 프레임의
점군으로 바꾸고, 물리적으로 무효한 점을 걸러내고, voxel 다운샘플 + `max_points` 상한을 건다.
`backproject` (58) 가 핀홀 역투영, `voxel_downsample` (146) 이 복셀당 대표점 하나,
`coverage_preserving_cap` (174) 이 "인덱스를 고르는 대신 복셀을 키워서" 상한을 맞춘다.

**`robot_filter.py`** — 점군에서 로봇 자신의 몸에 해당하는 점을 지운다. 제약을 쓰는 것과
**같은** sphere chain 으로 질의하므로 (`robot_sphere_mask`, 32), 필터가 지운 것과 제약이 보는
것이 어긋날 수 없다. 질의는 점마다가 아니라 **구마다** 돈다 (점 6만 개 vs 구 200 개 — 실측
257 ms 대 8 ms).

### 이 스텝의 핵심 질문 — ESDF 가 raw depth 를 먹는데 클라우드가 왜 아직 필요한가

전환 후에도 클라우드를 소비하는 곳이 셋 남는다. 전부 **남는** 쪽이다.

| 소비자 | 왜 점군이어야 하는가 |
|---|---|
| `support_surface` | RANSAC 평면 적합. 복셀 격자가 아니라 점 집합이 필요 |
| `attention_lifting` | 3D 점 <-> 픽셀 대응. `uv` 가 살아 있어야 성립 |
| `target_grounding` | attention 시드에서 3D 연결성으로 번지는 영역 성장 |

즉 **ESDF 는 "무엇을 피할 것인가" 를 답하고, 클라우드는 "무엇이 target 이고 무엇이 지지면인가"
를 답한다.** 둘은 같은 depth 를 서로 다른 목적으로 읽는 두 소비자이고, 그래서 아래 G1 이
생겼다 — 같은 입력의 **단위 해석이 갈라졌다**.

### 공부한 것

- `max_points` 상한을 "인덱스 고르기" 로 걸면 공간적 보장이 없다. 인덱스 인접성은 래스터
  인접성이지 공간 인접성이 아니라, 얇은 물체가 선택된 두 인덱스 사이로 통째로 빠질 수 있다.
  `cap_strategy: "voxel"` 이 기본인 이유 — 복셀을 키우면 **거친 클라우드**가 되지 부분적인
  클라우드가 되지 않는다.
- self-filter 의 보수적 방향은 "더 지우는 쪽" 이다. 덜 지우면 그리퍼에 용접된 유령 장애물이
  남는다.

### 사용자 질문과 답

- (이 스텝에서는 사용자 질문 없이 진행. 앞 스텝의 지시 두 가지가 적용됨 — 실시간성은 지금
  고려하지 않는다, 그리고 설명은 평이한 요약을 먼저.)

### 발견

| ID | 심각도 | 상태 | 요약 |
|----|--------|------|------|
| G1 | 심각 | **확정, 수정됨** | `pointcloud.depth_scale` 이 TSDF 경로에 적용되지 않는다 |
| G2 | 심각 | **확정, 수정됨** | 미관측 100% 필드가 `validity: valid` 로 보고된다 |
| E7 | 낮음 | **기각/재분류** | 중복 계산이 아니라 다른 해상도의 다른 계산이다 |
| G3 | 중간 | **확정, 수정됨** | 기여 0 인 카메라가 있어도 프레임이 정상으로 보고된다 |
| G4 | 중간 | **확정, 수정됨** | 격자 밖 로봇 구를 아무도 경고하지 않는다 (E4 의 미완분) |

### 판정 근거 — G1 (새로 발견)

**증상**: `depth_scale` 은 `backproject` (`reconstruction.py:88`) 에서만 적용되고,
`pipeline._depth_cameras_from` 이 TSDF 로 넘기는 depth 에는 적용되지 않았다. 같은 depth
이미지를 두 소비자가 **다른 단위로 읽는다.**

`depth_scale: 0.001` 은 가상의 설정이 아니다 — `config.py:143` 이 "0.001 for uint16 mm" 라고
직접 예시하고, `RUNBOOK.md` 의 `--record-depth` 가 **uint16 밀리미터**로 저장하며,
`trajopt/wire.py:57-60` 이 실제 depth 카메라 형식이라고 적어 둔 그 값이다.

**수정 전 실측** (`run_0004` step 9, 실제 RB-Y1 depth 를 mm 로 넘김, `depth_scale: 0.001`):

```
status           : ok
validity         : valid   geometry_certified=True
target grounded  : True
support surfaces : 2
esdf occupied vox: 0  unknown=100.0%
d_esdf probes    : [0.4 0.4 0.4]     <- max_distance, 즉 "아무것도 없음"
```

점군 경로는 정상 동작해서 (126,328 점, target 잡힘, 지지면 2개) 프레임 전체가 건강해 **보이는**
동안, 충돌 필드는 완전히 비어 있었다. `collision_backend: esdf` 에서는 필드가 **유일한** 장애물
원천이므로, 이 프레임은 "작업공간 전체가 자유" 라고 optimizer 에 말한다.

**수정**: `_depth_cameras_from` 이 `CameraDepth` 를 만들 때 `pointcloud.depth_scale` 을
적용한다. `_robot_mask_for` 에는 **스케일 전** depth 를 그대로 넘긴다 — 그쪽은 `backproject` 를
거치고, `backproject` 가 스케일을 자기가 적용하므로 이중 적용이 된다.

**수정 후 같은 입력**: `esdf occupied vox: 3067, unknown=72.8%,
d_esdf probes: [0.05, 0.034, -0.02]` — 미터로 넘긴 정상 실행(3076 / 72.8%) 과 일치한다
(3067 vs 3076 차이는 mm 반올림).

### 판정 근거 — G2 (새로 발견)

**증상**: 미관측 비율이 아무리 높아도 `notes` 에 문장 하나가 추가될 뿐, `ConstraintValidity`
에는 닿지 않았다 (`pipeline.py:455-460`). 그래서 위 G1 사례가 `geometry_certified=True` 로
나왔다 — 소비자가 실제로 게이트하는 플래그가 그것이다 (`live-integration.md`).

이것은 `ConstraintValidity` **자신의 docstring 이 이름 붙여 둔 실패**다:

> "an empty `VALID` set means 'AG3S looked and there is nothing there', while `INCOMPLETE` means
> 'AG3S cannot tell you what is there'. **Rendering the second as the first is how a robot drives
> into an unmodelled wall.**"

덧붙여 `live-integration.md` 는 "미관측 비율이 높으면 `degraded` 가 난다" 고 적고 있는데,
**그것도 사실이 아니었다** (실측 미관측 72.8% 에서 `validity: valid`). E3 의 docstring 과 같은
종류의 어긋남이다.

**수정**: 두 단계로 validity 에 반영한다.
- `n_free == 0 and n_occupied == 0` (한 복셀도 관측되지 않음) -> **INCOMPLETE**. 진짜로 빈
  작업공간이라도 광선이 통과한 자리는 FREE 로 남으므로, 이 조건은 **실패에만** 걸린다 — 잘못된
  depth 단위, 잘못된 외부 파라미터, 카메라가 볼 수 없는 bounds 상자.
- `unknown_fraction >= esdf.unknown_report_threshold` (기본 0.9) -> **DEGRADED**. enum 자신의
  표현으로 "represented, but ... partially observed". 여전히 쓸 수 있되 인증은 안 된다.

**수정 후 실측** (같은 프레임, `esdf.bounds_*` 를 카메라가 볼 수 없는 곳으로 옮겨 필드 적분을
실패시킴):

```
(a) 정상            : occupied=3076 free=145670 unknown=72.8%  status=ok      validity=valid      certified=True
(b) 격자를 로봇 뒤로 : occupied=0    free=0      unknown=100%   status=geometry_incomplete validity=incomplete certified=False
(c) 격자를 공중으로  : occupied=0    free=0      unknown=100%   status=geometry_incomplete validity=incomplete certified=False
```

정상 프레임(a)의 판정은 **바뀌지 않았다** — 회귀 없음.

**남겨 두는 것**: `live-integration.md` 가 암시하는 "72% 면 degraded" 는 여전히 아니다(임계값
0.9). 이 씬의 정상 미관측이 70~73% 라고 `esdf-full-scenario.md` 가 측정해 두었으므로 그
수준을 DEGRADED 로 부르면 모든 프레임이 미인증이 된다. 임계값을 어디에 둘 것인가는 별도 판단이
필요해 지금은 기존 `unknown_report_threshold` 를 재사용했고, 문서 쪽 표현이 코드와 어긋난다는
사실만 기록해 둔다.

### 판정 근거 — E7 (기각/재분류)

Step 2 에서 "`_robot_mask_for` 가 `filter_robot_points` 와 같은 일을 두 번 한다" 로 제기했는데,
**같은 일이 아니다.**

- `_robot_mask_for` -> `backproject` : **모든 유효 픽셀**을 역투영한다 (다운샘플 없음).
  이미지 공간 마스크를 만드는 것이 목적이므로 픽셀 해상도가 그대로 필요하다.
- 2단계 self-filter -> `reconstruct` : **voxel 다운샘플된** 클라우드에 대해 돈다 (복셀당 대표점
  하나).

같은 프레임에서 재어 보면:

```
image                       : 480x640 = 307200 px
backproject (mask path)     : 126331 pts -> 53225 px marked robot
reconstruct (stage-2 cloud) : 57616 pts (voxel 5.0 mm) -> 5717 px marked robot

robot px stage-2 mask would MISS: 47508 (89.3% of the real robot pixels)
```

즉 "2단계 결과를 재사용한다" 는 자명해 보이는 최적화는 **로봇 픽셀의 89.3% 를 놓치고**, 그
픽셀들이 TSDF 에 장애물로 적분된다 — `_robot_mask_for` 가 애초에 막으려던 바로 그 버그(주석에
기록된 실측: 194 구 중 130 구 위반). **하면 안 되는 수정**이라는 것을 수치로 남겨 둔다.

남는 것은 순수한 성능 문제뿐이고 — KD-tree ball query 가 서로 다른 두 점 집합에 대해 두 번
돈다 — 올바른 해법은 재사용이 아니라 **로봇 구를 이미지에 직접 래스터화**하는 것이다 (픽셀
수가 아니라 구 수에 비례). 이는 cuRoboV2/nvblox 의 GPU 작업과 같은 성격이라 "실시간성" 항목으로
넘긴다 (위 "참고 문헌과 적용 내역" 참고). **E7 은 제기된 형태로는 기각한다.**

### 이 스텝에서 수정한 것

`pipeline.py` 두 곳 — `_depth_cameras_from` 의 `depth_scale` 적용(G1), ESDF 미관측의
`ConstraintValidity` 반영(G2). `reconstruction.py` 와 `robot_filter.py` 자체는 **수정하지
않았다**.

**회귀 확인**: `run_0004` 15프레임 `esdf_rollout` — feasible 8 / violated 7, 위반값
(-28.7->-0.6, -68.0->+1.7, -140.3->+3.7, -73.9->+3.2 mm) 전부 Step 2 직후 실행과 **동일**.
정상 경로의 판정은 하나도 바뀌지 않았다.

### 후속 (같은 날) — Step 4 가 드러낸 **미해결분**을 마저 잡는다

위 G1·G2 를 고친 뒤에도 "필드가 조용히 틀리는" 경로가 두 개 더 남아 있었다. G2 는 필드가
**빈** 경우만 잡는데, **틀린** 필드는 비어 있지 않기 때문이다. 둘 다 실측으로 확인했다.

**(A) E4 의 훅이 죽은 코드였다.** Step 1 에서 `outside_query_fraction` 을 추가하며 "파이프라인에
연결하는 건 Step 2 의 몫" 이라고 적어 두고 **끝내 연결하지 않았다.**

```
$ grep -rn "outside_query_fraction" --include="*.py" benchmark/
benchmark/ag3s/esdf.py: (정의·증가만, 읽는 곳 없음)
```

실측 — 로봇 구가 격자 밖에 있을 때:
```
d_esdf = [0.4 0.4 0.4]            <- max_distance, 즉 "아무것도 없음"
outside_query_fraction = 100.0%   <- 값은 쌓인다
frame: status=ok validity=valid certified=True     <- 아무 영향 없음
```

**(B) 카메라가 죽어도 프레임은 정상.** 3대 중 2대의 외부 파라미터를 망가뜨려 기여를 0 으로
만들었는데도:
```
per-camera n_updated = {'head': 150873, 'left_wrist': 0, 'right_wrist': 0}
occupied=6416 free=144457 unknown=72.4%   -> G2 의 조건(둘 다 0)에 안 걸린다
```
**E5 를 고치지 않았다면 이 검사는 아예 불가능했다** — 그전에는 세 대가 `"camera"` 한 키로
뭉개져 죽은 카메라를 구분할 방법 자체가 없었다.

**오탐 위험 사전 측정** (검사를 넣기 전에 정상 경로에서 재 봄):
```
정상 3카메라 프레임의 n_updated : head=129893~148746, left_wrist=13399~40776, right_wrist=27592~38843
격자 밖 구 (arms 120구, 롤아웃 44프레임 전체) : 0
격자 밖 구 (전신 194구)                      : 25 — base, wheel_l, wheel_r, link_torso_0 (전부 격자 바닥 아래)
```
즉 `n_updated == 0` 도, `arms` 모델의 격자 밖 구도 정상 경로에서는 **한 번도** 나타나지 않는다.
전신 모델의 25구는 오탐이 아니라 사실이며, `--constraint-links arms` 가 기본인 문서화된 이유와
같은 것을 다른 각도에서 말해 준다.

**수정** — `pipeline._esdf_coverage()` 신설, ESDF 가 만들어진 직후 호출.
- **G3**: 공급된 카메라 중 `n_updated == 0` 인 것이 있으면(전부는 아닐 때 — 전부면 G2 가 잡는다)
  이름을 적어 note + `DEGRADED`.
- **G4**: `constraint_robot_model` 의 구를 현재 `q` 로 풀어 격자 경계와 비교하고, 밖에 있으면
  개수·링크 이름과 함께 note + `DEGRADED`. **질의 시점 카운터보다 이쪽이 낫다** — optimizer 가
  계획을 세우기 **전에** AG3S 가 말해 줄 수 있기 때문. `outside_query_fraction` 은 trajopt 쪽
  진단용으로 남겨 두되, 프레임 판정의 책임은 G4 가 진다.

**수정 후 실측**:
```
정상 3대           : n_updated head=148746 left=13399 right=29242  -> status=ok validity=valid certified=True
손목 2대 파손       : n_updated head=148746 left=0 right=0          -> status=degraded validity=degraded certified=False
                     note: camera(s) left_wrist, right_wrist contributed no voxels to the ESDF this frame ...
격자를 못 보는 위치 : note: 120 of 120 constraint sphere(s) lie outside the ESDF grid (ee_finger_l1, ...)
                     -> status=geometry_incomplete validity=incomplete certified=False
```
정상 프레임의 판정은 **바뀌지 않았다.**

**회귀**: `run_0004` 15프레임 `esdf_rollout` — feasible 8 / violated 7, 위반값
(-28.7->-0.6, -68.0->+1.7, -140.3->+3.7, -73.9->+3.2 mm) 전부 이전 실행과 동일.

### Step 4 이후에도 **남아 있는** 것 (정직하게)

- **G2 의 임계값이 미정.** `unknown_report_threshold` 0.9 를 재사용했는데, 이 씬의 정상 미관측이
  70~73% 다. 어느 선부터 "너무 못 봤다" 인지는 아직 아무도 정하지 않았다. `live-integration.md`
  는 72% 면 degraded 라고 적고 있어 **문서와 코드가 여전히 어긋난다.**
- **부분적으로 *틀린* 필드는 여전히 통과한다.** G3 는 기여가 **정확히 0** 인 카메라만 잡는다.
  외부 파라미터가 조금 틀려 기하를 엉뚱한 자리에 적분하는 카메라는 `n_updated` 가 크게 나오므로
  아무 검사에도 안 걸린다. 이것을 잡으려면 카메라 간 일치도(같은 표면을 서로 다른 곳에 그리는지)
  를 봐야 하고, 그건 Step 7(`multiview.py`)의 몫이다.
- **E7 의 성능 문제**는 정책상 GPU 작업으로 미룸 (기각된 것은 "중복 계산" 이라는 진단이지
  성능 문제 자체가 아니다).

---

## cuRoboV2 API 검증 (2026-09-11) — 2계층 ESDF 설계의 전제 확인

계획서 "방향 결정" 이 세운 가정 — **한 TSDF 에서 해상도가 다른 ESDF 를 여러 번 뽑아 합성할 수
있는가** — 을 실제 코드와 실행으로 확인했다. 결론: **가능하고, API 의 의도된 사용 방식이다.**

### 환경

- 이 GPU 서버: H200 NVL 140 GB, driver 580.173, torch 2.4.1+cu124 (conda)
- `git clone https://github.com/NVlabs/curobo` → `/tmp/curobo_src` (78fd485)
- **nvcc 불필요**: `USE_PYBIND` 기본 0, JIT 백엔드를 쓴다. `pip install -e . --no-build-isolation`
  + `pip install 'cuda-core[cu12]'` 로 끝.
- **openpi venv 를 건드리지 않았다** — `/tmp/curobo_venv` 를 따로 만들어 설치했다
  (curobo 의존성이 numpy/scipy 를 올릴 수 있어 격리가 필요했다).

### 소스에서 확인한 것

| 항목 | 확인 |
|---|---|
| TSDF 와 ESDF 해상도 분리 | `MapperCfg.voxel_size` (5 mm) vs `esdf_voxel_size` (기본 50 mm) |
| ESDF 범위를 따로 지정 | `extent_esdf_meters_xyz` 가 `extent_meters_xyz` 와 별개 |
| **호출마다 해상도·원점 변경** | `Mapper.compute_esdf(esdf_origin=..., esdf_voxel_size=...)` — docstring 이 "for sliding window" 라고 적고 있다 |
| **여러 ESDF 를 한 world 에** | `SceneCfg.voxel: Optional[List[VoxelGrid]]` — 실제로 2개 넣어 확인 |
| Mapper 공존 | `block_size` docstring: "two Mappers with different block sizes can coexist" |
| 보너스 | `RobotSegmenter` — 우리 `_robot_mask_for`(E7) 에 해당하는 것을 GPU 로 제공 |

### 실측 — RB-Y1 `run_0004` step 9, 실제 depth 3 대

**정상 상태 (워밍업 3회 후 10회 평균).** 첫 호출은 JIT 컴파일 + CUDA graph capture 때문에
1302 ms 가 나오므로 그 숫자를 쓰면 안 된다.

```
TSDF 적분 (3 카메라)              1.72 ± 0.10 ms
ESDF 전체 20 mm (2.56 m 창)       0.52 ± 0.03 ms
ESDF target 주변 5 mm (0.64 m)    0.74 ± 0.06 ms
────────────────────────────────────────────────
2계층 합계                        2.98 ms   / 예산 66.7 ms
(우리 numpy 구현 실측)           ~2600 ms
```

**약 870 배.** 예산 대비 22 배 여유가 남는다 — 지금 우리 구현은 예산의 40 배를 쓰고 있다.

### 정확성 검증 — 그리고 내가 두 번 틀린 것

처음 두 번의 검증 시도에서 **좌표계가 97 mm 어긋난다**는 결과가 나왔다. 원인은 curobo 가 아니라
**내 인덱싱 가정** 이었고, 그것을 확정하는 과정이 이 검증의 핵심이다.

1. **1차 오류** — `esdf_origin` 을 격자 **코너**로 가정했다. 실제로는 **중심**이다
   (`integrator_esdf.py:909` 주석 "Pose at center", 그리고 실행 결과 `vg.pose[:3]` 가 내가 넘긴
   `esdf_origin` 과 정확히 같다). 코너로 넣어서 target 이 창 모서리에 걸렸고 두 계층이
   130~257 mm 어긋났다.
2. **2차 오류** — 중심으로 고친 뒤에도 수직선 탐침이 테이블을 97 mm 위로 보고했다. 이번엔
   손으로 쓴 `(p - corner)/vs` 인덱싱이 문제였다.

**우회 진단**: 인덱싱을 쓰지 않고 `mapper.extract_occupied_voxels()` 로 점유 복셀을 직접 꺼내
실제 기하와 대조했다 — TSDF 는 처음부터 정확했다.

```
점유 복셀 82,216 개.  테이블 영역 z 분포 상위:
   z 0.82~0.84 : 16,228 개
   z 0.80~0.82 : 13,907 개      <- 우리 RANSAC 평면 z=0.823 과 일치
   z 0.00~0.02 : 16,856 개      <- 바닥
```

그래서 마지막에는 **curobo 자신의 좌표 생성기** `VoxelGrid.create_xyzr_tensor(transform_to_origin=True)`
를 써서 다시 쟀고, 그러자 전부 맞았다.

```
=== 거친 20 mm ===
  pose(중심) [0.45 0. 0.8]  dims [2.56 2.56 2.56]
  |d|<20mm 복셀의 z 중앙값 : 0.790 m      (참값 0.823 대비 -33 mm)
  target centroid d = +0.000 m

=== 미세 5 mm (target 중심) ===
  pose(중심) [0.554 0.302 0.854]  = target centroid  <- esdf_origin 이 중심임을 확증
  dims [0.64 0.64 0.64]
  |d|<5mm 복셀의 z 중앙값 : 0.821 m       (참값 0.823 대비 -2 mm)
  target centroid d = -0.005 m            (사과 표면 바로 안쪽 — 옳다)
```

**이것이 2계층 설계의 이득을 end-to-end 로 수치화한 것이다**: 같은 테이블 상판을 거친 계층은
**33 mm** 틀리게, 미세 계층은 **2 mm** 틀리게 본다. `esdf_margin` 50 mm 기준으로 마진 예산의
66% 를 먹던 것이 4% 로 떨어진다.

### 재현 — 스크립트는 저장소에 있다

검증 환경(`/tmp/curobo_src`, `/tmp/curobo_venv`)은 **휘발성**이므로 재부팅 후 다시 만들어야
한다. 스크립트와 절차는 `benchmark/ag3s/experiments/curobo/` 에 넣어 두었다.

| 파일 | 하는 일 | 실행 venv |
|---|---|---|
| `README.md` | 환경 구성 절차, 밟았던 함정 3가지 | — |
| `export_frame.py` | RB-Y1 한 프레임을 `/tmp/rby1_frame.npz` 로 내보냄 | **openpi** |
| `check_occupancy_frame.py` | 좌표계 일치 — 점유 복셀 vs 실제 테이블 높이 | curobo |
| `verify_two_tier.py` | 2계층 정확성 — 거친 33 mm 대 미세 2 mm | curobo |
| `bench_two_tier.py` | 정상 상태 시간 (워밍업 필수) | curobo |

`export_frame.py` 로 npz 를 다시 만들고 `verify_two_tier.py` 를 돌려 위 수치가 그대로
재현되는 것을 확인했다 (2026-09-11).

### 남은 가정과 다음 할 일

- **trajopt 연결은 아직 안 했다.** `VoxelGrid` 를 `SceneSnapshot` 에 어떻게 물릴지, `distance()`/
  `gradient()` 인터페이스를 어떻게 맞출지는 미확인. 우리 `_esdf_clearance` 는 그 두 메서드만
  요구하므로 어댑터 한 겹이면 될 것으로 보이나 **확인 전까지는 가정**이다.
- **두 계층의 합성 규칙**을 아직 안 정했다. `SceneCfg(voxel=[coarse, fine])` 가 받아들여지는 것은
  확인했지만, 충돌 질의가 둘을 min() 으로 합치는지 아니면 각각 별도 행이 되는지는 커널을 더
  읽어야 한다.
- `RobotSegmenter` 로 E7 의 성능 문제가 사라지는지 — 별도 확인 대상.
- 이 검증은 **한 프레임** 이다. 증분 갱신(프레임 간 재사용)의 정상 상태 비용은 따로 재야 한다.

---

## 문서 정합성 검토 + 컨테이너 이관 (2026-09-11)

### `VLA_ACTION_CONDITIONED_LOCAL_TSDF_ESDF_PLAN.md` 검토

같은 `docs/` 에 있던 이 문서(내가 쓰지 않은 것)를 읽고 이 검토와 대조했다.
**방향은 일치하고, 우리가 열어 둔 질문 하나에 답을 준다.**

- 일치: cuRoboV2 참고/사용, TSDF/ESDF 해상도 분리, TO 가 거리·기울기를 질의.
- **답을 준 것**: 미세 ESDF 의 중심을 무엇으로 잡을지. 그 문서는 **action chunk 의 swept
  volume** 이라고 정했다. 방향 결정 절에 "target centroid 냐 로봇 궤적이냐, 별도 판단 필요"
  로 남겨 뒀던 항목이고, 실측도 그쪽을 지지한다 — 같은 반경에서 장애물 유지율이
  로봇 궤적 기준 75.3% 대 target 중심 46.5%.
- **차이**: 그 문서는 full-workspace dense ESDF 를 local ROI 로 **대체**한다. 검토 중
  "target 주변만 남기면 장애물 84% 를 잃는다" 는 실측이 있었지만 **그 우려는 ROI 가 swept
  volume 일 때는 적용되지 않는다** — 충돌은 로봇이 지나가는 곳에서만 일어나므로
  `swept volume ⊕ (로봇 반경 + margin + activation band + trust region)` 밖은 그 청크 동안
  닿을 수 없다. 그 문서 9항이 TO 가 ROI 를 벗어나는 경우를 이미 짚고 있다.

**구현 시 지킬 것 둘** (`CONTAINER_SETUP.md` §7 에도 적었다):
1. ROI 패딩 ≥ `max_distance`. 잘린 ESDF 는 경계 근처 거리가 과대평가된다(진짜 최근접 표면이
   ROI 밖일 수 있으므로). 우리 `esdf.py` 의 `_dirty_blocks` 가 같은 이유로 부풀린다.
2. ROI 안에서는 해상도를 올릴 것. 그 문서는 ESDF 10~20 mm 로 적었는데 **이득의 대부분이
   5 mm 에서 나온다** — 실측으로 같은 테이블 상판을 20 mm 는 33 mm, 5 mm 는 2 mm 틀리게 본다.

그 문서 7항의 "진행 기록용 md 신규 생성" 은 **이 파일이 이미 그 역할** 이므로 새로 만들지 말고
여기 이어 쓰기를 권한다 — 기록이 갈라지면 어느 쪽이 최신인지 알 수 없게 된다.

### 컨테이너 이관 준비

다음 작업(trajopt 어댑터)은 다른 컨테이너에서 한다. 인계 문서 3종:
`CONTAINER_SETUP.md` (신규) + 이 로그 + `AG3S_REVIEW_PLAN.md`.

만든 것:

| 파일 | 내용 |
|---|---|
| `ag3s/docs/CONTAINER_SETUP.md` | 이관 대상·크기, venv 2개 구성, 동작 확인 3단계, 다음 작업 배경 |
| `requirements-ag3s.txt` | CPU 파이프라인 (numpy 2.4.6 / mujoco 3.11 / casadi 3.8 / osqp 1.1.3 …) |
| `requirements-curobo.txt` | GPU venv (`cuda-core[cu12]`, warp-lang). **환경을 분리해야 하는 이유**를 파일 안에 적었다 |

**검증**: `requirements-ag3s.txt` 로 빈 venv 를 만들어(`/tmp/req_test`) 문서의 확인 절차를
그대로 돌렸다.

```
(1) export_frame      -> target centroid [0.554 0.302 0.854], 제약 구 120, 카메라 3   OK
(2) esdf_rollout 15청크 -> -28.7→-0.6 / -68.0→+1.7 / -140.3→+3.7 / -73.9→+3.2 mm
                          해소 14, feasible 8 / violated 7                          OK (기준선과 동일)
```

즉 문서에 적은 기대값은 **빈 환경에서 재현되는 것을 확인한 값**이다.

**아직 커밋하지 않았다.** 작업 트리에 미커밋 변경 18건이 있고(Step 1~4 수정 + 문서 +
`experiments/curobo/`), 그중 일부는 내가 작성하지 않은 것이다 (`trajopt/safe_policy.py`,
`serve_safe.py`, `safe_replay.json`, `attention_policy.py`, 그리고 위 VLA 계획서).
**clone 만 하면 전부 사라지므로 이관 전에 반드시 push 해야 한다** — 절차는
`CONTAINER_SETUP.md` §1 에 있다.
