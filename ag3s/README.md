# AG3S — Attention-Guided 3D Safety Scene

> **Attention은 target이 무엇인지만 정한다. 무엇이 충돌할 수 있는지는 3D 기하가 정한다.**
> Attention이 낮거나 0인 지오메트리도 물리적으로 존재하면 충돌 후보가 된다.
> Attention 점수를 장애물 탐지 임계값으로 쓰는 일은 없다.
> **어떤 접촉이 지금 허용되는지는 phase와 외부에서 주입된 접촉 컨텍스트가 정한다.**

AG3S는 (depth 또는 포인트클라우드, 카메라 내부/외부 파라미터, 로봇 상태, attention map, phase)를
받아 **`CollisionConstraintSet`** — 궤적 최적화기가 그대로 소비할 수 있는 고정 구조 CasADi 제약
사양 — 을 만듭니다.

```
VLA → Action Chunk → SEAM → Reference Trajectory ─┐
                                                  ├→ [TO: 미구현] → Safe Chunk
AG3S → CollisionConstraintSet ────────────────────┘
```

**TO는 이번 범위가 아닙니다.** 구현 위치는 `benchmark/seam_vla/refinement/collision_avoidance.py`
(현재 `refine()`이 `NotImplementedError`)이고, AG3S의 책임은 그 인터페이스 계약까지입니다.
계약이 실제로 성립함은 `tests/ag3s/test_to_contract.py`의 CasADi/IPOPT 하네스가 검증합니다.

## 구조

```
benchmark/ag3s/
├── types.py                 전 스테이지 데이터 계약, RobotCollisionModel/AttentionAdapter Protocol
├── config.py                §9 YAML 스키마 + 검증 + ablation 오버라이드
├── reconstruction.py    1.  핀홀 백프로젝션, 유효성 필터, 결정론적 보셀/캡
├── robot_filter.py      2.  주입된 로봇 모델로 self-filter (KD-tree)
├── support_surface.py   3.  결정론적 RANSAC 평면 → half-space
├── attention_lifting.py 4.  A(u,v) → 점별 attention (어떤 점도 버리지 않음)
├── target_grounding.py  5.  seed → 3D 연결성 → 클러스터 → 스코어 / 실패 상태
├── collision_candidates.py 6. 잔차 클러스터링, unknown 보존, phase 규칙, 프레임 간 트래킹
├── geometry.py          7.  sphere→capsule→box→ellipsoid 피팅 + to_spheres
├── clearance.py             phase x manipulator x source x link → 요구 clearance (fail-closed)
├── multiview.py             카메라별 처리 → base frame 융합 (provenance CSR 보존)
├── attached.py              POST_GRASP: 잡은 물체를 부모 링크에 붙인 충돌 바디로
├── constraint_builder.py 8. 고정 슬롯 CasADi 제약 (구조 1회 빌드, 이후 파라미터만)
├── to_adapter.py            CollisionConstraintSet 조립, to_casadi, refresh, warm-start 맵
├── pipeline.py              오케스트레이션 전용 (알고리즘 없음)
├── profiler.py              8단계 latency
├── visualization.py         단계별 PNG 덤프 (matplotlib lazy import)
├── asset/image/             단계별 PNG 8장 (재생성 가능)
├── asset/doc/               구현 보고서(ag3s_report.html) · 이미지 해설 · RB-Y1 실험
├── experiments/             구체적 씬 위의 실행 실험 (RB-Y1 transport)
├── robot_models/            RB-Y1 URDF sphere chain (numpy/CasADi 공용 FK)
└── configs/                 default.yaml, rby1_three_camera.yaml
```

`pipeline.py`는 조율만 합니다. 모든 스테이지는 독립적으로 단위 테스트됩니다.

## 세 가지 안전 계약

### 1. 접촉은 "제약 삭제"가 아니라 (로봇 sphere, 후보 슬롯) 마진이다

이전 구현은 `GRASP`에서 target candidate에 `collision_enabled=False`를 설정했고, 그것은 마진을
완화한 것이 아니라 **NLP에서 target의 제약 row를 통째로 제거**했습니다. 잡는 동안 왼팔·전완·몸통이
물체를 관통해도 solver가 볼 수 있는 제약이 없었습니다.

지금은 target이 모든 phase에서 슬롯을 유지하고, `GRASP`에서 바뀌는 것은 `(로봇 sphere, 슬롯)`
마진 행렬의 한 칸뿐입니다.

```python
margin = policy.lookup(
    phase=Phase.GRASP,
    candidate_source=SourceType.TARGET,
    robot_link="ee_finger_r1",
    active_manipulators=[Manipulator.RIGHT],
    target_grounded=True,
)   # → contact_margin (기본 0.0)
```

**0은 "끔"이 아닙니다.** row는 그대로 남아 `‖p_r − p_c‖² ≥ (r_r + r_c + 0)²`를 강제합니다 —
닿는 것은 허용, 관통은 금지. 그 외 모든 링크(전완·몸통·반대팔·미지 링크)는 모든 phase에서 full
margin을 유지합니다. 알 수 없는 phase / source / link / manipulator 조합은 전부 full margin으로
**fail closed** 합니다. `OBJECT` · `UNKNOWN_GEOMETRY` · `OVERFLOW` · support surface는 어떤
경우에도 완화되지 않습니다.

**Active manipulator는 주입 입력입니다.** `Phase.GRASP`는 잡고 있다는 사실만 말할 뿐 어느 손인지는
말하지 않습니다. 아무도 지명되지 않으면 아무것도 완화되지 않습니다.

### 2. 지오메트리는 조용히 사라지지 않는다

유효한 non-support 지오메트리는 반드시 셋 중 하나로 귀결합니다.

| | |
|---|---|
| 1 | 정상 `CollisionCandidate` |
| 2 | `SourceType.OVERFLOW` 보수 집합체 — **모든 원본 점을 포함** |
| 3 | `PipelineStatus.GEOMETRY_INCOMPLETE` + `ConstraintValidity.INCOMPLETE` |

`ConstraintValidity`는 로그 문자열이 아니라 `CollisionConstraintSet`의 필드입니다. 빈 `VALID`
집합은 "봤고 아무것도 없다"이고 `INCOMPLETE`는 "무엇이 있는지 말할 수 없다"입니다 — 둘을 같게
보이게 하는 것이 로봇이 모델링되지 않은 벽으로 들어가는 방식입니다. **무엇을 할지는 AG3S가
정하지 않습니다.** 정지·유지·직전 인증 씬 재사용은 전부 호출자의 정책입니다.

`max_points`도 인덱스 선택이 아니라 **보셀을 키워서** 지킵니다(`cap_strategy: voxel`). 점유된 모든
보셀이 대표점을 남기므로 미표현 영역이 최종 보셀 크기 이내로 유계입니다.

불확실성 예산(`geometry.perception_uncertainty`)은 **primitive 반지름에만** 더합니다. `d_safe`에도
더하면 같은 1 cm를 두 번 세어 우회를 소리 없이 두 배로 만듭니다.

### 3. 세 카메라는 하나의 씬이다

카메라마다 `AG3S`를 따로 돌리면 crate가 head 뷰에서 `id=0`, wrist 뷰에서 table이 `id=0`이 됩니다.
지금은 카메라별로 **자기 캡처 시각의 로봇 상태**로 재구성·self-filter한 뒤 base frame 보셀 격자에서
융합하고, 이후 스테이지는 하나의 클라우드를 봅니다.

```
T_base_cam(t) = FK(q(t_capture), mount_link) @ T_link_cam
```

`q_now` 하나를 세 대에 재사용하지 않습니다 — 손목 카메라는 팔과 함께 움직이므로 그것은 클라우드를
어긋나게 하고, 동시에 self-filter를 어긋나게 해 팔 자신의 점이 그리퍼에 붙은 유령 장애물이 됩니다.

융합점은 자신을 만든 모든 observation(카메라·픽셀·타임스탬프·attention)을 CSR 블록으로 유지합니다.
대표점은 **실제 관측점**이며 합성 centroid가 아닙니다. Attention은 관측들에 대한 `max`로 융합합니다.

카메라 dropout · stale state · skew는 전부 `ConstraintValidity`를 낮추고 metrics에 남습니다.
**AG3S 내부에 정지 FSM은 없습니다.**

## POST_GRASP — attached collision geometry

잡은 뒤 물체는 세계의 장애물이 아니라 움직이는 로봇의 일부입니다.

```python
ag3s.attach(target, robot_state=q_grasp, parent_link="ee_right",
            allowed_contact_links=["ee_right", "ee_finger_r1", "ee_finger_r2"])
```

- 존재는 **인지에 의존하지 않습니다.** occlusion·grounding 실패·카메라 dropout은 attached를 제거하지
  않습니다. `detach()`만이 제거합니다. AG3S는 grasp 성공을 추론하지 않습니다.
- 접촉은 **allowlist**입니다. 잡고 있는 손가락만 면제되고 전완·몸통·반대팔은 계속 제약됩니다.
  인식할 수 없는 링크명은 면제되지 않습니다(fail closed).
- attach/detach는 **파라미터 변경**입니다. row 수·sparsity·solver 객체가 전후로 동일합니다.
  부모 링크는 심볼릭 식의 일부이므로 빌더 생성 시 `attached_parent_links=(...)`로 미리 선언합니다.
  선언하지 않은 호출자는 비용이 0입니다(row 수가 이전과 동일).

## 스테이지 순서에 대한 한 가지 의도적 차이

명세는 §3.3(grounding) → §3.4(평면 피팅) 순이지만, 파이프라인은 **평면을 먼저** 맞춥니다.
평면 피팅은 target에 의존하지 않는 반면 grounding은 평면에 의존하기 때문입니다 — 물체는 테이블
위에 서 있어 어떤 실용적인 `eps`에서도 연결성이 테이블 전체로 범람합니다(실측: 41,745점짜리
단일 클러스터가 씬 전체를 삼킴). 방법의 변경이 아니라 실행 시점의 선택입니다.

## 설치

이 워크스페이스의 기존 venv(`src/openpi/.venv`)에는 casadi만 없었으므로, 그것만 추가하면 됩니다.

```bash
uv pip install --python src/openpi/.venv/bin/python casadi
```

AG3S 전용 venv를 새로 만들 경우 (numpy 2.x에서도 동작합니다):

```bash
uv venv --python 3.11 benchmark/ag3s/.venv
uv pip install --python benchmark/ag3s/.venv/bin/python -r benchmark/ag3s/requirements.txt
```

`uv` 대신 표준 pip을 쓸 경우:

```bash
python3.11 -m venv benchmark/ag3s/.venv
benchmark/ag3s/.venv/bin/pip install -r benchmark/ag3s/requirements.txt
```

시각화가 필요 없으면 `matplotlib`을, 테스트를 돌리지 않으면 `pytest`를 빼도 됩니다 —
둘 다 코어 import 경로에 없습니다.

## 실행

```bash
# 단위 테스트 (422개)
src/openpi/.venv/bin/python -m pytest tests/ag3s/ -q

# TO 계약 하네스 — IPOPT 수렴 로그 포함
src/openpi/.venv/bin/python -m pytest tests/ag3s/test_to_contract.py -q -s

# e2e smoke + 8단계 latency 표
src/openpi/.venv/bin/python -m benchmark.ag3s.pipeline --frames 12 --profile

# 단계별 시각화 8종 → benchmark/ag3s/asset/image/ (해설: asset/doc/FIGURES.md)
src/openpi/.venv/bin/python -m benchmark.ag3s.visualization

# RB-Y1 transport 씬, 3대 카메라 융합 (ZED head + D435i 양 손목)
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.rby1_transport \
    --json benchmark/ag3s/asset/doc/rby1_transport_fused_index.json
```

## 사용법

```python
from benchmark.ag3s import AG3SConfig
from benchmark.ag3s.pipeline import AG3S
from benchmark.ag3s.robot_models import load_rby1

ag3s = AG3S(AG3SConfig.from_yaml("benchmark/ag3s/configs/default.yaml"),
            robot_model=load_rby1())

# 단일 카메라 — 호환 경로, 이전과 동일
constraint_set = ag3s.process(
    attention_map=attention, depth=depth, pointcloud=None,
    camera_intrinsics=K, T_base_cam=T_base_cam,
    robot_state=q, phase="approach",
    active_manipulators=[Manipulator.RIGHT],   # 접촉 권한은 주입 입력
)
```

3대 카메라를 하나의 씬으로 융합하려면:

```python
from benchmark.ag3s.types import CameraID, CameraObservation, Manipulator

observations = [
    CameraObservation(
        camera_id=CameraID.HEAD, depth=head_depth, camera_intrinsics=K_head,
        mount_link="link_head_2", T_link_cam=T_head_cam,   # hand-eye
        robot_state=q_at_head_capture, timestamp=t_head,   # 이 카메라 자신의 상태
        attention_map=head_attention, image_hw=head_depth.shape,
    ),
    CameraObservation(camera_id=CameraID.LEFT_WRIST,  mount_link="link_left_arm_6",  ...),
    CameraObservation(camera_id=CameraID.RIGHT_WRIST, mount_link="link_right_arm_6", ...),
]
constraint_set = ag3s.process_multi(observations, phase="approach",
                                    active_manipulators=[Manipulator.RIGHT])

assert constraint_set.geometry_certified   # 아니면 호출자가 정책을 결정
```

관측마다 `robot_state`와 `timestamp`를 따로 싣는 것이 핵심입니다. `process_multi`에는
`robot_state` 인자가 없습니다 — 세 대에 하나의 현재 `q`를 적용하는 것이 이 메서드가 막으려는
버그이기 때문입니다.

TO 쪽에서는:

```python
from benchmark.ag3s.to_adapter import to_casadi, refresh

fragment = to_casadi(constraint_set.constraints, Q)   # 셋업 시 1회
# nlp = {"x": ..., "f": ..., "g": fragment["g"], "p": fragment["p"]}
# 이후 매 프레임: solver(..., p=refresh(constraint_set, ag3s.builder))
```

구조는 한 번만 빌드되고 프레임마다 파라미터 값만 갱신됩니다. 이것이 warm-start와 실시간성의
전제입니다 — 측정치로 구조 빌드 6.1 ms(1회) 대 프레임당 제약 생성 0.06 ms.

## 의존성

numpy, scipy, casadi, PyYAML만 사용합니다. scikit-learn / open3d / pinocchio / hpp-fcl은 이
워크스페이스에 없으며 추가하지 않았습니다 — DBSCAN, 연결 성분, RANSAC, 보셀, KD-tree는
numpy/scipy로 직접 구현했습니다. matplotlib은 `visualization.py` 안에서만 lazy import됩니다.

## RB-Y1 3-카메라 융합 실측 (MuJoCo transport 씬)

| 항목 | 값 |
|---|---|
| 융합 | 3대 60,566점 → 12,528점 (12 mm 보셀), skew 30 ms |
| 카메라별 | head 25,850 / left_wrist 17,334 / right_wrist 17,382 |
| URDF vs MuJoCo 링크 위치 오차 | `link_head_2` / `link_left_arm_6` / `link_right_arm_6` **모두 0.0 mm** |
| self-filter | head 6,771점 제거, 잔존 로봇점 0. 손목 카메라는 로봇 픽셀을 아예 보지 않음(실측 0/307,200) |
| 융합 효과 | 23개 후보 중 **12개가 2대 이상의 카메라 점을 함께 보유** — 뷰마다 id 공간이 따로였을 때는 구조적으로 불가능 |
| 평면 | 바닥 d=0.0128 m (rms 3.07e-4), 테이블 d=0.8353 m (rms 2.41e-3) |
| 제약 | 41,480 row(팔 61구) → 손끝 capsule 포함 105구에서 71,400 row |
| 파라미터 갱신 | 0.054 ms/frame, 마진 행렬(61×32) 0.024 ms |

`(sphere, slot)` 마진 도입으로 파라미터는 202 → 2,122개로 늘었지만 **row 수·sparsity·solver 객체는
불변**이고 프레임당 비용은 0.06 ms 수준입니다.

정직하게 남겨두는 사실 두 가지:

- **crate가 후보 4개로 쪼개집니다.** 이것은 융합 문제가 아니라 클러스터링 결과입니다 — 속이 빈
  상자의 벽은 개구부를 가로질러 Euclidean 연결이 되지 않고, 과일을 분리할 만큼 작은 `eps`는
  벽도 분리합니다. 융합이 고친 것은 "같은 물체가 뷰마다 다른 id"이지 "한 물체가 한 클러스터"가
  아닙니다.
- **attention은 여전히 대역품입니다.** 세 카메라에 서로 다른 target을 지시했더니 `max` 융합이
  가장 강한 신호(right_wrist의 orange)를 골랐습니다. 예상된 동작이며, 이 실험이 검증하는 것은
  attention *이후* 단계입니다.

## 알려진 한계

- **정확도 수치는 전부 합성 fixture 및 MuJoCo 기준**입니다. 실측 캡처가 없습니다.
- **hand-eye calibration 오차·depth noise·실제 timestamp skew는 실기에서만 검증 가능**합니다.
  MuJoCo에서는 URDF FK와 시뮬레이터가 정확히 일치하므로(0.0 mm) 정합 오차가 0입니다.
- **RB-Y1 URDF는 torso + arm 0–5의 collision capsule만 제공**합니다. 접촉 허용 링크(`ee_*`)에
  sphere가 없으면 clearance 정책이 완화할 대상이 없어 grasp를 표현할 수 없습니다. 실험은
  `extra_capsules`로 손끝 capsule을 주입해 우회합니다 (`tests/ag3s/test_robot_models.py`가 이
  사실을 고정).
- **`eps` 운용 범위**: `voxel_size < eps < 최소 물체 간격`. 3 cm 기본값은 2 cm 간격 물체를 병합합니다
  (Euclidean 연결성의 본질적 한계이며 버그가 아님).
- **box/ellipsoid는 제약 목적으로 bounding sphere로 축약**됩니다. 제약 형식이 구-대-구이기
  때문이며, pose/dimensions는 온전히 전달되므로 해당 거리 함수를 지원하는 TO가 나오면 활용
  가능합니다.
- **제약 행 수 = horizon × robot_spheres × (slots + planes)**. RB-Y1 전체 61구 체인은 H=20에서
  약 39,000행이므로, 실배치에는 link-filtered 모델 주입이 필요합니다.
