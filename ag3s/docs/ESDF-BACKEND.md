# TSDF → ESDF 충돌 표현 (두 번째 backend)

**상태: 구현 완료, 기본값 아님.** `collision_backend: primitive` 가 그대로 기본이라 기존 결과와
테스트는 한 줄도 달라지지 않는다. `esdf` 또는 `both` 로 켠다.

## 왜

primitive backend 는 후보 하나를 도형 하나로 줄인다. 조밀하고 볼록한 물체에는 맞지만 두 가지로
깨진다 ([OPEN-geometry-representation.md](OPEN-geometry-representation.md) 의 측정값):

| 문제 | 증상 |
|---|---|
| 비등방성 — 테이블 잔여 슬랩 41 mm × 1.0 m × 0.8 m | 경계 구 **749 mm**, 과잉 부피 54배 |
| 중공 — 크레이트 | 구 229 mm 가 내부를 삼켜 **"바구니에 넣기"가 불가능** |
| 합쳐서 | 테이블 위 작업 평면의 **100%가 막힘** |

ESDF 는 그 축약을 하지 않는다. 관측된 표면을 복셀 격자에 그대로 적분하고 어느 점에서든 가장
가까운 표면까지의 거리를 답한다. **도형을 고르지 않으므로 도형이 틀릴 일이 없다.**

## 제약 형태

```
d_esdf(p(q)) - collision_radius - safety_margin >= 0
```

로봇 구 하나당 한 행이다. primitive 는 (로봇 구 × 슬롯) 이 행 수였는데 ESDF 는 슬롯이 없다 —
필드는 *점*에 대해 답하므로 열거할 것이 없다. RB-Y1 에서 61구 × 32슬롯 = 1,952 행이 61 행이 된다.

기울기는 `∇h = ∇d_esdf(p) · ∂p/∂q` 다. 거리장은 아이코널 방정식을 만족하므로 자유 공간에서
`|∇d_esdf| = 1` 이고, 측정값도 1.000 이다.

## 켜는 법

```yaml
# AG3S
collision_backend: esdf        # primitive | esdf | both
esdf:
  voxel_size: 0.010            # 5 / 10 / 20 mm 비교용 축
  max_distance: 0.5
  unknown_policy: free         # free | occupied
  exclude_target: auto         # auto | always | never
  incremental: true
  emit_candidates: false       # 아래 "esdf 는 primitive 를 돌리지 않는다" 참고
```

```yaml
# trajopt
collision:
  backend: esdf                # primitive | esdf | both
  esdf_margin: 0.05
```

### `esdf` 는 primitive 경로를 돌리지 않는다

`collision_backend: esdf` 면 후보 생성(클러스터링 + primitive fitting)을 **아예 건너뛴다**.
충돌 제약을 만드는 것은 **attention + ESDF** 이기 때문이다 — attention 이 target 을 지목하고,
`CollisionConstraintSet.target` 이 그것을 싣고, 거리장이 그에 맞춰 파인다. 그 어느 단계도 도형
근사를 필요로 하지 않는다. 전에는 후보를 만들어 놓고 `scene_from_constraint_set` 에서 전부 끄고
있었다. 계산만 버린 것이 아니라, 쓰지도 않는 표현을 산출물에 실어 "무엇이 제약을 만들었는가" 를
흐렸다.

측정(20 mm 복셀, 한 프레임, 3회 워밍업 후):

| | 합계 | primitive 경로 | ESDF |
|---|---|---|---|
| `emit_candidates: true` | 422.2 ms | 29.5 ms | 27.9 ms |
| 기본 (`false`) | **393.9 ms** | 0.0 ms | 27.7 ms |

두 경우 모두 `상태 ok`, target 있음, 필드 있음이고 후보 수만 9 → 0 이다.

**앞선 초안의 304 ms 절감 수치는 틀렸다.** 그때는 지지면 추출이 꺼진 설정에서 쟀고, 그러면
`ground_target` 이 exclude_mask 를 못 받아 영역 성장이 테이블을 타고 번지면서 클러스터링이 씬
전체를 훑는다. 지지면 추출을 정상화하면 primitive 경로는 29.5 ms 다. 절감은 실재하지만 작다 —
이 변경의 근거는 속도가 아니라 **무엇이 제약을 만드는가의 일관성**이다.

`emit_candidates: true` 는 그래서 최적화 스위치가 아니라 진단용 탈출구다. 같은 프레임에서 두
표현을 나란히 보고 싶을 때만 켠다.

### 지지면: 추출과 소비는 별개다

여기서 한 번 헷갈렸으니 적어 둔다. 지지면 **추출**(`support_surface.enabled`)은 어느 모드에서도
켜 두어야 한다. `ground_target` 이 그 평면 마스크를 `exclude_mask` 로 받지 못하면 영역 성장이
테이블을 타고 번져 씬 전체가 한 클러스터가 되고, target 을 아예 못 찾는다(`no_target`).

모드가 정하는 것은 그 다음 두 가지다:

| | `--support-surfaces plane` | `--support-surfaces field` (기본) |
|---|---|---|
| 평면 추출 | 켬 | 켬 |
| `esdf.exclude_support_surfaces` | `true` — 필드에서 파냄 | `false` — 거리장에 남김 |
| `trajopt.collision.use_support_planes` | `true` — half-space 행을 읽음 | `false` — 읽지 않음 |

`field` 가 기본인 이유는 half-space 가 **무한**하기 때문이다. 테이블 상판 평면 하나가 그 아래
공간 전부를 금지하고, RB-Y1 에서는 그것이 바닥에 놓인 베이스와 바퀴 구 57개를 명목상 위반으로
만든다. 거리장에는 관측된 상판만 들어가므로 그 문제가 없다.

`both` 는 삭제하지 않고 남긴 이유 그 자체다 — 두 표현이 **같은 씬**을 설명하므로 한 QP 안에
나란히 넣어 행 단위로 비교할 수 있다. 실행 간 비교보다 훨씬 강하다.

## 세 상태 점유 — 이 backend 에서 가장 중요한 설계

점유 / 자유 / **미관측**. 어느 카메라도 보지 못한 복셀은 자유가 아니다.

* 미관측을 자유로 두면 로봇이 못 본 공간을 지나간다.
* 점유로 두면 테이블 뒤가 전부 막혀 아무것도 못 한다.

AG3S 방식대로 **조용히 정하지 않고 보고한다**: `unknown_policy` 로 고르되 기본값은 `free` 이고,
미관측 비율이 `EsdfField.stats` 에 실려 `unknown_report_threshold` 를 넘으면 노트로 나온다.

기본값이 `free` 인 것은 **primitive backend 와 같은 성질**이기 때문이다 — 관측되지 않은 기하는
후보를 만들지 않으므로 그쪽도 암묵적으로 자유다. ESDF 가 이것을 나쁘게 만들지 않고, 처음으로
**셀 수 있게** 만든다. 이 씬에서 측정된 미관측 비율은 **72%** 다.

## 정확도 — 두 가지 오차가 서로 반대 방향이다

합성 구(반지름 200 mm)를 세 방향에서 본 씬으로 측정했다 (`tests/ag3s/test_esdf.py`).

**이산화 편향은 정확히 복셀 반 칸이고 보수적이다.** 표면을 점유 복셀의 *중심*에 찍으므로 거리가
참값보다 작게 나온다.

| 복셀 | 최대 오차 | 부호 |
|---|---|---|
| 5 mm | −2.5 mm | 보수적 (여유를 더 요구) |
| 10 mm | −5.0 mm | 보수적 |
| 20 mm | −10.0 mm | 보수적 |

제약이 `d - r - margin >= 0` 이므로 작게 나오는 것은 안전한 방향이다. **다만 20 mm 복셀은 그
편향이 10 mm 여서 50 mm 여유거리의 5분의 1을 이미 쓴다 — 해상도 선택이 곧 예산 선택이다.**

**관측되지 않은 표면은 반대로 낙관적이다.** 세 카메라 사이 대각선처럼 어느 카메라에도 정면으로
잡히지 않는 표면은 `UNKNOWN` 이라 점유가 아니고, 거리장은 더 먼 점유 복셀까지를 재므로 참값보다
**크게** 나온다. 이산화 편향은 복셀을 줄이면 사라지지만 **이쪽은 그렇지 않다 — 줄일 수 있는
것은 관측뿐이다.** 이 성질을 못박는 테스트가
`test_unobserved_surface_makes_the_field_optimistic_not_conservative` 다.

## target 제외

`exclude_target: auto` 는 **단계의 접촉 허가를 따른다** — `ClearancePolicy` 가 읽는 것과 같은
입력이다. 필드에서 target 을 파내는 것은 필드판 여유거리 완화이므로, 둘이 서로 다른 규칙으로
움직이면 안 된다.

**파내는 것은 제거가 아니라 이관이다.** target 의 기하는 `CollisionConstraintSet.candidates` 에
여전히 후보로 남아 있고(5단계에서 확인한 성질), 필드에서 빠지는 것은 "잡아야 하는 물체를
장애물로도 세지 않는다"는 접촉 정책의 결과일 뿐이다. `CollisionConstraintSet.esdf` 는
`candidates` **옆에** 있지 대신 있지 않다 — ESDF 는 "이 기하가 target 이다"를 표현할 수 없고
"가장 가까운 표면이 얼마나 머냐"만 답하므로, 후보를 지우면 여유거리 정책이 읽을 것이 사라진다.

## 국소 갱신 — 정확하지만, 이 씬에서는 이득이 없다

설계는 이렇다. 격자를 한 번 할당하고 프레임마다 같은 버퍼에 적분한다. 거리 변환은 **점유 상태가
실제로 바뀐** 복셀이 든 블록에서만 다시 하고, 각 블록을 `max_distance` 만큼 부풀린다. 그 여유가
정확히 `max_distance` 이므로 **부분격자 안의 `max_distance` 이하 거리는 전역 재계산과 같은 값**이
나온다. 측정된 차이는 **0.0000 mm** 다 (`test_local_update_reproduces_the_global_transform_exactly`).

두 번 고쳤다. 처음엔 "TSDF 가 닿은 복셀"을 dirty 로 잡아서 카메라가 보는 부피 전체가 매 프레임
dirty 가 됐고, 그다음엔 세 상태를 그대로 비교해서 `UNKNOWN → FREE` 전환(거리장을 전혀 바꾸지
않는다)까지 dirty 로 셌다. 지금은 EDT 가 실제로 보는 **이진 점유 집합**을 비교한다.

**그런데 이 씬에서는 이득이 없다.** 정직하게 적는다.

| 복셀 | max_distance | pad(복셀) | 최초 | 증분 | 블록 | 재계산 |
|---|---|---|---|---|---|---|
| 10 mm | 0.50 m | 51 | 2.33 s | 1.78 s | 0 | 100% |
| 10 mm | 0.10 m | 11 | 1.94 s | 2.12 s | 0 | 100% |
| 20 mm | 0.50 m | 26 | 0.39 s | 0.42 s | 0 | 100% |
| 20 mm | 0.10 m | 6 | 0.60 s | 0.40 s | 2 | 85% |

이유는 두 가지다. 첫째, 프레임당 바뀌는 복셀은 10 mm 에서 **822개**뿐인데 그것이 작업 공간에
**흩어져** 있어서 블록 수가 많다. 둘째, 각 블록의 padding 이 `max_distance / voxel_size` 이고
10 mm · 0.5 m 에서 51 복셀이라, 16³ 블록 하나를 부풀리면 118³ = 1.6 M 이 되어 격자(4.3 M)의
3분의 1이다. 그래서 되돌림 규칙(다시 계산할 복셀이 격자의 절반을 넘으면 전역)이 거의 항상 걸린다.

되돌림이 있으므로 **국소 갱신이 전역보다 느려지지는 않는다.** 이득이 나는 조건은 명확하다 —
변화가 공간적으로 뭉쳐 있고 `max_distance / voxel_size` 가 `block_voxels` 에 비해 작을 때다.
위 표의 마지막 줄이 그 문턱을 겨우 넘는 지점이다.

## primitive 와 나란히 — 이 backend 를 만든 이유가 숫자로

RB-Y1 씬 프레임 0, `approach`, 20 mm 복셀, `both` backend. 로봇 충돌 구 194개 각각에서 두 표현이
말하는 여유거리를 비교했다. 양쪽 모두 같은 지지면 half-space 행을 포함한다.

| | 위반 구 | 최소 여유 |
|---|---|---|
| primitive + 평면 | **185 / 194** | −1124 mm |
| ESDF + 평면 | **71 / 194** | −1124 mm |

**ESDF 만 위반이라고 보는 구는 0개다.** ESDF 가 위반이라 부른 71개는 primitive 도 전부 위반이라
부른다. primitive 는 거기에 **114개를 더** 얹는다.

여유거리 차이(primitive − ESDF)는 중앙값 **−281 mm**, 5~95 백분위 −377 ~ +0 mm 다. 즉 primitive
경로는 로봇 주변에 **평균 281 mm 두께의 없는 장애물**을 만든다. 그것이 테이블 잔여의 749 mm 구와
크레이트의 229 mm 구다.

이것이 `OPEN-geometry-representation.md` 가 "작업 공간의 100%가 막힌다"고 적은 것의 다른 얼굴이다.
ESDF 는 그 114개를 없애면서 **하나도 새로 만들지 않는다** — 과대근사를 걷어내는 것이지 과소근사로
바꾸는 것이 아니다.

(최소 여유 −1124 mm 가 양쪽에서 같은 것은 그것이 **공유된 평면 행**에서 나오기 때문이다. 테이블
상판의 half-space 는 무한히 뻗으므로 상판보다 낮은 모든 것이 위반이 된다 — 지지면 표현의 별개
한계이고 이 backend 가 만든 것이 아니다.)

## 지지면은 필드에서 파낸다

지지면은 이미 half-space 행으로 나가고 그쪽 여유거리는 10 mm 다. 같은 테이블을 필드에도 넣으면
**두 backend 가 같은 기하에 대해 서로 다른 여유거리를 요구한다** — 평면 행은 10 mm, 필드는 50 mm.

측정: 파내지 않으면 로봇 구 25개가 위반인데 그중 **23개가 지지면**이다 (바닥에 놓인 base 5 · 바퀴
6, 테이블 상판 근처 손가락 12). 파내면 **14개**로 줄고 남는 것은 크레이트 근처의 손가락뿐이다.

파내도 잃는 것은 없다. 지지면은 `CollisionConstraintSet.support_surfaces` 에 그대로 있고 어느
backend 에서든 평면 행이 된다. 기본값 `exclude_support_surfaces: true`.

## 로봇도 필드에서 뺀다

`robot_mask` 를 파이프라인이 자동으로 만든다 — 점군에 쓰는 것과 같은 `robot_sphere_mask` 를
이미지 공간에 적용한다. 없으면 팔이 장애물 필드에 들어가 **로봇이 자기 자신과 충돌한다고
보고한다**: 배선 전 194개 중 130개 위반, 최소 −158 mm. 배선 후 25개, −34 mm.

## 지연 시간 — 실시간 예산에 한참 못 미친다

제어 주기는 **66.7 ms** (15 Hz) 다.

| 복셀 | 격자 | 메모리 | 프레임당 |
|---|---|---|---|
| 5 mm | 301×360×320 (34.7 M) | 139 MB | **17.1 s** |
| 10 mm | 151×180×160 (4.3 M) | 17 MB | **1.8 s** |
| 20 mm | 76×90×80 (0.5 M) | 2 MB | **0.4 s** |

10 mm 는 예산의 **27배**, 20 mm 도 **6배**다. 내역은 TSDF 적분 0.85 s + 전역 EDT 0.86 s
(10 mm 기준)이고, **적분 쪽이 EDT 만큼 비싸다** — 완벽한 증분 EDT 를 만들어도 절반밖에 못 줄인다.

줄일 수 있는 것은 세 가지다.

1. **작업 상자.** 기본값은 앞쪽으로 치우친 상자다 (`default_bounds`). 사방 1.5 m 로 두면
   14.8 M 복셀에 6.5 s 다. 고정 베이스에 조작 대상이 전부 앞에 있으므로 뒤를 자르는 것이
   이 과제에서는 안전하지만, 베이스가 도는 과제에서는 넓혀야 하고 그때 비용이 다시 든다.
2. **복셀 크기.** 20 mm 는 0.4 s 이지만 이산화 편향이 10 mm 가 된다.
3. **적분 방식.** 지금은 모든 복셀을 카메라에 투영한다. 깊이 픽셀에서 표면 근처 띠만 갱신하는
   방식(voxblox 계열)은 O(전체 복셀) 대신 O(픽셀 × 절단폭/복셀) 이다. 대신 자유 공간을 파내지
   못하므로 `unknown_policy: occupied` 를 쓸 수 없게 된다. **아직 하지 않았다.**

## 로봇은 필드에 넣지 않는다

`CameraDepth.robot_mask` 로 로봇 픽셀을 빼면 그 광선은 **자유가 아니라 미관측**이 된다.
두 가지 이유에서 옳다. 팔은 자기 자신에 대한 장애물이 아니고(자기 충돌은 로봇 모델과
`attached.self_collision_mask` 가 따로 다룬다), 팔을 필드에 넣으면 팔이 자기 앞을 막는다.
그리고 팔은 **움직인다** — 정적인 씬에서 매 프레임 바뀌는 것이 팔뿐이면 국소 갱신이 의미를
가질 수 있는데, 팔을 남기면 변화가 격자 전체에 퍼진다.

## 파일

| 파일 | 역할 |
|---|---|
| `esdf.py` | `VoxelGrid` · `TsdfVolume` · `EsdfField` · `EsdfBuilder` · `carve` |
| `config.py` | `EsdfConfig`(`emit_candidates` 포함), `AG3SConfig.collision_backend` |
| `types.py` | `CollisionConstraintSet.esdf` (후보 **옆에**) |
| `pipeline.py` | 8b 단계. 후보 생성 *뒤*에 — target 파냄이 접촉 규칙을 따라야 하므로 |
| `trajopt/linearize.py` | `SceneSnapshot.esdf`, `ESDF_SLOT`, 선택·선형화의 세 번째 블록 |
| `trajopt/config.py` | `CollisionBackendConfig` |
| `tests/ag3s/test_esdf.py` | 23개 — 정확도·세 상태·국소 갱신·target 파냄·설정 |
| `tests/trajopt/test_esdf_backend.py` | 14개 — 두 backend 일치·행 라벨·기울기·`both`·해결 |

## 아직 하지 않은 것

* **작업 공간 막힘 비율의 before/after.** 로봇 구 기준 비교는 했지만(185 → 71),
  `OPEN-geometry-representation.md` 가 쓴 "테이블 위 평면 격자에서 막힌 비율 100%" 를 같은
  방식으로 다시 재지는 않았다. 6단계 스크립트에 backend 축을 붙이면 된다.
* **띠 기반 적분.** 위 3번.
* **여유거리의 단계 의존성.** primitive 경로는 `ClearancePolicy` 에서 (로봇 구 × 슬롯) 마진을
  받는데 ESDF 는 슬롯이 없어 하나의 보수적인 수(`esdf_margin`)를 쓴다. 같은 수를 재사용하면
  단계별 완화를 잃거나 파지 완화를 전 링크에 적용하게 되고, **두 번째가 여유거리 정책이 막으려는
  바로 그 실수다.** 복셀마다 의미 라벨을 갖기 전에는 이 상태로 둔다.
