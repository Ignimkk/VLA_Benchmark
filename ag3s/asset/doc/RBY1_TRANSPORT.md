# RB-Y1 transport 씬 실험 — ZED + D435i

합성 fixture가 아닌 **실제 로봇·실제 씬**에 AG3S를 태운 결과입니다. TO 이전까지가 범위이고,
산출물은 카메라별 `CollisionConstraintSet`, 그 뒤의 후보 색인, 단계별 그림입니다.

```bash
MUJOCO_GL=osmesa src/openpi/.venv/bin/python -m benchmark.ag3s.experiments.reports.rby1_transport
```

- 코드: [`benchmark/ag3s/experiments/`](../../experiments/)
- 그림: [`asset/image/rby1_transport/`](../image/rby1_transport/) — 카메라 3대 × 8단계 = 24장
- 기계 판독용 색인: [`rby1_transport_index.json`](rby1_transport_index.json)

## 씬과 센서

`src/rby1_description/models/rby1a/mujoco/model_transport.xml` (nq=66, geom 1,336). 기본 자세로
free body를 400스텝 안정화시킨 뒤 모델 자체의 `teleop` 키프레임으로 관절만 덮어씁니다 —
키프레임을 통째로 적용하면 crate와 과일이 원점으로 붕괴합니다.

| 카메라 | 위치 | fovy | 해상도 | 지정 target |
|---|---|---|---|---|
| `zed_left` | head (base 기준 z=1.45 m) | 90° | 640×480 | crate |
| `wrist_cam_l` | 왼손목 D435i | 90° | 640×480 | pear |
| `wrist_cam_r` | 오른손목 D435i | 90° | 640×480 | orange |

**Attention은 대역품입니다.** π0.5는 LIBERO의 224×224 agentview로 학습돼 있어 ZED 프레임에
들이대면 의미 없는 숫자가 나옵니다. 지정 물체의 투영 중심에 가우시안을 놓아, attention **이후의
모든 단계**(lifting, 3D grounding, 후보 생성, 제약 생성)를 검증합니다. 이 카메라들을 실제로
보는 정책이 생기면 `gaussian_attention` 하나만 교체하면 됩니다.

## 결과

### Grounding (MuJoCo GT 세그멘테이션 대비)

| 카메라 | target | IoU | purity | 점 수 | confidence |
|---|---|---|---|---|---|
| `wrist_cam_r` | orange | **1.0000** | 1.0000 | 211 | 0.903 |
| `wrist_cam_l` | pear | **0.9401** | 0.9401 | 267 | 0.868 |
| `zed_left` | crate | **0.8180** | 1.0000 | 3,319 | 0.549 |

crate는 purity 1.0(오염 없음)에 recall이 낮습니다 — 손잡이 때문에 클라우드가 조각나 일부가
별도 `unknown_geometry` 후보로 빠집니다. 버려지지는 않았습니다.

### Self-filter

| 카메라 | 로봇 점 | 제거율 | 씬 보존 |
|---|---|---|---|
| `zed_left` | 5,261 → 0 | **100.0%** | **99.8%** |
| `wrist_cam_l` / `wrist_cam_r` | 0 → 0 | — | 100% |

손목 카메라는 자기 몸을 보지 않으므로 제거할 것이 없습니다. head 카메라가 이 스테이지의
시험대이고, [`zed_left_02_self_filtered.png`](../image/rby1_transport/zed_left_02_self_filtered.png)에
팔이 사라지는 것이 보입니다.

### Support surface

바닥과 테이블 상판이 **두 평면 모두** 매 프레임 복원됩니다.

| 카메라 | 바닥 d (m) | inlier | RMS (m) | 테이블 d (m) | inlier | RMS (m) |
|---|---|---|---|---|---|---|
| `zed_left` | 0.0124 | 12,321 | 7.2e-05 | 0.8332 | 6,669 | 2.3e-03 |
| `wrist_cam_l` | 0.0119 | 7,821 | 6.3e-08 | 0.8435 | 7,645 | 2.2e-03 |
| `wrist_cam_r` | 0.0117 | 7,929 | 7.9e-08 | 0.8432 | 7,631 | 2.2e-03 |

### 충돌 후보 색인 — `zed_left`, phase=approach

| id | source_type | GT body | purity | 점 | r (m) | margin | 충돌 | 접촉 |
|---|---|---|---|---|---|---|---|---|
| -1 | support_surface | office(바닥) | 1.00 | 12,321 | — | 0.010 | ✓ | ✗ |
| -2 | support_surface | table(상판) | 0.89 | 6,669 | — | 0.010 | ✓ | ✗ |
| 5 | **target** | crate | 1.00 | 3,319 | 0.2321 | **0.020** | ✓ | ✗ |
| 0 | object | table(몸체·다리) | 1.00 | 2,894 | 0.6737 | 0.050 | ✓ | ✗ |
| 1 | object | pear+banana | 0.60 | 282 | 0.0839 | 0.050 | ✓ | ✗ |
| 2 | object | apple | 1.00 | 199 | 0.0410 | 0.050 | ✓ | ✗ |
| 3 | object | orange | 1.00 | 154 | 0.0356 | 0.050 | ✓ | ✗ |
| 4 | unknown_geometry | crate 조각 | 1.00 | 10 | 0.0191 | 0.050 | ✓ | ✗ |

제약: **41,480행, 32슬롯 중 6개 활성**, horizon 20, 제곱형.

target의 margin이 0.020(= 0.050 × APPROACH의 0.4)으로 줄어든 반면 나머지는 전부 0.050 그대로인
것이 phase 규칙이 배선돼 있다는 증거입니다.

## 단계별 latency (`zed_left` 640×480, 7프레임, warm-up 1)

| 단계 | ms |
|---|---|
| 1. scene reconstruction | 21.53 |
| 2. robot self-filter (194구) | 13.48 |
| 3. support surface | 10.28 |
| 4. attention lifting | 5.10 |
| 5. target grounding | 64.57 |
| 6. collision candidates | 17.66 |
| 7. primitive fitting | 0.51 |
| 8. constraint generation | **0.08** |
| **합계** | **133.21** |

## 이 실험이 드러낸 것

**1. URDF의 충돌 모델은 팔 전용입니다.** RB-Y1 URDF는 몸통과 팔 링크 0–5에만 캡슐 17개를
줍니다 — base, 바퀴, 손목, 그리퍼, 머리에는 없습니다. 자기충돌을 하지 않으니 시뮬레이터도
충돌 geom을 주지 않습니다. 그런데 head 카메라는 그것들을 봅니다: 필터가 로봇 점의 67.9%만
지웠고, 남은 1,645점이 **로봇에 붙어 다니는 유령 장애물**로 클러스터링됐습니다.
`UrdfSphereChain(extra_capsules=...)`로 주입 지점을 만들고, MuJoCo 시각 메시에서 바운딩 캡슐
30개를 유도해 채웠습니다(194구). 제거율 100%, 씬 보존 99.8%.

**2. self-filter와 제약은 서로 다른 로봇 모델을 원합니다.** 필터는 카메라에 보이는 전신이
필요하고(194구), 제약은 움직이는 팔만 있으면 됩니다(61구). 같은 모델을 쓰면 행 수가
41,480 → 132,000으로 뜁니다. `AG3S(robot_model=..., constraint_robot_model=...)`로 분리했습니다.

**3. `depth_max`는 z-depth를 자르지 반경을 자르지 않습니다.** 핀홀 규약대로이지만 fovy 90°
렌즈에서는 두 값이 크게 갈립니다 — 4:3 프레임의 코너 광선은 광축에서 58° 벌어져 있어,
z-depth 2.2 m 컷이 **4.17 m 떨어진 사무실 벽**을 통과시켰습니다(실측). 벽은 반지름 1.85 m
구 하나가 되어 작업공간 전체를 막습니다. `pointcloud.range_max`(반경 게이트)를 추가했습니다.

**4. 점당 Python 루프가 194구에서 터졌습니다.** `robot_sphere_mask`가 6만 점을 순회하며
257 ms를 썼습니다. 구를 순회하도록 뒤집어 **13.5 ms(19배)**가 됐고, 출력은 동일합니다.
계획대로 "정확한 baseline 먼저, 프로파일 보고 최적화"의 결과입니다.

**5. `eps < 최소 표면 간격` 한계가 실제 씬에서 재현됩니다.** banana와 pear가 하나의 후보로
병합됐습니다(113 + 169점). 중심 간 9.1 cm지만 바나나가 길쭉해 끝이 배에 3 cm(=eps) 이내로
닿습니다. apple/orange는 9.8 cm에 둘 다 뭉툭해서 분리됐습니다. **안전 방향의 손실은
아닙니다** — 병합된 구는 둘 다 포함하므로 보수적이고, 잃는 것은 정밀도입니다.

## 한계

- **Attention이 실물이 아닙니다** (위 참조). 이 실험은 attention 자체를 검증하지 않습니다.
- **벽은 여전히 구로 근사됩니다.** 수직 평면은 `normal_reference=(0,0,1)`, `max_normal_angle_deg=25`
  가 support surface에서 배제하므로 잔차 클러스터가 됩니다. 벽을 half-space로 다루려면
  `RESTRICTED_REGION`과 다중 법선 기준이 필요합니다 — 미구현입니다.
- **테이블 몸체가 반지름 0.67 m 구**입니다. 상판은 평면으로 정확히 처리되지만 다리·측면은
  잔차이고, 구-대-구 제약 형식에서는 이보다 잘 표현할 방법이 지금 없습니다(M4 한계와 동일).
- **MuJoCo depth에는 센서 노이즈가 없습니다.** 실제 D435i는 0.8 m에서 2–3 mm이고, 합성
  fixture에서 그 대역의 저하는 측정해 뒀습니다(IoU 0.998 @ 3 mm).
