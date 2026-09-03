# AG3S 검증 Step 1~3 — 발표자료 통합 소스

이 파일 하나로 슬라이드를 만들 수 있도록, 흩어져 있던 근거 텍스트와 수식·알고리즘 원문을
한곳에 모은 것이다. **여기 없는 숫자는 존재하지 않는 숫자다.**

수집 원본 (2026-09-02 기준, 전부 직접 확인):

| 구분 | 파일 |
|---|---|
| 시스템 정의 | `benchmark/ag3s/__init__.py`, `benchmark/ag3s/README.md` |
| 파이프라인 순서 | `benchmark/ag3s/pipeline.py` (모듈 docstring) |
| 검증 개요 | `benchmark/ag3s/docs/README.md` |
| Step 1 | `docs/step-01-attention.md` · `.json` · `docs/step-01-server-prompt.md` |
| Step 2 | `docs/step-02-backprojection.md` · `.json` |
| Step 3 | `docs/step-03-lifting.md` · `.json` |
| 수식/알고리즘 | `reconstruction.py`, `attention_lifting.py`, `support_surface.py`, `experiments/attention_report.py`, `experiments/backprojection_report.py`, `experiments/lifting_report.py` |
| 설정 | `configs/default.yaml`, `configs/rby1_three_camera.yaml` |

---

## 0. 번호 체계 주의 — 발표에서 반드시 구분할 것

이 저장소에는 "step" 번호가 두 개 있고, 혼동하면 발표가 무너진다.

| 체계 | 의미 | 출처 |
|---|---|---|
| **파이프라인 stage 1~8** | AG3S가 매 프레임 실행하는 처리 단계 | `pipeline.py` docstring |
| **검증 step 0~7** | 파인튜닝된 실제 정책 출력으로 각 단계를 ground truth와 대조한 실험 | `docs/step-0N-*.md` |

이 발표는 **검증 step 1~3**을 다룬다. 대응 관계:

| 검증 step | 검사 대상 | 파이프라인 위치 |
|---|---|---|
| Step 0 | 관측 기록이 GPU forward pass를 쓸 값어치가 있는가 | 파이프라인 이전 (입력 검사) |
| **Step 1** | π0.5가 프롬프트가 지목한 물체를 보는가 | 파이프라인의 **attention 입력원** (stage 4의 입력) |
| **Step 2** | 깊이 → 3D 점이 실제 물체 위치에 놓이는가 | **stage 1** scene reconstruction |
| **Step 3** | 2D attention이 옳은 3D 점에 붙는가 | **stage 4** attention lifting |
| Step 4~6 | grounding / separation / geometry | stage 5~7 (문서 완성, 이번 발표 범위 밖) |

---

## 1. 시스템 정의 — AG3S는 무엇인가

### 1.1 한 문장 정의 (`__init__.py`)

> Attention decides **what the target is**. 3D geometry decides **what can collide**.
> Geometry with low or zero attention is still a collision candidate if it physically exists;
> an attention score is never used as an obstacle-detection threshold.

한국어(README 서두):

> **Attention은 target이 무엇인지만 정한다. 무엇이 충돌할 수 있는지는 3D 기하가 정한다.**
> Attention이 낮거나 0인 지오메트리도 물리적으로 존재하면 충돌 후보가 된다.
> Attention 점수를 장애물 탐지 임계값으로 쓰는 일은 없다.
> **어떤 접촉이 지금 허용되는지는 phase와 외부에서 주입된 접촉 컨텍스트가 정한다.**

### 1.2 입출력 계약

입력: `(depth 또는 point cloud, 카메라 내부 파라미터 K, 외부 파라미터 T_base_cam, 로봇 상태 q,
attention map, phase)`

출력: **`CollisionConstraintSet`** — 궤적 최적화기(TO)가 그대로 소비할 수 있는 **고정 구조**
CasADi 제약 사양.

### 1.3 전체 시스템에서의 위치 (`__init__.py` / README 동일)

```
VLA → Action Chunk → SEAM → Reference Trajectory ─┐
                                                  ├→ [TO: 미구현] → Safe Chunk
AG3S → CollisionConstraintSet ────────────────────┘
```

**TO는 이번 범위가 아니다.** 구현 위치는 `benchmark/seam_vla/refinement/collision_avoidance.py`
(현재 `refine()`이 `NotImplementedError`)이고, AG3S의 책임은 그 **인터페이스 계약**까지다.
계약이 실제로 성립함은 `tests/ag3s/test_to_contract.py`의 CasADi/IPOPT 하네스가 검증한다.

---

## 2. 파이프라인 8단계와 데이터 흐름

### 2.1 실행 순서 (`pipeline.py` docstring 원문 번역)

```
1. scene reconstruction     depth 또는 point cloud → base frame의 필터링된 점군
2. robot self-filter        주입된 충돌 모델로 로봇 자신의 점 제거
3. support surface          RANSAC 평면; 그 마스크가 stage 5와 6으로 전달
4. attention lifting        A(u,v) → 점별 attention, 아무것도 버리지 않음
5. target grounding         seed → 3D 연결성 → 클러스터 → 점수화, 또는 실패 상태
6. collision candidates     잔차 클러스터링, unknown 보존, phase 규칙, 프레임 간 트래킹
7. primitive fitting        (6 안에서 실행; latency는 따로 측정)
8. constraint generation    고정 슬롯 CasADi 파라미터
```

**Phase는 모든 수준에서 주입 인자다. AG3S는 phase를 절대 추론하지 않는다.**

`pipeline.py`는 조율만 한다 — "No algorithm lives here." 모든 stage가 독립 모듈이고 독립적으로
단위 테스트된다. 이 제약을 문서에 적어 둔 이유도 원문에 있다: *"the first 'just a small special
case' added here is the point at which a stage stops being independently testable."*

### 2.2 명세와 순서가 다른 곳 한 군데 (발표에서 좋은 소재)

명세는 §3.3 target grounding → §3.4 평면 피팅 순인데, 파이프라인은 **평면을 먼저** 맞춘다.

- 평면 피팅은 target에 의존하지 않는다. 반대로 grounding은 평면에 의존한다.
- 물체는 테이블 **위에** 서 있고, 실용적인 어떤 `eps`에서도 유클리드 연결성이 테이블 전체로
  범람한다.
- **실측: 41,745점짜리 단일 클러스터가 씬 전체를 삼켰다.**
- 방법의 변경이 아니라 **실행 시점의 선택**이다.

### 2.3 데이터 흐름 한 줄 요약

```
depth×3 + K + T_base_cam + q + attention + phase
   → [1] 점군(base frame)      : 픽셀 (u,v) 보존
   → [2] 로봇 점 제거
   → [3] 평면(half-space) 분리 : n·p ≥ d + margin
   → [4] attention 점군        : 점 개수 불변
   → [5] target 하나 + 상태
   → [6,7] 충돌 후보 + primitive
   → [8] CollisionConstraintSet (고정 슬롯, 파라미터만 갱신)
```

---

## 3. 세 가지 안전 계약 (README §"세 가지 안전 계약")

발표에서 Step 1~3의 "왜 이렇게 검증하는가"를 지탱하는 근거다. 특히 **계약 2**가 Step 3 검사 A의
근거다.

### 계약 1 — 접촉은 "제약 삭제"가 아니라 (로봇 sphere, 후보 슬롯) 마진이다

이전 구현은 `GRASP`에서 target에 `collision_enabled=False`를 설정했고, 그것은 마진 완화가 아니라
**NLP에서 target의 제약 row를 통째로 제거**하는 것이었다. 잡는 동안 왼팔·전완·몸통이 물체를
관통해도 solver가 볼 수 있는 제약이 없었다.

지금은 target이 모든 phase에서 슬롯을 유지하고, `GRASP`에서 바뀌는 것은 `(로봇 sphere, 슬롯)`
마진 행렬의 **한 칸**뿐이다. **0은 "끔"이 아니다** — row는 그대로 남아

```
‖p_r − p_c‖² ≥ (r_r + r_c + 0)²
```

를 강제한다. 닿는 것은 허용, 관통은 금지. 알 수 없는 phase / source / link / manipulator 조합은
전부 full margin으로 **fail closed**.

### 계약 2 — 지오메트리는 조용히 사라지지 않는다

유효한 non-support 지오메트리는 반드시 셋 중 하나로 귀결한다.

| | |
|---|---|
| 1 | 정상 `CollisionCandidate` |
| 2 | `SourceType.OVERFLOW` 보수 집합체 — **모든 원본 점을 포함** |
| 3 | `PipelineStatus.GEOMETRY_INCOMPLETE` + `ConstraintValidity.INCOMPLETE` |

빈 `VALID` 집합은 "봤고 아무것도 없다"이고 `INCOMPLETE`는 "무엇이 있는지 말할 수 없다"이다 —
**둘을 같게 보이게 하는 것이 로봇이 모델링되지 않은 벽으로 들어가는 방식이다.**

`max_points`도 인덱스 선택이 아니라 **보셀을 키워서** 지킨다(`cap_strategy: voxel`). 점유된 모든
보셀이 대표점을 남기므로 미표현 영역이 최종 보셀 크기 이내로 유계다.

불확실성 예산(`geometry.perception_uncertainty`)은 **primitive 반지름에만** 더한다. `d_safe`에도
더하면 같은 1 cm를 두 번 세어 우회를 소리 없이 두 배로 만든다.

### 계약 3 — 세 카메라는 하나의 씬이다

카메라마다 AG3S를 따로 돌리면 crate가 head 뷰에서 `id=0`, wrist 뷰에서는 table이 `id=0`이 된다.
지금은 카메라별로 **자기 캡처 시각의 로봇 상태**로 재구성·self-filter한 뒤 base frame 보셀
격자에서 융합하고, 이후 stage는 하나의 클라우드를 본다.

```
T_base_cam(t) = FK(q(t_capture), mount_link) · T_link_cam
```

`q_now` 하나를 세 대에 재사용하지 않는다 — 손목 카메라는 팔과 함께 움직이므로 그것은 클라우드를
어긋나게 하고, 동시에 self-filter를 어긋나게 해 팔 자신의 점이 그리퍼에 붙은 **유령 장애물**이
된다. Attention은 관측들에 대한 `max`로 융합한다. ← **이것이 Step 3 발견 E가 문제 삼는 지점이다.**

---

## 4. 검증 방법론과 실험 셋업

### 4.1 왜 단계별로 검증하는가 (`docs/README.md` 원문)

> AG3S는 8단계 파이프라인이고, 각 단계는 앞 단계의 출력을 신뢰한다. 마지막 단계(제약 집합)만
> 보고 "된다/안 된다"를 판정하면, 실패했을 때 어느 단계가 원인인지 알 수 없고 성공했을 때도
> 운이 좋았던 것인지 알 수 없다. 그래서 **각 단계의 출력을 그 단계의 ground truth와 직접
> 대조**한다. MuJoCo 씬이므로 모든 단계에 정확한 ground truth가 존재한다는 것이 이 검증의
> 전제이자 최대 강점이다.

### 4.2 실험 조건 (Step 1~3 전부 동일한 단 하나의 기록)

| 항목 | 값 |
|---|---|
| 기록 | `outputs/rby1_atomic_infer/ag3s_step1/ag3s_records/run_0002` |
| 프롬프트 | `put the apple in the basket` |
| target body | `apple` |
| 로봇 | RB-Y1 (14-DoF 양팔) |
| 카메라 | head ZED (`zed_left`) + 양 손목 D435i (`wrist_cam_l`, `wrist_cam_r`) |
| 체크포인트 | `.../pi05_rby1_atomic_lora/rby1_atomic_basket_14d_v2_30k_20260825/29999` |
| 기록 프레임 | 44 (스텝당 `.npz`, 총 3.7 MB) |
| 채점 프레임 | Step 1: **28** (16개 제외 — target 200 px 미만) |
| Ground truth | MuJoCo 세그멘테이션 · geom 자세 · 메시 정점 |

**하나의 기록을 1~7단계가 전부 재사용한다.** 단계마다 롤아웃을 다시 돌리면 매번 다른 씬을
검증하게 되어 단계 간 비교가 불가능해진다.

### 4.3 기록 형식 — 기본값은 qpos만 저장한다

깊이와 세그멘테이션은 저장하지 않는다. MuJoCo는 결정론적이라 `qpos`가 로봇·크레이트·과일을 모두
고정하므로 `TransportScene`이 나중에 정확히 같은 깊이/세그멘테이션/내부·외부 파라미터를
재생성한다 — **AG3S가 이미 테스트를 통과해 온 바로 그 코드 경로로.** 저장하면 파일이 약 50배가
되고, 더 나쁘게는 기록 시점의 렌더링과 분석 시점의 렌더링이 아무도 모르게 어긋날 수 있다.

**정책 이미지 3장만은 그대로 저장한다.** 재생성하면 안 되는 유일한 것이다 — 네트워크에 실제로
들어간 텐서이고, attention 검증은 "일치할 것으로 기대되는 재렌더링"이 아니라 **모델이 본 픽셀
위에서** 이뤄져야 한다.

`--record-depth`를 켜면 카메라별 depth를 **uint16 밀리미터**로 저장한다 — 실제 depth 카메라가
내보내는 형식이고 `PointCloudConfig.depth_scale = 0.001`이 존재하는 이유다. 스텝당 약 0.13 MB
(44스텝 +6 MB). 재생 depth와 1 mm 이내로 일치하며, 그 1 mm가 정확히 양자화 폭이다.

### 4.4 좌표 규약 — 모든 단계가 여기에 의존한다

`render_cam(..., match_rby1_dataset=True)`는 **299×224(4:3)로 렌더한 뒤 224×224로 찌그러뜨린다**
— crop도 pad도 아닌 순수 가로 압축. 따라서 **정규화 좌표는 보존된다**: 정책 열 `u_p`는 정규화 열
`u_p/224`이고, AG3S가 재구성하는 640×480(역시 4:3) 프레임에서도 같은 정규화 열이다.
**attention은 offset 없이 단순 정규화 스케일링으로 점군에 올라간다.**

### 4.5 토큰 배치 — 추측이 아니다

`AlohaInputs`가 base → left wrist → right wrist 고정 순서로 이미지 dict를 만들고
`Pi0.embed_prefix`가 그 순서로 이어붙이므로:

| 정책 키 | MuJoCo 카메라 | AG3S CameraID | prefix 토큰 |
|---|---|---|---|
| `cam_high` | `zed_left` | head | 0–255 |
| `cam_left_wrist` | `wrist_cam_l` | left_wrist | 256–511 |
| `cam_right_wrist` | `wrist_cam_r` | right_wrist | 512–767 |
| (언어) | — | — | 768– |

**이 표가 Step 3 발견 E의 원인 설명이다** — (L8, h2)는 head 블록 [0,256)에서 고른 셀이고,
같은 (층, 헤드)가 왼쪽 손목 블록 [256,512)에서도 target을 찾는다는 보장은 어디에도 없다.

### 4.6 왜 attention 추출만 GPU 서버인가

`localhost:8123`은 VS Code SSH 포워딩이고 실제 정책 서버는 GPU 서버에 있다. websocket 프로토콜은
`actions`만 돌려주므로 **실행 중인 서버에서 attention을 꺼낼 방법이 없다.** attention은
`Pi0Config.return_attn_probs=True`로 prefix/suffix를 직접 구동해야 나오고, 그러려면 체크포인트가
있는 곳에서 forward를 돌려야 한다. 서버 재시작은 필요 없다 — 기록된 관측에 대해 **오프라인으로
한 번 돌리는 별도 스크립트**다.

추출 결과 크기: 약 81 MB (44프레임 × Euler 3 × pooling 3 × 18층 × 8헤드 × 3카메라 × 256패치).

### 4.7 Step 0 — 기록 검사 (`record_check`)

**결과가 아니라 입력 검사다.** GPU forward pass를 쓸 값어치가 있는 기록인지, 채점 가능한 프레임이
몇 개인지, 가림이 어느 구간인지를 서버에 보내기 **전에** 로컬에서 확인한다.

- 판정: **run_0002 검사 통과** — 44프레임 중 **28프레임 채점 가능**
- 그림 `fig0_record_sanity.png`: 위 패널은 제어 스텝별 target 가시 픽셀 수 + 채점 하한(점선) +
  가려진 구간(음영), 아래 패널은 같은 시간축의 16×16 패치 점유 히트맵.

---

## 5. Step 1 — attention map

### 5.1 질문과 동기

**질문.** 파인튜닝된 `pi05_rby1_atomic_lora` 정책은 프롬프트가 지목한 물체를 실제로 보는가?

AG3S가 attention을 쓰는 곳은 **단 하나** — 재구성된 클러스터 중 어느 것이 target인지 **이름을
붙이는 일**뿐이다. attention이 틀려도 파이프라인의 안전 성질은 전부 유지되지만, 틀리면 파이프라인이
**쓸모가 없다.** 그래서 이것이 가장 먼저 재야 할 값이고, 실제 attention이 합성 대역품
`mujoco_source.gaussian_attention`(target의 알려진 투영 위치에 놓은 가우시안)을 대체할 수 있는지를
결정하는 값이다.

### 5.2 방법 — 1296개 셀 스윕

| 축 | 값 | 개수 |
|---|---|---|
| 층 (layer) | 0–17 | 18 |
| 헤드 (head) | 0–7 | 8 |
| Euler step (flow-matching 시점) | [0, 4, 9] | 3 |
| query pooling | ['mean', 'first', 'last'] | 3 |
| noise seed | [0] | 1 |
| **조합** | | **1296** |

attention 블록은 층당 16×16 = 256 패치. 한 프레임의 attention 맵은 `(16,16)`.

### 5.3 지표 정의 (`experiments/attention_report.py: score_maps`)

프레임마다 attention 맵을 합이 1이 되도록 정규화한 뒤, 물체 `i`에 대해:

```
mass_i       = Σ_(패치 p) A_norm(p) · coverage_i(p)      # 물체 i에 떨어진 attention 비율
area_share_i = Σ_p coverage_i(p) / 256                   # 물체 i의 화면(패치) 점유 비율
lift_i       = mass_i / area_share_i                     # 균등 attention 대비 배율
score_i(β)   = mass_i / (area_share_i)^β                 # β 스윕용 점수
hit_rate(β)  = P[ argmax_i score_i(β) == target ]        # 물체 단위 적중률
peak_on_target = P[ argmax 패치가 target을 5% 이상 덮음 ]
advantage    = mean( mass_target − max_(i≠target) mass_i )
```

각 항의 의미:

- `mass_i` — 그 물체 위에 실제로 떨어진 attention의 **양**. 큰 물체가 유리하다.
- `area_share_i` — 그 물체가 화면에서 차지하는 **면적 비율**. 크기 보정항.
- `lift_i` — **균등 attention이 줄 밀도의 몇 배인가.** `lift ≈ 1.0`이면 raw mass가 얼마든
  그 헤드는 **아무것도 선택하지 않은 것**이다.
- `β` — 크기 보정의 강도. **β = 0은 raw mass**(큰 크레이트가 이김), **β = 1은 밀도**(작은 과일이
  이김). 이 정의는 KNOWS 논문 Eq. (2)–(3)이 하는 것과 같다.
- `peak_on_target` — "가장 뜨거운 패치 하나가 target을 포함하는가". 16×16 격자를 480×640에
  올리면 패치 하나가 30×40 px이고, 배(pear)는 총 380 px, 그 밑의 테이블이 모든 패치에서 이긴다.
  그래서 "지배"가 아니라 **"존재"**를 묻는 것이 이 해상도에서 답할 수 있게 만드는 유일한 방법이다.

**자격 게이트.** `competitor_mask`에 든 물체만, 그리고 최소 `min_body_px = 100` 픽셀 이상 보일
때만 이길 수 있다. *"A body reduced to a few stray pixels can post an enormous density score off a
single patch, which is noise wearing the shape of a result."*

경쟁 집합은 `['crate', 'apple', 'banana', 'orange', 'pear']`. **테이블을 제외한 이유**: AG3S
7단계가 테이블을 support surface로 뽑아내고 4단계 target grounding이 애초에 후보로 보지 않는다 —
**파이프라인이 결코 고르지 않을 집합을 상대로 채점하면 아무 단계도 내리지 않는 판단을 재게 된다.**
다만 물체별 표에는 테이블을 남겨 두어, 테이블만 쳐다보는 헤드가 있다면 보이도록 했다.

### 5.4 왜 peak-on-target으로 순위를 매기는가 (발표의 핵심 논리)

| 지표 | 스윕 최댓값 | 최댓값에 닿은 셀 | 고유값 개수 |
|---|---|---|---|
| peak-on-target | 1.000 | **3 / 1296** | 20 |
| hit β=0 | 0.321 | 31 / 1296 | 10 |
| hit β=0.5 | 0.929 | 6 / 1296 | 27 |
| hit β=1 | 1.000 | **105 / 1296** | 29 |

- **β = 0은 헤드가 아니라 씬이 상한을 정한다.** 1296개 셀 전부가 같은 값에서 멈춘다. 이길 수 있는
  프레임은 **파지 전** 프레임뿐이다 — target이 크레이트 안으로 들어간 뒤에는 target을 덮는 어떤
  attention 덩어리도 크레이트를 더 많이 덮으므로, 헤드가 무엇을 하든 raw mass는 target을 고를 수
  없다.
- **β = 1은 포화한다.** 105개 셀이 정확히 1.000에 닿아 좋은 헤드끼리를 구분하지 못한다.
- **peak-on-target에는 두 문제가 다 없다.** 실질적으로 상한이 없고, 이후 target grounding이 실제로
  소비할 정보에 가장 가깝다.

→ 슬라이드 메시지: **"지표를 결과에 맞춰 고르지 않았다. 지표가 왜 실패하는지를 먼저 보였다."**

### 5.5 판정 규칙

베이스라인 둘:

- **uniform (모델 없음)** — "target이 그냥 화면에서 제일 큰 것 아닌가"에 답한다.
- **head-average (18×8 전체 평균)** — "헤드를 고른 것이 무엇을 벌어줬나"에 답한다.

게이트: 선택된 헤드는 **어떤 β에서도 두 베이스라인보다 나빠서는 안 되며**, **peak-on-target에서
head-average를 0.15 이상** 앞서고 **lift > 2**여야 한다.

**마진을 peak-on-target에만 요구한 것은 의도된 선택이다.** β = 1에서 head-average가 이미 1.000이라
어떤 헤드가 낼 수 있는 최대 마진이 0.000이다 — **아무것도 통과할 수 없는 게이트는 증거가 아니라
고장난 규칙이다.**

### 5.6 결과 — **PASS**

| 맵 | hit β=0 | hit β=0.5 | hit β=1 | peak-on-target | target mass | target lift | advantage |
|---|---|---|---|---|---|---|---|
| head-average (18×8) | 0.036 | 0.179 | 1.000 | 0.107 | 0.009 | 5.34 | −0.037 |
| uniform (모델 없음) | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 1.00 | −0.035 |
| **best: L8 h2** (Euler 0, `last`) | **0.286** | **0.929** | **1.000** | **1.000** | **0.141** | **91.82** | **−0.091** |

핵심 한 줄: **L8 h2는 균등 attention이 사과에 줄 밀도의 91.82배를 싣고, 채점한 28프레임 전부에서
가장 뜨거운 패치가 사과 위에 있었다 (peak-on-target 1.000).**

상위 5개 셀 (1296개 중):

| # | 헤드 | Euler | pooling | peak-on-target | target lift |
|---|---|---|---|---|---|
| 1 | L8 h2 | 0 | last | 1.000 | 91.82 |
| 2 | L8 h2 | 4 | last | 1.000 | 80.00 |
| 3 | L8 h2 | 4 | mean | 1.000 | 65.14 |
| 4 | L8 h2 | 0 | mean | 0.857 | 71.31 |
| 5 | L8 h2 | 9 | last | 0.821 | 52.34 |

→ 상위 5개가 전부 **같은 (층, 헤드) = (8, 2)**. 샘플링 선택(Euler step, pooling)은 그 다음 문제다.

물체별 attention 분포 (채점 28프레임 평균):

| body | 평균 가시 픽셀 | attention mass | lift | argmax 획득 프레임 |
|---|---|---|---|---|
| crate | 11042 | 0.225 | 6.35 | 0 |
| **apple (target)** | **431** | **0.141** | **91.82** | **28** |
| banana | 205 | 0.011 | 14.84 | 0 |
| orange | 491 | 0.002 | 1.52 | 0 |
| pear | 303 | 0.001 | 0.90 | 0 |
| table | 24513 | 0.288 | 3.53 | 0 |
| shelf | 0 | 0.000 | 0.00 | 0 |

### 5.7 반드시 함께 설명할 것 — advantage가 음수인 이유

`advantage = mass_target − max(다른 경쟁자 mass)` 가 −0.091로 **음수**다. 이것은 **예상된 것이며
실패가 아니다.** 크레이트가 target의 약 **26배** 픽셀을 차지하므로 raw mass는 더 많이 가져가면서
*픽셀당* attention은 훨씬 적게 받는다. `lift`가 재는 것이 정확히 그 비교이고 (crate 6.35 vs
apple 91.82), **두 열은 반드시 함께 읽어야 한다.**

### 5.8 이 결과가 아래 단계에서 무엇을 고치는가 (인계)

AG3S 3단계(=파이프라인 stage 4)의 attention 소스는

```
attention[frame, 0, "last", 8, 2, camera]
```

`GridAttentionAdapter`가 이미 16×16 그리드를 받으므로 **어댑터 코드는 바뀌지 않는다.** 합성
대역품 `gaussian_attention`을 실제 모델 attention으로 교체 가능.

---

## 6. Step 2 — 3D back-projection

### 6.1 질문과 동기

**질문.** 깊이에서 나온 3D 점이 실제로 물체가 있는 곳에 놓이는가?

1단계는 attention이 옳은 물체를 가리키는지 확인했다. **그 이름표가 붙는 대상은 이 점들이고, 점이
틀린 곳에 있으면 이후 모든 단계 — 클러스터링, primitive 근사, 여유거리 — 가 틀린 기하 위에서
정확하게 계산될 뿐이다.**

검사 프레임: 44개 중 균등 간격 **8개** × 카메라 3대 = 24회.
역투영 코드: `benchmark.ag3s.reconstruction.backproject` — **AG3S 본체 코드 그대로.**

### 6.2 수식 — 핀홀 역투영 (`reconstruction.py: backproject`)

```
X = (u − cx) · D / fx
Y = (v − cy) · D / fy
Z = D
P_base = T_base_cam · P_cam
```

`u`는 열(column), `v`는 행(row) — 내부 파라미터 규약과 일치. `D = depth · depth_scale`.

**유효성 필터가 여기 있는 이유:** depth가 0이거나 비유한(non-finite)이거나 `[depth_min, depth_max]`
밖인 픽셀은 **여기서 버려지고 클라우드에 들어가지 않는다** — 원문: *"a zero-depth pixel
back-projects to the camera origin, which would put a phantom obstacle inside the robot."*
(0-depth 픽셀은 카메라 원점으로 역투영되어 **로봇 안에 유령 장애물**을 만든다.)

**이미지 뒤집기가 없다.** `benchmark/knows_vla`는 robosuite가 관측을 미리 뒤집어 저장하기 때문에
`flip_row`/`flip_col`이 필요하지만, AG3S는 내부 파라미터가 기술하는 방향 그대로 depth를 받고
수정은 GT로 검증 가능한 호출자에게 맡긴다 — **추측은 선택지가 아니다.**

**`range_max` (반경 게이트):** `depth_max`는 z-깊이를 자르지 실제 거리를 자르지 않는다. 90° 렌즈에서
둘은 크게 벌어진다 — **2.5 m z-컷이 프레임 모서리에서는 4.2 m 거리의 점을 통과시키고(실측),
사무실 벽을 1.85 m 구로 작업공간에 끌어들이기에 충분하다.** 그래서 명시적 반경 게이트가 있다:

```
‖P_cam‖ ≤ range_max
```

### 6.3 검사 설계 — 4개가 서로 다른 실패 방식에 대응한다

**한 가지만 재면 어느 것이 깨졌는지 알 수 없다.**

| 검사 | 무엇이 깨지면 걸리는가 | 기준 |
|---|---|---|
| **A. 재투영 왕복** | 역투영 **식 자체** | < 1e-6 px |
| **B. 지지면 평면** | 카메라 **회전 규약** | 기울기 < 0.5°, 높이 오차 < 5 mm |
| **C. 물체 표면 오차** | **깊이 스케일 · 내부 파라미터** | 95 백분위 < 10 mm |
| **D. 카메라 간 정합** | 손목 카메라 **외부 파라미터 (FK 사슬)** | 융합 시 두께 증가 < +5 mm |

### 6.4 검사 A — 산술과 규약을 분리한다

역투영한 점을 같은 `K`, `T_base_cam`으로 다시 투영한다:

```
p_cam = R_base_camᵀ (P_base − t_base_cam)
u' = fx · p_cam.x / p_cam.z + cx
v' = fy · p_cam.y / p_cam.z + cy
err = max(|u' − u|, |v' − v|)
```

**이 검사는 규약이 옳은지는 전혀 묻지 않는다** — 규약이 통째로 틀려 있어도 왕복은 완벽하게 닫힌다.
**그래서 유용하다:** A가 통과하고 B가 실패하면 문제는 **카메라 규약**이고, A가 실패하면 **식 자체**다.
이 둘을 구분하지 못하면 이후 디버깅이 추측이 된다.

**결과: 24회 최대 오차 6.82e-13 px** — 배정밀도 반올림 수준.

### 6.5 검사 B — 지지면 평면 (회전 오차가 가장 먼저 드러나는 곳)

테이블 픽셀만 역투영해 지배 평면을 맞추고, 법선과 base +z가 이루는 각, 실제 상판 높이와의 차이,
잔차를 잰다. 넓고 평평한 면이라 회전 오차가 크게 증폭된다.

**두 번 좁혀야 했다 — 두 번 다 역투영은 옳고 검사가 틀렸다:**

1. `table`은 하나의 body이지만 **geom이 5개** — 상판 하나 + 다리 넷. body 라벨로 고르면 수직인
   다리가 들어온다. **첫 판: 기울기 37.55°.**
2. 상판 geom 자체가 **두께 40 mm의 box**라 **옆면**도 보이고, 그 점들은 윗면보다 최대 40 mm 아래에
   있으면서 수직이다. **두 번째 판: 기울기 0.84°, 높이 −9 mm.**

**해결:** 상판 geom의 점에 **AG3S 자신의 RANSAC 평면 추정기**(`support_surface.fit_plane_ransac`)를
건다. 점 수가 압도적인 윗면을 지배 평면으로 잡고 옆면을 이상점으로 버린다. **이것은 파이프라인
stage 3(support surface)이 실제로 쓰는 바로 그 코드이므로, 여기서 통과한다는 것은 그 stage가 같은
씬에서 옳은 평면을 잡는다는 뜻이기도 하다.**

MuJoCo geom에서 읽은 실제 상판 윗면 높이: **0.820000 m** (base 프레임)

| 프레임 | 상판 geom 점 | 평면 내점 | 기울기 (°) | 측정 높이 (m) | 높이 오차 (mm) | RMS 잔차 (mm) | 95p 잔차 (mm) |
|---|---|---|---|---|---|---|---|
| 0 | 21476 | 19658 | 0.1516 | 0.819433 | −0.57 | 0.78 | 0.51 |
| 1 | 21389 | 19557 | 0.1278 | 0.819717 | −0.28 | 0.81 | 0.39 |
| 2 | 20629 | 19012 | 0.1151 | 0.819884 | −0.12 | 0.76 | 0.29 |
| 3 | 16336 | 14903 | 0.1324 | 0.819698 | −0.30 | 0.78 | 0.40 |
| 4 | 18416 | 16887 | 0.1362 | 0.819642 | −0.36 | 0.82 | 0.45 |
| 5 | 19089 | 17362 | 0.1047 | 0.820046 | **+0.05** | 0.74 | 0.22 |
| 6 | 21547 | 19703 | 0.1639 | 0.819269 | **−0.73** | 0.83 | 0.59 |
| 7 | 24682 | 22883 | 0.1557 | 0.819391 | −0.61 | 0.76 | 0.53 |

**요약: 기울기 0.10–0.16°, 높이 오차 −0.73 ~ +0.05 mm.**

그림 읽는 법 주의: 잔차를 최댓값이 아니라 95 백분위로 그린 이유는 **최댓값이 RANSAC 내점
임계값(8 mm)에 눌려 매 프레임 같은 값이 나오기 때문**이다 — 측정값이 아니라 설정값이 되어 아무것도
알려주지 않는다.

### 6.6 검사 C — 중심이 아니라 표면과 비교한다

각 점에서 그 물체의 **실제 표면**까지의 거리를 잰다.

**중심 좌표와 비교하지 않는 이유:** 카메라는 앞면만 본다. 보이는 점들의 무게중심은 카메라 쪽으로
대략 반지름만큼 치우쳐 있으므로, **역투영이 완벽해도 중심과의 거리는 0이 되지 않는다.** 그 편차를
오차로 보고하면 **없는 문제를 만들어내게 된다.**

정답 표면을 geom 종류별로 다르게 만든다:

- **과일 (메시, 볼록)** — 정점의 볼록 껍질. 점에서 껍질 표면까지 부호 없는 거리:
  ```
  signed = max_f (n_f · p + d_f)        # f = 껍질의 각 면
  dist   = |signed|
  ```
- **크레이트 (상자 36개로 만든 속이 빈 통)** — 볼록 껍질을 쓰면 **열린 입구가 덮여 내부가 통째로
  안쪽**이 되고, 안쪽 벽의 점이 크레이트 반너비만큼 떨어진 것으로 계산된다. **역투영이 완벽해도
  수십 mm의 가짜 오차가 난다.** 그래서 점-상자 표면 거리를 정확히 계산해 geom별 최솟값을 취한다:
  ```
  local = R_boxᵀ (p − t_box)
  q     = |local| − half_extents
  dist  = ‖max(q, 0)‖              (밖일 때)
        = −max(q)                  (안일 때, 가장 가까운 면까지)
  ```
  **속이 빈 형상이 자연히 처리되고 근사가 아니다.**

**카메라별로 나눠 싣는 이유:** 한 카메라의 외부 파라미터만 틀린 경우가, 합쳐 놓으면 나머지 뒤에
숨는다.

| body | 카메라 | 점 수 | 중앙값 (mm) | 95p (mm) | 최대 (mm) |
|---|---|---|---|---|---|
| crate | zed_left | 85393 | 1.01 | 1.34 | 1.86 |
| crate | wrist_cam_l | 502825 | 0.33 | 0.57 | 0.86 |
| crate | wrist_cam_r | 510961 | 0.27 | 0.38 | 0.80 |
| apple | zed_left | 2511 | 0.91 | 4.29 | 11.00 |
| apple | wrist_cam_l | 55277 | 1.64 | **7.23** | 10.01 |
| apple | wrist_cam_r | 11926 | 0.40 | 0.99 | 4.38 |
| banana | zed_left | 1652 | 2.81 | 5.47 | 6.69 |
| banana | wrist_cam_l | 15695 | 2.84 | 5.81 | 7.04 |
| orange | zed_left | 3931 | 0.71 | 1.79 | 1.93 |
| orange | wrist_cam_l | 321 | 0.35 | 0.47 | 0.49 |
| orange | wrist_cam_r | 724 | 0.36 | 0.46 | 0.49 |
| pear | zed_left | 1799 | 0.74 | 1.86 | 6.97 |
| pear | wrist_cam_l | 5441 | 0.29 | 0.79 | 4.29 |
| pear | wrist_cam_r | 33486 | 0.33 | 4.33 | 8.01 |

**요약: crate 0.38–1.34 mm, 과일 0.46–7.23 mm (95 백분위). 전부 기준 10 mm 이내.**

### 6.7 검사 D — 외부 파라미터 사슬 전체

손목 카메라의 외부 파라미터는 `T_base_cam(t) = FK(q, mount_link) · T_link_cam`을 타고 온다.
마운트 링크나 `T_link_cam`이 틀리면 **head 카메라만으로는 A·B·C가 전부 통과하면서 손목만
어긋난다.**

**무게중심을 비교하면 안 된다.** 카메라마다 물체의 **다른 면**을 보므로 무게중심은 정상적으로
다르다 — 이 씬에서 크레이트의 카메라별 무게중심은 **최대 14 cm** 벌어지는데, 그것은 보정 오차가
아니라 **시점 차이**다. (이 보고서의 첫 판이 그 값을 오차로 보고했다.)

**대신 "세 카메라를 합쳤을 때 표면이 두꺼워지는가"를 본다.** 외부 파라미터가 서로 맞으면 융합
오차는 개별 카메라 오차 중 최댓값 근처에 머물고, 하나라도 틀어져 있으면 그 위로 뛴다. **각 카메라가
다른 면을 보는 것은 이 지표를 흔들지 않는다 — 그 면들이 같은 표면 위에 있는지만 묻기 때문이다.**

| body | 합친 카메라 | 융합 95p (mm) | 개별 최댓값 (mm) | 증가 (mm) |
|---|---|---|---|---|
| crate | 3대 | 0.92 | 1.34 | **−0.43** |
| apple | 3대 | 6.98 | 7.23 | −0.25 |
| banana | 2대 | 5.78 | 5.81 | −0.03 |
| orange | 3대 | 1.70 | 1.79 | −0.09 |
| pear | 3대 | 3.95 | 4.33 | −0.38 |

**증가량이 전부 음수 — 융합이 오히려 표면을 얇게 만든다.** 세 카메라의 외부 파라미터가 서로
일치한다는 뜻이다.

### 6.8 판정 — **PASS** (A/B/C/D 전부)

### 6.9 인계

옳은 위치의 3D 점 + 픽셀 대응 `(u,v)`가 보존된 클라우드. `uv`는 **Step 3 lifting이 쓰라고**
reconstruction이 남겨둔 것이다.

---

## 7. Step 3 — attention lifting (2D → 3D)

### 7.1 질문과 동기

**질문.** 1단계에서 옳다고 확인한 attention이, 2단계에서 옳다고 확인한 점들 중 **옳은 점**에
붙는가?

**두 입력이 모두 옳아도 잇는 방식이 틀리면 4단계는 틀린 점을 seed로 삼는다.** 실패 방식은 셋이다 —
픽셀 대응이 어긋나거나, 정규화가 순위를 바꾸거나, 점이 조용히 사라지거나.

사용한 셀: **L8 h2, Euler 0, `last`** (1단계가 고른 것을 그대로 이어받음).
검사 프레임: 20개 × 카메라 3대. 순위 지표는 target이 200 px 이상 보이는 프레임만
(cam_high 12, cam_left_wrist 17, cam_right_wrist 10).

### 7.2 수식 — lifting 파이프라인 (`attention_lifting.py`)

```
(1) 셀 선택   raw (L, Hd, T, P) → 토큰축 평균 → [layer, head] → (P,) → (G,G),  G=√P=16
(2) 업샘플    (16,16) → (H,W)   bilinear 또는 nearest
(3) 표본추출  값 = pixel_map[v, u]  (점마다, cloud.uv 사용)
(4) 정규화    normalize_attention(값)
출력: AttentionPointCloud(cloud, normalized, raw)   ← 개수는 len(cloud), 언제나
```

**bilinear 업샘플의 픽셀 중심 정렬 (`_resample`):**

```
fy = clip((y + 0.5) · h / H − 0.5, 0, h−1)
fx = clip((x + 0.5) · w / W − 0.5, 0, w−1)
```

`+0.5 … −0.5` 시프트가 필요한 이유: **16×16 격자를 480×640에 펼치면 배율이 30–40배이고,
corner-aligned 보간은 모든 패치 중심을 반 패치, 즉 약 15 픽셀 편향시킨다.**

**정규화 네 모드 (`normalize_attention`) — 전부 순위를 보존한다고 주장:**

```
percentile (기본) : lo,hi = percentile(a, [5,99]);  clip((a−lo)/(hi−lo), 0, 1)
minmax            : lo,hi = min(a), max(a)         (교과서판, ablation 기준선)
softmax           : p = softmax((a − max a)/T);  p / max(p)
none              : 통과 (이미 정규화된 VLA용)
```

- `percentile`이 기본인 이유: **뜨거운 패치 하나가 맵 전체를 지배하는 것을 막는다**
  (minmax는 막지 못한다).
- `softmax`에서 `/max(p)`로 다시 스케일하는 이유: 그것이 없으면 값이 ~1/N이 되어 **어떤 절대
  임계값도 무의미해진다.**
- 퇴화 입력(`hi − lo ≈ 0`)에서 나누지 않고 **0을 반환하는 이유**: 그래야 `target_grounding`이
  수치 잡음으로 target을 만들어내는 대신 **`NO_ATTENTION`을 보고**한다.

**정규화값과 raw값을 둘 다 보관하는 이유:** 정규화는 상단에서 손실이 있다(percentile clipping이
동점을 만든다). 그런데 `target_grounding`은 **모호하지 않은 peak**가 필요하다.

**seed 추출 (4단계의 개념이지만 여기서 계산됨):**

```
seed = { i : attention_i ≥ percentile(attention, seed_percentile) }    # 기본 95
```

### 7.3 검사 A — 점 보존 (안전 계약이 걸려 있는 한 줄)

`lift`의 docstring: *"The returned cloud has `len(cloud)` entries. **Always.**"*

**attention은 target의 *이름*을 정할 뿐 기하를 지우지 않는다**는 AG3S의 안전 계약(§3 계약 2)이
이 문장에 걸려 있다. 낮은 attention을 걸러내는 최적화가 언젠가 여기 들어가면, **물리적으로 존재하는
장애물이 "VLA가 안 봤다"는 이유로 충돌 후보에서 사라진다.**

**결과: 60회(20프레임 × 3카메라) 전부 입력 점 수 = 출력 점 수 = attention 값 개수.
합계 3,038,175개 점, 손실 0개.**

### 7.4 검사 B — 픽셀 대응 (보간은 값을 만들어내지 않는다)

- `nearest`: 점에 붙은 값이 그 점이 나온 픽셀의 패치를 **직접 조회한 값과 정확히 같아야** 한다.
  → **최대 오차 0.00e+00**
- `bilinear`: 정확한 일치는 요구할 수 없지만, 값이 **이웃 네 패치의 [최솟값, 최댓값] 밖으로
  나가면 안 된다.** → **범위 이탈 최대 1.82e-12**

**이 두 검사는 Step 2의 A와 같은 역할이다.** B가 통과하면 대응은 옳고, 이후 문제는 attention
자체이거나 점 자체다.

### 7.5 검사 C — 정규화가 순위를 보존하는가 (설정 두 개의 독립성)

주장: 네 모드 전부가 순위를 보존한다. **이 주장이 `attention.normalization`과
`clustering.seed_percentile`을 독립적으로 고를 수 있는 근거다** — 백분위 기준 컷은 값이 아니라
순위만 보므로, 주장이 사실이면 어떤 모드에서도 같은 점이 뽑혀야 한다.

**`argsort`로 비교하면 안 된다.** `percentile` 모드는 상·하위를 잘라 **의도적으로 동점을 만들고**,
동점의 순서는 정렬 구현이 임의로 정한다. 주장은 "순위 역전이 없다"(약한 순서 보존)이지 "동점이
없다"가 아니다. (**첫 판이 `argsort` 비교로 이 검사를 실패시켰다.**) 그래서 raw 오름차순으로
정렬했을 때 정규화 값이 **감소하지 않는지**를 본다.

**단조성 240/240 전부 성립.** seed 집합 일치도(기준 = percentile):

| 모드 | Jaccard | 완전 일치 |
|---|---|---|
| percentile | 1.0000 | 예 |
| minmax | 1.0000 | 예 |
| **softmax** | **0.9801 ~ 1.0000** | **아니오** |
| none | 1.0000 | 예 |

**softmax만 갈리는 원인은 순위 역전이 아니라 float32다.** softmax는 값의 범위를 크게 압축하고,
컷 근처에서 서로 다른 raw 값들이 **같은 float32로 뭉개져** 함께 임계를 넘는다. 약 2,600개 seed 중
최대 수십 개.

**판정 게이트는 단조성과 세 모드의 완전 일치에만 걸었다.** *"softmax를 통과시키려고 Jaccard
임계값을 고르는 것은 지표를 결과에 맞추는 일이므로 하지 않았다."* 대신 어긋남을 측정해 남긴다.

실질적 영향은 없다 — seed는 영역 성장의 출발점일 뿐이고 수십 개가 달라져도 성장 결과는 같은
클러스터에 수렴한다. 다만 docstring의 주장은 **정신은 옳고 문자 그대로는 조금 과하다.**

### 7.6 검사 D — 순위 지표 (그리고 첫 판이 틀렸던 이유)

**seed 순도를 판정 기준으로 삼았던 첫 판은 틀렸다.** 이유 둘:

1. **4단계는 seed를 그대로 target으로 쓰지 않는다.**
   ```
   seed 점 ──grow_region(eps)──▶ 성장한 영역 ──DBSCAN──▶ 여러 클러스터 ──점수화──▶ target 하나
   ```
   seed는 "여기서부터 기하를 따라가 보라"는 **지시**일 뿐이다. seed 하나가 물체 위에 있으면 그
   물체 전체가 성장으로 딸려 오고, seed가 엉뚱한 곳에 있어도 그 클러스터는 점수화에서 탈락한다.
   **그래서 seed는 순수할 필요가 없다.**
2. **순도는 물체 크기가 천장을 정한다.** 사과는 화면의 약 **0.14%**뿐이라, 상위 5% 점이 전부
   뽑혀도 순도는 **구조적으로 0.028을 넘을 수 없다.** (1단계에서 β=0이 씬에 눌려 헤드를 구분하지
   못했던 것과 **같은 함정**이다.) 첫 판은 이 값을 0.007로 보고했는데, 천장 0.028을 옆에 적지
   않으면 **재앙처럼 읽힌다.**
3. 더 근본적으로: **3단계의 질문은 "attention이 옳은 점에 붙는가"이지 "4단계가 좋은 씨앗을
   받는가"가 아니다.** 후자를 재려면 4단계의 성장·군집·점수화까지 봐야 하는데, 그러면 두 단계를
   한 번에 재는 셈이라 실패 시 원인을 가릴 수 없다.

**그래서 크기에 영향받지 않는 세 지표로 바꿨다:**

| 지표 | 무엇을 묻는가 | 천장 | 우연 수준 |
|---|---|---|---|
| **peak가 target 위** | `ground_target`이 실제로 쓰는 최고 attention 점이 target 위인가 | 1.0 | 물체 면적 비율 (<0.1%) |
| **AUC** | 무작위 target 점이 무작위 비-target 점보다 높은 attention을 가질 확률 | 1.0 | 0.5 |
| **precision@N** | 상위 N개(N = target 점 수) 중 target 비율 | 1.0 | 물체 면적 비율 |

AUC는 Mann–Whitney 순위합으로 계산한다:

```
ranks = argsort(argsort(attention)) + 1
AUC = ( Σ_(i∈target) ranks_i − n_t(n_t+1)/2 ) / (n_t · n_o)
```
(`n_t` = target 점 수, `n_o` = 비-target 점 수)

**세 지표를 함께 두는 이유:** 각각 다른 것에 둔감하다 — `peak`는 한 점만 보므로 분포를 못 보고,
`AUC`는 전체 분포를 보므로 극단값에 둔하며, `precision@N`은 그 사이다.

**결과:**

| 카메라 | peak가 target 위 | AUC | precision@N | 평균 target 점 수 |
|---|---|---|---|---|
| **cam_high** | **1.000** | **0.99976** | **0.553** | 27 |
| cam_left_wrist | **0.059** | 0.80576 | 0.091 | 25 |
| cam_right_wrist | 0.800 | 0.96390 | 0.438 | 15 |

판정 기준: peak ≥ 0.8, AUC ≥ 0.9 (head 카메라 기준) → **통과.**

**seed 진단 (판정 아님, 4단계가 받게 될 것):** 각 칸은 `순도 / 천장 → 회수율`.
회수율 = 순도/천장 = **target 점 중 몇 %가 seed에 들어갔는가** — 크기 효과가 약분되어 바로 읽힌다.

| 카메라 | 상위 10% | 상위 5% | 상위 2% | 상위 1% |
|---|---|---|---|---|
| cam_high | 0.0033 / 0.0035 → **0.87** | 0.0065 / 0.0069 → **0.83** | 0.0160 / 0.0173 → **0.79** | 0.0314 / 0.0346 → **0.77** |
| cam_left_wrist | 0.0026 / 0.0043 → 0.69 | 0.0047 / 0.0086 → 0.63 | 0.0090 / 0.0214 → 0.51 | 0.0156 / 0.0428 → 0.45 |
| cam_right_wrist | 0.0014 / 0.0016 → 0.88 | 0.0028 / 0.0033 → 0.84 | 0.0069 / 0.0081 → 0.84 | 0.0136 / 0.0162 → 0.83 |

상위 5% seed가 실제로 앉는 곳 (개수):

| 카메라 | target | 다른 물체 | 배경 | robot |
|---|---|---|---|---|
| cam_high | 355 | 9589 | 44110 | 1922 |
| cam_left_wrist | 247 | 1199 | 49340 | 152 |
| cam_right_wrist | 133 | 1142 | 43793 | 2 |

**`robot`을 따로 센 것은 의도적이다.** attention이 그리퍼에 몰려 seed가 로봇 위에 앉는 것은
"엉뚱한 물체를 골랐다"와는 **다른 종류의 실패**이고, 로봇 점은 self-filter에서 제거되므로 4단계에
도달하지도 않는다 — **즉 seed만 낭비된다.**

### 7.7 발견 E — 왼쪽 손목 카메라 attention은 target을 가리키지 않는다

1단계는 **head 카메라만** 검증했다. 그런데 `multiview.fuse`는 세 카메라의 attention을 **`max`로
합친다.**

- `cam_left_wrist`: **peak 0.059** — 17개 채점 프레임 중 단 1개에서만 최고점이 사과 위에 온다.
  AUC 0.806은 우연(0.5)보다는 낫지만 head(0.99976)와 비교가 되지 않는다.

**원인은 셀 선택이다.** 1단계는 **head 카메라의 토큰 블록 [0, 256)** 위에서 (L8, h2)를 골랐다.
같은 (층, 헤드)가 왼쪽 손목 블록 [256, 512)에서도 target을 찾는다는 보장은 **어디에도 없다** —
오른쪽 손목에서는 우연히 옮겨갔고(0.800), 왼쪽에서는 그러지 않았다.

**따라서 attention을 `max`로 융합하면 왼쪽 손목의 잘못된 peak가 그대로 들어온다.** 선택지 셋:

1. 카메라마다 셀을 따로 고른다 — 카메라별로 1단계를 다시 돌려야 하고 정답 대조가 필요하다.
2. 검증된 카메라만 융합에 넣는다 — 여기서는 `cam_high`와 `cam_right_wrist`.
3. **attention은 head 카메라 것만 쓰고, 점군만 세 카메라로 융합한다.** ← **채택**

**3번이 가장 방어하기 쉬운 이유:** attention은 target의 *이름*을 정할 뿐이고 그 일은 한 카메라로
충분하며(head peak 1.000), **기하 융합은 Step 2 검사 D에서 세 카메라가 서로 일치함을 이미
확인했다.** 4단계는 이 구성으로 진행한다.

→ **슬라이드 메시지: 검증이 설계를 실제로 바꾼 사례.** 이 발견이 없었다면 파이프라인 기본 동작
(`max` 융합)이 조용히 잘못된 target을 지목할 수 있었다.

### 7.8 판정 — **PASS** (A/B/C/D 전부), 인계

3단계의 출력 = 4단계의 입력: **점 개수가 보존된 attention 점군** + head 카메라만 사용한다는
구성 결정.

---

## 8. 종합 · 한계

### 8.1 Step 1→2→3 인계 사슬

| Step | 받은 것 | 확인한 것 | 넘긴 것 |
|---|---|---|---|
| 1 | 기록된 정책 이미지 3장 + 세그멘테이션 GT | 프롬프트가 지목한 물체를 본다 (peak 1.000, lift 91.82) | **셀 (L8, h2, Euler 0, last)** |
| 2 | 기록의 qpos → 재생 depth + K + T_base_cam | 점이 실제 물체 위에 놓인다 (0.7 mm / 7.2 mm) | **uv가 보존된 base frame 점군** |
| 3 | 1의 셀 + 2의 점군 | 옳은 attention이 옳은 점에 붙는다 (손실 0, AUC 0.99976) | **attention 점군 + head-only 구성 결정** |

**세 단계 전부 PASS.** 그리고 각 단계가 다음 단계의 실패 원인을 좁혀 주므로, 4단계에서 문제가
생기면 그것은 4단계의 문제라고 말할 수 있다.

### 8.2 이 검증이 재지 *않은* 것 (반드시 발표에 포함)

**1. depth에 잡음이 없다.** MuJoCo 렌더러의 깊이에는 실제 ZED의 잡음, 반사면 결측, 물체 경계의
flying pixel, 스테레오 정합 실패가 없다. **Step 2의 "표면 오차 0.4–7 mm"는 역투영 수학과 좌표
규약의 정확도이지 실기 정확도가 아니다.** `--record-depth`의 uint16 mm는 1 mm 양자화를 반영하지만
**표현 형식이지 잡음 모형이 아니다.**

**2. 정답이 시뮬레이터에서 온다.** 물체 자세·메시 정점·세그멘테이션이 전부 MuJoCo가 알려준 값이다.
실기에는 이 정답이 없으므로, 넘어갈 때는 정답을 다른 방식으로 마련하거나(마커, 수동 라벨) 정답
없이 **자기일관성 검사**만으로 만족해야 한다.

> **이 검증의 최대 강점이 시뮬레이터라는 점이고, 최대 한계도 같은 점이다.**

**3. 파이프라인 전반의 알려진 한계** (README §"알려진 한계"에서 Step 1~3과 관련된 것):
- MuJoCo에서는 URDF FK와 시뮬레이터가 **정확히 일치**(링크 위치 오차 0.0 mm)하므로 정합 오차가
  0이다. hand-eye calibration 오차·depth noise·실제 timestamp skew는 **실기에서만 검증 가능**하다.
- `eps` 운용 범위는 `voxel_size < eps < 최소 물체 간격`. 3 cm 기본값은 **2 cm 간격 물체를
  병합**한다 (유클리드 연결성의 본질적 한계이며 버그가 아님).

### 8.3 다음 단계 (문서는 이미 존재, 이번 발표 범위 밖)

| Step | 판정 | 한 줄 |
|---|---|---|
| 4 target grounding | PASS (파지 전) | 채점 가능 8프레임 **8/8**, IoU 0.953, 오선택 0건 |
| 5 target/obstacle 분리 | PASS | 덮는 점 차이 **0점** — attention이 완전히 틀려도 기하는 남는다 |
| 6 geometry (primitive) | **부분 통과** | 관측 점 포함 1.000 / **실제 물체 포함 0.666**, 최대 침투 46.7 mm |
| 7 constraints + TO | 미착수 | 도구 없음 |

---

## 9. Step 1~3에 관계된 설정값 (`configs/`)

| 항목 | default.yaml | rby1_three_camera.yaml | 근거 |
|---|---|---|---|
| `attention.normalization` | `percentile` | — | 뜨거운 패치 하나의 지배를 막는다 |
| `attention.percentile_range` | `[5.0, 99.0]` | — | |
| `attention.seed_percentile` | 95 | — | 상위 5%가 seed |
| `attention.interpolation` | `bilinear` | — | 픽셀 중심 정렬 (§7.2) |
| `pointcloud.voxel_size` | 0.005 | 0.006 | |
| `pointcloud.max_points` | 60000 | 60000 | 실시간 요구 (640×480 = 307k) |
| `pointcloud.depth_max` | 3.0 | 2.5 | z-깊이 컷 |
| `pointcloud.range_max` | (없음) | **1.6** | 90° 렌즈에서 2.5 m z-컷이 4.2 m 점을 통과시킨다 (실측) |
| `pointcloud.cap_strategy` | — | `voxel` | 인덱스 선택이 아니라 보셀 확대 |
| `timing.fusion_voxel_size` | — | 0.012 | **카메라 *간* 정합 오차가 카메라 *내* 양자화보다 크다.** 6 mm면 같은 테이블 표면이 몇 mm 떨어진 세 장의 시트가 된다 |
| `support_surface.distance_threshold` | 0.008 | 0.008 | RANSAC 내점 임계값 = Step 2 fig1이 최댓값 대신 95p를 쓰는 이유 |
| `support_surface.max_normal_angle_deg` | 25.0 | — | 없으면 씬에서 가장 큰 평면이 벽이나 상자 옆면일 수 있다 |
| `clustering.eps` | 0.03 | 0.03 | `voxel_size < eps < 최소 물체 간격` |
| `geometry.perception_uncertainty` | — | **0.0** | MuJoCo에는 잡음이 없으므로 0. **실기에서 올려야 할 곳은 `d_safe`가 아니라 여기다** |

---

## 10. 그림 인벤토리 (존재 확인 완료)

경로는 저장소 루트 기준. 슬라이드에는 **Placeholder + 경로 + 캡션**만 넣는다.

### Step 0 · Step 1
| 경로 | 무엇을 보여주나 | 읽는 법 |
|---|---|---|
| `benchmark/ag3s/asset/image/attention/fig0_record_sanity.png` | 제어 스텝별 target 가시 픽셀 수 + 16×16 패치 점유 히트맵 | **결과가 아니라 입력 검사.** 채점 가능 프레임 수와 가림 구간 |
| `.../attention/fig1_layer_head_peak_on_target.png` | 18층×8헤드 격자, 칸 색 = peak-on-target | 진한 칸이 많으면 지표가 **포화**한 것 → 지표를 바꿔야 한다. 주황 테두리는 **순위가 고른 칸**이지 격자의 최댓값이 아니다 |
| `.../attention/fig2_attention_overlay.png` | 정책이 실제로 본 224×224 위의 attention + 정답 target 윤곽 | **○가 주황 윤곽 안에 있으면 그 프레임은 peak-on-target 성공** |
| `.../attention/fig3_attention_lift.png` | 물체별 attention **밀도**(mass/면적비)의 시간 변화 | 1.0(점선)이 "균등 attention과 같음". raw mass 축이면 테이블·크레이트가 크기만으로 위를 차지하고 과일이 전부 0 근처에 눌린다 |
| `.../attention/fig4_choice_sweep.png` | Euler step × pooling 조합별 최고 hit rate | 막대 높이가 크게 다르면 그 선택이 결과를 좌우한다는 뜻 |

### Step 2
| 경로 | 무엇을 보여주나 | 읽는 법 |
|---|---|---|
| `benchmark/ag3s/asset/image/backprojection/fig1_table_plane.png` | 프레임별 평면 기울기(°)와 잔차(RMS·95p, mm) | 최댓값 대신 95p를 그린 이유는 최댓값이 RANSAC 임계값 8 mm에 눌려 **측정값이 아니라 설정값**이 되기 때문 |
| `.../backprojection/fig2_surface_error.png` | 물체별 × 카메라 3대의 95p 표면 오차 막대 | **한 카메라만 틀린 경우가 합쳐 놓으면 뒤에 숨는다.** 나눠 놓으면 막대 하나로 보인다 |
| `.../backprojection/fig3_topdown.png` | 점군의 x–y 투영, ×는 MuJoCo 실제 물체 중심 | **점이 ×를 둘러싸지 않고 한쪽에 몰리는 것이 정상** — 카메라는 앞면만 본다. 크레이트가 속이 빈 사각 링으로 나오는 것도 같은 이유 |

### Step 3
| 경로 | 무엇을 보여주나 | 읽는 법 |
|---|---|---|
| `benchmark/ag3s/asset/image/lifting/fig1_ranking.png` | peak / AUC / precision@N × 카메라 3대 | 1.0이 최선, AUC 우연 수준 0.5(점선). **왼쪽 손목만 무너진 것이 한눈에 보인다** |
| `.../lifting/fig2_seed_points.png` | seed가 어디에 앉는가 (2패널: 전체 / target 확대) | 왼쪽은 seed 대부분이 **배경**에 있음을, 오른쪽은 그 와중에도 target 위에 앉음을 보여준다. **seed 순도를 지표로 쓸 수 없는 이유를 눈으로** |

### 파이프라인 개관 · 직관용 (갤러리)
| 경로 | 무엇을 보여주나 |
|---|---|
| `benchmark/ag3s/asset/image/gallery/g1_per_camera.png` | 카메라 3대가 각각 무엇을 보는가. head는 테이블 전체와 과일 넷, 손목은 크레이트 근처 좁은 조각만 — **Step 3에서 왼쪽 손목 attention이 실패한 이유가 여기 있다** |
| `.../gallery/g2_fusion.png` | 세 점군을 겹쳐 **출처 카메라로 색칠**. 색이 갈라지지 않으면 외부 파라미터가 맞는 것 — **Step 2 검사 D를 숫자가 아니라 그림으로** |
| `.../gallery/g3_attention_2d.png` | RGB · depth · attention 나란히. **3D로 올라가기 직전의 상태** |
| `.../gallery/g4_attention_3d.png` | 점군을 attention으로 색칠. **낮은 점도 전부 그린다** — 아무것도 버리지 않았다는 불변식의 시각적 증거 |
| `.../gallery/g5_target.png` | 고른 클러스터 + 맞춘 primitive + 정답 표면 |
| `.../gallery/g6_stages.png` | 원본 → 자기 필터 → 지지면 제외 → seed → target. 무엇이 언제 걸러지는가 |
| `benchmark/ag3s/asset/image/ag3s_01_raw_cloud.png` … `ag3s_08_constraints.png` | 합성 fixture 위 8단계 (파이프라인 개관용) |

**그림 규격 (`experiments/figstyle.py`)**: 크기·비율은 파랑 한 색의 명도 램프(순차형), 정체는
검증된 8색 팔레트를 **고정 순서**로 — **순서 자체가 색각 이상 안전 장치**(인접 슬롯 쌍 CVD ΔE ≥ 8).
그래서 차트마다 색을 재배열하지 않는다. 회색은 데이터가 아닌 것.

---

## 11. 발표에서 쓰면 좋은 "한 줄" 모음

- "Attention은 target이 **무엇인지**만 정하고, **무엇이 충돌할 수 있는지**는 3D 기하가 정한다."
- "마지막 단계만 보면 실패해도 원인을 모르고, 성공해도 운인지 모른다."
- "아무것도 통과할 수 없는 게이트는 증거가 아니라 고장난 규칙이다." (Step 1 판정 규칙)
- "A가 통과하고 B가 실패하면 문제는 규약이고, A가 실패하면 식 자체다." (Step 2 검사 설계)
- "역투영은 옳았고 검사가 틀렸다." (Step 2 테이블 다리·옆면)
- "보간은 섞을 뿐 만들어내지 않는다." (Step 3 검사 B)
- "softmax를 통과시키려고 임계값을 고르는 것은 지표를 결과에 맞추는 일이다." (Step 3 검사 C)
- "0.0065라는 순도는 실패가 아니라 사과가 화면의 0.14%라는 뜻이다." (Step 3 검사 D)
- "이 검증의 최대 강점이 시뮬레이터라는 점이고, 최대 한계도 같은 점이다."

---

## 12. 이 문서에 **없는** 정보 (슬라이드에 필요하면 사람에게 물어볼 것)

- π0.5 파인튜닝 설정의 세부 — 데이터셋 규모, LoRA rank, 학습 시간. 문서에는 체크포인트 경로
  (`rby1_atomic_basket_14d_v2_30k_20260825/29999`)만 있다.
- 이 연구의 상위 목적 — 목표 논문/학회, 과제명, KNOWS 논문과의 정확한 관계.
- RB-Y1 실물 사진·하드웨어 사양. 현재 시각자료는 **전부 MuJoCo 렌더**다.
- Step 1~3의 **실행 시간 / 계산 비용**. `profiler.py`가 stage latency를 재지만 이 세 검증
  스크립트의 소요 시간은 문서에 기록되어 있지 않다.
- 정량적 baseline 비교 대상(다른 논문·방법). 이 검증은 **자기 파이프라인의 내부 정합성** 검증이지
  경쟁 방법과의 비교 실험이 아니다.
