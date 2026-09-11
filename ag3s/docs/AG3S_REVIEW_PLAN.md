# AG3S 공동 코드 검토 — attention + TSDF/ESDF 전환 전제

> **실행 기록** — [AG3S_REVIEW_LOG.md](AG3S_REVIEW_LOG.md). 스텝별 진행 현황, 누적 발견 표,
> 모듈 처분 표는 그쪽에서 갱신된다. 이 파일은 검토를 시작할 때 합의한 **계획**이고,
> 계획이 바뀌면 여기를 고친다.

## Context

사용자의 결정 세 가지.

1. **씬(장애물) 쪽 primitive 근사는 쓰지 않는다.** 포인트 클러스터에 구를 씌우는 경로를 걷어내고
   TSDF/ESDF로 대체한다.
2. **로봇 모델 자체는 primitive를 계속 쓴다.** sphere chain, `robot_radii`, 캡슐→구 이산화,
   잡은 물체의 구 표현은 전부 유지된다. 제약은 `d_esdf(p_robot(q)) − r_robot − margin ≥ 0`
   형태이므로 로봇 쪽 구는 오히려 **필수**다.
3. **attention과 TSDF/ESDF를 제대로 결합한다.** 이것이 이 검토의 목표다 — 지금 결합 방식이
   틀려 있다는 것이 사전 조사의 가장 큰 발견이다.
4. **지금 단계에서 실시간성(지연)은 최적화 대상이 아니다** (2026-09-07 결정). 이 스택은 어차피
   지금 CPU/numpy 기반이라 실시간에 못 미친다 — 그 사실 자체는 이미 알려진 것이고 이 검토가
   풀 문제가 아니다. **나중에** cuRoboV2/nvblox 류의 GPU 가속 패턴(block-sparse voxel hashing,
   voxel-centric projection, PBA+ 등 — Step 1 E6 판정 근거, `papers/curobo_report_v1.pdf`,
   `papers/cuRoboV2_...pdf` 참고)을 참조해 별도로 개선한다. 그동안 이 검토·수정에서 성능은
   "판단 결과를 안 바꾸면서 자릿수 개선이 되면 좋음" 정도로만 다루고, 판단(무엇을 장애물로 볼
   것인가, 언제 완화할 것인가)의 정확성을 우선한다.

그 전제로 다시 조사한 결과, 코드의 실제 구조가 문서가 말하는 것과 다르다는 것이 드러났다.

- 실제 최적화기는 AG3S 안의 CasADi `ConstraintBuilder` 가 **아니다.** `benchmark/trajopt/`
  (4.2k LOC) 의 SQP이고, `linearize.py:531 scene_from_constraint_set` 이 AG3S 출력을 받아
  `SceneSnapshot` 으로 바꾼 뒤 수치 선형화한다. ESDF는 이미 이 경로로 end-to-end 연결되어 있다
  (`linearize.py:324 _esdf_clearance` — `d_esdf(p) − r − margin`, 기울기는 `∇d·J`).
- 즉 ESDF는 CasADi 심볼릭 브리지가 필요 없다. AG3S 의 `constraint_builder.py` 심볼릭 기계장치는
  **씬 primitive 전용**이고, 그것이 삭제 대상이다.
- 그런데 삭제하면 딸려 나가는 것이 있다. 평면(지지면) 행과 (sphere, slot) 마진 행렬이 그 심볼릭
  파라미터 벡터 안에 얹혀 있다.

사전 조사에서 **검증된 결함 7건 + 잔존 의심 4건**을 찾았다. 셋은 전환 설계 자체를 바꿔야 한다.

> **E1** GRASP에서 ESDF는 target 복셀을 파낸다 — 손끝뿐 아니라 **몸통·전완·반대팔에 대해서도**.
> 이것은 `clearance.py:1-27` 이 "AG3S가 하던 가장 위험한 일"이라며 고쳤다고 선언한 바로 그 동작이다.
> **attention과 ESDF를 결합하는 현재 방식이 정확히 여기서 틀려 있다.**
>
> **E2** 테이블이 무한 half-space다. 실측: 홈 자세 61 sphere 중 **26개가 최대 0.677 m 위반**.
> 그리고 ESDF 모드에서는 테이블이 필드에서 파내지므로 이 평면 행이 **유일한** 테이블 제약이다.
>
> **E3** 잡은 물체(attached)는 지금 실제 optimizer에 **전혀 도달하지 않는다.** trajopt에
> attached 블록을 읽는 코드가 없다.

산출물은 두 가지다. (1) 사용자가 ESDF 경로를 스텝별로 파악한 상태, (2) 그 과정과 판단의 기록.

**계획 변경 (2026-09-07): 확정된 결함은 그 스텝 안에서 바로 고친다.** 원래는 "수정은 검토가
끝난 뒤"였는데, Step 1(E4·E6)·Step 2(E1·E5)에서 사용자가 그때그때 "고치고 넘어가자"고 지시해
방식이 바뀌었다. 지금부터 각 스텝은 `설명 → 질문 → 공동 판정(확정/기각) → (확정된 것은 그
자리에서 수정 + 검증) → 기록` 순서다. 수정이 아직 검토 안 한 스텝의 모듈까지 건드려야 할 때는
(E1이 Step 3·10을 건드렸듯) 사용자에게 범위를 먼저 확인한다 — 계획서의 "스텝을 합치지 않는다"
원칙은 **설명/판정 순서**에 대한 것이지 수정 타이밍에 대한 것이 아니게 됐다.

---

## 방향 결정 (2026-09-07) — cuRoboV2 를 인프라로 쓰고, attention 은 해상도를 배분한다

검토 중에 드러난 사실 때문에 전환의 목표 자체를 다시 잡았다.

**사실 1 — 우리가 직접 만들고 있던 ESDF 기계장치는 이미 오픈소스로 존재한다.**
cuRoboV2 (`papers/cuRoboV2_...pdf`, 코드 `https://github.com/NVlabs/curobo`) 가 매니퓰레이터용
GPU-native TSDF/ESDF 를 제공하고, nvblox 대비 10배 빠르고 메모리 8배 적다고 보고한다. 우리
발견 중 **E4(고정 상자 밖 = 자유)와 E6(전 복셀 매 프레임 투영)은 그쪽 구조에서는 애초에
생기지 않는다** — block-sparse 할당이라 벗어날 고정 상자가 없고, block discovery 가 전 복셀
투영을 대체한다.

**사실 2 — 우리 발견의 대부분은 cuRoboV2 에 대응물이 없는 영역에서 나왔다.**
cuRoboV2 에는 attention 이 없다. target grounding 도, phase x manipulator 별 접촉 권한도,
`ClearancePolicy` 도 없다. E1 · E2 · E3 · F2 · F8 · F9 · F10 이 전부 그 영역이다.

### 결정

1. **ESDF 인프라는 cuRoboV2 를 최대한 가져다 쓴다.** block-sparse TSDF, voxel-centric 적분,
   PBA+ dense ESDF, GPU 실행. 우리가 다시 만들지 않는다.
2. **우리 기여는 attention 쪽에 둔다.** 무엇이 target 인가, 어느 링크가 그것을 만져도 되는가,
   그리고 아래 3번.
3. **attention 이 ESDF 의 해상도 예산을 배분한다 (2계층 ESDF).** cuRoboV2 는 TSDF 와 ESDF 의
   해상도를 분리해 두었지만(§5.4), ESDF 자체는 단일 해상도다. 거기에 "관심 영역만 미세하게"를
   얹는다.

### 2계층 ESDF — 무엇을 하고 무엇을 하지 않는가

**하지 않는 것: 범위를 줄이지 않는다.** 초안에서 "target 주변만 ESDF 로 만들자" 를 검토했다가
실측으로 기각했다 — 장애물은 target 이 아니라 **target 이 아닌 것들**이고, 실제로 장애물의
절반이 target 에서 0.54 m 넘게 떨어져 있다. 반경 0.20 m 로 자르면 연산은 99% 줄지만 장애물의
84% 가 사라진다.

```
장애물(OCCUPIED) 의 target 까지 거리:  50% 분위 0.54 m,  75% 0.84 m,  90% 0.99 m,  최대 1.47 m

반경   | target 주변 남는 장애물 | 로봇 궤적 주변 남는 장애물
0.20 m |        15.5%          |        31.6%
0.30 m |        26.8%          |        54.4%
0.50 m |        46.5%          |        75.3%
```

**하는 것: 정밀도를 몰아준다.** 작업공간 전체를 거친 복셀로 **그대로 유지**하고(먼 장애물도
계속 회피된다), attention 이 가리킨 target 주변만 미세 복셀로 한 겹 더 얹어 질의에서
`min()` 으로 합친다 — cuRoboV2 가 depth 채널과 analytic 채널을 합치는 그 방식과 같다.

```
균일 20 mm         547,200 복셀   이산화 편향 -10.0 mm  (50 mm 마진의 20% 소모)
균일  5 mm      34,675,200 복셀   이산화 편향  -2.5 mm  <- 이상적이지만 35 M, 불가능
2계층 R=0.15 m     763,200 복셀   균일 5 mm 대비 45.4x 절감,  target 주변 장애물 9% 가 5 mm
2계층 R=0.20 m   1,059,200 복셀   균일 5 mm 대비 32.7x 절감,  target 주변 장애물 15% 가 5 mm
2계층 R=0.30 m   2,275,200 복셀   균일 5 mm 대비 15.2x 절감,  target 주변 장애물 27% 가 5 mm
```

**이득의 성격**: 지금 20 mm 복셀의 편향 -10 mm 가 `esdf_margin` 50 mm 의 20% 를 먹고 있고,
그래서 `esdf-full-scenario.md` 가 "잔여 -5.1 mm 가 복셀 노이즈인지 진짜 관통인지 구분되지
않는다" 고 적었다. 손끝이 실제로 다투는 자리에서만 그 편향이 -2.5 mm(5%)로 떨어진다.
**잃는 것은 없다** — 범위가 아니라 해상도를 바꾸는 것이므로.

### API 검증 완료 (2026-09-11, 실제 H200 에서 실행)

`https://github.com/NVlabs/curobo` (commit `78fd485`) 를 받아 이 장비에서 직접 돌렸다.
**2계층 ESDF 는 stock API 로 그대로 된다 — 포크도 개조도 필요 없다.**

확인한 것:

1. **nvcc 없이 설치·실행된다.** `setup.py` 의 `USE_PYBIND` 기본값이 `0` 이라 CUDA 확장 컴파일은
   선택이고, 기본은 Warp/cuda.core JIT 백엔드다. 이 서버에 CUDA toolkit 이 없는데도 `pip install
   -e .` 만으로 동작했다. (openpi venv 를 건드리지 않으려고 `--system-site-packages` 로 별도
   venv 를 만들어 설치했다 — `numpy`/`scipy` 가 업그레이드되면 openpi 가 깨진다.)
2. **`MapperCfg` 가 TSDF 와 ESDF 의 해상도·범위를 처음부터 분리해 둔다.**
   `voxel_size`(TSDF, 기본 5 mm) 대 `esdf_voxel_size`(기본 50 mm),
   `extent_meters_xyz` 대 `extent_esdf_meters_xyz`.
3. **같은 TSDF 에서 ESDF 를 해상도·원점을 바꿔 여러 번 뽑을 수 있다.**
   `Mapper.compute_esdf(esdf_origin=..., esdf_voxel_size=...)` 가 호출마다 오버라이드를 받고,
   docstring 이 그 용도를 "sliding window" 라고 적고 있다. **적분은 한 번, 추출은 여러 번**이다.
4. **두 ESDF 가 한 씬에 장애물로 공존한다.** `SceneCfg.voxel: Optional[List[VoxelGrid]]` 이고
   `VoxelGrid` 는 `Obstacle` 서브클래스라 각자 pose·dims·voxel_size 를 갖는다.
5. **ESDF 버퍼는 고정 128³ 이고 물리적 범위는 `voxel_size` 가 정한다.** 20 mm 면 2.56 m 상자,
   5 mm 면 0.64 m 상자. 우리 작업공간(1.5 x 1.8 x 1.6 m)이 거친 계층에 그대로 들어가고, 미세
   계층 0.64 m 는 target 반경 0.32 m 에 해당한다 — 앞서 계산한 R=0.30 m 와 거의 일치한다.

실측 시간 (카메라 1대, JIT 워밍업 후 10회 평균, H200 NVL):

```
TSDF 적분              0.61 ms
ESDF 거친 20 mm        0.40 ms
ESDF 미세  5 mm        0.45 ms      <- 2계층의 추가 비용은 이것뿐
2계층 합계             1.46 ms      메모리 191.8 MB
```

비교) 우리 numpy 구현은 카메라 3대·10 mm 에서 정상 상태 약 940 ms 다. 카메라 수가 달라 그대로
비교할 수는 없지만(cuRobo 쪽은 1대), **자릿수가 세 개 차이**라는 것은 분명하다. 청크 예산
66.7 ms 대비 1.46 ms 는 **예산의 2%** 이고, 지금 우리가 40배 초과인 상태에서 예산 안으로 들어온다.

즉 "2계층은 비용이 든다" 는 걱정이 실측으로 기각됐다 — 미세 계층 추가 비용이 **0.45 ms** 다.

> **측정 두 벌이 있다 — 카메라 수가 다르다.** 위 표는 **카메라 1대** 기준이고,
> `AG3S_REVIEW_LOG.md` 의 "cuRoboV2 API 검증" 절에는 **실제 RB-Y1 3대**(head + 손목 둘) 기준
> 측정이 있다: TSDF 적분 1.72 ms, ESDF 거친 0.52 ms, 미세 0.74 ms, **합계 2.98 ms**.
> 결론은 같다(예산 66.7 ms 의 4%) — 두 숫자를 섞어 인용하지 말 것.
> 로그 쪽에는 **정확성 검증**(좌표계 일치, 2계층 오차 33 mm 대 2 mm)도 함께 있고, 그것이
> 이 설계의 이득을 실제로 뒷받침하는 부분이다.

### 아직 확인하지 않은 것
- 미세 계층의 중심을 target centroid 로 둘지, 로봇 손끝 궤적으로 둘지. 실측상 로봇 궤적 기준이
  장애물을 더 많이 담지만(0.50 m 에서 75.3% 대 46.5%), **해상도 배분에서는 범위 손실이 없으므로
  이 비교가 그대로 적용되지 않는다.** 별도 판단이 필요하다.
- 기존 검토(Step 5~11)는 계속 진행한다. cuRoboV2 를 쓰더라도 attention 앞단(Step 5~7)과 접촉
  정책(Step 10)은 그대로 우리 것이고, 거기 있는 미판정 발견들이 이 방향에서 오히려 더 중요해진다.

---

## 진행 방식

**속도**: 천천히, 꼼꼼하게. 스텝을 합치지 않는다.
**순서**: 흐름의 종점이 ESDF이므로 `ESDF 코어 → 그 앞단(attention 포함) → 삭제 대상과 그 여파`.
**기록 파일**: `benchmark/ag3s/docs/AG3S_REVIEW_LOG.md` — 한국어.

각 스텝은 항상 같은 4단계.

1. **내가 설명** — 그 모듈이 무엇을 받아 무엇을 내놓는지, 어떤 설계 결정이 들어 있는지.
   코드는 `파일:줄` 로 짚어 사용자가 직접 열어볼 수 있게 한다.
2. **사용자 질문** — 사용자가 묻는다. **여기서 멈춘다.** 답이 끝나기 전에 다음 스텝으로 가지 않는다.
3. **문제 확정** — 후보를 제시하고 실제 문제인지 / 의도된 트레이드오프인지 **같이** 판정한다.
   추측으로 남기지 않는다. 가능하면 그 자리에서 실행해 수치로 확정한다 (E2가 그 방식이었다).
4. **로그 기록** — 확정 내용과 그 스텝의 완료 시각을 남긴다.

각 스텝마다 답해야 하는 질문이 둘 더 있다.

- **"ESDF 전환 후 이 모듈은 필요 / 축소 / 삭제 중 무엇인가."** 검토가 끝나면 이것이 삭제 계획이 된다.
- **"여기서 attention은 무엇을 하고 있고, 무엇을 해야 하는가."** 목표 3번이 이 질문의 누적이다.

---

## 삭제 경계 — 무엇이 남고 무엇이 가는가

"primitive를 안 쓴다"가 `geometry.py` 전체 삭제를 뜻하지 않는다. 확인한 경계는 이렇다.

| 남는다 (로봇 쪽) | 간다 (씬 쪽) |
|---|---|
| `robot_models/urdf_sphere_chain.py` — 자체적으로 캡슐→구 이산화. `geometry.py` 를 임포트하지 않음 | `fit_sphere` / `fit_box` / `fit_ellipsoid` — 포인트 클러스터에 도형 씌우기 |
| `geometry.to_spheres` — 잡은 물체 구 표현 (`attached.py:113`) 과 로봇 캡슐 | `inflate` / `containment_report` / `contains` — 씬 근사의 보수성 검증용 |
| `geometry.fit_capsule` — MuJoCo에서 로봇 캡슐 생성 (`mujoco_source.py:378`) | `collision_candidates.py` 전체 — 클러스터링 · 트래커 · overflow |
| `robot_radii`, `SceneSnapshot.robot_radii` | `constraint_builder.py` 의 후보 슬롯 블록 |

경계선이 애매한 것 하나: **평면(지지면) 행**. 지금 심볼릭 파라미터 벡터 안에 얹혀 있어서
후보 블록을 지우면 운반체를 잃는다. Step 8과 10에서 새 운반체를 정해야 한다.

**갱신 (2026-09-07, 방향 결정 이후)** — 위 표는 "씬 primitive 대 우리 ESDF" 구도로 그린 것이고,
cuRoboV2 를 인프라로 쓰기로 하면서 한 줄이 더 움직인다.

| 모듈 | 이전 처분 | 방향 결정 후 |
|---|---|---|
| `ag3s/esdf.py` (TSDF/ESDF 코어, 588줄) | 필요 (Step 1) | **대체 후보** — cuRoboV2 가 같은 일을 GPU 에서 한다. Step 1 에서 고친 E4·E6 도 그쪽에서는 구조적으로 안 생기는 문제였다 |
| `ag3s/pipeline.py` 의 ESDF 조립부 | 필요 (Step 2) | **유지하되 백엔드 교체** — `_depth_cameras_from`·`_esdf_coverage`·`target_link_margin` 배선은 그대로 쓰고, 그 안에서 부르는 필드 구현만 바뀐다 |
| `trajopt/linearize.py` 의 `_esdf_clearance` | 필요 (Step 3) | **유지** — `distance()`/`gradient()` 인터페이스만 맞으면 백엔드는 무관하다 |

즉 **우리가 유지하는 것은 배선과 정책이고, 교체되는 것은 필드 구현 하나**다. Step 1~4 의 수정이
헛되지 않은 이유이기도 하다 — E1(접촉 권한), G1(단위), G2~G4(검증 플래그)는 전부 백엔드가
바뀌어도 그대로 필요한 것들이다. 반대로 E4·E6 는 백엔드와 함께 사라진다.

---

## 기록 파일 구조

```markdown
## Step N — <모듈>
- 시작: 2026-09-05 14:20 KST / 완료: 2026-09-05 15:05 KST
- 처분: 필요 | 축소 | 삭제
- attention의 역할: <이 스텝에서 attention이 하는 일, 또는 "없음">

### 이 모듈이 하는 일
<입력 → 출력, 3~5줄>

### 공부한 것
<사용자가 새로 이해한 개념>

### 사용자 질문과 답
- Q: ... / A: ...

### 발견
| ID | 심각도 | 상태 | 요약 | 위치 |
|----|--------|------|------|------|

### 판정 근거
<확정/기각 이유. 수치를 찍었다면 그 명령과 출력>
```

파일 맨 위에 누적 발견 표 · 스텝 진행 현황 · **모듈 처분 표**를 두어 중간에 들어와도 위치를 알 수 있게 한다.

---

## 스텝 분해

### A. ESDF 경로가 실제로 무엇인가

| Step | 대상 | 초점 |
|---|---|---|
| 0 | 로그 생성 + 경계 지도 | `ag3s` ↔ `trajopt` 분업. 무엇이 심볼릭이고 무엇이 수치 선형화인가. 위 삭제 경계표 확정 |
| 1 | `esdf.py` (527) | `TsdfVolume` 투영 적분 · 세 상태 점유 · `EsdfField` 삼선형 보간/기울기 · `EsdfBuilder` 국소 갱신. **E4 · E6** |
| 2 | `pipeline.py` ESDF 조립부 | `_robot_mask_for:597` · `_depth_cameras_from:627` · `_build_esdf:668`. **E1 · E5 · E7** |
| 3 | `trajopt/linearize.py` ESDF 소비부 | `SceneSnapshot:73` · `_esdf_clearance:324` · `scene_from_constraint_set:531`. 로봇 구가 여기서 어떻게 쓰이는지. **E2 · E3** |

### B. attention 앞단 — ESDF와 결합되는 지점

| Step | 대상 | 초점 |
|---|---|---|
| 4 | `reconstruction.py` + `robot_filter.py` | ESDF는 raw depth를 먹는데 클라우드가 왜 아직 필요한가 (평면 · grounding) |
| 5 | `attention_lifting.py` | 어댑터 경계 · 정규화 위치 · `image_hw` 기본값. **F2 · F9** |
| 6 | `target_grounding.py` | 시드 → 3D 연결성 → 스코어. **이 출력이 곧 carving 입력이다.** **F8** |
| 7 | `multiview.py` | 3-카메라 융합 · capture-time FK. ESDF는 카메라별로 따로 적분한다는 비대칭 |
| 8 | `support_surface.py` + 평면 행 | **E2**. RANSAC 결정론, half-space의 무한성, 평면 행의 운반체 |

### C. 삭제 대상과 그 여파

| Step | 대상 | 초점 |
|---|---|---|
| 9 | `collision_candidates.py` + `geometry.py` | 삭제 범위 확정. 함께 잃는 것: 트래커 id 안정성 · overflow 보존 · unknown geometry. `to_spheres`/`fit_capsule` 은 남긴다 |
| 10 | `clearance.py` + `constraint_builder.py` + `to_adapter.py` + `attached.py` | **E1 · E3**. phase×link 접촉 정책을 ESDF에서 어떻게 표현할 것인가 |
| 11 | `tests/ag3s/` + `tests/trajopt/` | 무엇이 삭제되고 무엇이 새로 필요한가. **445개가 왜 E1~E3을 못 잡았는가** |

Step 11을 별도로 두는 것이 중요하다. 이 결함들이 전부 통과하는 스위트를 뚫고 남아 있다는 사실
자체가, 앞으로 코드를 고칠 때 무엇을 믿으면 안 되는지를 알려준다.

---

## 사전 조사에서 이미 찾은 것

Step 0에서 로그에 "미확정 후보"로 먼저 적고, 해당 스텝에서 하나씩 판정한다. **사용자와 함께
판정하기 전까지는 어느 것도 "확정"으로 쓰지 않는다.**

### 전환 설계를 바꿔야 하는 것

- **E1 · target carving이 전신에 대해 target을 지운다 (심각) — Step 2**
  `pipeline.py:684-696` 이 `rule.contact_permission` 을 보고 `esdf.py:339 carve()` 를 호출해
  target 복셀을 `OCCUPIED → FREE` 로 만든다. 필드는 **누가 묻는지를 모른다.** GRASP에서 그 물체는
  손끝뿐 아니라 몸통·전완·반대팔에게도 사라진다.
  `clearance.py:1-27` 은 이것을 "AG3S가 하던 가장 위험한 일"로 규정하고 (sphere, slot) 마진
  행렬로 고쳤다고 선언한다. 씬 primitive를 지우면 그 마진 행렬도 함께 사라지므로,
  **ESDF에는 링크별 접촉 권한을 표현할 수단이 하나도 남지 않는다.**
  목표 3번("attention + ESDF를 잘 쓰기")이 정면으로 걸리는 지점이고, 대안 설계가 필요하다.

- **E2 · 테이블이 무한 half-space, 그리고 ESDF에서는 그것이 유일한 테이블 제약 (심각) — Step 8**
  평면 행은 `n·c − offset − r` 이고 **모든 로봇 sphere × 모든 horizon step** 에 걸린다
  (`constraint_builder.py:216`, trajopt의 plane 블록도 같은 형태). 바닥에는 맞지만 테이블에는
  틀리다 — 상판은 유한한데 half-space는 무한해서 "로봇의 어떤 부위도 테이블 높이 아래로 내려갈
  수 없다"를 작업공간 전체에 강제한다.
  **실측** (`load_rby1()`, 홈 자세, 테이블 d=0.8353, margin 0.01): 61 sphere 중 **26개 위반,
  최악 −0.677 m**. 바닥(d=0.0128)은 0개.
  ESDF 모드에서는 `EsdfConfig.exclude_support_surfaces=True` (기본, `config.py:401`) 가 테이블을
  필드에서 파내므로 이 평면 행이 유일한 테이블 제약이다. 더구나 `linearize.py:551-565` 의
  spec=None 경로는 평면 행을 **빈 배열**로 만든다.

- **E3 · attached object가 실제 optimizer에 도달하지 않는다 (심각) — Step 10**
  `constraint_builder._attached_rows:220` 는 CasADi 표현식인데, trajopt는
  `SceneSnapshot.from_spec:126` 에서 pos / radius / d_safe / active / plane 블록만 읽는다.
  `benchmark/trajopt/` 전체에 attached 블록을 읽는 코드가 없다. 잡은 물체는 지금 최적화기에 없다.
  (로봇 쪽 primitive는 유지되므로 이건 표현의 문제가 아니라 **배선의 문제**다.)

### ESDF 코어 자체

- **E4 · 격자 밖은 무조건 자유 (중간) — Step 1**
  `outside_distance = cfg.max_distance` (`esdf.py:516`), `default_bounds` 는 x ∈ [−0.3, +1.2],
  y ∈ ±0.9 로 앞쪽만 자른다 (`esdf.py:400`). 상자 밖은 전부 "0.5 m 떨어짐"으로 답한다.
  씬 primitive는 클러스터가 어디에 있든 후보를 만들었으므로, 이 구멍은 전환으로 **새로 생긴다.**
- **E5 · ESDF per-camera 통계가 세 대 모두 `"camera"` 로 붕괴 (중간) — Step 2**
  `pipeline.py:658` 이 `getattr(obs, "camera", getattr(obs, "name", "camera"))` 로 읽는데
  `CameraObservation` 에는 `camera_id` 만 있다 (`types.py:365`). 세 대가 같은 키가 되어
  `esdf.py:466` 의 `per_camera[cam.name]` 에서 서로 덮어쓴다.
- **E6 · TSDF 적분이 전 복셀 중심을 매 카메라·매 프레임 투영 (중간) — Step 1**
  `esdf.py:127-141`. 4.3 M 복셀 × 3 카메라. 측정된 1213 ms/frame 의 주범.
- **E7 · 로봇 마스크를 만들려고 매 카메라마다 클라우드를 재구성 (낮음) — Step 2**
  `pipeline.py:597-625` 가 `backproject` 를 통째로 다시 돈다. self-filter가 이미 같은 일을 했다.

### attention 앞단 — 전환 후 오히려 중요도가 오른다

grounding 출력이 곧 carving 입력이므로, 목표 지정이 틀리면 필드에서 **엉뚱한 물체가 파인다.**

- **F2 · attention 정규화가 카메라별 (중간) — Step 5**
  `multiview.py:124-126` 주석은 "one normalization for all of them" 이라고 하지만, `lift()` 는
  `normalize_attention` 을 그 카메라의 점들에만 적용한다 (`attention_lifting.py:280`).
  percentile lo/hi가 카메라마다 달라져, 융합 `max` 는 "가장 강한 증거"가 아니라 **가장 관대하게
  스케일된 카메라**를 고른다. README가 정직하게 적어둔 "max가 right_wrist의 orange를 골랐다"의
  원인일 수 있다.
- **F8 · `extract_seeds` 퍼센타일 경로 (`target_grounding.py:183`) — Step 6**
  attention이 희소하면 p95 컷이 0.0이 되어 0값 점이 시드가 될 수 있다. `NO_ATTENTION` 검사는
  완전 평탄한 맵만 거른다.
- **F9 · `lift()` 의 `image_hw` 기본값 (`attention_lifting.py:265`) — Step 5**
  다운샘플 후 `uv` 가 이미지 경계에 못 닿으면 attention 맵이 잘못된 해상도로 리샘플된다.
  경고만 있고 검증이 없다.
- **F10 · status 우선순위 (`to_adapter.py:176-179`) — Step 10**
  `DEGRADED` 가 `NO_TARGET` 보다 먼저라 타깃 없는 degraded 프레임이 `NO_TARGET` 을 가린다.

### 삭제와 함께 소멸하므로 기록만 하고 넘길 것

`geometry.py:255` 의 폴백 순서가 문서와 반대인 것, `constraint_builder.py:466` 의 overflow
집합체가 빈 슬롯과 구분되지 않는 것. 둘 다 씬 primitive 전용이다.

---

## 검증 방법

이 작업은 코드를 고치는 것이 아니라 **판정**하는 것이므로, 검증은 발견마다 다르다.

- **수치 확정** — 의심 단계의 발견은 가능한 한 실행으로 확정한다. E2를 확정한 방식이 표준이다:
  로봇 모델을 로드하고, 문제의 제약 행을 직접 평가하고, 위반 개수와 최악값을 찍는다.
  ```bash
  src/openpi/.venv/bin/python -c "<재현 스니펫>"
  ```
  확정된 스니펫과 그 출력은 로그의 "판정 근거"에 남긴다.
- **회귀 기준선** — 검토 시작 시점의 상태를 로그에 고정한다. 나중에 코드를 고칠 때 비교 대상이 된다.
  ```bash
  src/openpi/.venv/bin/python -m pytest tests/ag3s/ tests/trajopt/ -q   # ag3s 현재 445 passed
  src/openpi/.venv/bin/python -m benchmark.ag3s.pipeline --frames 12 --profile
  ```
- **ESDF 경로 실측** — Step 1~3 에서는 실제로 필드를 만들어 확인한다.
  `benchmark/trajopt/experiments/esdf_rollout.py` 와 `tests/trajopt/test_esdf_backend.py` 가
  이미 그 배선을 하고 있으므로, 새 하네스를 쓰기 전에 이 둘을 먼저 읽고 재사용한다.
- **스텝 완료 조건** — 그 스텝의 발견이 전부 "확정" 또는 "기각"으로 판정되었고, 모듈 처분과
  attention의 역할이 정해졌고, 사용자의 질문이 소진되었고, 로그에 시각이 기록되었을 때만
  다음 스텝으로 넘어간다.

수정은 이 검토의 범위가 아니다. 사용자가 "이건 고치자"라고 지시한 항목만, 지시한 시점에 다룬다.

---

## 시작 지점

승인되면 Step 0을 실행한다: 회귀 기준선을 측정하고, `AG3S_REVIEW_LOG.md` 를 생성해 위의 발견
목록을 "미확정 후보"로 적고, `ag3s ↔ trajopt` 경계 지도를 설명한다. 그리고 **사용자의 질문을
기다린다.**
