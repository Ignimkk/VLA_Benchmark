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
**기능 수정은 이 검토가 끝난 뒤 사용자가 지시하는 범위에서 별도로 진행한다.**

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
