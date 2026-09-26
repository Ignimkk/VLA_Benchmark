# T5e — 구현

> writer: ag3s-implementer (A1) · 2026-09-25 · 읽는 쪽: verifier, scribe, lead

## 무엇을 했나 (평이한 요약 먼저)

**target 을 점수로 거부하지 않고 1 등으로 고른다. 그리고 다른 물체가 target 을 가져가려면
3 프레임 연속 1 등이어야 한다.**

두 조각은 서로 독립이고 따로 끌 수 있다.

1. **점수 거부를 끈다** — `target_score_threshold` 의 기본값을 `0.25` → `0.0` 으로 내렸다.
   비교가 `<` 이고 점수는 `w_geometry > 0` 이면 항상 양수라 **0.0 은 아무것도 거부하지 않는다.**
   `LOW_SCORE` 분기와 설정 키는 **그대로 있다** — 0 보다 큰 값을 주면 예전과 똑같이 거부한다.
   지운 것은 하나도 없고, 판정을 되돌리는 데 필요한 것은 숫자 하나뿐이다.
2. **프레임 간 hysteresis 를 새로 넣었다** — `TargetConfirm` 이 "지금 target 인 물체의 무게중심"
   하나만 기억한다. 1 등이 그 물체면 계속 쓰고, 다른 물체면 그 도전자가 `target_confirm_frames`
   (3) 프레임 **연속** 1 등이 될 때까지 기존 물체를 유지한다.

**cluster 가 0 개면 여전히 target 이 없다** (`NO_CLUSTER`/`NO_GEOMETRY`/`NO_SEED`/`NO_ATTENTION`
넷은 그대로다). `target_score` 계산·`metrics` 기록·`runner_up_score` 키도 그대로다.

### confirm 상태를 어디에 두었나 — **`ag3s/stages/target_grounding.py` 의 새 클래스, 소유자는 AG3S pipeline**

`grasp_latch.py` 의 `_Confirm` 을 재사용하지 않았다. 근거 넷:

| | 왜 안 되나 |
|---|---|
| **잠금 대 hysteresis** | `_Confirm` 은 한 번 `locked` 되면 `release()` 전까지 **절대 안 바뀐다.** 조작 대상에는 맞지만 target 에는 틀리다 — ep1807 의 target 은 apple **다음에** crate 로 정당하게 바뀐다. 필요한 것은 "바뀌지만 한 프레임으로는 안 바뀐다" 다 |
| **점수 문턱이 박혀 있다** | `_Confirm.update` 는 매 프레임 `score >= score_ratio * runner_up` 를 먼저 본다 (`grasp_latch.py:143`). T5e 가 없애려는 바로 그 점수 거부다. 1·2 위 격차가 꾸준히 1.1 배인 구간에서는 **영원히 안 바뀐다** |
| **이름이 없다** | `_Confirm` 은 `label` 문자열을 비교하는데 grounding 에는 이름이 없다 — `TargetGeometry.id` 는 그 프레임의 cluster 번호라 프레임 간에 뜻이 없다. 그래서 정체는 `CentroidIdentity` 와 **같은 규칙·같은 tolerance(0.06 m)** 로 무게중심 근접으로 판정한다 |
| **의존 방향** | `benchmark/ag3s` 는 `benchmark/trajopt` 를 import 하지 않는다 (반대로 `trajopt/safe_policy.py` 가 pipeline 을 쓴다). 위로 손을 뻗으면 방향이 뒤집힌다 |

상태의 **소유자**는 AG3S pipeline 이다 (`pipeline.py:113`, `reset()` 에서 청소 — `pipeline.py:157`).
이유: `reset()` 이 곧 에피소드 경계이고, 그 경계를 아는 것은 pipeline 뿐이다. `ground_target` 은
**여전히 프레임 독립**이다 — `confirm=` 를 주지 않으면 예전과 한 글자도 다르지 않게 동작하므로
report·diagram·ablation 스크립트는 전부 순수 함수를 그대로 받는다.

**무게중심은 물체를 따라간다.** 잡힌 사과는 움직인다. 처음 본 자리에 못을 박으면 몇 프레임 뒤
tolerance 를 벗어나 정체가 끊긴다. `CentroidIdentity` 가 앵커를 갱신하지 **않는** 것은 그것이
영구 잠금을 먹이기 때문이고, 여기서는 결정을 매 프레임 다시 하므로 추적하는 쪽이 안전한 오류다.

## 바뀐 파일

> 모두 `/mnt/dev/work/benchmark/` repo (별도 git). 커밋하지 않았다.

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/config.py:105` | `target_score_threshold` 기본값 `0.25` → **`0.0`** | 이 한 줄이 35/51 chunk 에 `LOW_SCORE` 를 붙이고 있었다. 키는 남긴다 |
| `benchmark/ag3s/config.py:112` | **새로** `target_confirm_frames: int = 3` | 흔들림을 누르는 값. `GraspLatch` 의 `confirm_frames` 와 같은 값·같은 근거 |
| `benchmark/ag3s/config.py:117` | **새로** `target_identity_tolerance: float = 0.06` | "같은 물체인가" 의 반경. `CentroidIdentity.tolerance` 와 같은 값 |
| `benchmark/ag3s/config.py:139-150` | `validate()` 에 셋 검사 추가 | `threshold >= 0` · `confirm_frames >= 1` · `tolerance > 0` |
| `benchmark/ag3s/configs/default.yaml:25-27` | 같은 셋 (문서용 사본) | `test_config.py` 가 dataclass 와 일치를 강제한다 |
| `benchmark/ag3s/configs/rby1_three_camera.yaml:55-57` | 같은 셋 | live profile 문서. **코드에서 이 yaml 을 읽는 곳은 없다** (아래 참조) |
| `benchmark/ag3s/stages/target_grounding.py:250` | **새 dataclass** `ConfirmDecision` | 어느 cluster 를 왜 골랐나 — `mode`·`rank`·`streak`·`frames` |
| `benchmark/ag3s/stages/target_grounding.py:276` | **새 class** `TargetConfirm` | 프레임 간 상태 전부. 모듈 docstring 에 설계 근거 |
| `benchmark/ag3s/stages/target_grounding.py:462` | `ground_target(..., confirm=None)` | keyword-only·기본 None → **기존 호출자 전부 무변경** |
| `benchmark/ag3s/stages/target_grounding.py:563` | `LOW_SCORE` 분기 **그대로**, 위치도 그대로 (confirm 앞) | 거부한 프레임이 전환 카운트를 올려서는 안 된다 |
| `benchmark/ag3s/stages/target_grounding.py:571-576` | 고른 cluster = `confirm.select(clusters).cluster`, 없으면 `clusters[0]` | 1 등 선택 |
| `benchmark/ag3s/stages/target_grounding.py:599-600` | `metrics` 에 **`target_rank`·`target_confirm_streak`** 추가 | hysteresis 가 실제로 일한 프레임을 로그에서 구별할 수 있게 |
| `benchmark/ag3s/stages/target_grounding.py:616` | `runner_up_score` = **고른 것을 뺀 나머지 중 최고점** | 1 등을 골랐을 때는 `clusters[1]` 과 **완전히 동일**. hold 프레임에서는 `clusters[1]` 이 곧 target 이라 그대로 두면 비율이 항상 1.0 이 되어 `GraspLatch` 의 애매함 판정이 죽는다 |
| `benchmark/ag3s/runtime/pipeline.py:113·157·405` | `TargetConfirm` 하나를 들고 `reset()` 에서 청소, `ground_target` 에 넘김 | 상태의 소유자 |
| `benchmark/ag3s/runtime/pipeline.py:417-426` | `hold`/`switch`/`unobserved` 일 때만 **평범한 note** 한 줄 | `reason(...)` 이 **아니다** — 저하가 아니다. validity 를 건드리지 않는다 |
| `tests/ag3s/test_target_confirm.py` (새 파일, 18 개) | T5e 전부 | 아래 |
| `tests/ag3s/test_pipeline.py:159·187` | pipeline 에 배선됐는지 + `reset()` | 같은 기하·attention 만 sphere→box 로 옮겨 3 프레임을 센다 |
| `tests/ag3s/test_target_grounding.py:249` | **기존 테스트 하나를 고쳤다** | 아래 "깨진 테스트" |

### 결정의 다섯 갈래 (`ConfirmDecision.mode`)

| mode | 언제 | 무엇을 내나 |
|---|---|---|
| `first` | 아직 잡고 있는 것이 없다 (에피소드 첫 target) | 1 등. 즉시 |
| `keep` | 1 등이 잡고 있는 그 물체다 | 1 등 (무게중심 갱신) |
| `hold` | 다른 것이 1 등인데 연속이 `frames` 미만 | **잡고 있던 물체** (rank ≥ 1) |
| `switch` | 도전자가 `frames` 연속 1 등 | 1 등. target 이 넘어간다 |
| `unobserved` | 다른 것이 1 등이고 **잡고 있던 물체가 이 프레임에 없다** | 1 등. **카운트는 안 지운다** |

`unobserved` 가 필요한 이유: 없는 기하를 target 이라고 부르지 않는다. 그러면서도 카운트를
유지하므로 정말 사라진 물체는 예정대로(3 프레임) 넘겨준다.

### 깨진 테스트 하나 — 이것이 이 변경의 **대가**다

`tests/ag3s/test_target_grounding.py:249`
(`test_without_the_support_surface_removed_grounding_declines`) 가 유일하게 깨졌고,
**의도된 것이라 판단해 테스트를 새 동작으로 고쳤다** (이름도 `..._the_flood_becomes_the_target` 로).

- 옛 주장: `exclude_mask` 없이 돌리면 연결 성장이 테이블 전체로 번지는데, 그 덩어리가 너무
  안 뭉쳐서 **점수가 거부**하므로 `LOW_SCORE` 가 난다 — 즉 점수가 2 차 방어선이었다.
- 지금: 그 덩어리(10,000 점 이상, `spatial_compactness < 0.01`)가 **target 이 된다** — 그 점수는
  단위 테스트 픽스처에서 0.25 보다 훨씬 낮게 관측된다(픽스처 값이고 기준선 수치가 아니다).
  **테이블 홍수를 막는 것은 평면 적합이지 점수가 아니다.** 파이프라인은 `support_mask` 를 항상
  넘기므로(`pipeline.py:402`) live 경로는 덮여 있지만, `exclude_mask` 를 주지 않는 호출자에게는
  2 차 방어선이 없어졌다. 고친 테스트가 그 사실 자체를 주장하고, `threshold=0.25` 를 주면
  예전처럼 거부되는 것도 같은 테스트에서 확인한다.

## 단위 검증

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -u -m pytest tests/ -q
```

**740 passed** (직전 **720** + 새로 더한 **20** = 새 파일 `test_target_confirm.py` 18 + `test_pipeline.py` 2).
기존 테스트는 하나를 **고쳤고**(아래) 지운 것은 없다. `tests/` 전체 단독 실행, 실패 0.

지시서가 요구한 넷은 전부 테스트가 있다:

| 요구 | 테스트 |
|---|---|
| 1 등이 threshold 아래여도 target | `test_target_confirm.py:112` (점수 < 0.25 를 **주장**하고, 같은 프레임을 `threshold=0.25` 로 돌려 `LOW_SCORE` 가 나오는 것도 `:122`) |
| 3 번 연속이어야 바뀐다 (2 번은 안 바뀐다) | `:154` (2 번) · `:165` (3 번) · `:175` (2+1+2 로는 안 바뀐다 — 연속이라는 말의 뜻) |
| cluster 0 개면 target 없음 | `:131` (`NO_CLUSTER`) · `:137` (`NO_GEOMETRY`) · `:145` (`select([])` 는 예외) |
| `obj0` 정체 유지 | `:255` — `CentroidIdentity` 를 통과시켜 5 프레임 전부 `obj0`. **대조군**으로 confirm 없이 같은 5 프레임을 돌려 `obj0·obj1·obj1·obj0·obj0` 이 나오는 것까지 박았다 |

그 밖에: `first` 프레임은 즉시 잡는다(`:190`) · `reset()`(`:202`) · `confirm_frames=1` 이 옛 동작
(`:211`) · `unobserved`(`:218`) · 움직이는 물체 추적(`:232`) · `runner_up_score`(`:272`) ·
**confirm 없이는 여전히 프레임 독립**(`:291`) · tolerance 안에 둘이 있으면 가까운 쪽(`:321`).

## verifier 가 알아야 할 것

- **기본값 변경 셋** (새 CLI 플래그는 없다):
  `clustering.target_score_threshold` **0.25 → 0.0** · `clustering.target_confirm_frames` **3** (새로) ·
  `clustering.target_identity_tolerance` **0.06** (새로).
  회귀 기준선 명령은 `AG3SConfig.from_dict` 로 **dataclass 기본값에서 시작**하므로
  (`esdf_rollout.py:182`) yaml 을 거치지 않고 바로 적용된다. `configs/*.yaml` 을 읽는 코드는 없다.
- **`ground_target` 시그니처는 하위 호환**이다. `confirm=` 은 keyword-only·기본 None.
  `confirm` 없이 부르는 곳(report·diagram·studies 전부)은 **동작이 바뀌지 않는다** —
  단 `target_score_threshold` 기본값 변경은 그쪽에도 적용된다.
- **새로 볼 수 있는 것**: target 이 있는 프레임의 `metrics["target_rank"]`(0 이 아니면 1 등이
  아닌 것을 유지했다는 뜻) · `metrics["target_confirm_streak"]` · `constraint_set.notes` 의
  `target hold (rank 1, challenger at 2/3 frames)` 같은 줄. **`reason(...)` 이 아니라 평범한
  note 라서 `ConstraintValidity` 와 `geometry_certified` 에 영향이 없다.**
- **재생산이 필요한 산출물: 없음.** npz·기록·자산을 건드리지 않았다.
- **옛 기록과 호환: 깨지지 않는다.** 읽는 형식을 바꾸지 않았다.
- **회귀 기준선 셋 중 `has_target 9/15` 는 이 변경으로 바뀐다 — 회귀가 아니라 의도다.**
  나머지 둘에 대한 내 예상은 아래 절에 있고, **측정 전에 읽지 마라.**
- 관련이 있지만 **안 건드린 것**: `benchmark/trajopt/safe_policy.py:411` 의
  `column = 6 if str(hand) == "left" else 13` — 14D 배치이고 16D 는 `wire.gripper_columns()` 가
  `(7, 15)` 를 준다. **봤고 안 건드렸다** (별건, 이번 STEP 아님). `camera_skew` /
  `camera_transform_stale` 판정과 self-filter 도 **안 건드렸다** (사용자 판정).

## 내가 기대하는 결과

> **verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.** 아래는 전부 **코드를 읽은 예상**이고
> **측정값이 아니다.** 하나라도 숫자로 쓰지 마라 — 재는 것은 A2 다.

| 기준선 | 예상 | 코드 근거 |
|---|---|---|
| `has_target 9/15` | **올라간다.** 옛 6 건 중 `LOW_SCORE` 였던 것은 전부 target 이 생긴다. 6 건 전부가 `LOW_SCORE` 였다면 15/15 | 거부 분기가 `target_score_threshold=0.0` 에서 절대 참이 안 된다 (`target_grounding.py:563`). `NO_SEED`/`NO_CLUSTER`/`NO_ATTENTION` 이었던 프레임은 **안 변한다** — 그래서 15/15 를 단정하지 않는다 |
| `위반으로 시작 14/15` | **안 바뀌는 쪽에 기운다.** 다만 바뀔 수 있는 경로가 하나 있다 | 이 개수는 clearance 만이 아니라 **기하 인증**도 본다 (X3, `sqp.py:287-293`). 내가 더한 note 는 `reason(...)` 이 아니라 validity 를 안 건드린다. 바뀔 수 있는 경로: target 이 새로 생긴 프레임은 그 물체가 phase 규칙에 따라 접촉 허가를 받아 clearance 가 완화되고 필드에서 파일 수 있다(`pipeline.py` 의 `_build_esdf(..., grounding.target)`). **그 물체 때문에 위반이던 프레임이라면 내려갈 수 있다** |
| frame0 `clearance_before` +0.15718632962849477 mm | **frame 0 에 이미 target 이 있었다면 완전히 같다.** 없었다면 바뀔 수 있다 | frame 0 은 항상 `first` 분기다 — 잡고 있는 것이 없으니 1 등을 즉시 쓴다. 옛 동작도 (점수를 통과했다면) 1 등을 썼다. 즉 **frame 0 에서 hysteresis 는 정의상 아무 일도 하지 않는다.** 남는 차이는 "옛날엔 `LOW_SCORE` 였나" 하나뿐이고, 나는 ep1800 frame 0 의 옛 status 를 모른다 (X3·T1 의 verify.json 에 프레임별 기록이 없다) |

추가로 **새로 볼 수 있을 것으로 기대하는 것** (기준선 아님, 있으면 hysteresis 가 일한 증거):
`notes` 의 `target hold ...` 줄과 `metrics["target_rank"] == 1.0` 인 프레임. 15 프레임 rollout
에서 몇 건이 나올지는 **모른다** — ep1800 의 1 등 순서를 측정하지 않았다.

**내가 재지 않은 것**: 회귀 기준선 · 긴 rollout · fine layer 5 mm 가 실제로 붙는지.
단위 테스트까지가 내 몫이다.
