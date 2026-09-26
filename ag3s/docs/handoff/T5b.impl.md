# T5b — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-09-25
> **`T5a`(shadow) 위에 올렸다.** 서버 재기동은 둘을 합쳐 lead 가 한 번만 한다.

## 무엇을 했나 (평이한 요약 먼저)

`degraded` 의 **사유**가 로컬까지 온다. `ag3s_status` 가 `ok` 가 아닐 때만 응답에 선택 키
`ag3s` 블록이 실리고, 그 안에 `status` · `validity` · `grounding_status` · **`reasons`**
(`[{code, detail}]`) · `notes`(전문)가 있다. 로컬은 그것을 HOLD 한 줄에 찍고 프레임마다
기록한다. 서버도 같은 프레임에 한 줄 찍는다.

**원인은 안 고쳤다.** cap 을 올리지도 판정을 느슨하게 하지도 않았다 — 이 STEP 은 보이게 만드는
것뿐이다.

**진짜 원인은 두 개였고 둘 다 "사유 없음" 이 아니었다.**

1. **사유는 이미 거의 다 있었다** — `DEGRADED` 를 내리는 자리는 열 곳 중 아홉이 그 옆에서
   `notes` 에 수치까지 든 문장을 넣고 있었다. **나가는 길이 없었을 뿐이다:** 응답의 `notes` 는
   `TrajOptResult.notes`(**최적화기** 쪽)이고 `CollisionConstraintSet.notes`(**지각** 쪽)는
   서버 프로세스 밖으로 나가는 경로가 하나도 없었다. 이것이 smoke 에서 본 증상의 전부다.
2. **딱 한 자리는 정말로 노트가 없었다** — `multiview.py` 의 *"카메라 중 하나가 `max_points`
   cap 에 걸렸다"* 분기. `validity` 만 내리고 조용히 지나갔다.

## 다섯 자리를 다 읽었다 — 실제로는 **열** 이다 (지시서가 요구한 표)

지시서가 준 다섯(`pipeline.py:346·443·538·662·669`)을 읽고 나서 `ConstraintValidity.DEGRADED`
를 전부 grep 했다. **다섯이 아니라 열이다.** `:662`/`:669` 는 같은 분기(status 와 validity)이고,
`pipeline` 에 두 자리(`_esdf_coverage`)와 `multiview` 에 세 자리, `to_adapter` 에 한 자리가 더
있다. 줄번호는 코드를 넣은 **뒤** 기준이다.

| 자리 (수정 후) | 무슨 조건인가 | 코드 | 노트가 있었나 |
|---|---|---|---|
| `pipeline.py:339` (옛 `:338/:346`) | 단일/융합 점군이 `pointcloud.max_points` cap 에 걸려 voxel 을 키웠다. **거칠어졌을 뿐 빠진 것은 없다** (점유 셀마다 대표가 남는다 — 버리는 경로는 `INCOMPLETE` 다) | `pointcloud_capped` | 있었다 |
| `pipeline.py:439` (옛 `:437/:443`) | 클러스터가 `collision_candidate.max_clusters`/`unknown_min_points` 를 넘어 보수적 aggregate 로 접혔다 | `candidate_overflow` | 있었다 |
| `pipeline.py:535` (옛 `:532/:540`) | ESDF 미관측 비율이 `esdf.unknown_report_threshold` 이상. **dense 계층에만 있는 판정** — block-sparse 는 `unknown_fraction` 이 `None` 이라 이 분기에 안 들어오고 노트만 남긴다 | `esdf_unknown_fraction` | 있었다 |
| `pipeline.py:673-680` (옛 `:662/:669`) | `self._builder is None` — 로봇 모델이 안 주입돼 제약을 쓸 대상이 없다. 기하는 있는데 제약이 없는 것을 `VALID` 로 두면 소비자가 "제약 없음" 을 "깨끗함" 으로 읽는다 | `no_robot_model` | 있었다 |
| `pipeline.py:743` (`_esdf_coverage` G3) | 어떤 카메라가 이 프레임에 voxel 을 **하나도** 안 썼다 (정상은 대당 13k~150k) — extrinsics · 죽은 드라이버 · depth 범위 밖 | `esdf_dead_camera` | 있었다 |
| `pipeline.py:788` (`_esdf_coverage` G4) | 제약 구 일부가 **coverage 격자** 밖. 거리장이 그 구에 무엇이 있든 `outside_distance` 를 답하므로 제약을 받지 않는다 (E4) | `spheres_outside_grid` | 있었다 |
| `multiview.py:195` | 어떤 카메라의 `robot_state` 가 그 이미지 촬영 시각에서 `timing.max_state_age_sec` 이상 떨어졌다 — 클라우드가 엉뚱한 자세에 놓인다 | `camera_state_stale` | 있었다 |
| `multiview.py:203` | 어떤 카메라의 이미지가 프레임 기준 시각보다 `timing.max_transform_age_sec` 이상 뒤처졌다 | `camera_transform_stale` | 있었다 |
| `multiview.py:210` | 카메라 간 촬영 시각 차이가 `timing.max_camera_skew_sec` 초과 | `camera_skew` | 있었다 |
| `multiview.py:224` | `timing.expected_cameras` 중 보고하지 않은 카메라가 있다 | `camera_missing` | 있었다 |
| **`multiview.py:389`** | **카메라 중 하나가 자기 점군에서 `max_points` cap 에 걸렸다** | `fused_pointcloud_capped` | **없었다 — 이 STEP 이 만든 유일한 새 노트** |
| `to_adapter.py:157` | 제약 구가 `builder.max_candidates` 를 넘어 overflow 슬롯으로 접혔다 | `constraint_sphere_overflow` | 있었다 |

`INCOMPLETE` 를 내는 네 자리(`pipeline:352·450·499`, `to_adapter:152`)는 **전부 노트가 있다**
(읽고 확인했다). 이 STEP 의 범위가 `degraded` 라 코드는 안 달았다 — 달려면 같은 방식이다.

## 왜 병렬 자료구조를 만들지 않았나 (설계 판단)

`notes: list[str]` 는 `CollisionConstraintSet` · `FusionResult` · 기록 npz · 그림 스크립트까지
이미 관통한다. 여기에 `degradations: tuple[Degradation, ...]` 를 따로 만들면 **자리마다 두 개를
갱신해야 하고, 빠뜨린 쪽이 곧 "사유가 빈 경로"** 가 된다 — 막으려는 그 실패다. 게다가
`check_freshness` 는 `(validity, notes, metrics)` 를 돌려주는 공개 함수라 arity 를 바꾸면
기존 테스트가 깨진다.

그래서 **매체는 `notes` 그대로 두고 문장 앞에 코드를 붙였다**: `"<code>: <detail>"`.
`reason()` 이 `CODES` 에 없는 코드를 **거절**하므로 새 분기를 만드는 사람이 등록을 피해 갈 수
없고, 등록 한 줄이 곧 "이 코드가 어느 분기인가" 의 문서다. 옛 소비자는 문장이 그대로라 안
깨진다 (`"capped at max_points" in note` 같은 기존 테스트 둘이 그대로 통과한다).

## 사유가 빈 경로를 **구조로** 없앴다

`degradation.ensure_reason(notes, degraded=...)` 를 **두 관문**에 뒀다:
`to_adapter.py:198`(모든 정상 경로의 `validity` 가 여기서 확정된다)과
`pipeline.py:673`(`build_constraint_set` 을 거치지 않는 유일한 경로).

degraded 인데 코드가 하나도 없으면 `degraded_without_reason` 을 달아 보낸다. **예외를 던지지
않는다** — 진단이 빠진 것으로 지각을 죽이면 그 프레임의 기하가 통째로 사라지고 그게 더 나쁘다
(`unpack_camera_observations` 가 카메라 누락에 예외를 안 내는 것과 같은 판단). 그 코드가 보이면
씬의 성질이 아니라 **AG3S 의 배선 결함**이고, 단위 테스트는 그것이 **나오지 않는 것**을 검사한다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/runtime/degradation.py` (**새 파일**) | `CODES` 등록부 · `reason()` · `reasons()` · `codes()` · `explain()` · `ensure_reason()` | 코드 문자열이 한 곳에만 있게. 등록되지 않은 코드는 못 쓴다 |
| `benchmark/ag3s/runtime/pipeline.py:339,439,535,673,743,788` | 여섯 자리에 코드. `:673` 은 `ensure_reason` 도 | 위 표 |
| `benchmark/ag3s/runtime/multiview.py:195,203,210,224` | 네 자리에 코드 | 위 표 |
| **`benchmark/ag3s/runtime/multiview.py:378-396`** | **없던 노트를 만들었다** — 어느 카메라가 몇 점에서 cap 에 걸려 voxel 이 얼마로 커졌는지 | 수치가 없으면 다음 사람이 다시 서버에 들어가서 재야 한다 |
| `benchmark/ag3s/constraints/to_adapter.py:157,198` | 코드 + **마지막 관문** | |
| `benchmark/trajopt/wire.py:100,44-70` | `AG3S_BLOCK = "ag3s"` 상수 + 응답 표의 행·절 | 와이어의 정본은 이 파일 머리말이다 |
| `benchmark/trajopt/wire.py` (`pack_response`) | `ag3s=None` 인자 — **`None` 이면 키를 안 싣는다** | `ag3s_status == "ok"` 인 응답이 T0 과 같아야 한다 |
| `benchmark/trajopt/wire.py` (`unpack_ag3s`) | 없으면 `{}`. `unpack_field` 처럼 상태 객체를 만들지 않는다 | `ag3s_status != "ok"` + 빈 블록 = 옛 서버. 읽는 쪽이 그 조합을 볼 수 있어야 한다 |
| `benchmark/trajopt/safe_policy.py` (`_ag3s_block`) | `ok` 면 `None`, 아니면 `status`·`validity`·`grounding_status`·`reasons`·`notes`. 제약 집합이 아예 없으면 `refiner.last_failure` 를 싣는다 | `no_geometry` 만 나가면 **왜 지각이 안 돌았는지**가 다시 서버 안에 남는다 |
| `benchmark/trajopt/safe_policy.py` (`infer`) | 서버 로그에 한 줄 (`[safe_policy] seq=… ag3s=… — <사유>`) | smoke 때 **서버 로그에도 없었다**. 로컬 기록을 못 얻는 세션에서 같은 일이 되풀이된다 |
| `benchmark/trajopt/client.py` (`last_ag3s`) | 블록을 붙든다. **hold 일 때도.** 응답이 아예 없으면 `{}` 로 비운다 | 지난 프레임 사유가 남으면 이 프레임이 그 사유로 멈춘 것처럼 읽힌다 |
| `benchmark/trajopt/client.py` (`_explain`) | 인증 실패 문장에 `grounding=` 과 사유를 붙인다 (`detail_chars=64` 로 자르고, 자른 것은 `…`+`+N more` 로 말한다) | HOLD 한 줄이 프레임마다 찍히므로 전문을 넣으면 다른 것을 다 밀어낸다. 전문은 블록과 기록에 있다 |
| `pi05_TO_hybrid/rby1_bringup/pi05_infer.py` (observation frame) | `grounding_status` 를 **실제 값**으로. `notes` 를 서버 노트로. `extra` 에 `ag3s_reasons`·`ag3s_reason_codes` | 이 자리에 `"unavailable-on-client"` 와 *"T1 에서 와이어에 싣는다"* 가 적혀 있었다. 그 T1 이 이것이다 |
| 같은 파일 (planning frame) | `extra` 에 `ag3s_reason_codes` | 청크 단위로도 센다 |

## 기본 동작을 바꾸지 않았다 (요구사항 1)

* `ag3s_status == "ok"` → `ag3s` 키가 **아예 없다.** `pack_response` 수준에서 키 집합을 상수로
  고정해 테스트했다.
* `SafetyVerdict` · `last_verdict` 의 키 집합을 **건드리지 않았다** — T0 기록의 `verdict` 가 그
  모양이다. 블록은 `last_ag3s` 라는 **별도** 속성으로 들고, 기록에는 `extra` 로 넘긴다.
* 노트 문장은 코드 접두만 붙었고 내용은 그대로다.
* `T5a` 의 `actions_reference` 와 충돌하지 않는다 — **두 선택 키가 한 응답에 같이 실릴 수
  있음**을 테스트로 고정했다 (shadow 실행의 unsafe 프레임이 정확히 그 조합이고, 그것이 T5 에서
  보려는 프레임이다).
* `T5a` 의 `test_without_shadow_no_reference_key_is_added` 만 손봤다. 그 테스트는 **카메라가
  없는** 합성 씬을 쓰는데 그 프레임은 `ok` 가 아니어서 `ag3s` 블록이 정당하게 생긴다. 기대 키
  집합에 그것을 더하고, `ok` 인 응답이 T0 과 같다는 쪽은 `pack_response` 수준의 새 테스트가
  지킨다 — 합성 씬으로 `ok` 를 만들 수 없으므로 와이어에서 고정하는 것이 맞다.

## 단위 검증

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -u -m pytest tests/ -q
```

**720 passed** (`T5a` 뒤 692 + 새로 28). 새 테스트:

* `tests/ag3s/test_degradation_reason.py` (19) — **자리별로 분기를 태워 그 자리의 코드가 나오는지**
  본다: `camera_skew` · `camera_state_stale` · `camera_transform_stale` · `camera_missing` ·
  `fused_pointcloud_capped` · `pointcloud_capped` · `no_robot_model` · `esdf_dead_camera` ·
  `spheres_outside_grid`. 코드가 **나오는 것만으로는 부족하다** — 엉뚱한 코드가 나오면 다음
  사람이 엉뚱한 곳을 보므로 기대 코드를 이름으로 못 박았다. 등록 안 된 코드는 `KeyError`.
  `ensure_reason` 이 사유 없는 degraded 를 채우는 것과, **정상 시나리오에서
  `degraded_without_reason` 이 안 나오는 것**(= 자리별 배선이 살아 있다) 둘 다 검사한다.
* `tests/trajopt/test_safe_policy.py` (+4) — `ok` 면 키 없음 / 블록 왕복 / **`actions_reference`
  와 공존** / 실제 `SafePolicy` 가 non-ok 프레임에 사유를 싣는다.
* `tests/trajopt/test_safe_client.py` (+5) — HOLD 사유가 코드·수치·`grounding` 을 담는다 /
  hold 에서도 블록을 붙든다 / **옛 서버면 "블록을 모르는 서버" 라고 말한다**(빈칸으로 두면
  "사유 없는 degraded" 와 구별되지 않는다) / 정상 프레임은 `{}` / 다음 프레임이 지난 사유를
  물려받지 않는다.

`G3`·`G4`(`_esdf_coverage`)는 실제 cuRobo 필드가 있어야 나는 자리라, 그 검사에게 보여 주는
면(`stats["per_camera"]` · `coverage_grid` · `outside_distance`)만 흉내내 **분기 자체**를 태웠다.

**실제 `SafePolicy` → `SafeRemoteClient` 한 왕복**을 한 프로세스에서 돌려 눈으로 봤다 (합성 씬,
임시 스크립트, cap 을 낮춰 degraded 를 만들었다 — 원인을 고치는 것이 아니라 사유가 오는지 보려는
것이다). 세 코드가 수치와 함께 왔다: `fused_pointcloud_capped` · `pointcloud_capped` ·
`esdf_unknown_fraction`, `grounding=no_attention`. HOLD 한 줄이 그 셋을 이름으로 말한다.

## verifier 가 알아야 할 것

* **새 플래그 없음.** `T5b` 는 플래그가 아니라 항상 켜진 진단이다. `ag3s_status != "ok"` 인
  프레임에만 블록이 붙는다.
* **재기동이 필요하다** — `T5a`+`T5b` 합쳐 **한 번.** lead 가 한다. 포트 8000 의 서버(PID
  1191230, 실행 45 분)는 건드리지 않았다.
* **smoke 의 `degraded` 원인은 이제 서버 로그 첫 줄에 나온다.** 재기동 뒤 `[safe_policy] seq=…
  ag3s=degraded (validity=…, grounding=…) — <코드>(…)` 를 찾으면 된다. 로컬에서는 HOLD 줄과
  프레임 기록의 `ag3s_reason_codes` 다. **그 코드가 무엇으로 나오는지가 다음 판정의 입력이고,
  고치는 것은 사용자 판정 뒤다.**
* **옛 기록과 호환** — 깨지지 않는다. 응답은 non-ok 프레임에 키가 하나 **늘고**, frame record 는
  `ag3s_reasons`·`ag3s_reason_codes` 가 늘고(정상 프레임에서는 빈 목록), `grounding_status` 가
  `"unavailable-on-client"` 대신 실제 값이 들어간다. `plot_t0.py` 는 `kind`·`t_step`·`field`·
  `camera_skew_sec` 만 `.get()` 으로 읽으므로 영향이 없다.
* **`ag3s_reason_codes` 는 키를 항상 둔다** (빈 목록이어도). shadow 키를 조건부로 둔 것과 다른
  선택이고 이유가 다르다: 이쪽은 A2 가 프레임을 **세로로 세는** 값이라 있다/없다가 섞이면
  파서가 두 갈래로 갈린다.
* **`degraded_without_reason` 이 실측에 나오면 그것은 씬이 아니라 코드 결함이다.** 보이면
  바로 알려 주기 바란다 — 내가 놓친 분기가 있다는 뜻이다.
* 측정은 안 했다. 회귀 기준선·clearance·지연 전부 A2 다.

## 내가 기대하는 결과

> **verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.** 기대가 보이면 판정이 뒤집힌다 (C2·D2).

`ok` 프레임의 응답이 그대로이므로 세 회귀 기준선(`위반으로 시작 14/15` · `has_target 9/15` ·
`frame0 clearance_before +0.157 mm`)은 재현될 것으로 본다. smoke 의 `t=8` 에서 어느 코드가
나올지는 **예상하지 않는다** — 그것을 모르는 것이 이 STEP 이 있는 이유다. 다만 `max_violation_m`
이 0.0 이었으므로 후보는 궤적이 아니라 **관측** 쪽일 것이다 (`spheres_outside_grid` ·
`esdf_dead_camera` · 카메라 신선도 셋 중 하나). `X3`(점군 cap)는 `serve_safe.py` 가
`max_points` 를 기본값 200000 으로 두므로 후보에서 빠진다 — 그것은 lead 가 이미 확인했다.
