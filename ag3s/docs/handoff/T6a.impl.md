# T6a — 구현

> writer: ag3s-implementer (A1) · 2026-09-26 · 읽는 쪽: verifier, scribe, lead

## 무엇을 했나 (평이한 요약 먼저)

기록기가 **청크와 MuJoCo 참값을 남기게** 했다. planning record 마다 `actions`(서버가 계산한
refined) · `actions_reference`(shadow 일 때만) · `qpos`(66개 전체) · `object_poses`(과일 넷 +
crate 의 위치와 자세) · `max_violation_pair` 가 실리고, control record 마다 `qpos` 가 실린다.

**`max_violation_pair` 는 꺼낼 수 있었다.** 최악 위반을 만든 행은 `linearize.py` 의
`full_violation` 이 min 을 취하던 바로 그 배열 안에 있었고, 같은 자리에서 argmin 을 함께
읽으면 된다. 그래서 `CollisionLinearizer.worst_row` 를 새로 두고 `sqp._finish` 가
`full_violation` **대신** 그것을 부른다 — clearance sweep 도 forward kinematics 도 늘지
않는다(argmin 세 번만 더 돈다). 신원은 구 인덱스가 아니라 **link 이름**이고, ESDF 행이면
label 층이 답하는 **obstacle 이름**까지 붙는다.

**키를 더한 것이고 바꾸지 않았다.** 옛 키는 하나도 움직이지 않았고 completeness 표의 항목도
그대로다. 세 가지를 테스트로 고정했다: 새 키가 실린다 · 옛 키와 completeness 항목이 그대로다 ·
`--record-frames` 없이는 참값 코드가 **한 줄도 돌지 않는다**(호출부를 AST 로 검사한다).

**크기는 컸다. 3.70 MB/episode 다** (75 planning + 600 control, shadow). 아래 "크기 — 실측"
절에 전부 있다. 1 MB 를 넘으므로 줄이지 않고 보고한다 — 줄이는 판단은 lead 의 것이다.

## 바뀐 파일

### 기록기 — 새 키가 실린다

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/runtime/scene_truth.py` (신규) | `ObjectPoseProbe` · `joint_positions` | MuJoCo 참값을 읽는 **유일한** 자리. 기록기는 MuJoCo 를 모르고, `pi05_infer.py` 에 박으면 아무도 테스트할 수 없다 |
| `frame_record.py:220-262` | `planning(actions, actions_reference, qpos, object_poses, max_violation_pair)` | 다섯 키. `actions_reference` **만** 없을 때 키가 빠진다 (shadow 신호) |
| `frame_record.py:276-290` | `control(qpos=...)` | `executed` 는 "실행하기로 했나", `qpos` 는 "그래서 팔이 어디 있었나" |
| `frame_record.py:86-108, 182` | `_jsonable` 을 `_write` 에 물림 | `json.dumps(default=str)` 는 `ndarray` 를 만나도 **예외를 안 내고** `"[[0.1 … 0.9]]"` 를 적는다. 청크를 싣기 시작하면서 그 경로가 실제로 열렸다 |

### `max_violation_pair` — SQP 안에서 꺼냈다

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/trajopt/linearize.py:623-676` | `CollisionLinearizer.worst_row` | `full_violation` 과 **같은 배열**에서 argmin 까지 읽는다. 값이 갈라지면 그 신원은 보고된 숫자의 신원이 아니다 |
| `linearize.py:677-731` | `_identify` · `_query_name` · `_esdf_label_name` | 인덱스를 이름으로. link 이름 / `attached:<link>[i]` / ESDF label 이름 |
| `benchmark/trajopt/sqp.py:277` | `_finish` 가 `full_violation` → `worst_row` | **추가 비용 없음** — 어차피 한 번 부르던 자리다 |
| `sqp.py:325`, `sqp.py:260` | `metrics["max_violation_pair"]` | `_passthrough` 에서도 키를 둔다 (값은 `None`) |
| `benchmark/trajopt/safe_policy.py:521, 528` | `_verdict` 가 metrics 에서 집어 verdict 로 | 서버가 판정과 함께 신원을 싣는다 |
| `benchmark/trajopt/wire.py:144-148, 263-288, 369-379` | `VIOLATION_PAIR` · `SafetyVerdict.max_violation_pair` · `unpack_violation_pair` | 응답의 **선택 키**. 신원이 없는 프레임의 응답은 T0 때와 한 바이트도 다르지 않다 (`ag3s` 블록·`actions_reference` 와 같은 규약) |

### 클라이언트 — 두 청크와 신원을 붙든다

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/trajopt/client.py:96-111` | `last_actions_refined` · `last_actions_reference` · `last_violation_pair` | `infer` 의 반환값과 겸할 수 없다: **shadow 에서 그 반환값의 `actions` 는 reference 로 바뀌어 나간다**. 호출부가 모드마다 다른 키를 물으면 `actions` 의 뜻이 기록에서 갈라진다 |
| `client.py:195-199` | 성공 경로에서 세 값을 붙든다 | shadow 갈래가 `actions` 를 덮기 **전** 자리다 |
| `client.py:344-359` | `_hold` — 응답이 있으면 남기고 없으면 **비운다** | 지난 프레임 것을 남기면 이 프레임이 그것 때문에 멈춘 것처럼 읽힌다 (`last_ag3s` 와 같은 규칙) |
| `last_verdict` 의 키 집합 | **손대지 않았다** | 그 dict 가 T0 기록의 `verdict` 다. 신원은 자기 키로 간다 |

### 호출부

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `pi05_infer.py:1187-1201` | `scene_truth = ObjectPoseProbe(m)` — `if args.record_frames:` **안쪽** | 기록 안 하는 실행은 `scene_truth` 를 import 조차 하지 않는다 |
| `pi05_infer.py:1261` | manifest `extra.object_truth` | 무엇을 쟀고 무엇이 이 모델에 없었나 (`describe()`) |
| `pi05_infer.py:1556-1565` | planning record 의 다섯 키 | 청크는 **클라이언트에서** 가져온다 (`result["actions"]` 는 모드에 따라 뜻이 다르다) |
| `pi05_infer.py:1670` | control record 의 `qpos` | `apply_action` 직후 · `mj_step` **전** — 이 action 이 지령된 순간의 자세 |

## `max_violation_pair` 의 내용

키 집합은 프레임마다 고정이다 (`test_the_key_set_is_fixed` 가 고정한다).

| 키 | 값 | 비고 |
|---|---|---|
| `block` | `candidate` · `plane` · `esdf` | 어느 제약 계열 |
| `clearance_m` | 그 행의 여유거리 | `max_violation_m = max(0, -clearance_m)` |
| `step` | chunk 안 몇 번째 스텝 (0-based) | |
| `query` / `link` | 질의점 번호 / 그 구가 붙은 **link 이름** | 쥔 물체의 점이면 `attached:<parent_link>[i]` |
| `slot` | candidate slot 또는 plane 번호 | ESDF 는 `None` (slot 을 열거하지 않는다) |
| `candidate_id` | 그 slot 을 소유한 AG3S candidate 번호 | candidate 행에서만 |
| `obstacle` | ESDF label 층이 답한 **가장 가까운 물체 이름** | label 없는 backend 에서는 `None` |
| `point_m` | 그 질의점의 base frame 좌표 | 그림에 바로 찍을 수 있다 |

**꺼내지 못한 것 하나**: candidate 행의 **이름**이다. `SceneSnapshot` 은 `candidate_ids`(정수)
까지만 들고 있고 AG3S 쪽 label 을 싣지 않는다. 그래서 candidate 행의 `obstacle` 은 `None` 이고
`candidate_id` 로만 되짚어야 한다. **억지로 이름을 만들지 않았다** — ESDF backend 로 도는
실행(T5·T6 가 그렇다)에서는 `block == "esdf"` 이므로 label 층이 이름을 준다.

`pair` 가 `None` 인 경우는 하나다: **활성 제약이 하나도 없었다** (모든 행이 `+inf`). 그때 응답에
키가 실리지 않고 기록에는 `null` 이 남는다. 없는 신원을 지어내지 않는다.

## 크기 — 실측

`.venv-ag3s` 에서 `FrameRecorder` 로 실제 한 줄씩 쓰고 바이트를 셌다 (H=50, `ACTION_WIDTH`=16,
`nq`=66, body 5개, 값은 정규분포 — 실제 관절값과 자릿수가 같다). `nq` 는
`model_transport.xml`/`model_transport_pick_place_obstacles.xml` 둘 다 **66** 이고 과일 넷 +
crate 는 둘 다 있다 (body id 48~52).

| record | T0 기준 | T6a closed loop | T6a shadow |
|---|---|---|---|
| planning 한 줄 | 561 B | **19 837 B** | **36 704 B** |
| control 한 줄 | 203 B | **1 581 B** | 1 581 B |
| **75 planning + 600 control** | 0.164 MB | **2.436 MB** | **3.701 MB** |

- 지시서의 `3.2 KB/chunk` 는 **float32 binary** 기준이다. JSON 은 float32 값이 double 로
  펼쳐지며 값 하나가 약 20 자가 된다 — `[50, 16]` 이 16 840 B 다 (binary 의 5.3배).
- `qpos` 66개는 1 382 B, `object_poses` 5개는 874 B, `max_violation_pair` 는 약 200 B.

**1 MB 를 넘으므로 줄이지 않고 보고한다.** 참고 수치만 적어 둔다 — 소수점 6자리로 내리면
(1e-6 rad = 0.06 milli-degree, actuator 분해능보다 훨씬 아래) planning 한 줄이 19 837 → 약
10 700 B 가 되고 episode 총량이 **closed 1.336 MB · shadow 1.969 MB** 다. **지금은 full
precision 이다** — 자르는 판단은 lead 의 것이고, 추측으로 줄이지 않았다.

쓰기 비용은 줄당 flush 가 이미 있던 구조이고 planning 이 청크당 한 번(533 ms 예산)이라
19~37 KB 의 `json.dumps` 는 예산 안에 든다. control 은 1.6 KB/step 이다.

## 단위 검증

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work \
  .venv-ag3s/bin/python -u -m pytest tests/ -q
```

**777 passed** (직전 740 + 새 37). 실패 0.

새 테스트 37개:

| 파일 | 개수 | 무엇을 고정하나 |
|---|---|---|
| `tests/ag3s/test_frame_record.py` | 6 | 옛 키 집합 · completeness 표의 항목과 통과 조건 · 새 키가 **값으로** 실린다 · `ndarray` 가 문자열 repr 로 새지 않는다 |
| `tests/ag3s/test_scene_truth.py` | 7 | `data.xpos`/`xquat` 그대로인가 · 없는 body 는 `missing` 이고 예외가 아니다 · 시뮬레이션을 건드리지 않는다 |
| `tests/ag3s/test_record_cost_guard.py` | 4 | **`--record-frames` 없이는 비용이 0** — 참값 이름의 모든 사용이 기록 분기 안에 있고 module-level import 가 없고 probe 가 딱 한 번 만들어진다 (AST 검사) |
| `tests/trajopt/test_violation_pair.py` | 14 | `worst_row` 값이 `full_violation` 과 같다 · 신원이 실제 행을 가리킨다 · link/obstacle 이름 · 빈 씬은 `None` · SQP metrics · 와이어 왕복 |
| `tests/trajopt/test_safe_client.py` (추가) | 6 | shadow 에서 두 청크가 갈라지지 않는다 · 옛 서버는 `None` · hold 에도 남는다 · 응답 없으면 비운다 · `last_verdict` 키 집합 불변 |

**긴 rollout 은 돌리지 않았다.** 회귀 측정과 figure 는 verifier 의 것이다.

## verifier 가 알아야 할 것

- **새로 생긴 플래그·기본값 변경**: 없음. `--record-frames` 의 뜻과 사용법이 그대로다.
- **서버를 다시 띄워야 한다.** `max_violation_pair` 는 `serve_safe`(→ `safe_policy` → `wire`)
  에서 나온다. 포트 8000 의 PID 4004754 는 **옛 코드로 떠 있다** — 그 서버에 붙으면
  `max_violation_pair` 가 전부 `null` 이다 (그것도 정직한 기록이지만 T5f 는 안 닫힌다).
  나는 죽이지 않았다. 재기동은 lead 의 것이다.
- **재생산이 필요한 산출물**: 없음. npz·ESDF 자산은 건드리지 않았다.
- **옛 기록과 호환이 깨지는가**: **아니다.** 새 기록에 키가 더 있을 뿐이고 `plot_t0.py` ·
  `verify_t0.py` · `verify_frame_record.py` · `plot_i4.py` 는 모두 키를 하나씩 이름으로 읽는다
  (키 집합 비교를 하는 곳이 없다 — 확인했다). 반대 방향(옛 `frames.jsonl` 을 새 코드로 읽기)도
  읽는 쪽이 `.get` 을 쓰므로 그대로다.
- **`--trajopt` in-process 경로의 `actions` 는 `None` 이다.** 그쪽의 수정은 planning record
  **다음에** 일어나므로 이 자리에서 적을 수 있는 refined 가 없다. 지어내지 않고 비웠다.
  T5·T6 은 `--safe-remote` 경로이므로 영향이 없다.
- **`object_poses` 의 좌표계는 MuJoCo world frame** 이다 (`data.xpos`/`xquat`). 로봇 base frame
  이 아니다 — 제약 쪽 좌표와 비교할 때 변환이 필요하다. manifest 의
  `extra.object_truth.frame` 에 그대로 적혀 있다.
- **`qpos` 는 자유물체까지 포함한 66개 전체**다. 팔 관절만이 아니다.
- **control record 의 `qpos` 는 `mj_step` 전 값**이다 — 그 action 이 지령된 순간의 자세.

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다. 기대가 보이면 판정이 뒤집힌다. -->

<details>
<summary>펼치지 말 것 (측정 후)</summary>

`seq 38` 이후의 `qpos` 가 control record 사이에서 거의 변하지 않을 것으로 본다 (hold 가
현재 관절을 목표로 주므로). 그렇다면 멈춤은 제어가 아니라 판정 쪽이고, `max_violation_pair` 가
어느 link 에서 나오는지가 다음 단서다. 다만 **그 반대도 가능하다**: `qpos` 가 움직이는데
과제가 진행되지 않는 경우(같은 자리를 왕복)라면 원인이 정책 쪽이다. 나는 어느 쪽인지 모른다.

</details>
