# T43HM — 구현: 손에 든 사과 row 에 손가락과 같은 분류 · margin (`--held-obstacle-margin`)

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-10-07 · 기준 benchmark `a297155`
> task: [T43.task.md](T43.task.md) §"사용자 판정 (2026-10-07) — HM" · 오프라인 산출물: `outputs/impl/T43HM/` (p/ 스크립트 · raw/ 결과 · fig/)

## 무엇을 했나 (평이한 요약 먼저)

쥔 사과의 질의 구 행도 손가락 행처럼 "최근접 표면이 무엇인가" 로 가른다. 장애물(허들 등)이면
`max(그 행의 margin, obstacle_margin)` = 10 mm 를 받는다. 두 가지는 margin 을 받지 않는다.
- **지지면(테이블)**: 손가락 행과 같은 평면 판정 (2 voxel). `--obstacle-margin-support` 와 관계없이 **늘** 뺀다 (T34 J1: 쥔 구는 테이블 띠 8.7 mm 만큼만 올라가 있다).
- **사과 자신의 잔상 (O7)**: 최근접 표면점 `p − d·∇d/|∇d|` 가 **attach 때 AG3S 가 TSDF 에서 비운 부피** (capture 자세의 fit 구 + 잡기 전 자리, `AG3S.held_capture_spheres`) + 2 voxel (10 mm) 안이면 사과 자신이다.

목적지 규칙(20 mm)은 그대로이고 겹치면 큰 값이다. flag 는 기본 off, off 면 비트 동일하다.
오프라인 (V7g · V8 기록, 생산 코드로 재계산): 운반 중 사과–막대 접촉 8 건 모두 **계획 자세에서 TO 가 위반을 보게 된다** (off +3.8…+8.2 → on −1.8…−6.2 mm). attach · 들어 올림 105 청크에서 새 위반 0. 대신 운반 · crate 진입에서 TO 가 계획을 고쳐야 하는 청크가 생기고, 그 일부는 **장애물이 아닌 표면(사과 자신의 상 · 정체 불명 잔상)** 때문이다 (§오프라인 (c), 열린 위험 1).

## 규칙

`linearize.CollisionLinearizer._held_rows_class` (held 열만, `held_obstacle_margin` 일 때):

| 순서 | class | 조건 | margin |
|---|---|---|---|
| 1 | `support` | 어느 지지면 k: `|d − (n_k·p − o_k)| ≤ 2 voxel` (항상) | 기존 그대로 (0 또는 목적지 20) |
| 2 | `held` (자기 잔상) | `∇d ≠ 0` 이고 최근접 표면점이 `held_capture_spheres` + 2 voxel 안 | 기존 그대로 |
| 3 | `obstacle` | 나머지 | `max(기존, obstacle_margin)` |

- target 검사 (`|d − d_object| ≤ voxel`) 는 held 열에 하지 않는다 — 비교 대상이 자기 자신이다.
- capture 부피가 없거나 (prebuilt attach, `bringup.py` 경로) 기울기가 0 이면 잔상으로 보지 않는다 = 10 mm 를 붙이는 쪽 (fail-closed).

### 잔상 규칙의 근거 (V7g · V8, 35 grasp run, `raw/survey_*.json`)

held 행 중 `d − r < 10 mm` 인 행의 최근접 표면점 → attach 부피 거리 (planned + servo 자세):

| 표면 (관측 시점 장면의 GT) | 행 수 | attach 부피까지 |
|---|---:|---|
| lift 의 사과 자신의 상 (O7) | 6 | −9.2 … **7.3 mm** |
| hurdle | 589 | ≥ **199.1 mm** |
| crate | 363 | ≥ 307.7 mm |
| 기타 (apple 근처 · robot) | 608 | ≥ 216.0 mm |

**쓰지 않은 대안: "지금 (q_now) 쥔 구 근처의 표면"**. planned hurdle 행 254 중 22–24 행이 q_now 의 쥔 구에서 3–5 mm 안이다 — 막대가 사과에 가장 가까운 순간에 margin 을 빼게 된다. self-filter fit 구 기준도 같다 (5 mm 에서 18/250).
figure: `outputs/impl/T43HM/fig/t43hm-evidence.png` (왼쪽 = 이 표, 오른쪽 = (a)).

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/trajopt/config.py:597-610` · `:654-657` | `collision.held_obstacle_margin: bool = False` · `obstacle_margin > 0` 이 아니면 `ValueError` | flag |
| `benchmark/trajopt/linearize.py:76-79` | `_HELD_SELF_TOL_VOXELS = 2.0` | 잔상 허용오차 |
| `…/linearize.py:200-208` | `SceneSnapshot.held_obstacle_margin` · `held_capture_spheres` (K,4) | 씬 |
| `…/linearize.py:803-805` | `_esdf_clearance` — 켜졌으면 `_last_held_rows` 기록 | metrics |
| `…/linearize.py:869-945` | `_obstacle_rows` 가 held 열을 `_held_rows_class` 로; `_held_rows_record` · `held_rows_record()` | 분류 |
| `…/linearize.py:1122-1124` | `worst_row` 신원에 `held_obstacle_margin: True` (켜졌을 때만) | 기록 |
| `…/linearize.py:1614-1631` · `:1720-1725` | `scene_from_constraint_set(..., held_capture=None)` — 켜졌을 때만 flag, 쥐고 있을 때만 capture | 배선 |
| `benchmark/trajopt/sqp.py:641-646` · `:741` | `to.metrics["held_rows"] = {obstacle/support/held: {n, min_clearance_m}}` (켜졌을 때만) | 기록 |
| `benchmark/trajopt/safe_policy.py:328-329` · `:803` · `:817-823` | `_held_capture_for_scene()` → scene; meta `held_obstacle_margin` | 배선 |
| `benchmark/trajopt/serve_safe.py:1124-1129` · `:1770-1775` · `:2085-2089` | `--held-obstacle-margin` (store_true, `--obstacle-margin` 없으면 SystemExit) · 시작 로그 한 줄 | CLI |
| `benchmark/ag3s/runtime/pipeline.py:1225-1236` | `AG3S.held_capture_spheres` (읽기 전용, `_held_first_free`) | capture 부피 |
| `tests/trajopt/test_t43hm_held_obstacle_margin.py` (새, 24) · `tests/ag3s/test_config_roundtrip.py` (+3 줄) | | |

## 단위 검증

```bash
cd /mnt/dev/work && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa \
  PYTHONPATH=/mnt/dev/work .venv-openpi-live/bin/python -m pytest -q -p no:cacheprovider tests/trajopt tests/ag3s
```
- 전체: **2437 passed, 2 skipped, 1 failed (566 s)**. 실패 1 개는 새 파일의 기록 재생 1 건이다. survey 를 다시 쓰는 동안 json 을 읽은 경합이었고, 단독 재실행에서 24/24 pass 다.
- 새 파일 24 개가 고정하는 것:
  - off 비트 동일
  - 분류 · margin (bar / table / 잔상 / crate)
  - 로봇 행 불변
  - support 는 support 스위치와 무관하게 늘 빠진다
  - 잔상 허용오차 9 mm → held, 11 mm → obstacle
  - 최근접 표면점 기준
  - 목적지 20 vs 4 mm 일 때 큰 값
  - fail-closed (capture 없음 · 기울기 0 · 평면 없음)
  - `worst_row` key · `held_rows_record` · `to.metrics["held_rows"]` (실제 `TrajectoryOptimizer.solve`)
  - config 왕복 · CLI · `scene_from_constraint_set` · `AG3S.held_capture_spheres` · `SafePolicy._held_capture_for_scene`
  - V8 기록 4 청크 (막대 2 → obstacle, lift 잔상 2 → held, 값 ±0.05 mm)
- 재구성 충실도: 기록의 worst row 가 held 행인 97 청크에서 off 재계산과 기록 clearance 의 차이는 margin {0, 20} 을 고르면 **max 0.0013 mm** 다.

## 오프라인 (V7g · V8 기록, 생산 `_esdf_clearance`, `p/hm_survey.py` → `p/hm_eval.py` → `p/hm_agg.py`, `raw/hm_summary.json`)

자세는 두 가지로 쟀다.
- planned = 그 청크 refined 행 k
- servo = HX 의 `pred_q14` (`benchmark/trajopt/servo.py`)

label 층은 기록에 없어서 두 bracket 으로 쟀다.
- **B0**: 아무 행도 목적지 label 이 아니다 (상한).
- **B1**: GT crate 가 최근접인 held 행은 모두 목적지 label 이다 (하한, on/off 모두 20 mm).

### (a) 운반 중 사과–막대 접촉 (CT, V7g 3 · V8 5): 창 = first_t−2 … last_t, held 행 최소 clearance (mm)

| run | 접촉 t | planned off → **on** | servo off → **on** | on 의 class / GT |
|---|---|---|---|---|
| V7g 1925 s19253 | 174–175 | +4.6 → **−5.4** | −9.3 → −19.3 | obstacle / hurdle |
| V7g 1968 s19681 | 206–208 | +4.7 → **−5.3** | −14.3 → −24.3 | obstacle / hurdle |
| V7g 1982 s19822 | 168–169 | +5.0 → **−5.0** | −16.1 → −26.1 | obstacle / hurdle |
| V8 1807 s18071 | 206–208 | +4.9 → **−5.1** | −18.0 → −28.0 | obstacle / hurdle |
| V8 1807 s18073 | 174 | +8.2 → **−1.8** | −14.7 → −24.7 | obstacle / hurdle |
| V8 1967 s19672 | 199 | +4.7 → **−5.3** | −6.3 → −16.3 | obstacle / hurdle |
| V8 1968 s19683 | 164–165 | +3.8 → **−6.2** | −12.5 → −22.5 | obstacle / hurdle |
| V8 1982 s19823 | 175–177 | +4.7 → **−5.3** | −16.3 → −26.3 | obstacle / hurdle |

- planned 자세에서 TO 가 보는 것: 0/8 → **8/8**.
- servo 자세는 off 에서도 이미 8/8 이다 (HX 와 같다). on 은 10 mm 더 깊어진다.
- 8 건 모두 class `obstacle` 이다. 잔상 · 지지면으로 빠진 행은 없다.

### (b) attach · 들어 올림 (attach 뒤 첫 3 청크, 105 청크)

| | B0 | B1 |
|---|---:|---:|
| 새 위반 청크 (on < 0 ≤ off) | **0** | **0** |
| 이미 위반 (off < 0) | 0 | 1 (B1 의 label 가정 탓) |
| 잔상 규칙이 살린 행 (`held`, off < 10 mm) | 6 | — |
| 잔상 규칙이 없었다면 새 위반 청크 | 4 | — |

- 테이블: `d − r < 10 mm` 인 held 행이 0 이다 (J1 lift 가 지킨다). support 판정은 늘 빠진다.
- 이웃 과일: GT fruit 가 최근접인 held 행은 0 이다.

### (c) crate 진입 (사과 중심이 crate geom 에서 80 mm 안, 78 청크) · (참고) 나머지 운반 (248 청크)

| 구간 | 새 위반 B0 | GT 원인 (B0) | 새 위반 B1 | GT 원인 (B1) | 이미 위반 B0 |
|---|---:|---|---:|---|---:|
| crate | **39** | crate 18 · crate~apple 5 · apple 2 · apple~crate 2 · hurdle 4 · **정체 불명 8** | **11** | hurdle 4 · 정체 불명 6 · apple~crate 1 | 1 |
| carry | 49 | hurdle 27 · hurdle~apple 6 · crate 7 · apple 3 · apple~hurdle 1 · **정체 불명 5** | 42 | hurdle 27 · hurdle~apple 6 · apple 3 · 정체 불명 5 · apple~hurdle 1 | 12 |

- GT 원인은 binding held 행의 최근접 표면점에 가장 가까운 MuJoCo geom 무리다 (관측 시점 장면). 라벨 뜻:
  - `a~b` — 두 무리가 5 mm 안에서 겹친다.
  - 정체 불명 — 20 mm 안에 어떤 geom 도 없다 (로봇 geom 포함).
- 새 위반의 깊이: on −0.4 … −9.5 mm, 대부분 −5 mm 근처다 (off 가 +5 mm 근처, 즉 이미 binding 이던 held 행).
- 새 위반이 곧 HOLD 는 아니다. TO 가 그만큼 계획을 옮겨야 한다는 뜻이다. held 질의가 FIA 재풀이에 없어서 실제 HOLD 수는 오프라인으로 재현할 수 없다 → V 라운드에서 잰다.
- 기록 HOLD (on/off 무관, 참고): lift 1 · carry 15 · crate 2.

## verifier 가 알아야 할 것
- **새 flag**: `--held-obstacle-margin`.
  - `--obstacle-margin M (> 0)` 과 함께 준다. config 는 `collision.held_obstacle_margin`.
  - 시작 로그 `TO held_obstacle_margin: ON …` 한 줄이 생긴다.
  - meta `held_obstacle_margin: true` 가 생긴다.
- **새 기록 키** (켰을 때만):
  - `to.metrics.max_violation_pair.held_obstacle_margin: true` — 이때 held 행의 `obstacle_margin_class` 는 `obstacle` · `support` · `held`(= 자기 잔상) 중 하나다.
  - `to.metrics.held_rows = {obstacle|support|held: {n, min_clearance_m}}` — 반환 궤적의 waypoint held 행 기준이다.
- servo / sweep 경로 행 (`PathChecker`) 도 같은 `_esdf_clearance` 를 쓰므로 같은 분류 · margin 을 받는다.
- 재생산할 산출물: 없음. 옛 기록과의 호환: 깨지지 않음 (off 비트 동일, AG3S 쪽은 읽기 전용 property 하나).
- 오프라인 한계:
  - runtime 의 capture 부피는 fit 구 + **잡기 전 자리 구 (+ guard pad)** 다. 기록에는 fit 구만 있어서 오프라인은 fit 구만 썼다.
  - runtime 의 support_planes 는 그 프레임의 `support_surfaces` 다. 오프라인은 attach 기록의 테이블 평면을 썼다.

## 열린 위험 (lead 판단 필요)
1. **장애물이 아닌 표면이 margin 을 받는다.**
   - 운반 · crate 진입 중 held 행 옆 field 표면 중 일부는 attach 부피 밖이다.
     - 사과 자신의 현재 상: apple 2–3 청크
     - 정체 불명 잔상: crate 8 · carry 5 청크 (B0)
   - 이 표면에 10 mm 가 붙는다 (위반 −0.4 … −8.8 mm).
   - 한 프레임 기하로는 막대와 가를 수 없었다 (q_now 쥔 구 · filter 구 · attach 부피 모두 시험).
   - V 라운드에서 새 HOLD 의 binding 행이 이 무리인지 봐야 한다.
2. **기존 버그 (HM 과 무관, 고치지 않음)**: `scene_from_constraint_set` 이 `--target-volume-exempt` 와 `--obstacle-margin` 을 함께 켜면 `target_volume` 을 버린다.
   - 위치: `linearize.py:1704-1719`. `obstacle = dict(...)` 가 앞에서 채운 `target_volume` 을 덮는다.
   - V8 은 target-volume-exempt 를 쓰지 않아 영향이 없었다.
3. 기록 재생 테스트 4 건은 `outputs/impl/T43HM/raw/survey_V8.json` 에 기댄다 (없으면 skip).

## 내가 기대하는 결과
(verifier 는 측정이 끝나기 전에 읽지 않는다.) 운반 중 사과–막대 접촉은 줄 것이다. 대신 crate 진입과 허들 통과 직후에 TO 가 사과를 더 띄우느라 궤적이 바뀌고, 열린 위험 1 의 무리에서 HOLD 가 몇 청크 생길 수 있다.
