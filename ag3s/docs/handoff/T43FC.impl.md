# T43 FC — 구현: 손가락마다 따로, 실제 굵기의 몸통 구 (`--finger-cover`, 기본 꺼짐)

> writer: ag3s-implementer (A1) · 2026-10-06 · 읽는 쪽: verifier, scribe, lead
> task: [T43.task.md](T43.task.md) "사용자 판정 (2026-10-06 10:00) — 손가락 구 = B" · 기준 benchmark `f2f335b`
> 원자료 · 스크립트: `/mnt/dev/work/outputs/impl/T43FC/` (CPU only, `.venv-openpi-live`)

## 무엇을 했나 (평이한 요약 먼저)

손가락 link 4 개 (`ee_finger_l1/l2`, `ee_finger_r1/r2`) 의 구를 **손가락마다 따로** 다시 만든다. 새 구는 그
손가락 collision mesh 를 전부 담는다. 파지 면도 담는다. mesh 밖으로는 최대 2.5 mm 만 나간다.
- 두 손가락을 한 덩어리로 감싸지 않는다. 열린 손에서 두 손가락의 구 합집합 사이는 비어 있다. 최대 개도는 94.3 mm 다.
- 손바닥 (`ee_left/right`) 과 다른 link 는 그대로다.

기존 수단으로는 되지 않았다 (§1 표).
- `--capsule-radius-scale-link`: capsule 축이 손가락 판의 폭 방향 (y) 이다. 그래서 판 길이를 덮으려면 반지름이 10 mm 이상이어야 한다. 배율 1.0 (URDF 상당) 은 mesh 밖으로 19.7 mm 넘치고, 손끝 아래로 14 mm 내려간다.
- T43 R `--gripper-cover`: 내접 덮개라 넘치지 않는다. 대신 파지 면을 일부러 덮지 않고, 반지름 상한이 3.25 mm 다. 손가락 표본의 3.4 % 만 구 안이다.

그래서 새 flag 를 넣었다. `--finger-cover [both|left|right]` · `--finger-cover-overflow MM` 이고, 기본은 꺼짐이다. 꺼지면 모델이 비트 동일하다.

오프라인 재생 결과 (W · Z3V · V7g + E0, 150 run).
- (a) 막대 접촉 순간: 실제 기하로는 새 구가 접촉 57/57 을 침투로 본다 (축선 사슬은 47/57). 기록 거리장으로는 새 구 34/44 · 축선 사슬 36/44 다. 둘 다 못 본 8 건은 거리장에 막대가 없던 순간이다.
- (b) closing · attach: margin 0 위반 청크는 2 개 늘었다 (5 → 7). 그 청크는 모두 서버가 이미 HOLD 한 청크다. 10 mm 띠에서는 새로 걸리는 청크가 12/1254 개다. 테이블 행은 band 를 포함해도 실제로는 양수다.
- (c) 구는 122 → 270 개다. TO 벽시계는 CPU 자가 측정으로 +17–29 ms (×1.4–1.7) 다.

## 1. 기준 = 손가락 collision mesh — 수단 비교

손가락 collision mesh 의 크기는 다음과 같다 (URDF `ee_finger_l1` frame, 네 손가락 같음).
- 볼록 조각 5 개다.
- 판 본체: x −3.3…6.2 mm (두께 9.5 mm, x = 닫히는 방향의 반대), y ±9.6…±16.3 mm (손끝 쪽이 좁다), z −60.8…−2 mm.
- 뿌리 블록: z > −10 mm, x 는 13.3 mm 까지.

덮임은 겉면 표본 전부 (0.5 mm, 파지 면 포함, 맞붙은 조각 사이 면 제외 = 20,774 점) 로 잰다. 넘침은 구 표면 표본 중 mesh 밖 점에서 가장 가까운 mesh 표본까지 거리의 최대다.
원자료: `alt_mechanisms.{py,json,log}`.

| 수단 | 손가락 구 / 전체 | 반지름 mm | 구 안 | ≤ 2.5 mm | 최대 틈 mm | **넘침 mm** | 안쪽 면 x mm (mesh −3.3) | 손끝 z mm (mesh −60.8) | 최대 개도 mm |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| E3b (지금, 배율 0.05) | 19 / 122 | 2.6–3.3 | 0.032 | 0.220 | 13.58 | 5.05 | −1.8 | −60.5 | 102.3 |
| `--capsule-radius-scale-link ee_finger_*=0.5` | 19 / 122 | 6.9–9.3 | 0.465 | 0.755 | 8.40 | 11.13 | −7.9 | −66.6 | 90.2 |
| 같은 flag `=1.0` (URDF 상당) | 19 / 122 | 13.2–17.9 | 0.983 | 1.000 | 1.02 | **19.67** | −16.5 | −75.2 | 73.1 |
| T43 R `--gripper-cover` (손가락 부분) | 103 / 1594 | 0.8–3.3 | 0.034 | 0.736 | 5.73 | 5.05 | −1.8 | −60.5 | 102.3 |
| **FC `--finger-cover` (2.5 mm, 기본)** | **56 / 270** | **2.5–8.4** | **1.000** | 1.000 | **0.00** | **2.55** | −5.8 | −63.3 | **94.3** |
| FC `--finger-cover-overflow 2` | 91 / 410 | 2.0–7.9 | 1.000 | 1.000 | 0.00 | 2.11 | −5.3 | −62.8 | 95.3 |

- E3b 와 T43 R 의 넘침 5 mm 는 축선 사슬 끝 구가 y 방향 (손가락 폭) 으로 mesh 밖에 나와 있는 것이다. FC 는 그 사슬을 **대신하므로** 이 넘침도 사라진다.
- T43 R 의 덮개는 이 사슬 위에 붙는 것이다.

## 2. 고른 구 — 규칙과 손가락별 결과

규칙 (`gripper_cover.finger_body_spheres`) 은 세 단계다.
1. 후보 중심: mesh 안 0.5 mm 격자점 `c` 다. 반지름은 `depth(c) + overflow` 이다.
   - `depth` = max(한 볼록 조각 안의 정확한 깊이 `hull_depth`, 겉면 표본까지 거리 − 0.5 mm).
   - `ball(c, depth) ⊂ mesh` 이므로 이 구는 mesh 에서 overflow 보다 멀리 나가지 않는다.
   - 뒤 항은 조각이 맞붙은 뿌리 블록에서 구를 크게 하려는 것이다. 표본 간격 0.5 mm 만큼의 근사다.
2. 대상: 겉면 표본 전부다. 표본에서 바깥 법선으로 0.3 mm 나간 점이 mesh 안이면 맞붙은 면으로 보고 뺀다.
3. 고르기: greedy set cover 로 대상 전부를 구 **안에** (틈 0) 담는다. 다른 구로 이미 다 담기는 구는 뒤에서부터 지운다.

결과 (overflow 2.5 mm, `build_check.json`):

| | 값 |
|---|---|
| 손가락 하나 | **56 구, r 2.54–8.44 mm** (판 가운데 구 7.2 mm = 반두께 4.75 + 2.5, 뿌리 블록은 8.4 mm, 모서리는 작다) |
| 네 손가락 | link frame 에서 같은 구다 (캐시 1 회 풀이). 손가락 구 76 → 224, 전체 122 → 270 |
| 덮지 못한 표본 | **0 / 20,774** (최대 틈 0.000 mm). collision mesh 정점 971 개도 전부 구 안 |
| 넘침 | 상한 2.5 mm (구성상), 실측 2.55 mm (표본 오차 포함) |
| 구 상자 | x −5.84…15.67 · y ±18.84 · z −63.34…4.33 mm (mesh 상자 + 2.5 mm) |
| 손가락 사이 | 열린 손 (한쪽 50 · 35 · 16.7 mm) 에서 두 합집합 사이 > 0, 가운데 점은 어느 구에도 안 든다 (test) |
| 최대 개도 (T26) | 102.35 → **94.32 mm** = mesh 안쪽 간격 99.4 − 2 × 2.5 (사과 66.6 mm) |
| 생성 시간 | 모델 짓기 +8 s (CPU, 프로세스당 1 회) |
| overflow 별 손가락 구 | 1.0 mm 253 · 2.0 mm 91 · **2.5 mm 56** · 3.0 mm 48 |

2.5 mm 를 고른 이유는 세 가지다.
- T43 R 덮개 틈과 같은 크기다 (사용자 "최대 틈 약 2–3 mm").
- 2.0 mm 는 구가 1.6 배이고 TO 가 +16–18 ms 더 든다 (§5).
- 3.0 mm 는 구가 8 개만 준다.

## 3. 오프라인 (a) — CT 손가락–막대 접촉 순간에 새 구가 막대를 보나

**이름 주의:** CT 의 `ee_finger_l2` (geom 1086) 는 **MJCF** body 이름이다. URDF 로는 `ee_finger_l1` 이다 (T11 `MJCF_BODY_ALIASES`). 그러니 "release 뒤 l2 28 건" 은 URDF `ee_finger_l1` 이다.

방법 (`fc_offline.py` (a)) 은 다음과 같다. CT 의 손가락 접촉 event 마다 `first_t…last_t` 의 각 행에서 pre 와 post 상태를 잰다.
- pre = 그 control 행 `qpos` 다.
- post = CT replay 의 post 상태다. 손 site 자세 `tcp/tcp_R` 와 개도를 쓰고, 행 동안의 손 강체 운동 `M` 으로 옮긴다.
  - 접촉은 대부분 run 의 마지막 행이라 다음 행 `qpos` 가 없다.
  - post 를 재현할 수 있는 행에서 재현 오차는 0 mm 다.
- 실제 기하: 그 link 구에서 막대 collision geom (bar capsule, leg cylinder 2) 까지의 거리 − r 이다. 막대 pose 는 client reset (mocap) 에서 읽는다.
  - mesh 는 `mj_geomDistance(손가락 collision geom, 막대)` 다.
- 거리장: t 를 덮는 청크 (planning `t_step ≤ t` 의 마지막) 의 기록 격자를 쓰고, TO 합성 (`FIA.contact_policy`) 대로 d − r 을 잰다. "본다" = d − r < 0 이다.
- event 값 = 행 · 상태의 최솟값이다.

| 단계 (CT) | event | 실제 mesh 침투 (< 0) | 실제 기하: 축선 사슬 < 0 | 실제 기하: **FC < 0** | FC ≤ mesh | 거리장 있는 event | 거리장: 축선 사슬 본다 | 거리장: **FC 본다** | 중앙값 d − r (사슬 → FC) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 쥐기 전 (MJCF l2) | 5 | 3 | 2 | **5** | 5 | 3 | 3 | 3 | −2.3 → −3.6 mm |
| 운반 (MJCF l2 15 · r2 1; E0 11, V1 5) | 16 | 13 | 9 | **16** | 16 | 5 (V1) | 5 | 5 | −2.6 → −4.9 mm |
| release 뒤 · MJCF l2 (= URDF l1) | 28 | 23 | 28 | 28 | 28 | 28 | 20 | 18 | −2.8 → −2.9 mm |
| release 뒤 · MJCF l1 (= URDF l2) | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | −3.3 → −5.9 mm |
| **합** | **57** | 47 | 47 | **57** | 57 | 44 | **36** | **34** | |

- **실제 기하로는 FC 가 57/57 을 본다** (모든 event 에서 FC ≤ mesh, 합집합이 mesh 를 담으므로). 축선 사슬이 놓친 10 건이 있다.
  - E0 운반 4 · 쥐기 전 1: mesh −0.7…−1.6 mm 인데 사슬은 +0.1…+1.3 mm.
  - V1 운반 3 · V5B/V5S 쥐기 전 2.
- **거리장에서는 FC 34 / 사슬 36** 이다. 차이 2 건은 V5 · V5D ep1982 s19823 t300 (같은 순간) 이다.
  - mesh +0.03 mm (실제로는 닿지 않음) 인데 사슬은 −2.0 mm, FC 는 +1.7 mm 다.
  - 사슬 끝 구가 손가락 폭 밖으로 5 mm 나와 있던 것이 "보았던" 것이다.
- 둘 다 못 본 8 건은 거리장의 d − r 이 +4…+56 mm 다. 그 청크의 거리장에 그 자리의 막대가 없다 (자기 필터 · 시야 · 청크 시점). 구 모델로 고칠 수 없다.
- E0 는 서버 기록이 없어 거리장 열이 비었다 (E0 = 정책만).

## 4. 오프라인 (b) — 정상 파지 (closing · attach) 에서 새 HOLD 를 만들까

방법 (`fc_offline.py` (b)) 은 다음과 같다.
- 대상: grasp `state_before` 나 `state` 가 `closing` 인 서버 청크, 또는 `attach` 청크다. 1254 청크 (closing 1141, attach 113; arm 사이에 같은 순간이 많아 고유 1050) 이고, 서버 기록이 있는 137 run 이다.
- 행: 활성 손 손가락 구를 TO 계획 창 (refined 행 0–7) 에 둔다.
  - 개도는 서버와 같은 envelope 를 쓴다 (`SafePolicy._set_finger_joints`: 스텝 0 측정, 이후 max(측정, 명령 누적 최대)).
- 거리장 합성은 TO 와 같다.
  - closing: 권한 손가락이 **target-free 층** 을 본다 (1141/1141).
  - attach: 쥔 것이 있어 전체 층이다 (113/113).
- 충실도: 서버 기록의 TO 최악 행 (`to.metrics.max_violation_pair` 의 step · query · clearance) 을 축선 사슬로 다시 지으면 차이가 **0 · 10 · 20 mm 중 하나이고, 그 값에서 벗어난 양은 0.000 mm** 다 (1251 청크, 3 청크는 최악 행이 로봇 ESDF 행이 아니라 잴 수 없음). 10 · 20 mm 는 그 행의 label margin (장애물 10, destination 20) 이다.
  - **label 층은 기록에 없다.** 그래서 margin 0 (확실히 위반) 과 10 · 20 mm 띠 (label 에 따라 위반) 를 따로 센다.
  - 기록 npz 의 `clearance` 배열은 전체 층 진단값이고 TO 의 행이 아니다. 쓰지 않았다.
- 위반 행 원인: 그 행의 거리장 값과 가장 맞는 실제 물체 (|d_obj − d_field| ≤ 10 mm) 로 고른다. 후보는 테이블 (정적 geom) · crate 13 box · 과일 hull · 막대다. target-free 로 답한 행은 사과를 후보에서 뺀다.
  - 거리장은 테이블 윗면을 실제보다 3–6 mm 높게 본다 (support band, T43R2).

| | 청크 | margin 0 위반 청크: 사슬 → FC | **FC 에서만** | 10 mm 띠 청크: 사슬 → FC | FC 에서만 | 20 mm 띠: 사슬 → FC | FC 에서만 | 계획 창 최소 d − r 중앙값 |
|---|---:|---|---:|---|---:|---|---:|---|
| closing | 1141 | 2 → 5 | 3 | 92 → 87 | 10 | 354 → 350 | 4 | 61.5 → 61.3 mm |
| attach | 113 | 3 → 2 | 0 | 10 → 10 | 2 | 48 → 53 | 8 | 21.2 → 21.1 mm |
| 합 | 1254 | 5 → 7 | **3** | 102 → 97 | **12** | 402 → 403 | 12 | |

- **margin 0 에서 FC 에서만 위반하는 3 청크 (고유 2) 는 서버가 이미 HOLD 한 청크다.**
  - V5B ep1995 s19953 t208: 막대, −1.16 mm.
  - V1 · V4 ep1834 s18341 t112: 같은 순간이다. 오른손 closing, −4.35 mm, 원인 미확인 22 행이다.
  - 그러니 서버가 실행했던 청크에서 새로 HOLD 가 되는 것은 0 이다.
- 반대 방향도 1 건 있다. V5B ep1982 s19823 t160 attach 는 사슬 −0.67 mm (사과) 였고 FC 는 +0.73 mm 다.
- 테이블: margin 0 행 62 개가 테이블로 설명된다. 전부 V3 B ep1995 s19952 t152 · 160 (서버 HOLD) 이고, 사슬도 같은 청크에서 8 · 21 행이 위반했다.
- **10 mm 띠에서 FC 에서만 걸리는 12 청크 (고유 9) 는 서버가 실행한 청크다.** FC 최소 d − r 은 6.0–9.9 mm 다.
  - 원인: 테이블 (band, 4), 막대 (V5B ep1995 t224 36 행, V3 1), crate (2), attach 의 사과 (2), 미확인 (3).
  - 그 행이 "장애물 10 mm" label 이면 HOLD 가 된다. label 을 몰라 확정하지 못한다. 이것이 새 HOLD 위험의 상한이다.
- 전체로 보면 10 mm 띠 청크 수는 줄었다 (102 → 97). 축선 사슬이 y 방향으로 넘치던 5 mm 가 없어졌기 때문이다.
- 테이블 쪽 손끝: FC 손끝은 mesh 보다 2.5 mm 아래다 (사슬은 mesh 보다 0.3 mm 위). closing 에서 테이블로 설명되는 FC 행의 최소 d − r 은 V3 의 위 청크 밖에서 +1.2 mm 다 (V5 · V5D · V7g ep1982 s19823 t120). margin 0 위반은 V3 의 위 청크뿐이다.
- 사과: closing 중에는 target-free 층이라 사과가 거리장에 없다. attach (전체 층) 에서 사과로 설명되는 margin 0 행은 14 개이고, 모두 서버 HOLD 청크다.

## 5. 오프라인 (c) — 비용

| 항목 | 축선 사슬 (E3b) | FC 2.5 mm | FC 2.0 mm |
|---|---:|---:|---:|
| 제약 구 | 122 | 270 (+148) | 410 |
| ESDF 질의 · FK 행 (구 × 스텝 × 계층) | 1 × | 2.2 × | 3.4 × |
| QP 행 (`rows_per_step` 24) | 그대로 | 그대로 | 그대로 |
| TO 벽시계 중앙값, C1807 / B1968 (ms) | 43.7 / 39.7 | **72.7 / 56.7** | 91.0 / 72.6 |
| 그중 linearize · check (ms, C1807) | 8.2 · 8.7 | 11.0 · 15.8 | 13.9 · 22.5 |
| 서버 시작 | — | +8 s | +8 s |

- 측정: `to_time.py` 는 T43 R 과 같은 24 청크 (T40 G E3b C1807 119–130, B1968 140–151) 다. 반복 수는 기록과 같고, 시계를 끄고 3 회 중 최솟값을 썼다. CPU · nice 19 · 공유 pod 에서 잰 **자가 측정** 이다.
  - 결과: `to_time.{json,log}`.
  - 기록 서버 시간과 절대값은 다르다. 같은 환경 안의 비교로만 읽는다.
- C1807 의 qp 가 12 → 28 ms 다. 막대 근처에서 활성 행이 늘었기 때문으로 본다 (재지 않음).
- T43 R `--gripper-cover` 는 1594 구, 224–264 ms 였다. FC 의 비용은 그 1/4 정도다.
- `time_budget_ms` 에 걸리면 SQP 반복이 줄 수 있다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/robot_models/urdf_sphere_chain.py:214-223` | `FINGER_BODY_ROLE` · `EXACT_SPHERE_ROLES` | 몸통 구는 반지름 그대로 (배율 0.05 · 간격 inflation 없음) |
| `…/urdf_sphere_chain.py:887` · `:1001` | `== EXACT_SPHERE_ROLE` → `in EXACT_SPHERE_ROLES` | 꺼지면 이 role 의 capsule 이 없어 동작이 같다 |
| `benchmark/ag3s/experiments/sources/gripper_cover.py:581-767` (추가) | `FINGER_COVER_*` 상수 · `finger_body_targets` · `_greedy_exact_cover` · `_sampled_overflow` · `finger_body_spheres` · `finger_body_capsules` (캐시) | §2 |
| `benchmark/ag3s/experiments/reports/grounding_report.py:138-145` | `FINGER_COVER_HANDS` | `--finger-cover left/right` |
| `…/grounding_report.py:147` · `:176-182` · `:193-227` · `:245-246` | `build_constraint_robot_model(…, finger_cover=None)`. 켜면 손가락 gap-filling capsule 자리에 몸통 구를 넣고 (손바닥 순서 그대로) `finger_cover_report` 를 남긴다. `gripper_cover` 와 같이 주면 덮개는 바뀐 모델 위에서 만든다 | |
| `benchmark/trajopt/serve_safe.py:542-574` | `announce_finger_cover` (켰을 때만 WARNING) · `finger_cover_options` | |
| `…/serve_safe.py:756` · `:813-815` · `:893` | `build_ag3s(finger_cover=None)`. None 이면 인자를 안 넘긴다. `--exclude-links` 재빌드에도 같은 `cover` 가 간다 | |
| `…/serve_safe.py:1246-1254` | `--finger-cover [both\|left\|right]` · `--finger-cover-overflow MM` | CLI |
| `…/serve_safe.py:1548-1551` | `--no-safe` · `--no-perception` 과 함께면 거절 | T43 R 과 같은 규율 |
| `…/serve_safe.py:1980-1982` · `:2061` | client metadata `finger_cover` (켰을 때만) · `build_ag3s` 호출 | |
| `tests/trajopt/test_t43fc_finger_cover.py` (새, 27) | 아래 | |

**AG3SConfig 왕복 테스트는 늘리지 않았다.** 이 flag 는 제약 로봇 모델 (`build_constraint_robot_model`) 의 옵션이고 AG3SConfig 필드가 아니다.
- T43 R `--gripper-cover` 와 같은 배선이다.
- `assert_ag3s_config` · `with_overrides` 경로를 지나지 않는다. 그래서 그 버그 (T2 의 `collision_backend` 유실) 가 생길 자리가 없다.
- 대신 CLI → `build_ag3s` → `build_constraint_robot_model(finger_cover=…)` 배선을 test 로 고정했다. 끄면 키가 없다.

## 단위 검증

```bash
cd /mnt/dev/work
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=2 \
PYTHONPATH=/mnt/dev/work nice -n 19 .venv-openpi-live/bin/python -m pytest -q -p no:cacheprovider tests/trajopt/test_t43fc_finger_cover.py
# 27 passed (≈ 50 s)
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=2 \
PYTHONPATH=/mnt/dev/work nice -n 19 .venv-openpi-live/bin/python -m pytest -q -p no:cacheprovider tests/trajopt tests/ag3s
# 2393 passed, 2 skipped, 0 failed (7 min 10 s) — outputs/impl/T43FC/pytest_all.log
```

기존 테스트는 한 줄도 고치지 않았다.

새 테스트 27 개가 고정하는 것:

1. 꺼짐
   - 구 이름 · 반지름이 E3b 기록과 float32 비트로 같다.
   - `None/False/{}` 가 같은 모델이다 (FK 중심 `array_equal`).
   - CLI 기본값은 `None` 이다. overflow 만 주거나 0 · 음수 · nan 을 주면 `SystemExit` 다. 꺼져 있으면 announce 가 조용하다.
   - `build_ag3s` 는 끄면 키를 넘기지 않는다. `--no-safe` · `--no-perception` 과 함께면 거절한다.
2. 덮임 · 넘침 (네 손가락 각각)
   - 겉면 표본 > 15,000 개와 mesh 정점이 전부 구 안 (gap ≤ 1e-9) 이다.
   - 넘침 실측 ≤ 2.5 + 0.3 mm 이고, 구 중심은 mesh 안이다.
   - 구 상자 ⊂ mesh 상자 + 2.5 mm 다.
   - 축선 사슬은 표본 5 % 미만이다.
3. 손가락은 따로
   - 열림 50 · 35 · 16.7 mm 에서 두 합집합이 떨어져 있고, 가운데 점은 비어 있다.
   - 손바닥 구가 같다. 손가락은 전부 `finger_body` role 이고, 손가락 coverage shortfall 은 없다.
   - 네 손가락이 같은 구다. 배율 1.0 을 줘도 몸통 구는 같다.
   - 최대 개도는 mesh 간격 − 2 overflow 이고, 사과보다 20 mm 넘게 크다.
   - `hands` · `overflow` 옵션이 동작하고, 모르는 키 · 손은 거절한다.
   - `--gripper-cover` 와 함께 주면 몸통 구는 그대로이고 손바닥 덮개가 붙는다.
4. 기록 재생
   - V5 ep1925 s19251 t270 (CT 접촉, MJCF l2 = URDF l1) 의 post 상태를 쓴다.
   - 기록 거리장의 층은 기록 행과 맞는 쪽을 고른다 (오차 < 0.1 mm).
   - 그 거리장에서 FC 의 d − r < −3 mm (실측 −4.2) 이고, 사슬 (−3.3) 보다 깊다.

## verifier 가 알아야 할 것

- **새 flag**: `--finger-cover [both|left|right]` (값 없이 주면 both), `--finger-cover-overflow MM` (기본 2.5). 기본은 꺼짐이고 기존 기본값은 하나도 안 바꿨다.
  - 시험이라면 V7g flag 에 `--finger-cover` 만 더한다.
- 켜면 생기는 것은 넷이다.
  - 시작 로그: `constraint finger cover (T43 FC): 구 122 → 270 …` (WARNING).
  - client metadata `finger_cover`.
  - npz 의 `sphere_centres` · `sphere_radii` · `clearance` 열이 270 이 된다. **앞 46 열 (손바닥) 은 같고 손가락 열은 바뀐다.** T43 R 처럼 뒤에 붙인 것이 아니다.
  - T26 max opening 이 102.35 → 94.32 mm 다 (`gripper_max_opening`, admissibility).
- 함께 바뀌는 측정 · 판단 경로는 세 가지다. 모두 "손가락이 실제 굵기가 된다" 는 방향이다.
  - `hand_spheres` (T31b/T32b hand occlusion) 가 굵은 손가락을 본다.
  - `tip_alignment` (측정 전용) 은 손끝 구 중심이 바뀐다.
  - grasp latch 의 reach 는 link 원점 기준이라 그대로다.
- `fixed_input_ablation` 은 FC 실행을 다시 짓지 못한다. 구 수 270 ≠ 122 라 "sphere option 이 다릅니다" 로 멈춘다. 안전한 실패다. `outputs/impl/T43FC/to_time.py` 방식을 쓴다.
- 오프라인 도구 메모 두 가지.
  - 기록 npz 의 `clearance` 배열은 **전체 층 진단값** 이다. TO 행 (권한 손가락 = target-free) 이 아니다. 그래서 충실도는 `to.metrics.max_violation_pair` 로 잰다.
  - V5 · V5D · V7g (그리고 V1 · V4) 는 attach 전까지 같은 seed 에서 같은 궤적이라 closing 청크가 겹친다 (고유 1050/1254).
- 재생산이 필요한 산출물: 없음. 옛 기록과 호환: 꺼지면 키 · 값 모두 같다.
- **나는 재지 않았다**: closed-loop (GPU 라운드), GPU 서버 TO 시간, figure, label 층이 있는 margin 판정.

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->

- release 뒤 · 운반 중 손가락–막대 접촉은 TO 가 손가락 등과 옆을 막대에서 2–3 mm 더 떼려 할 것이므로 줄 것이다. 다만 거리장에 막대가 안 잡힌 순간 (§3 의 8 건) 은 그대로일 것이다.
- closing · attach 의 HOLD 는 거의 그대로일 것이다. 늘어난다면 10 mm label 띠의 테이블 · 막대 행 (§4 의 12 청크 상한) 에서 올 것이다.
- TO 시간이 +17–29 ms (CPU) 늘어난다. 서버 시간 예산 근처의 청크에서 반복이 줄 수 있다.
