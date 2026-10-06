# T43 FR — 구현: 손가락 몸통 구 수를 줄이는 선택지 표 (기본 `--finger-cover` 는 그대로)

> writer: ag3s-implementer (A1) · 2026-10-06 · 읽는 쪽: lead, verifier, scribe
> task: [T43.task.md](T43.task.md) "사용자 판정 (2026-10-06 11:10)" 의 **FR** · 기준 benchmark `786cf1b`
> 원자료 · 스크립트: `/mnt/dev/work/outputs/impl/T43FR/` (CPU only, `.venv-openpi-live`)

## 무엇을 했나 (평이한 요약 먼저)

FC 의 손가락 몸통 구는 손가락 하나에 56 개다. 모델 전체로는 122 → 270 구다. 이 구 수를 줄이는 방법을 여러 가지 만들고 같은 잣대로 쟀다.
- 잣대: 덮임 · 넘침 · 최대 개도 · CPU TO 시간 · 오프라인 (a) · (b).
- 표 (§1) 는 V8 의 GPU TO 시간이 나오면 사용자가 하나를 고르게 하려는 것이다.

결과는 셋이다.
1. **효과가 큰 것은 두 가지다.** 넘침 (overflow) 을 키우는 것과 정수계획 (`ilp`) 으로 고르는 것이다.
   - `ilp` 는 FC 와 같은 후보 · 같은 보장 (mesh 전부 덮음, 넘침 ≤ overflow) 에서 구 수를 최소로 한다.
   - greedy 대비 손가락당 구 수: 2.5 mm 56 → 46 · 3 mm 48 → 32 · 4 mm 28 → **19** · 5 mm 24 → 14 · 6 mm 18 → 11.
   - `ilp` overflow 4 mm 는 전체 122 구다. 축선 사슬 모델과 같은 수이고, CPU TO 도 사슬과 비슷하다 (FC 기본의 ×0.70).
2. 효과가 없는 것도 둘이다.
   - 반지름 하한: 손가락당 55–57 구로 그대로다.
   - 내포 구 제거: 0 개 지워진다. greedy 뒤의 중복 제거가 이미 더 강한 조건이다.
3. 덮임을 포기하는 것도 쟀다.
   - 개수 상한 K: 덮임 보장이 없다. K 8 이면 표본 56 % 가 최대 9 mm 밖에 남는다.
   - 덮임 허용 틈 (gap): 표본이 합집합에서 gap 안이면 덮인 것으로 친다.

선택지는 새 flag 셋이다. 기본은 전부 꺼짐이고, 안 주면 FC 와 비트로 같다.
- `--finger-cover-solver {greedy,ilp}`
- `--finger-cover-gap MM`
- `--finger-cover-max-spheres N`

`ilp` 는 5 분 넘게 걸릴 수 있다. 그래서 7 개 설정의 답 (구 중심의 격자 index) 을 `finger_cover_presets.py` 에 미리 풀어 두었다.
- 서버는 그 중심의 반지름을 다시 계산하고 덮임을 검사한 뒤 쓴다. 시작 시간은 FC 와 같다 (+8–13 s).

## 1. 선택지 표

손가락 하나 (URDF `ee_finger_l1`, 네 손가락 같은 구) 기준이다. 전체 = 손바닥 46 + 4 × 손가락.

열 읽는 법은 다음과 같다.
- "안 덮인 표본": FC 의 대상 표본 20,774 개 (겉면 0.5 mm, 파지 면 포함) 중 구 합집합 밖인 것.
- "최대 미덮음": 그 표본에서 합집합 표면까지의 최대 거리.
- "넘침": 구 표면 표본 중 mesh 밖 점에서 mesh 까지의 최대 거리 (실측).
- "최대 개도": T26 `gripper_openings` 이다.
- CPU TO: `fr_time.py` 로 쟀다. FC `to_time.py` 와 같은 24 청크 (T40 G E3b C1807 119–130 / B1968 140–151) 이고, 반복 수는 기록과 같고 시계는 껐다.
  - 순서: pass · 청크마다 19 모델을 **무작위 순서로 번갈아** 풀었다 (pass 0 은 버림, 5 pass, 청크마다 최솟값).
  - 2 회 (seed 1 · 2) 반복해 평균했다. 두 회의 중앙값 차이는 최대 2.9 % 다.
  - "×FC 기본" 은 같은 청크끼리 짝지은 비율의 중앙값이다.
- (a): CT 손가락–막대 접촉 57 건을 실제 기하로 본 것이다.
  - "본다" = 구 − 막대 거리 < 0.
  - "mesh 까지 닿음" = 구 거리 ≤ 실제 손가락 mesh 거리, 즉 합집합이 그 순간 실제 손가락만큼 나와 있다.
- (b): closing · attach 1254 청크다. margin 0 = verdict 가 허락하지 않는 계획 행 (FC 의 "non-allowed") 이고, 10 mm 띠 = label 이 장애물이면 HOLD 가 되는 행이다.
  - "새" = 축선 사슬에서는 안 걸리던 청크다. 괄호 · "미HOLD" 는 그중 서버가 이미 HOLD 하지 않은 청크 = **새 HOLD 의 상한** 이다.
  - 마지막 수는 FC 기본 대비 새로 걸리는 미HOLD 청크다.

| 선택지 | 구 / 손가락 | 전체 | 안 덮인 표본 | 최대 미덮음 mm | 넘침 mm | 최대 개도 mm | CPU TO 중앙 C/B ms | p90 C/B ms | ×FC 기본 C/B | (a) 본다 /57 | (a) mesh 까지 /57 | (b) margin 0: 청크 · 새 (미HOLD) | (b) 10 mm 띠: 청크 · 새 미HOLD · FC 대비 새 미HOLD |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|---|
| 축선 사슬 (finger-cover 끔) | 19 | 122 | (> 95 %) | 13.58 | 5.05 | 102.3 | 44.4 / 40.3 | 95.6 / 41.2 | 0.71 / 0.70 | 47 | 35 | 5 · — | 102 · — · 17 |
| **FC 기본 (greedy 2.5)** | **56** | **270** | **0** | 0.00 | 2.55 | 94.3 | **74.5 / 58.0** | 89.9 / 60.4 | **1.00 / 1.00** | 57 | 57 | 7 · 3 (0) | 97 · 12 · 0 |
| greedy overflow 3 | 48 | 238 | 0 | 0.00 | 3.02 | 93.3 | 65.0 / 54.2 | 88.3 / 56.2 | 0.93 / 0.93 | 57 | 57 | 7 · 3 (0) | 102 · 15 · 6 |
| greedy overflow 4 | 28 | 158 | 0 | 0.00 | 4.06 | 91.3 | 51.6 / 45.3 | 70.0 / 46.9 | 0.77 / 0.78 | 57 | 57 | 8 · 4 (0) | 118 · 30 · 21 |
| greedy overflow 5 | 24 | 142 | 0 | 0.00 | 5.01 | 89.3 | 51.1 / 43.4 | 73.8 / 44.1 | 0.76 / 0.75 | 57 | 57 | 19 · 14 (**7**) | 148 · 46 · 50 |
| greedy overflow 6 | 18 | 118 | 0 | 0.00 | 6.00 | 87.3 | 50.7 / 39.8 | 78.8 / 41.4 | 0.71 / 0.70 | 57 | 57 | 25 · 20 (**11**) | 204 · 101 · 106 |
| **ilp overflow 2.5** | 46 | 230 | 0 | 0.00 | 2.52 | 94.3 | 57.6 / 53.6 | 88.2 / 55.1 | 0.89 / 0.92 | 56 ¹ | 57 | 7 · 3 (0) | 94 · 10 · 1 |
| **ilp overflow 3** | 32 | 174 | 0 | 0.00 | 3.01 | 93.3 | 55.7 / 46.0 | 80.4 / 48.5 | 0.81 / 0.81 | 57 | 57 | 7 · 3 (0) | 100 · 15 · 7 |
| **ilp overflow 4** | **19** | **122** | 0 | 0.00 | 4.01 | 91.3 | **47.3 / 40.6** | 67.6 / 42.2 | **0.70 / 0.70** | 57 | 57 | 7 · 3 (0) | 135 · 38 · 38 |
| ilp overflow 5 | 14 | 102 | 0 | 0.00 | 4.98 | 89.3 | 45.2 / 38.0 | 67.5 / 39.2 | 0.67 / 0.66 | 57 | 57 | 13 · 8 (**4**) | 157 · 63 · 60 |
| ilp overflow 6 | 11 | 90 | 0 | 0.00 | 5.98 | 87.3 | 43.8 / 36.3 | 78.6 / 37.5 | 0.64 / 0.63 | 57 | 57 | 11 · 7 (**3**) | 135 · 48 · 38 |
| greedy gap 0.5 (2.5) | 48 | 238 | 737 | 0.50 | 2.53 | 94.3 | 70.5 / 53.8 | 93.7 / 56.2 | 0.94 / 0.94 | 57 | 51 | 7 · 3 (0) | 90 · 6 · 0 |
| greedy gap 1.0 (2.5) | 32 | 174 | 2,566 | 1.00 | 2.52 | 94.3 | 53.0 / 46.9 | 89.8 / 48.9 | 0.83 / 0.81 | 47 | 32 | 7 · 3 (0) | 76 · 5 · 0 |
| ilp gap 0.5 (2.5) | 32 | 174 | 769 | 0.50 | 2.51 | 94.3 | 54.5 / 46.2 | 78.9 / 48.3 | 0.83 / 0.81 | 57 | 51 | 7 · 3 (0) | 87 · 5 · 0 |
| ilp gap 1.0 (2.5) | 26 | 150 | 2,290 | 1.00 | 2.50 | 94.3 | 49.2 / 43.8 | 82.9 / 45.3 | 0.76 / 0.76 | 51 | 43 | 7 · 3 (0) | 77 · 5 · 0 |
| greedy K 24 (2.5) | 24 | 142 | 1,808 | 1.97 | 2.49 | 94.3 | 56.3 / 42.7 | 78.9 / 44.3 | 0.81 / 0.74 | 57 | 45 | 7 · 3 (0) | 94 · 11 · 0 |
| greedy K 16 (2.5) | 16 | 110 | 5,072 | 3.87 | 2.49 | 94.9 | 48.4 / 39.0 | 74.2 / 40.6 | 0.69 / 0.68 | 57 | 45 | 6 · 2 (0) | 93 · 10 · 0 |
| greedy K 12 (2.5) | 12 | 94 | 7,763 | 5.34 | 2.49 | 94.9 | 48.5 / 36.7 | 65.5 / 37.7 | 0.68 / 0.64 | 57 | 45 | 6 · 2 (0) | 93 · 10 · 0 |
| greedy K 8 (2.5) | 8 | 78 | 11,673 | 8.99 | 2.49 | 94.9 | 41.6 / 35.0 | 73.2 / 35.9 | 0.63 / 0.61 | 57 | 43 | 6 · 2 (0) | 93 · 10 · 0 |
| 반지름 하한 3–5 mm (후보 제외 / 반지름 올림) | 55–57 | 266–274 | 0–1 | ≤ 0.09 | 2.5 / 2.8–4.9 | — | (구 수 그대로, 재지 않음) | | | | | | |
| 내포 구 제거 | 56 | 270 | 0 | 0 | 2.55 | 94.3 | (지워지는 구 0) | | | | | | |

¹ ilp 2.5 의 (a) 56/57 은 V5B ep1995 s19953 t233 쥐기 전 한 건이다. 그 순간 실제 mesh 거리는 **+0.62 mm** (닿지 않음) 이고, ilp 구는 +0.04 mm 다.
- FC 기본 (−0.77 mm) 은 넘침 2.5 mm 가 그 사이를 메워 "본다" 였던 것이다.
- "mesh 까지 닿음" 은 57/57 이다.

원자료는 다음과 같다.
- 표: `fr_table.{py,json,md}`
- 덮임: `fr_build.{json,log}`
- 시간: `fr_time_{A,B}.{json,log}`
- 오프라인: `offline/fr_*.json` · `offline_summary.{json,log}`
- 탐색: `proto_options.*` · `proto_ilp_*`

`fc25` 행은 FC 의 숫자를 그대로 재현한다. (a) 57/57 · 34/44, (b) margin 0 7 · 새 3 (미HOLD 0), 10 mm 띠 97 · 새 12, CPU 74.5/58.0 (FC 72.7/56.7) ms 다.

### GPU 시간으로 바꾸는 법

V8 이 FC 기본의 GPU TO 중앙값 `t` 를 내면, 각 선택지는 대략 `t × (×FC 기본)` 이다.
- 예: `t` = 40 ms 이면 ilp 4 mm ≈ 28 ms, ilp 2.5 mm ≈ 36–37 ms.
- 단 이 비율은 CPU 에서 잰 것이다. GPU 는 구 수에 비례하는 부분 (ESDF 질의 · FK · check) 의 비중이 다를 수 있다 (§4 위험).
- CPU 단계별 중앙값: linearize · check 가 구 수를 따른다 (사슬 8 · 9 → FC 11 · 17 → ilp 4 mm 8 · 9 ms). qp 는 B1968 에서 7 ms 로 그대로다. C1807 (막대 근처) 에서는 활성 행에 따라 12 (사슬) · 28 (FC) · 15 (ilp 4) · 21 (greedy 6) ms 로 달랐다.

### 읽는 법 (implementer 의 정리, 판정은 사용자)

- **완전 덮임 (안 덮인 표본 0)**: FC 기본, greedy overflow 3–6, ilp overflow 2.5–6.
  - 이 중 사슬 대비 margin 0 의 새 미HOLD 청크가 0 인 것: FC · greedy 3 · greedy 4 · ilp 2.5 · ilp 3 · ilp 4.
  - 그중 가장 빠른 것은 **ilp overflow 4** (×0.70, 122 구) 다.
  - 대가는 둘이다. 최대 개도가 94.3 → 91.3 mm 가 된다 (사과 66.6 mm). 10 mm 띠 청크가 97 → 135 로 는다 (FC 대비 새 미HOLD 38, label 에 따라 HOLD).
  - **보장 · 넘침을 FC 와 똑같이 두고** 줄이는 것은 ilp 2.5 mm (46 구, ×0.89–0.92) 뿐이다.
- 넘침 5 mm 이상은 margin 0 의 새 미HOLD 가 생긴다 (3–11 청크). 손가락이 실제보다 5–6 mm 두꺼워져 테이블 · crate 쪽에서 걸리는 것으로 보인다 (원인 귀속은 재지 않음).
- gap · K 는 (b) 가 FC 와 같거나 적다. 대신 손가락 mesh 일부가 구 밖에 남는다. K 는 9 mm 까지 남고, gap 1.0 은 (a) "mesh 까지" 32–43/57 이다.

## 2. 고른 방법 — 규칙

모두 FC 의 후보 (mesh 안 0.5 mm 격자점 `c`, 반지름 `depth(c) + overflow`) 와 대상 (겉면 표본) 을 그대로 쓴다. 그래서 넘침 ≤ overflow 는 어느 선택지에서도 구성상 그대로다.

| 선택지 | 무엇을 바꾸나 |
|---|---|
| overflow (FC 의 flag) | 후보 반지름 = depth + overflow. 클수록 구 하나가 더 넓게 덮는다 |
| `solver="ilp"` | 후보 = 1 mm 격자 (0.5 mm 격자에서 축마다 하나 건너, 1.6 만) ∪ greedy 가 고른 중심. 0/1 정수계획 `min Σx s.t. 모든 대상이 고른 구 하나 이상 안` (scipy `milp`, HiGHS). greedy 보다 적고 덮임 검사를 통과할 때만 쓴다 |
| `gap` | 덮었다고 치는 반지름 = r + gap (구 자체는 그대로). 남는 표본의 깊이 ≤ gap |
| `max_spheres` | greedy 를 K 개에서 멈춘다 (뒤 중복 제거는 그대로) |

`ilp` 미리 푼 답 (`finger_cover_presets.py`) 은 다음과 같다.

| (overflow, gap) mm | 구 (greedy) | 최적 증명 | 하한 | 푼 시간 s |
|---|---|---|---|---:|
| 2.5, 0 | 46 (56) | 아니오 (600 s 상한; 900 s 에서도 46) | 43 | 600.7 |
| 2.5, 0.5 | 32 (48) | 예 | 32 | 268 |
| 2.5, 1.0 | 26 (32) | 예 | 26 | 152 |
| 3, 0 | 32 (48) | 예 | 32 | 268 |
| 4, 0 | 19 (28) | 예 | 19 | 51 |
| 5, 0 | 14 (24) | 예 | 14 | 59 |
| 6, 0 | 11 (18) | 예 | 11 | 36 |

- 답은 좌표가 아니라 **격자 index** 다. 런타임은 FC 와 똑같이 격자를 만들고 그 점을 고른다.
  - 그 점의 반지름을 `depth + overflow` 로 다시 계산하고, 대상 전부가 `gap` 안인지 검사한다.
  - 검사에 떨어지면 (mesh 가 바뀌면) WARNING 을 내고 그 자리에서 푼다 (`FINGER_COVER_ILP_TIME_LIMIT` 300 s).
  - 풀이가 greedy 보다 많거나 실패하면 greedy 답을 쓴다 (`source = greedy_fallback`).
- 위 표에 없는 overflow 를 `ilp` 로 주면 서버 시작 때 최대 5 분 푼다. 시간 상한에 걸리면 결과가 CPU 속도에 따라 달라질 수 있다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/ag3s/experiments/sources/gripper_cover.py:637-655` | `_greedy_exact_cover(…, pad=0.0, max_count=None)` | gap · K. 기본값이면 같은 계산이다 (`float(r)` 그대로) |
| `…/gripper_cover.py:696-754` (추가) | `FINGER_COVER_SOLVERS` · `FINGER_COVER_ILP_STRIDE` · `FINGER_COVER_ILP_TIME_LIMIT` · `_finger_candidates` (FC 후보 계산을 함수로 뺌 + 격자 index) · `_ilp_cover` · `_covered` | ilp |
| `…/gripper_cover.py:756-850` | `finger_body_spheres(…, gap=0.0, max_spheres=None, solver="greedy", ilp_time_limit, use_presets=True)` | 보고서에는 기본이 아닐 때만 새 키가 붙는다 (`solver` · `source` · `ilp` · `greedy_n_spheres` · `gap_mm` · `max_spheres`) |
| `…/gripper_cover.py:852-888` | `finger_body_capsules(…, gap, max_spheres, solver)`, 캐시 키에 셋을 더함 | |
| `benchmark/ag3s/experiments/sources/finger_cover_presets.py` (새) | `FINGER_BODY_ILP_PRESETS` 7 개 | 서버 시작 때 정수계획을 풀지 않게 |
| `benchmark/ag3s/experiments/reports/grounding_report.py:178-179` · `:205-212` · `:229` | `finger_cover` dict 의 `gap` · `max_spheres` · `solver` 를 준 것만 넘기고, `options` 보고서에도 준 것만 적는다 | 안 주면 FC 와 글자 그대로 같다 |
| `benchmark/trajopt/serve_safe.py:556-564` | `announce_finger_cover`: 선택지를 줬을 때만 `T43 FR` WARNING 한 줄 (ilp 답 출처 포함) | |
| `…/serve_safe.py:567-604` | `finger_cover_options`: 새 flag 셋. `--finger-cover` 없이 주면 `SystemExit`, gap < 0 · nan, max-spheres < 1, max-spheres + ilp 도 거절 | |
| `…/serve_safe.py:1285-1294` | `--finger-cover-solver {greedy,ilp}` · `--finger-cover-gap MM` · `--finger-cover-max-spheres N` | CLI |
| `tests/trajopt/test_t43fr_finger_cover_reduce.py` (새, 21) | 아래 | |

FC 와 같은 배선이다. 이 선택지들은 제약 로봇 모델의 옵션이고 AG3SConfig 필드가 아니다. 그래서 AG3SConfig 왕복 테스트는 늘리지 않았다.
- `--no-safe` · `--no-perception` 과의 조합은 FC 의 거절이 그대로 막는다. `finger_cover_options` 가 dict 를 내면 거절된다.

## 단위 검증

```bash
cd /mnt/dev/work
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa OMP_NUM_THREADS=2 \
PYTHONPATH=/mnt/dev/work nice -n 19 .venv-openpi-live/bin/python -m pytest -q -p no:cacheprovider tests/trajopt/test_t43fr_finger_cover_reduce.py
# 21 passed (152 s)
… -m pytest -q -p no:cacheprovider tests/trajopt/test_t43fc_finger_cover.py tests/trajopt/test_t43r_gripper_cover.py
# 48 passed (58 s) — FC · R 테스트 그대로
… -m pytest -q -p no:cacheprovider tests/trajopt tests/ag3s
# 2414 passed, 2 skipped, 0 failed (9 min 22 s) — outputs/impl/T43FR/pytest_all.log
```

기존 테스트는 한 줄도 고치지 않았다. FC 의 "꺼짐 = E3b 비트 동일" · "FC 2.5 mm = 56 구, 전부 덮음" 테스트가 그대로 통과한다.
- 비트 동일은 따로도 확인했다. 원본 `gripper_cover.py` 와 새 파일의 `finger_body_spheres` (2.5 · 4 mm) 는 중심 · 반지름 `array_equal`, 보고서 `==` 다.

새 테스트 21 개가 고정하는 것:
1. 안 주면 FC 그대로
   - 기본값을 명시해 불러도 구 · 보고서가 비트로 같고, 새 키가 없다. 모델 `options` 는 `{hands, overflow}` 이고 270 구다.
   - CLI 기본은 셋 다 `None` 이고 넘기는 키가 없다. 혼자 주거나 잘못된 값이면 `SystemExit` 다.
   - announce 는 선택지를 줬을 때만 `T43 FR` 줄을 낸다. 모르는 값 · 키 (`min_radius`) 는 거절한다.
2. ilp
   - 미리 푼 답 7 개 **각각**: `source == "preset"`, 구 수 = 답 = 기록 < greedy 다.
   - 보장도 각각 확인한다. 대상 전부 gap 안, 넘침 ≤ overflow + 0.3 mm, 중심이 mesh 안이다.
   - 답이 mesh 에 안 맞을 때: WARNING 을 내고 그 자리에서 풀이를 부른다. 풀이가 실패하면 greedy 와 비트로 같다.
   - 풀이가 덮지 못하는 답을 내면 쓰지 않는다.
   - 작은 문제에서 ilp 는 2 개 (최적), greedy 는 3 개다.
3. gap 0.5 · 1.0: 구 ≤ 48 · 32, 남는 표본 > 0, 깊이 ≤ gap, 넘침 보장 그대로다.
4. K 8 · 16 · 24: 구 ≤ K 이고, 보고서의 안 덮인 수 · 최대 깊이가 직접 잰 것과 같다.
5. 모델 배선
   - ilp 4 mm 는 122 구, 네 손가락 같은 구 (`preset`), 손바닥 그대로, 전부 덮음, 개도 = mesh 간격 − 2 × 4 mm 다.
   - `hands` · `gap` · `max_spheres` 가 보고서까지 간다.

## verifier 가 알아야 할 것

- **새 flag** (기본은 전부 꺼짐이고, 기존 기본값은 하나도 안 바꿨다):
  - `--finger-cover-solver {greedy,ilp}`
  - `--finger-cover-gap MM`
  - `--finger-cover-max-spheres N`
  - 모두 `--finger-cover` 와 함께만 쓴다.
  - 예: 사용자가 "ilp 4 mm" 를 고르면 V8 flag 에 `--finger-cover-overflow 4 --finger-cover-solver ilp` 를 더한다.
- 켜면 생기는 것: 시작 로그에 FC 줄 다음으로 `constraint finger cover (T43 FR): 구 수 선택지 {...}, ilp 답 출처 ['preset']` (WARNING) 이 나온다.
  - client metadata `finger_cover` 에 그 키가 들어간다.
  - npz 구 열 수는 표의 "전체" 다. 앞 46 열 (손바닥) 은 같다.
  - **`ilp` 4 mm 는 열 수가 122 로 축선 사슬과 같지만 구는 다르다.** 열 수로 FC 여부를 판별하지 말고 `finger_cover` metadata 를 본다. 같은 이유로 `fixed_input_ablation` 이 "sphere option 이 다릅니다" 로 멈추지 않을 수 있다. `outputs/impl/T43FR/fr_time.py` 방식을 쓴다.
- 미리 푼 답이 없는 overflow 로 `ilp` 를 주면 서버 시작이 최대 5 분 늦어진다 (WARNING 은 없다. 시작 로그의 `ilp` 보고서 `source: solved` · `sec` 를 본다).
- 재생산이 필요한 산출물: 없음. 옛 기록과 호환: 안 주면 키 · 값 모두 같다.
- **나는 재지 않았다**: GPU TO 시간, closed-loop, figure, label 층이 있는 margin 판정, (b) 의 새 행 원인 귀속 (FC 의 attribution 단계는 이번에 뺐다).

## 남은 위험

1. **CPU 비율 ≠ GPU 비율.** GPU 에서는 구 수에 비례하는 부분이 병렬이라, 줄인 만큼 덜 빨라질 수 있다. V8 의 GPU 시간과 함께, 고른 선택지 하나를 GPU 에서 한 번 재는 것이 확실하다.
2. ilp 2.5 mm 답은 최적이 증명되지 않았다 (46, 하한 43). 쓰는 데 문제는 없다 (보장은 검사한다).
3. 넘침이 클수록 (b) 의 10 mm 띠가 는다. 4 mm: FC 대비 새 미HOLD 38 청크다. label 층이 기록에 없어 실제 HOLD 수는 closed-loop 에서만 안다.
4. gap · K 는 손가락 mesh 일부를 덮지 않는다. CT 접촉 57 건에서는 K 8 까지도 "본다" 였지만, 접촉 위치가 손가락 등 · 옆이었기 때문이다. 모서리 접촉에서는 못 볼 수 있다.

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->

- GPU 에서 FC 기본 대비 비율은 CPU 보다 1 에 가까울 것이다. ilp 4 mm 는 ×0.75–0.9 정도로 본다.
- ilp 2.5 mm 는 closed-loop 지표 (접촉 · HOLD) 가 FC 기본과 거의 같을 것이다. 보장 · 넘침이 같고 구 배치만 다르기 때문이다.
