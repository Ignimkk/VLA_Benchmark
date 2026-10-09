# T43TP — 구현: TO 연산 시간 단축 (V11AB 행동 유지, TO p90 ≤ V8 수준)

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe · 2026-10-09
> 기준 benchmark `3ca1b21` (V11AB = TA) · task: [T43.task.md](T43.task.md) §"사용자 판정 (2026-10-09)" **TP**
> **커밋**: `166124c` 는 lead 가 사용자 요청으로 만든 **진행 중 스냅샷** (py_compile 만 확인) 이다. 최종은 그 위의 커밋이다 (아래 "커밋").
> 산출물: `outputs/impl/T43TP/` — `timing/{cpu,cuda}/` (통제 측정 원자료) · `raw/` (품질, reset) · `raw_keep/` (품질, keep-state) · `agg/*.json` · `prof/` (OSQP · ESDF · FK 마이크로 측정) · `p/` (스크립트)

## 무엇을 했나 (평이한 요약 먼저)

V11AB 의 TO 가 느린 이유를 같은 기계 · 같은 부하에서 단계별로 쟀다. 서버 기록의 V11AB 526 ms 는 대부분 **pod 부하** 탓이었다 (V11 라운드 load1 중앙 35–44, p90 190–228). 같은 부하에서 다시 풀면 V11AB 는 V8 의 1.7 배다 (중앙 98 vs 58 ms, GPU 필드 경로).

V8 대비 늘어난 몫은 셋이다.
- 16 step 이 FK · ESDF 질의를 두 배로 만든다.
- 같은 궤적의 FK · ESDF 평가를 한 solve 안에서 **2–3 번씩 다시** 한다 (gate · reference 위반 · 첫 iterate 는 같은 reference 다).
- eps 1e-4 의 OSQP 꼬리 (한 QP 최대 4000 반복) 가 있다.

그래서 **결과를 하나도 바꾸지 않는** 두 flag 를 만들었다.
- `--to-fast-eval`: 한 solve 안에서 각 평가를 한 번만 한다. FK · ESDF clearance 를 memo 하고, FK 는 CasADi zero-copy buffer 로 읽는다. destination label 은 distance 와 한 번에 읽는다. QP 는 solve 당 한 번 만들고 box 행만 다시 계산한다.
- `--esdf-host-mirror`: GPU 필드 격자를 청크마다 host 로 한 번 복사한다 (~1.3 ms). 그 뒤 모든 질의는 host 산술이다. 같은 값이다.

둘을 켜면 V11AB 와 **bit 동일** (1018/1018 청크, 궤적 SHA-1) 하다. TO 는 GPU 필드 경로에서 중앙 98 → 39 ms, p90 157 → 69 ms 다. V8 (58 / 74) 보다 낮다.

그 밖의 단축 후보 (QP 설정 · gate · 꼬리 성김 · 짧은 지평 · warm start · 반복 수 · ilp) 도 각각 flag 로 만들어 재고 TA 오프라인 지표로 V11AB 와 대조했다. bit 동일한 두 flag 만으로 목표에 닿으므로 **나머지는 권장하지 않는다**. 표는 §3 에 있다.

## 1. 통제된 측정 — V8 vs V11AB, 단계별 (같은 청크 · 같은 프로세스 · 순서 회전)

- 도구: `benchmark/trajopt/experiments/tp_timing.py`. TA 재구성 (`ta_offline.Run`) 위에서 단계 타이머를 클래스에 붙인다.
- 서버 방식으로 푼다 (`--mode server`): 설정마다 optimizer 하나를 run 전체에 유지한다. 그래서 OSQP 인수분해 · warm start 가 온라인처럼 이어지고, continuity 는 자기 직전 청크다.
- 청크마다 모든 설정을 회전 순서로 연달아 푼다 → 같은 부하다.
- 각 run 의 첫 청크는 뺐다. V8 기록 19 run, 1018 청크다.
- destination label: 하네스의 근사 label (nearest surface) 시간은 빼고, 서버와 같은 `CuroboEsdfField.label` 읽기를 같은 점에서 재서 넣었다 (`wall_ms`).
- `--field cuda`: 필드를 서버와 같은 `DeviceEsdfField` (GPU 격자, 질의점 칸만 gather) 로 올렸다. 단일 프로세스, < 1 GB.
- `--field cpu`: host `EsdfField` 다. whole-grid gradient cache 는 시간 밖에서 미리 만든다.

| 경로 (부하) | 설정 | TO 중앙 | p90 | p99 | FK | ESDF 읽기 | clearance 분류 | linearize | QP 조립 | OSQP (평균) | 기타 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **GPU 필드 = 서버** (load1 중앙 32 · p90 172, throttle 1.7 %) | V8 | 57.7 | 73.8 | 133 | 14.9 | 23.6 | 3.1 | 0.9 | 7.4 | 3.1 (3.5) | 1.6 |
| | V11AB | 97.9 | 156.7 | 268 | 31.2 | 42.6 | 7.2 | 1.0 | 9.0 | 5.4 (12.1) | 2.0 |
| | V11AB + fast-eval | 39.8 | 75.4 | 195 | 7.5 | 14.2 | 2.6 | 0.9 | 4.1 | 5.3 (12.3) | 1.8 |
| | **V11AB + fast-eval + host-mirror** | **38.9** | **69.2** | 173 | 7.1 | 12.0 | 2.5 | 0.9 | 4.0 | 5.2 (12.1) | 4.7 ¹ |
| CPU 필드 (load1 중앙 38 · p90 269, throttle 3.0 %) | V8 | 51.5 | 65.9 | 80 | 15.4 | 18.9 | 2.8 | 0.8 | 7.4 | 3.1 (3.5) | 1.5 |
| | V11AB | 85.9 | 136.9 | 229 | 29.6 | 33.0 | 6.6 | 0.9 | 8.9 | 5.3 (12.1) | 1.9 |
| | V11AB + fast-eval | 34.2 | 61.4 | 157 | 7.2 | 9.5 | 2.5 | 0.8 | 4.0 | 5.1 (12.1) | 1.7 |

단계 열은 청크별 배타 시간의 중앙값 (ms) 이다. OSQP 는 괄호에 평균을 함께 적었다 (꼬리가 크다).
¹ 기타에 청크당 host 복사 (~3 ms, 두 128³ 격자 + label) 가 들어 있다.
원자료: `agg/timing_cuda.json` · `agg/timing_cpu.json` (`stage_mean_ms` · `stage_p90_ms` · `calls_median` · 부하 표본 `timing/load_*.txt`).

읽는 법:
- **GPU 와 CPU 의 구분.** 서버 경로에서 GPU 를 쓰는 것은 ESDF 격자 읽기 (`DeviceEsdfField` gather) 뿐이다. FK (CasADi) · 분류 · linearize · QP 조립 · OSQP · check 는 전부 CPU 다.
- GPU 읽기는 호출마다 왕복 + sync 다. 1–4 k 점에서 distance 0.6–1.3 ms, gradient 2.1–3.9 ms 이고, 같은 점의 host 읽기 (0.4–1.3 / 0.9–2.7 ms) 보다 느리다 (`prof/esdf_bench.json`). 그래서 GPU 로 batch 하는 것보다 host 로 한 번 내리는 쪽이 빠르다 → host-mirror.
- **오프라인 대 서버 기록.** 서버 기록 V8 TO 는 중앙 70, p90 103 ms 다 (load ~15). 같은 청크의 오프라인 GPU 경로는 58 / 74 ms 다. 이 비 (×1.2–1.4) 를 V11AB + fast-eval + host-mirror 에 적용하면 **온라인 TO 기대 중앙 ≈ 47 ms, p90 ≈ 97 ms** 이다. V8 기록 (70 / 103) 이하다.
- 호출 수 (solve 당 중앙): FK 6 → 실제 계산 4, ESDF clearance 10 → 계산 4, field 읽기 29 → 13 (V11AB → fast-eval).
- **B 비용의 sparse 구조**: H 16 · nq 14 의 `P` 는 W 0 / 20, 경계 항 유무와 관계없이 nnz 1036, 대역폭 2 step 으로 같다. smoothing 의 D2 가 이미 그 자리를 차지한다. B 는 QP 구조 비용을 더하지 않는다.

## 2. 무엇을 바꿨나 — 단축 후보 (전부 기본 off, off ⇒ 비트 동일)

| flag (serve_safe) | config | 하는 일 | V11AB 대비 결과 |
|---|---|---|---|
| **`--to-fast-eval`** | `sqp.fast_eval` | 아래 넷. 한 solve 동안만 산다 (`solve` 가 `begin_memo` / `end_memo`, 예외에도 정리) | **bit 동일** |
| ↳ FK memo | `linearize.py:693` | 입력 값 (궤적 · q_now · 손가락 경로 · 쥔 점) 이 같으면 같은 배열을 돌려준다. read-only 로 만든다 | |
| ↳ ESDF clearance memo | `linearize.py:774` | read-only states 의 identity 를 key 로 쓴다. 부수효과 (`_last_obstacle_class` · `_last_lookahead` · `query_radii` …) 는 hit 때 다시 건다 | |
| ↳ buffered FK | `linearize.py:82` `_BufferedCall` | `Function.buffer()` 로 같은 컴파일된 함수를 numpy 에 직접 쓴다. `DM.nonzeros()` list 를 거치지 않는다 (H16 FK 5.2 → 1.6 ms) | |
| ↳ distance + label 한 번 | `curobo_field.py:311` `distance_and_label` · `linearize.py:886` | `_evaluate` 를 한 번만 돈다 | |
| ↳ QP 한 번 조립 | `problem.py:193` `box_bounds` · `:232` `rebox_problem` · `sqp.py:439` | 반복 사이에 box 행 (iterate · trust radius) 만 다시 계산한다 | |
| **`--esdf-host-mirror`** | `collision.esdf_host_mirror` | `DeviceEsdfField.mirror_to_host` (`device_field.py:209`) 가 청크당 한 번 D2H 한다. 그 뒤 `_gather` · `label` 은 host 를 읽고, 산술은 그 클래스의 host 경로다 (클래스 계약상 `EsdfField` 와 bit 동일). 호출은 `sqp.py:201` | **bit 동일** |
| `--qp-warm-iterate` | `qp.warm_start_iterate` | solve 의 첫 QP 를 직전 청크의 해가 아니라 `[iterate, 0]` 에서 시작한다 (dual 은 유지) | 근사 |
| `--qp-eps-after-first EPS` | `qp.eps_after_first` | 2 번째 이후 QP 의 eps | 근사 |
| `--qp-max-iter N` · `--qp-accept-max-iter` | `qp.max_iter` · `qp.accept_max_iter` | 상한에서 멈춘 iterate 를 부정확한 해로 받는다 (merit 가 판정) | 근사 |
| `--lookahead-stride S` | `horizon.lookahead_stride` | 꼬리 step 을 S 간격으로만 질의하고 행을 둔다. 창은 그대로다 | 근사 |
| `--sqp-warm-lookahead` | `sqp.warm_start_lookahead` | look-ahead 에서도 직전 계획을 K 만큼 밀어 seed 로 쓴다. continuity 기준이 정확히 그 계획일 때만 쓴다 (`sqp.py:570`) | 근사 |
| (기존) `--lookahead-gate-mm` · `--plan-horizon 12` · `--sqp-max/min-iterations 2` · `--finger-cover-solver ilp` | | gate 강화 · 짧은 꼬리 · 반복 수 · FR ilp 구 수 | 근사 |

## 3. 후보별 시간과 품질 (V11AB 대비)

시간은 CPU 필드, 서버 방식, V8 19 run 1018 청크다. 품질은 `ta_offline` (V8 19 + V7g 19 run, 2183 청크, 교차 439, HX 접촉 22) 를 `p/tp_agg.py` 로 V11AB (`A16qB20`) 와 대조했다.
- **keep** = optimizer · continuity 를 청크 사이에 유지한 오프라인 (온라인과 같은 warm start). QP warm start 계열은 이것으로만 뜻이 있다.
- **reset** = TA 와 같이 청크마다 reset 한 오프라인.

| 설정 | TO 중앙 / p90 / p99 (ms) | servo 허들 p10 (mm) · <0 청크 | 접촉 clear /22 | 새 / 해소 충돌 청크 (vs V11AB) | 사과 앞 141: 0.5° 넘게 움직임 · outward > 2 mm · 최대 | closing 132 움직임 | 판정 |
|---|---|---|---|---|---|---|---|
| V8 | 51.5 / 65.9 / 80 | — | — | — | — | — | 기준 시간 |
| V11AB | 85.9 / 136.9 / 229 | keep +1.4 · 30 / reset +0.9 · 37 | 14 / 15 | — | — | — | 기준 품질 |
| **fast-eval (+ host-mirror)** | **34.2 / 61.4 / 157** | 같음 | 같음 | 0 / 0 | 0 · 0 · 0 | 0 | **bit 동일, 권장** |
| qp-warm-iterate | 35.3 / 52.6 / 184 | keep +1.1 · 33 | 14 | 1 / 1 | 5 · 0 · 1.9 | 6 | 동등, 이득 작음 |
| qp-eps-after-first 1e-3 | 33.9 / 59.0 / 134 | keep +2.2 · 30 | 15 | 2 / 7 | 10 · 3 · 4.3 | 13 | 파지 이동 ↑ |
| qp-max-iter 1000 + accept | 34.2 / 63.5 / **115** | keep +1.6 · 31 | 14 | **0 / 2** | 1 · 0 · 0.3 | 9 | 동등, p99 만 줄임 (선택) |
| qp-max-iter 500 + accept | 33.9 / 60.2 / 89 | reset +1.1 · 35 | 14 | 3 / 6 | 6 · 2 · 15.5 | 9 | 파지 outward 15 mm |
| gate 10 mm | 31.3 / 61.5 / 175 | reset +1.0 · 36 | 15 | 0 / 0 | 2 · 4 · 4.1 | 9 | 사과 앞 outward 4 건 |
| gate 5 mm | 30.2 / 61.4 / 180 | reset +0.8 · 37 | 15 | 0 / 0 | 7 · 7 · 4.1 | 11 | 사과 앞 outward 7 건 |
| plan-horizon 12 | 28.7 / 46.8 / 112 | reset +1.0 · 38 | 15 | 6 / 2 | 4 · 6 · 4.6 | 10 | 새 충돌 6 |
| lookahead-stride 2 | 32.1 / 59.6 / 155 | reset +1.7 · 32 | 15 | 3 / 4 | 2 · 2 · 4.8 | 9 | 이득 작음 |
| lookahead-stride 4 | 31.1 / 55.9 / 140 | reset +2.2 · 32 | 15 | 6 / 4 | 8 · 3 · 5.5 | 8 | 새 충돌 6 |
| SQP 2 반복 | 29.1 / 52.2 / 139 | reset +1.0 · 37 | 14 | **15** / 0 | 4 · 2 · 3.1 | 12 | 새 충돌 15, 기각 |
| warm-lookahead + 2 반복 | 30.0 / 53.4 / 132 | keep +1.8 · 32 | 16 | 4 / 7 | 9 · 2 · **22.7** | 17 | 파지 outward 23 mm, 기각 |
| FR ilp (V8 run 만) | 31.6 / 59.1 / 148 | reset +3.2 · 13 (V8) | 10/14 | 0 / 1 | 3 · 3 · 4.3 (72) | 2/49 | 모델 변경, 이득 작음 |
| warm-iterate + eps-after + cap 1000 | 35.1 / 50.1 / 93 | keep +1.2 · 35 | 15 | 2 / 6 | 14 · 1 · 3.7 | 15 | 파지 이동 ↑ |
| 위 + warm-lookahead + 2 반복 | 29.4 / 46.0 / 81 | keep +1.9 · 31 | 15 | 4 / 9 | 18 · 1 · 6.3 | 22 | 파지 이동 ↑ |

- 비교 잣대 (solver 잡음): TA 에서 QP 정확도만 바꾼 Q4 가 base 대비 새 / 해소 2 / 2, 사과 앞 14/72 움직임을 냈다.
- 이 패스에서 V11AB 자체는 OSQP 상한 실패가 reset 60 · keep 16 건이다. cap 계열은 0 이다.
- GPU 경로의 조합 측정: fast-eval + host-mirror + warm-iterate + eps-after + cap1k 는 38.9 / 56.7 / 93 ms, 여기에 warm-lookahead 를 더하면 32.2 / 51.8 / 75 ms 다 (`agg/timing_cuda.json`). 둘 다 위 표처럼 파지 청크를 움직인다.
- GPU-batched 질의: GPU 왕복이 호출당 비용을 지배한다 (§1). 그래서 batch 대신 host mirror 로 구현했다. FK 를 GPU 로 옮기는 것은 buffered FK (H16 1.2 ms) 보다 빠를 수 없어 하지 않았다.

## 4. 권장 조합과 서버 flag

V11AB flag (`$E3B --obstacle-margin 0.010 --held-body-cover $T2 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20`) 위에 다음을 더한다.

```
--to-fast-eval --esdf-host-mirror
```

- 행동: V11AB 와 bit 동일하다 (오프라인 1018/1018 청크 궤적 SHA-1, 단위 테스트 §5). 온라인 A+B 행동은 V11AB 와 같아야 한다.
- 기대 TO: 오프라인 GPU 경로 (load1 중앙 32) 중앙 39 · p90 69 ms. V8 같은 조건은 58 / 74 ms 다. 서버 기록 보정 (×1.2–1.4) 후 **온라인 중앙 ≈ 47, p90 ≈ 97 ms** 이다 (V8 기록 70 / 103).
- 선택 (p99 꼬리만): `--qp-max-iter 1000 --qp-accept-max-iter` 를 더하면 CPU p99 157 → 115 ms 다. keep 품질은 새 충돌 0, 사과 앞 outward > 2 mm 0 이다. 다만 결과는 OSQP 허용오차 수준에서 바뀐다.
- **왕복 533 ms 의 경로** (V8 서버 기록, run 중앙의 중앙):

| 단계 | V8 기록 | 권장 조합 기대 | 533 ms 에 필요한 것 |
|---|---|---|---|
| π0.5 infer | 398 | 398 | 왕복 안에 다 둘 수 없다 |
| AG3S (지각 · ESDF) | 258 | 258 | |
| TO | 76 | ≈ 47 (p90 ≈ 97) | 이번 작업으로 V8 수준 |
| client · 전송 · 기록 (왕복 − server_total) | 209 | 209 | |
| 왕복 | 948 | ≈ 920 | |

TO 를 0 으로 해도 infer + AG3S 가 656 ms 라, 동기 왕복은 533 ms 에 들어갈 수 없다.
경로는 **chunk pipelining** 이다. 청크 k 실행을 시작할 때 청크 k+1 을 요청하고, 8 행 실행 (533 ms) 동안 infer → AG3S → TO 를 끝낸다. 이때 필요한 것은 server_total + 전송 ≤ 533 ms 이다.
그러려면 TO (p90 ≤ 100) 밖에서 약 390 ms 를 줄여야 한다:
- infer: flow step 수 · compile.
- AG3S: 258 → ~150.
- 전송 · 기록: 209 → ~50. `--record-constraints` 의 npz 기록이 여기에 들어 있는지 확인해야 한다.

이것은 측정하지 않은 요구치다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/trajopt/config.py:111` · `:630` · `:639` · `:702–711` · `:838` | `horizon.lookahead_stride` · `sqp.fast_eval` · `sqp.warm_start_lookahead` · `qp.warm_start_iterate/eps_after_first/accept_max_iter` · `collision.esdf_host_mirror` (+ 검증) | 후보 flag |
| `benchmark/trajopt/linearize.py:82` · `:540–566` · `:693` · `:774–826` · `:886` | `_BufferedCall` · memo · buffered FK · clearance memo (부수효과 재적용) · stride wrapper · fused label | fast-eval · stride |
| `benchmark/trajopt/sqp.py:198` · `:201` · `:300–310` · `:439` · `:464` · `:570` | memo 범위 · host mirror · warm-lookahead (정렬 검증) · rebox · 첫 QP x0 / 이후 eps | |
| `benchmark/trajopt/problem.py:193` · `:232` | `box_bounds` 분리 (build_problem 이 같은 함수를 부른다) · `rebox_problem` | QP 한 번 조립 |
| `benchmark/trajopt/qp.py:106` · `:125–185` | `solve(x0=, eps=)` · max-iter 수락 | QP 후보 |
| `benchmark/trajopt/serve_safe.py:1188–1218` · `:1895–1960` | 새 CLI 8 개 · `tp_overrides` · `tp_collision_overrides` · `announce_tp` | |
| `benchmark/ag3s/fields/curobo_field.py:299` · `:311` | `mirror_to_host` · `distance_and_label` | |
| `benchmark/ag3s/fields/device_field.py:209–232` · `_gather` · `label` | host mirror (label 격자를 바꾸면 mirror 도 버린다) | |
| `benchmark/trajopt/experiments/ta_offline.py:124–175` · `:272` · `:913–920` | TP 변형 (`_tp`, `_server_args` = 변형의 로봇 모델) · `--keep-state` (자기 직전 청크를 continuity 로) · `--ref-variant` | 품질 비교 |
| `benchmark/trajopt/experiments/tp_timing.py` (새) | 단계 타이머 · 서버 방식 · `--field cuda` · 부하 기록 | 통제 측정 |
| `tests/trajopt/test_t43tp_fast_to.py` (새, 38) · `tests/ag3s/test_config_roundtrip.py` (+TP 키 왕복) | | |

## 단위 검증

```bash
cd /mnt/dev/work && JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa \
  PYTHONPATH=/mnt/dev/work .venv-openpi-live/bin/python -m pytest -q -p no:cacheprovider tests/trajopt tests/ag3s
```

- TP 파일 38 passed. 다루는 것:
  - 기본 off · 왕복 · 거부 · CLI.
  - fast-eval bit 동일: primitive, 그리고 필드 + 쥔 점 + destination + obstacle margin + gate 열림 / 닫힘 / 없음 + B + C, 연속 3 solve.
  - memo 범위 · read-only · 부수효과 재적용.
  - buffered FK = CasADi call. `distance_and_label` = 두 읽기 (static shape 포함). `rebox_problem` = `build_problem`.
  - QP x0 · eps 복귀 · max-iter 수락. SQP 의 x0 / eps 전달 시점.
  - stride. warm-lookahead 정렬 조건.
  - host mirror bit 동일 (CPU tensor 로 `DeviceEsdfField`).
- 전체 `tests/trajopt` + `tests/ag3s` (CPU): 2514 passed, 2 skipped (`outputs/impl/T43TP/logs/pytest_full_2.log`).
- **off ⇒ 비트 동일 (실기록)**: 1018 청크에서 V11AB 와 fast-eval 의 궤적 SHA-1 이 모두 같다. host-mirror 를 더한 쪽은 GPU 필드에서 1018/1018 이다. V8 도 같다 (`agg/timing_*.json` `bit_identical_to_V11AB`, keep / reset 품질 지표도 같다).

## verifier 가 알아야 할 것

- 새 flag (전부 기본 off): `--to-fast-eval` · `--esdf-host-mirror` · `--qp-warm-iterate` · `--qp-eps-after-first` · `--qp-max-iter` · `--qp-accept-max-iter` · `--lookahead-stride` · `--sqp-warm-lookahead`. 기록 meta 의 `trajopt_config` 에 새 키가 실린다.
- 재생산할 산출물 · 옛 기록 호환: 없음 / 깨지지 않음. `ta_offline` 출력은 새 옵션을 줄 때만 키 (`keep_state` · `ref_variant` · `vs_ref_joint_deg_max`) 가 생긴다.
- 측정 한계:
  - 부하가 높고 변동이 컸다 (CPU 패스 load1 p90 269). 설정 사이는 같은 프로세스에서 회전 순서로 쟀으므로 비율은 공정하지만, 절대값은 부하에 묶인다.
  - 온라인 기대값은 V8 의 오프라인 / 기록 비로 환산한 것이다.
  - 서버의 TO 밖 시간 (verdict · 기록) 은 재지 않았다.
- **커밋**: `166124c` 는 lead 의 진행 중 스냅샷이다 (py_compile 만 확인). 최종 커밋은 이 문서와 함께 그 위에 올린 커밋이다.

## 열린 위험

1. 온라인 p90 은 환산값 ≈ 97 ms 로 목표 100 에 여유가 작다. 부하가 V11 라운드 수준 (load1 90+) 이면 넘는다. TO 는 단일 스레드 CPU 라 부하에 그대로 늘어난다.
2. host mirror 는 청크마다 격자 2–3 장 (~25 MB) 을 host 로 복사한다. 기록기의 worker 도 같은 격자를 내려받는다 (`cache=False`). 둘이 겹쳐도 값은 같지만 PCIe 사용이 두 번이다.
3. `--sqp-warm-lookahead` 의 정렬 검증은 continuity 기준 (`previous_physical_chunk`) 이 있을 때만 seed 한다. grasp continuity off 청크에서는 seed 가 없다 (안전한 쪽).
4. QP 근사 후보의 품질은 open-loop 오프라인 판정이다. 채택하려면 온라인 짝 비교가 필요하다.

## 내가 기대하는 결과

(verifier 는 측정 전에 읽지 않는다.)
온라인 V11AB + `--to-fast-eval --esdf-host-mirror` 는 V11AB 와 같은 청크 · 같은 접촉 · 같은 G 를 낼 것이다 (같은 seed · 같은 입력이면 궤적이 같다). TO 중앙은 40–50 ms, p90 은 V8 기록 (103 ms) 이하일 것이다.
