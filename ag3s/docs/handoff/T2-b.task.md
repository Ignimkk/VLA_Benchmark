# T2-b — T2 의 미측정 두 항목을 채운다 (ep1807)

> writer: lead (A0) · 2026-09-25 · 주 담당 **A2(verifier)** · 산출 파일 **하나**: `handoff/T2-b.verify.json`

## 왜 이것인가

`T2.verify.json` 이 nine events 를 다 냈는데 **acceptance 조건 두 개가 측정되지 않은 채로 남았다**
(`not_measured` 의 1·3 번). 사용자 판정(2026-09-25): **이 둘을 채운 뒤에 T2 를 판정한다.**

## 자산 — `T2.verify.json` 과 **같은 기록**이다

| | |
|---|---|
| record | `outputs/live_test/20260925_ep1807/run_0000/run_0000` — 75 observations / 600 control steps |
| attention | `benchmark/ag3s/asset/data/attention_16d_ep1807.npz` `(75,3,3,18,8,3,16,16)` |
| venv | `.venv-ag3s` (numpy 2.4.6). `MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work` |
| `self_filter_inflation` | **0.05 m** (2026-09-25 부터의 기본값). 0.02 로 구운 npz 와 섞지 마라 |

**`T2.verify.json` 이 확정한 frame 번호 — 이 밖의 frame 번호를 새로 지어내지 마라.**
관측 frame index 는 0..74, `t_step` 은 control step.

| event | frame | t_step |
|---|---|---|
| target_acquired | 0 | 0 |
| target_latched | 2 | 16 |
| grasp_contact | **19** | 152 |
| attached | **19** | 152 |
| transport_start | 21 | 168 |
| destination_attention_raw | 28 | 224 |
| destination_attention_locked | **None** | — |
| placement | **32** | 256 |
| detached | **32** | 256 |
| task_state_reset | **None** | — |

HELD 구간은 **frame 19~31** 이다. frame 32~74 는 `latch.phase == 'placed'` 로 유지된다.

---

## 측정 1 — attach 이후 held object 가 **robot collision geometry 에 들어가고 obstacle field 에서 빠지는가**

T2 의 acceptance 조건인데 아직 아무 수치가 없다. **E3(쥔 물체가 optimizer 에 도달하지
않는다 — 과거 발견)와 직결되는 항목이다.**

**frame 별로 두 쪽을 나란히 낸다** (frame 15~40 을 최소 범위로, 되면 0~74 전부):

| 무엇 | 어디서 읽나 |
|---|---|
| `ag3s.attached` 가 `None` 이 아닌가 | `benchmark/ag3s/runtime/pipeline.py:158` (`attached` property) |
| attached sphere 개수와 반지름 | `benchmark/ag3s/constraints/attached.py:212` (`attached_spheres`) |
| attached point 가 robot base frame 에서 어디 있나 | `benchmark/ag3s/constraints/attached.py:53` (`attached_points_in_base`) |
| **같은 점이 obstacle field 에도 남아 있나** | held object 위치에서 ESDF/obstacle primitive 를 조회해, 그 자리가 장애물로도 잡혀 있는지 |

**핵심 수치 셋** — 이 셋이 이 측정의 답이다:
1. `attached` 가 처음 `not None` 이 되는 frame (frame 19 와 일치하는가)
2. 그 frame 에서 **attached sphere 개수** 와 **obstacle cluster 개수의 변화** — held object 가
   obstacle 쪽에서 실제로 빠졌으면 cluster 가 하나 줄거나 target cluster 가 obstacle 목록에서
   사라진다. **줄지 않으면 double counting 이다 — 그 사실을 그대로 적어라.**
3. `detached` (frame 32) 이후 `attached` 가 `None` 으로 돌아오는 frame

**double counting 이면 그것이 발견이다.** 고치지 마라 — 수치로 적고 넘겨라.

## 측정 2 — accumulated TSDF 의 **residual(잔상) 지속 frame 수**

decay 는 기본 꺼져 있다: `benchmark/ag3s/config.py:480` `time_decay = 1.0`,
`:484` `frustum_decay = 1.0` (1.0 = 감쇠 없음). **끈 상태에서 잔상이 몇 frame 남는지**가 답이다.

apple 은 frame 0 에 `[0.564, 0.320, 0.850]` 에 있고 frame 74 에 `[0.490, -0.081, 0.858]` 로
간다 (총 408.5 mm). **떠난 자리에 TSDF occupancy 가 얼마나 오래 남나**를 센다:

- apple 의 **frame 0 위치**에 반지름 5 cm 구를 두고, frame 별로 그 구 안의 TSDF/ESDF
  occupancy(또는 obstacle point 수)를 센다. transport_start(frame 21) 이후에도 0 이 아니면 잔상이다.
- **답의 형태**: `residual_frames_after_transport_start` = 정수, 그리고 frame 별 occupancy 계열.
- 비교군으로 움직이지 않은 과일 하나(pear, `[0.465, -0.324, 0.856]`, drift 0.6 mm)를 같은 방법으로
  재서 **"원래 거기 있는 것"과 "잔상"의 신호 크기를 나란히** 둔다.

---

## 동시에 A1 이 같은 tree 를 고치고 있다 — **파일 지문을 남겨라**

A1 이 `benchmark/ag3s/stages/target_grounding.py` 의 `metrics` dict 에 `runner_up_score` 를
더하는 중이다. 그 변경은 latch 의 `confident` 판정을 바꿔 **attach frame 을 옮길 수 있다.**
네 측정은 **변경 전 코드**의 것이어야 한다 (`T2.verify.json` 과 같은 조건).

**각 측정 run 직전·직후에 이것을 찍고 `verify.json` 의 `code_state` 에 넣어라:**

```bash
md5sum /mnt/dev/work/benchmark/ag3s/stages/target_grounding.py /mnt/dev/work/benchmark/trajopt/grasp_latch.py
```

**두 값이 run 전후로 다르면 그 run 은 버리고 다시 돌린다.** 그 사실도 적는다.

## 규칙

- **해석·판정을 쓰지 않는다.** 수치와 경로만 `benchmark/ag3s/docs/handoff/T2-b.verify.json` 에.
  schema 는 `handoff/_SCHEMA.verify.json`.
- **figure 3 종**(규칙 A: 실제 씬 · 그래프 · 표)을 `benchmark/ag3s/docs/figures/t2b/` 아래
  **새 이름**으로. `figures/t2/` 의 것을 덮지 마라. `make-figure` skill 을 읽고 스타일을 맞춘다.
- raw 와 script 는 `outputs/verify/T2b/`.
- 코드를 고치지 않는다. `benchmark/**/*.py` 는 A1 의 것이다 — 측정 script 는 `outputs/verify/T2b/` 에 쓴다.
- **`--max-steps` 를 늘리지 않는다.** 600 이 맞는 조건이다 (사용자 판정 2026-09-25).
- `pkill -f` 금지 (자기 셸을 죽인다). 긴 실행은 `python -u`. PID 를 뽑아 `kill <PID>`.
- **여기 박힌 frame 번호 밖의 값을 지어내지 마라.** 모르면 `not_measured` 에 적는다.

## 이 task 가 답하는 질문

**held object 가 robot 쪽으로 옮겨 앉는 것이 실제로 일어나는가, 그리고 떠난 자리의 잔상이
몇 frame 남는가.**
