# R-port — 재측정이 실제로 16D 를 먹도록 이식한다

> writer: lead (A0) · 2026-09-25 · 사용자 판정으로 확정됨

## 왜 쪼갰나

1 차 스모크 결과: 분류상 `broken` 은 1 건인데 **쓸 수 있는 수치는 7 건 중 1 건**(R6 의 N2)이었다.
`R.task.md` 의 되돌아올 지점이 *"`broken` 3 건 이상이면 이식이 본체"* 였고, 글자는 안 맞지만
취지가 맞는 자리다. 사용자 판정(2026-09-25): **R-port → R-measure 로 쪼갠다.**

무효였던 이유는 전부 **스크립트가 16D 를 안 먹었기 때문**이지 값이 틀려서가 아니다.

## A1 이 할 것 — 이식 8 건

**이번에는 고친다.** 각 항목은 `R.impl.md` 의 예측과 `R.smoke.verify.json` 의 실측을 근거로 한다.

| # | 무엇 | 왜 |
|---|---|---|
| P1 | **R1 을 프로세스 분리 구조로 바꾼다** (사용자 판정) | `pipeline.py:906` 의 T0 불변식(backend=curobo 인데 legacy EsdfBuilder 생성)이 한 프로세스 비교를 금지한다. **불변식을 건드리지 않는다** — legacy 로 한 번, cuRobo 로 한 번 따로 돌려 각각 npz 로 내고, 대조는 **밖에서** 한다. `verify_backend.py` 를 `dump` 단계와 `compare` 단계로 가른다 |
| P2 | **R4 `ground_truth.py`** — 출력 경로(`:32` 의 박힌 `OUT`, savefig `:412`)를 인자로 뺀다 | 지금은 archive 의 `archive/14d-era-20260923/figures/curobo-ground-truth.png` 을 덮어쓴다. 기본값은 `figures/r-16d/` |
| P3 | **R4 자산 짝 검사를 추가한다** | 1 차가 **자세만 16D, depth·필드는 09-12 자 `/tmp` npz** 인 혼합 실행이었는데 **스크립트가 그것을 안 잡았다.** 세 인자의 출처가 같은 기록인지 확인하고, 아니면 **즉시 실패**시킨다. 조용히 섞이는 것이 이 검토가 반복해서 만난 실패다 |
| P4 | **R5 `verify_two_tier.py:7`** — `/tmp/rby1_frame.npz` 를 인자로 뺀다 (argparse 없음) | 09-11 자 14D 산물을 말없이 읽는다 |
| P5 | **R3 이 기록을 먹게 한다** | 지금은 `TransportScene(settle_steps=400)` + seed 로 **새 씬을 만든다**(`:61`, `used_saved_run: false`). 14D 값 1.5 복셀은 *실제 파지* sweep 에서 나왔으므로, 16D 긴 기록의 파지 구간(정책 호출 33~36, `t_step` 264~288)을 먹어야 같은 것을 재는 것이다. **`--thresholds` 상한도 넓힌다** — 1 차가 3.0 을 골랐는데 그것이 후보 목록의 상한값이라 잘린 값이지 측정이 아니다 |
| P6 | **PHASE-1 — 단계 경계를 기록에서 파생한다** | `ground_truth.py:_phase(t, b=(24,56,72))` 는 플래그가 없다. 긴 기록에서 `t_step ≥ 72` 면 전부 `grasp` 이라 **호출 9~49 (41 프레임)** 이 잘못 라벨된다. 실제 파지는 왼 그리퍼 `state[7]` 가 1.0 → 0.44 로 닫히는 **호출 33~36**. 경계를 그리퍼에서 뽑아라 |
| P7 | **R7 `step7_state_lag.py:52` 의 `STEP_MS = 16.0` 단위를 확정한다** (사용자 판정) | 기록의 `t_step` 은 0·8·16…392 로 **제어 스텝**을 8 씩 센다. `meta.json` 의 `ctrl_hz=15`, `open_loop_horizon=8` → 제어 스텝 하나가 66.7 ms, 기록 한 스텝이 **533 ms**. **먼저 `STEP_MS` 가 무엇에 곱해지는지 코드로 확정하고**, 그 다음 기록의 `ctrl_hz`/`open_loop_horizon` 에서 파생시켜라. 상수로 박지 마라. 14D·16D 공통 문제다 |
| P8 | **R6 의 N1 을 재는 스크립트를 새로 쓴다** | self-collision 제약이 `benchmark/trajopt/` 에 **코드 0 줄로 아예 없다**(잠복). 14D 값 *"움직일 수 있는 구의 최소 여유 185.6 mm"* 는 문서에만 있고 내는 스크립트가 repo 에 없다. `outputs/verify/` 가 아니라 `benchmark/ag3s/experiments/` 에 둔다 — 반복해서 쓸 것이다 |

**P7 을 고칠 때 주의**: 단위가 바뀌면 14D 수치와의 비교 기준도 같이 바뀐다. 옛 값을 다시 계산해
바꾸지 말고, **"14D 값은 옛 단위, 16D 값은 새 단위" 를 `R.impl.md` 에 명시**해라. A3 가 그걸 표에 적는다.

## A2 가 지금 할 것 — 코드 변경 없이 되는 둘

| # | 무엇 | 어떻게 |
|---|---|---|
| M1 | **R2 를 실제 attention 으로 다시 돌린다** | 1 차는 `attention=synthetic` 이었다. **npz 는 있다** — `benchmark/ag3s/asset/data/attention_16d_long.npz` (50 프레임, `records` 가 16D 긴 기록을 가리키고 `t_step` 0…392 일치). `--attention` 에 그것을 준다. cuRobo 와 legacy 대조 둘 다 |
| M2 | **F20(잔상)을 잰다** | 1 차에서 안 돌렸다. `studies/a4_task_tsdf.py` 계열. 14D 값은 decay 끔에서 −6.7 mm |

figure 는 **전부 `figures/r-16d/`** 아래. archive 의 이름을 덮어쓰지 않는다.

## A3 가 지금 할 것

1 차 결과를 로그의 `## R` 절에 **확정으로** 적는다 (사용자 판정이 났다). 쓸 것:
- smoke 분류 7 건과 **"쓸 수 있는 수치는 1 건"** 이라는 결과.
- **R6 의 N2 만 16D 값으로 채운다**: arms 120 구 해석적 채널 끔 최대 **+258.2 mm** · 중앙 **+47.7 mm** ·
  낙관 질의 **1241** → 켬 **0**. all 194 구 **+408.0 mm** → **0**. 출처는 `R.smoke.verify.json`.
- 나머지 6 건은 16D 칸에 **"R-port 대기"** 와 그 이유 한 줄.
- 사용자 판정 셋 (R 쪼개기 · R1 프로세스 분리 · R7 단위 확정 후 재측정).
- **CLAUDE.md 의 venv 표가 틀렸던 것**: 둘이 아니라 셋이고 `.venv-openpi-live` 에 mujoco 와 curobo 가
  같이 있다 (2026-09-25 실측). 이것은 환경 사실이므로 로그에 남긴다.

**`R.smoke.verify.json` 의 `numbers` 에 없는 숫자는 여전히 쓰지 않는다.**

## 이 STEP 이 답하는 질문

**1 차에서 무효였던 6 건이 이식 뒤에 16D 를 실제로 먹는가.** 먹은 뒤의 값이 14D 와 같은지 다른지는
R-measure 가 답한다.
