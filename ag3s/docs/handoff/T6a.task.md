# T6a — 기록기가 **참값과 chunk 를 남기게** 한다

> writer: lead (A0) · 2026-09-26 · 주 담당 **A1(implementer)** · 산출 **하나**: `handoff/T6a.impl.md`
> 사용자 판정 2026-09-26: *"기록기를 먼저 고친다."*

## 왜 — 세 번 연속으로 같은 자리에서 막혔다

| 언제 | 못 잰 것 | 왜 |
|---|---|---|
| `T5c` | `refined` 대 `reference` clearance | `actions` 배열이 없다 |
| `T5c` | 과제가 실제로 완결됐나 | `qpos`·물체 자세가 없다 |
| `T5f` | `violated` 가 **어느 제약**인가 | verdict 가 위반 제약의 신원을 안 싣는다 |
| **`T6`** | **seq 38 부터 영구히 멈춘 자리가 어디인가** | 위 셋 전부 |

**T6 에서 75 chunk 중 42 번 멈췄고 seq 38 부터 끝까지 38 chunk 연속이다.** 그 자리가 어디인지
알아야 고칠 수 있는데, 기록에 로봇 자세도 물체 자세도 없어 **측정할 방법이 없다.**
`T5` 의 핵심 합격 조건 하나도 같은 이유로 미측정으로 닫혔다.

## 무엇을 남기나

**planning record 마다:**

| 키 | 무엇 |
|---|---|
| `actions` | 서버가 준 refined chunk `[H, ACTION_WIDTH]` |
| `actions_reference` | shadow 일 때만. 정책 원본 chunk |
| `qpos` | 그 planning frame 의 로봇 관절 전체 |
| `object_poses` | MuJoCo 참값 — 과일 넷 + crate 의 body 위치(그리고 가능하면 자세) |
| `max_violation_pair` | `max_violation_m` 을 만든 **제약의 신원** — 어느 link 대 어느 candidate/obstacle 인가 |

**control record 마다**: `qpos` (600 개). 이것이 있어야 멈춘 구간에서 팔이 정말 안 움직였는지
확인된다.

## 크기를 걱정하지 마라 — 재 보고 정해라

`[50, 16]` float32 는 chunk 당 3.2 KB 다. 75 chunk 면 240 KB, 양쪽 chunk 를 다 실어도 0.5 MB.
`qpos` 600 개도 작다. **실제 크기를 재서 `impl.md` 에 적어라.** 1 MB 를 크게 넘으면 그때
말해라 — 지금 추측으로 줄이지 마라.

## 깨뜨리면 안 되는 것

- **`plot_t0.py` 가 읽는 형식** (`benchmark/ag3s/experiments/live/plot_t0.py:37` 이
  `frames.jsonl` 을 읽는다). 키를 **더하는 것**이지 바꾸는 것이 아니다.
- **completeness 표** — `expected_observation_frames` · `unexplained_carried_or_stale` 등
  지금 나오는 항목이 그대로 나와야 한다.
- **`--record-frames` 없이 돌 때의 동작.** 기록을 안 할 때는 아무것도 계산하지 않는다 —
  MuJoCo 참값을 매 frame 뽑는 비용을 기록 안 할 때까지 물리지 마라.

## `max_violation_pair` — 어디서 오나

`T5f` 가 *"기록의 verdict 가 위반 제약의 신원을 안 싣는다"* 로 막혔다. SQP 가 어느 제약에서
최대 위반을 냈는지는 최적화기 안에 있다 (`benchmark/trajopt/` 의 `sqp.py`·`linearize.py`
근처). **코드를 읽고 그 신원을 꺼낼 수 있는지 보고, 못 꺼내면 왜인지 `impl.md` 에 적어라.**
억지로 만들지 마라 — 없으면 없다고 적는 것이 낫다.

## 검증 (네가 하는 것)

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -u -m pytest tests/ -q
```

직전 **740 passed**. 테스트를 더해라 — 새 키가 실린다 · 옛 키가 그대로다 ·
`--record-frames` 없이는 아무 비용도 안 든다.

**긴 rollout 을 돌리지 마라.** 재실행은 lead 가 한다.

## 규칙

- 네 소유: `benchmark/ag3s/**/*.py` · `benchmark/trajopt/*.py` · `tests/**` ·
  `pi05_TO_hybrid/rby1_bringup/pi05_infer.py` · `handoff/T6a.impl.md`.
- **포트 8000 서버(PID 4004754)를 죽이지 마라.** 재기동은 lead 가 한다. `pkill -f` 금지.
- 커밋하지 않는다. 규칙 I — technical term 은 영어로.
- 박힌 수치(75 · 42 · 38 · 740 · 3.2 KB) 밖의 값을 지어내지 마라.
