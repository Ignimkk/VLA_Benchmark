# T5e — target 을 **1 등으로 고르고 `confirm_frames = 3` 으로 누른다**

> writer: lead (A0) · 2026-09-25 · 주 담당 **A1(implementer)** · 산출 **하나**: `handoff/T5e.impl.md`
> 사용자 판정 2026-09-25.

## 왜 — 측정이 확정한 것

`T5d` 가 ep1807 의 seq 1..51 을 전수 측정했다. **score 를 무시하고 1 등만 보면 빈 chunk 가 0 개다.**

```
seq  1-15 (15) : apple
seq 16-30 (15) : crate
seq 31-32 ( 2) : pear     <- 흔들림
seq 33-34 ( 2) : crate
seq 35    ( 1) : apple    <- 흔들림
seq 36-51 (16) : crate
```

**`target_score_threshold = 0.25` 하나가 이 중 35 chunk 에 `LOW_SCORE`/"target 없음" 이라는
이름을 붙이고 있었다.** 그리고 그 35 건이 fine layer 5 mm 가 안 붙는 원인이기도 하다
(fine window 의 중심이 grounding target 이라 target 이 없으면 fine layer 자체가 없다 —
`T5c` 의 `one_tier_set_equals_low_score_set: true`).

**새 label 이 3 번 연속 1 등이어야 바꾼다** 를 적용하면 흔들림 둘이 다 눌린다:

```
seq  1-17 (17) : apple
seq 18-51 (34) : crate
```

`N=2` 로는 pear 2 chunk 가 살아남는다. **3 이 맞는 값이고, `GraspLatch` 가 이미 쓰는 값이다**
(`benchmark/trajopt/grasp_latch.py` 의 `confirm_frames: int = 3`).

## 고칠 자리

| 어디 | 지금 |
|---|---|
| `benchmark/ag3s/stages/target_grounding.py:375` | `clusters.sort(key=lambda c: (-c.target_score, int(c.point_indices[0])))` — **정렬은 그대로 둔다** |
| `benchmark/ag3s/stages/target_grounding.py:378-381` | `if best.target_score < cfg.target_score_threshold: return LOW_SCORE` — **여기가 문제다** |
| `benchmark/ag3s/config.py:91` | `target_score_threshold: float = 0.25` |
| `benchmark/ag3s/configs/default.yaml:25` · `configs/rby1_three_camera.yaml:55` | 같은 값 (`tests/ag3s/test_config.py:45` 가 일치를 강제한다) |

## 무엇을 하나

**1 등을 target 으로 쓴다. 그리고 새 label 이 `confirm_frames` 만큼 연속 1 등이어야 바꾼다.**

- cluster 가 **하나도 없으면** 여전히 target 없음이다. 그건 진짜 없는 것이다.
- **정체(label)는 그대로 유지해라** — `obj0` 같은 id 를 `GraspLatch` 가 쓰고 있다.
- `target_score` 는 계속 계산하고 `metrics` 에도 남겨라. **버리는 것은 "그 값으로 거부하는 것"
  이지 값 자체가 아니다.** `runner_up_score`(오늘 더한 키)도 그대로 둔다.

**설계에서 네가 정할 것 — `confirm` 상태를 어디에 두나.** `target_grounding` 은 지금 frame 마다
독립이라 상태가 없다. 후보는 둘이고 **코드를 읽고 골라 `impl.md` 에 근거를 적어라**:
- `AG3S` pipeline 이 `frame_index` 를 들고 있으니 거기에 둔다
- `grasp_latch.py` 의 `_Confirm` 을 재사용한다 (같은 규칙을 두 번 구현하지 않는 이점)

**`target_score_threshold` 를 지우지 마라.** 설정은 남기되 **기본 동작에서 거부에 쓰지 않는다.**
지우면 되돌릴 수 없고, 이 판정이 뒤집힐 여지를 없앤다.

## 반드시 보고할 것 — 회귀 기준선이 **바뀐다**

`has_target 9/15` 는 **바로 이 threshold 가 만든 값이다.** 이 변경으로 **의도대로 올라간다.**
**그것은 회귀가 아니다.** 기준선 셋 중:

| | 예상 |
|---|---|
| `has_target 9/15` | **바뀐다 (의도한 것)** |
| `위반으로 시작 14/15` | ? — 네가 코드를 읽고 예상을 적어라 |
| frame0 `clearance_before` +0.15718632962849477 mm | ? — 같다 |

**예상을 `impl.md` 에 적고 사실로 쓰지 마라.** 측정은 A2 가 한다. 기준선 갱신은 사용자 판정이다.

## 이번에 하지 않는 것

- **`camera_skew` / `camera_transform_stale` 판정을 건드리지 않는다** (사용자 판정). head cam 이
  팔에 가려 사과를 seq 18 에 7 pixel 밖에 못 보는 구간이 있지만, 이번 라운드 대상이 아니다.
- **self-filter 를 건드리지 않는다.** 사과가 집게 안에 들어온 뒤 지워지는 것은 **옳은 동작**이다
  (사용자 판정) — 쥔 사과를 장애물로 두면 로봇 자신과의 충돌로 읽힌다.
- `safe_policy.py:411` 의 gripper 열 번호 **6 (14D 배치)** 문제 — `wire.gripper_columns()` 가
  16D 에서 `(7, 15)` 를 준다. **`A2` 가 찾아 둔 별건이고 이번 STEP 이 아니다.** `impl.md` 에
  "봤고 안 건드렸다" 만 적어라.

## 검증 (네가 하는 것)

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -u -m pytest tests/ -q
```

직전 **720 passed**. 테스트를 더해라:
- 1 등이 threshold 아래여도 target 이 된다
- 새 label 이 3 번 연속이어야 바뀐다 (2 번으로는 안 바뀐다)
- cluster 가 0 개면 여전히 target 없음
- `obj0` 같은 label 정체가 유지된다

## 규칙

- 네 소유: `benchmark/ag3s/**/*.py` · `benchmark/trajopt/*.py` · `tests/**` · `handoff/T5e.impl.md`.
  **`docs/*.md`·figure·`*.verify.json` 을 건드리지 않는다.**
- **측정하지 않는다.** 긴 rollout 을 돌리지 않는다 — 단위 테스트까지다.
- **포트 8000 서버(PID 1987912)를 죽이지 마라.** 재기동은 lead 가 한다. `pkill -f` 금지.
- 커밋하지 않는다.
- 규칙 I — technical term 은 영어로 (`threshold`·`cluster`·`grounding`·`latch`·`confirm_frames`).
- 박힌 수치(0.25 · 3 · 720 · seq 구간 · 9/15 · 14/15 · +0.157 mm) 밖의 값을 지어내지 마라.
