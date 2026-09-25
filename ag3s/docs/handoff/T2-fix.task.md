# T2-fix — `runner_up_score` 를 metrics 에 넣는다 (사용자 판정 2026-09-25)

> writer: lead (A0) · 2026-09-25 · 주 담당 **A1(implementer)** · 산출 파일 **하나**: `handoff/T2-fix.impl.md`

## 확정된 사실 — 이것이 수정 근거다 (`T2.verify.json` 출처)

`target.metrics` dict 에 **`runner_up_score` 키가 없다.** 그래서:

| 어디 | 무엇 |
|---|---|
| `benchmark/ag3s/stages/target_grounding.py:394-402` | `metrics` dict 를 만드는 자리. 키 7 개 — `mean_attention` · `max_attention` · `attention_mass` · `point_count` · `spatial_compactness` · `distance_from_attention_peak` · `rms_radius`. **`runner_up_score` 가 없다** |
| `benchmark/trajopt/safe_policy.py:341` | `runner_up=float(metrics.get("runner_up_score", 0.0) or 0.0)` — 없으니 **항상 0.0** |
| `benchmark/trajopt/grasp_latch.py:142` | `confident = score >= self._cfg.score_ratio * max(runner_up, 1e-9)` |
| `benchmark/trajopt/grasp_latch.py:68` | `score_ratio: float = 1.3` |

`runner_up = 0.0` 이면 문턱이 `1.3 × 1e-9 = 1.3e-9` 다 — **`score > 0` 이면 무조건 `confident`.**
*"1 등과 2 등의 격차로 확신을 판단한다"* 는 장치가 **ep1800·ep1807 두 기록 모두 75/75 frame 에서
0.0 으로 무력화돼 있었다.** ep1807 frame 0 의 `target_score` 는 **0.8051** 이다.

## 무엇을 하나

**`runner_up_score` 를 `metrics` 에 채운다.** 1 등이 `best` 이므로 **2 등의 `target_score`** 다.

- `target_grounding.py` 의 같은 함수 안에 후보 cluster 들이 이미 있다 (`clusters`, 그리고 `best` 를
  고른 정렬). **2 등을 어디서 얻는지 코드로 확인하고** 그 값을 넣는다.
- **후보가 하나뿐이면** 2 등이 없다. 그 경우의 값을 **네가 정해 근거를 적어라** — `0.0` 으로 두면
  지금과 같은 "항상 confident" 이고, 그것이 옳을 수도 있다(경쟁자가 없으면 확신해도 된다).
  **어느 쪽을 골랐든 `impl.md` 에 왜인지 한 줄.**
- `LOW_SCORE` 로 빠지는 경로(`target_grounding.py:380` 부근, `target` 이 `None`)는 `metrics` 를
  만들지 않는다 — 건드리지 마라.

## 기대되는 부작용을 먼저 적어라 (고치기 전에)

이 변경은 **latch 가 언제 잠기는지를 바꿀 수 있다.** 지금까지 모든 frame 이 `confident` 였으니,
2 등이 1 등의 1/1.3 = 0.77 배보다 크면 그 frame 은 **이제 confident 가 아니다.** 그러면:

- `target_latched` 가 늦어지거나 아예 안 걸릴 수 있다 (ep1807 은 frame 2 에 걸렸다)
- 회귀 기준선이 움직일 수 있다: **`위반으로 시작 14/15` · `has_target 9/15` · `frame0 clearance_before +0.157 mm`**

**측정은 네 몫이 아니다.** 단위 테스트까지만 하고, 위 셋이 어떻게 될지에 대한 **예상을
`impl.md` 에 적어 A2 가 확인할 것을 명시**한다. **추측한 수치를 사실처럼 쓰지 마라.**

## 같이 확인할 것 — 판정 (a) 는 코드가 **이미 따라가 있다**

사용자 판정: *"gap_filling_capsules 의 조용한 삼킴을 소리 나게 만든다."*
`benchmark/ag3s/experiments/sources/mujoco_source.py` 에 `_warn_no_capsules`(`:493`)와
`_WARNED_NO_CAPSULES` 가 **uncommitted 로 이미 들어가 있다.** 확인만 한다:

1. `KeyError` 경로와 `bounding_capsules` 가 빈 list 를 내는 경로 **둘 다** 경고하는가
2. `tests/ag3s/test_gap_filling_capsules.py` (guard 7 개) 가 통과하는가
3. `UNCOVERED_LINKS` 전부가 실제로 capsule 을 내는가 — **경고가 지금 하나도 안 떠야 정상이다**

**다르면 고쳐라. 같으면 "이미 됐다" 만 적어라 — 다시 쓰지 마라.**

## 판정 (b) 는 **하지 않는다** (사용자 판정 2026-09-25)

손바닥(`ee_left`/`ee_right`)을 **constraint model 에 넣지 않는다.** 기존 동작 유지.
`benchmark/ag3s/experiments/reports/rby1_transport.py:454` 는 `ee_finger_` 만 보는 채로 둔다.
`config.py:367-370` 의 `DEFAULT_CONTACT_LINKS` 와 어긋나는 것은 **알려진 상태로 남긴다** —
`impl.md` 에 "판정으로 유지" 한 줄만 적는다.

## 검증 (네가 하는 것)

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -u -m pytest tests/ -q
```

직전 기준 **659 passed**. 떨어지는 것이 있으면 숫자와 이름을 적는다.
`runner_up_score` 에 대한 **단위 테스트를 새로 하나 더한다** (`tests/ag3s/` 아래).

## 동시에 A2 가 같은 tree 로 측정 중이다

A2 가 `ep1807` 기록으로 T2 의 남은 두 항목을 재고 있고, **변경 전 코드의 수치여야 한다.**
`target_grounding.py` 를 저장하는 **시각(초까지)** 을 `impl.md` 에 적어라 — A2 가 run 전후
`md5sum` 을 찍고 있어서, 겹쳤는지 그 둘로 판정한다.

## 규칙

- 네 소유: `benchmark/**/*.py` · `tests/**` · `handoff/T2-fix.impl.md`. **`docs/*.md`·figure 는
  건드리지 않는다** (A3·A2 의 것).
- **측정하지 않는다.** 긴 rollout 을 돌리지 않는다 — 단위 테스트까지다.
- 커밋하지 않는다. lead 가 한다.
- `pkill -f` 금지. 긴 실행은 `python -u`.
- **여기 박힌 수치(0.8051 · 1.3 · 14/15 · 9/15 · +0.157 mm · 659) 밖의 숫자를 지어내지 마라.**
