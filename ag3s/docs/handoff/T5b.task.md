# T5b — `degraded` 의 **사유**를 와이어에 싣는다

> writer: lead (A0) · 2026-09-25 · 주 담당 **A1(implementer)** · 산출 **하나**: `handoff/T5b.impl.md`
> **`T5a`(shadow) 를 끝낸 뒤 이어서 한다.** 서버 재기동은 둘을 합쳐 lead 가 한 번만 한다.

## 왜 — 첫 live 실행에서 바로 막혔다

2026-09-25 smoke (ep1800, 2 청크, `--safe-remote`, 서버 = `serve_safe.py` cuRobo backend):

| | t=0 | **t=8** |
|---|---|---|
| `ag3s_status` | ok | **degraded** |
| `geometry_certified` | True | **False** |
| `trajopt_status` | feasible | **violated** |
| `max_violation_m` | 0.0 | **0.0** |
| `safe` | True | **False** |

**`max_violation_m` 이 0.0 인데 `violated` 다.** 궤적은 깨끗하고 **기하 인증이 실패한 것**이다
(판정 note: *"the trajectory clears every constraint, but AG3S could not certify the geometry
behind them"*). X3(`status` 가 clearance 가 아니라 기하 인증을 보고 뒤집힌다)의 구조가 live
경로에서도 그대로 나타난다.

**그런데 왜 degraded 인지가 어디에도 없다.** 관측 프레임 note 는 이렇게만 적혀 있다:

> `"grounding 상태·점 개수는 서버 안에만 있다 (T1 에서 와이어에 싣는다)"`

서버 로그에도 없다 (lead 가 확인했다). **X3 의 원인은 아니다** — `serve_safe.py:61` 이
`pointcloud` 에 `range_max` 만 넣으므로 `max_points` 는 기본값 **200000**(`config.py:148`,
X3 에서 올린 값)을 그대로 받는다.

## 무엇을 하나

**로컬이 `degraded` 를 받았을 때 "어디서, 왜" 를 알 수 있게 한다.**

`pipeline.py` 에서 `ConstraintValidity.DEGRADED` 로 내려가는 자리가 최소 다섯이다 —
**`:346` · `:443` · `:538` · `:662` · `:669`**. 먼저 **다섯을 다 읽고 각각이 무슨 조건인지
`impl.md` 에 한 줄씩 표로 적어라.** 그것이 이 STEP 의 절반이다.

그다음 **각 자리가 사유를 구조화된 형태로 남기게** 하고, 그것을 와이어에 싣는다:

- 이미 `notes` 가 있는 자리는 그것을 쓰고, **없는 자리는 만들어라.** "degraded 인데 사유 없음"
  이 하나라도 남으면 이 STEP 은 실패다.
- 사유에는 **그 자리를 특정할 수 있는 것**이 들어가야 한다 — 어떤 cap 에 걸렸는지, 어떤 단계가
  부분 관측이었는지, 수치가 있으면 그 수치(점 개수 · cap 값 · 카메라 이름).
- `wire.py` 응답에 실어라. `field` 가 선택 키인 것과 같은 방식으로 **`ag3s`(또는 그에 준하는)
  블록**을 두고, `ag3s_status` 가 `ok` 가 아닐 때의 사유를 담는다. **`T5a` 에서 더한
  `actions_reference` 와 충돌하지 않게** 해라.
- `frame_record.py` 가 그것을 프레임별로 저장하게 해라 — `plot_t0.py` 가 읽는 형식을 깨지 않고.

## 경계 — 원인을 **고치지 마라**

이 STEP 은 **보이게 만드는 것**이다. `degraded` 를 없애는 수정(cap 을 올리거나 판정을 느슨하게)
은 **하지 않는다.** 왜 degraded 인지 수치로 확정한 뒤에 사용자 판정으로 고친다.
X3 에서 `max_points` 를 올린 것도 그 순서를 밟았다.

## 요구사항

1. **기본 동작은 한 바이트도 바뀌지 않는다** — `ag3s_status == "ok"` 인 응답은 예전과 같아야
   한다. 회귀 기준선과 T0 기록이 거기 달려 있다.
2. **`degraded` 인데 사유가 비어 있는 경로가 없다.** 다섯 자리 전부.
3. 로컬이 사유를 **출력하고 기록한다** — HOLD 를 찍을 때 사유를 같이 찍어라. 지금은
   `[safe] t=8 HOLD — AG3S could not certify the geometry (status=degraded)` 까지만 나온다.

## 검증 (네가 하는 것)

```bash
cd /mnt/dev/work && MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work .venv-ag3s/bin/python -u -m pytest tests/ -q
```

`T5a` 뒤의 숫자에서 시작한다. **다섯 자리 각각이 사유를 남기는지 단위 테스트로 잡아라** —
`degraded` 를 내면서 사유가 비면 실패하는 테스트가 하나는 있어야 한다.

## 규칙

- 네 소유: `benchmark/ag3s/**/*.py` · `benchmark/trajopt/*.py` · `tests/**` ·
  `pi05_TO_hybrid/rby1_bringup/pi05_infer.py` · `handoff/T5b.impl.md`.
  **`docs/*.md`·figure·`*.verify.json` 은 건드리지 않는다.**
- **측정하지 않는다.** 긴 rollout 을 돌리지 않는다.
- **포트 8000 의 서버를 죽이지 말아라.** 재기동은 lead 가 `T5a`+`T5b` 를 합쳐 한 번만 한다.
- 커밋하지 않는다. `pkill -f` 금지 — PID 를 뽑아 `kill <PID>`.
- 위에 박힌 수치(200000 · 0.0 · 다섯 자리의 줄번호) 밖의 숫자를 지어내지 마라.
